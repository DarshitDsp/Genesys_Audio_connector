from __future__ import annotations

import asyncio
import base64
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aiohttp import web

from azure_realtime_client import AzureRealtimeClient
import logger as log


TRANSCRIPTS_DIR = Path.cwd() / "transcripts"
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_BINARY_MESSAGE_SIZE = int(os.environ.get("MAX_BINARY_MESSAGE_SIZE", "64000"))
MIN_BINARY_MESSAGE_SIZE = int(os.environ.get("MIN_BINARY_MESSAGE_SIZE", "8000"))
NO_INPUT_TIMEOUT = int(os.environ.get("NO_INPUT_TIMEOUT", "15000"))


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_transcript(session_id: str, speaker: str, text: str) -> None:
    file_path = TRANSCRIPTS_DIR / f"{session_id}.txt"
    line = f"[{_iso_now()}] {speaker}: {text}\n"
    with file_path.open("a", encoding="utf-8") as f:
        f.write(line)


class SessionHandler:
    def __init__(self, ws: web.WebSocketResponse, session_id: str, org_id: str | None, correlation_id: str | None):
        self.ws = ws
        self.client_session_id = session_id
        self.org_id = org_id
        self.correlation_id = correlation_id

        self.last_server_sequence_number = 0
        self.last_client_sequence_number = 0

        self.azure_client: AzureRealtimeClient | None = None
        self.selected_media: dict[str, Any] | None = None
        self.conversation_id: str | None = None
        self.input_variables: dict[str, Any] = {}

        self.disconnecting = False
        self.closed = False
        self.is_audio_playing = False

        self.buffer: list[bytes] = []
        self._flush_task: asyncio.Task[None] | None = None
        self._no_input_timer: asyncio.Task[None] | None = None

        self.start_time = asyncio.get_running_loop().time()
        self.bytes_received = 0

        log.info("[Session] Created", {"sessionId": self.client_session_id, "orgId": org_id})

    async def handle_text_message(self, msg: dict[str, Any]) -> None:
        if self.closed:
            return

        if isinstance(msg.get("seq"), int) and msg["seq"] != self.last_client_sequence_number + 1:
            log.warn(
                "[Session] Invalid client seq",
                {
                    "sessionId": self.client_session_id,
                    "expected": self.last_client_sequence_number + 1,
                    "got": msg["seq"],
                },
            )
            await self.send_disconnect("error", "Invalid client sequence number.", {})
            return

        if isinstance(msg.get("seq"), int):
            self.last_client_sequence_number = msg["seq"]

        if isinstance(msg.get("serverseq"), int) and msg["serverseq"] > self.last_server_sequence_number:
            log.warn("[Session] Invalid server seq ack", {"sessionId": self.client_session_id})
            await self.send_disconnect("error", "Invalid server sequence number.", {})
            return

        if msg.get("id") and msg["id"] != self.client_session_id:
            log.warn(
                "[Session] Session ID mismatch",
                {"sessionId": self.client_session_id, "got": msg["id"]},
            )
            await self.send_disconnect("error", "Invalid ID specified.", {})
            return

        log.info(
            f"[Session] Received '{msg.get('type')}'",
            {"sessionId": self.client_session_id, "params": json.dumps(msg.get("parameters"))},
        )

        msg_type = msg.get("type")
        if msg_type == "open":
            await self._handle_open(msg)
        elif msg_type == "ping":
            await self._handle_ping()
        elif msg_type == "close":
            await self._handle_close(msg)
        elif msg_type == "playback_completed":
            await self._handle_playback_completed()
        elif msg_type == "playback_started":
            self.is_audio_playing = False
        elif msg_type == "dtmf":
            log.info("[Session] DTMF", {"sessionId": self.client_session_id, "digit": (msg.get("parameters") or {}).get("digit")})
        elif msg_type == "update":
            await self._send("updated", {})
        elif msg_type == "error":
            log.error("[Session] Genesys error", {"sessionId": self.client_session_id, "params": msg.get("parameters")})
        else:
            log.warn("[Session] Unknown message type", {"sessionId": self.client_session_id, "type": msg_type})

    async def handle_binary_message(self, data: bytes) -> None:
        if self.disconnecting or self.closed:
            return

        self.bytes_received += len(data)
        log.debug("[Session] Binary audio", {"sessionId": self.client_session_id, "bytes": len(data)})

        if self.azure_client:
            b64 = base64.b64encode(data).decode("ascii")
            await self.azure_client.append_audio(b64)

    async def _handle_open(self, msg: dict[str, Any]) -> None:
        params = msg.get("parameters") or {}
        self.conversation_id = params.get("conversationId")
        self.input_variables = params.get("inputVariables") or {}

        participant = params.get("participant") or {}
        log.info(
            "[Session] Open",
            {
                "sessionId": self.client_session_id,
                "conversationId": self.conversation_id,
                "ani": participant.get("ani"),
                "dnis": participant.get("dnis"),
            },
        )

        media_list = params.get("media") or []
        self.selected_media = (
            next((m for m in media_list if m.get("format") == "PCMU" and m.get("rate") == 8000), None)
            or next((m for m in media_list if m.get("format") == "L16" and m.get("rate") == 8000), None)
            or (media_list[0] if media_list else None)
        )

        if not self.selected_media:
            log.error("[Session] No supported media", {"sessionId": self.client_session_id, "mediaList": media_list})
            await self.send_disconnect("error", "No supported media type was found.", {})
            return

        log.info("[Session] Selected media", {"sessionId": self.client_session_id, "media": self.selected_media})
        append_transcript(
            self.client_session_id,
            "SESSION",
            f"started | conversationId={self.conversation_id} | ani={participant.get('ani','')} | dnis={participant.get('dnis','')}",
        )

        await self._send("opened", {"media": [self.selected_media]})
        asyncio.create_task(self._connect_azure())

    async def _handle_ping(self) -> None:
        await self._send("pong", {})

    async def _handle_close(self, msg: dict[str, Any]) -> None:
        reason = (msg.get("parameters") or {}).get("reason")
        log.info("[Session] Close", {"sessionId": self.client_session_id, "reason": reason})
        await self._send("closed", {})
        await self.cleanup()

    async def _handle_playback_completed(self) -> None:
        log.info("[Session] Playback completed", {"sessionId": self.client_session_id})
        self.is_audio_playing = False
        self._start_no_input_timer()

    async def _connect_azure(self) -> None:
        try:
            self.azure_client = AzureRealtimeClient(
                self.client_session_id,
                callbacks={
                    "ready": self._on_azure_ready,
                    "audio_delta": self._on_azure_audio_delta,
                    "audio_done": self._on_azure_audio_done,
                    "speech_started": self._on_azure_speech_started,
                    "speech_stopped": self._on_azure_speech_stopped,
                    "transcript": self._on_azure_transcript,
                    "ai_text": self._on_azure_ai_text,
                    "error": self._on_azure_error,
                    "closed": self._on_azure_closed,
                },
            )
            await self.azure_client.connect()
        except Exception as err:
            log.error("[Session] Azure connect failed", {"sessionId": self.client_session_id, "error": str(err)})
            await self.send_disconnect("error", "Azure connection failed.", {})

    async def _on_azure_ready(self) -> None:
        log.info("[Session] Azure ready", {"sessionId": self.client_session_id})

    async def _on_azure_audio_delta(self, audio_buf: bytes) -> None:
        self.is_audio_playing = True
        self._stop_no_input_timer()
        await self.send_audio(audio_buf)

    async def _on_azure_audio_done(self) -> None:
        log.info("[Session] Azure audio done - flushing buffer", {"sessionId": self.client_session_id})
        await self.flush_buffer()

    async def _on_azure_speech_started(self) -> None:
        self._stop_no_input_timer()
        await self.send_barge_in()
        self.is_audio_playing = True

    async def _on_azure_speech_stopped(self) -> None:
        return

    async def _on_azure_transcript(self, text: str) -> None:
        log.info("[Session] Transcript", {"sessionId": self.client_session_id, "text": text})
        append_transcript(self.client_session_id, "User", text)
        await self.send_transcript(text, 1.0, True)

    async def _on_azure_ai_text(self, text: str) -> None:
        log.info("[Session] AI response text", {"sessionId": self.client_session_id, "text": text})
        append_transcript(self.client_session_id, "Assistant", text)

    async def _on_azure_error(self, err: Exception) -> None:
        log.error("[Session] Azure error", {"sessionId": self.client_session_id, "error": str(err)})

    async def _on_azure_closed(self) -> None:
        log.info("[Session] Azure closed", {"sessionId": self.client_session_id})

    async def send_audio(self, bytes_data: bytes) -> None:
        self.buffer.append(bytes_data)
        total_length = sum(len(b) for b in self.buffer)

        if total_length < MIN_BINARY_MESSAGE_SIZE:
            if not self._flush_task or self._flush_task.done():
                self._flush_task = asyncio.create_task(self._delayed_flush())
            return

        await self.flush_buffer()

    async def _delayed_flush(self) -> None:
        await asyncio.sleep(0.5)
        await self.flush_buffer()

    async def flush_buffer(self) -> None:
        total_length = sum(len(b) for b in self.buffer)
        if total_length <= 0:
            return

        data = b"".join(self.buffer)
        self.buffer = []

        pos = 0
        while pos < len(data):
            chunk = data[pos : pos + MAX_BINARY_MESSAGE_SIZE]
            log.debug("[Session] Sending audio chunk", {"sessionId": self.client_session_id, "bytes": len(chunk)})
            await self.ws.send_bytes(chunk)
            pos += MAX_BINARY_MESSAGE_SIZE

    async def send_barge_in(self) -> None:
        self.buffer = []
        log.info("[Session] Sending barge-in", {"sessionId": self.client_session_id})
        await self._send("event", {"entities": [{"type": "barge_in", "data": {}}]})

    async def send_transcript(self, transcript: str, confidence: float, is_final: bool) -> None:
        channel = ((self.selected_media or {}).get("channels") or ["external"])[0]
        log.info(
            "[Session] Sending transcript",
            {"sessionId": self.client_session_id, "transcript": transcript, "isFinal": is_final},
        )
        await self._send(
            "event",
            {
                "entities": [
                    {
                        "type": "transcript",
                        "data": {
                            "id": str(uuid.uuid4()),
                            "channel": channel,
                            "isFinal": is_final,
                            "alternatives": [
                                {
                                    "confidence": confidence,
                                    "interpretations": [{"type": "normalized", "transcript": transcript}],
                                }
                            ],
                        },
                    }
                ]
            },
        )

    async def send_disconnect(self, reason: str, info: str, output_variables: dict[str, Any]) -> None:
        self.disconnecting = True
        log.info("[Session] Sending disconnect", {"sessionId": self.client_session_id, "reason": reason, "info": info})
        await self._send("disconnect", {"reason": reason, "info": info, "outputVariables": output_variables or {}})

    def _start_no_input_timer(self) -> None:
        self._stop_no_input_timer()

        async def timer() -> None:
            await asyncio.sleep(NO_INPUT_TIMEOUT / 1000.0)
            log.info("[Session] No-input timeout - triggering Azure response", {"sessionId": self.client_session_id})
            if self.azure_client:
                await self.azure_client.send_no_input()

        self._no_input_timer = asyncio.create_task(timer())

    def _stop_no_input_timer(self) -> None:
        if self._no_input_timer and not self._no_input_timer.done():
            self._no_input_timer.cancel()
        self._no_input_timer = None

    async def cleanup(self) -> None:
        if self.closed:
            return
        self.closed = True

        self._stop_no_input_timer()

        duration_ms = int((asyncio.get_running_loop().time() - self.start_time) * 1000)
        log.info(
            "[Session] Cleanup",
            {
                "sessionId": self.client_session_id,
                "durationMs": duration_ms,
                "bytesReceived": self.bytes_received,
            },
        )
        append_transcript(
            self.client_session_id,
            "SESSION",
            f"ended | durationMs={duration_ms} | bytesReceived={self.bytes_received}",
        )

        if self.azure_client:
            await self.azure_client.disconnect()
            self.azure_client = None

    async def _send(self, msg_type: str, parameters: dict[str, Any]) -> None:
        if self.ws.closed:
            return

        self.last_server_sequence_number += 1
        msg = {
            "version": "2",
            "id": self.client_session_id,
            "type": msg_type,
            "seq": self.last_server_sequence_number,
            "clientseq": self.last_client_sequence_number,
            "parameters": parameters or {},
        }

        if msg_type == "event":
            entity_type = ((msg.get("parameters") or {}).get("entities") or [{}])[0].get("type")
            log.info(f"[Session] Sending event: {entity_type}", {"sessionId": self.client_session_id})
        else:
            log.info(f"[Session] Sending '{msg_type}'", {"sessionId": self.client_session_id})

        await self.ws.send_str(json.dumps(msg))
