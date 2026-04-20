"""
Azure OpenAI Realtime WebSocket client (mirrors src/azureRealtimeClient.js).
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import os
import urllib.parse
from typing import Any, Awaitable, Callable

import websockets
from websockets.client import WebSocketClientProtocol

import logger as log

LOG_EVENT_TYPES = frozenset(
    {
        "error",
        "response.content.done",
        "rate_limits.updated",
        "response.done",
        "input_audio_buffer.committed",
        "input_audio_buffer.speech_stopped",
        "input_audio_buffer.speech_started",
        "session.created",
    }
)

INITIAL_GREETING = os.environ.get("INITIAL_GREETING", "Hello, how can I help you today?")
VOICE = os.environ.get("OPENAI_VOICE_ID", "alloy")

Callback = Callable[..., Any] | None


async def _maybe_await(fn: Callback, *args: Any) -> None:
    if fn is None:
        return
    result = fn(*args)
    if inspect.isawaitable(result):
        await result


class AzureRealtimeClient:
    """
    Emits the same logical events as the JS EventEmitter via async callbacks:
    ready, audio_delta(bytes), audio_done, speech_started, speech_stopped,
    transcript(str), ai_text(str), error(Exception), closed
    """

    def __init__(self, session_id: str, callbacks: dict[str, Callback]) -> None:
        self.session_id = session_id
        self._callbacks = callbacks
        self._ws: WebSocketClientProtocol | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._connect_lock = asyncio.Lock()

    async def connect(self) -> None:
        async with self._connect_lock:
            if self._ws and not self._ws.closed:
                return

            endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            api_key = os.environ.get("AZURE_OPENAI_API_KEY")
            deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
            api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-01-preview")

            if not endpoint or not api_key or not deployment:
                raise RuntimeError(
                    "Missing env vars: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT"
                )

            host = urllib.parse.urlparse(endpoint).netloc
            q = urllib.parse.urlencode({"api-version": api_version, "deployment": deployment})
            url = f"wss://{host}/openai/realtime?{q}"

            log.info("[Azure] Connecting", {"sessionId": self.session_id, "host": host, "deployment": deployment})

            extra_headers = {
                "api-key": api_key,
                "OpenAI-Beta": "realtime=v1",
            }

            self._ws = await websockets.connect(url, additional_headers=extra_headers)

        log.info("[Azure] WebSocket open", {"sessionId": self.session_id})

        async def delayed_init() -> None:
            await asyncio.sleep(0.1)
            await self._initialize_session()

        asyncio.create_task(delayed_init())

        self._reader_task = asyncio.create_task(self._reader_loop())

    async def _reader_loop(self) -> None:
        assert self._ws is not None
        try:
            async for message in self._ws:
                if isinstance(message, bytes):
                    await self._on_message(message.decode("utf-8"))
                else:
                    await self._on_message(message)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.error("[Azure] Reader error", {"sessionId": self.session_id, "error": str(e)})
            await _maybe_await(self._callbacks.get("error"), e)
        finally:
            log.info("[Azure] WebSocket closed", {"sessionId": self.session_id})
            await _maybe_await(self._callbacks.get("closed"))

    async def _initialize_session(self) -> None:
        session_update = {
            "type": "session.update",
            "session": {
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.9,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                    "create_response": True,
                    "interrupt_response": True,
                },
                "input_audio_format": "g711_ulaw",
                "output_audio_format": "g711_ulaw",
                "input_audio_transcription": {"model": "whisper-1"},
                "voice": VOICE,
                "instructions": os.environ.get("DEFAULT_PROMPT_INSTRUCTIONS")
                or "You are a helpful voice assistant on a phone call. Keep responses concise and conversational.",
                "modalities": ["text", "audio"],
                "temperature": 0.8,
            },
        }

        log.info("[Azure] Sending session.update", {"sessionId": self.session_id})
        await self._send(session_update)

        initial_conversation_item = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": INITIAL_GREETING}],
            },
        }

        log.info("[Azure] Sending initial conversation item", {"sessionId": self.session_id})
        await self._send(initial_conversation_item)
        await self._send({"type": "response.create"})

        await _maybe_await(self._callbacks.get("ready"))

    async def _on_message(self, raw: str) -> None:
        try:
            response: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError:
            return

        rtype = response.get("type", "")
        if rtype in LOG_EVENT_TYPES:
            log.info("[Azure] Event", {"sessionId": self.session_id, "type": rtype})
        else:
            log.debug("[Azure] Event", {"sessionId": self.session_id, "type": rtype})

        if rtype == "response.audio.delta" and response.get("delta"):
            audio_bytes = base64.b64decode(response["delta"])
            await _maybe_await(self._callbacks.get("audio_delta"), audio_bytes)

        if rtype == "response.done":
            status = (response.get("response") or {}).get("status")
            log.info("[Azure] response.done", {"sessionId": self.session_id, "status": status})

            if status == "completed":
                output = (response.get("response") or {}).get("output") or []
                texts: list[str] = []
                for o in output:
                    if o.get("type") == "message" and o.get("role") == "assistant":
                        for c in o.get("content") or []:
                            if c.get("type") in ("text", "audio"):
                                t = c.get("transcript") or c.get("text") or ""
                                if t:
                                    texts.append(t)
                output_text = " ".join(texts).strip()
                if output_text:
                    await _maybe_await(self._callbacks.get("ai_text"), output_text)
                await _maybe_await(self._callbacks.get("audio_done"))
            elif status == "failed":
                log.error(
                    "[Azure] Response failed",
                    {"sessionId": self.session_id, "details": (response.get("response") or {}).get("status_details")},
                )
                await _maybe_await(self._callbacks.get("error"), RuntimeError("Azure response failed"))

        if rtype == "input_audio_buffer.speech_started":
            log.info("[Azure] Speech started — sending barge-in", {"sessionId": self.session_id})
            await _maybe_await(self._callbacks.get("speech_started"))

        if rtype == "input_audio_buffer.speech_stopped":
            log.info("[Azure] Speech stopped", {"sessionId": self.session_id})
            await _maybe_await(self._callbacks.get("speech_stopped"))

        if rtype == "conversation.item.input_audio_transcription.completed":
            text = response.get("transcript") or ""
            log.info("[Azure] Transcript", {"sessionId": self.session_id, "text": text})
            await _maybe_await(self._callbacks.get("transcript"), text)

        if rtype == "error":
            log.error("[Azure] API error", {"sessionId": self.session_id, "error": response.get("error")})
            err = response.get("error") or {}
            msg = err.get("message") if isinstance(err, dict) else str(err)
            await _maybe_await(self._callbacks.get("error"), RuntimeError(msg or json.dumps(err)))

    async def append_audio(self, base64_audio: str) -> None:
        await self._send({"type": "input_audio_buffer.append", "audio": base64_audio})

    async def send_no_input(self) -> None:
        no_input_message = os.environ.get(
            "NO_INPUT_MESSAGE", "User did not provide any input. Act accordingly."
        )
        await self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": no_input_message}],
                },
            }
        )
        await self._send({"type": "response.create"})

    async def _send(self, obj: dict[str, Any]) -> None:
        if not self._ws or self._ws.closed:
            log.warn("[Azure] Send on closed WS", {"sessionId": self.session_id})
            return
        await self._ws.send(json.dumps(obj))

    async def disconnect(self) -> None:
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None

        if self._ws and not self._ws.closed:
            log.info("[Azure] Closing connection", {"sessionId": self.session_id})
            await self._ws.close()
        self._ws = None
