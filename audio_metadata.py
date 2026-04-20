"""
Audio metadata helpers (mirrors src/audioMetadata.js).
WAV header parsing and format mapping for Azure Realtime-style labels.
"""

from __future__ import annotations

import struct
from typing import Any


def is_wav_file(buf: bytes | bytearray | memoryview) -> bool:
    if not buf or len(buf) < 12:
        return False
    b = bytes(buf[:12])
    return b[:4] == b"RIFF" and b[8:12] == b"WAVE"


def find_chunk(buf: bytes, chunk_id: str) -> int:
    cid = chunk_id.encode("ascii")
    for i in range(12, len(buf) - 8):
        if buf[i : i + 4] == cid:
            return i
    return -1


def resolve_format(code: int, bits_per_sample: int) -> str:
    if code == 6:
        return "g711_alaw"
    if code == 7:
        return "g711_ulaw"
    return "pcm16"


def parse_wav_header(buf: bytes) -> dict[str, Any]:
    if len(buf) < 44:
        raise ValueError("Buffer too small to contain a valid WAV header")

    fmt_offset = find_chunk(buf, "fmt ")
    if fmt_offset == -1:
        raise ValueError("WAV file missing fmt chunk")

    fmt_data = fmt_offset + 8
    audio_format_code = struct.unpack_from("<H", buf, fmt_data)[0]
    channels = struct.unpack_from("<H", buf, fmt_data + 2)[0]
    sample_rate = struct.unpack_from("<I", buf, fmt_data + 4)[0]
    bits_per_sample = struct.unpack_from("<H", buf, fmt_data + 14)[0]
    audio_format = resolve_format(audio_format_code, bits_per_sample)

    data_chunk_offset = find_chunk(buf, "data")
    data_offset = data_chunk_offset + 8 if data_chunk_offset != -1 else 44

    return {
        "format": audio_format,
        "sampleRate": sample_rate,
        "channels": channels,
        "bitsPerSample": bits_per_sample,
        "dataOffset": data_offset,
        "raw": False,
        "source": "wav_header",
        "audioFormatCode": audio_format_code,
    }


def extract_metadata(
    first_chunk: bytes | bytearray | memoryview,
    client_declared: dict[str, Any] | None = None,
) -> dict[str, Any]:
    client_declared = client_declared or {}
    if is_wav_file(first_chunk):
        return parse_wav_header(bytes(first_chunk))

    fmt = client_declared.get("format") or "pcm16"
    sample_rate = client_declared.get("sampleRate") or 24000
    channels = client_declared.get("channels") or 1
    bits = client_declared.get("bitsPerSample") or 16

    return {
        "format": fmt,
        "sampleRate": sample_rate,
        "channels": channels,
        "bitsPerSample": bits,
        "dataOffset": 0,
        "raw": True,
        "source": "client_declared" if client_declared.get("format") else "default",
    }


def summarize(metadata: dict[str, Any]) -> str:
    return (
        f"format={metadata.get('format')}, sampleRate={metadata.get('sampleRate')}Hz, "
        f"channels={metadata.get('channels')}, bits={metadata.get('bitsPerSample')}, "
        f"source={metadata.get('source')}"
    )
