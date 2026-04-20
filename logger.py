"""Structured JSON logging (mirrors src/logger.js)."""

import json
import os
import sys
from datetime import datetime, timezone


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


DEBUG = os.environ.get("DEBUG", "").lower() == "true"


def _emit(stream, level: str, msg: str, data: dict) -> None:
    payload = {"ts": _ts(), "level": level, "msg": msg, **data}
    stream.write(json.dumps(payload) + "\n")
    stream.flush()


def info(msg: str, data: dict | None = None) -> None:
    _emit(sys.stdout, "INFO", msg, data or {})


def error(msg: str, data: dict | None = None) -> None:
    _emit(sys.stderr, "ERROR", msg, data or {})


def warn(msg: str, data: dict | None = None) -> None:
    _emit(sys.stderr, "WARN", msg, data or {})


def debug(msg: str, data: dict | None = None) -> None:
    if DEBUG:
        _emit(sys.stdout, "DEBUG", msg, data or {})
