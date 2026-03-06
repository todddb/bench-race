from __future__ import annotations

import json
import logging
import os
import traceback
from datetime import datetime, timezone
from typing import Any, Dict


class JsonFormatter(logging.Formatter):
    """Minimal JSON formatter for structured service logs."""

    _base_keys = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "level": record.levelname,
            "event": record.getMessage(),
        }

        for key in ("endpoint", "model_id", "backend", "error"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value

        for key, value in record.__dict__.items():
            if key not in self._base_keys and key not in payload:
                payload[key] = value

        if record.exc_info:
            payload["stacktrace"] = "".join(traceback.format_exception(*record.exc_info)).strip()

        return json.dumps(payload, default=str)


def configure_logging() -> None:
    level_name = os.getenv("WRAPPER_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root_logger.addHandler(handler)

