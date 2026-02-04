# agent/logging_middleware.py
import time
import json
import os
from typing import Callable, List, Dict, Any, Awaitable
from starlette.types import ASGIApp, Receive, Scope, Send

# Config via env
LOG_JSON = os.environ.get("LOG_JSON", "1") == "1"
LOG_HTTP_BODY = os.environ.get("LOG_HTTP_BODY", "0") == "1"
LOG_HTTP_MAXLEN = int(os.environ.get("LOG_HTTP_MAXLEN", "2000"))

def _safe_truncate(b: bytes) -> str:
    if not b:
        return ""
    try:
        s = b.decode(errors="replace")
    except Exception:
        s = str(b)
    if len(s) > LOG_HTTP_MAXLEN:
        return s[:LOG_HTTP_MAXLEN] + "…(truncated)"
    return s

def _emit_log(level: str, event: str, details: dict):
    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "level": level,
        "event": event,
        "details": details,
    }
    if LOG_JSON:
        print(json.dumps(rec, default=str), flush=True)
    else:
        print(f"[{rec['ts']}] {level} {event} {details}", flush=True)


class SafeLoggingMiddleware:
    """
    ASGI middleware that safely reads & logs HTTP request bodies and replays
    the original ASGI messages to the downstream app to avoid consume/replay bugs.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Only handle http requests here; pass websockets unchanged.
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # 1) Drain/collect the incoming http.request messages fully
        received_messages: List[Dict[str, Any]] = []
        body_bytes = b""
        while True:
            msg = await receive()
            received_messages.append(msg)
            if msg.get("type") == "http.request":
                body_bytes += msg.get("body", b"")
                if not msg.get("more_body", False):
                    break
            else:
                # Not an http.request (rare) — keep it and break.
                break

        # Build a replay queue of messages identical to what we consumed.
        replay_queue = list(received_messages)

        # 2) Create a receive wrapper that replays the buffered messages first,
        #    then returns a harmless disconnect if called again.
        async def replay_receive() -> Dict[str, Any]:
            if replay_queue:
                return replay_queue.pop(0)
            # Downstream may call receive again — return a disconnect message.
            return {"type": "http.disconnect"}

        # 3) Log inbound request metadata safely
        try:
            method = scope.get("method")
            path = scope.get("path")
            client = scope.get("client")
            remote = f"{client[0]}:{client[1]}" if client else None
            headers = {k.decode(): v.decode() for k, v in scope.get("headers", [])}
            details = {
                "method": method,
                "path": path,
                "remote": remote,
            }
            if LOG_HTTP_BODY:
                details["body"] = _safe_truncate(body_bytes)
            else:
                details["body_len"] = len(body_bytes)
            _emit_log("INFO", "http_in", details)
        except Exception as exc:
            _emit_log("ERROR", "http_in_log_failed", {"error": str(exc)})

        # 4) Call downstream app with the replay_receive.
        # We'll capture the response status and time by wrapping send.
        started = None
        status_code = None

        async def send_wrapper(message: Dict[str, Any]) -> None:
            nonlocal started, status_code
            # Intercept http.response.start to get status code
            if message["type"] == "http.response.start":
                started = time.time()
                status_code = message.get("status")
            await send(message)

        try:
            await self.app(scope, replay_receive, send_wrapper)
            duration_ms = None
            if started is not None:
                duration_ms = int((time.time() - started) * 1000)
            _emit_log("INFO", "http_out", {
                "path": scope.get("path"),
                "status_code": status_code,
                "duration_ms": duration_ms,
            })
        except Exception as exc:
            # log exception with stack-like info
            _emit_log("ERROR", "http_app_error", {"error": repr(exc)})
            # re-raise so upstream frameworks handle the error normally
            raise
