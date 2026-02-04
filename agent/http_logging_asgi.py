"""
ASGI-safe HTTP logging middleware.

Replaces the problematic BaseHTTPMiddleware-based logger that caused
"Unexpected message received: http.request" errors.

This middleware:
- Drains all incoming http.request messages
- Replays buffered messages to downstream via replay_receive
- Never interferes with WebSocket connections
- Logs http_in with full request details
"""

import time
import secrets
from typing import Any, Dict, List, Optional
from starlette.types import ASGIApp, Receive, Scope, Send

from agent.logging_utils import get_logger


class HTTPLoggingASGIMiddleware:
    """
    ASGI-safe HTTP logger that drains & replays request messages.
    Avoids BaseHTTPMiddleware request-stream pitfalls that can cause:
      RuntimeError: Unexpected message received: http.request
    """

    def __init__(self, app: ASGIApp, *, agent_id: str, hostname: str):
        self.app = app
        self.agent_id = agent_id
        self.hostname = hostname
        self.logger = get_logger()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # --- Drain entire incoming request stream ---
        received: List[Dict[str, Any]] = []
        body = b""
        while True:
            msg = await receive()
            received.append(msg)

            if msg.get("type") == "http.request":
                body += msg.get("body", b"")
                if not msg.get("more_body", False):
                    break
                continue

            # anything else - stop (rare)
            break

        replay_queue = list(received)

        async def replay_receive() -> Dict[str, Any]:
            if replay_queue:
                return replay_queue.pop(0)
            return {"type": "http.disconnect"}

        # --- build log fields similar to existing logger ---
        t0 = time.time()
        request_id = secrets.token_hex(4)

        method = scope.get("method")
        path = scope.get("path")
        client = scope.get("client")
        remote_addr = None
        if client and len(client) >= 2:
            remote_addr = f"{client[0]}:{client[1]}"

        headers = {}
        for k, v in scope.get("headers", []):
            try:
                headers[k.decode()] = v.decode()
            except Exception:
                headers[str(k)] = str(v)

        # We'll capture status_code from response.start
        status_code: Optional[int] = None

        async def send_wrapper(message: Dict[str, Any]) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message.get("status", 0) or 0)
            await send(message)

        # Call downstream app with replayed receive
        try:
            await self.app(scope, replay_receive, send_wrapper)
        finally:
            dur_ms = (time.time() - t0) * 1000.0

            # Skip logging for health checks (same behavior as old middleware)
            if path and path.endswith("/health"):
                self.logger.debug(
                    "http_in",
                    request_id=request_id,
                    method=method,
                    path=path,
                    status_code=status_code,
                    duration_ms=dur_ms,
                )
            else:
                # Parse body as JSON if applicable
                body_obj = None
                try:
                    if body and headers.get("content-type", "").startswith("application/json"):
                        import json
                        body_obj = json.loads(body)
                except Exception:
                    pass

                # Use existing logger's http_in method
                self.logger.http_in(
                    method=method,
                    path=path,
                    remote_addr=remote_addr or "unknown",
                    request_id=request_id,
                    headers=headers,
                    body=body_obj,
                    status_code=status_code,
                    duration_ms=dur_ms,
                )
