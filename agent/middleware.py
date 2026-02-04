"""
FastAPI middleware for bench-race agent logging.

Provides automatic logging of:
- HTTP requests (method, path, headers, body)
- HTTP responses (status, duration)
- WebSocket connections (connect, disconnect)
- Errors and exceptions
"""

import json
import time
import traceback
import uuid
from typing import Callable

from fastapi import Request, Response, WebSocket
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from agent.logging_utils import get_logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that logs all HTTP requests and responses.

    Logs:
    - Request: method, path, remote address, headers, body
    - Response: status code, duration
    - Errors: exception details with stack trace
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = get_logger()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process HTTP request and log details."""
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        # Extract request details
        method = request.method
        path = str(request.url.path)
        query_params = str(request.url.query) if request.url.query else None
        remote_addr = request.client.host if request.client else "unknown"

        # Get headers (sanitized)
        headers = dict(request.headers)

        # Get request body (if applicable and LOG_HTTP_BODY is enabled)
        body = None
        try:
            if request.method in ("POST", "PUT", "PATCH"):
                # Try to read body as JSON
                content_type = headers.get("content-type", "")
                if "application/json" in content_type:
                    # Read raw body
                    raw_body = await request.body()
                    if raw_body:
                        body = json.loads(raw_body)

                        # Re-populate request body for downstream handlers
                        async def receive():
                            return {"type": "http.request", "body": raw_body}
                        request._receive = receive
        except Exception:
            # If we can't parse body, skip it
            pass

        # Add query params to path if present
        full_path = f"{path}?{query_params}" if query_params else path

        try:
            # Process request
            response = await call_next(request)

            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000

            # Log successful request
            # Health checks at DEBUG level to reduce spam
            if path.endswith("/health"):
                self.logger.debug(
                    "http_in",
                    request_id=request_id,
                    method=method,
                    path=full_path,
                    status_code=response.status_code,
                    duration_ms=duration_ms,
                )
            else:
                self.logger.http_in(
                    method=method,
                    path=full_path,
                    remote_addr=remote_addr,
                    request_id=request_id,
                    headers=headers,
                    body=body,
                    status_code=response.status_code,
                    duration_ms=duration_ms,
                )

            return response

        except Exception as e:
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000

            # Log error
            error_msg = f"{type(e).__name__}: {str(e)}"
            stack = traceback.format_exc()

            self.logger.http_in(
                method=method,
                path=full_path,
                remote_addr=remote_addr,
                request_id=request_id,
                headers=headers,
                body=body,
                duration_ms=duration_ms,
                error=error_msg,
            )

            self.logger.error(
                "request_exception",
                request_id=request_id,
                error=error_msg,
                stack_trace=stack,
                method=method,
                path=full_path,
            )

            # Return 500 error
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "request_id": request_id,
                    "details": str(e),
                }
            )


class WebSocketLoggingMiddleware:
    """
    Middleware for logging WebSocket connections.

    Note: This is not a standard middleware but provides logging utilities
    for WebSocket endpoints.
    """

    def __init__(self):
        self.logger = get_logger()

    def log_connect(self, websocket: WebSocket, client_id: str):
        """Log WebSocket connection."""
        client_addr = websocket.client.host if websocket.client else "unknown"

        self.logger.info(
            "websocket_connect",
            client_id=client_id,
            client_addr=client_addr,
            path=str(websocket.url.path),
        )

    def log_disconnect(self, client_id: str, reason: str = "normal"):
        """Log WebSocket disconnection."""
        self.logger.info(
            "websocket_disconnect",
            client_id=client_id,
            reason=reason,
        )

    def log_error(self, client_id: str, error: str):
        """Log WebSocket error."""
        self.logger.error(
            "websocket_error",
            client_id=client_id,
            error=error,
            stack_trace=traceback.format_exc(),
        )

    def log_message(self, client_id: str, direction: str, message_type: str, size: int):
        """
        Log WebSocket message (DEBUG level).

        Args:
            client_id: WebSocket client ID
            direction: "send" or "receive"
            message_type: Type of message (e.g., "event", "ping", "pong")
            size: Message size in bytes
        """
        self.logger.debug(
            f"websocket_{direction}",
            client_id=client_id,
            message_type=message_type,
            size_bytes=size,
        )


# Global WebSocket logger instance
ws_logger = WebSocketLoggingMiddleware()
