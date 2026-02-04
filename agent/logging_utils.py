"""
Structured logging utility for bench-race agents.

Provides consistent JSON logging for:
- HTTP requests (inbound/outbound)
- Job lifecycle events
- Progress tracking
- Error handling

Environment variables:
- LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR) [default: INFO]
- LOG_JSON: Enable JSON output (1) or pretty text (0) [default: 1]
- LOG_HTTP_BODY: Include request/response bodies in logs [default: 0]
- LOG_HTTP_MAXLEN: Max length for HTTP body logging [default: 2000]
"""

import json
import logging
import os
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from contextlib import contextmanager

# Configuration from environment
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_JSON = os.getenv("LOG_JSON", "1") == "1"
LOG_HTTP_BODY = os.getenv("LOG_HTTP_BODY", "0") == "1"
LOG_HTTP_MAXLEN = int(os.getenv("LOG_HTTP_MAXLEN", "2000"))

# Get hostname for agent identification
HOSTNAME = socket.gethostname()

# Agent ID (will be set from config)
AGENT_ID: Optional[str] = None


def set_agent_id(agent_id: str):
    """Set the global agent ID for logging."""
    global AGENT_ID
    AGENT_ID = agent_id


class StructuredLogger:
    """
    Structured logger that outputs JSON logs to stdout.

    Each log line includes:
    - ts: ISO8601 timestamp
    - level: Log level (DEBUG, INFO, WARNING, ERROR)
    - event: Event name
    - agent_id: Agent identifier
    - hostname: Machine hostname
    - Additional context fields
    """

    def __init__(self, name: str = "bench-agent"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, LOG_LEVEL))

        # Remove existing handlers
        self.logger.handlers.clear()

        # Add stdout handler with our custom formatter
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, LOG_LEVEL))

        if LOG_JSON:
            handler.setFormatter(JSONFormatter())
        else:
            handler.setFormatter(PrettyFormatter())

        self.logger.addHandler(handler)
        self.logger.propagate = False

    def _truncate_body(self, body: Any, max_len: int = LOG_HTTP_MAXLEN) -> str:
        """Truncate body for logging."""
        if body is None:
            return None

        body_str = str(body)
        if len(body_str) > max_len:
            return body_str[:max_len] + f"... (truncated, {len(body_str)} total chars)"
        return body_str

    def _sanitize_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Remove sensitive headers like Authorization tokens."""
        if not headers:
            return {}

        sanitized = {}
        sensitive_keys = {"authorization", "x-api-key", "api-key", "token"}

        for key, value in headers.items():
            if key.lower() in sensitive_keys:
                sanitized[key] = "***REDACTED***"
            else:
                sanitized[key] = value

        return sanitized

    def log(
        self,
        level: str,
        event: str,
        job_id: Optional[str] = None,
        request_id: Optional[str] = None,
        duration_ms: Optional[float] = None,
        error: Optional[str] = None,
        stack_trace: Optional[str] = None,
        **details
    ):
        """
        Log a structured event.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR)
            event: Event name (e.g., "job_received", "http_out")
            job_id: Job identifier (if applicable)
            request_id: Request identifier (if applicable)
            duration_ms: Duration in milliseconds (if applicable)
            error: Error message (if applicable)
            stack_trace: Stack trace (if applicable)
            **details: Additional event-specific fields
        """
        log_data = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "event": event,
            "agent_id": AGENT_ID,
            "hostname": HOSTNAME,
        }

        if job_id:
            log_data["job_id"] = job_id
        if request_id:
            log_data["request_id"] = request_id
        if duration_ms is not None:
            log_data["duration_ms"] = round(duration_ms, 2)
        if error:
            log_data["error"] = error
        if stack_trace:
            log_data["stack_trace"] = stack_trace

        # Add additional details
        if details:
            log_data["details"] = details

        # Log at appropriate level
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method("", extra={"structured": log_data})

    def debug(self, event: str, **kwargs):
        """Log at DEBUG level."""
        self.log("DEBUG", event, **kwargs)

    def info(self, event: str, **kwargs):
        """Log at INFO level."""
        self.log("INFO", event, **kwargs)

    def warning(self, event: str, **kwargs):
        """Log at WARNING level."""
        self.log("WARNING", event, **kwargs)

    def error(self, event: str, **kwargs):
        """Log at ERROR level."""
        self.log("ERROR", event, **kwargs)

    def http_in(
        self,
        method: str,
        path: str,
        remote_addr: str,
        request_id: str,
        headers: Optional[Dict[str, str]] = None,
        body: Any = None,
        status_code: Optional[int] = None,
        duration_ms: Optional[float] = None,
        error: Optional[str] = None,
    ):
        """Log inbound HTTP request."""
        details = {
            "method": method,
            "path": path,
            "remote_addr": remote_addr,
        }

        if headers:
            details["headers"] = self._sanitize_headers(headers)

        if LOG_HTTP_BODY and body is not None:
            details["request_body"] = self._truncate_body(body)

        if status_code is not None:
            details["status_code"] = status_code

        if error:
            self.error(
                "http_in_error",
                request_id=request_id,
                duration_ms=duration_ms,
                error=error,
                **details
            )
        else:
            self.info(
                "http_in",
                request_id=request_id,
                duration_ms=duration_ms,
                **details
            )

    def http_out(
        self,
        service: str,
        method: str,
        url: str,
        request_id: str,
        timeout: Optional[float] = None,
        request_body: Any = None,
        status_code: Optional[int] = None,
        response_body: Any = None,
        duration_ms: Optional[float] = None,
        error: Optional[str] = None,
    ):
        """Log outbound HTTP request."""
        details = {
            "service": service,
            "method": method,
            "url": url,
        }

        if timeout is not None:
            details["timeout"] = timeout

        if LOG_HTTP_BODY and request_body is not None:
            details["request_body"] = self._truncate_body(request_body)

        if status_code is not None:
            details["status_code"] = status_code

        if LOG_HTTP_BODY and response_body is not None:
            details["response_body"] = self._truncate_body(response_body)

        if error:
            self.error(
                "http_out_error",
                request_id=request_id,
                duration_ms=duration_ms,
                error=error,
                **details
            )
        else:
            self.info(
                "http_out",
                request_id=request_id,
                duration_ms=duration_ms,
                **details
            )

    @contextmanager
    def job_context(self, job_id: str, job_type: str, **initial_details):
        """
        Context manager for tracking job lifecycle.

        Usage:
            with logger.job_context(job_id="123", job_type="llm") as ctx:
                ctx.milestone("job_started")
                # do work
                ctx.milestone("job_completed", tokens=100)
        """
        start_time = time.time()

        class JobContext:
            def __init__(self, logger: StructuredLogger, job_id: str, job_type: str):
                self.logger = logger
                self.job_id = job_id
                self.job_type = job_type
                self.start_time = start_time

            def milestone(self, event: str, **details):
                """Log a job milestone."""
                duration_ms = (time.time() - self.start_time) * 1000
                self.logger.info(
                    event,
                    job_id=self.job_id,
                    duration_ms=duration_ms,
                    job_type=self.job_type,
                    **details
                )

            def error(self, event: str, error: str, **details):
                """Log a job error."""
                duration_ms = (time.time() - self.start_time) * 1000
                self.logger.error(
                    event,
                    job_id=self.job_id,
                    duration_ms=duration_ms,
                    job_type=self.job_type,
                    error=error,
                    stack_trace=traceback.format_exc(),
                    **details
                )

        ctx = JobContext(self, job_id, job_type)
        self.info(
            "job_received",
            job_id=job_id,
            job_type=job_type,
            **initial_details
        )

        try:
            yield ctx
        except Exception as e:
            ctx.error("job_failed", error=str(e))
            raise


class JSONFormatter(logging.Formatter):
    """Formats log records as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        if hasattr(record, "structured"):
            return json.dumps(record.structured)

        # Fallback for non-structured logs
        log_data = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class PrettyFormatter(logging.Formatter):
    """Formats log records as human-readable text."""

    def format(self, record: logging.LogRecord) -> str:
        if hasattr(record, "structured"):
            data = record.structured
            timestamp = data.get("ts", "")[:19]  # Trim microseconds
            level = data.get("level", "INFO")
            event = data.get("event", "")
            job_id = data.get("job_id", "")

            parts = [f"[{timestamp}]", f"[{level}]", f"[{event}]"]

            if job_id:
                parts.append(f"[job:{job_id[:8]}]")

            details = data.get("details", {})
            if details:
                details_str = " ".join(f"{k}={v}" for k, v in details.items())
                parts.append(details_str)

            if data.get("error"):
                parts.append(f"ERROR: {data['error']}")

            return " ".join(parts)

        return super().format(record)


# Global logger instance
_logger: Optional[StructuredLogger] = None


def get_logger() -> StructuredLogger:
    """Get the global structured logger instance."""
    global _logger
    if _logger is None:
        _logger = StructuredLogger()
    return _logger


def init_logging(agent_id: Optional[str] = None):
    """
    Initialize logging system.

    Args:
        agent_id: Optional agent identifier
    """
    if agent_id:
        set_agent_id(agent_id)

    logger = get_logger()

    # Log configuration on startup
    logger.info(
        "logger_config",
        log_level=LOG_LEVEL,
        log_json=LOG_JSON,
        log_http_body=LOG_HTTP_BODY,
        log_http_maxlen=LOG_HTTP_MAXLEN,
        hostname=HOSTNAME,
    )

    return logger


# Timer context manager for measuring durations
@contextmanager
def timer():
    """
    Context manager to measure duration.

    Usage:
        with timer() as t:
            # do work
        duration_ms = t.elapsed_ms
    """
    class Timer:
        def __init__(self):
            self.start = time.time()
            self.elapsed_ms = 0

        def stop(self):
            self.elapsed_ms = (time.time() - self.start) * 1000
            return self.elapsed_ms

    t = Timer()
    try:
        yield t
    finally:
        t.stop()
