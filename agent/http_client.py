"""
Instrumented HTTP client wrapper for bench-race agents.

Wraps httpx with structured logging for all outbound HTTP requests
to ComfyUI, Ollama, and other services.
"""

import uuid
from typing import Any, Dict, Optional
import httpx
from contextlib import asynccontextmanager

from agent.logging_utils import get_logger, timer


class LoggedHTTPClient:
    """
    HTTP client that logs all requests and responses.

    Wraps httpx.AsyncClient with automatic logging of:
    - Request method, URL, headers, body
    - Response status, body, duration
    - Errors and exceptions
    """

    def __init__(
        self,
        service: str,
        base_url: Optional[str] = None,
        timeout: Optional[httpx.Timeout] = None,
        **client_kwargs
    ):
        """
        Initialize logged HTTP client.

        Args:
            service: Service name for logging (e.g., "comfyui", "ollama", "central")
            base_url: Base URL for the service
            timeout: Request timeout configuration
            **client_kwargs: Additional arguments for httpx.AsyncClient
        """
        self.service = service
        self.logger = get_logger()

        # Build client kwargs
        kwargs = client_kwargs.copy()
        if base_url:
            kwargs["base_url"] = base_url
        if timeout:
            kwargs["timeout"] = timeout

        self._client_kwargs = kwargs
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self._client = httpx.AsyncClient(**self._client_kwargs)
        await self._client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get the underlying httpx client."""
        if self._client is None:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")
        return self._client

    async def request(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> httpx.Response:
        """
        Make an HTTP request with logging.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: URL to request
            **kwargs: Additional arguments for httpx.request

        Returns:
            httpx.Response object
        """
        client = self._get_client()
        request_id = str(uuid.uuid4())[:8]

        # Extract request details for logging
        timeout = kwargs.get("timeout")
        if isinstance(timeout, httpx.Timeout):
            timeout_value = timeout.read or timeout.connect
        else:
            timeout_value = timeout

        request_body = kwargs.get("json") or kwargs.get("data") or kwargs.get("content")

        # Log the request
        with timer() as t:
            try:
                response = await client.request(method, url, **kwargs)

                # Log successful response
                self.logger.http_out(
                    service=self.service,
                    method=method,
                    url=str(url),
                    request_id=request_id,
                    timeout=timeout_value,
                    request_body=request_body,
                    status_code=response.status_code,
                    response_body=response.text if response.status_code >= 400 else None,
                    duration_ms=t.elapsed_ms,
                )

                return response

            except httpx.TimeoutException as e:
                # Log timeout error
                self.logger.http_out(
                    service=self.service,
                    method=method,
                    url=str(url),
                    request_id=request_id,
                    timeout=timeout_value,
                    request_body=request_body,
                    duration_ms=t.elapsed_ms,
                    error=f"Timeout: {str(e)}",
                )
                raise

            except httpx.ConnectError as e:
                # Log connection error
                self.logger.http_out(
                    service=self.service,
                    method=method,
                    url=str(url),
                    request_id=request_id,
                    timeout=timeout_value,
                    request_body=request_body,
                    duration_ms=t.elapsed_ms,
                    error=f"Connection error: {str(e)}",
                )
                raise

            except Exception as e:
                # Log other errors
                self.logger.http_out(
                    service=self.service,
                    method=method,
                    url=str(url),
                    request_id=request_id,
                    timeout=timeout_value,
                    request_body=request_body,
                    duration_ms=t.elapsed_ms,
                    error=str(e),
                )
                raise

    async def get(self, url: str, **kwargs) -> httpx.Response:
        """Make a GET request."""
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs) -> httpx.Response:
        """Make a POST request."""
        return await self.request("POST", url, **kwargs)

    async def put(self, url: str, **kwargs) -> httpx.Response:
        """Make a PUT request."""
        return await self.request("PUT", url, **kwargs)

    async def delete(self, url: str, **kwargs) -> httpx.Response:
        """Make a DELETE request."""
        return await self.request("DELETE", url, **kwargs)

    @asynccontextmanager
    async def stream(
        self,
        method: str,
        url: str,
        **kwargs
    ):
        """
        Stream an HTTP request with logging.

        Args:
            method: HTTP method
            url: URL to request
            **kwargs: Additional arguments for httpx.stream

        Yields:
            httpx.Response object in streaming mode
        """
        client = self._get_client()
        request_id = str(uuid.uuid4())[:8]

        # Extract request details for logging
        timeout = kwargs.get("timeout")
        if isinstance(timeout, httpx.Timeout):
            timeout_value = timeout.read or timeout.connect
        else:
            timeout_value = timeout

        request_body = kwargs.get("json") or kwargs.get("data") or kwargs.get("content")

        # Log the request start
        start_time = 0
        with timer() as t:
            try:
                async with client.stream(method, url, **kwargs) as response:
                    # Log stream start
                    self.logger.http_out(
                        service=self.service,
                        method=method,
                        url=str(url),
                        request_id=request_id,
                        timeout=timeout_value,
                        request_body=request_body,
                        status_code=response.status_code,
                        duration_ms=t.elapsed_ms,
                    )

                    yield response

            except httpx.TimeoutException as e:
                self.logger.http_out(
                    service=self.service,
                    method=method,
                    url=str(url),
                    request_id=request_id,
                    timeout=timeout_value,
                    request_body=request_body,
                    duration_ms=t.elapsed_ms,
                    error=f"Timeout: {str(e)}",
                )
                raise

            except httpx.ConnectError as e:
                self.logger.http_out(
                    service=self.service,
                    method=method,
                    url=str(url),
                    request_id=request_id,
                    timeout=timeout_value,
                    request_body=request_body,
                    duration_ms=t.elapsed_ms,
                    error=f"Connection error: {str(e)}",
                )
                raise

            except Exception as e:
                self.logger.http_out(
                    service=self.service,
                    method=method,
                    url=str(url),
                    request_id=request_id,
                    timeout=timeout_value,
                    request_body=request_body,
                    duration_ms=t.elapsed_ms,
                    error=str(e),
                )
                raise


# Convenience functions for creating service-specific clients

def comfyui_client(base_url: str, timeout: Optional[httpx.Timeout] = None) -> LoggedHTTPClient:
    """Create a logged HTTP client for ComfyUI."""
    return LoggedHTTPClient(
        service="comfyui",
        base_url=base_url,
        timeout=timeout or httpx.Timeout(connect=10.0, read=300.0, write=60.0, pool=10.0),
    )


def ollama_client(base_url: str, timeout: Optional[httpx.Timeout] = None) -> LoggedHTTPClient:
    """Create a logged HTTP client for Ollama."""
    return LoggedHTTPClient(
        service="ollama",
        base_url=base_url,
        timeout=timeout or httpx.Timeout(connect=5.0, read=None, write=60.0, pool=10.0),
    )


def central_client(base_url: str, timeout: Optional[httpx.Timeout] = None) -> LoggedHTTPClient:
    """Create a logged HTTP client for central server."""
    return LoggedHTTPClient(
        service="central",
        base_url=base_url,
        timeout=timeout or httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=10.0),
    )
