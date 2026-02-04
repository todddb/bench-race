"""
ComfyUI WebSocket helper for tracking job completion and progress.

This module provides proper completion detection for ComfyUI jobs by:
1. Listening for websocket events from ComfyUI
2. Filtering messages by prompt_id to track only our job
3. Falling back to history polling if websocket fails
4. Validating that history contains actual outputs before marking complete
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

log = logging.getLogger("bench-agent.comfy_ws")


@dataclass
class ComfyWSResult:
    """Result of waiting for a ComfyUI prompt to complete."""
    completed: bool = False
    error: Optional[str] = None
    max_step: int = 0
    total_steps: int = 0
    events_seen: int = 0
    execution_time_ms: float = 0.0
    timed_out: bool = False
    prompt_id: str = ""
    # Debug info
    history_status: Optional[Dict] = None
    queue_position: Optional[int] = None
    was_in_queue: bool = False


@dataclass
class ProgressEvent:
    """Progress event from ComfyUI."""
    current_step: int
    total_steps: int
    node_id: Optional[str] = None


@dataclass
class ComfyWSTracker:
    """
    Tracks ComfyUI job execution via websocket and/or polling.

    ComfyUI websocket message types:
    - status: Queue status update (includes queue info)
    - execution_start: Job started executing (includes prompt_id)
    - executing: Currently executing a node (includes prompt_id, node)
    - progress: Sampler progress (includes value, max)
    - executed: Node completed (includes prompt_id, node, output)
    - execution_error: Execution failed (includes prompt_id, exception info)
    - execution_cached: Nodes were cached (includes prompt_id, nodes)
    - execution_interrupted: Job was interrupted
    """
    prompt_id: str
    client_id: str
    base_url: str
    timeout_seconds: float = 300.0  # 5 minutes default
    poll_interval: float = 0.5  # How often to poll /history

    # Callbacks
    on_progress: Optional[Callable[[ProgressEvent], None]] = None
    on_preview: Optional[Callable[[bytes, int, int], None]] = None

    # Internal state
    _completed: bool = field(default=False, init=False)
    _error: Optional[str] = field(default=None, init=False)
    _max_step: int = field(default=0, init=False)
    _total_steps: int = field(default=0, init=False)
    _events_seen: int = field(default=0, init=False)
    _start_time: float = field(default=0.0, init=False)
    _execution_started: bool = field(default=False, init=False)
    _ws_connected: bool = field(default=False, init=False)

    def _get_ws_url(self) -> str:
        """Convert HTTP URL to WebSocket URL."""
        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        return f"{ws_url}/ws?clientId={self.client_id}"

    async def _handle_message(self, data: Dict[str, Any]) -> bool:
        """
        Handle a parsed websocket message.

        Returns True if we should stop listening (completion or error).
        """
        msg_type = data.get("type", "")
        msg_data = data.get("data", {})
        self._events_seen += 1

        # Log all messages for debugging
        log.debug(f"ComfyUI WS message: type={msg_type}, prompt_id in msg={msg_data.get('prompt_id')}, our_prompt_id={self.prompt_id}")

        if msg_type == "status":
            # Queue status update - check if our prompt is in queue
            queue_info = msg_data.get("status", {}).get("exec_info", {})
            queue_remaining = queue_info.get("queue_remaining", 0)
            log.debug(f"Queue status: {queue_remaining} remaining")
            return False

        if msg_type == "execution_start":
            # Check if this is for our prompt
            msg_prompt_id = msg_data.get("prompt_id", "")
            if msg_prompt_id == self.prompt_id:
                self._execution_started = True
                log.info(f"Execution started for prompt {self.prompt_id}")
            return False

        if msg_type == "executing":
            # Node is being executed - check prompt_id
            msg_prompt_id = msg_data.get("prompt_id")
            if msg_prompt_id == self.prompt_id:
                node_id = msg_data.get("node")
                if node_id is None:
                    # When node is None, it means execution for this prompt finished
                    log.info(f"Execution finished for prompt {self.prompt_id} (node=None signal)")
                    self._completed = True
                    return True
                else:
                    log.debug(f"Executing node {node_id} for prompt {self.prompt_id}")
            return False

        if msg_type == "progress":
            # Sampler progress - this doesn't include prompt_id but occurs during our execution
            if self._execution_started:
                value = msg_data.get("value", 0)
                max_val = msg_data.get("max", 0)
                node_id = msg_data.get("node")

                self._max_step = max(self._max_step, value)
                self._total_steps = max(self._total_steps, max_val)

                if self.on_progress:
                    try:
                        self.on_progress(ProgressEvent(
                            current_step=value,
                            total_steps=max_val,
                            node_id=node_id,
                        ))
                    except Exception as e:
                        log.warning(f"Progress callback error: {e}")
            return False

        if msg_type == "executed":
            # A node finished execution
            msg_prompt_id = msg_data.get("prompt_id")
            if msg_prompt_id == self.prompt_id:
                node_id = msg_data.get("node")
                output = msg_data.get("output", {})
                log.debug(f"Node {node_id} executed for prompt {self.prompt_id}, output keys: {list(output.keys())}")
                # Don't mark complete yet - wait for executing with node=None
            return False

        if msg_type == "execution_cached":
            # Some nodes were cached
            msg_prompt_id = msg_data.get("prompt_id")
            if msg_prompt_id == self.prompt_id:
                nodes = msg_data.get("nodes", [])
                log.debug(f"Cached nodes for prompt {self.prompt_id}: {nodes}")
            return False

        if msg_type == "execution_error":
            # Execution failed
            msg_prompt_id = msg_data.get("prompt_id")
            if msg_prompt_id == self.prompt_id:
                exception_type = msg_data.get("exception_type", "Unknown")
                exception_message = msg_data.get("exception_message", "No message")
                node_id = msg_data.get("node_id")
                node_type = msg_data.get("node_type")

                error_parts = [f"ComfyUI execution error: {exception_type}: {exception_message}"]
                if node_type:
                    error_parts.append(f"Node: {node_type} (id={node_id})")

                self._error = " | ".join(error_parts)
                log.error(f"Execution error for prompt {self.prompt_id}: {self._error}")
                return True
            return False

        if msg_type == "execution_interrupted":
            msg_prompt_id = msg_data.get("prompt_id")
            if msg_prompt_id == self.prompt_id:
                self._error = "Execution was interrupted"
                log.warning(f"Execution interrupted for prompt {self.prompt_id}")
                return True
            return False

        return False

    async def wait_for_completion(
        self,
        http_client: httpx.AsyncClient,
        structured_logger: Any = None,
    ) -> ComfyWSResult:
        """
        Wait for the prompt to complete using websocket + polling fallback.

        This method:
        1. Connects to ComfyUI websocket
        2. Listens for completion events for our specific prompt_id
        3. Falls back to /history polling if websocket fails
        4. Validates that history contains outputs before returning success
        """
        self._start_time = time.perf_counter()
        ws_url = self._get_ws_url()

        try:
            await self._wait_via_websocket(ws_url, http_client, structured_logger)
        except Exception as e:
            log.warning(f"WebSocket tracking failed, falling back to polling: {e}")
            if not self._completed and not self._error:
                await self._wait_via_polling(http_client, structured_logger)

        execution_time_ms = (time.perf_counter() - self._start_time) * 1000.0

        # Final validation: check history for actual outputs
        history_status, has_outputs = await self._validate_history(http_client, structured_logger)

        result = ComfyWSResult(
            completed=self._completed and has_outputs and not self._error,
            error=self._error,
            max_step=self._max_step,
            total_steps=self._total_steps,
            events_seen=self._events_seen,
            execution_time_ms=execution_time_ms,
            timed_out=False,
            prompt_id=self.prompt_id,
            history_status=history_status,
        )

        # If we thought we completed but have no outputs, it's an error
        if self._completed and not has_outputs and not self._error:
            result.completed = False
            result.error = "Job reported complete but history contains no outputs"
            log.error(f"Prompt {self.prompt_id} completed but has no outputs in history")

        return result

    async def _wait_via_websocket(
        self,
        ws_url: str,
        http_client: httpx.AsyncClient,
        structured_logger: Any = None,
    ) -> None:
        """Wait for completion via websocket events."""
        deadline = time.perf_counter() + self.timeout_seconds

        log.info(f"Connecting to ComfyUI websocket: {ws_url}")

        async with websockets.connect(ws_url, ping_interval=20, ping_timeout=30) as ws:
            self._ws_connected = True
            log.info(f"Connected to ComfyUI websocket for prompt {self.prompt_id}")

            while time.perf_counter() < deadline:
                try:
                    # Use a shorter timeout for each receive to allow checking deadline
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)

                    if isinstance(msg, bytes):
                        # Binary message = preview image
                        if self._execution_started and self.on_preview:
                            try:
                                self.on_preview(msg, self._max_step, self._total_steps)
                            except Exception as e:
                                log.debug(f"Preview callback error: {e}")
                        continue

                    try:
                        data = json.loads(msg)
                    except json.JSONDecodeError:
                        log.debug(f"Non-JSON websocket message: {msg[:100]}")
                        continue

                    should_stop = await self._handle_message(data)
                    if should_stop:
                        return

                except asyncio.TimeoutError:
                    # Check if we should keep waiting
                    if time.perf_counter() >= deadline:
                        self._error = f"Timeout waiting for ComfyUI (>{self.timeout_seconds}s)"
                        return
                    continue
                except ConnectionClosed:
                    log.warning("ComfyUI websocket connection closed unexpectedly")
                    raise

        # If we get here, we timed out
        self._error = f"Timeout waiting for ComfyUI (>{self.timeout_seconds}s)"

    async def _wait_via_polling(
        self,
        http_client: httpx.AsyncClient,
        structured_logger: Any = None,
    ) -> None:
        """Fallback: poll /history until completion or timeout."""
        deadline = time.perf_counter() + self.timeout_seconds
        poll_count = 0

        log.info(f"Polling /history for prompt {self.prompt_id}")

        while time.perf_counter() < deadline:
            poll_count += 1

            try:
                # Check history
                history_resp = await http_client.get(
                    f"{self.base_url}/history/{self.prompt_id}",
                    timeout=10.0,
                )
                history_resp.raise_for_status()
                history = history_resp.json() or {}

                if self.prompt_id in history:
                    prompt_history = history[self.prompt_id]
                    status = prompt_history.get("status", {})
                    outputs = prompt_history.get("outputs", {})

                    # Check for completion
                    status_str = status.get("status_str", "")
                    completed = status.get("completed", False)

                    # Check if there are actual outputs
                    has_images = any(
                        "images" in node_outputs and node_outputs["images"]
                        for node_outputs in outputs.values()
                    )

                    # Check for errors in status
                    if "error" in status or status_str == "error":
                        error_msg = status.get("error", {})
                        if isinstance(error_msg, dict):
                            error_msg = error_msg.get("message", str(error_msg))
                        self._error = f"ComfyUI error: {error_msg}"
                        return

                    if completed or has_images:
                        self._completed = True
                        log.info(f"Prompt {self.prompt_id} completed via polling (poll #{poll_count})")
                        return

                # Also check queue to see if job is still pending
                queue_resp = await http_client.get(f"{self.base_url}/queue", timeout=5.0)
                queue_resp.raise_for_status()
                queue = queue_resp.json()

                # Check if prompt is in running or pending queue
                running = queue.get("queue_running", [])
                pending = queue.get("queue_pending", [])

                in_running = any(item[1] == self.prompt_id for item in running)
                in_pending = any(item[1] == self.prompt_id for item in pending)

                if not in_running and not in_pending:
                    # Not in queue and not in history - might have failed silently
                    if self.prompt_id not in history:
                        log.warning(f"Prompt {self.prompt_id} not in queue or history - checking more carefully")
                        # Give it a moment and check history again
                        await asyncio.sleep(0.5)
                        continue

                log.debug(
                    f"Polling {self.prompt_id}: in_queue={in_pending}, "
                    f"running={in_running}, poll_count={poll_count}"
                )

            except Exception as e:
                log.warning(f"Poll error: {e}")

            await asyncio.sleep(self.poll_interval)

        # Timed out
        self._error = f"Timeout waiting for ComfyUI via polling (>{self.timeout_seconds}s)"

    async def _validate_history(
        self,
        http_client: httpx.AsyncClient,
        structured_logger: Any = None,
    ) -> Tuple[Optional[Dict], bool]:
        """
        Validate that history contains actual outputs.

        Returns (history_status, has_outputs)
        """
        try:
            history_resp = await http_client.get(
                f"{self.base_url}/history/{self.prompt_id}",
                timeout=10.0,
            )
            history_resp.raise_for_status()
            history = history_resp.json() or {}

            if self.prompt_id not in history:
                log.warning(f"Prompt {self.prompt_id} not found in history after completion")
                return None, False

            prompt_history = history[self.prompt_id]
            status = prompt_history.get("status", {})
            outputs = prompt_history.get("outputs", {})

            # Count images in outputs
            image_count = 0
            for node_id, node_outputs in outputs.items():
                images = node_outputs.get("images", [])
                image_count += len(images)

            has_outputs = image_count > 0

            if not has_outputs:
                log.warning(
                    f"History for {self.prompt_id} has no images. "
                    f"Status: {status}, Output node IDs: {list(outputs.keys())}"
                )

            return status, has_outputs

        except Exception as e:
            log.error(f"Failed to validate history for {self.prompt_id}: {e}")
            return None, False


async def wait_for_prompt(
    prompt_id: str,
    client_id: str,
    base_url: str,
    http_client: httpx.AsyncClient,
    timeout_seconds: float = 300.0,
    on_progress: Optional[Callable[[ProgressEvent], None]] = None,
    on_preview: Optional[Callable[[bytes, int, int], None]] = None,
    structured_logger: Any = None,
) -> ComfyWSResult:
    """
    Convenience function to wait for a ComfyUI prompt to complete.

    Args:
        prompt_id: The prompt_id returned from POST /prompt
        client_id: UUID used when submitting the prompt
        base_url: ComfyUI base URL (e.g., http://127.0.0.1:8188)
        http_client: httpx.AsyncClient for HTTP requests
        timeout_seconds: Maximum time to wait
        on_progress: Callback for progress updates
        on_preview: Callback for preview images (bytes, step, total_steps)
        structured_logger: Optional structured logger for detailed logging

    Returns:
        ComfyWSResult with completion status, any errors, and step tracking info
    """
    tracker = ComfyWSTracker(
        prompt_id=prompt_id,
        client_id=client_id,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        on_progress=on_progress,
        on_preview=on_preview,
    )

    return await tracker.wait_for_completion(http_client, structured_logger)


async def check_queue_status(
    prompt_id: str,
    base_url: str,
    http_client: httpx.AsyncClient,
) -> Dict[str, Any]:
    """
    Check if a prompt is in the ComfyUI queue.

    Returns dict with:
        - in_queue: bool - prompt is in pending queue
        - is_running: bool - prompt is currently running
        - queue_position: int or None - position in pending queue
        - queue_remaining: int - total items in queue
    """
    try:
        resp = await http_client.get(f"{base_url}/queue", timeout=5.0)
        resp.raise_for_status()
        queue = resp.json()

        running = queue.get("queue_running", [])
        pending = queue.get("queue_pending", [])

        is_running = any(item[1] == prompt_id for item in running)

        queue_position = None
        for i, item in enumerate(pending):
            if item[1] == prompt_id:
                queue_position = i
                break

        in_queue = queue_position is not None

        return {
            "in_queue": in_queue,
            "is_running": is_running,
            "queue_position": queue_position,
            "queue_remaining": len(running) + len(pending),
        }
    except Exception as e:
        log.error(f"Failed to check queue status: {e}")
        return {
            "in_queue": False,
            "is_running": False,
            "queue_position": None,
            "queue_remaining": 0,
            "error": str(e),
        }


def extract_images_from_history(history: Dict, prompt_id: str) -> List[Dict[str, str]]:
    """
    Extract image information from ComfyUI history response.

    Returns list of dicts with keys: filename, subfolder, type
    """
    images = []

    if prompt_id not in history:
        return images

    prompt_history = history[prompt_id]
    outputs = prompt_history.get("outputs", {})

    for node_id, node_outputs in outputs.items():
        for image_info in node_outputs.get("images", []):
            filename = image_info.get("filename")
            if not filename:
                continue

            images.append({
                "filename": filename,
                "subfolder": image_info.get("subfolder", ""),
                "type": image_info.get("type", "output"),
            })

    return images
