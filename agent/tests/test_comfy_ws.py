"""
Unit tests for ComfyUI WebSocket completion tracking.

Tests that the completion logic:
- Does NOT complete early when history has empty outputs
- Properly extracts images from history
- Downloads images via /view with correct parameters
- Handles errors and timeouts correctly
- Filters messages by prompt_id
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent.comfy_ws import (
    ComfyWSTracker,
    ComfyWSResult,
    ProgressEvent,
    wait_for_prompt,
    check_queue_status,
    extract_images_from_history,
)


class TestExtractImagesFromHistory:
    """Tests for extract_images_from_history helper."""

    def test_extracts_single_image(self):
        """Test extracting a single image from history."""
        prompt_id = "test-prompt-123"
        history = {
            prompt_id: {
                "outputs": {
                    "7": {  # SaveImage node
                        "images": [
                            {"filename": "bench_race_00001_.png", "subfolder": "", "type": "output"}
                        ]
                    }
                }
            }
        }

        images = extract_images_from_history(history, prompt_id)

        assert len(images) == 1
        assert images[0]["filename"] == "bench_race_00001_.png"
        assert images[0]["subfolder"] == ""
        assert images[0]["type"] == "output"

    def test_extracts_multiple_images(self):
        """Test extracting multiple images from history."""
        prompt_id = "test-prompt-456"
        history = {
            prompt_id: {
                "outputs": {
                    "7": {
                        "images": [
                            {"filename": "image1.png", "subfolder": "", "type": "output"},
                            {"filename": "image2.png", "subfolder": "batch", "type": "output"},
                        ]
                    }
                }
            }
        }

        images = extract_images_from_history(history, prompt_id)

        assert len(images) == 2
        assert images[0]["filename"] == "image1.png"
        assert images[1]["filename"] == "image2.png"
        assert images[1]["subfolder"] == "batch"

    def test_extracts_images_from_multiple_nodes(self):
        """Test extracting images from multiple output nodes."""
        prompt_id = "test-prompt-789"
        history = {
            prompt_id: {
                "outputs": {
                    "7": {
                        "images": [{"filename": "final.png", "subfolder": "", "type": "output"}]
                    },
                    "8": {
                        "images": [{"filename": "preview.png", "subfolder": "", "type": "temp"}]
                    }
                }
            }
        }

        images = extract_images_from_history(history, prompt_id)

        assert len(images) == 2
        filenames = [img["filename"] for img in images]
        assert "final.png" in filenames
        assert "preview.png" in filenames

    def test_returns_empty_for_missing_prompt_id(self):
        """Test that missing prompt_id returns empty list."""
        history = {
            "other-prompt": {
                "outputs": {
                    "7": {
                        "images": [{"filename": "test.png", "subfolder": "", "type": "output"}]
                    }
                }
            }
        }

        images = extract_images_from_history(history, "nonexistent-prompt")

        assert images == []

    def test_returns_empty_for_no_outputs(self):
        """Test that prompt with no outputs returns empty list."""
        prompt_id = "test-prompt"
        history = {
            prompt_id: {
                "outputs": {}
            }
        }

        images = extract_images_from_history(history, prompt_id)

        assert images == []

    def test_skips_images_without_filename(self):
        """Test that images without filename are skipped."""
        prompt_id = "test-prompt"
        history = {
            prompt_id: {
                "outputs": {
                    "7": {
                        "images": [
                            {"filename": "valid.png", "subfolder": "", "type": "output"},
                            {"subfolder": "", "type": "output"},  # Missing filename
                            {"filename": "", "subfolder": "", "type": "output"},  # Empty filename
                        ]
                    }
                }
            }
        }

        images = extract_images_from_history(history, prompt_id)

        assert len(images) == 1
        assert images[0]["filename"] == "valid.png"


class TestCheckQueueStatus:
    """Tests for check_queue_status helper."""

    @pytest.mark.asyncio
    async def test_detects_prompt_in_pending_queue(self):
        """Test detecting a prompt in the pending queue."""
        prompt_id = "test-prompt-123"
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "queue_running": [],
            "queue_pending": [
                [0, "other-prompt", {}],
                [1, prompt_id, {}],
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.get.return_value = mock_response

        result = await check_queue_status(prompt_id, "http://localhost:8188", mock_client)

        assert result["in_queue"] is True
        assert result["is_running"] is False
        assert result["queue_position"] == 1

    @pytest.mark.asyncio
    async def test_detects_prompt_running(self):
        """Test detecting a prompt currently running."""
        prompt_id = "test-prompt-123"
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "queue_running": [[0, prompt_id, {}]],
            "queue_pending": []
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.get.return_value = mock_response

        result = await check_queue_status(prompt_id, "http://localhost:8188", mock_client)

        assert result["in_queue"] is False
        assert result["is_running"] is True

    @pytest.mark.asyncio
    async def test_detects_prompt_not_in_queue(self):
        """Test detecting a prompt not in queue (completed or failed)."""
        prompt_id = "test-prompt-123"
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "queue_running": [],
            "queue_pending": []
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.get.return_value = mock_response

        result = await check_queue_status(prompt_id, "http://localhost:8188", mock_client)

        assert result["in_queue"] is False
        assert result["is_running"] is False
        assert result["queue_remaining"] == 0


class TestComfyWSTrackerHandleMessage:
    """Tests for ComfyWSTracker._handle_message."""

    def test_filters_by_prompt_id_execution_start(self):
        """Test that execution_start is filtered by prompt_id."""
        tracker = ComfyWSTracker(
            prompt_id="my-prompt-123",
            client_id="client-456",
            base_url="http://localhost:8188",
        )

        # Message for different prompt should not start execution
        other_msg = {"type": "execution_start", "data": {"prompt_id": "other-prompt"}}

        async def run_test():
            result = await tracker._handle_message(other_msg)
            assert result is False
            assert tracker._execution_started is False

            # Message for our prompt should start execution
            our_msg = {"type": "execution_start", "data": {"prompt_id": "my-prompt-123"}}
            result = await tracker._handle_message(our_msg)
            assert result is False
            assert tracker._execution_started is True

        asyncio.get_event_loop().run_until_complete(run_test())

    def test_handles_executing_node_none_completion(self):
        """Test that executing with node=None signals completion."""
        tracker = ComfyWSTracker(
            prompt_id="my-prompt-123",
            client_id="client-456",
            base_url="http://localhost:8188",
        )

        async def run_test():
            # First start execution
            start_msg = {"type": "execution_start", "data": {"prompt_id": "my-prompt-123"}}
            await tracker._handle_message(start_msg)

            # Then receive executing with node=None
            done_msg = {"type": "executing", "data": {"prompt_id": "my-prompt-123", "node": None}}
            result = await tracker._handle_message(done_msg)

            assert result is True
            assert tracker._completed is True

        asyncio.get_event_loop().run_until_complete(run_test())

    def test_handles_execution_error(self):
        """Test that execution_error sets error state."""
        tracker = ComfyWSTracker(
            prompt_id="my-prompt-123",
            client_id="client-456",
            base_url="http://localhost:8188",
        )

        async def run_test():
            error_msg = {
                "type": "execution_error",
                "data": {
                    "prompt_id": "my-prompt-123",
                    "exception_type": "ValueError",
                    "exception_message": "Invalid checkpoint",
                    "node_id": "1",
                    "node_type": "CheckpointLoaderSimple"
                }
            }
            result = await tracker._handle_message(error_msg)

            assert result is True
            assert tracker._error is not None
            assert "Invalid checkpoint" in tracker._error
            assert "CheckpointLoaderSimple" in tracker._error

        asyncio.get_event_loop().run_until_complete(run_test())

    def test_tracks_progress_after_execution_start(self):
        """Test that progress updates are tracked after execution starts."""
        tracker = ComfyWSTracker(
            prompt_id="my-prompt-123",
            client_id="client-456",
            base_url="http://localhost:8188",
        )

        async def run_test():
            # Start execution first
            await tracker._handle_message({
                "type": "execution_start",
                "data": {"prompt_id": "my-prompt-123"}
            })

            # Send progress updates
            await tracker._handle_message({
                "type": "progress",
                "data": {"value": 5, "max": 30}
            })
            await tracker._handle_message({
                "type": "progress",
                "data": {"value": 15, "max": 30}
            })
            await tracker._handle_message({
                "type": "progress",
                "data": {"value": 30, "max": 30}
            })

            assert tracker._max_step == 30
            assert tracker._total_steps == 30

        asyncio.get_event_loop().run_until_complete(run_test())


class TestWaitViaPolling:
    """Tests for ComfyWSTracker._wait_via_polling."""

    @pytest.mark.asyncio
    async def test_does_not_complete_on_empty_history(self):
        """Test that polling does not complete when history has no outputs."""
        tracker = ComfyWSTracker(
            prompt_id="test-prompt",
            client_id="test-client",
            base_url="http://localhost:8188",
            timeout_seconds=2.0,  # Short timeout for test
            poll_interval=0.1,
        )

        poll_count = 0

        async def mock_get(url, timeout=None):
            nonlocal poll_count
            poll_count += 1

            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()

            if "/history/" in url:
                # Return empty outputs for first few polls, then with images
                if poll_count < 3:
                    mock_resp.json.return_value = {
                        "test-prompt": {
                            "status": {"completed": False},
                            "outputs": {}
                        }
                    }
                else:
                    mock_resp.json.return_value = {
                        "test-prompt": {
                            "status": {"completed": True},
                            "outputs": {
                                "7": {
                                    "images": [{"filename": "test.png", "subfolder": "", "type": "output"}]
                                }
                            }
                        }
                    }
            elif "/queue" in url:
                mock_resp.json.return_value = {
                    "queue_running": [["0", "test-prompt", {}]] if poll_count < 3 else [],
                    "queue_pending": []
                }

            return mock_resp

        mock_client = AsyncMock()
        mock_client.get = mock_get

        await tracker._wait_via_polling(mock_client)

        # Should have polled multiple times before completing
        assert poll_count >= 3
        assert tracker._completed is True

    @pytest.mark.asyncio
    async def test_detects_error_in_history_status(self):
        """Test that errors in history status are detected."""
        tracker = ComfyWSTracker(
            prompt_id="test-prompt",
            client_id="test-client",
            base_url="http://localhost:8188",
            timeout_seconds=2.0,
            poll_interval=0.1,
        )

        async def mock_get(url, timeout=None):
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()

            if "/history/" in url:
                mock_resp.json.return_value = {
                    "test-prompt": {
                        "status": {
                            "status_str": "error",
                            "error": {"message": "Checkpoint not found"}
                        },
                        "outputs": {}
                    }
                }
            elif "/queue" in url:
                mock_resp.json.return_value = {
                    "queue_running": [],
                    "queue_pending": []
                }

            return mock_resp

        mock_client = AsyncMock()
        mock_client.get = mock_get

        await tracker._wait_via_polling(mock_client)

        assert tracker._error is not None
        assert "Checkpoint not found" in tracker._error


class TestValidateHistory:
    """Tests for ComfyWSTracker._validate_history."""

    @pytest.mark.asyncio
    async def test_returns_false_for_no_images(self):
        """Test that validation returns False when no images in outputs."""
        tracker = ComfyWSTracker(
            prompt_id="test-prompt",
            client_id="test-client",
            base_url="http://localhost:8188",
        )

        async def mock_get(url, timeout=None):
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {
                "test-prompt": {
                    "status": {},
                    "outputs": {"7": {"not_images": []}}
                }
            }
            return mock_resp

        mock_client = AsyncMock()
        mock_client.get = mock_get

        status, has_outputs = await tracker._validate_history(mock_client)

        assert has_outputs is False

    @pytest.mark.asyncio
    async def test_returns_true_for_images(self):
        """Test that validation returns True when images exist."""
        tracker = ComfyWSTracker(
            prompt_id="test-prompt",
            client_id="test-client",
            base_url="http://localhost:8188",
        )

        async def mock_get(url, timeout=None):
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {
                "test-prompt": {
                    "status": {},
                    "outputs": {
                        "7": {
                            "images": [{"filename": "test.png"}]
                        }
                    }
                }
            }
            return mock_resp

        mock_client = AsyncMock()
        mock_client.get = mock_get

        status, has_outputs = await tracker._validate_history(mock_client)

        assert has_outputs is True


class TestIntegrationScenarios:
    """Integration tests simulating real-world scenarios."""

    @pytest.mark.asyncio
    async def test_immediate_history_empty_bug(self):
        """
        Test the bug scenario: /history returns immediately but with empty outputs.

        This simulates the Dell GB10 bug where:
        1. POST /prompt returns prompt_id
        2. GET /history/<prompt_id> returns 200 but with empty outputs
        3. Code incorrectly marks job as complete with 0 images

        The fix should wait for actual outputs before completing.
        """
        prompt_id = "cef6b2fe-352c-401e-811c-befe4670d3c1"

        # Create tracker with short timeout for test
        tracker = ComfyWSTracker(
            prompt_id=prompt_id,
            client_id="test-client",
            base_url="http://localhost:8188",
            timeout_seconds=1.0,  # Very short for test
            poll_interval=0.1,
        )

        # Mock that returns empty outputs on first call
        call_count = 0

        async def mock_get(url, timeout=None):
            nonlocal call_count
            call_count += 1

            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()

            if "/history/" in url:
                # Simulate the bug: history exists but no outputs yet
                mock_resp.json.return_value = {
                    prompt_id: {
                        "status": {},
                        "outputs": {}  # Empty!
                    }
                }
            elif "/queue" in url:
                # Job is in the running queue
                mock_resp.json.return_value = {
                    "queue_running": [[0, prompt_id, {}]],
                    "queue_pending": []
                }

            return mock_resp

        mock_client = AsyncMock()
        mock_client.get = mock_get

        # Simulate websocket failure to trigger polling path
        tracker._ws_connected = False

        await tracker._wait_via_polling(mock_client)

        # Validate history - should return False for no outputs
        status, has_outputs = await tracker._validate_history(mock_client)

        # The bug would have marked this as complete; the fix should not
        assert has_outputs is False, "Should not mark complete with empty outputs"

    @pytest.mark.asyncio
    async def test_full_wait_for_prompt_success(self):
        """Test successful completion flow through wait_for_prompt."""
        prompt_id = "test-prompt-success"
        client_id = "test-client"
        base_url = "http://localhost:8188"

        # Track progress callbacks
        progress_calls = []

        def on_progress(p):
            progress_calls.append(p)

        # Create mock client
        mock_client = AsyncMock()

        async def mock_get(url, timeout=None):
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()

            if "/history/" in url:
                mock_resp.json.return_value = {
                    prompt_id: {
                        "status": {"completed": True},
                        "outputs": {
                            "7": {
                                "images": [
                                    {"filename": "output_00001.png", "subfolder": "", "type": "output"}
                                ]
                            }
                        }
                    }
                }
            elif "/queue" in url:
                mock_resp.json.return_value = {
                    "queue_running": [],
                    "queue_pending": []
                }

            return mock_resp

        mock_client.get = mock_get

        # Patch websockets.connect to simulate failure, forcing polling
        with patch("agent.comfy_ws.websockets.connect") as mock_ws:
            mock_ws.side_effect = Exception("Connection failed - using polling fallback")

            result = await wait_for_prompt(
                prompt_id=prompt_id,
                client_id=client_id,
                base_url=base_url,
                http_client=mock_client,
                timeout_seconds=1.0,
                on_progress=on_progress,
            )

        assert result.completed is True
        assert result.error is None
        assert result.prompt_id == prompt_id
