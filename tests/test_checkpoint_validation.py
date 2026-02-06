from __future__ import annotations

import json
from types import SimpleNamespace
from unittest import mock

import central.app as central_app


class DummyResponse:
    def __init__(self, status_code=200, url="https://example.com/test.safetensors", headers=None, reason="OK"):
        self.status_code = status_code
        self.url = url
        self.headers = headers or {}
        self.reason = reason

    def close(self) -> None:
        return None


def test_checkpoint_filename_from_url():
    name = central_app._checkpoint_filename_from_url(
        "https://example.com/models/sd_xl_base_1.0.safetensors?download=1"
    )
    assert name == "sd_xl_base_1.0.safetensors"


def test_validate_checkpoint_url_caches(monkeypatch):
    calls = SimpleNamespace(count=0)

    def fake_head(url, allow_redirects=True, timeout=10):
        calls.count += 1
        return DummyResponse(headers={"Content-Length": "123"})

    monkeypatch.setattr(central_app.requests, "head", fake_head)
    central_app.CHECKPOINT_VALIDATION_CACHE.clear()

    url = "https://example.com/test.safetensors"
    first = central_app._validate_checkpoint_url(url)
    second = central_app._validate_checkpoint_url(url)

    assert calls.count == 1
    assert first["valid"] is True
    assert second["valid"] is True


def test_checkpoint_validation_allows_redirect_without_extension(monkeypatch):
    url = "https://huggingface.co/foo/bar/resolve/main/sd_xl_base_1.0.safetensors"
    resolved = "https://cas-bridge.example.com/3d6f7c00078891"
    headers = {
        "Content-Disposition": 'inline; filename="sd_xl_base_1.0.safetensors"',
        "Content-Length": "1234",
    }

    def fake_head(u, allow_redirects=True, timeout=10):
        assert u == url
        return DummyResponse(status_code=200, url=resolved, headers=headers)

    monkeypatch.setattr(central_app.requests, "head", fake_head)
    monkeypatch.setattr(
        central_app.requests, "get", lambda *a, **k: DummyResponse(status_code=200, url=resolved, headers=headers)
    )

    result = central_app._validate_checkpoint_url(url, force=True)
    assert result["valid"] is True
    assert result["error"] is None
    assert result["resolved_url"] == resolved


def test_image_run_allowed_with_synced_checkpoint_and_missing_url(monkeypatch):
    """
    Test that image runs are allowed when checkpoint is synced to agent
    but no URL entry exists in the config.

    This validates that URL lookup is only required for sync operations,
    not for run-time execution.
    """

    class MockHealthResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            # Agent reports checkpoint is available locally
            return {"checkpoints": ["sd_xl_base_1.0.safetensors"]}

    class MockTxt2ImgResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"agent_job_id": "test-job-123"}

    request_log = []

    def mock_requests_get(url, timeout=None):
        request_log.append(("get", url))
        return MockHealthResponse()

    def mock_requests_post(url, json=None, timeout=None):
        request_log.append(("post", url))
        return MockTxt2ImgResponse()

    # Mock MACHINES to have one agent
    test_machines = [
        {
            "machine_id": "test-machine",
            "label": "Test Machine",
            "agent_base_url": "http://localhost:9999",
        }
    ]

    # Mock _checkpoint_entries to return empty (no URL configured)
    monkeypatch.setattr(central_app, "_checkpoint_entries", lambda *args, **kwargs: [])

    # Mock requests
    monkeypatch.setattr(central_app.requests, "get", mock_requests_get)
    monkeypatch.setattr(central_app.requests, "post", mock_requests_post)

    # Mock MACHINES
    monkeypatch.setattr(central_app, "MACHINES", test_machines)

    # Use Flask test client
    central_app.app.config["TESTING"] = True
    with central_app.app.test_client() as client:
        response = client.post(
            "/api/start_image",
            data=json.dumps({
                "prompt": "test prompt",
                "checkpoint": "sd_xl_base_1.0.safetensors",
                "seed": 12345,
                "steps": 1,
                "width": 512,
                "height": 512,
                "num_images": 1,
                "repeat": 1,
            }),
            content_type="application/json",
        )

        data = response.get_json()

        # Should NOT get a "Checkpoint URL not configured" error
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {data}"
        assert "error" not in data or data.get("error") is None
        assert "run_id" in data

        # Verify the agent was called
        assert any("txt2img" in url for method, url in request_log if method == "post")


def test_resolve_checkpoint_id_by_filename(monkeypatch):
    entries = [
        {
            "url": "https://example.com/sd_xl_base_1.0.safetensors",
            "filename": "sd_xl_base_1.0.safetensors",
            "label": "SDXL Base",
        }
    ]
    monkeypatch.setattr(central_app, "_checkpoint_entries", lambda settings=None: entries)

    resolved, error = central_app._resolve_checkpoint_id("sd_xl_base_1.0.safetensors")

    assert resolved == "sd_xl_base_1.0.safetensors"
    assert error is None


def test_resolve_checkpoint_id_by_digest(monkeypatch):
    url = "https://example.com/sd_xl_refiner_1.0.safetensors"
    digest = central_app.hashlib.sha256(url.encode("utf-8")).hexdigest()
    entries = [
        {
            "url": url,
            "filename": "sd_xl_refiner_1.0.safetensors",
            "label": "SDXL Refiner",
        }
    ]
    monkeypatch.setattr(central_app, "_checkpoint_entries", lambda settings=None: entries)

    resolved, error = central_app._resolve_checkpoint_id(digest)

    assert resolved == "sd_xl_refiner_1.0.safetensors"
    assert error is None


def test_resolve_checkpoint_id_unknown_digest(monkeypatch):
    monkeypatch.setattr(central_app, "_checkpoint_entries", lambda settings=None: [])

    resolved, error = central_app._resolve_checkpoint_id("c" * 64)

    assert resolved is None
    assert error == "unknown checkpoint digest"


def test_resolve_checkpoint_id_unknown_name(monkeypatch):
    entries = [
        {
            "url": "https://example.com/known.safetensors",
            "filename": "known.safetensors",
            "label": "Known",
        }
    ]
    monkeypatch.setattr(central_app, "_checkpoint_entries", lambda settings=None: entries)

    resolved, error = central_app._resolve_checkpoint_id("missing.safetensors")

    assert resolved is None
    assert error == "unknown checkpoint name"


def test_resolve_checkpoint_id_rejects_paths(monkeypatch):
    monkeypatch.setattr(central_app, "_checkpoint_entries", lambda settings=None: [])

    resolved, error = central_app._resolve_checkpoint_id("../oops.safetensors")

    assert resolved is None
    assert error == "checkpoint must be a filename, not a path"
