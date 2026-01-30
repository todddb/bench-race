from __future__ import annotations

from types import SimpleNamespace

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
