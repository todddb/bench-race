from __future__ import annotations

import asyncio
from pathlib import Path
import importlib


def _load_agent_module():
    repo_root = Path(__file__).resolve().parent.parent
    config_path = repo_root / "agent" / "config" / "agent.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        config_path.write_text(
            "machine_id: test-agent\nlabel: Test Agent\ncomfyui:\n  enabled: true\n",
            encoding="utf-8",
        )
    return importlib.import_module("agent.agent_app")


class DummyResponse:
    def __init__(self, status_code=200, chunks=None):
        self.status_code = status_code
        self._chunks = chunks or []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aiter_bytes(self, chunk_size=1024):
        for chunk in self._chunks:
            yield chunk


class DummyClient:
    def __init__(self, response: DummyResponse):
        self._response = response

    def stream(self, method, url, headers=None, timeout=None):
        return self._response


def test_should_skip_checkpoint_when_size_matches(tmp_path: Path):
    agent_app = _load_agent_module()
    target = tmp_path / "model.safetensors"
    target.write_bytes(b"1234")
    assert agent_app._should_skip_checkpoint(target, 4) is True


def test_download_checkpoint_writes_part_then_renames(tmp_path: Path):
    agent_app = _load_agent_module()
    item = agent_app.ComfyCheckpointItem(
        name="model.safetensors",
        url="https://example.com/model.safetensors",
        size_bytes=6,
    )
    response = DummyResponse(chunks=[b"abc", b"def"])
    client = DummyClient(response)

    async def run_download():
        return await agent_app._download_checkpoint_file(item, tmp_path, client)

    result = asyncio.run(run_download())
    final_path = tmp_path / "model.safetensors"
    temp_path = tmp_path / "model.safetensors.part"

    assert result["status"] == "downloaded"
    assert final_path.exists()
    assert not temp_path.exists()
