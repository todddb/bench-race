import asyncio
import importlib
from pathlib import Path


def _ensure_agent_config():
    cfg = Path("agent/config/agent.yaml")
    if cfg.exists():
        return
    example = Path("agent/config/agent.yaml.example")
    if example.exists():
        cfg.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")


class _FakeBackend:
    async def list_models(self):
        return ["demo-model"]

    async def generate(self, model, messages, stream):
        for token in ["hello", " world"]:
            yield token


class _FakeManager:
    def get_active_backend(self, _active):
        return _FakeBackend()


def test_v1_models_and_chat_completion(monkeypatch):
    _ensure_agent_config()
    mod = importlib.import_module("agent.agent_app")
    monkeypatch.setattr(mod, "backend_manager", _FakeManager())
    monkeypatch.setattr(mod, "_ACTIVE_BACKEND", "ollama")

    models = asyncio.run(mod.v1_models())
    assert models["data"][0]["id"] == "demo-model"

    resp = asyncio.run(mod.v1_chat_completions({"model": "demo-model", "messages": [{"role": "user", "content": "hi"}]}))
    assert resp.media_type == "text/event-stream"
