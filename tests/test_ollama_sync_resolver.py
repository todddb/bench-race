import asyncio
import importlib
from pathlib import Path


def test_sync_models_uses_resolved_ollama_tag(monkeypatch, capsys):
    cfg_path = Path("agent/config/agent.yaml")
    if not cfg_path.exists():
        cfg_path.write_text(Path("agent/config/agent.yaml.example").read_text(encoding="utf-8"), encoding="utf-8")

    agent_app = importlib.import_module("agent.agent_app")

    pulled = []

    def fake_fetch_installed_ollama_tags(_url="http://127.0.0.1:11434/api/tags"):
        return {"llama3.1:8b-instruct-q4_K_M"}

    def fake_subprocess_run(cmd, capture_output, text, check):
        pulled.append(cmd)
        return None

    async def fake_broadcast_sync_event(*_args, **_kwargs):
        return None

    def fake_ensure_model_layout():
        return None

    monkeypatch.setattr(agent_app, "fetch_installed_ollama_tags", fake_fetch_installed_ollama_tags)
    monkeypatch.setattr(agent_app.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(agent_app, "_broadcast_sync_event", fake_broadcast_sync_event)
    monkeypatch.setattr(agent_app, "_ensure_model_layout", fake_ensure_model_layout)

    req = agent_app.SyncRequest(llm=["llama3.1:8b-instruct"], backend="ollama")

    asyncio.run(agent_app._sync_models("sync-1", req))

    assert pulled == [["ollama", "pull", "llama3.1:8b-instruct-q4_K_M"]]
    out = capsys.readouterr().out
    assert "[sync] Mapped requested model 'llama3.1:8b-instruct' -> 'llama3.1:8b-instruct-q4_K_M'" in out
