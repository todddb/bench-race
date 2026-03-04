import asyncio
import importlib
from pathlib import Path

from backends.ollama_backend import resolved_ollama_pull_name


def test_job_dispatch_uses_resolved_tag(monkeypatch):
    cfg_path = Path("agent/config/agent.yaml")
    if not cfg_path.exists():
        cfg_path.write_text(Path("agent/config/agent.yaml.example").read_text(encoding="utf-8"), encoding="utf-8")

    installed = ["llama3.1:8b-instruct-q4_K_M"]

    def mock_fetch():
        return installed

    resolved = resolved_ollama_pull_name(
        {"model": "llama3.1:8b-instruct"},
        installed,
    )

    assert resolved == "llama3.1:8b-instruct-q4_K_M"

    agent_app = importlib.import_module("agent.agent_app")

    async def fake_check_ollama_available(_base_url):
        return True

    async def fake_get_ollama_models(_base_url):
        return installed

    called = {"ollama": False, "mock": False, "model": None}

    async def fake_stream_ollama_generate(*, model, **kwargs):
        called["ollama"] = True
        called["model"] = model
        return {"tokens_generated": 1, "total_ms": 1, "engine": "ollama", "model": model}

    async def fake_run_mock_stream(*args, **kwargs):
        called["mock"] = True
        return {"tokens_generated": 1, "total_ms": 1, "engine": "mock", "model": "mock"}

    async def fake_broadcast_event(_ev):
        return None

    monkeypatch.setattr(agent_app, "check_ollama_available", fake_check_ollama_available)
    monkeypatch.setattr(agent_app, "get_ollama_models", fake_get_ollama_models)
    monkeypatch.setattr(agent_app, "fetch_installed_ollama_tags", mock_fetch)
    monkeypatch.setattr(agent_app, "stream_ollama_generate", fake_stream_ollama_generate)
    monkeypatch.setattr(agent_app, "_run_mock_stream", fake_run_mock_stream)
    monkeypatch.setattr(agent_app, "_broadcast_event", fake_broadcast_event)
    monkeypatch.setattr(agent_app, "_reset_idle_timer", lambda: None)
    monkeypatch.setattr(agent_app, "_ACTIVE_BACKEND", "ollama")

    req = agent_app.LLMRequest(model="llama3.1:8b-instruct", prompt="hello")
    asyncio.run(agent_app._job_runner_llm("job-1", req))

    assert called["ollama"] is True
    assert called["mock"] is False
    assert called["model"] == "llama3.1:8b-instruct-q4_K_M"


def test_job_dispatch_requires_active_backend(monkeypatch):
    agent_app = importlib.import_module("agent.agent_app")

    called = {"mock": False, "events": []}

    async def fake_run_mock_stream(*args, **kwargs):
        called["mock"] = True
        return {"tokens_generated": 1}

    async def fake_broadcast_event(ev):
        called["events"].append(ev.model_dump())

    monkeypatch.setattr(agent_app, "_run_mock_stream", fake_run_mock_stream)
    monkeypatch.setattr(agent_app, "_broadcast_event", fake_broadcast_event)
    monkeypatch.setattr(agent_app, "_ACTIVE_BACKEND", None)

    req = agent_app.LLMRequest(model="llama3.1:8b-instruct", prompt="hello")
    asyncio.run(agent_app._job_runner_llm("job-no-backend", req))

    assert called["mock"] is False
    assert called["events"][-1]["type"] == "job_done"
    assert called["events"][-1]["payload"]["error"] == "No active backend"


def test_engine_start_rejects_invalid_backend():
    agent_app = importlib.import_module("agent.agent_app")

    resp = asyncio.run(agent_app.start_engine(agent_app.EngineStartRequest(backend="custom", model="x")))
    assert resp.status_code == 400
    assert b"Invalid backend 'custom'" in resp.body
