"""Tests for job dispatch model handling.

IMPORTANT ARCHITECTURAL RULE:
Central is the sole authority for model resolution.
Agents receive fully resolved model strings and execute them directly.
The agent's _job_runner_llm passes the model string as-is to the backend.
"""
import asyncio
import importlib
from pathlib import Path


def test_job_dispatch_uses_model_as_provided(monkeypatch):
    """Agent's _job_runner_llm passes the model string from Central as-is.
    Central has already resolved the abstract ID to a runtime string."""
    cfg_path = Path("agent/config/agent.yaml")
    if not cfg_path.exists():
        cfg_path.write_text(Path("agent/config/agent.yaml.example").read_text(encoding="utf-8"), encoding="utf-8")

    agent_app = importlib.import_module("agent.agent_app")

    # Central sends the fully resolved model string
    resolved_model = "llama3.1:8b-instruct-q4_K_M"

    async def fake_check_ollama_available(_base_url):
        return True

    async def fake_get_ollama_models(_base_url):
        return [resolved_model]

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
    monkeypatch.setattr(agent_app, "stream_ollama_generate", fake_stream_ollama_generate)
    monkeypatch.setattr(agent_app, "_run_mock_stream", fake_run_mock_stream)
    monkeypatch.setattr(agent_app, "_broadcast_event", fake_broadcast_event)
    monkeypatch.setattr(agent_app, "_reset_idle_timer", lambda: None)
    monkeypatch.setattr(agent_app, "_ACTIVE_BACKEND", "ollama")

    # Agent receives the already-resolved model string from Central
    req = agent_app.LLMRequest(model=resolved_model, prompt="hello")
    asyncio.run(agent_app._job_runner_llm("job-1", req))

    assert called["ollama"] is True
    assert called["mock"] is False
    # Agent passes the resolved model string directly — no reinterpretation
    assert called["model"] == resolved_model


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


def test_engine_start_rejects_unsupported_backend():
    """Agent's engine start rejects backends that aren't 'ollama' or 'custom'."""
    agent_app = importlib.import_module("agent.agent_app")

    resp = asyncio.run(agent_app.start_engine(agent_app.EngineStartRequest(backend="vllm", model="x")))
    assert resp.status_code == 400
