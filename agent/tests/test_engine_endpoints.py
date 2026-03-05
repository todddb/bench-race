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
    else:
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text("machine_id: test-machine\nlabel: Test\n", encoding="utf-8")


def test_engine_start_rejects_unsupported_backend():
    _ensure_agent_config()
    mod = importlib.import_module("agent.agent_app")

    try:
        asyncio.run(mod.start_engine({"backend": "mlx", "model": "anything"}))
        assert False, "Expected HTTPException"
    except mod.HTTPException as exc:
        assert exc.status_code == 400
        assert "Unsupported backend" in exc.detail


def test_engine_stop_comfyui_returns_immediately(monkeypatch):
    _ensure_agent_config()
    mod = importlib.import_module("agent.agent_app")

    async def fake_stop():
        await asyncio.sleep(0.01)
        return {"stopped": True}

    monkeypatch.setattr(mod, "_stop_comfyui", fake_stop)
    mod._ENGINE_TASKS.clear()

    response = asyncio.run(mod.stop_engine(mod.EngineStopRequest(engine="comfyui")))

    assert response["ok"] is True


def test_comfy_health_exposes_status(monkeypatch):
    _ensure_agent_config()
    mod = importlib.import_module("agent.agent_app")

    monkeypatch.setattr(mod, "_comfy_config", lambda: {"enabled": True})
    monkeypatch.setattr(mod, "_comfy_base_url", lambda: "http://127.0.0.1:8188")

    class _Resp:
        status_code = 200

        def json(self):
            return {"version": "x"}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, _url):
            return _Resp()

    monkeypatch.setattr(mod.httpx, "AsyncClient", lambda timeout=2.0: _Client())

    payload = asyncio.run(mod.comfy_health())

    assert payload["running"] is True
    assert payload["status"] == "healthy"


def test_engine_start_missing_required_fields():
    _ensure_agent_config()
    mod = importlib.import_module("agent.agent_app")

    try:
        asyncio.run(mod.start_engine({"backend": "custom"}))
        assert False, "Expected HTTPException"
    except mod.HTTPException as exc:
        assert exc.status_code == 400
        assert exc.detail == "Missing required field: model"


def test_engine_start_resolution_failure(monkeypatch):
    _ensure_agent_config()
    mod = importlib.import_module("agent.agent_app")

    monkeypatch.setattr(mod, "resolve_model_for_machine", lambda model_id, backend: None)

    try:
        asyncio.run(mod.start_engine({"backend": "custom", "model": "bad-model"}))
        assert False, "Expected HTTPException"
    except mod.HTTPException as exc:
        assert exc.status_code == 400
        assert "Model bad-model not valid for backend custom" == exc.detail


def test_engine_start_custom_uses_resolved_model(monkeypatch):
    _ensure_agent_config()
    mod = importlib.import_module("agent.agent_app")

    monkeypatch.setattr(mod, "resolve_model_for_machine", lambda _model_id, _backend: "resolved-model")

    async def fake_start_backend_engine(backend, model):
        return {"status": "started", "engine": backend, "model": model, "accepted": True}

    monkeypatch.setattr(mod, "start_backend_engine", fake_start_backend_engine)

    response = asyncio.run(mod.start_engine({"backend": "custom", "model": "llama3.1-8b-custom"}))

    assert response["status"] == "started"
    assert response["model"] == "resolved-model"
    assert response["accepted"] is True


def test_backend_status_selected_backend_only(monkeypatch):
    _ensure_agent_config()
    mod = importlib.import_module("agent.agent_app")

    async def fake_check_backend_health(backend):
        return {"name": backend, "healthy": True}

    monkeypatch.setattr(mod, "_check_backend_health", fake_check_backend_health)

    payload = asyncio.run(mod.backend_status(type("Req", (), {"query_params": {}})()))

    assert payload["backends"] == {}

    mod._ACTIVE_BACKEND = "ollama"
    payload = asyncio.run(mod.backend_status(type("Req", (), {"query_params": {}})()))
    assert "ollama" in payload["backends"]
