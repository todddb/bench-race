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


def test_engine_start_comfyui_returns_immediately(monkeypatch):
    _ensure_agent_config()
    mod = importlib.import_module("agent.agent_app")

    async def fake_start():
        await asyncio.sleep(0.01)
        return {"started": True}

    monkeypatch.setattr(mod, "_start_comfyui", fake_start)
    mod._ENGINE_TASKS.clear()

    response = asyncio.run(mod.start_engine(mod.EngineStartRequest(engine="comfyui")))

    assert response["engine"] == "comfyui"
    assert response["status"] == "starting"
    assert response["accepted"] is True


def test_engine_stop_comfyui_returns_immediately(monkeypatch):
    _ensure_agent_config()
    mod = importlib.import_module("agent.agent_app")

    async def fake_stop():
        await asyncio.sleep(0.01)
        return {"stopped": True}

    monkeypatch.setattr(mod, "_stop_comfyui", fake_stop)
    mod._ENGINE_TASKS.clear()

    response = asyncio.run(mod.stop_engine(mod.EngineStopRequest(engine="comfyui")))

    assert response["engine"] == "comfyui"
    assert response["status"] == "stopping"
    assert response["accepted"] is True


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
