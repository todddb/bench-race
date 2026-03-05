"""Tests validating legacy central sync endpoints are removed."""
import importlib
from pathlib import Path


def _ensure_config():
    cfg = Path("config/machines.yaml")
    cfg.parent.mkdir(parents=True, exist_ok=True)
    if not cfg.exists():
        cfg.write_text(
            "machines:\n"
            "  - machine_id: agent1\n"
            "    label: Agent 1\n"
            "    agent_base_url: http://127.0.0.1:9001\n",
            encoding="utf-8",
        )


def test_sync_models_route_removed():
    _ensure_config()
    central_app = importlib.import_module("central.app")
    central_app.app.config["TESTING"] = True

    with central_app.app.test_client() as client:
        resp = client.post("/api/agents/agent1/sync_models", json={"models": ["m1"]})
    assert resp.status_code == 404


def test_sync_status_route_removed():
    _ensure_config()
    central_app = importlib.import_module("central.app")
    central_app.app.config["TESTING"] = True

    with central_app.app.test_client() as client:
        resp = client.get("/api/agents/agent1/sync_status/job-1")
    assert resp.status_code == 404


def test_model_status_unreachable_agent(monkeypatch):
    _ensure_config()
    central_app = importlib.import_module("central.app")
    import central.app as ca

    ca.MACHINES = [{"machine_id": "agent1", "agent_base_url": "http://127.0.0.1:9999"}]
    monkeypatch.setattr(ca, "_required_models", lambda: {"llm": ["llama3.1:8b"], "whisper": [], "sdxl_profiles": []})

    def fail_get(url, timeout=None):
        raise ConnectionError("unreachable")

    monkeypatch.setattr(ca.requests, "get", fail_get)

    ca.app.config["TESTING"] = True
    with ca.app.test_client() as client:
        resp = client.get("/api/agents/agent1/model_status")
    assert resp.status_code == 502
