"""Tests for POST /api/models/validate central endpoint."""
import importlib
from pathlib import Path


def test_validate_models_basic(monkeypatch):
    cfg = Path("central/config/machines.yaml")
    cfg.parent.mkdir(parents=True, exist_ok=True)
    if not cfg.exists():
        cfg.write_text(
            "machines:\n"
            "  - machine_id: test1\n"
            "    label: Test\n"
            "    agent_base_url: http://127.0.0.1:9001\n",
            encoding="utf-8",
        )

    central_app = importlib.import_module("central.app")
    central_app.app.config["TESTING"] = True

    with central_app.app.test_client() as client:
        resp = client.post(
            "/api/models/validate",
            json={"models": ["llama3.1:8b-instruct", "kimi2.5:cloud"]},
        )

    assert resp.status_code == 200
    data = resp.get_json()
    results = data["results"]
    assert len(results) == 2

    assert results[0]["model"] == "llama3.1:8b-instruct"
    assert results[0]["valid"] is True
    assert results[0]["sanitized_name"] == "llama3.1_8b-instruct"

    assert results[1]["model"] == "kimi2.5:cloud"
    assert results[1]["valid"] is True
    assert results[1]["sanitized_name"] == "kimi2.5_cloud"


def test_validate_empty_model_name(monkeypatch):
    cfg = Path("central/config/machines.yaml")
    cfg.parent.mkdir(parents=True, exist_ok=True)
    if not cfg.exists():
        cfg.write_text(
            "machines:\n"
            "  - machine_id: test1\n"
            "    label: Test\n"
            "    agent_base_url: http://127.0.0.1:9001\n",
            encoding="utf-8",
        )

    central_app = importlib.import_module("central.app")
    central_app.app.config["TESTING"] = True

    with central_app.app.test_client() as client:
        resp = client.post(
            "/api/models/validate",
            json={"models": ["", "valid-model"]},
        )

    assert resp.status_code == 200
    data = resp.get_json()
    results = data["results"]
    assert results[0]["valid"] is False
    assert results[1]["valid"] is True


def test_validate_with_agent_check(monkeypatch):
    cfg = Path("central/config/machines.yaml")
    cfg.parent.mkdir(parents=True, exist_ok=True)
    if not cfg.exists():
        cfg.write_text(
            "machines:\n"
            "  - machine_id: test1\n"
            "    label: Test\n"
            "    agent_base_url: http://127.0.0.1:9001\n",
            encoding="utf-8",
        )

    central_app = importlib.import_module("central.app")

    class _Resp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {"ollama_models": ["llama3.1:8b-instruct"]}

    monkeypatch.setattr(
        central_app,
        "MACHINES",
        [{"machine_id": "m1", "agent_base_url": "http://agent"}],
    )
    monkeypatch.setattr(central_app.requests, "get", lambda url, timeout=None: _Resp())

    central_app.app.config["TESTING"] = True
    with central_app.app.test_client() as client:
        resp = client.post(
            "/api/models/validate",
            json={"models": ["llama3.1:8b-instruct"], "check_agents": True},
        )

    data = resp.get_json()
    result = data["results"][0]
    assert result["valid"] is True
    assert len(result["agents"]) == 1
    assert result["agents"][0]["machine_id"] == "m1"
    assert result["agents"][0]["has_model"] is True
