import importlib
import sys
from pathlib import Path


CFG_PATH = Path("config/machines.yaml")


def _load_central_app():
    CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
    original = CFG_PATH.read_text(encoding="utf-8") if CFG_PATH.exists() else None
    try:
        CFG_PATH.write_text(
            "machines:\n"
            "  - machine_id: test1\n"
            "    label: Test\n"
            "    agent_base_url: http://127.0.0.1:9001\n",
            encoding="utf-8",
        )
        sys.modules.pop("central.app", None)
        return importlib.import_module("central.app")
    finally:
        if original is None:
            CFG_PATH.unlink(missing_ok=True)
        else:
            CFG_PATH.write_text(original, encoding="utf-8")


def test_api_status_validates_ollama_registry_id_against_resolved_tag(monkeypatch):
    central_app = _load_central_app()

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "machine_id": "m1",
                "label": "Mac Agent",
                "active_backend": "ollama",
                "ollama_models": ["llama3.1:8b-instruct-q4_K_M"],
                "llm_models": [],
            }

    monkeypatch.setattr(
        central_app,
        "MACHINES",
        [
            {
                "machine_id": "m1",
                "label": "Mac Agent",
                "vendor": "apple",
                "agent_base_url": "http://agent",
                "excluded": False,
            }
        ],
    )
    monkeypatch.setattr(central_app.requests, "get", lambda url, timeout=2: _Resp())

    central_app.app.config["TESTING"] = True
    with central_app.app.test_client() as client:
        resp = client.get("/api/status?model=llama3.1-8b-q4")

    payload = resp.get_json()
    machine = payload["machines"][0]
    assert machine["selected_model"] == "llama3.1-8b-q4"
    assert machine["resolved_selected_model"] == "llama3.1:8b-instruct-q4_K_M"
    assert machine["has_selected_model"] is True


def test_api_status_keeps_non_ollama_model_validation_as_selected(monkeypatch):
    central_app = _load_central_app()

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "machine_id": "m1",
                "label": "Custom Agent",
                "active_backend": "mlx",
                "ollama_models": [],
                "llm_models": ["llama3.1-8b-custom"],
            }

    monkeypatch.setattr(
        central_app,
        "MACHINES",
        [{"machine_id": "m1", "label": "Custom Agent", "agent_base_url": "http://agent", "excluded": False}],
    )
    monkeypatch.setattr(central_app.requests, "get", lambda url, timeout=2: _Resp())

    central_app.app.config["TESTING"] = True
    with central_app.app.test_client() as client:
        resp = client.get("/api/status?model=llama3.1-8b-custom")

    payload = resp.get_json()
    machine = payload["machines"][0]
    assert machine["resolved_selected_model"] == "llama3.1-8b-custom"
    assert machine["has_selected_model"] is True
