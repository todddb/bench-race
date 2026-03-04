from pathlib import Path
import importlib


def test_load_models_manifest_and_api(monkeypatch, tmp_path):
    machines_cfg = Path("config/machines.yaml")
    machines_cfg.parent.mkdir(parents=True, exist_ok=True)
    if not machines_cfg.exists():
        machines_cfg.write_text(
            "machines:\n"
            "  - machine_id: test1\n"
            "    label: Test\n"
            "    agent_base_url: http://127.0.0.1:9001\n",
            encoding="utf-8",
        )

    manifest_path = tmp_path / "models.json"
    manifest_path.write_text(
        '{"models":[{"id":"m1","ollama":{"quant":"q4"},"custom":{"MLX":{"quant":"fp16"}}}]}',
        encoding="utf-8",
    )

    loader = importlib.import_module("central.config_loader")
    manifest = loader.load_models_manifest(manifest_path)
    assert manifest["models"][0]["id"] == "m1"

    app_mod = importlib.import_module("central.app")
    monkeypatch.setattr(app_mod, "load_models_manifest", lambda: manifest)
    app_mod.app.config["TESTING"] = True

    with app_mod.app.test_client() as client:
        resp = client.get("/api/models")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["models"][0]["id"] == "m1"


def test_agent_switch_and_load_model_proxy(monkeypatch):
    app_mod = importlib.import_module("central.app")

    machine = {
        "machine_id": "agent1",
        "label": "Agent 1",
        "agent_base_url": "http://agent1:9001",
    }
    monkeypatch.setattr(app_mod, "MACHINES", [machine])
    monkeypatch.setattr(app_mod, "_proxy_backend_status", lambda _: {"backend": "ollama"})
    monkeypatch.setattr(app_mod, "_resolve_model_for_machine", lambda machine, backend, model_id: ("ollama", model_id))

    called = []

    class _Resp:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    def fake_post(url, json=None, timeout=None):
        called.append({"url": url, "json": json, "timeout": timeout})
        return _Resp({"ok": True})

    monkeypatch.setattr(app_mod.requests, "post", fake_post)
    app_mod.app.config["TESTING"] = True

    with app_mod.app.test_client() as client:
        switch_resp = client.post("/api/agent/agent1/switch_backend", json={"backend": "ollama", "model_id": "m1"})
        assert switch_resp.status_code == 200
        load_resp = client.post("/api/agent/agent1/load_model", json={"model_id": "m2"})
        assert load_resp.status_code == 200

    assert called[0]["url"].endswith("/api/backend/select")
    assert called[0]["json"]["backend"] == "ollama"
    assert called[0]["json"]["model"] == "m1"
    assert called[1]["json"]["model"] == "m2"
