from pathlib import Path
import importlib
import sys
import uuid


CFG_PATH = Path("config/machines.yaml")


def _load_central_app_with_test_machines():
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


def test_build_backend_switch_message_targeting():
    proto = importlib.import_module("central.agent_protocol")
    rid = str(uuid.uuid4())

    msg_mac = proto.build_backend_switch_message({"label": "Mac Studio"}, "custom", rid)
    assert msg_mac["type"] == "backend_switch"
    assert msg_mac["payload"]["target"] == "mlx"

    msg_nv = proto.build_backend_switch_message({"label": "NVIDIA RTX"}, "custom", rid)
    assert msg_nv["payload"]["target"] == "trtllm"


def test_api_backend_switch_dispatch(monkeypatch):
    app_mod = _load_central_app_with_test_machines()
    app_mod.app.config["TESTING"] = True

    monkeypatch.setattr(app_mod, "MACHINES", [{"machine_id": "a1", "agent_base_url": "http://agent"}])
    monkeypatch.setattr(app_mod, "resolve_runtime_model", lambda machine, backend, model_id: "llama3:8b")
    monkeypatch.setattr(app_mod, "load_models_registry", lambda: {"ollama": [{"id": "m1"}], "custom": [{"id": "m1"}]})

    class _Resp:
        status_code = 200
        text = "ok"

    monkeypatch.setattr(app_mod.requests, "post", lambda *args, **kwargs: _Resp())

    with app_mod.app.test_client() as client:
        resp = client.post("/api/backend/switch", json={"backend": "ollama", "model_id": "m1"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"

        custom_resp = client.post("/api/backend/switch", json={"backend": "custom", "model_id": "m1"})
        assert custom_resp.status_code == 200
        custom_data = custom_resp.get_json()
        assert custom_data["status"] == "ok"
