from pathlib import Path
import importlib
import uuid


def test_build_backend_switch_message_targeting():
    proto = importlib.import_module("central.agent_protocol")
    rid = str(uuid.uuid4())

    msg_mac = proto.build_backend_switch_message({"label": "Mac Studio"}, "custom", rid)
    assert msg_mac["type"] == "backend_switch"
    assert msg_mac["payload"]["target"] == "mlx"

    msg_nv = proto.build_backend_switch_message({"label": "NVIDIA RTX"}, "custom", rid)
    assert msg_nv["payload"]["target"] == "trtllm"


def test_api_backend_switch_dispatch(monkeypatch):
    machines_cfg = Path("central/config/machines.yaml")
    machines_cfg.parent.mkdir(parents=True, exist_ok=True)
    if not machines_cfg.exists():
        machines_cfg.write_text(
            "machines:\n"
            "  - machine_id: test1\n"
            "    label: Test\n"
            "    agent_base_url: http://127.0.0.1:9001\n",
            encoding="utf-8",
        )

    app_mod = importlib.import_module("central.app")
    app_mod.app.config["TESTING"] = True

    monkeypatch.setattr(app_mod, "MACHINES", [{"machine_id": "a1", "agent_base_url": "http://agent"}])
    monkeypatch.setattr(
        app_mod,
        "do_backend_switch",
        lambda agents, backend: {"request_id": "req-1", "dispatch": [{"machine_id": "a1", "ok": True}]},
    )

    with app_mod.app.test_client() as client:
        resp = client.post("/api/backend/switch", json={"backend": "ollama"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["switch_id"] == "req-1"
