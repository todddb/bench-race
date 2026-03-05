from pathlib import Path
import importlib
import sys

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


def test_load_model_resolves_runtime_name_not_registry_id(monkeypatch):
    app_mod = _load_central_app_with_test_machines()
    app_mod.app.config["TESTING"] = True

    machine = {"machine_id": "m1", "agent_base_url": "http://agent", "gpu_type": "apple"}
    monkeypatch.setattr(app_mod, "MACHINES", [machine])
    monkeypatch.setattr(app_mod, "_proxy_backend_status", lambda *_args, **_kwargs: {"backend": "ollama"})
    monkeypatch.setattr(
        app_mod,
        "load_models_registry",
        lambda: {
            "ollama": [
                {
                    "id": "llama3.1-8b-custom",
                    "display_name": "Llama 3.1 8B Custom",
                    "apple": "llama3.1:8b-instruct-q4_K_M",
                    "nvidia": "llama3.1:8b-instruct-fp16",
                }
            ]
        },
    )

    posted_payloads = []

    class _Resp:
        status_code = 200

        def json(self):
            return {"ok": True}

    def fake_post(_url, *args, **kwargs):
        posted_payloads.append(kwargs.get("json") or {})
        return _Resp()

    monkeypatch.setattr(app_mod.requests, "post", fake_post)

    with app_mod.app.test_client() as client:
        resp = client.post("/api/agent/m1/load_model", json={"model_id": "llama3.1-8b-custom"})
        assert resp.status_code == 200

    assert posted_payloads
    assert posted_payloads[0]["backend"] == "ollama"
    assert posted_payloads[0]["model"] == "llama3.1:8b-instruct-q4_K_M"
    assert posted_payloads[0]["model"] != "llama3.1-8b-custom"
    assert posted_payloads[0]["model"] != "Llama 3.1 8B Custom"


def test_backend_switch_and_load_model_share_resolution_logic(monkeypatch):
    app_mod = _load_central_app_with_test_machines()
    app_mod.app.config["TESTING"] = True

    machine = {"machine_id": "m1", "agent_base_url": "http://agent", "gpu_type": "nvidia"}
    monkeypatch.setattr(app_mod, "MACHINES", [machine])
    monkeypatch.setattr(app_mod, "load_machines_config", lambda: {"machines": [machine]})
    monkeypatch.setattr(app_mod, "_proxy_backend_status", lambda *_args, **_kwargs: {"backend": "custom"})
    monkeypatch.setattr(app_mod, "_wait_for_backend_ready", lambda *_args, **_kwargs: (True, "ready"))

    resolve_calls = []

    def fake_resolve(machine_arg, backend_arg, model_id_arg):
        resolve_calls.append((machine_arg.get("machine_id"), backend_arg, model_id_arg))
        return "trtllm", "meta-llama/Llama-3.1-8B-Instruct"

    monkeypatch.setattr(app_mod, "_resolve_model_for_machine", fake_resolve)

    class _Resp:
        status_code = 200
        text = "ok"

        def json(self):
            return {"ok": True}

    monkeypatch.setattr(app_mod.requests, "post", lambda *_args, **_kwargs: _Resp())

    with app_mod.app.test_client() as client:
        switch_resp = client.post("/api/backend/switch", json={"backend": "custom", "model_id": "llama3.1-8b-custom"})
        assert switch_resp.status_code == 200

        load_resp = client.post("/api/agent/m1/load_model", json={"model_id": "llama3.1-8b-custom"})
        assert load_resp.status_code == 200

    assert len(resolve_calls) >= 2
    assert ("m1", "custom", "llama3.1-8b-custom") in resolve_calls
