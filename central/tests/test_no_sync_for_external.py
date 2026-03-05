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


def test_backend_switch_never_calls_legacy_sync(monkeypatch):
    app_mod = _load_central_app_with_test_machines()
    app_mod.app.config["TESTING"] = True

    monkeypatch.setattr(app_mod, "load_machines_config", lambda: {"machines": [{"machine_id": "test1", "agent_base_url": "http://agent"}]})
    monkeypatch.setattr(app_mod, "_wait_for_backend_ready", lambda *_args, **_kwargs: (True, "ready"))
    monkeypatch.setattr(app_mod, "resolve_runtime_model", lambda *_args, **_kwargs: "llama3:8b")
    monkeypatch.setattr(app_mod, "_machine_llm_hardware", lambda *_args, **_kwargs: "nvidia")
    monkeypatch.setattr(app_mod, "load_models_registry", lambda: {"ollama": [{"id": "m1"}], "custom": [{"id": "m1"}]})

    called_urls = []

    class _Resp:
        status_code = 200
        text = "ok"

        def json(self):
            return {"ok": True}

    def fake_post(url, *args, **kwargs):
        called_urls.append(url)
        return _Resp()

    monkeypatch.setattr(app_mod.requests, "post", fake_post)

    with app_mod.app.test_client() as client:
        resp = client.post("/api/backend/switch", json={"backend": "ollama", "model_id": "m1"})
        assert resp.status_code == 200
        resp = client.post("/api/backend/switch", json={"backend": "custom", "model_id": "m1"})
        assert resp.status_code == 200

    assert any("/api/engine/start" in url for url in called_urls)
    assert not any("/api/agents/test1/sync_models" in url for url in called_urls)
