import importlib


def test_resolve_model_for_machine_uses_vendor_specific_ollama_tag(monkeypatch):
    app_mod = importlib.import_module("central.app")

    monkeypatch.setattr(
        app_mod,
        "load_models_registry",
        lambda: {
            "ollama": [
                {
                    "id": "llama3.1-8b-q4",
                    "display_name": "Llama 3.1 8B Instruct (Q4_K_M)",
                    "apple": "llama3.1:8b-instruct-q4_K_M",
                    "nvidia": "llama3.1:8b-instruct-q4_k_m-nv",
                }
            ],
            "custom": [],
        },
    )

    apple = {"machine_id": "m1", "vendor": "apple"}
    nvidia = {"machine_id": "m2", "vendor": "nvidia"}

    assert app_mod._resolve_model_for_machine(apple, "ollama", "Llama 3.1 8B Instruct (Q4_K_M)") == (
        "ollama",
        "llama3.1:8b-instruct-q4_K_M",
    )
    assert app_mod._resolve_model_for_machine(nvidia, "ollama", "Llama 3.1 8B Instruct (Q4_K_M)") == (
        "ollama",
        "llama3.1:8b-instruct-q4_k_m-nv",
    )


def test_resolve_model_for_machine_custom_uses_vendor_runtime_model(monkeypatch):
    app_mod = importlib.import_module("central.app")

    monkeypatch.setattr(
        app_mod,
        "load_models_registry",
        lambda: {
            "ollama": [],
            "custom": [
                {
                    "id": "llama3.1-8b-custom",
                    "display_name": "Llama 3.1 8B (Custom Quants)",
                    "apple": "mlx-community/Llama-3.1-8B-Instruct-4bit",
                    "nvidia": "Llama-3.1-8B-Instruct-NVFP4-engine",
                }
            ],
        },
    )

    apple = {"machine_id": "m1", "vendor": "apple"}
    nvidia = {"machine_id": "m2", "vendor": "nvidia"}

    assert app_mod._resolve_model_for_machine(apple, "custom", "llama3.1-8b-custom") == (
        "custom",
        "mlx-community/Llama-3.1-8B-Instruct-4bit",
    )
    assert app_mod._resolve_model_for_machine(nvidia, "custom", "llama3.1-8b-custom") == (
        "custom",
        "Llama-3.1-8B-Instruct-NVFP4-engine",
    )


def test_api_engine_start_resolves_registry_model_before_agent_call(monkeypatch):
    app_mod = importlib.import_module("central.app")
    app_mod.app.config["TESTING"] = True

    monkeypatch.setattr(
        app_mod,
        "MACHINES",
        [{"machine_id": "m1", "agent_base_url": "http://agent", "vendor": "apple"}],
    )

    captured = {}

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"ok": True}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return _Resp()

    monkeypatch.setattr(app_mod.requests, "post", fake_post)

    with app_mod.app.test_client() as client:
        resp = client.post(
            "/api/agents/m1/engine/start",
            json={"backend": "custom", "model": "llama3.1-8b-custom"},
        )

    assert resp.status_code == 200
    assert captured["url"].endswith("/api/engine/start")
    assert captured["json"]["backend"] == "custom"
    assert captured["json"]["model"] == "mlx-community/Llama-3.1-8B-Instruct-4bit"


def test_api_engine_start_returns_400_for_unknown_registry_id(monkeypatch):
    app_mod = importlib.import_module("central.app")
    app_mod.app.config["TESTING"] = True

    monkeypatch.setattr(
        app_mod,
        "MACHINES",
        [{"machine_id": "m1", "agent_base_url": "http://agent", "vendor": "apple"}],
    )

    with app_mod.app.test_client() as client:
        resp = client.post(
            "/api/agents/m1/engine/start",
            json={"backend": "custom", "model": "missing-model-id"},
        )

    assert resp.status_code == 400
    assert "Unknown custom model id" in (resp.get_json() or {}).get("error", "")
