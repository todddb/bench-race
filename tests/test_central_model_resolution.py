import importlib


def test_resolve_model_for_machine_uses_vendor_specific_ollama_tag(monkeypatch):
    """Central must resolve abstract registry IDs (not display_names) to
    architecture-specific runtime strings using machines.yaml gpu.type."""
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

    apple = {"machine_id": "m1", "gpu": {"type": "apple"}}
    nvidia = {"machine_id": "m2", "gpu": {"type": "nvidia"}}

    # Resolution uses abstract ID, never display_name
    assert app_mod._resolve_model_for_machine(apple, "ollama", "llama3.1-8b-q4") == (
        "ollama",
        "llama3.1:8b-instruct-q4_K_M",
    )
    assert app_mod._resolve_model_for_machine(nvidia, "ollama", "llama3.1-8b-q4") == (
        "ollama",
        "llama3.1:8b-instruct-q4_k_m-nv",
    )


def test_resolve_rejects_display_name(monkeypatch):
    """display_name is a Central-only UI abstraction and must never be
    accepted as a resolution key.  Only abstract 'id' is valid."""
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

    machine = {"machine_id": "m1", "gpu": {"type": "apple"}}

    try:
        app_mod.resolve_runtime_model(machine, "ollama", "Llama 3.1 8B Instruct (Q4_K_M)")
        assert False, "Should have raised ValueError for display_name lookup"
    except ValueError as exc:
        assert "Unknown ollama model id" in str(exc)


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

    apple = {"machine_id": "m1", "gpu": {"type": "apple"}}
    nvidia = {"machine_id": "m2", "gpu": {"type": "nvidia"}}

    assert app_mod._resolve_model_for_machine(apple, "custom", "llama3.1-8b-custom") == (
        "custom",
        "mlx-community/Llama-3.1-8B-Instruct-4bit",
    )
    assert app_mod._resolve_model_for_machine(nvidia, "custom", "llama3.1-8b-custom") == (
        "custom",
        "Llama-3.1-8B-Instruct-NVFP4-engine",
    )


def test_namespace_isolation_custom_does_not_search_ollama(monkeypatch):
    """When backend == 'custom', resolution must only search registry['custom'],
    never registry['ollama']."""
    app_mod = importlib.import_module("central.app")

    monkeypatch.setattr(
        app_mod,
        "load_models_registry",
        lambda: {
            "ollama": [
                {
                    "id": "llama3.1-8b-q4",
                    "display_name": "Llama 3.1 8B Q4",
                    "apple": "llama3.1:8b-instruct-q4_K_M",
                    "nvidia": "llama3.1:8b-instruct-q4_K_M",
                }
            ],
            "custom": [],
        },
    )

    machine = {"machine_id": "m1", "gpu": {"type": "apple"}}

    try:
        app_mod.resolve_runtime_model(machine, "custom", "llama3.1-8b-q4")
        assert False, "Should have raised ValueError — model exists in ollama but not custom"
    except ValueError as exc:
        assert "Unknown custom model id" in str(exc)


def test_namespace_isolation_ollama_does_not_search_custom(monkeypatch):
    """When backend == 'ollama', resolution must only search registry['ollama'],
    never registry['custom']."""
    app_mod = importlib.import_module("central.app")

    monkeypatch.setattr(
        app_mod,
        "load_models_registry",
        lambda: {
            "ollama": [],
            "custom": [
                {
                    "id": "llama3.1-8b-custom",
                    "display_name": "Llama 3.1 8B Custom",
                    "apple": "mlx-community/Llama-3.1-8B-Instruct-4bit",
                    "nvidia": "Llama-3.1-8B-Instruct-NVFP4-engine",
                }
            ],
        },
    )

    machine = {"machine_id": "m1", "gpu": {"type": "apple"}}

    try:
        app_mod.resolve_runtime_model(machine, "ollama", "llama3.1-8b-custom")
        assert False, "Should have raised ValueError — model exists in custom but not ollama"
    except ValueError as exc:
        assert "Unknown ollama model id" in str(exc)


def test_api_engine_start_resolves_registry_model_before_agent_call(monkeypatch):
    app_mod = importlib.import_module("central.app")
    app_mod.app.config["TESTING"] = True

    monkeypatch.setattr(
        app_mod,
        "MACHINES",
        [{"machine_id": "m1", "agent_base_url": "http://agent", "gpu": {"type": "apple"}}],
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
    # Central must send resolved runtime string, NOT the abstract ID
    assert captured["json"]["model"] == "mlx-community/Llama-3.1-8B-Instruct-4bit"
    assert captured["json"]["model"] != "llama3.1-8b-custom"


def test_api_engine_start_returns_400_for_unknown_registry_id(monkeypatch):
    app_mod = importlib.import_module("central.app")
    app_mod.app.config["TESTING"] = True

    monkeypatch.setattr(
        app_mod,
        "MACHINES",
        [{"machine_id": "m1", "agent_base_url": "http://agent", "gpu": {"type": "apple"}}],
    )

    with app_mod.app.test_client() as client:
        resp = client.post(
            "/api/agents/m1/engine/start",
            json={"backend": "custom", "model": "missing-model-id"},
        )

    assert resp.status_code == 400
    assert "Unknown custom model id" in (resp.get_json() or {}).get("error", "")
