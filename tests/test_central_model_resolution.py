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

    assert app_mod._resolve_model_for_machine(apple, "ollama", "Llama 3.1 8B Instruct (Q4_K_M)") == "llama3.1:8b-instruct-q4_K_M"
    assert app_mod._resolve_model_for_machine(nvidia, "ollama", "Llama 3.1 8B Instruct (Q4_K_M)") == "llama3.1:8b-instruct-q4_k_m-nv"


def test_resolve_model_for_machine_custom_returns_registry_id(monkeypatch):
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
                }
            ],
        },
    )

    machine = {"machine_id": "m1", "vendor": "apple"}
    assert app_mod._resolve_model_for_machine(machine, "mlx", "Llama 3.1 8B (Custom Quants)") == "llama3.1-8b-custom"
