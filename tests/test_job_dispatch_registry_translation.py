"""Verify that registry ID → runtime string translation happens in Central,
not in the agent.

IMPORTANT ARCHITECTURAL RULE:
Central is the sole authority for model resolution.
Agents must never resolve abstract IDs or map architectures.
"""
import importlib


def test_central_resolves_registry_id_to_runtime_string():
    """Central's resolve_runtime_model translates abstract registry IDs
    into concrete runtime strings using backend namespace + architecture."""
    app_mod = importlib.import_module("central.app")

    registry = {
        "ollama": [
            {
                "id": "llama3.1-8b-q4",
                "display_name": "Llama 3.1 8B Instruct (Q4_K_M)",
                "apple": "llama3.1:8b-instruct-q4_K_M",
                "nvidia": "llama3.1:8b-instruct-q4_K_M",
            }
        ],
        "custom": [],
    }

    machine = {"machine_id": "m1", "gpu": {"type": "apple"}}

    import unittest.mock as mock
    with mock.patch.object(app_mod, "load_models_registry", return_value=registry):
        resolved = app_mod.resolve_runtime_model(machine, "ollama", "llama3.1-8b-q4")

    assert resolved == "llama3.1:8b-instruct-q4_K_M"
