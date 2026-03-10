"""Tests for backend selection and model resolution.

IMPORTANT ARCHITECTURAL RULE:
Central is the sole authority for model resolution.
Agents receive only fully resolved model strings.
The select_backend endpoint on the agent receives runtime model strings
(already resolved by Central) and uses Ollama tag matching only for
local installation verification — NOT for registry-based resolution.
"""
import asyncio
import importlib
from pathlib import Path


def test_select_backend_resolves_ollama_model_before_start(monkeypatch):
    """Agent's select_backend uses resolved_ollama_pull_name for Ollama
    tag matching (local installation check), which is allowed."""
    cfg_path = Path("agent/config/agent.yaml")
    if not cfg_path.exists():
        cfg_path.write_text(Path("agent/config/agent.yaml.example").read_text(encoding="utf-8"), encoding="utf-8")

    agent_app = importlib.import_module("agent.agent_app")

    captured = {}

    installed_tags = {
        "llama3.1:8b-instruct-q4_K_M",
        "llama3.1:8b-instruct-q8_0",
    }

    def fake_fetch_installed_ollama_tags(_url):
        return installed_tags

    def fake_resolved_ollama_pull_name(model_entry, tags):
        model = model_entry.get("model", "")
        # First call: resolve original model; subsequent calls: passthrough
        if model in installed_tags:
            return model
        return "llama3.1:8b-instruct-q4_K_M"

    async def fake_run_agent_script(command, *args):
        captured["command"] = command
        captured["args"] = args
        return {"ok": True}

    async def fake_check_backend_health(_backend):
        return {"healthy": True}

    monkeypatch.setitem(agent_app.CFG, "ollama", {"base_url": "http://127.0.0.1:11434"})
    monkeypatch.setattr(agent_app, "fetch_installed_ollama_tags", fake_fetch_installed_ollama_tags)
    monkeypatch.setattr(agent_app, "resolved_ollama_pull_name", fake_resolved_ollama_pull_name)
    monkeypatch.setattr(agent_app, "_run_agent_script", fake_run_agent_script)
    monkeypatch.setattr(agent_app, "_check_backend_health", fake_check_backend_health)

    req = agent_app.BackendSelectRequest(
        backend="ollama",
        model="llama3.1:8b-instruct",
    )
    resp = asyncio.run(agent_app.select_backend(req))

    assert resp["ok"] is True
    assert captured["command"] == "start-backend"
    assert captured["args"] == ("ollama", "llama3.1:8b-instruct-q4_K_M")


def test_central_resolves_registry_id_to_ollama_tag():
    """Central resolves abstract registry IDs to Ollama runtime tags.
    This is the ONLY place where registry ID → runtime string translation
    should happen."""
    app_mod = importlib.import_module("central.app")

    import unittest.mock as mock
    registry = {
        "ollama": [
            {
                "id": "llama3.1-8b-q4",
                "apple": "llama3.1:8b-instruct-q4_K_M",
                "nvidia": "llama3.1:8b-instruct-q4_K_M",
            }
        ],
        "custom": [],
    }

    machine = {"machine_id": "m1", "gpu": {"type": "apple"}}
    with mock.patch.object(app_mod, "load_models_registry", return_value=registry):
        resolved = app_mod.resolve_runtime_model(machine, "ollama", "llama3.1-8b-q4")

    assert resolved == "llama3.1:8b-instruct-q4_K_M"


def test_central_resolves_registry_id_before_agent_dispatch():
    """Central translates registry ID and sends resolved string to agent.
    Agent never sees the abstract ID."""
    app_mod = importlib.import_module("central.app")

    import unittest.mock as mock
    registry = {
        "ollama": [
            {
                "id": "llama3.1-8b-q4",
                "apple": "llama3.1:8b-instruct-q4_K_M",
                "nvidia": "llama3.1:8b-instruct-q4_K_M",
            }
        ],
        "custom": [],
    }

    machine = {"machine_id": "m1", "gpu": {"type": "apple"}}
    with mock.patch.object(app_mod, "load_models_registry", return_value=registry):
        backend, resolved = app_mod._resolve_model_for_machine(machine, "ollama", "llama3.1-8b-q4")

    assert backend == "ollama"
    assert resolved == "llama3.1:8b-instruct-q4_K_M"
    # The resolved string is what should be sent to the agent, never the abstract ID
    assert resolved != "llama3.1-8b-q4"
