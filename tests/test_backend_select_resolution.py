import asyncio
import importlib
from pathlib import Path


def test_select_backend_resolves_ollama_model_before_start(monkeypatch):
    cfg_path = Path("agent/config/agent.yaml")
    if not cfg_path.exists():
        cfg_path.write_text(Path("agent/config/agent.yaml.example").read_text(encoding="utf-8"), encoding="utf-8")

    agent_app = importlib.import_module("agent.agent_app")

    captured = {}

    def fake_fetch_installed_ollama_tags(_url):
        return {
            "llama3.1:8b-instruct-q4_K_M",
            "llama3.1:8b-instruct-q8_0",
        }

    def fake_resolved_ollama_pull_name(model_entry, installed_tags):
        assert model_entry["model"] == "llama3.1:8b-instruct"
        assert installed_tags == {
            "llama3.1:8b-instruct-q4_K_M",
            "llama3.1:8b-instruct-q8_0",
        }
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



def test_registry_id_to_ollama_tag_returns_tag_for_ollama_entry(monkeypatch):
    agent_app = importlib.import_module("agent.agent_app")

    monkeypatch.setattr(
        agent_app,
        "load_models_registry",
        lambda: {
            "ollama": [
                {
                    "id": "llama3.1-8b-q4",
                    "apple": "llama3.1:8b-instruct-q4_K_M",
                    "nvidia": "llama3.1:8b-instruct-q4_K_M",
                }
            ]
        },
    )

    assert agent_app.registry_id_to_ollama_tag("llama3.1-8b-q4") == "llama3.1:8b-instruct-q4_K_M"


def test_registry_id_translates_before_ollama_resolution(monkeypatch):
    cfg_path = Path("agent/config/agent.yaml")
    if not cfg_path.exists():
        cfg_path.write_text(Path("agent/config/agent.yaml.example").read_text(encoding="utf-8"), encoding="utf-8")

    agent_app = importlib.import_module("agent.agent_app")

    captured = {}

    def fake_load_models_registry():
        return {
            "ollama": [
                {
                    "id": "llama3.1-8b-q4",
                    "apple": "llama3.1:8b-instruct-q4_K_M",
                    "nvidia": "llama3.1:8b-instruct-q4_K_M",
                }
            ]
        }

    def fake_fetch_installed_ollama_tags(_url):
        return {"llama3.1:8b-instruct-q4_K_M"}

    def fake_resolved_ollama_pull_name(model_entry, _installed_tags):
        captured["model_entry"] = model_entry
        return model_entry["model"]

    async def fake_run_agent_script(command, *args):
        captured["command"] = command
        captured["args"] = args
        return {"ok": True}

    async def fake_check_backend_health(_backend):
        return {"healthy": True}

    monkeypatch.setitem(agent_app.CFG, "ollama", {"base_url": "http://127.0.0.1:11434"})
    monkeypatch.setattr(agent_app, "load_models_registry", fake_load_models_registry)
    monkeypatch.setattr(agent_app, "fetch_installed_ollama_tags", fake_fetch_installed_ollama_tags)
    monkeypatch.setattr(agent_app, "resolved_ollama_pull_name", fake_resolved_ollama_pull_name)
    monkeypatch.setattr(agent_app, "_run_agent_script", fake_run_agent_script)
    monkeypatch.setattr(agent_app, "_check_backend_health", fake_check_backend_health)

    req = agent_app.BackendSelectRequest(backend="ollama", model="llama3.1-8b-q4")
    resp = asyncio.run(agent_app.select_backend(req))

    assert resp["ok"] is True
    assert captured["model_entry"] == {"model": "llama3.1:8b-instruct-q4_K_M", "id": "llama3.1-8b-q4"}
    assert captured["command"] == "start-backend"
    assert captured["args"] == ("ollama", "llama3.1:8b-instruct-q4_K_M")
