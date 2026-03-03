import importlib
from pathlib import Path


def test_job_dispatch_registry_id_translates_to_ollama(monkeypatch):
    cfg_path = Path("agent/config/agent.yaml")
    if not cfg_path.exists():
        cfg_path.write_text(Path("agent/config/agent.yaml.example").read_text(encoding="utf-8"), encoding="utf-8")

    agent_app = importlib.import_module("agent.agent_app")

    installed = ["llama3.1:8b-instruct-q4_K_M"]

    def mock_fetch(*_args, **_kwargs):
        return installed

    monkeypatch.setattr(agent_app, "fetch_installed_ollama_tags", mock_fetch)

    model = "llama3.1-8b-q4"

    translated = agent_app.registry_id_to_ollama_tag(model)
    resolved = agent_app.resolved_ollama_pull_name({"model": translated}, installed)

    assert resolved == "llama3.1:8b-instruct-q4_K_M"
