import importlib
from pathlib import Path

from fastapi.testclient import TestClient


def test_agent_legacy_sync_routes_removed():
    cfg_path = Path("agent/config/agent.yaml")
    if not cfg_path.exists():
        cfg_path.write_text(Path("agent/config/agent.yaml.example").read_text(encoding="utf-8"), encoding="utf-8")

    agent_app = importlib.import_module("agent.agent_app")
    client = TestClient(agent_app.app)

    assert client.post("/models/sync", json={"llm": ["m1"]}).status_code == 404
    assert client.post("/_internal/sync_models", json={"job_id": "j1", "models": []}).status_code == 404
