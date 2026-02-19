import importlib
from pathlib import Path


def test_machine_sync_posts_target_dir_and_sanitize(monkeypatch):
    cfg = Path("central/config/machines.yaml")
    cfg.parent.mkdir(parents=True, exist_ok=True)
    if not cfg.exists():
        cfg.write_text(
            "machines:\n"
            "  - machine_id: bootstrap\n"
            "    label: Bootstrap\n"
            "    agent_base_url: http://127.0.0.1:9001\n",
            encoding="utf-8",
        )

    central_app = importlib.import_module("central.app")

    class _Resp:
        def __init__(self, data):
            self._data = data
            self.status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return self._data

    posted = {}

    monkeypatch.setattr(
        central_app,
        "_required_models",
        lambda: {"llm": ["llama3.1:8b"], "whisper": [], "sdxl_profiles": []},
    )
    monkeypatch.setattr(
        central_app,
        "MACHINES",
        [{"machine_id": "m1", "agent_base_url": "http://agent"}],
    )

    def fake_get(url, timeout=None):
        return _Resp({"ollama_models": []})

    def fake_post(url, json=None, timeout=None):
        posted["url"] = url
        posted["json"] = json
        return _Resp({"sync_id": "abc"})

    monkeypatch.setattr(central_app.requests, "get", fake_get)
    monkeypatch.setattr(central_app.requests, "post", fake_post)

    central_app.app.config["TESTING"] = True
    with central_app.app.test_client() as client:
        response = client.post("/api/machines/m1/sync")

    assert response.status_code == 200
    assert posted["url"] == "http://agent/models/sync"
    assert posted["json"]["target_dir"] == "ollama"
    assert posted["json"]["sanitize_names"] is True
    assert posted["json"]["llm"] == ["llama3.1:8b"]
