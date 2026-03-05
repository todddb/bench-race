import importlib
from pathlib import Path


def test_machine_sync_route_removed():
    cfg = Path("config/machines.yaml")
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
    central_app.app.config["TESTING"] = True
    with central_app.app.test_client() as client:
        response = client.post("/api/machines/bootstrap/sync")
    assert response.status_code == 404
