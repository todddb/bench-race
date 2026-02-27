from pathlib import Path
import importlib


def test_load_models_map_and_filter_endpoint(tmp_path, monkeypatch):
    machines_cfg = Path("central/config/machines.yaml")
    machines_cfg.parent.mkdir(parents=True, exist_ok=True)
    if not machines_cfg.exists():
        machines_cfg.write_text(
            "machines:\n"
            "  - machine_id: test1\n"
            "    label: Test\n"
            "    agent_base_url: http://127.0.0.1:9001\n",
            encoding="utf-8",
        )

    cfg = tmp_path / "models_map.json"
    cfg.write_text(
        '{"models":[{"display_name":"m1","backend":"ollama"},{"display_name":"m2","backend":"custom"}]}',
        encoding="utf-8",
    )

    loader = importlib.import_module("central.config_loader")
    models = loader.load_models_map(cfg)
    assert len(models) == 2

    app_mod = importlib.import_module("central.app")
    monkeypatch.setattr(app_mod, "load_models_map", lambda: models)
    app_mod.app.config["TESTING"] = True

    with app_mod.app.test_client() as client:
        resp = client.get("/api/models?backend=ollama")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data) == 1
        assert data[0]["display_name"] == "m1"
