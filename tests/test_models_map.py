from pathlib import Path
import importlib


def test_load_models_map(tmp_path):
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

