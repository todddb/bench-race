from pathlib import Path
import importlib


def _ensure_machines_yaml():
    machines_cfg = Path("config/machines.yaml")
    machines_cfg.parent.mkdir(parents=True, exist_ok=True)
    if not machines_cfg.exists():
        machines_cfg.write_text(
            "machines:\n"
            "  - machine_id: test1\n"
            "    label: Test\n"
            "    agent_base_url: http://127.0.0.1:9001\n",
            encoding="utf-8",
        )


def test_load_models_registry(tmp_path):
    cfg = tmp_path / "models.json"
    cfg.write_text(
        '{"version":2,"shared_baseline":[{"id":"m1","display_name":"Model 1"}],"architectures":{"apple_silicon":{"backend":"mlx","models":[{"id":"m2"}]}}}',
        encoding="utf-8",
    )

    loader = importlib.import_module("central.config_loader")
    registry = loader.load_models_registry(cfg)
    assert registry["version"] == 2
    assert registry["shared_baseline"][0]["id"] == "m1"
    assert registry["architectures"]["apple_silicon"]["backend"] == "mlx"


def test_load_models_registry_missing_returns_empty(tmp_path):
    loader = importlib.import_module("central.config_loader")
    registry = loader.load_models_registry(tmp_path / "missing.json")
    assert registry == {"version": 0, "shared_baseline": [], "architectures": {}}


def test_api_models_config_and_backend_filtering(monkeypatch):
    _ensure_machines_yaml()
    app_mod = importlib.import_module("central.app")
    registry = {
        "version": 2,
        "shared_baseline": [{"id": "ollama-a", "display_name": "Ollama A"}],
        "architectures": {
            "apple_silicon": {"backend": "mlx", "models": [{"id": "mlx-a"}]},
            "nvidia_blackwell": {"backend": "trtllm", "models": [{"id": "trt-a"}]},
        },
    }
    monkeypatch.setattr(app_mod, "load_models_registry", lambda: registry)
    app_mod.app.config["TESTING"] = True

    with app_mod.app.test_client() as client:
        cfg_resp = client.get("/api/models/config")
        assert cfg_resp.status_code == 200
        assert cfg_resp.get_json()["version"] == 2

        ollama_resp = client.get("/api/models?backend=ollama")
        assert ollama_resp.status_code == 200
        assert ollama_resp.get_json()["models"][0]["id"] == "ollama-a"

        mlx_resp = client.get("/api/models?backend=mlx&architecture=apple_silicon")
        assert mlx_resp.status_code == 200
        assert mlx_resp.get_json()["models"][0]["id"] == "mlx-a"

        trt_resp = client.get("/api/models?backend=trtllm&architecture=nvidia_blackwell")
        assert trt_resp.status_code == 200
        assert trt_resp.get_json()["models"][0]["id"] == "trt-a"
