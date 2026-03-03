from pathlib import Path
import importlib
import sys


CFG_PATH = Path("config/machines.yaml")


def _load_central_app_with_test_machines():
    CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
    original = CFG_PATH.read_text(encoding="utf-8") if CFG_PATH.exists() else None
    try:
        CFG_PATH.write_text(
            "machines:\n"
            "  - machine_id: test1\n"
            "    label: Test\n"
            "    agent_base_url: http://127.0.0.1:9001\n",
            encoding="utf-8",
        )
        sys.modules.pop("central.app", None)
        return importlib.import_module("central.app")
    finally:
        if original is None:
            CFG_PATH.unlink(missing_ok=True)
        else:
            CFG_PATH.write_text(original, encoding="utf-8")


def test_load_models_registry_v3(tmp_path):
    cfg = tmp_path / "models.json"
    cfg.write_text(
        '{"version":3,"ollama":[{"id":"m1","display_name":"Model 1","apple":"m1:tag","nvidia":"m1:tag"}],"custom":[{"id":"c1","display_name":"Custom 1","mlx_hf_id":"mlx/c1","trt-llm_hf_id":"meta/c1"}],"comfyui":[{"id":"sdxl","display_name":"SDXL","download_url":"https://example.com/sdxl.safetensors"}]}',
        encoding="utf-8",
    )

    loader = importlib.import_module("central.config_loader")
    registry = loader.load_models_registry(cfg)
    assert registry["version"] == 3
    assert registry["ollama"][0]["id"] == "m1"
    assert registry["custom"][0]["id"] == "c1"
    assert registry["comfyui"][0]["id"] == "sdxl"


def test_load_models_registry_missing_returns_empty(tmp_path):
    loader = importlib.import_module("central.config_loader")
    registry = loader.load_models_registry(tmp_path / "missing.json")
    assert registry == {"version": 0, "ollama": [], "custom": [], "comfyui": []}


def test_api_models_config_and_backend_filtering_v3(monkeypatch):
    app_mod = _load_central_app_with_test_machines()
    registry = {
        "version": 3,
        "ollama": [
            {"id": "ollama-a", "display_name": "Ollama A", "apple": "ollama-a:tag", "nvidia": "ollama-a:tag"},
            {"id": "ollama-b", "display_name": "Ollama B", "apple": "ollama-b:tag", "nvidia": "ollama-b:tag"},
        ],
        "custom": [
            {"id": "custom-a", "display_name": "Custom A", "mlx_hf_id": "mlx/custom-a", "trt-llm_hf_id": "meta/custom-a"},
            {"id": "custom-b", "display_name": "Custom B", "mlx_hf_id": "mlx/custom-b", "trt-llm_hf_id": "meta/custom-b"},
        ],
    }
    monkeypatch.setattr(app_mod, "load_models_registry", lambda: registry)
    app_mod.app.config["TESTING"] = True

    with app_mod.app.test_client() as client:
        cfg_resp = client.get("/api/models/config")
        assert cfg_resp.status_code == 200
        assert cfg_resp.get_json()["version"] == 3

        ollama_resp = client.get("/api/models?backend=ollama")
        assert ollama_resp.status_code == 200
        assert [m["id"] for m in ollama_resp.get_json()["models"]] == ["ollama-a", "ollama-b"]

        custom_resp = client.get("/api/models?backend=custom")
        assert custom_resp.status_code == 200
        assert [m["id"] for m in custom_resp.get_json()["models"]] == ["custom-a", "custom-b"]

        mlx_resp = client.get("/api/models?backend=mlx")
        assert mlx_resp.status_code == 200
        assert [m["id"] for m in mlx_resp.get_json()["models"]] == ["custom-a", "custom-b"]

        trt_resp = client.get("/api/models?backend=trtllm")
        assert trt_resp.status_code == 200
        assert [m["id"] for m in trt_resp.get_json()["models"]] == ["custom-a", "custom-b"]
