from central import config_loader


def test_load_backends_config_missing_returns_empty(tmp_path):
    missing = tmp_path / "missing.yaml"
    assert config_loader.load_backends_config(missing) == {}


def test_load_backends_config_malformed_returns_empty(tmp_path):
    bad = tmp_path / "backends.yaml"
    bad.write_text("trtllm: [", encoding="utf-8")
    assert config_loader.load_backends_config(bad) == {}


def test_load_backends_config_happy_path(tmp_path):
    cfg = tmp_path / "backends.yaml"
    cfg.write_text(
        "trtllm:\n"
        "  image: example\n"
        "  port: 8000\n"
        "mlx:\n"
        "  host: 127.0.0.1\n"
        "  port: 8321\n",
        encoding="utf-8",
    )
    loaded = config_loader.load_backends_config(cfg)
    assert loaded["trtllm"]["image"] == "example"
    assert loaded["mlx"]["port"] == 8321
