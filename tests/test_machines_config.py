from central import config_loader


def test_load_machines_config_missing_returns_empty(tmp_path):
    assert config_loader.load_machines_config(tmp_path / "machines.yaml") == {}


def test_load_machines_config_malformed_returns_empty(tmp_path):
    bad = tmp_path / "machines.yaml"
    bad.write_text("machines: [", encoding="utf-8")
    assert config_loader.load_machines_config(bad) == {}


def test_load_machines_config_happy_path(tmp_path):
    cfg = tmp_path / "machines.yaml"
    cfg.write_text(
        "machines:\n"
        "  - machine_id: test1\n"
        "    label: Test\n"
        "    agent_base_url: http://127.0.0.1:9001\n",
        encoding="utf-8",
    )
    loaded = config_loader.load_machines_config(cfg)
    assert loaded["machines"][0]["machine_id"] == "test1"
