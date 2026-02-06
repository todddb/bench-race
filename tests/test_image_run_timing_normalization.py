def test_normalize_image_machine_timing_fallbacks():
    import importlib
    from pathlib import Path

    config_path = Path(__file__).resolve().parents[1] / "central" / "config" / "machines.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        config_path.write_text("machines: []\n", encoding="utf-8")

    central_app = importlib.import_module("central.app")

    machine_entry = {
        "dispatch_at_ms": 1_700_000_000_000,
        "queue_latency_ms": 2_500,
        "total_ms": 4_000,
    }

    central_app._normalize_image_machine_timing(machine_entry)

    assert machine_entry["started_at_ms"] == 1_700_000_002_500
    assert machine_entry["first_progress_at_ms"] == 1_700_000_002_500
    assert machine_entry["completed_at_ms"] == 1_700_000_006_500
