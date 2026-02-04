from types import SimpleNamespace

from agent.runtime_sampler import RuntimeSampler, RuntimeSamplerConfig


def test_runtime_sampler_schema(monkeypatch):
    monkeypatch.setattr("psutil.cpu_percent", lambda interval=None: 12.5)
    monkeypatch.setattr(
        "psutil.virtual_memory",
        lambda: SimpleNamespace(
            used=1024 * 1024 * 3,
            total=1024 * 1024 * 8,
            available=1024 * 1024 * 5,
        ),
    )

    sampler = RuntimeSampler(RuntimeSamplerConfig(interval_s=1, buffer_len=3))
    sampler.sample_once()
    snapshot = sampler.snapshot()

    assert snapshot["sampler_interval_s"] == 1
    assert snapshot["cpu_pct"] == [12.5]
    assert snapshot["gpu_pct"] == [None]
    assert snapshot["vram_used_mib"] == [None]
    assert snapshot["system_mem_used_mib"] == [3.0]
    assert snapshot["ram_used_bytes"] == [1024 * 1024 * 3]
    assert len(snapshot["timestamps"]) == 1
    assert snapshot["gpu_metrics_available"] is False
