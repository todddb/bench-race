from central.runtime_metrics import normalize_runtime_metrics


def test_runtime_metrics_normalizes_gpu_metrics_available():
    payload = {
        "timestamps": [1.0, 2.0],
        "cpu_pct": [10, 20],
        "gpu_metrics_available": False,
        "sampler_interval_s": [1.0, 1.0],
    }

    normalized = normalize_runtime_metrics(payload)

    assert normalized["timestamps"] == [1.0, 2.0]
    assert normalized["cpu_pct"] == [10, 20]
    assert normalized["gpu_metrics_available"] == [False, False]
    assert normalized["sampler_interval_s"] == 1.0


def test_runtime_metrics_normalizes_single_sample_series():
    payload = {
        "timestamps": [5.0],
        "cpu_pct": 33,
        "gpu_pct": None,
        "gpu_metrics_available": True,
    }

    normalized = normalize_runtime_metrics(payload)

    assert normalized["timestamps"] == [5.0]
    assert normalized["cpu_pct"] == [33]
    assert normalized["gpu_pct"] == [None]
    assert normalized["gpu_metrics_available"] == [True]
