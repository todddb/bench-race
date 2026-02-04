from central.fit_util import compute_model_fit_score


def _fit_ratio_for(model_bytes: int, vram_bytes: int, activation_multiplier=1.7, usable_fraction=0.9) -> float:
    usable = vram_bytes * usable_fraction
    peak = model_bytes * activation_multiplier
    return peak / usable


def test_compute_model_fit_score_boundaries():
    vram_bytes = 1000
    usable = vram_bytes * 0.9

    def model_for_ratio(ratio):
        return int((ratio * usable) / 1.7)

    good = compute_model_fit_score(model_for_ratio(0.6), vram_bytes)
    marginal = compute_model_fit_score(model_for_ratio(0.8), vram_bytes)
    risk = compute_model_fit_score(model_for_ratio(1.0), vram_bytes)
    fail = compute_model_fit_score(model_for_ratio(1.1), vram_bytes)

    assert good["label"] == "good"
    assert marginal["label"] == "marginal"
    assert risk["label"] == "risk"
    assert fail["label"] == "fail"
    assert _fit_ratio_for(model_for_ratio(0.6), vram_bytes) <= 0.6
