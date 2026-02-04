from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional


def compute_model_fit_score(
    model_bytes: int,
    vram_bytes: int,
    activation_multiplier: float = 1.7,
    usable_vram_fraction: float = 0.9,
) -> Dict[str, float | int | str]:
    usable_vram = int(vram_bytes * usable_vram_fraction)
    estimated_peak = int(model_bytes * activation_multiplier)
    if usable_vram <= 0:
        return {
            "estimated_peak_bytes": estimated_peak,
            "usable_vram_bytes": usable_vram,
            "fit_ratio": float("inf"),
            "fit_score": 0.0,
            "label": "fail",
            "color": "#DC2626",
        }
    if estimated_peak <= 0:
        fit_ratio = float("inf")
    else:
        fit_ratio = usable_vram / estimated_peak
    fit_score = fit_ratio
    if fit_ratio >= 1.2:
        label, color = "good", "#059669"
    elif fit_ratio >= 1.0:
        label, color = "risk", "#F97316"
    else:
        label, color = "fail", "#DC2626"
    return {
        "estimated_peak_bytes": estimated_peak,
        "usable_vram_bytes": usable_vram,
        "fit_ratio": fit_ratio,
        "fit_score": fit_score,
        "label": label,
        "color": color,
    }


def get_model_size_bytes(model_name_or_path: str, fallback_gb: Optional[float] = None) -> Optional[int]:
    if not model_name_or_path:
        return None
    path = Path(model_name_or_path)
    if path.exists() and path.is_file():
        return path.stat().st_size
    if fallback_gb is None:
        return None
    return int(fallback_gb * 1024**3)


def safe_ratio(numerator: Optional[int], denominator: Optional[int]) -> float:
    if not numerator or not denominator:
        return math.inf
    return numerator / denominator
