from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

SERIES_FIELDS = [
    "cpu_pct",
    "gpu_pct",
    "system_mem_used_mib",
    "system_mem_total_mib",
    "vram_used_mib",
    "vram_total_mib",
    "ram_used_bytes",
    "ram_total_bytes",
]


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _infer_length(metrics: Dict[str, Any]) -> int:
    length = 0
    for key in SERIES_FIELDS + ["gpu_metrics_available"]:
        value = metrics.get(key)
        if isinstance(value, list):
            length = max(length, len(value))
        elif value is not None:
            length = max(length, 1)
    return length


def _pad_or_trim(values: List[Any], length: int, pad_value: Any) -> List[Any]:
    if length <= 0:
        return []
    if len(values) >= length:
        return values[:length]
    return values + [pad_value] * (length - len(values))


def _normalize_sampler_interval(value: Any) -> Optional[float]:
    if isinstance(value, list):
        for item in value:
            try:
                resolved = float(item)
            except (TypeError, ValueError):
                continue
            if resolved > 0:
                return resolved
        return None
    if value is None:
        return None
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    return resolved if resolved > 0 else None


def _normalize_series(values: Any, length: int) -> List[Any]:
    series = _as_list(values)
    if length <= 0:
        return []
    if len(series) == 1 and not isinstance(values, list):
        return [series[0]] * length
    return _pad_or_trim(series, length, None)


def _normalize_bool_series(values: Any, length: int) -> List[bool]:
    series = _as_list(values)
    if length <= 0:
        return []
    if len(series) == 1 and not isinstance(values, list):
        return [bool(series[0])] * length
    if len(series) < length:
        pad_value = bool(series[-1]) if series else False
        return _pad_or_trim([bool(item) for item in series], length, pad_value)
    return [bool(item) for item in series[:length]]


def normalize_runtime_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(metrics, dict):
        return {}

    normalized = dict(metrics)
    raw_timestamps = metrics.get("timestamps")
    timestamps = _as_list(raw_timestamps)
    length = len(timestamps)
    if length == 0:
        length = _infer_length(metrics)
        if length > 0:
            timestamps = [None] * length

    normalized["timestamps"] = _pad_or_trim(timestamps, length, None)
    normalized["sampler_interval_s"] = _normalize_sampler_interval(metrics.get("sampler_interval_s"))

    for key in SERIES_FIELDS:
        normalized[key] = _normalize_series(metrics.get(key), length)

    normalized["gpu_metrics_available"] = _normalize_bool_series(metrics.get("gpu_metrics_available"), length)

    return normalized


def normalize_runtime_metrics_map(metrics_map: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(metrics_map, dict):
        return {}
    return {machine_id: normalize_runtime_metrics(payload or {}) for machine_id, payload in metrics_map.items()}
