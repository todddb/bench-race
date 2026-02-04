from __future__ import annotations

import json
import platform
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Tuple

import psutil


def _parse_memory_string(value: str) -> Optional[int]:
    if not value:
        return None
    parts = value.replace(",", "").split()
    if not parts:
        return None
    try:
        number = float(parts[0])
    except ValueError:
        return None
    unit = parts[1].lower() if len(parts) > 1 else "b"
    if unit.startswith("gb"):
        return int(number * 1024**3)
    if unit.startswith("mb"):
        return int(number * 1024**2)
    if unit.startswith("kb"):
        return int(number * 1024)
    return int(number)


def _try_load_nvml() -> Tuple[Optional[Any], Optional[Any]]:
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml, handle
    except Exception:
        return None, None


def _powermetrics_gpu_available() -> bool:
    return platform.system() == "Darwin" and bool(shutil.which("powermetrics"))


@dataclass
class RuntimeSamplerConfig:
    interval_s: float = 1.0
    buffer_len: int = 120


class RuntimeSampler:
    def __init__(self, config: RuntimeSamplerConfig):
        self.interval_s = max(config.interval_s, 0.25)
        self.buffer_len = max(config.buffer_len, 10)
        self.cpu_pct: Deque[Optional[float]] = deque(maxlen=self.buffer_len)
        self.gpu_pct: Deque[Optional[float]] = deque(maxlen=self.buffer_len)
        self.vram_used_mib: Deque[Optional[float]] = deque(maxlen=self.buffer_len)
        self.vram_total_mib: Deque[Optional[float]] = deque(maxlen=self.buffer_len)
        self.system_mem_used_mib: Deque[Optional[float]] = deque(maxlen=self.buffer_len)
        self.ram_used_bytes: Deque[Optional[int]] = deque(maxlen=self.buffer_len)
        self.timestamps: Deque[float] = deque(maxlen=self.buffer_len)
        self._nvml, self._nvml_handle = _try_load_nvml()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._gpu_metrics_available = self._nvml is not None or _powermetrics_gpu_available()

    def _sample_cpu(self) -> float:
        return float(psutil.cpu_percent(interval=None))

    def _sample_system_memory(self) -> Tuple[int, float]:
        mem = psutil.virtual_memory()
        used_bytes = int(mem.used)
        return used_bytes, float(used_bytes) / (1024 * 1024)

    def _sample_gpu(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        if self._nvml and self._nvml_handle:
            util = self._nvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
            mem = self._nvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            return float(util.gpu), float(mem.used) / (1024 * 1024), float(mem.total) / (1024 * 1024)
        return None, None, None

    def sample_once(self) -> None:
        self.cpu_pct.append(self._sample_cpu())
        used_bytes, used_mib = self._sample_system_memory()
        self.ram_used_bytes.append(used_bytes)
        self.system_mem_used_mib.append(used_mib)
        gpu_pct, vram_used, vram_total = self._sample_gpu()
        self.gpu_pct.append(gpu_pct)
        self.vram_used_mib.append(vram_used)
        self.vram_total_mib.append(vram_total)
        self.timestamps.append(time.time())

    def run(self) -> None:
        while not self._stop_event.is_set():
            self.sample_once()
            time.sleep(self.interval_s)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def snapshot(self) -> Dict[str, Any]:
        has_gpu_samples = any(value is not None for value in self.gpu_pct)
        return {
            "sampler_interval_s": self.interval_s,
            "cpu_pct": list(self.cpu_pct),
            "gpu_pct": list(self.gpu_pct),
            "vram_used_mib": list(self.vram_used_mib),
            "vram_total_mib": list(self.vram_total_mib),
            "system_mem_used_mib": list(self.system_mem_used_mib),
            "ram_used_bytes": list(self.ram_used_bytes),
            "timestamps": list(self.timestamps),
            "gpu_metrics_available": self._gpu_metrics_available and has_gpu_samples,
        }

    def serialize(self) -> str:
        return json.dumps(self.snapshot())
