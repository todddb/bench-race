from __future__ import annotations

import json
import platform
import re
import shutil
import subprocess
from typing import Any, Dict, Optional, Tuple

import psutil


def _parse_memory_string(value: str) -> Optional[int]:
    if not value:
        return None
    cleaned = value.replace(",", "").strip()
    match = re.match(r"([\d.]+)\s*([A-Za-z]+)?", cleaned)
    if not match:
        return None
    number = float(match.group(1))
    unit = (match.group(2) or "b").lower()
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


def _read_system_profiler() -> Dict[str, Any]:
    if not shutil.which("system_profiler"):
        return {}
    try:
        output = subprocess.check_output(["system_profiler", "SPDisplaysDataType", "-json"], text=True)
        return json.loads(output)
    except Exception:
        return {}


def discover_hardware() -> Dict[str, Any]:
    cpu_cores = psutil.cpu_count(logical=True)
    cpu_physical = psutil.cpu_count(logical=False)
    total_ram = psutil.virtual_memory().total

    gpu_name = None
    gpu_vram_bytes = None
    gpu_type = None
    cuda_compute = None

    system = platform.system()

    nvml, handle = _try_load_nvml()
    if nvml and handle:
        try:
            gpu_name = nvml.nvmlDeviceGetName(handle).decode("utf-8")
        except Exception:
            gpu_name = None
        try:
            mem = nvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_vram_bytes = int(mem.total)
        except Exception:
            gpu_vram_bytes = None
        try:
            major, minor = nvml.nvmlDeviceGetCudaComputeCapability(handle)
            cuda_compute = [int(major), int(minor)]
        except Exception:
            cuda_compute = None
        gpu_type = "discrete"
    elif system == "Darwin":
        profiler = _read_system_profiler()
        displays = profiler.get("SPDisplaysDataType", [])
        if displays:
            first = displays[0]
            gpu_name = first.get("sppci_model") or first.get("spdisplays_chipset_model")
            vram_label = first.get("spdisplays_vram_shared") or first.get("spdisplays_vram")
            gpu_vram_bytes = _parse_memory_string(str(vram_label)) if vram_label else None
            gpu_type = "unified"

    return {
        "cpu_cores": cpu_cores,
        "cpu_physical_cores": cpu_physical,
        "total_system_ram_bytes": total_ram,
        "gpu_name": gpu_name,
        "gpu_vram_bytes": gpu_vram_bytes,
        "gpu_type": gpu_type,
        "cuda_compute": cuda_compute,
    }
