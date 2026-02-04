from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from agent.errors import classify_comfy_error


@dataclass
class ComfyPreflightResult:
    comfyui_gpu_ok: bool
    comfyui_cpu_ok: bool
    error: Optional[str] = None


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _comfyui_python(install_dir: Path) -> Optional[Path]:
    venv_python = install_dir / ".venv" / "bin" / "python"
    if venv_python.exists():
        return venv_python
    return None


def run_comfyui_cuda_probe(install_dir: Path) -> Optional[ComfyPreflightResult]:
    """Run a lightweight CUDA probe using ComfyUI's venv if available."""
    if platform.system() != "Linux":
        return None
    if not shutil.which("nvidia-smi"):
        return None

    python_path = _comfyui_python(install_dir)
    if not python_path:
        return None

    env = os.environ.copy()

    command = [
        str(python_path),
        "-c",
        "import torch; torch.cuda.is_available(); torch.randn(1, device='cuda'); print('ok')",
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, env=env, timeout=20)
        return ComfyPreflightResult(comfyui_gpu_ok=True, comfyui_cpu_ok=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or exc.stdout or "").strip()
        classified = classify_comfy_error(stderr)
        if classified.get("category") == "cuda_unsupported_arch":
            return ComfyPreflightResult(comfyui_gpu_ok=False, comfyui_cpu_ok=True, error=stderr)
        return ComfyPreflightResult(comfyui_gpu_ok=False, comfyui_cpu_ok=True, error=stderr)
    except Exception as exc:
        return ComfyPreflightResult(comfyui_gpu_ok=False, comfyui_cpu_ok=False, error=str(exc))


def force_cpu_enabled() -> bool:
    return _truthy(os.getenv("COMFY_FORCE_CPU", ""))
