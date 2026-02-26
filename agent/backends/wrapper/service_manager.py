from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


class ServiceManager:
    """Shell helper that reuses existing bench-race lifecycle scripts."""

    def __init__(self, repo_root: Optional[Path] = None) -> None:
        self.repo_root = repo_root or Path(__file__).resolve().parents[3]

    def _run(self, command: str, env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(self.repo_root),
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, **(env or {})},
        )
        return {
            "ok": proc.returncode == 0,
            "rc": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "command": command,
        }

    @staticmethod
    def trt_engine_id(model_id: str) -> str:
        return model_id.replace("/", "__")

    def start_backend(self, backend: str, model_id: Optional[str] = None) -> Dict[str, Any]:
        if backend == "mlx":
            return self._run("./scripts/agent start-mlx")
        if backend == "trt":
            if not model_id:
                return {"ok": False, "rc": 2, "stdout": "", "stderr": "model_id required for trt"}
            return self._run(
                f"./agent/backends/trtllm_run.sh restart",
                env={"TRTLLM_MODEL": self.trt_engine_id(model_id)},
            )
        return {"ok": False, "rc": 2, "stdout": "", "stderr": f"unknown backend: {backend}"}

    def stop_backend(self, backend: str) -> Dict[str, Any]:
        if backend == "mlx":
            return self._run("./scripts/agent stop-mlx")
        if backend == "trt":
            return self._run("./agent/backends/trtllm_run.sh stop")
        return {"ok": False, "rc": 2, "stdout": "", "stderr": f"unknown backend: {backend}"}

    def switch_backend(self, backend: str, model_id: str) -> Dict[str, Any]:
        return self.start_backend(backend=backend, model_id=model_id)
