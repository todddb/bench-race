from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


class ServiceManager:
    """Deterministic backend lifecycle manager for wrapper-managed engines."""

    TRT_CONTAINER_NAME = "bench-race-trtllm"

    def __init__(self, repo_root: Optional[Path] = None) -> None:
        self.repo_root = repo_root or Path(__file__).resolve().parents[3]
        self.mlx_pidfile = self.repo_root / "agent" / "run" / "mlx.pid"
        self.ollama_base_url = (
            os.getenv("OLLAMA_BASE_URL")
            or os.getenv("OLLAMA_API_BASE_URL")
            or "http://127.0.0.1:11434"
        ).rstrip("/")
        self._active_models: Dict[str, Optional[str]] = {"ollama": None, "mlx": None, "trt": None}

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
        result = {
            "ok": proc.returncode == 0,
            "rc": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "command": command,
        }
        if not result["ok"]:
            logger.error("backend_command_failed", extra={"error": result["stderr"] or result["stdout"], "command": command})
        return result

    @staticmethod
    def trt_engine_id(model_id: str) -> str:
        return model_id.replace("/", "__")

    @staticmethod
    def _normalize_backend(backend: str) -> str:
        normalized = (backend or "").strip().lower()
        if normalized == "trtllm":
            return "trt"
        return normalized

    def _is_mlx_running(self) -> bool:
        if not self.mlx_pidfile.exists():
            return False
        try:
            pid_raw = self.mlx_pidfile.read_text(encoding="utf-8").strip()
            if not pid_raw:
                return False
            pid = int(pid_raw)
            os.kill(pid, 0)
            return True
        except Exception:
            return False

    def _is_trt_running(self) -> bool:
        result = self._run(
            f"docker ps --filter name=^{self.TRT_CONTAINER_NAME}$ --filter status=running --format '{{{{.Names}}}}'"
        )
        if not result.get("ok", False):
            return False
        names = [line.strip() for line in (result.get("stdout") or "").splitlines() if line.strip()]
        return self.TRT_CONTAINER_NAME in names

    def start_backend(self, backend: str, model_id: Optional[str] = None) -> Dict[str, Any]:
        backend = self._normalize_backend(backend)
        if backend == "mlx":
            try:
                model_arg = model_id or ""
                return self._run(f"./agent/backends/mlx_run.sh start {model_arg}".strip())
            except Exception as exc:
                logger.exception("mlx_engine_load_failed", extra={"backend": "mlx", "model_id": model_id, "error": str(exc)})
                raise RuntimeError(f"MLX load failed: {exc}") from exc
        if backend == "trt":
            if not model_id:
                return {"ok": False, "rc": 2, "stdout": "", "stderr": "model_id required for trt"}
            engine_id = self.trt_engine_id(model_id)
            engine_dir = self.repo_root / "agent" / "models" / "trtllm" / engine_id
            if not engine_dir.is_dir():
                msg = f"Model directory not found: {engine_dir}"
                logger.error("trt_engine_load_failed", extra={"backend": "trt", "model_id": model_id, "error": msg})
                return {"ok": False, "rc": 2, "stdout": "", "stderr": msg}
            try:
                return self._run(f"./agent/backends/trtllm_run.sh start {engine_id}")
            except Exception as exc:
                logger.exception("trt_engine_load_failed", extra={"backend": "trt", "model_id": model_id, "error": str(exc)})
                raise RuntimeError(f"TRT load failed: {exc}") from exc
        if backend == "ollama":
            return {"ok": True, "rc": 0, "stdout": "", "stderr": "", "command": "noop: ollama daemon externally managed"}
        return {"ok": False, "rc": 2, "stdout": "", "stderr": f"unknown backend: {backend}"}

    def stop_backend(self, backend: str) -> Dict[str, Any]:
        backend = self._normalize_backend(backend)
        if backend == "mlx":
            return self._run("./agent/backends/mlx_run.sh stop")
        if backend == "trt":
            return self._run("./agent/backends/trtllm_run.sh stop")
        if backend == "ollama":
            return {"ok": True, "rc": 0, "stdout": "", "stderr": "", "command": "noop: ollama daemon externally managed"}
        return {"ok": False, "rc": 2, "stdout": "", "stderr": f"unknown backend: {backend}"}

    async def _load_ollama_model(self, model_id: str) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{self.ollama_base_url}/api/generate",
                json={"model": model_id, "prompt": "", "stream": False},
            )
            resp.raise_for_status()
            return {"ok": True, "status_code": resp.status_code, "action": "load_model", "backend": "ollama", "model": model_id}

    async def _unload_ollama_model(self, model_id: str) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{self.ollama_base_url}/api/generate",
                json={"model": model_id, "prompt": "", "stream": False, "keep_alive": 0},
            )
            resp.raise_for_status()
            return {"ok": True, "status_code": resp.status_code, "action": "unload_model", "backend": "ollama", "model": model_id}

    async def ensure_backend_running(self, backend: str, model_id: str) -> Dict[str, Any]:
        backend = self._normalize_backend(backend)
        if backend == "ollama":
            if self._active_models.get("ollama") == model_id:
                return {"ok": True, "backend": "ollama", "action": "noop", "reason": "model_already_active"}
            result = await self._load_ollama_model(model_id)
            self._active_models["ollama"] = model_id
            return result

        if backend == "trt":
            current_model = self._active_models.get("trt")
            running = self._is_trt_running()
            if running and current_model == model_id:
                return {"ok": True, "backend": "trt", "action": "noop", "reason": "already_running"}
            started = self.start_backend("trt", model_id=model_id)
            if started.get("ok", False):
                self._active_models["trt"] = model_id
            return started

        if backend == "mlx":
            running = self._is_mlx_running()
            if not running:
                started = self.start_backend("mlx", model_id=model_id)
                if not started.get("ok", False):
                    return started
            self._active_models["mlx"] = model_id
            return {"ok": True, "backend": "mlx", "action": "started" if not running else "noop"}

        return {"ok": False, "rc": 2, "stdout": "", "stderr": f"unknown backend: {backend}"}

    async def ensure_backend_stopped(self, backend: str) -> Dict[str, Any]:
        backend = self._normalize_backend(backend)
        if backend == "ollama":
            model_id = self._active_models.get("ollama")
            if not model_id:
                return {"ok": True, "backend": "ollama", "action": "noop", "reason": "no_active_model"}
            result = await self._unload_ollama_model(model_id)
            self._active_models["ollama"] = None
            return result

        if backend == "trt":
            if not self._is_trt_running():
                self._active_models["trt"] = None
                return {"ok": True, "backend": "trt", "action": "noop", "reason": "already_stopped"}
            stopped = self.stop_backend("trt")
            if stopped.get("ok", False):
                self._active_models["trt"] = None
            return stopped

        if backend == "mlx":
            if not self._is_mlx_running():
                self._active_models["mlx"] = None
                return {"ok": True, "backend": "mlx", "action": "noop", "reason": "already_stopped"}
            stopped = self.stop_backend("mlx")
            if stopped.get("ok", False):
                self._active_models["mlx"] = None
            return stopped

        return {"ok": False, "rc": 2, "stdout": "", "stderr": f"unknown backend: {backend}"}

    def switch_backend(self, backend: str, model_id: str) -> Dict[str, Any]:
        return self.start_backend(backend=backend, model_id=model_id)
