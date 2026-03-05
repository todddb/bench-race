from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_CANDIDATE_PATHS = (
    ROOT_DIR / "config" / "registry" / "models.json",
    ROOT_DIR / "central" / "config" / "registry" / "models.json",
    ROOT_DIR / "central" / "models" / "models.json",
    ROOT_DIR / "agent" / "models" / "metadata.yaml",
)


def _registry_path() -> Optional[Path]:
    override = (os.getenv("MODEL_REGISTRY_PATH") or "").strip()
    if override:
        path = Path(override)
        return path if path.exists() else None
    for candidate in _DEFAULT_CANDIDATE_PATHS:
        if candidate.exists():
            return candidate
    return None


def _load_registry() -> Dict[str, Dict[str, Any]]:
    path = _registry_path()
    if not path:
        return {}

    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return _normalize_json_registry(payload)

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return _normalize_yaml_registry(payload)


def _normalize_json_registry(payload: Any) -> Dict[str, Dict[str, Any]]:
    registry: Dict[str, Dict[str, Any]] = {}
    if not isinstance(payload, dict):
        return registry

    # Canonical central registry format: {"custom": [...], "ollama": [...]}.
    ollama = payload.get("ollama")
    if isinstance(ollama, list):
        for entry in ollama:
            if not isinstance(entry, dict):
                continue
            model_id = str(entry.get("id") or "").strip()
            if not model_id:
                continue
            registry.setdefault(model_id, {})
            registry[model_id]["ollama"] = {
                "apple": str(entry.get("apple") or "").strip(),
                "nvidia": str(entry.get("nvidia") or "").strip(),
            }

    # Canonical central registry format: {"custom": [{"id": ..., "apple": ..., "nvidia": ...}]}
    custom = payload.get("custom")
    if isinstance(custom, list):
        for entry in custom:
            if not isinstance(entry, dict):
                continue
            model_id = str(entry.get("id") or "").strip()
            if not model_id:
                continue
            registry.setdefault(model_id, {})
            registry[model_id]["custom"] = {
                "apple": str(entry.get("apple") or "").strip(),
                "nvidia": str(entry.get("nvidia") or "").strip(),
            }

    # Legacy manifest format: {"models": [{"id": ..., "custom": {"MLX":..., "TensorRT-LLM":...}}]}
    models = payload.get("models")
    if isinstance(models, list):
        for entry in models:
            if not isinstance(entry, dict):
                continue
            model_id = str(entry.get("id") or "").strip()
            custom_cfg = entry.get("custom")
            if not model_id or not isinstance(custom_cfg, dict):
                continue
            mlx = custom_cfg.get("MLX") if isinstance(custom_cfg.get("MLX"), dict) else {}
            trt = custom_cfg.get("TensorRT-LLM") if isinstance(custom_cfg.get("TensorRT-LLM"), dict) else {}
            registry.setdefault(model_id, {})
            registry[model_id].setdefault("custom", {
                "apple": str((mlx or {}).get("engine_model_name") or "").strip(),
                "nvidia": str((trt or {}).get("engine_model_name") or "").strip(),
            })

    return registry


def _normalize_yaml_registry(payload: Any) -> Dict[str, Dict[str, Any]]:
    registry: Dict[str, Dict[str, Any]] = {}
    if not isinstance(payload, dict):
        return registry
    models = payload.get("models")
    if not isinstance(models, list):
        return registry
    for entry in models:
        if not isinstance(entry, dict):
            continue
        model_id = str(entry.get("id") or entry.get("model_id") or "").strip()
        if not model_id:
            continue
        registry[model_id] = {
            "custom": {
                "apple": str(entry.get("apple") or "").strip(),
                "nvidia": str(entry.get("nvidia") or "").strip(),
            }
        }
    return registry


def get_registry_entry(model_id: str) -> Optional[Dict[str, Any]]:
    return _load_registry().get((model_id or "").strip())


def get_machine_architecture() -> str:
    env = (os.getenv("BENCH_AGENT_PLATFORM") or "").strip().lower()
    if env in {"apple", "mac", "darwin"}:
        return "apple"
    if env in {"nvidia", "linux"}:
        return "nvidia"
    return "apple" if platform.system().lower() == "darwin" else "nvidia"


def resolve_model_for_machine(model_id: str, backend: str):
    entry = get_registry_entry(model_id)

    if not entry:
        return None

    arch = get_machine_architecture()  # returns "apple" or "nvidia"

    if backend == "ollama":
        return (entry.get("ollama") or {}).get(arch)

    if backend == "custom":
        return (entry.get("custom") or {}).get(arch)

    return None


def registry_entry_matches_backend(model_id: str, backend: str) -> bool:
    entry = get_registry_entry(model_id)
    if not entry:
        return False
    return isinstance(entry.get((backend or "").strip().lower()), dict)
