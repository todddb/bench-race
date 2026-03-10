# IMPORTANT ARCHITECTURAL RULE:
# Central is the sole authority for model resolution.
# Agents must never resolve abstract IDs or map architectures.
# Agents execute only the fully resolved model string provided by Central.
#
# This module retains registry-loading helpers for validation and tests only.
# No agent runtime code should call resolve_model_for_machine() or
# get_machine_architecture() for model resolution purposes.  Central
# resolves all models and sends agents the concrete execution string.
from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def get_all_registry_entries() -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    for model_id, entry in _load_registry().items():
        if not isinstance(entry, dict):
            continue
        combined_apple = str(((entry.get("ollama") or {}).get("apple") or "")).strip()
        combined_nvidia = str(((entry.get("ollama") or {}).get("nvidia") or "")).strip()
        if not combined_apple and not combined_nvidia:
            combined_apple = str(((entry.get("custom") or {}).get("apple") or "")).strip()
            combined_nvidia = str(((entry.get("custom") or {}).get("nvidia") or "")).strip()
        entries.append({"id": model_id, "apple": combined_apple, "nvidia": combined_nvidia})
    return entries


def resolve_standard_id_from_runtime(runtime_id: str) -> Optional[str]:
    runtime_id = (runtime_id or "").strip()
    if not runtime_id:
        return None

    for entry in get_all_registry_entries():
        for field in ("apple", "nvidia"):
            if str(entry.get(field) or "").strip() == runtime_id:
                return str(entry.get("id") or "").strip() or None
    return None


def get_machine_architecture() -> str:
    """Return the local machine architecture label ('apple' or 'nvidia').

    NOTE: This is used only for local hardware detection (e.g. test
    validation).  Model resolution must be performed by Central using
    machines.yaml, NOT by agents inspecting their own hardware.
    """
    env = (os.getenv("BENCH_AGENT_PLATFORM") or "").strip().lower()
    if env in {"apple", "mac", "darwin"}:
        return "apple"
    if env in {"nvidia", "linux"}:
        return "nvidia"
    return "apple" if platform.system().lower() == "darwin" else "nvidia"


# ---------------------------------------------------------------------------
# DEPRECATED — Agent must NOT use these for runtime model resolution.
# Central is the sole authority.  These are retained only so that existing
# tests that monkeypatch them continue to import without error.
# ---------------------------------------------------------------------------

def resolve_model_for_machine(model_id: str, backend: str):
    """DEPRECATED: Agent must not resolve models.  Central resolves all models.

    Retained for backward-compatible test imports only.
    """
    entry = get_registry_entry(model_id)

    if not entry:
        return None

    arch = get_machine_architecture()

    if backend == "ollama":
        return (entry.get("ollama") or {}).get(arch)

    if backend == "custom":
        return (entry.get("custom") or {}).get(arch)

    return None


def registry_entry_matches_backend(model_id: str, backend: str) -> bool:
    """DEPRECATED: Agent must not resolve models.  Central resolves all models.

    Retained for backward-compatible test imports only.
    """

    backend = (backend or "").strip().lower()
    if backend not in {"custom", "ollama"}:
        return False

    from agent.backends.base import BackendType

    try:
        from agent.agent_app import backend_manager as active_backend_manager
    except Exception:
        return False

    active_backend = active_backend_manager.get_active_backend()

    if active_backend.backend_type == BackendType.MANAGED:
        entry = get_registry_entry(model_id)
        return entry is not None

    if active_backend.backend_type == BackendType.EXTERNAL:
        machine_arch = get_machine_architecture()
        for entry in get_all_registry_entries():
            if str(entry.get(machine_arch) or "").strip() == (model_id or "").strip():
                return True
        return False

    return False
