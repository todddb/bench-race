from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
MODELS_REGISTRY_PATH = CONFIG_DIR / "registry" / "models.json"
MACHINES_PATH = CONFIG_DIR / "machines.yaml"
BACKENDS_PATH = CONFIG_DIR / "backends.yaml"
POLICY_PATH = CONFIG_DIR / "policy.yaml"
CENTRAL_DIR = Path(__file__).resolve().parent
MODELS_MANIFEST_PATH = CENTRAL_DIR / "models" / "models.json"


def _load_yaml_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError:
        logger.warning("Malformed YAML at %s; using empty config.", path)
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def load_models_registry(path: Path | None = None) -> Dict[str, Any]:
    """Load canonical model registry from ``config/registry/models.json``.

    Returns an empty registry when config is missing or malformed.
    """
    target = path or MODELS_REGISTRY_PATH
    default_registry: Dict[str, Any] = {
        "version": 0,
        "ollama": [],
        "custom": [],
        "comfyui": [],
    }

    if not target.exists():
        logger.warning("models.json not found at %s; using empty registry.", target)
        return default_registry

    try:
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        logger.warning("models.json is malformed at %s; using empty registry.", target)
        return default_registry

    if not isinstance(data, dict):
        return default_registry

    ollama = data.get("ollama")
    custom = data.get("custom")
    comfyui = data.get("comfyui")
    version = data.get("version", 0)

    return {
        "version": version if isinstance(version, int) else 0,
        "ollama": ollama if isinstance(ollama, list) else [],
        "custom": custom if isinstance(custom, list) else [],
        "comfyui": comfyui if isinstance(comfyui, list) else [],
    }


def load_machines_config(path: Path | None = None) -> Dict[str, Any]:
    return _load_yaml_file(path or MACHINES_PATH)


def load_backends_config(path: Path | None = None) -> Dict[str, Any]:
    return _load_yaml_file(path or BACKENDS_PATH)


def load_policy_config(path: Path | None = None) -> Dict[str, Any]:
    return _load_yaml_file(path or POLICY_PATH)


def load_models_manifest(path: Path | None = None) -> Dict[str, Any]:
    """Load canonical models manifest from JSON config.

    Returns a dict containing a ``models`` list.
    """
    target = path or MODELS_MANIFEST_PATH
    try:
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"models": []}

    if not isinstance(data, dict):
        return {"models": []}

    models = data.get("models", [])
    if not isinstance(models, list):
        models = []
    return {"models": [m for m in models if isinstance(m, dict)]}
