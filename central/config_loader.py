from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_REGISTRY_PATH = BASE_DIR.parent / "config" / "models.json"
CENTRAL_DIR = Path(__file__).resolve().parent
MODELS_MANIFEST_PATH = CENTRAL_DIR / "models" / "models.json"


def load_models_registry(path: Path | None = None) -> Dict[str, Any]:
    """Load canonical model registry from ``config/models.json``.

    Returns an empty registry when config is missing or malformed.
    """
    target = path or MODELS_REGISTRY_PATH
    default_registry: Dict[str, Any] = {
        "version": 0,
        "shared_baseline": [],
        "architectures": {},
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

    shared_baseline = data.get("shared_baseline")
    architectures = data.get("architectures")
    version = data.get("version", 0)

    return {
        "version": version if isinstance(version, int) else 0,
        "shared_baseline": shared_baseline if isinstance(shared_baseline, list) else [],
        "architectures": architectures if isinstance(architectures, dict) else {},
    }


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
