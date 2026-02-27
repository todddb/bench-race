from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

CENTRAL_DIR = Path(__file__).resolve().parent
MODELS_MAP_PATH = CENTRAL_DIR / "config" / "models_map.json"
MODELS_MANIFEST_PATH = CENTRAL_DIR / "models" / "models.json"


def load_models_map(path: Path | None = None) -> List[Dict[str, Any]]:
    """Load model mapping entries from JSON config.

    Returns an empty list if config is missing or malformed.
    """
    target = path or MODELS_MAP_PATH
    try:
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        return []
    models = data.get("models", []) if isinstance(data, dict) else []
    return [m for m in models if isinstance(m, dict)]


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
