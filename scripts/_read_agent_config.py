#!/usr/bin/env python3
"""
Helper script to read agent/config/agent.yaml and output JSON for shell scripts.

Usage:
    python3 scripts/_read_agent_config.py [config_path]

Output: JSON object with:
    {
        "ollama_base_url": "http://127.0.0.1:11434",
        "llm_models": ["model1", "model2"],
        "whisper_models": ["large-v3"],
        "sdxl_profiles": ["profile1"]
    }

If config file doesn't exist or parsing fails, exits with non-zero status and error message.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def find_config_path() -> Path:
    """Find agent config file path."""
    # Check environment variable first
    env_path = os.environ.get("AGENT_CONFIG")
    if env_path:
        return Path(env_path)

    # Check command-line argument
    if len(sys.argv) > 1:
        return Path(sys.argv[1])

    # Default: relative to this script's location
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    return repo_root / "agent" / "config" / "agent.yaml"


def main() -> int:
    config_path = find_config_path()

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        return 1

    try:
        import yaml
    except ImportError:
        # If PyYAML is not available, try a simple parser
        print("Error: PyYAML not installed. Install with: pip install pyyaml", file=sys.stderr)
        return 1

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Error: Failed to parse YAML: {e}", file=sys.stderr)
        return 1

    output = {
        "ollama_base_url": config.get("ollama_base_url", "http://127.0.0.1:11434"),
        "llm_models": config.get("llm_models", []),
        "whisper_models": config.get("whisper_models", []),
        "sdxl_profiles": config.get("sdxl_profiles", []),
    }

    print(json.dumps(output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
