"""
Unified Model Registry for bench-race.

Single source of truth mapping logical model IDs to per-backend artifacts.
Supports Ollama and vLLM backends with artifact metadata, availability flags,
quantization info, checksums, and license metadata.

The registry can be loaded from a YAML/JSON file or populated programmatically.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

log = logging.getLogger("model_registry")


class BackendArtifact(BaseModel):
    """Metadata for a model artifact on a specific backend."""
    backend: str  # "ollama" or "vllm"
    artifact_id: str  # Backend-specific model name/path
    quantization: Optional[str] = None  # e.g., "q8_0", "q4_K_M", "fp16", "awq"
    format: Optional[str] = None  # e.g., "gguf", "safetensors", "gptq"
    size_bytes: Optional[int] = None
    checksum: Optional[str] = None  # "sha256:..."
    source_url: Optional[str] = None  # HuggingFace / registry URL
    provisioning_status: Optional[str] = None  # "available", "downloading", "missing", "error"
    license: Optional[str] = None  # e.g., "llama3.1", "apache-2.0"
    notes: Optional[str] = None


class ModelEntry(BaseModel):
    """A logical model with per-backend artifact mappings."""
    model_id: str  # Logical identifier (e.g., "meta-llama-3.1-8b-instruct")
    display_name: Optional[str] = None  # Human-readable name
    family: Optional[str] = None  # e.g., "llama", "mistral", "qwen"
    parameter_count: Optional[str] = None  # e.g., "8B", "70B"
    license: Optional[str] = None
    artifacts: List[BackendArtifact] = Field(default_factory=list)

    def get_artifact(self, backend: str) -> Optional[BackendArtifact]:
        """Return the artifact for the given backend, or None."""
        for a in self.artifacts:
            if a.backend == backend:
                return a
        return None

    def available_backends(self) -> List[str]:
        """Return list of backends that have artifacts defined."""
        return [a.backend for a in self.artifacts]


class ModelRegistry(BaseModel):
    """
    Central model registry: maps logical model IDs to backend artifacts.
    """
    version: str = "1"
    models: List[ModelEntry] = Field(default_factory=list)

    def get_model(self, model_id: str) -> Optional[ModelEntry]:
        """Lookup a model by its logical ID."""
        for m in self.models:
            if m.model_id == model_id:
                return m
        return None

    def list_models(self, backend: Optional[str] = None) -> List[ModelEntry]:
        """List all models, optionally filtered by backend availability."""
        if backend is None:
            return list(self.models)
        return [m for m in self.models if any(a.backend == backend for a in m.artifacts)]

    def list_model_ids(self, backend: Optional[str] = None) -> List[str]:
        """List all model IDs, optionally filtered by backend."""
        return [m.model_id for m in self.list_models(backend)]

    def add_model(self, entry: ModelEntry) -> None:
        """Add or update a model entry (by model_id)."""
        existing = self.get_model(entry.model_id)
        if existing:
            self.models = [m for m in self.models if m.model_id != entry.model_id]
        self.models.append(entry)

    def to_api_response(self) -> List[Dict[str, Any]]:
        """Serialize registry for API responses."""
        result = []
        for m in self.models:
            entry = {
                "model_id": m.model_id,
                "display_name": m.display_name or m.model_id,
                "family": m.family,
                "parameter_count": m.parameter_count,
                "license": m.license,
                "backends": {},
            }
            for a in m.artifacts:
                entry["backends"][a.backend] = {
                    "artifact_id": a.artifact_id,
                    "quantization": a.quantization,
                    "format": a.format,
                    "provisioning_status": a.provisioning_status,
                }
            result.append(entry)
        return result


def build_registry_from_policy(
    policy: Dict[str, Any],
    ollama_models: Optional[List[str]] = None,
    vllm_models: Optional[List[str]] = None,
) -> ModelRegistry:
    """
    Build a model registry from the existing model_policy.yaml format,
    augmented with live availability from backend model lists.

    This bridges the current model_policy.yaml format with the new
    unified registry without breaking existing workflows.
    """
    registry = ModelRegistry()

    required = policy.get("required", {})
    llm_models = required.get("llm", [])

    ollama_available = set(ollama_models or [])
    vllm_available = set(vllm_models or [])

    for model_name in llm_models:
        # Parse model name to extract metadata
        # Format: "llama3.1:8b-instruct-q8_0"
        parts = model_name.split(":")
        family = parts[0] if parts else model_name
        variant = parts[1] if len(parts) > 1 else ""

        # Extract parameter count and quantization from variant
        param_count = None
        quantization = None
        if variant:
            for token in variant.split("-"):
                if token.endswith("b") and token[:-1].replace(".", "").isdigit():
                    param_count = token.upper()
                elif token.startswith("q") and "_" in token:
                    quantization = token
                elif token.startswith("q") and token[1:].replace("_", "").replace(".", "").isdigit():
                    quantization = token

        artifacts = []

        # Ollama artifact
        ollama_status = "available" if model_name in ollama_available else "missing"
        artifacts.append(BackendArtifact(
            backend="ollama",
            artifact_id=model_name,
            quantization=quantization,
            format="gguf",
            provisioning_status=ollama_status,
        ))

        # vLLM artifact (if the model or a matching ID is available)
        for vm in vllm_available:
            if _models_match(model_name, vm):
                artifacts.append(BackendArtifact(
                    backend="vllm",
                    artifact_id=vm,
                    format="safetensors",
                    provisioning_status="available",
                ))
                break

        # Generate a logical model ID
        model_id = model_name.replace(":", "-").replace("_", "-")

        entry = ModelEntry(
            model_id=model_id,
            display_name=model_name,
            family=family,
            parameter_count=param_count,
            artifacts=artifacts,
        )
        registry.add_model(entry)

    return registry


def _models_match(ollama_name: str, vllm_name: str) -> bool:
    """
    Heuristic check whether an Ollama model name and a vLLM model ID
    refer to the same underlying model. This is intentionally loose
    since naming conventions differ between backends.
    """
    # Normalize: lowercase, strip common prefixes/suffixes
    a = ollama_name.lower().replace(":", "-").replace("_", "-")
    b = vllm_name.lower().replace("/", "-").replace("_", "-")

    # Extract key tokens
    a_tokens = set(a.split("-"))
    b_tokens = set(b.split("-"))

    # Require at least family + size overlap
    overlap = a_tokens & b_tokens
    return len(overlap) >= 2


def load_registry_from_file(path: str) -> ModelRegistry:
    """Load a model registry from a JSON file."""
    p = Path(path)
    if not p.exists():
        log.warning("Registry file not found: %s, returning empty registry", path)
        return ModelRegistry()
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ModelRegistry.model_validate(data)


def save_registry_to_file(registry: ModelRegistry, path: str) -> None:
    """Save a model registry to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(registry.model_dump(), f, indent=2)
