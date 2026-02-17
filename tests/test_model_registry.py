"""Tests for shared.model_registry — _models_match heuristic and manual overrides."""
from __future__ import annotations

import pytest
from shared.model_registry import (
    _models_match,
    build_registry_from_policy,
    ModelRegistry,
)


# ---------------------------------------------------------------------------
# _models_match: positive cases (should return True)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "ollama_name, vllm_name",
    [
        # Standard Llama 3.1 naming
        ("llama3.1:8b-instruct-q8_0", "meta-llama/Llama-3.1-8B-Instruct"),
        # Mistral variants
        ("mistral:7b-instruct", "mistralai/Mistral-7B-Instruct-v0.3"),
        # Qwen
        ("qwen2:7b-instruct", "Qwen/Qwen2-7B-Instruct"),
    ],
    ids=["llama3.1-8b", "mistral-7b", "qwen2-7b"],
)
def test_models_match_positive(ollama_name: str, vllm_name: str):
    assert _models_match(ollama_name, vllm_name) is True


# ---------------------------------------------------------------------------
# _models_match: negative cases (should return False)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "ollama_name, vllm_name",
    [
        # Different families
        ("llama3.1:8b-instruct-q8_0", "mistralai/Mistral-7B-Instruct-v0.3"),
        # Different sizes
        ("llama3.1:8b-instruct", "meta-llama/Llama-3.1-70B-Instruct"),
        # Completely unrelated
        ("phi3:mini", "meta-llama/Llama-3.1-8B-Instruct"),
    ],
    ids=["different-family", "different-size", "unrelated"],
)
def test_models_match_negative(ollama_name: str, vllm_name: str):
    assert _models_match(ollama_name, vllm_name) is False


# ---------------------------------------------------------------------------
# _models_match: edge cases
# ---------------------------------------------------------------------------
def test_models_match_empty_strings():
    assert _models_match("", "") is False


def test_models_match_single_token():
    # Single token overlap is insufficient (need >= 2)
    assert _models_match("llama", "llama") is False


def test_models_match_exact_same():
    # Same name in both backends — should match if it has 2+ tokens
    assert _models_match("llama:8b", "llama-8b") is True


# ---------------------------------------------------------------------------
# build_registry_from_policy: manual override
# ---------------------------------------------------------------------------
def test_build_registry_manual_override():
    policy = {"required": {"llm": ["custom-model:7b"]}}
    vllm_models = ["org/CustomModel-7B"]

    # Without override, heuristic may not match (depends on token overlap)
    # With override, it should always match
    registry = build_registry_from_policy(
        policy,
        ollama_models=["custom-model:7b"],
        vllm_models=vllm_models,
        artifact_overrides={"custom-model:7b": "org/CustomModel-7B"},
    )

    entry = registry.get_model("custom-model-7b")
    assert entry is not None
    backends = entry.available_backends()
    assert "ollama" in backends
    assert "vllm" in backends
    vllm_artifact = entry.get_artifact("vllm")
    assert vllm_artifact is not None
    assert vllm_artifact.artifact_id == "org/CustomModel-7B"


def test_build_registry_override_not_in_available():
    """Override pointing to a model not in vllm_models should not create artifact."""
    policy = {"required": {"llm": ["custom-model:7b"]}}
    registry = build_registry_from_policy(
        policy,
        ollama_models=["custom-model:7b"],
        vllm_models=[],  # vLLM has no models
        artifact_overrides={"custom-model:7b": "org/CustomModel-7B"},
    )

    entry = registry.get_model("custom-model-7b")
    assert entry is not None
    assert "vllm" not in entry.available_backends()


def test_build_registry_heuristic_match():
    """Standard heuristic matching without overrides."""
    policy = {"required": {"llm": ["llama3.1:8b-instruct-q8_0"]}}
    registry = build_registry_from_policy(
        policy,
        ollama_models=["llama3.1:8b-instruct-q8_0"],
        vllm_models=["meta-llama/Llama-3.1-8B-Instruct"],
    )

    entry = registry.get_model("llama3.1-8b-instruct-q8-0")
    assert entry is not None
    assert "vllm" in entry.available_backends()


def test_build_registry_no_vllm():
    """No vLLM models available — only Ollama artifacts created."""
    policy = {"required": {"llm": ["llama3.1:8b-instruct-q8_0"]}}
    registry = build_registry_from_policy(
        policy,
        ollama_models=["llama3.1:8b-instruct-q8_0"],
        vllm_models=[],
    )

    entry = registry.get_model("llama3.1-8b-instruct-q8-0")
    assert entry is not None
    assert "ollama" in entry.available_backends()
    assert "vllm" not in entry.available_backends()
