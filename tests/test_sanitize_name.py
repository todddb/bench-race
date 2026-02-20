"""Unit tests for the shared sanitize_model_name helper."""
import sys
from pathlib import Path

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.sanitize_name import sanitize_model_name


def test_colon_replaced():
    assert sanitize_model_name("kimi2.5:cloud") == "kimi2.5_cloud"


def test_colon_in_instruct_tag():
    assert sanitize_model_name("llama3.1:8b-instruct") == "llama3.1_8b-instruct"


def test_slash_replaced():
    assert sanitize_model_name("user/name-with spaces") == "user_name-with_spaces"


def test_hf_style_id():
    assert sanitize_model_name("meta-llama/Llama-2-8b-chat-hf") == "meta-llama_Llama-2-8b-chat-hf"


def test_already_clean():
    assert sanitize_model_name("my_model.v2") == "my_model.v2"


def test_multiple_special_chars():
    assert sanitize_model_name("a@b#c$d") == "a_b_c_d"


def test_leading_trailing_underscores_stripped():
    assert sanitize_model_name("///model///") == "model"


def test_consecutive_underscores_collapsed():
    assert sanitize_model_name("a::b") == "a_b"


def test_truncation():
    long_name = "a" * 200
    result = sanitize_model_name(long_name)
    assert len(result) == 90


def test_empty_string():
    assert sanitize_model_name("") == ""


def test_only_special_chars():
    assert sanitize_model_name(":::") == ""


def test_quantization_suffix():
    assert sanitize_model_name("llama3.1:70b-instruct-q4_K_M") == "llama3.1_70b-instruct-q4_K_M"


def test_deterministic():
    """Same input always produces same output."""
    name = "kimi2.5:cloud"
    assert sanitize_model_name(name) == sanitize_model_name(name)
