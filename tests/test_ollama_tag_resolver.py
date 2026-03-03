from backends.ollama_backend import choose_best_tag, resolved_ollama_pull_name


def test_choose_best_tag_exact_match():
    available = {"llama3.1:8b-instruct", "llama3.1:8b-instruct-q4_K_M"}
    assert choose_best_tag("llama3.1:8b-instruct", available) == "llama3.1:8b-instruct"


def test_choose_best_tag_prefers_q4_candidate():
    available = {
        "llama3.1:8b-instruct-q8_0",
        "llama3.1:8b-instruct-q4_K_M",
        "llama3.1:8b-instruct-fp16",
    }
    assert choose_best_tag("llama3.1:8b-instruct", available) == "llama3.1:8b-instruct-q4_K_M"


def test_choose_best_tag_without_q4_prefers_shortest_then_lexical():
    available = {
        "llama3.1:8b-instruct-q8_0",
        "llama3.1:8b-instruct-fp16",
        "llama3.1:8b-instruct-awq",
    }
    assert choose_best_tag("llama3.1:8b-instruct", available) == "llama3.1:8b-instruct-awq"


def test_choose_best_tag_no_candidates_returns_none():
    available = {"mistral:7b-instruct-q4_K_M"}
    assert choose_best_tag("llama3.1:8b-instruct", available) is None


def test_resolved_ollama_pull_name_prefers_ollama_tag_field_when_installed():
    model_entry = {
        "id": "llama3.1:8b-instruct",
        "ollama_tag": "llama3.1:8b-instruct-q4_K_M",
    }
    installed = {"llama3.1:8b-instruct-q4_K_M"}
    assert resolved_ollama_pull_name(model_entry, installed) == "llama3.1:8b-instruct-q4_K_M"


def test_resolved_ollama_pull_name_resolves_from_id_prefix():
    model_entry = {"id": "llama3.1:8b-instruct"}
    installed = {"llama3.1:8b-instruct-q4_K_M", "llama3.1:8b-instruct-q8_0"}
    assert resolved_ollama_pull_name(model_entry, installed) == "llama3.1:8b-instruct-q4_K_M"


def test_resolved_ollama_pull_name_returns_none_when_no_match():
    model_entry = {"id": "llama3.1:8b-instruct"}
    installed = {"mistral:7b-instruct-q4_K_M"}
    assert resolved_ollama_pull_name(model_entry, installed) is None
