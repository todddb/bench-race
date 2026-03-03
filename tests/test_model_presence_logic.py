import importlib
import sys
from pathlib import Path


CFG_PATH = Path("config/machines.yaml")


def _load_central_app():
    CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
    original = CFG_PATH.read_text(encoding="utf-8") if CFG_PATH.exists() else None
    try:
        CFG_PATH.write_text(
            "machines:\n"
            "  - machine_id: test1\n"
            "    label: Test\n"
            "    agent_base_url: http://127.0.0.1:9001\n",
            encoding="utf-8",
        )
        sys.modules.pop("central.app", None)
        return importlib.import_module("central.app")
    finally:
        if original is None:
            CFG_PATH.unlink(missing_ok=True)
        else:
            CFG_PATH.write_text(original, encoding="utf-8")


def test_model_satisfied_exact_match():
    central_app = _load_central_app()
    installed = ["llama3.1:8b-instruct"]
    assert central_app.model_satisfied("llama3.1:8b-instruct", installed)


def test_model_satisfied_quantized_variant():
    central_app = _load_central_app()
    installed = ["llama3.1:8b-instruct-q4_K_M"]
    assert central_app.model_satisfied("llama3.1:8b-instruct", installed)


def test_model_not_satisfied():
    central_app = _load_central_app()
    installed = ["llama3.1:70b-instruct-q4_K_M"]
    assert not central_app.model_satisfied("llama3.1:8b-instruct", installed)
