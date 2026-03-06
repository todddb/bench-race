"""Tests verifying the EXTERNAL vs MANAGED backend separation.

After the refactor:
- Ollama (EXTERNAL) must never have lifecycle managed by the agent.
- MLX/TRT-LLM (MANAGED) must have full lifecycle enforcement.
- The wrapper exposes Ollama-compatible API endpoints on a separate port.
"""
import asyncio
import importlib
from pathlib import Path


def _ensure_agent_config():
    cfg = Path("agent/config/agent.yaml")
    if cfg.exists():
        return
    example = Path("agent/config/agent.yaml.example")
    if example.exists():
        cfg.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text("machine_id: test-machine\nlabel: Test\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# BackendType enum tests
# ---------------------------------------------------------------------------


def test_backend_type_values():
    from agent.backends.base import BackendType

    assert BackendType.EXTERNAL == "external"
    assert BackendType.MANAGED == "managed"
    assert BackendType.EXTERNAL != BackendType.MANAGED


def test_ollama_is_external():
    from agent.backends.ollama_wrapper import OllamaBackendWrapper

    backend = OllamaBackendWrapper()
    assert backend.backend_type.value == "external"


def test_mlx_is_managed():
    from agent.backends.mlx_wrapper import MLXBackendWrapper

    backend = MLXBackendWrapper()
    assert backend.backend_type.value == "managed"


def test_trtllm_is_managed():
    from agent.backends.trtllm_wrapper import TRTLLMBackendWrapper

    backend = TRTLLMBackendWrapper()
    assert backend.backend_type.value == "managed"


# ---------------------------------------------------------------------------
# External backend (Ollama) lifecycle isolation
# ---------------------------------------------------------------------------


def test_ollama_start_is_noop():
    """Ollama.start() must not invoke any subprocess or lifecycle action."""
    from agent.backends.ollama_wrapper import OllamaBackendWrapper

    backend = OllamaBackendWrapper()
    result = asyncio.run(backend.start("some-model"))
    assert result == {"ok": True, "selected_model": "some-model"}


def test_ollama_stop_is_noop():
    """Ollama.stop() must not invoke any subprocess or lifecycle action."""
    from agent.backends.ollama_wrapper import OllamaBackendWrapper

    backend = OllamaBackendWrapper()
    result = asyncio.run(backend.stop())
    assert result == {"ok": True}


def test_start_backend_engine_external_sets_running(monkeypatch):
    """start_backend_engine for EXTERNAL should mark engine as ready/running."""
    _ensure_agent_config()
    mod = importlib.import_module("agent.agent_app")

    # Reset state
    mod.agent_state.running = False
    mod.agent_state.current_model = None

    class _ExternalBackend:
        backend_type = mod.BackendType.EXTERNAL

        async def is_available(self):
            return True

        async def list_models(self):
            return ["test-model"]

        async def start(self, model):
            return {"ok": True, "selected_model": model}

        async def stop(self):
            return {"ok": True}

    class _Manager:
        def create_backend(self, name):
            return _ExternalBackend()

        def set_active_backend(self, backend, name):
            pass

        def get_active_backend(self):
            return _ExternalBackend()

        def get_active_backend_name(self):
            return "ollama"

    monkeypatch.setattr(mod, "backend_manager", _Manager())
    monkeypatch.setattr(mod, "broadcast_status", lambda msg: asyncio.sleep(0))

    result = asyncio.run(mod.start_backend_engine("ollama", "test-model"))

    assert result["status"] == "ok"
    assert mod.agent_state.current_model == "test-model"
    assert mod.agent_state.running is True


def test_stop_engine_external_does_not_run_script(monkeypatch):
    """stop_engine for EXTERNAL must not call _run_agent_script."""
    _ensure_agent_config()
    mod = importlib.import_module("agent.agent_app")

    mod._ACTIVE_BACKEND = "ollama"
    mod.agent_state.running = False
    mod.agent_state.current_model = "model-a"

    class _Backend:
        backend_type = mod.BackendType.EXTERNAL

    class _Manager:
        def get_active_backend_name(self):
            return "ollama"

        def get_active_backend(self):
            return _Backend()

        def create_backend(self, _name):
            return _Backend()

        def clear_active_backend(self):
            return None

    monkeypatch.setattr(mod, "backend_manager", _Manager())

    async def _boom(*_args, **_kwargs):
        raise AssertionError("stop script should not run for external backend")

    monkeypatch.setattr(mod, "_run_agent_script", _boom)

    response = asyncio.run(mod.stop_engine(mod.EngineStopRequest(engine="ollama")))
    assert response["status"] == "ok"


# ---------------------------------------------------------------------------
# Managed backend lifecycle enforcement
# ---------------------------------------------------------------------------


def test_jobs_reject_managed_not_running(monkeypatch):
    """POST /jobs must reject requests when MANAGED backend not running."""
    _ensure_agent_config()
    mod = importlib.import_module("agent.agent_app")

    mod._ACTIVE_BACKEND = "mlx"
    mod.agent_state.running = False
    mod.agent_state.current_model = "model-a"

    class _ManagedBackend:
        backend_type = mod.BackendType.MANAGED

    class _Manager:
        def get_active_backend_name(self):
            return "mlx"

        def get_active_backend(self):
            return _ManagedBackend()

    monkeypatch.setattr(mod, "backend_manager", _Manager())
    monkeypatch.setattr(mod, "registry_entry_matches_backend", lambda *_a, **_k: True)

    req = mod.LLMRequest(model="model-a", prompt="hello")
    try:
        asyncio.run(mod.start_job(req))
        assert False, "Expected HTTPException"
    except mod.HTTPException as exc:
        assert exc.status_code == 400
        assert exc.detail == "Engine not started"


def test_jobs_allow_external_without_running(monkeypatch):
    """POST /jobs must allow EXTERNAL backend even when agent_state.running is False."""
    _ensure_agent_config()
    mod = importlib.import_module("agent.agent_app")

    mod._ACTIVE_BACKEND = "ollama"
    mod.agent_state.running = False
    mod.agent_state.current_model = "model-a"

    class _ExternalBackend:
        backend_type = mod.BackendType.EXTERNAL

        async def is_available(self):
            return True

        async def list_models(self):
            return ["model-a"]

        async def generate(self, model, messages, stream):
            yield "hello"

    class _Manager:
        def get_active_backend_name(self):
            return "ollama"

        def get_active_backend(self):
            return _ExternalBackend()

    monkeypatch.setattr(mod, "backend_manager", _Manager())
    monkeypatch.setattr(mod, "registry_entry_matches_backend", lambda *_a, **_k: True)

    req = mod.LLMRequest(model="model-a", prompt="hello")
    # Should NOT raise — external backends don't need agent_state.running
    result = asyncio.run(mod.start_job(req))
    assert hasattr(result, "job_id")


def test_jobs_allow_external_without_agent_state_model(monkeypatch):
    """POST /jobs must allow EXTERNAL backend without agent_state.current_model."""
    _ensure_agent_config()
    mod = importlib.import_module("agent.agent_app")

    mod._ACTIVE_BACKEND = "ollama"
    mod.agent_state.running = False
    mod.agent_state.current_model = None

    class _ExternalBackend:
        backend_type = mod.BackendType.EXTERNAL

    class _Manager:
        def get_active_backend_name(self):
            return "ollama"

        def get_active_backend(self):
            return _ExternalBackend()

    monkeypatch.setattr(mod, "backend_manager", _Manager())
    monkeypatch.setattr(mod, "registry_entry_matches_backend", lambda *_a, **_k: True)

    req = mod.LLMRequest(model="model-a", prompt="hello")
    result = asyncio.run(mod.start_job(req))
    assert hasattr(result, "job_id")


def test_jobs_allow_external_without_availability_probe(monkeypatch):
    """POST /jobs must not gate EXTERNAL backend on is_available checks."""
    _ensure_agent_config()
    mod = importlib.import_module("agent.agent_app")

    mod._ACTIVE_BACKEND = "ollama"
    mod.agent_state.running = False
    mod.agent_state.current_model = "model-a"

    class _ExternalBackend:
        backend_type = mod.BackendType.EXTERNAL

    class _Manager:
        def get_active_backend_name(self):
            return "ollama"

        def get_active_backend(self):
            return _ExternalBackend()

    monkeypatch.setattr(mod, "backend_manager", _Manager())
    monkeypatch.setattr(mod, "registry_entry_matches_backend", lambda *_a, **_k: True)

    req = mod.LLMRequest(model="model-a", prompt="hello")
    result = asyncio.run(mod.start_job(req))
    assert hasattr(result, "job_id")


# ---------------------------------------------------------------------------
# Registry validation semantics by backend type
# ---------------------------------------------------------------------------


def test_external_runtime_id_allowed(monkeypatch):
    """EXTERNAL backends must accept runtime IDs present in registry mappings."""
    _ensure_agent_config()
    reg = importlib.import_module("agent.model_registry")
    mod = importlib.import_module("agent.agent_app")

    class _ExternalBackend:
        backend_type = mod.BackendType.EXTERNAL

    class _Manager:
        def get_active_backend(self):
            return _ExternalBackend()

    monkeypatch.setattr(mod, "backend_manager", _Manager())
    monkeypatch.setattr(reg, "get_machine_architecture", lambda: "apple")
    monkeypatch.setattr(
        reg,
        "get_all_registry_entries",
        lambda: [{"id": "llama3.1-8b-q4", "apple": "llama3.1:8b-instruct-q4_K_M", "nvidia": "llama3.1:8b-instruct-q4_K_M"}],
    )

    assert reg.registry_entry_matches_backend("llama3.1:8b-instruct-q4_K_M", "ollama") is True


def test_managed_rejects_runtime_id(monkeypatch):
    """MANAGED backends must accept only standardized IDs, not runtime IDs."""
    _ensure_agent_config()
    reg = importlib.import_module("agent.model_registry")
    mod = importlib.import_module("agent.agent_app")

    class _ManagedBackend:
        backend_type = mod.BackendType.MANAGED

    class _Manager:
        def get_active_backend(self):
            return _ManagedBackend()

    monkeypatch.setattr(mod, "backend_manager", _Manager())
    monkeypatch.setattr(reg, "get_registry_entry", lambda model_id: {"custom": {}} if model_id == "llama3.1-8b-q4" else None)

    assert reg.registry_entry_matches_backend("llama3.1:8b-instruct-q4_K_M", "custom") is False
    assert reg.registry_entry_matches_backend("llama3.1-8b-q4", "custom") is True


# ---------------------------------------------------------------------------
# Atomic switch: external backends are never stopped
# ---------------------------------------------------------------------------


def test_atomic_switch_does_not_stop_external_backend(monkeypatch):
    """_atomic_switch_backend must not call stop() on the previous EXTERNAL backend."""
    _ensure_agent_config()
    mod = importlib.import_module("agent.agent_app")

    stop_called = {"called": False}

    class _ExternalOld:
        backend_type = mod.BackendType.EXTERNAL

        async def stop(self):
            stop_called["called"] = True

    class _ManagedNew:
        backend_type = mod.BackendType.MANAGED

        async def start(self, model):
            return {"ok": True}

        async def stop(self):
            pass

    class _Manager:
        _active = _ExternalOld()
        _active_name = "ollama"

        def create_backend(self, name):
            if name == "mlx":
                return _ManagedNew()
            return _ExternalOld()

        def get_active_backend(self):
            return self._active

        def get_active_backend_name(self):
            return self._active_name

        def set_active_backend(self, backend, name):
            self._active = backend
            self._active_name = name

    monkeypatch.setattr(mod, "backend_manager", _Manager())

    asyncio.run(mod._atomic_switch_backend("mlx", "some-model"))

    assert not stop_called["called"], "stop() should not be called on external backend"


# ---------------------------------------------------------------------------
# Wrapper Ollama-compatible endpoints
# ---------------------------------------------------------------------------


def test_wrapper_has_ollama_api_tags():
    """Wrapper must expose GET /api/tags."""
    from agent.backends.wrapper.app import app

    routes = {r.path for r in app.routes}
    assert "/api/tags" in routes


def test_wrapper_has_ollama_api_generate():
    """Wrapper must expose POST /api/generate."""
    from agent.backends.wrapper.app import app

    routes = {r.path for r in app.routes}
    assert "/api/generate" in routes


def test_wrapper_has_ollama_api_chat():
    """Wrapper must expose POST /api/chat."""
    from agent.backends.wrapper.app import app

    routes = {r.path for r in app.routes}
    assert "/api/chat" in routes


def test_wrapper_has_v1_models():
    """Wrapper must expose GET /v1/models."""
    from agent.backends.wrapper.app import app

    routes = {r.path for r in app.routes}
    assert "/v1/models" in routes


def test_wrapper_port_defaults_to_9002():
    """Wrapper default port must be 9002, not Ollama's 11434."""
    from agent.backends.wrapper.app import WRAPPER_PORT

    assert WRAPPER_PORT == 9002


# ---------------------------------------------------------------------------
# Port separation
# ---------------------------------------------------------------------------


def test_port_separation():
    """Ollama and wrapper ports must be different."""
    from agent.backends.wrapper.app import WRAPPER_PORT

    OLLAMA_PORT = 11434
    assert WRAPPER_PORT != OLLAMA_PORT
