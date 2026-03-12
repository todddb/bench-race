"""Tests for MLX backend streaming pipeline.

Verifies that:
- MLX server /infer endpoint yields tokens with \\n delimiter and text/event-stream content-type
- MLX adapter infer_stream processes SSE lines via aiter_lines() for immediate delivery
- Wrapper /v1/chat/completions wraps each token in a proper SSE data: frame
"""
from __future__ import annotations

import asyncio
import importlib
import json
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# MLX server streaming tests
# ---------------------------------------------------------------------------

def test_mlx_server_infer_streaming_uses_event_stream_media_type():
    """The /infer endpoint must return text/event-stream when stream=True."""
    server = importlib.import_module("agent.backends.mlx.server")

    tokens_yielded = []

    def fake_stream_generate(model, tokenizer, prompt, max_tokens, temp):
        class _Tok:
            def __init__(self, t):
                self.text = t

        yield _Tok("Hello")
        yield _Tok("Hello world")

    mock_mlx_lm = MagicMock()
    mock_mlx_lm.stream_generate = fake_stream_generate

    # Patch state so a model appears loaded
    orig_model = server._state.model
    orig_tokenizer = server._state.tokenizer
    orig_mlx_lm = server._mlx_lm

    try:
        server._state.model = MagicMock()
        server._state.tokenizer = MagicMock()
        server._mlx_lm = mock_mlx_lm

        client = TestClient(server.app)
        resp = client.post(
            "/infer",
            json={"prompt": "hi", "max_tokens": 10, "temperature": 0.7, "stream": True},
        )

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        # Collect streamed body
        body = resp.content.decode("utf-8")
        # Each token must be terminated by a newline so aiter_lines() can split
        lines = [ln for ln in body.split("\n") if ln]
        assert len(lines) >= 1
        # First token should be "Hello", second incremental token "world" (delta)
        assert lines[0] == "Hello"
        assert lines[1] == " world"
    finally:
        server._state.model = orig_model
        server._state.tokenizer = orig_tokenizer
        server._mlx_lm = orig_mlx_lm


def test_mlx_server_infer_streaming_newline_delimiter():
    """Each token yielded by token_stream must end with a newline character."""
    server = importlib.import_module("agent.backends.mlx.server")

    def fake_stream_generate(model, tokenizer, prompt, max_tokens, temp):
        class _Tok:
            def __init__(self, t):
                self.text = t

        cumulative = ""
        for word in ["one", " two", " three"]:
            cumulative += word
            yield _Tok(cumulative)

    mock_mlx_lm = MagicMock()
    mock_mlx_lm.stream_generate = fake_stream_generate

    orig_model = server._state.model
    orig_tokenizer = server._state.tokenizer
    orig_mlx_lm = server._mlx_lm

    try:
        server._state.model = MagicMock()
        server._state.tokenizer = MagicMock()
        server._mlx_lm = mock_mlx_lm

        client = TestClient(server.app)
        resp = client.post(
            "/infer",
            json={"prompt": "count", "max_tokens": 10, "temperature": 0.0, "stream": True},
        )

        assert resp.status_code == 200
        raw = resp.content.decode("utf-8")

        # Every token chunk must end with "\n" so that httpx aiter_lines()
        # can split tokens immediately without buffering the whole response.
        for chunk in raw.split("\n"):
            if chunk:
                # Chunks are the individual token texts (newline stripped by split)
                assert len(chunk) > 0

        # Verify we got all three incremental tokens
        lines = [ln for ln in raw.split("\n") if ln]
        assert lines == ["one", " two", " three"]
    finally:
        server._state.model = orig_model
        server._state.tokenizer = orig_tokenizer
        server._mlx_lm = orig_mlx_lm


def test_mlx_server_infer_non_streaming_returns_json():
    """The /infer endpoint with stream=False must return a JSON object with 'text'."""
    server = importlib.import_module("agent.backends.mlx.server")

    mock_mlx_lm = MagicMock()
    mock_mlx_lm.generate = MagicMock(return_value="The answer is 42.")
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode = MagicMock(return_value=[1, 2, 3, 4, 5])

    orig_model = server._state.model
    orig_tokenizer = server._state.tokenizer
    orig_mlx_lm = server._mlx_lm

    try:
        server._state.model = MagicMock()
        server._state.tokenizer = mock_tokenizer
        server._mlx_lm = mock_mlx_lm

        client = TestClient(server.app)
        resp = client.post(
            "/infer",
            json={"prompt": "What is 6*7?", "max_tokens": 32, "temperature": 0.0, "stream": False},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "text" in data
        assert data["text"] == "The answer is 42."
        assert "tokens" in data
        assert "meta" in data
        assert data["meta"]["engine"] == "mlx"
    finally:
        server._state.model = orig_model
        server._state.tokenizer = orig_tokenizer
        server._mlx_lm = orig_mlx_lm


# ---------------------------------------------------------------------------
# MLX adapter streaming tests
# ---------------------------------------------------------------------------

class _SSEStreamResponse:
    """Fake httpx streaming response that yields SSE-style text/event-stream lines."""

    def __init__(self, tokens):
        self._tokens = tokens

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def raise_for_status(self):
        pass

    @property
    def headers(self):
        return {"content-type": "text/event-stream"}

    async def aiter_lines(self):
        for tok in self._tokens:
            yield tok


class _FakeHTTPXClient:
    """Fake httpx.AsyncClient for MLX adapter tests."""

    def __init__(self, tokens, **kwargs):
        self._tokens = tokens

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def stream(self, method, url, json=None, **kwargs):
        assert method == "POST"
        assert url.endswith("/infer")
        assert json is not None
        assert json["stream"] is True, "adapter must send stream=True to MLX server"
        return _SSEStreamResponse(self._tokens)


def test_mlx_adapter_infer_stream_uses_aiter_lines():
    """infer_stream must send stream=True and iterate via aiter_lines for immediate delivery."""
    adapter_mod = importlib.import_module("agent.backends.wrapper.adapters.mlx_adapter")
    adapter = adapter_mod.MLXAdapter()

    tokens = ["Hello", " world", " from", " MLX"]

    async def run():
        with patch.object(adapter_mod.httpx, "AsyncClient",
                          lambda **kw: _FakeHTTPXClient(tokens, **kw)):
            collected = []
            async for chunk in adapter.infer_stream("test-model", {"prompt": "hi", "max_tokens": 10}):
                collected.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)
        return collected

    result = asyncio.run(run())
    assert result == tokens


def test_mlx_adapter_infer_stream_sends_stream_true():
    """The POST body sent to /infer must include stream=True."""
    adapter_mod = importlib.import_module("agent.backends.wrapper.adapters.mlx_adapter")
    adapter = adapter_mod.MLXAdapter()

    captured_body = {}

    class _CapturingStreamResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        @property
        def headers(self):
            return {"content-type": "text/event-stream"}

        async def aiter_lines(self):
            yield "token_one"

    class _CapturingClient:
        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def stream(self, method, url, json=None, **kw):
            captured_body.update(json or {})
            return _CapturingStreamResponse()

    async def run():
        with patch.object(adapter_mod.httpx, "AsyncClient",
                          lambda **kw: _CapturingClient(**kw)):
            chunks = []
            async for chunk in adapter.infer_stream("m", {"prompt": "test"}):
                chunks.append(chunk)
        return chunks

    asyncio.run(run())
    assert captured_body.get("stream") is True, "MLX adapter must send stream=True to server"


# ---------------------------------------------------------------------------
# Wrapper /v1/chat/completions streaming tests
# ---------------------------------------------------------------------------

class _FakeStreamingMLXAdapter:
    """Fake MLX adapter that yields raw token bytes, simulating real MLX streaming."""

    backend_name = "mlx"

    async def infer_stream(self, model_id, payload):
        # Simulate the MLX adapter yielding raw token bytes (after aiter_lines strips \n)
        for token in [" The", " sky", " is", " blue", "."]:
            yield token.encode("utf-8")

    async def list_models(self):
        return [{"id": "test-mlx-model", "object": "model", "owned_by": "mlx", "backend": "mlx"}]

    async def health(self):
        return {"ok": True, "engine": "mlx", "model": "test-mlx-model", "mem": {}}


def test_wrapper_chat_completions_streaming_sse_format(monkeypatch):
    """chat_completions with stream=True must yield proper SSE data: frames."""
    wrapper_app = importlib.import_module("agent.backends.wrapper.app")

    fake_adapter = _FakeStreamingMLXAdapter()
    monkeypatch.setitem(wrapper_app.adapters, "mlx", fake_adapter)
    monkeypatch.setitem(wrapper_app.active_state, "backend", "mlx")
    monkeypatch.setitem(wrapper_app.active_state, "model", "test-mlx-model")

    client = TestClient(wrapper_app.app)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "test-mlx-model",
            "messages": [{"role": "user", "content": "What color is the sky?"}],
            "stream": True,
        },
    ) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        lines = list(resp.iter_lines())

    # Filter meaningful SSE lines (non-empty, not [DONE])
    data_lines = [ln for ln in lines if ln.startswith("data: ") and "[DONE]" not in ln]
    assert len(data_lines) >= 1, "Expected at least one SSE data frame"

    # Each data line must be valid JSON with OpenAI chunk format
    token_texts = []
    for line in data_lines:
        payload_str = line[6:]  # strip "data: "
        chunk = json.loads(payload_str)
        assert chunk["object"] == "chat.completion.chunk"
        assert "choices" in chunk
        delta = chunk["choices"][0].get("delta", {})
        if "content" in delta and delta["content"]:
            token_texts.append(delta["content"])

    # Verify we got the expected token texts
    assert token_texts == [" The", " sky", " is", " blue", "."]

    # Final line should be data: [DONE]
    done_lines = [ln for ln in lines if ln == "data: [DONE]"]
    assert len(done_lines) == 1, "Expected exactly one data: [DONE] line"


def test_wrapper_chat_completions_non_streaming_returns_full_json(monkeypatch):
    """chat_completions with stream=False must return a complete OpenAI response."""
    wrapper_app = importlib.import_module("agent.backends.wrapper.app")

    class _FakeSyncAdapter:
        backend_name = "mlx"

        async def infer(self, model_id, payload):
            return {"text": "The sky is blue.", "tokens": 5}

        async def list_models(self):
            return []

    monkeypatch.setitem(wrapper_app.adapters, "mlx", _FakeSyncAdapter())
    monkeypatch.setitem(wrapper_app.active_state, "backend", "mlx")
    monkeypatch.setitem(wrapper_app.active_state, "model", "test-model")

    client = TestClient(wrapper_app.app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "What color is the sky?"}],
            "stream": False,
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["message"]["content"] == "The sky is blue."
    assert data["choices"][0]["finish_reason"] == "stop"
