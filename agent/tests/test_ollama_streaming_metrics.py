import asyncio
import json
import time

from agent.backends.ollama_backend import stream_ollama_generate


class _StreamResponse:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    async def aiter_lines(self):
        yield json.dumps({"response": "hello "})
        await asyncio.sleep(0.01)
        yield json.dumps({"response": "world", "done": True})


class _Client:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def stream(self, method, url, json):
        assert method == "POST"
        assert url.endswith("/api/generate")
        assert json["stream"] is True
        return _StreamResponse()


def test_streaming_ollama_metrics(monkeypatch):
    from agent.backends import ollama_backend as mod

    monkeypatch.setattr(mod.httpx, "AsyncClient", _Client)

    collected = []

    async def on_token(text, now):
        collected.append((text, now))

    result = asyncio.run(
        stream_ollama_generate(
            job_id="job-1",
            model="llama3",
            prompt="hello",
            max_tokens=16,
            temperature=0.1,
            num_ctx=2048,
            base_url="http://127.0.0.1:11434",
            on_token=on_token,
        )
    )

    assert len(collected) == 2
    assert result["ttft_ms"] is not None and result["ttft_ms"] > 0
    assert result["total_ms"] > result["ttft_ms"]
    assert result["gen_tokens"] >= 2
    assert result["engine"] == "ollama"
