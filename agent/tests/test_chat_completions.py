import importlib

from fastapi.testclient import TestClient

wrapper_app = importlib.import_module("agent.backends.wrapper.app")


class _FakeAdapter:
    async def infer(self, model_id, payload):
        return {"text": f"echo:{payload.get('prompt', '')}", "tokens": 1}


def test_chat_endpoint_exists(monkeypatch):
    client = TestClient(wrapper_app.app)
    monkeypatch.setitem(wrapper_app.adapters, "mlx", _FakeAdapter())
    monkeypatch.setitem(wrapper_app.active_state, "backend", "mlx")

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert resp.status_code in (200, 501)


def test_chat_completions_openai_response_shape(monkeypatch):
    client = TestClient(wrapper_app.app)
    monkeypatch.setitem(wrapper_app.adapters, "mlx", _FakeAdapter())
    monkeypatch.setitem(wrapper_app.active_state, "backend", "mlx")

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert "choices" in data
    assert data["choices"][0]["message"]["role"] == "assistant"
