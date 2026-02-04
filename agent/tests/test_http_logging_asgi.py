"""
Unit tests for ASGI-safe HTTP logging middleware.

Tests that the middleware can:
- Replay request bodies without crashing
- Handle large POST bodies
- Work with FastAPI endpoints that read request.body()
"""

from fastapi import FastAPI, Request
from starlette.testclient import TestClient
from agent.http_logging_asgi import HTTPLoggingASGIMiddleware


def test_replays_body_without_crashing():
    """Test that middleware replays request body correctly."""
    app = FastAPI()
    app.add_middleware(HTTPLoggingASGIMiddleware, agent_id="t", hostname="h")

    @app.post("/echo")
    async def echo(req: Request):
        b = await req.body()
        return {"n": len(b), "start": b[:10].decode(errors="replace")}

    c = TestClient(app)
    payload = b"x" * 4096
    r = c.post("/echo", data=payload)
    assert r.status_code == 200
    assert r.json()["n"] == len(payload)


def test_handles_json_body():
    """Test that middleware handles JSON bodies correctly."""
    app = FastAPI()
    app.add_middleware(HTTPLoggingASGIMiddleware, agent_id="test", hostname="testhost")

    @app.post("/json_echo")
    async def json_echo(req: Request):
        data = await req.json()
        return {"received": data}

    c = TestClient(app)
    payload = {"prompt": "test prompt", "seed": 12345}
    r = c.post("/json_echo", json=payload)
    assert r.status_code == 200
    assert r.json()["received"] == payload


def test_handles_empty_body():
    """Test that middleware handles requests with no body."""
    app = FastAPI()
    app.add_middleware(HTTPLoggingASGIMiddleware, agent_id="test", hostname="testhost")

    @app.get("/test")
    async def test():
        return {"ok": True}

    c = TestClient(app)
    r = c.get("/test")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_handles_large_body():
    """Test that middleware handles large request bodies."""
    app = FastAPI()
    app.add_middleware(HTTPLoggingASGIMiddleware, agent_id="test", hostname="testhost")

    @app.post("/large")
    async def large(req: Request):
        b = await req.body()
        return {"size": len(b)}

    c = TestClient(app)
    # 1MB payload
    payload = b"x" * (1024 * 1024)
    r = c.post("/large", data=payload)
    assert r.status_code == 200
    assert r.json()["size"] == len(payload)
