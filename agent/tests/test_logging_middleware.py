# agent/tests/test_logging_middleware.py
import os
import json
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from fastapi.responses import JSONResponse
import sys
from pathlib import Path

# Add parent directory to path to import the middleware
sys.path.insert(0, str(Path(__file__).parent.parent))
from logging_middleware import SafeLoggingMiddleware


@pytest.fixture
def app_with_middleware():
    """Create a FastAPI app with SafeLoggingMiddleware for testing."""
    app = FastAPI()

    @app.post("/echo")
    async def echo(request: Request):
        """Echo endpoint that reads and returns the request body."""
        body = await request.json()
        return JSONResponse({"received": body, "status": "ok"})

    @app.post("/echo-form")
    async def echo_form(request: Request):
        """Echo endpoint for form data."""
        form = await request.form()
        return JSONResponse({"received": dict(form), "status": "ok"})

    @app.get("/health")
    async def health():
        """Simple health check endpoint."""
        return JSONResponse({"status": "healthy"})

    @app.post("/large-body")
    async def large_body(request: Request):
        """Endpoint that accepts large request bodies."""
        body = await request.body()
        return JSONResponse({"size": len(body), "status": "ok"})

    # Add the middleware
    app.add_middleware(SafeLoggingMiddleware)

    return app


def test_middleware_allows_body_reading(app_with_middleware, capsys):
    """Test that middleware properly replays body so endpoint can read it."""
    client = TestClient(app_with_middleware)

    # Send a POST request with JSON body
    test_data = {"message": "Hello, World!", "value": 42}
    response = client.post("/echo", json=test_data)

    # Verify the endpoint received and echoed the data
    assert response.status_code == 200
    data = response.json()
    assert data["received"] == test_data
    assert data["status"] == "ok"

    # Verify logging output
    captured = capsys.readouterr()
    assert "http_in" in captured.out
    assert "http_out" in captured.out


def test_middleware_handles_large_bodies(app_with_middleware):
    """Test that middleware handles large request bodies correctly."""
    client = TestClient(app_with_middleware)

    # Create a large body (10KB)
    large_data = "x" * 10000
    response = client.post("/large-body", content=large_data)

    # Verify the endpoint received the full body
    assert response.status_code == 200
    data = response.json()
    assert data["size"] == 10000
    assert data["status"] == "ok"


def test_middleware_handles_form_data(app_with_middleware):
    """Test that middleware handles form data correctly."""
    client = TestClient(app_with_middleware)

    # Send form data
    form_data = {"field1": "value1", "field2": "value2"}
    response = client.post("/echo-form", data=form_data)

    # Verify the endpoint received the form data
    assert response.status_code == 200
    data = response.json()
    assert data["received"] == form_data
    assert data["status"] == "ok"


def test_middleware_handles_get_requests(app_with_middleware):
    """Test that middleware handles GET requests (no body)."""
    client = TestClient(app_with_middleware)

    response = client.get("/health")

    # Verify the endpoint works correctly
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_middleware_logs_json_format(app_with_middleware, capsys, monkeypatch):
    """Test that middleware logs in JSON format when LOG_JSON=1."""
    monkeypatch.setenv("LOG_JSON", "1")
    client = TestClient(app_with_middleware)

    response = client.post("/echo", json={"test": "data"})
    assert response.status_code == 200

    # Check that output is valid JSON
    captured = capsys.readouterr()
    lines = [line for line in captured.out.strip().split("\n") if line]

    # Verify at least some lines are valid JSON
    json_lines = []
    for line in lines:
        try:
            parsed = json.loads(line)
            json_lines.append(parsed)
        except json.JSONDecodeError:
            pass  # Some lines might be from TestClient, skip them

    # Should have at least http_in and http_out events
    assert len(json_lines) >= 2
    events = [log.get("event") for log in json_lines if "event" in log]
    assert "http_in" in events
    assert "http_out" in events


def test_middleware_respects_log_http_body_setting(app_with_middleware, capsys, monkeypatch):
    """Test that middleware respects LOG_HTTP_BODY environment variable."""
    # Test with LOG_HTTP_BODY=1 (should log body)
    monkeypatch.setenv("LOG_HTTP_BODY", "1")
    monkeypatch.setenv("LOG_JSON", "1")

    # Need to reload the module to pick up env changes
    import importlib
    import logging_middleware
    importlib.reload(logging_middleware)

    app = FastAPI()

    @app.post("/test")
    async def test_endpoint(request: Request):
        body = await request.json()
        return JSONResponse({"ok": True})

    app.add_middleware(logging_middleware.SafeLoggingMiddleware)
    client = TestClient(app)

    capsys.readouterr()  # Clear previous output
    response = client.post("/test", json={"secret": "data"})

    captured = capsys.readouterr()
    # Should contain body in logs
    assert "secret" in captured.out or "body" in captured.out


def test_middleware_websocket_passthrough(app_with_middleware):
    """Test that middleware passes WebSocket connections unchanged."""
    from fastapi import WebSocket

    app = app_with_middleware

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        data = await websocket.receive_text()
        await websocket.send_text(f"Echo: {data}")
        await websocket.close()

    client = TestClient(app)

    # Test WebSocket connection
    with client.websocket_connect("/ws") as websocket:
        websocket.send_text("Hello")
        data = websocket.receive_text()
        assert data == "Echo: Hello"


def test_middleware_multiple_requests(app_with_middleware):
    """Test that middleware handles multiple sequential requests correctly."""
    client = TestClient(app_with_middleware)

    # Send multiple requests
    for i in range(5):
        test_data = {"iteration": i}
        response = client.post("/echo", json=test_data)
        assert response.status_code == 200
        data = response.json()
        assert data["received"]["iteration"] == i


def test_middleware_empty_body(app_with_middleware):
    """Test that middleware handles requests with empty bodies."""
    client = TestClient(app_with_middleware)

    response = client.post("/large-body", content=b"")
    assert response.status_code == 200
    data = response.json()
    assert data["size"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
