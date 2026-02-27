import asyncio
import importlib


def test_handle_backend_switch_emits_running(monkeypatch):
    handler = importlib.import_module("agent.agent_ws_handler")

    events = []

    async def fake_run(send, req_id, phase, cmd):
        await send({"type": "backend_switch_status", "payload": {"request_id": req_id, "phase": phase, "detail": cmd, "timestamp": 1}})
        return 0

    async def fake_send(payload):
        events.append(payload)

    monkeypatch.setattr(handler, "_run_script_and_stream", fake_run)

    asyncio.run(handler.handle_backend_switch({"backend": "ollama", "request_id": "r1"}, fake_send))

    phases = [e["payload"]["phase"] for e in events if e.get("type") == "backend_switch_status"]
    assert "offline" in phases
    assert "stopping" in phases
    assert "starting" in phases
    assert phases[-1] == "running"
