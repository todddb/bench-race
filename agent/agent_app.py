from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
import sys
from typing import Dict, List

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---------------------------
# Path and import robustness
# ---------------------------
AGENT_DIR = Path(__file__).resolve().parent            # .../bench-race/agent
ROOT_DIR = AGENT_DIR.parent                            # .../bench-race
CONFIG_PATH = AGENT_DIR / "config" / "agent.yaml"

# Make repo root importable so `shared` works
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# now import shared pydantic schemas
from shared.schemas import Capabilities, Event, JobStartResponse, LLMRequest  # type: ignore
from backends.ollama_backend import check_ollama_available, stream_ollama_generate, get_ollama_models

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bench-agent")

# ---------------------------
# Load config
# ---------------------------
def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG = load_config()

# ---------------------------
# FastAPI app + websockets
# ---------------------------
app = FastAPI(title="bench-race agent")

# Manage connected websocket clients
# Each entry: client_id -> WebSocket
WS_CLIENTS: Dict[str, WebSocket] = {}

# Track running jobs (job_id -> asyncio.Task)
RUNNING_JOBS: Dict[str, asyncio.Task] = {}


# ---------------------------
# Helpers: websocket broadcast
# ---------------------------
async def _broadcast_event(event: Event):
    """Send event to all connected websocket clients (best-effort)."""
    to_remove: List[str] = []
    payload = event.model_dump()
    txt = json.dumps(payload)
    for cid, ws in list(WS_CLIENTS.items()):
        try:
            await ws.send_text(txt)
        except Exception as exc:
            log.warning("WebSocket send failed for %s: %s", cid, exc)
            to_remove.append(cid)
    for cid in to_remove:
        WS_CLIENTS.pop(cid, None)


# ---------------------------
# HTTP endpoints
# ---------------------------
@app.get("/health")
async def health():
    return {"ok": True, "machine_id": CFG.get("machine_id"), "label": CFG.get("label")}


@app.get("/capabilities")
async def capabilities():
    base_url = CFG.get("ollama_base_url", "http://127.0.0.1:11434")

    # Check Ollama reachability and get available models
    ollama_reachable = await check_ollama_available(base_url)
    ollama_models = []
    if ollama_reachable:
        ollama_models = await get_ollama_models(base_url)

    cap = Capabilities(
        machine_id=CFG.get("machine_id"),
        label=CFG.get("label"),
        tests=CFG.get("tests", ["llm_generate"]),
        llm_models=CFG.get("llm_models", []),
        whisper_models=CFG.get("whisper_models", []),
        sdxl_profiles=CFG.get("sdxl_profiles", []),
        ollama_reachable=ollama_reachable,
        ollama_models=ollama_models,
    )
    return JSONResponse(cap.model_dump())


# ---------------------------
# Background job: LLM streaming
# ---------------------------
async def _run_mock_stream(job_id: str, model: str, prompt: str, max_tokens: int, temperature: float, num_ctx: int, fallback_reason: str = "unknown"):
    """
    Fallback streaming for development when Ollama isn't available.
    Produces a few token chunks with small sleeps to simulate behavior.

    Args:
        fallback_reason: Why mock was used ("ollama_unreachable", "missing_model", "stream_error")
    """
    ttft = None
    gen_tokens = 0
    start_time = time.perf_counter()
    t_first = None

    # Simple deterministic split: break prompt into sentences or words
    chunks = []
    if len(prompt) < 200:
        # split by phrase length
        words = prompt.split()
        chunk_size = max(1, len(words) // 6)
        for i in range(0, len(words), chunk_size):
            chunks.append(" ".join(words[i : i + chunk_size]))
    else:
        # long prompt -> sample substrings
        for i in range(0, min(len(prompt), 300), 40):
            chunks.append(prompt[i : i + 40])

    # Add a few generated-sounding chunks
    fake_responses = [
        "Here are the main tradeoffs: unified memory simplifies large model usage.",
        "Discrete VRAM can provide higher per-GPU throughput for small models.",
        "Unified memory reduces the need for explicit sharding and can reduce engineering complexity.",
        "However it may have higher latencies in some workloads.",
        "In practice, measure both on your workload and pick what's simpler and faster."
    ]
    chunks.extend(fake_responses)

    for chunk in chunks:
        await asyncio.sleep(0.35)  # simulate chunk delay
        now = time.perf_counter()
        if ttft is None:
            ttft = (now - start_time) * 1000.0
            t_first = now
        toks = len(chunk.split())
        gen_tokens += toks
        ev = Event(job_id=job_id, type="llm_token", payload={"text": chunk, "timestamp_s": now})
        await _broadcast_event(ev)

    end_time = time.perf_counter()
    total_ms = (end_time - start_time) * 1000.0
    gen_tps = gen_tokens / (end_time - t_first) if t_first and end_time > t_first else None
    result = {
        "ttft_ms": ttft,
        "gen_tokens": gen_tokens,
        "gen_tokens_per_s": gen_tps,
        "total_ms": total_ms,
        "model": model,
        "engine": "mock",
        "fallback_reason": fallback_reason,
    }
    return result


async def _job_runner_llm(job_id: str, req: LLMRequest):
    """
    Background runner for llm_generate jobs.
    Tries Ollama streaming first; falls back to mock streaming on failure.
    Emits events while running and a final 'job_done' event with metrics.
    """
    model = req.model
    prompt = req.prompt
    max_tokens = req.max_tokens
    temperature = req.temperature
    num_ctx = req.num_ctx

    log.info("Starting job %s model=%s", job_id, model)
    base_url = CFG.get("ollama_base_url", "http://127.0.0.1:11434")
    try:
        available = await check_ollama_available(base_url)
        if not available:
            log.warning("Ollama unreachable; falling back to mock backend for job %s", job_id)
            result = await _run_mock_stream(job_id, model, prompt, max_tokens, temperature, num_ctx, fallback_reason="ollama_unreachable")
        else:
            # Check if model is available on Ollama
            ollama_models = await get_ollama_models(base_url)
            if model not in ollama_models:
                log.warning("Model %s not found on Ollama (available: %s); falling back to mock for job %s", model, ollama_models, job_id)
                result = await _run_mock_stream(job_id, model, prompt, max_tokens, temperature, num_ctx, fallback_reason="missing_model")
            else:
                log.info("Using Ollama backend for job %s", job_id)
                async def _on_token(text: str, timestamp_s: float) -> None:
                    ev = Event(job_id=job_id, type="llm_token", payload={"text": text, "timestamp_s": timestamp_s})
                    await _broadcast_event(ev)

                result = await stream_ollama_generate(
                    job_id=job_id,
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    num_ctx=num_ctx,
                    base_url=base_url,
                    on_token=_on_token,
                )
    except Exception as e:
        log.warning("Ollama stream failed (%s); falling back to mock stream", e)
        result = await _run_mock_stream(job_id, model, prompt, max_tokens, temperature, num_ctx, fallback_reason="stream_error")

    # Emit final metrics event
    ev = Event(job_id=job_id, type="job_done", payload=result)
    await _broadcast_event(ev)
    log.info("Job %s done. result=%s", job_id, result)


# ---------------------------
# Jobs endpoint
# ---------------------------
@app.post("/jobs", response_model=JobStartResponse)
async def start_job(req: LLMRequest):
    """
    Start a job and run it in the background. Returns job_id immediately.
    Streaming events are sent to connected WebSocket clients.
    """
    job_id = str(uuid.uuid4())

    # If a job_id already running, reject
    # (not likely, but defensive)
    if job_id in RUNNING_JOBS:
        raise HTTPException(status_code=409, detail="job already running")

    # Launch background runner
    task = asyncio.create_task(_job_runner_llm(job_id, req))
    RUNNING_JOBS[job_id] = task

    # When task finishes, remove it from RUNNING_JOBS
    def _on_done(t: asyncio.Task):
        RUNNING_JOBS.pop(job_id, None)
        if t.exception():
            log.exception("Job %s terminated with exception", job_id)

    task.add_done_callback(_on_done)

    return JobStartResponse(job_id=job_id)


# ---------------------------
# WebSocket endpoint
# ---------------------------
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    """
    Simple pubsub: any connected client receives all events.
    The central server should connect here to receive token streams and job_done events.
    """
    await ws.accept()
    client_id = str(uuid.uuid4())
    WS_CLIENTS[client_id] = ws
    log.info("WS client connected: %s (total=%d)", client_id, len(WS_CLIENTS))
    try:
        while True:
            # keep the connection alive by reading; the central may send pings/commands later
            data = await ws.receive_text()
            # we ignore received messages for now; could implement control messages
            log.debug("WS recv from %s: %s", client_id, data)
    except WebSocketDisconnect:
        log.info("WS client disconnected: %s", client_id)
        WS_CLIENTS.pop(client_id, None)
    except Exception as exc:
        log.exception("WS error for %s: %s", client_id, exc)
        WS_CLIENTS.pop(client_id, None)


# ---------------------------
# Graceful shutdown
# ---------------------------
@app.on_event("shutdown")
async def _shutdown():
    log.info("Shutting down: cancelling %d running jobs", len(RUNNING_JOBS))
    for job_id, t in list(RUNNING_JOBS.items()):
        t.cancel()
    # close websockets
    for cid, ws in list(WS_CLIENTS.items()):
        try:
            await ws.close()
        except Exception:
            pass
    WS_CLIENTS.clear()
