from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ----- Common -----
class MachineInfo(BaseModel):
    machine_id: str
    label: str
    agent_base_url: str
    notes: Optional[str] = None


class Capabilities(BaseModel):
    machine_id: str
    label: str
    tests: List[str] = Field(default_factory=list)
    llm_models: List[str] = Field(default_factory=list)
    whisper_models: List[str] = Field(default_factory=list)
    sdxl_profiles: List[str] = Field(default_factory=list)
    accelerator_type: Optional[str] = None
    accelerator_memory_gb: Optional[float] = None
    system_memory_gb: Optional[float] = None
    gpu_name: Optional[str] = None
    gpu_vram_bytes: Optional[int] = None
    gpu_type: Optional[Literal["discrete", "unified"]] = None
    cuda_compute: Optional[List[int]] = None
    cpu_cores: Optional[int] = None
    cpu_physical_cores: Optional[int] = None
    total_system_ram_bytes: Optional[int] = None
    # Extended capability fields for preflight checks
    ollama_reachable: Optional[bool] = None
    ollama_models: List[str] = Field(default_factory=list)  # Actually available on Ollama
    agent_reachable: Optional[bool] = None  # Set by central when aggregating
    comfyui_gpu_ok: Optional[bool] = None
    comfyui_cpu_ok: Optional[bool] = None


# ----- Job Requests -----
class LLMRequest(BaseModel):
    test_type: Literal["llm_generate"] = "llm_generate"
    model: str
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.2
    num_ctx: int = 4096
    repeat: int = 1
    stream: bool = True


class WhisperRequest(BaseModel):
    test_type: Literal["whisper_transcribe"] = "whisper_transcribe"
    asset_id: str
    model: str = "large-v3"
    language: Optional[str] = None
    stream: bool = True


class SDXLRequest(BaseModel):
    test_type: Literal["sdxl_generate"] = "sdxl_generate"
    profile: str = "sdxl_1024_30steps"
    positive_prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 30
    seed: int = 12345
    n_images: int = 1
    stream: bool = True


# ----- Job & Events -----
class JobStartResponse(BaseModel):
    job_id: str


class Event(BaseModel):
    job_id: str
    type: str
    payload: Dict[str, Any] = Field(default_factory=dict)


class LLMResult(BaseModel):
    ttft_ms: Optional[float] = None
    gen_tokens: Optional[int] = None
    gen_tokens_per_s: Optional[float] = None
    prompt_tokens: Optional[int] = None
    total_ms: Optional[float] = None
    model: str
    engine: str = "ollama"
    fallback_reason: Optional[str] = None  # "ollama_unreachable", "missing_model", "stream_error"


class WhisperResult(BaseModel):
    audio_seconds: Optional[float] = None
    wall_seconds: Optional[float] = None
    x_realtime: Optional[float] = None
    model: str
    engine: str = "faster-whisper"


class SDXLResult(BaseModel):
    wall_seconds: Optional[float] = None
    seconds_per_image: Optional[float] = None
    images: int = 1
    steps: int = 30
    width: int = 1024
    height: int = 1024
    engine: str = "comfyui"
