# GPU Benchmarking Guide

This document covers GPU-specific setup for bench-race, with particular attention
to the NVIDIA GB10 (Blackwell architecture), PyTorch nightly requirements, and
Ollama optimization considerations across different hardware.

---

## Table of Contents

- [Supported GPU Architectures](#supported-gpu-architectures)
- [The NVIDIA GB10 (Blackwell)](#the-nvidia-gb10-blackwell)
- [PyTorch Nightly Builds for GB10](#pytorch-nightly-builds-for-gb10)
- [Ollama on the GB10](#ollama-on-the-gb10)
- [ComfyUI on the GB10](#comfyui-on-the-gb10)
- [Apple Silicon (Unified Memory)](#apple-silicon-unified-memory)
- [Model Fit Scoring](#model-fit-scoring)
- [Runtime Metrics Collection](#runtime-metrics-collection)
- [Suggested Screenshots](#suggested-screenshots)

---

## Supported GPU Architectures

bench-race supports three GPU categories:

| Category | Examples | Memory Model | Detection |
|----------|----------|-------------|-----------|
| **Apple Silicon** | M1, M2, M4 (all variants) | Unified (shared with CPU) | `system_profiler SPDisplaysDataType` |
| **NVIDIA Discrete** | RTX 3090, 4090, 5090, GB10 | Dedicated VRAM | `nvidia-smi` / `pynvml` |
| **CPU-only** | Any machine without a GPU | System RAM only | Fallback when no GPU detected |

The agent auto-detects GPU hardware at startup via `agent/hardware_discovery.py`
and reports it to central in the `/capabilities` response. Central uses this
information (along with any overrides in `machines.yaml`) for model fit scoring.

---

## The NVIDIA GB10 (Blackwell)

The NVIDIA GB10 is a Blackwell-architecture workstation GPU with
CUDA compute capability **12.1**. It is a powerful chip, but its newness presents
several practical challenges for the bench-race stack:

### What Makes the GB10 Different

1. **Compute capability 12.1** -- Most stable PyTorch releases (as of early 2026)
   ship pre-compiled CUDA kernels only up to SM 9.0 (Hopper). The GB10's
   SM 12.x architecture is **not** included in stable wheel builds. Attempting
   to run `torch.randn(1, device='cuda')` with a stable PyTorch on a GB10 will
   fail with:

   ```
   no kernel image is available for execution on the device
   ```

2. **Large memory** -- The GB10 typically ships with 64 GiB or more of VRAM
   (plus substantial system RAM, e.g., 128-512 GiB). This is enough to run
   70B-parameter models that won't fit on consumer GPUs.

3. **Grace CPU pairing** -- Many GB10 systems use an NVIDIA Grace ARM CPU
   rather than x86. The installer detects this by checking `lscpu` for
   "GB10" or "Grace" in the model name.

### GB10 Detection

The installer script (`scripts/install_agent.sh`) auto-detects GB10 hardware:

```bash
# From install_agent.sh
cpu_model=$(lscpu | grep "Model name" | sed 's/Model name:[[:space:]]*//' | head -1)

if [[ "$cpu_model" =~ GB10 ]] || [[ "$cpu_model" =~ "Grace" ]]; then
    IS_GB10=true
    PLATFORM="linux-gb10"
fi
```

You can also force GB10 mode manually:

```bash
./scripts/install_agent.sh --platform linux-gb10
```

### GB10 Configuration in machines.yaml

```yaml
- machine_id: "gb10"
  label: "Dell ProMax (GB10, 128GB)"
  logo: "nvidia"
  agent_base_url: "http://10.0.0.5:9001"

  cpu_cores: 64
  cpu_physical_cores: 32
  total_system_ram_bytes: 137438953472  # 128 GiB

  gpu:
    name: "NVIDIA GB10"
    type: "discrete"
    vram_bytes: 68719476736              # 64 GiB
    cuda_compute: [12, 1]                # Blackwell SM 12.1
    driver_version: "545.101"
    pci_bus: "0000:01:00.0"
```

---

## PyTorch Nightly Builds for GB10

### Why Nightly Is Required

Stable PyTorch releases include pre-compiled CUDA kernels for established GPU
architectures (e.g., Ampere SM 8.0, Ada Lovelace SM 8.9, Hopper SM 9.0). The
GB10's Blackwell architecture (SM 12.x) is only supported in **nightly
(pre-release) builds** that include CUDA 13.0 support.

### How the Installer Selects the Right Build

The installer reads CUDA compute capability from `nvidia-smi` and selects the
correct PyTorch index URL:

```bash
# Detect compute capability
cc_raw=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -1)
# Parses to: compute_major=12, compute_minor=1

# Selection logic
if [[ "$compute_major" -ge 13 ]] || \
   [[ "$compute_major" -eq 12 && "$compute_minor" -ge 1 ]]; then
    # GB10 (12.1) or future Blackwell variants (13.x)
    PYTORCH_INDEX_URL="https://download.pytorch.org/whl/nightly/cu130"
    PYTORCH_TAG="nightly/cu130"

elif [[ "$compute_major" -eq 12 ]]; then
    # RTX 5090 and similar (12.0) -- stable CUDA 12.9
    PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu129"
    PYTORCH_TAG="stable/cu129"

else
    # Older architectures (Ampere, Ada, etc.) -- stable CUDA 12.1
    PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
    PYTORCH_TAG="stable/cu121"
fi
```

### PyTorch Build Matrix

| GPU | Compute Capability | PyTorch Build | CUDA Version | Stable? |
|-----|-------------------|---------------|-------------|---------|
| RTX 3090, A100 | 8.x | `stable/cu121` | CUDA 12.1 | Yes |
| RTX 4090, L40 | 8.9 | `stable/cu121` | CUDA 12.1 | Yes |
| RTX 5090 | 12.0 | `stable/cu129` | CUDA 12.9 | Yes |
| **GB10** | **12.1** | **`nightly/cu130`** | **CUDA 13.0** | **No (nightly)** |
| Future Blackwell | 13.x+ | `nightly/cu130` | CUDA 13.0 | No (nightly) |

### The `--pre` Flag

For nightly builds, pip needs the `--pre` flag to accept pre-release packages:

```bash
# Nightly install (GB10)
pip install --pre \
    --index-url https://download.pytorch.org/whl/nightly/cu130 \
    --upgrade --force-reinstall --no-cache-dir \
    torch torchvision torchaudio

# Stable install (other GPUs)
pip install \
    --index-url https://download.pytorch.org/whl/cu121 \
    --upgrade --force-reinstall --no-cache-dir \
    torch torchvision torchaudio
```

### Implications of Using Nightly Builds

- **Nightly builds are not release-quality.** They may contain regressions,
  API changes, or bugs that won't be present in stable releases.
- **Reproducibility.** Nightly builds change daily. To pin a specific nightly
  version, note the date or wheel filename after installation.
- **Disk space.** PyTorch with CUDA 13.0 support is large (~2.5 GB+).
  The installer uses `--force-reinstall --no-cache-dir` to avoid stale caches.
- **Check periodically.** Once PyTorch ships a stable release with SM 12.1
  support, you should switch to stable builds. Update the installer logic or
  re-run the installer to pick up the change.

### CUDA Probe at Startup

The agent runs a lightweight CUDA probe at startup
(`agent/startup_checks.py`) to verify that the installed PyTorch can actually
allocate GPU memory:

```python
command = [
    python_path, "-c",
    "import torch; torch.cuda.is_available(); torch.randn(1, device='cuda'); print('ok')",
]
```

If this fails with `"no kernel image is available for execution on the device"`,
the error is classified as `cuda_unsupported_arch`, and the agent can
optionally fall back to CPU mode for ComfyUI workloads (see
[ComfyUI on the GB10](#comfyui-on-the-gb10)).

---

## Ollama on the GB10

### The Optimization Gap

Ollama is highly optimized for Apple Silicon and common NVIDIA architectures
(Ampere, Ada Lovelace). On these platforms, Ollama uses hand-tuned CUDA/Metal
kernels and memory management that extracts maximum performance from the
hardware.

On the GB10, Ollama's performance characteristics differ:

1. **Kernel coverage.** Ollama's bundled `llama.cpp` backend compiles CUDA
   kernels for a set of target architectures. If the GB10's SM 12.1 is not
   in that set, Ollama may fall back to PTX JIT compilation, which is
   functional but slower than native kernels.

2. **Memory management.** Ollama on Apple Silicon benefits from unified memory
   -- the GPU and CPU share the same physical RAM with zero-copy access. On
   the GB10, model weights must reside in GPU VRAM. Although the GB10 has
   generous VRAM (64 GiB), Ollama's memory allocation strategy may not yet
   be tuned for Blackwell's memory hierarchy.

3. **Quantization kernels.** The GGUF quantization kernels in `llama.cpp`
   (which Ollama uses internally) are optimized per-architecture. Blackwell
   may not yet have architecture-specific quantization fast-paths, resulting
   in slightly lower tokens/sec compared to theoretical peak.

### Practical Impact

In real-world benchmarks, expect:

- **Time to First Token (TTFT)** may be higher on GB10 compared to Apple
  Silicon for small models, due to model loading overhead. For large models
  (70B+), GB10's larger VRAM can actually yield lower TTFT because the
  model fits entirely in VRAM without offloading.

- **Throughput (tokens/sec)** may not reach the GB10's theoretical peak
  until Ollama ships Blackwell-optimized kernels. Compare results across
  hardware to quantify the gap.

- **Ollama version matters.** Newer Ollama releases progressively add
  support for newer architectures. Always use the latest Ollama on GB10.

### Recommendations for GB10 + Ollama

1. **Keep Ollama updated.** Run `ollama update` or reinstall regularly:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. **Monitor token throughput.** Use bench-race's comparison UI to track
   tokens/sec over time as Ollama releases improve GB10 support.

3. **Test different quantizations.** The performance gap between Q4 and Q8
   may differ on GB10 compared to other GPUs. Use bench-race to find the
   best speed/quality tradeoff for your workload.

4. **Consider the 70B advantage.** Where GB10 shines is running models that
   don't fit in other GPUs' VRAM. A 70B Q4 model (~40 GiB) fits comfortably
   in 64 GiB of GB10 VRAM but requires offloading on a 24 GiB consumer GPU.

---

## ComfyUI on the GB10

### The Dell GB10 Bug

During testing with ComfyUI on Dell GB10 hardware, a specific bug was
discovered and fixed:

**Problem:** After submitting a prompt to ComfyUI, polling
`GET /history/{prompt_id}` would immediately return HTTP 200 with a valid
response body -- but the `outputs` dictionary was empty. The previous code
incorrectly interpreted this as a completed job with zero images.

**Root cause:** On the GB10, ComfyUI's history endpoint becomes available
before the actual image generation has started, likely due to differences in
GPU scheduling or initialization timing.

**Fix:** The polling logic now validates that history contains actual image
outputs before marking a job as complete:

```python
# agent/comfy_ws.py
has_images = any(
    "images" in node_outputs and node_outputs["images"]
    for node_outputs in outputs.values()
)

completed_flag = status.get("completed", False) or status_str == "completed"
if completed_flag and not has_images:
    self._error = "Job reported complete but history contains no outputs"
    return  # Continue polling

if has_images:
    self._completed = True
    return
```

### CPU Fallback for Unsupported Architectures

If the CUDA startup probe fails (e.g., because PyTorch doesn't have SM 12.1
kernels), the agent can fall back to CPU-only mode for ComfyUI:

```yaml
# agent/config/agent.yaml
comfyui:
  allow_cpu_fallback: false          # Set to true to enable fallback
  cpu_fallback_on:
    - "cuda_unsupported_arch"        # Trigger condition
```

When CPU fallback activates, the agent sets `CUDA_VISIBLE_DEVICES=""` to force
ComfyUI to use the CPU. Image generation will be significantly slower but
functional.

---

## Apple Silicon (Unified Memory)

Apple Silicon machines (M1/M2/M4 and their Pro/Max/Ultra variants) use
**unified memory** -- the CPU and GPU share the same physical RAM.

### Key Differences from Discrete GPUs

| Aspect | Unified (Apple) | Discrete (NVIDIA) |
|--------|----------------|-------------------|
| VRAM | Shares system RAM | Dedicated GPU memory |
| Data transfer | Zero-copy | PCIe bus transfer |
| Max memory | System RAM (up to 192 GiB) | GPU VRAM (8-80 GiB typical) |
| Memory type | LPDDR5/LPDDR5X | GDDR6X / HBM3 |

### Configuration

In `machines.yaml`, set `gpu.type: "unified"` and `gpu.vram_bytes` equal to
`total_system_ram_bytes`:

```yaml
- machine_id: "macbook"
  gpu:
    name: "Apple M4 Max"
    type: "unified"
    vram_bytes: 137438953472  # Same as total_system_ram_bytes
```

### Unified Memory Implications for Benchmarking

- **Model fit scoring** uses `vram_bytes` for both unified and discrete GPUs.
  On unified systems, a 70B model can use most of system RAM, but the OS and
  other processes also need memory.
- **Runtime metrics** -- On Apple Silicon, GPU utilization comes from
  `powermetrics` (requires sudo or specific permissions). If unavailable, GPU
  fields show as `null` and the UI uses system memory as a proxy.

---

## Model Fit Scoring

Central computes a memory-first fit score for each machine + model combination:

```
usable_vram = vram_bytes * usable_vram_fraction      (default: 0.9)
estimated_peak = model_bytes * activation_multiplier  (default: 1.7)
fit_ratio = usable_vram / estimated_peak
```

| Fit Ratio | Label | UI Color | Meaning |
|-----------|-------|----------|---------|
| >= 1.2 | GOOD | Green | Comfortable headroom |
| 1.0 - 1.2 | RISK | Orange | Tight; may work with some swapping |
| < 1.0 | FAIL | Red | Model likely won't fit; expect OOM |

This scoring is displayed as badges in the web UI preflight panel, helping you
decide which machines can handle a given model before running the benchmark.

---

## Runtime Metrics Collection

Agents sample hardware metrics at a configurable interval:

```yaml
runtime_sampler:
  enabled: true
  interval_s: 1          # Sample every second
  buffer_len: 120        # Keep 2 minutes of history
```

### Metrics Collected

| Metric | NVIDIA | Apple Silicon | CPU-only |
|--------|--------|-------------|----------|
| CPU % | psutil | psutil | psutil |
| GPU % | pynvml / nvidia-smi | powermetrics | N/A |
| VRAM used | pynvml / nvidia-smi | (system memory proxy) | N/A |
| System RAM | psutil | psutil | psutil |

Metrics are exposed at `GET /api/agent/runtime_metrics` and streamed to central
via WebSocket for real-time display in the UI.

---

## Suggested Screenshots

The following screenshots would be helpful for this documentation:

1. **Web UI machine cards during a benchmark** -- showing side-by-side
   comparison of Apple Silicon, NVIDIA discrete, and GB10 machines with live
   token streaming and metrics.

2. **FIT badges on machine cards** -- showing GOOD (green), RISK (orange),
   and FAIL (red) badges for different model/machine combinations.

3. **Compute benchmark results** -- showing the compute results panel with
   elapsed time, primes/sec, and progress bars for multiple machines.

4. **Reset diagnostics modal** -- showing the detailed diagnostics when an
   agent reset encounters the CUDA unsupported architecture error.

5. **Preflight warning banner** -- showing the "Missing model" and "Agent
   unreachable" warnings before a run starts.
