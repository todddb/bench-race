# Changelog

## Unreleased

### Changed
- **vLLM backend**: Replaced legacy wrapper-based Docker image with upstream vLLM
  source build (`bench-race/vllm:blackwell`). Adds Blackwell GPU support (RTX Pro 6000,
  RTX 5090, sm_120 / CUDA capability 12.0). Build time ~30-60 min via
  `scripts/install_x64_vllm.sh`.
- **vLLM port**: Default port changed from 8010 to 8000 (native OpenAI API, no wrapper).
- **vLLM model argument**: `start-vllm` now accepts HuggingFace model IDs directly
  (e.g., `Qwen/Qwen2.5-72B-Instruct`) in addition to local paths.
- **install_x64_vllm.sh**: Now clones vLLM source and builds with `CUDA_VERSION=12.8.1`
  and `torch_cuda_arch_list="9.0 12.0"`. Accepts `--fresh` and `--arch` flags.

- Installer: optional sudoers drop-in with sudo -n fail-fast handling; generated agent config now includes central_base_url and fixes ownership when run as root.
- Runtime: non-interactive sudo failures return clearer remediation; ComfyUI history polling is primary completion signal with timeout events.
- Image benchmarks: checkpoint identifiers resolve to filenames (digest fallback supported) and status polling no longer hangs on unknown digests.
- Image sparklines now remain robust when timing fields are missing.
