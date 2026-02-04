from __future__ import annotations

from typing import Dict, List


CUDA_UNSUPPORTED_REMEDIATION = [
    "This ComfyUI install is using a PyTorch/CUDA build that doesn’t support this GPU architecture.",
    "Fix: reinstall PyTorch with a CUDA build that supports your GPU (or build from source), then restart ComfyUI.",
    "If you just upgraded GPUs (e.g., RTX 50xx), older wheels often won’t include the needed SM kernels.",
]


def _contains_any(message: str, needles: List[str]) -> bool:
    lowered = message.lower()
    return any(needle.lower() in lowered for needle in needles)


def classify_comfy_error(msg: str) -> Dict[str, object]:
    """Classify ComfyUI error messages for UI-safe display + remediation guidance."""
    message = msg or ""
    if _contains_any(message, ["no kernel image is available for execution on the device"]):
        return {
            "category": "cuda_unsupported_arch",
            "short": "Torch/CUDA build doesn’t support this GPU (kernel image not available).",
            "action": CUDA_UNSUPPORTED_REMEDIATION,
        }
    if _contains_any(message, ["cuda out of memory"]):
        return {
            "category": "oom",
            "short": "CUDA out of memory.",
            "action": [
                "Reduce resolution, steps, or batch size.",
                "Close other GPU workloads and retry.",
            ],
        }
    if _contains_any(message, ["could not find checkpoint", "checkpoint not found", "checkpoints dir missing"]):
        return {
            "category": "missing_checkpoint",
            "short": "Checkpoint not found.",
            "action": [
                "Sync the checkpoint to this agent and retry.",
                "Verify the checkpoint filename matches the job configuration.",
            ],
        }
    return {
        "category": "unknown",
        "short": "ComfyUI execution error.",
        "action": ["Check agent logs for the full traceback and details."],
    }
