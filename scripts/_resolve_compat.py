#!/usr/bin/env python3
"""_resolve_compat.py – Platform-aware vLLM/torch compatibility resolver.

Reads scripts/vllm_compat.json, detects the current platform (OS+arch) and
CUDA driver version, and outputs the best matching compatibility entry as
key=value lines on stdout.  Shell callers parse these lines to set variables.

Resolution algorithm:
  1. Determine ARCH_PLATFORM string (e.g. linux-x86_64, linux-aarch64, macos-arm64).
  2. Look up platform_mappings[ARCH_PLATFORM].  Entries are ordered from most
     specific (highest CUDA requirement) to most general (CPU fallback with
     cuda_min=null).
  3. Return the first entry whose cuda_min <= detected CUDA version, or the
     first entry with cuda_min=null as the CPU fallback.

Environment variables (override auto-detection; useful for tests and dry-runs):
  BENCH_RACE_ARCH_PLATFORM   Canonical platform string (e.g. linux-x86_64)
  BENCH_RACE_CUDA_VERSION    CUDA driver version string (e.g. 12.8) or empty
  BENCH_RACE_IS_BLACKWELL    Set to 1 if a Blackwell/GB10 GPU is detected

Output on stdout (key=value, one per line):
  vllm=<version>
  torch=<version>
  torchvision=<version or empty>
  torchaudio=<version or empty>
  setuptools=<version>
  torch_index_url=<url or empty>
  torch_tag=<tag>
  nightly=<true|false>
  conda_preferred=<true|false>

Diagnostic info is written to stderr.
Exit codes:
  0  – match found, output written
  1  – compat file not found or invalid JSON
  2  – no matching entry for the detected platform/CUDA combination
"""

import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Platform detection helpers
# ---------------------------------------------------------------------------

def _detect_platform() -> str:
    """Return canonical platform string like linux-x86_64 or macos-arm64."""
    override = os.environ.get("BENCH_RACE_ARCH_PLATFORM", "").strip()
    if override:
        return override
    try:
        import platform as _pl
        system = _pl.system().lower()
        machine = _pl.machine().lower()
        # Normalise arm64/aarch64 to aarch64 for consistency
        if machine in ("arm64", "aarch64"):
            machine = "aarch64"
        if system == "darwin":
            return f"macos-{machine}"
        return f"{system}-{machine}"
    except Exception:
        return "linux-x86_64"


def _detect_cuda_version() -> str:
    """Return CUDA driver version like '12.8', or '' when no GPU is present."""
    override = os.environ.get("BENCH_RACE_CUDA_VERSION", None)
    if override is not None:
        return override.strip()
    try:
        out = subprocess.check_output(
            ["nvidia-smi"],
            stderr=subprocess.STDOUT,
            timeout=10,
        ).decode(errors="replace")
        import re
        m = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", out)
        if m:
            return m.group(1)
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# Version comparison helper
# ---------------------------------------------------------------------------

def _version_tuple(ver_str: str) -> tuple:
    """Convert a version string like '12.8' to a comparable int tuple."""
    parts = str(ver_str).split(".")
    try:
        return tuple(int(p) for p in parts[:2])
    except ValueError:
        return (0, 0)


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

def resolve(data: dict, platform: str, cuda_ver: str) -> dict:
    """Select the best platform_mappings entry for the given platform+CUDA.

    Entries must be ordered from highest CUDA requirement down to the CPU
    fallback (cuda_min=null).  The first entry whose cuda_min is satisfied is
    returned.

    If the exact platform is not found, a prefix match on the OS portion is
    attempted (e.g. linux-aarch64 -> linux-*).
    """
    platform_mappings = data.get("platform_mappings", {})

    candidates = platform_mappings.get(platform, [])
    if not candidates:
        # Try OS-prefix match (e.g. "linux" matches "linux-x86_64")
        os_prefix = platform.split("-")[0]
        for key, entries in platform_mappings.items():
            if key.startswith(os_prefix):
                candidates = entries
                break

    if not candidates:
        return {}

    cuda_tuple = _version_tuple(cuda_ver) if cuda_ver else (0, 0)

    for entry in candidates:
        cuda_min = entry.get("cuda_min")
        if cuda_min is None:
            # CPU fallback – always matches
            return entry
        if cuda_tuple >= _version_tuple(str(cuda_min)):
            return entry

    return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    compat_file = SCRIPT_DIR / "vllm_compat.json"
    if not compat_file.exists():
        print(f"ERROR: {compat_file} not found", file=sys.stderr)
        return 1

    try:
        data = json.loads(compat_file.read_text())
    except json.JSONDecodeError as exc:
        print(f"ERROR: vllm_compat.json is invalid JSON: {exc}", file=sys.stderr)
        return 1

    platform = _detect_platform()
    cuda_ver = _detect_cuda_version()

    print(f"resolver: platform={platform} cuda={cuda_ver!r}", file=sys.stderr)

    entry = resolve(data, platform, cuda_ver)
    if not entry:
        print(
            f"ERROR: No compatible entry found in platform_mappings for "
            f"platform={platform!r} cuda={cuda_ver!r}",
            file=sys.stderr,
        )
        print("", file=sys.stderr)
        print(
            "Available platforms: "
            + ", ".join(data.get("platform_mappings", {}).keys()),
            file=sys.stderr,
        )
        print(
            "Fallback: set BENCH_RACE_ARCH_PLATFORM to one of the above, "
            "or add a new entry to scripts/vllm_compat.json.",
            file=sys.stderr,
        )
        return 2

    # Emit conda advisory if the resolved entry prefers conda
    if entry.get("conda_preferred") and not os.environ.get("BENCH_RACE_CUDA_VERSION"):
        print(
            f"advisory: conda_preferred=true for {platform}+CUDA {cuda_ver}. "
            "Consider re-running with --use-conda for more reliable wheel availability.",
            file=sys.stderr,
        )

    print(f"resolver: notes={entry.get('notes', '')!r}", file=sys.stderr)

    # Output fields as key=value (empty string for null/missing values)
    fields = [
        "vllm",
        "torch",
        "torchvision",
        "torchaudio",
        "setuptools",
        "torch_index_url",
        "torch_tag",
        "nightly",
        "conda_preferred",
    ]
    for field in fields:
        val = entry.get(field)
        if val is None:
            val = ""
        elif isinstance(val, bool):
            val = "true" if val else "false"
        print(f"{field}={val}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
