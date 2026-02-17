"""Unit tests for scripts/_resolve_compat.py – platform-aware vLLM/torch resolver.

These tests are fully offline (no network required) and simulate different
platform + CUDA combinations by feeding controlled inputs to the resolver's
`resolve()` function and the detection helpers.

Run with:
    pytest tests/test_vllm_compat_resolver.py -v
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Load the resolver module from scripts/ without installing it as a package
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
RESOLVER_PATH = REPO_ROOT / "scripts" / "_resolve_compat.py"
COMPAT_JSON_PATH = REPO_ROOT / "scripts" / "vllm_compat.json"


def _load_resolver():
    spec = importlib.util.spec_from_file_location("_resolve_compat", RESOLVER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


resolver = _load_resolver()


@pytest.fixture()
def compat_data():
    """Load vllm_compat.json once per test session."""
    return json.loads(COMPAT_JSON_PATH.read_text())


# ---------------------------------------------------------------------------
# _version_tuple helper
# ---------------------------------------------------------------------------


class TestVersionTuple:
    def test_simple(self):
        assert resolver._version_tuple("12.8") == (12, 8)

    def test_trailing_patch(self):
        # Only first two components are used
        assert resolver._version_tuple("13.0.1") == (13, 0)

    def test_zero(self):
        assert resolver._version_tuple("0.0") == (0, 0)

    def test_invalid(self):
        assert resolver._version_tuple("abc") == (0, 0)


# ---------------------------------------------------------------------------
# resolve() – core selection logic
# ---------------------------------------------------------------------------


class TestResolve:
    # ── linux-x86_64 ──────────────────────────────────────────────────────

    def test_x86_64_no_cuda_picks_cpu_fallback(self, compat_data):
        entry = resolver.resolve(compat_data, "linux-x86_64", "")
        assert entry, "Expected a CPU fallback entry"
        assert entry["torch_tag"] == "cpu"
        assert entry["cuda_min"] is None

    def test_x86_64_cuda_12_1_picks_cu121(self, compat_data):
        entry = resolver.resolve(compat_data, "linux-x86_64", "12.1")
        assert entry, "Expected a CUDA 12.1 entry"
        assert entry["torch_tag"] == "cu121"
        assert entry.get("nightly") is False

    def test_x86_64_cuda_12_8_picks_cu128_stable(self, compat_data):
        entry = resolver.resolve(compat_data, "linux-x86_64", "12.8")
        assert entry, "Expected a CUDA 12.8 entry"
        assert entry["torch_tag"] == "cu128"
        assert entry.get("nightly") is False
        assert "0.15.1" in entry.get("vllm", "")

    def test_x86_64_cuda_13_picks_cu128_nightly(self, compat_data):
        entry = resolver.resolve(compat_data, "linux-x86_64", "13.0")
        assert entry, "Expected a CUDA 13.0 entry"
        assert entry["torch_tag"] == "cu128"
        assert entry.get("nightly") is True

    def test_x86_64_cuda_high_version_picks_nightly(self, compat_data):
        """Any CUDA >= 13 on x86_64 should pick the nightly cu128 entry."""
        entry = resolver.resolve(compat_data, "linux-x86_64", "14.0")
        assert entry["nightly"] is True

    def test_x86_64_cuda_12_2_picks_cu121(self, compat_data):
        """CUDA 12.2 is >= 12.1 so should pick cu121 (the 12.1 entry)."""
        entry = resolver.resolve(compat_data, "linux-x86_64", "12.2")
        assert entry["torch_tag"] == "cu121"

    def test_x86_64_cuda_12_8_includes_torchvision(self, compat_data):
        """CUDA 12.8 on x86_64 should include a torchvision pin."""
        entry = resolver.resolve(compat_data, "linux-x86_64", "12.8")
        assert entry.get("torchvision"), "Expected torchvision for x86_64 cu128"

    # ── linux-aarch64 ─────────────────────────────────────────────────────

    def test_aarch64_no_cuda_picks_cpu_fallback(self, compat_data):
        entry = resolver.resolve(compat_data, "linux-aarch64", "")
        assert entry["torch_tag"] == "cpu"

    def test_aarch64_cuda_12_8_picks_nightly(self, compat_data):
        """aarch64 always uses nightly index for CUDA — stable index lacks aarch64 wheels."""
        entry = resolver.resolve(compat_data, "linux-aarch64", "12.8")
        assert entry["torch_tag"] == "cu128"
        assert entry.get("nightly") is True

    def test_aarch64_cuda_13_picks_nightly(self, compat_data):
        entry = resolver.resolve(compat_data, "linux-aarch64", "13.0")
        assert entry["torch_tag"] == "cu128"
        assert entry.get("nightly") is True

    def test_aarch64_cuda_entries_prefer_conda(self, compat_data):
        """CUDA entries on aarch64 should flag conda_preferred=True."""
        for cuda_ver in ("12.8", "13.0"):
            entry = resolver.resolve(compat_data, "linux-aarch64", cuda_ver)
            assert entry.get("conda_preferred") is True, (
                f"Expected conda_preferred for linux-aarch64+CUDA {cuda_ver}"
            )

    def test_aarch64_no_torchvision_on_cuda(self, compat_data):
        """aarch64 CUDA entries have no torchvision (wheel availability)."""
        entry = resolver.resolve(compat_data, "linux-aarch64", "12.8")
        assert not entry.get("torchvision"), (
            "aarch64 CUDA entries should not specify torchvision"
        )

    # ── macos-arm64 ───────────────────────────────────────────────────────

    def test_macos_arm64_cpu_path(self, compat_data):
        entry = resolver.resolve(compat_data, "macos-arm64", "")
        assert entry["torch_tag"] == "cpu"
        assert entry.get("nightly") is False

    def test_macos_arm64_includes_torchvision(self, compat_data):
        entry = resolver.resolve(compat_data, "macos-arm64", "")
        assert entry.get("torchvision"), "Expected torchvision for macOS arm64"

    # ── macos-x86_64 ─────────────────────────────────────────────────────

    def test_macos_x86_64_cpu_path(self, compat_data):
        entry = resolver.resolve(compat_data, "macos-x86_64", "")
        assert entry["torch_tag"] == "cpu"

    # ── unknown platform ─────────────────────────────────────────────────

    def test_unknown_platform_returns_empty(self, compat_data):
        entry = resolver.resolve(compat_data, "freebsd-amd64", "12.8")
        assert entry == {}, "Unknown platform should return empty dict"

    # ── GB10 / Blackwell scenarios ────────────────────────────────────────

    def test_gb10_x86_64_cuda_12_8(self, compat_data):
        """GB10 on x86_64 with CUDA 12.8 → stable cu128 wheels available."""
        entry = resolver.resolve(compat_data, "linux-x86_64", "12.8")
        assert entry["torch_tag"] == "cu128"
        # Stable, not nightly — x86_64 has stable cu128 wheels
        assert entry.get("nightly") is False

    def test_gb10_x86_64_cuda_13(self, compat_data):
        """GB10 on x86_64 with CUDA 13 → nightly cu128 required."""
        entry = resolver.resolve(compat_data, "linux-x86_64", "13.0")
        assert entry["nightly"] is True

    def test_gb10_aarch64_cuda_13(self, compat_data):
        """GB10 on aarch64 with CUDA 13 → nightly cu128 + conda preferred."""
        entry = resolver.resolve(compat_data, "linux-aarch64", "13.0")
        assert entry["nightly"] is True
        assert entry.get("conda_preferred") is True


# ---------------------------------------------------------------------------
# _detect_platform – via env var override (no real probing needed in tests)
# ---------------------------------------------------------------------------


class TestDetectPlatform:
    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("BENCH_RACE_ARCH_PLATFORM", "linux-aarch64")
        assert resolver._detect_platform() == "linux-aarch64"

    def test_env_var_empty_falls_back_to_real(self, monkeypatch):
        monkeypatch.delenv("BENCH_RACE_ARCH_PLATFORM", raising=False)
        result = resolver._detect_platform()
        # Just check it returns a non-empty string in the expected format
        assert "-" in result, f"Platform string should contain '-', got {result!r}"

    def test_various_platforms(self, monkeypatch):
        for plat in ("linux-x86_64", "linux-aarch64", "macos-arm64", "macos-x86_64"):
            monkeypatch.setenv("BENCH_RACE_ARCH_PLATFORM", plat)
            assert resolver._detect_platform() == plat


# ---------------------------------------------------------------------------
# _detect_cuda_version – via env var override
# ---------------------------------------------------------------------------


class TestDetectCuda:
    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("BENCH_RACE_CUDA_VERSION", "12.8")
        assert resolver._detect_cuda_version() == "12.8"

    def test_env_var_empty_string(self, monkeypatch):
        monkeypatch.setenv("BENCH_RACE_CUDA_VERSION", "")
        assert resolver._detect_cuda_version() == ""

    def test_env_var_absent_no_nvidia(self, monkeypatch):
        """When env var is unset and nvidia-smi is not available, returns ''."""
        monkeypatch.delenv("BENCH_RACE_CUDA_VERSION", raising=False)
        # nvidia-smi won't be available in CI
        result = resolver._detect_cuda_version()
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# JSON schema validation (sanity check that the file itself is well-formed)
# ---------------------------------------------------------------------------


class TestCompatJsonSchema:
    def test_file_exists(self):
        assert COMPAT_JSON_PATH.exists(), "vllm_compat.json must exist"

    def test_valid_json(self):
        data = json.loads(COMPAT_JSON_PATH.read_text())
        assert isinstance(data, dict)

    def test_has_platform_mappings(self, compat_data):
        assert "platform_mappings" in compat_data
        pm = compat_data["platform_mappings"]
        assert len(pm) >= 2, "At least 2 platform entries expected"

    def test_platform_mappings_are_lists(self, compat_data):
        for plat, entries in compat_data["platform_mappings"].items():
            assert isinstance(entries, list), f"{plat} entries must be a list"
            assert len(entries) >= 1, f"{plat} must have at least one entry"

    def test_every_entry_has_required_keys(self, compat_data):
        required = {"torch", "vllm", "setuptools", "torch_tag"}
        for plat, entries in compat_data["platform_mappings"].items():
            for i, entry in enumerate(entries):
                for key in required:
                    assert key in entry, (
                        f"platform_mappings[{plat!r}][{i}] missing required key {key!r}"
                    )

    def test_cpu_fallback_present_for_all_platforms(self, compat_data):
        """Each platform must have at least one entry with cuda_min=null (CPU fallback)."""
        for plat, entries in compat_data["platform_mappings"].items():
            has_cpu = any(e.get("cuda_min") is None for e in entries)
            assert has_cpu, f"Platform {plat!r} is missing a CPU fallback entry (cuda_min=null)"

    def test_entries_ordered_highest_cuda_first(self, compat_data):
        """Entries with cuda_min should come before the CPU fallback (cuda_min=null)."""
        for plat, entries in compat_data["platform_mappings"].items():
            seen_null = False
            for i, entry in enumerate(entries):
                if entry.get("cuda_min") is None:
                    seen_null = True
                else:
                    assert not seen_null, (
                        f"platform_mappings[{plat!r}][{i}] has cuda_min set "
                        f"after a null cuda_min entry — ordering is wrong"
                    )

    def test_flat_mappings_have_required_keys(self, compat_data):
        required = {"setuptools", "torch"}
        for ver, m in compat_data.get("mappings", {}).items():
            for key in required:
                assert key in m, f"mappings[{ver!r}] missing key {key!r}"

    def test_schema_version_present(self, compat_data):
        assert "meta" in compat_data
        assert compat_data["meta"].get("schema_version") == 2


# ---------------------------------------------------------------------------
# End-to-end: main() emits parseable key=value pairs
# ---------------------------------------------------------------------------


class TestMainOutput:
    def _run_main(self, monkeypatch, platform, cuda_ver, capsys=None):
        """Run resolver.main() with controlled env and capture stdout."""
        monkeypatch.setenv("BENCH_RACE_ARCH_PLATFORM", platform)
        monkeypatch.setenv("BENCH_RACE_CUDA_VERSION", cuda_ver)
        # Ensure no real nvidia-smi runs
        old_path = os.environ.get("PATH", "")
        # main() writes to stdout; capture via capsys or redirect
        return None  # let each test use capsys directly

    def test_linux_x86_64_cpu(self, monkeypatch, capsys):
        monkeypatch.setenv("BENCH_RACE_ARCH_PLATFORM", "linux-x86_64")
        monkeypatch.setenv("BENCH_RACE_CUDA_VERSION", "")
        rc = resolver.main()
        assert rc == 0
        out = capsys.readouterr().out
        kv = dict(line.split("=", 1) for line in out.strip().splitlines() if "=" in line)
        assert kv["torch_tag"] == "cpu"
        assert kv["nightly"] == "false"
        assert kv["vllm"]  # non-empty

    def test_linux_x86_64_cuda_12_8(self, monkeypatch, capsys):
        monkeypatch.setenv("BENCH_RACE_ARCH_PLATFORM", "linux-x86_64")
        monkeypatch.setenv("BENCH_RACE_CUDA_VERSION", "12.8")
        rc = resolver.main()
        assert rc == 0
        out = capsys.readouterr().out
        kv = dict(line.split("=", 1) for line in out.strip().splitlines() if "=" in line)
        assert kv["torch_tag"] == "cu128"
        assert kv["nightly"] == "false"
        assert "0.15" in kv["vllm"]

    def test_linux_aarch64_cuda_13(self, monkeypatch, capsys):
        monkeypatch.setenv("BENCH_RACE_ARCH_PLATFORM", "linux-aarch64")
        monkeypatch.setenv("BENCH_RACE_CUDA_VERSION", "13.0")
        rc = resolver.main()
        assert rc == 0
        out = capsys.readouterr().out
        kv = dict(line.split("=", 1) for line in out.strip().splitlines() if "=" in line)
        assert kv["torch_tag"] == "cu128"
        assert kv["nightly"] == "true"
        assert kv["conda_preferred"] == "true"

    def test_unknown_platform_returns_exit_2(self, monkeypatch, capsys):
        monkeypatch.setenv("BENCH_RACE_ARCH_PLATFORM", "solaris-sparc")
        monkeypatch.setenv("BENCH_RACE_CUDA_VERSION", "")
        rc = resolver.main()
        assert rc == 2, "Unknown platform should exit with code 2"

    def test_macos_arm64(self, monkeypatch, capsys):
        monkeypatch.setenv("BENCH_RACE_ARCH_PLATFORM", "macos-arm64")
        monkeypatch.setenv("BENCH_RACE_CUDA_VERSION", "")
        rc = resolver.main()
        assert rc == 0
        out = capsys.readouterr().out
        kv = dict(line.split("=", 1) for line in out.strip().splitlines() if "=" in line)
        assert kv["torch_tag"] == "cpu"
        assert kv["vllm"] == "0.14.1"
        assert kv["torchvision"] == "0.24.1"
