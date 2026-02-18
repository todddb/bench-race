# Local-Only Install Mode

The `--local-only` flag for `install_agent.sh` keeps all files within the
repo directory or the user's home directory. No system-level paths (`/etc`,
`/opt`, `/Library`) are touched and `sudo` is never invoked for service
installation.

## Quick Start

```bash
# Preview what will be created (no writes)
./scripts/install_agent.sh --preflight --local-only

# Install (non-root only)
./scripts/install_agent.sh --local-only

# Start the agent (respects local-only automatically)
./scripts/agents start
```

## New CLI Flags

| Flag | Description |
|------|-------------|
| `--local-only` | Force all paths to user-local locations; refuse sudo/system writes |
| `--preflight` | Print planned paths and exit without writing anything |
| `--uninstall-all` | Remove all managed artifacts (venvs, models, services, logs, pids) |
| `--confirm` | Non-interactive confirmation for `--uninstall-all` (same as `-y`) |

## What `--local-only` Does

- Sets `VENV_PATH` to `~/bench-race/vllm-venv` (unless overridden with `--venv-path`)
- Sets `MODEL_DIR` to `~/bench-race/models/vllm` (unless overridden with `--model-dir`)
- Installs systemd units as **user** services (`~/.config/systemd/user/`) or
  macOS LaunchAgents (`~/Library/LaunchAgents/`) instead of system-level
- Writes a marker file (`.bench-race-local-only`) in the repo root so that
  `scripts/agents` automatically avoids `sudo` and system-level `systemctl`
- Refuses to run as root (pass `--system` explicitly for system-wide installs)

## Preflight

Check what will be created before running:

```bash
./scripts/install_agent.sh --preflight --local-only
```

Example output:

```
[PREFLIGHT] Mode:             local-only
[PREFLIGHT] VENV_PATH:        /home/you/bench-race/vllm-venv
[PREFLIGHT] MODEL_DIR:        /home/you/bench-race/models/vllm
[PREFLIGHT] Agent venv:       /path/to/repo/agent/.venv
[PREFLIGHT] Agent config:     /path/to/repo/agent/config/agent.yaml
[PREFLIGHT] SERVICE_TARGET:   user (systemd user)
[PREFLIGHT] SERVICE_PATH:     ~/.config/systemd/user/bench-race-vllm.service
[PREFLIGHT] Will NOT write to /etc, /opt, or /Library (local-only mode).
```

## Uninstall / Cleanup

Remove everything the installer created:

```bash
# Interactive (asks for confirmation)
./scripts/install_agent.sh --uninstall-all --local-only

# Non-interactive
./scripts/install_agent.sh --uninstall-all --local-only --confirm
```

This removes:

- vLLM virtual environment (`VENV_PATH`)
- Agent virtual environment (`agent/.venv`)
- Model directory (`MODEL_DIR`)
- User-level systemd units / LaunchAgents
- PID files (`run/`)
- Log files (`logs/`)
- The `.bench-race-local-only` marker
- Conda env `bench-race-vllm` (if created by this installer)

To also remove system-level artifacts (created by `--system`):

```bash
sudo ./scripts/install_agent.sh --uninstall-all --system --confirm
```

### Verify cleanup

```bash
# Check for system-wide files
ls -l /etc/systemd/system/bench-race-vllm.service 2>/dev/null || echo "no systemd system unit found"
ls -l ~/.config/systemd/user/bench-race-vllm.service 2>/dev/null || echo "no user unit found"
ls -ld /opt/bench-race* 2>/dev/null || echo "no /opt install found"
```

## Root Safety Guard

Running as root without `--system` is blocked:

```bash
# This will be REFUSED with a helpful message:
sudo ./scripts/install_agent.sh

# These are allowed:
sudo ./scripts/install_agent.sh --system        # explicit system install
./scripts/install_agent.sh --local-only          # as non-root user
```

## System Install (unchanged behavior)

The existing `--system` flag continues to work identically:

```bash
sudo ./scripts/install_agent.sh --system
```

This writes to `/opt/bench-race/vllm-venv`, `/etc/systemd/system/`, etc.

## Running Tests

```bash
./scripts/test_local_install.sh
```

This runs acceptance tests verifying preflight output, flag conflicts,
uninstall behavior, and local-only marker detection.
