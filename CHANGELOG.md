# Changelog

## Unreleased

- Installer: optional sudoers drop-in with sudo -n fail-fast handling; generated agent config now includes central_base_url and fixes ownership when run as root.
- Runtime: non-interactive sudo failures return clearer remediation; ComfyUI history polling is primary completion signal with timeout events.
- Image benchmarks: checkpoint identifiers resolve to filenames (digest fallback supported) and status polling no longer hangs on unknown digests.
- Image sparklines now remain robust when timing fields are missing.
