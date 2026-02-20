"""Deterministic model-name sanitizer shared by central and agents.

Produces a filesystem-safe name from arbitrary model identifiers such as
``kimi2.5:cloud`` or ``meta-llama/Llama-2-8b-chat-hf``.

Rules:
  1. Replace any character not in ``[A-Za-z0-9._-]`` with ``_``.
  2. Collapse consecutive underscores.
  3. Strip leading/trailing underscores.
  4. Truncate to 90 characters.

The result is safe for use as a directory name, a vLLM model id, and a
symlink target.
"""
from __future__ import annotations

import re

_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._-]")
_MULTI_UNDERSCORE_RE = re.compile(r"_+")


def sanitize_model_name(name: str) -> str:
    """Return a filesystem-safe version of *name*.

    >>> sanitize_model_name("kimi2.5:cloud")
    'kimi2.5_cloud'
    >>> sanitize_model_name("llama3.1:8b-instruct")
    'llama3.1_8b-instruct'
    >>> sanitize_model_name("user/name-with spaces")
    'user_name-with_spaces'
    """
    s = _UNSAFE_RE.sub("_", name)
    s = _MULTI_UNDERSCORE_RE.sub("_", s)
    s = s.strip("_")
    return s[:90]
