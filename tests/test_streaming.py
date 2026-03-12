"""Unit tests for streaming token delta emission (BPE fragmentation prevention).

Verifies that stream_tokens_as_text buffers token ids and emits proper
incremental text deltas so BPE/wordpiece merges and whitespace markers are
preserved across the full streaming pipeline.
"""
from __future__ import annotations

import importlib


def _get_stream_tokens_as_text():
    server = importlib.import_module("agent.backends.mlx.server")
    return server.stream_tokens_as_text


def test_streaming_delta_with_bpe():
    """Buffered decode must reconstruct correct human-readable text from BPE fragments."""
    stream_tokens_as_text = _get_stream_tokens_as_text()

    # A minimal stub tokenizer that simulates BPE pieces.
    # The Ġ prefix (U+0120) is the GPT-2/LLaMA convention for a leading space.
    class StubTokenizer:
        def __init__(self):
            self.vocab = {
                1: "Here",
                2: "Ġare",
                3: "Ġthe",
                4: "Ġkey",
                5: "Ġdif",
                6: "fer",
                7: "ences",
            }

        def decode(self, ids, skip_special_tokens=True):
            return "".join(self.vocab[i] for i in ids).replace("Ġ", " ")

    emitted = []

    def emit(x):
        emitted.append(x)

    generator = iter([1, 2, 3, 4, 5, 6, 7])
    stream_tokens_as_text(generator, StubTokenizer(), emit)

    # The concatenated deltas must reconstruct the full phrase without any
    # BPE fragment leak (e.g. "dif" and "fer" must merge into " differ").
    assert "".join(emitted).strip() == "Here are the key differences"


def test_streaming_delta_preserves_leading_whitespace():
    """Leading spaces on tokens must not be stripped — they are semantically significant."""
    stream_tokens_as_text = _get_stream_tokens_as_text()

    class SpaceTokenizer:
        """Every token after the first is prefixed with a space."""
        _words = ["The", " quick", " brown", " fox"]

        def decode(self, ids, skip_special_tokens=True):
            return "".join(self._words[i - 1] for i in ids)

    emitted = []
    stream_tokens_as_text(iter([1, 2, 3, 4]), SpaceTokenizer(), emitted.append)

    # Leading spaces must survive intact so clients render correct spacing
    assert emitted[1] == " quick"
    assert emitted[2] == " brown"
    assert emitted[3] == " fox"
    assert "".join(emitted) == "The quick brown fox"


def test_streaming_delta_empty_generator():
    """An empty token stream must not call emit_fn at all."""
    stream_tokens_as_text = _get_stream_tokens_as_text()

    class DummyTokenizer:
        def decode(self, ids, skip_special_tokens=True):
            return ""

    emitted = []
    stream_tokens_as_text(iter([]), DummyTokenizer(), emitted.append)
    assert emitted == []


def test_streaming_delta_single_token():
    """A single token must be emitted as-is without fragmentation."""
    stream_tokens_as_text = _get_stream_tokens_as_text()

    class SingleTokenizer:
        def decode(self, ids, skip_special_tokens=True):
            return "hello"

    emitted = []
    stream_tokens_as_text(iter([42]), SingleTokenizer(), emitted.append)
    assert emitted == ["hello"]


def test_streaming_delta_fallback_on_non_prefix():
    """If decoded text is not a prefix of previous text, emit full text as fallback."""
    stream_tokens_as_text = _get_stream_tokens_as_text()

    call_count = [0]

    class NormalisingTokenizer:
        """Simulates a tokenizer that may normalise text unexpectedly."""
        _responses = ["hello", "world"]  # non-cumulative (abnormal tokenizer)

        def decode(self, ids, skip_special_tokens=True):
            idx = len(ids) - 1
            return self._responses[idx] if idx < len(self._responses) else ""

    emitted = []
    stream_tokens_as_text(iter([1, 2]), NormalisingTokenizer(), emitted.append)

    # Both deltas must be non-empty
    assert len(emitted) == 2
    assert all(e for e in emitted)
