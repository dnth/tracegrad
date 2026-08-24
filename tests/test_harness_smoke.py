"""An opt-in round trip against a real local `claude` harness.

Skipped by default: it costs a real model call and needs the user to be logged
in.  Run it with ``TRACEGRAD_HARNESS_SMOKE=1 uv run pytest -m harness`` when
changing the harness preset, because the isolation flags and the JSON envelope
shape are exactly the things a unit test with an injected runner cannot pin.
"""

from __future__ import annotations

import os
import shutil

import pytest

from tracegrad.llm import CommandBackend, LLMError

pytestmark = [
    pytest.mark.harness,
    pytest.mark.skipif(
        os.environ.get("TRACEGRAD_HARNESS_SMOKE") != "1",
        reason="set TRACEGRAD_HARNESS_SMOKE=1 to run the real harness round trip",
    ),
    pytest.mark.skipif(shutil.which("claude") is None, reason="the claude CLI is not installed"),
]


def test_the_isolated_claude_preset_completes_and_returns_json() -> None:
    backend = CommandBackend.claude(timeout=120.0)

    completion = backend.complete(
        'Reply with JSON only: {"ok": true}. No prose, no code fence.',
        "Reply now.",
    )

    assert completion.backend == "claude"
    assert completion.json() == {"ok": True}


def test_an_error_envelope_surfaces_as_an_llm_error() -> None:
    backend = CommandBackend.claude(timeout=120.0)
    backend.command = (*backend.command, "--model", "not-a-real-model-id")

    with pytest.raises(LLMError):
        backend.complete("Reply with anything.", "Reply now.")
