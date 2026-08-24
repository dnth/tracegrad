import json
from pathlib import Path

import pytest

from tracegrad.aggregate import aggregate
from tracegrad.distill import DistillConfig, distill_trace
from tracegrad.inventory import build_inventory
from tracegrad.llm import FakeBackend, LLMError
from tracegrad.schema import AttributionResult, Trace
from tracegrad.synthesize import SynthesisError, render_evidence, synthesize

PROMPT = "1. Always cite sources.\n2. Never speculate.\n"


def _inventory():
    return build_inventory(PROMPT)


def _trace(trace_id: str, output: str) -> Trace:
    return Trace(
        trace_id=trace_id,
        input="What is the capital of France?",
        output=output,
        judge={"score": 0.2, "rationale": "hallucinated, no citation"},
        prompt_hash="sha256:abc",
    )


def _attribution(trace_id: str, theme: str, quote: str) -> AttributionResult:
    return AttributionResult(
        trace_id=trace_id,
        violations=[
            {
                "instruction_id": "1",
                "theme_slug": theme,
                "quote": quote,
                "quote_source": "output",
            }
        ],
    )


# ------------------------------------------------------------------------ render_evidence


def test_render_evidence_includes_counts_and_only_verifying_quotes() -> None:
    trace = _trace("trace-1", "I made this up without a source.")
    distilled = distill_trace(trace, DistillConfig())
    attribution = _attribution("trace-1", "no-citation", "made this up without a source")
    aggregation = aggregate([attribution], denominator=10)

    evidence = render_evidence(aggregation, [attribution], {"trace-1": distilled})

    assert "1/10" in evidence
    assert "made this up without a source" in evidence
    # A confabulated quote that never appeared in the distilled output must not
    # be rendered, even though the theme was attributed.
    confabulated = _attribution("trace-1", "no-citation", "this exact text was never said")
    aggregation_two = aggregate([confabulated], denominator=10)
    evidence_two = render_evidence(aggregation_two, [confabulated], {"trace-1": distilled})
    assert "this exact text was never said" not in evidence_two


# ------------------------------------------------------------------------------ synthesize


def test_synthesize_empty_edit_list_yields_proposed_nothing() -> None:
    inventory = _inventory()
    trace = _trace("trace-1", "I made this up without a source.")
    distilled = distill_trace(trace, DistillConfig())
    attribution = _attribution("trace-1", "no-citation", "made this up without a source")
    aggregation = aggregate([attribution], denominator=10)
    backend = FakeBackend(responses=['{"edits": [], "reasoning": "no strong signal"}'])

    result = synthesize(
        inventory,
        aggregation,
        backend,
        attributions=[attribution],
        distilled={"trace-1": distilled},
    )

    assert result.proposed_nothing
    assert result.outcome.rejected == ()
    assert result.edits == ()


def test_synthesize_reprompts_on_gate_rejection_with_reason_and_writes_autopsy(
    tmp_path: Path,
) -> None:
    inventory = _inventory()
    trace = _trace("trace-1", "I made this up without a source.")
    distilled = distill_trace(trace, DistillConfig())
    # The evidence table will not contain the "unrelated-theme" theme this edit
    # claims to cover, so G4 (no-evidence) rejects it every time.
    attribution = _attribution("trace-1", "no-citation", "made this up without a source")
    aggregation = aggregate([attribution], denominator=10)
    anchor_id = inventory.instructions[0].instruction_id
    bad_edit_response = json.dumps(
        {
            "edits": [
                {
                    "instruction_id": anchor_id,
                    "operation": "REWRITE",
                    "text": "Cite every claim with a source.",
                    "covers_theme": "unrelated-theme",
                    "watch_metric": "unrelated-theme",
                }
            ],
            "reasoning": "trying to fix it",
        }
    )
    backend = FakeBackend(responses=[bad_edit_response, bad_edit_response, bad_edit_response])

    result = synthesize(
        inventory,
        aggregation,
        backend,
        attributions=[attribution],
        distilled={"trace-1": distilled},
        project_root=tmp_path,
        run_id="run-1",
    )

    # 1 initial attempt + 2 re-prompts = 3 calls, never more.
    assert len(backend.calls) == 3
    assert result.rounds == 3
    second_call_user_message = backend.calls[1][1]
    assert "REJECTED BY THE GATES" in second_call_user_message
    assert "unrelated-theme" in second_call_user_message
    assert result.autopsy_path is not None
    assert result.autopsy_path.exists()
    autopsy = json.loads(result.autopsy_path.read_text())
    assert autopsy["rejected"]


def test_synthesize_malformed_json_raises_synthesis_error() -> None:
    inventory = _inventory()
    aggregation = aggregate([], denominator=10)
    backend = FakeBackend(responses=["not json at all"])

    # A response with no JSON in it at all is a synthesis failure to every
    # caller, so it must not escape as the transport-level LLMError that
    # parse_json_response raises.
    with pytest.raises(SynthesisError):
        synthesize(inventory, aggregation, backend)


def test_synthesize_non_json_object_raises_synthesis_error() -> None:
    # A response that *is* valid JSON but not an object takes _parse_edits' own
    # SynthesisError path rather than the parse_json_response one.
    inventory = _inventory()
    aggregation = aggregate([], denominator=10)
    backend = FakeBackend(responses=["[1, 2, 3]"])

    with pytest.raises(SynthesisError):
        synthesize(inventory, aggregation, backend)


def test_synthesize_llm_error_raises_synthesis_error() -> None:
    inventory = _inventory()
    aggregation = aggregate([], denominator=10)

    def handler(system: str, user: str) -> str:
        raise LLMError("backend is down")

    backend = FakeBackend(handler=handler)

    with pytest.raises(SynthesisError):
        synthesize(inventory, aggregation, backend)
