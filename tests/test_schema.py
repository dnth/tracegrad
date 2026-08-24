from pathlib import Path

import pytest
from pydantic import ValidationError

from tracegrad.schema import (
    AttributionResult,
    Edit,
    Manifest,
    Report,
    StepVerdict,
    Trace,
)


def test_trace_requires_prompt_hash() -> None:
    with pytest.raises(ValidationError):
        Trace(
            trace_id="trace-1",
            input="hello",
            output="world",
            judge={"score": 1.0, "rationale": "good"},
        )


def test_trace_round_trips_with_nested_judge_and_meta() -> None:
    trace = Trace(
        trace_id="trace-1",
        input="hello",
        output="world",
        judge={"score": 0.75, "rationale": "mostly good"},
        prompt_hash="sha256:abc",
        meta={"model": "test-model"},
    )

    restored = Trace.model_validate_json(trace.model_dump_json())

    assert restored == trace
    assert restored.judge.score == 0.75
    assert restored.judge.rationale == "mostly good"
    assert restored.meta is not None
    assert restored.meta.model == "test-model"


def test_trace_rejects_out_of_range_judge_score() -> None:
    with pytest.raises(ValidationError):
        Trace(
            trace_id="trace-1",
            input="hello",
            output="world",
            judge={"score": 1.5, "rationale": "too generous"},
            prompt_hash="sha256:abc",
        )


def test_manifest_rejects_unsupported_engine(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        Manifest(
            template_file=tmp_path / "prompt.txt",
            engine="jinja-basic",
            judge_fingerprint="judge-v1",
        )


def test_manifest_accepts_supported_engines(tmp_path: Path) -> None:
    for engine in ("none", "format"):
        manifest = Manifest(
            template_file=tmp_path / "prompt.txt",
            engine=engine,
            vars={"name": "Ada"},
            sampling={"temperature": 0.0},
            judge_fingerprint="judge-v1",
        )
        assert manifest.engine.value == engine


def test_attribution_entries_have_declared_sources() -> None:
    attribution = AttributionResult(
        trace_id="trace-1",
        violations=[
            {
                "instruction_id": "i-1",
                "theme_slug": "missing-context",
                "quote": "the answer",
                "quote_source": "output",
            }
        ],
        harmful=[
            {
                "theme_slug": "unsafe-advice",
                "quote": "unsafe output",
                "quote_source": "output",
            }
        ],
    )

    assert attribution.violations[0].theme_slug == "missing-context"
    assert attribution.violations[0].quote_source.value == "output"
    assert attribution.harmful[0].quote_source.value == "output"


def test_edit_report_and_step_verdict_contracts() -> None:
    edit = Edit(
        instruction_id="i-1",
        operation="REWRITE",
        text="Answer with a concise summary.",
        covers_theme="verbosity",
        watch_metric="verbosity",
    )
    report = Report(
        applied_prompt_hash="sha256:prompt",
        clusters=[{"theme": "verbosity", "numerator": 2, "denominator": 10}],
    )
    verdict = StepVerdict(verdict="improved", theme="verbosity")

    assert edit.instruction_id == "i-1"
    assert report.clusters[0].numerator == 2
    assert verdict.verdict.value == "improved"
