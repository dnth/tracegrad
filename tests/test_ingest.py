import json
from pathlib import Path
from typing import Any

import pytest

from tracegrad.ingest import IngestError, ingest_traces, read_trace_lines
from tracegrad.schema import Manifest, Trace

GOOD_RATIONALE = "This rationale is long enough and explains the judge's reasoning clearly."


def _trace_dict(
    trace_id: str = "trace-1",
    prompt_hash: str = "sha256:p1",
    rationale: str = GOOD_RATIONALE,
    score: float = 0.9,
    model: str | None = "gpt-4",
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "trace_id": trace_id,
        "input": "some input",
        "output": "some output",
        "judge": {"score": score, "rationale": rationale},
        "prompt_hash": prompt_hash,
    }
    if model is not None:
        record["meta"] = {"model": model}
    return record


# --- read_trace_lines -------------------------------------------------------


def test_read_trace_lines_parses_and_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "traces.jsonl"
    path.write_text('{"a":1}\n\n{"b":2}\n', encoding="utf-8")

    assert read_trace_lines(path) == [(1, {"a": 1}), (3, {"b": 2})]


def test_read_trace_lines_raises_ingest_error_on_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"a":1}\nnot json\n', encoding="utf-8")

    with pytest.raises(IngestError, match="malformed JSON"):
        read_trace_lines(path)


def test_read_trace_lines_raises_ingest_error_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="could not read traces"):
        read_trace_lines(tmp_path / "missing.jsonl")


# --- schema validation and dropping -----------------------------------------


def test_ingest_traces_drops_non_object_and_invalid_records() -> None:
    records: list[tuple[int, object]] = [
        (1, ["not", "an", "object"]),
        (2, {"trace_id": "t1"}),  # missing required fields
    ]

    result = ingest_traces(records)

    assert result.accepted_count == 0
    assert {drop.reason for drop in result.dropped} == {"invalid-schema"}
    assert result.dropped_reasons["invalid-schema"] == 2


def test_ingest_traces_drops_duplicate_trace_id() -> None:
    records = [(1, _trace_dict(trace_id="t1")), (2, _trace_dict(trace_id="t1"))]

    result = ingest_traces(records)

    assert result.accepted_count == 1
    assert len(result.dropped) == 1
    assert result.dropped[0].reason == "duplicate-trace-id"
    assert result.dropped[0].trace_id == "t1"


def test_ingest_traces_accepts_pre_validated_trace_objects() -> None:
    trace = Trace.model_validate(_trace_dict(trace_id="t1"))

    result = ingest_traces([trace])

    assert result.accepted_count == 1
    assert result.dropped == ()


def test_ingest_traces_reads_from_jsonl_path(tmp_path: Path) -> None:
    path = tmp_path / "batch.jsonl"
    lines = [json.dumps(_trace_dict(trace_id=f"t{i}")) for i in range(2)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = ingest_traces(path)

    assert result.accepted_count == 2


# --- rationale quality floor -------------------------------------------------


def test_ingest_traces_drops_short_rationale() -> None:
    records = [(1, _trace_dict(trace_id="t1", rationale="too short"))]

    result = ingest_traces(records)

    assert result.accepted_count == 0
    assert result.dropped[0].reason == "rationale-below-quality-floor"


def test_ingest_traces_drops_punctuation_only_rationale() -> None:
    records = [(1, _trace_dict(trace_id="t1", rationale="-" * 30))]

    result = ingest_traces(records)

    assert result.accepted_count == 0
    assert result.dropped[0].reason == "rationale-below-quality-floor"


# --- prompt_hash partitioning -------------------------------------------------


def test_ingest_traces_keeps_dominant_prompt_hash_partition() -> None:
    records = [
        (1, _trace_dict(trace_id="t1", prompt_hash="sha256:p1")),
        (2, _trace_dict(trace_id="t2", prompt_hash="sha256:p1")),
        (3, _trace_dict(trace_id="t3", prompt_hash="sha256:p2")),
    ]

    result = ingest_traces(records)

    assert result.prompt_hash == "sha256:p1"
    assert result.accepted_count == 2
    assert result.prompt_hash_partitions == {"sha256:p1": 2, "sha256:p2": 1}
    minority_drops = [d for d in result.dropped if d.reason == "prompt-hash-partition"]
    assert len(minority_drops) == 1
    assert minority_drops[0].trace_id == "t3"


def test_ingest_traces_breaks_partition_tie_by_sort_order() -> None:
    records = [
        (1, _trace_dict(trace_id="t1", prompt_hash="sha256:zzz")),
        (2, _trace_dict(trace_id="t2", prompt_hash="sha256:aaa")),
    ]

    result = ingest_traces(records)

    assert result.prompt_hash == "sha256:aaa"
    assert result.accepted_count == 1


# --- model partitions ---------------------------------------------------------


def test_ingest_traces_counts_model_partitions_and_flags_mixed() -> None:
    records = [
        (1, _trace_dict(trace_id="t1", model="gpt-4")),
        (2, _trace_dict(trace_id="t2", model="gpt-4")),
        (3, _trace_dict(trace_id="t3", model="claude")),
    ]

    result = ingest_traces(records)

    assert result.model_partitions == {"gpt-4": 2, "claude": 1}
    assert result.is_mixed_model is True


def test_ingest_traces_single_model_is_not_mixed() -> None:
    records = [
        (1, _trace_dict(trace_id="t1", model="gpt-4")),
        (2, _trace_dict(trace_id="t2", model="gpt-4")),
    ]

    result = ingest_traces(records)

    assert result.model_partitions == {"gpt-4": 2}
    assert result.is_mixed_model is False


def test_ingest_traces_unspecified_model_partition() -> None:
    records = [(1, _trace_dict(trace_id="t1", model=None))]

    result = ingest_traces(records)

    assert result.model_partitions == {"unspecified": 1}


# --- judge fingerprint ---------------------------------------------------------


def test_ingest_traces_detects_judge_fingerprint_change(tmp_path: Path) -> None:
    manifest = Manifest(template_file=tmp_path / "t.txt", judge_fingerprint="judge-v2")

    result = ingest_traces([], manifest=manifest, previous_judge_fingerprint="judge-v1")

    assert result.judge_fingerprint == "judge-v2"
    assert result.previous_judge_fingerprint == "judge-v1"
    assert result.judge_fingerprint_changed is True


def test_ingest_traces_no_change_when_fingerprints_match(tmp_path: Path) -> None:
    manifest = Manifest(template_file=tmp_path / "t.txt", judge_fingerprint="judge-v1")

    result = ingest_traces([], manifest=manifest, previous_judge_fingerprint="judge-v1")

    assert result.judge_fingerprint_changed is False


def test_ingest_traces_no_manifest_means_no_fingerprint() -> None:
    result = ingest_traces([])

    assert result.judge_fingerprint is None
    assert result.judge_fingerprint_changed is False


# --- canary failures ------------------------------------------------------------


def test_ingest_traces_canary_failure_outside_tolerance() -> None:
    records = [(1, _trace_dict(trace_id="t1", score=0.9))]

    result = ingest_traces(records, canary_scores={"t1": 0.5}, canary_tolerance=0.1)

    assert len(result.canary_failures) == 1
    failure = result.canary_failures[0]
    assert failure.trace_id == "t1"
    assert failure.expected == 0.5
    assert failure.observed == 0.9
    assert failure.tolerance == 0.1


def test_ingest_traces_canary_within_tolerance_has_no_failure() -> None:
    records = [(1, _trace_dict(trace_id="t1", score=0.9))]

    result = ingest_traces(records, canary_scores={"t1": 0.85}, canary_tolerance=0.1)

    assert result.canary_failures == ()


# --- dropped_reasons -------------------------------------------------------------


def test_ingest_traces_dropped_reasons_counter() -> None:
    records: list[tuple[int, object]] = [
        (1, {"bad": "shape"}),
        (2, _trace_dict(trace_id="t1", rationale="short")),
    ]

    result = ingest_traces(records)

    assert result.dropped_reasons["invalid-schema"] == 1
    assert result.dropped_reasons["rationale-below-quality-floor"] == 1
