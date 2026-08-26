"""Snapshot reuse for the Kitaru source (ADR 0004)."""

from __future__ import annotations

from tracegrad.canonical import text_hash
from tracegrad.integrations.kitaru.accounting import format_source_table
from tracegrad.integrations.kitaru.mapping import MappedTrace, SourceDrop
from tracegrad.integrations.kitaru.snapshot import (
    SourceFingerprint,
    SourceMeta,
    fingerprints_compatible,
    load_fingerprint,
    load_source_drops,
    snapshot_exists,
    write_snapshot,
)
from tracegrad.schema import Trace
from tracegrad.state import initialize


def test_snapshot_round_trips_and_is_reusable(tmp_path) -> None:
    layout = initialize(tmp_path)
    trace = Trace(
        trace_id="s1",
        input="q",
        output="a",
        judge={"score": 0.2, "rationale": "needs a citation in the answer now"},
        prompt_hash=text_hash("prompt"),
    )
    mapped = [
        MappedTrace(
            trace=trace,
            session_number=12,
            evaluator_name="quality",
            evaluator_version=3,
            evaluator_version_id="ev",
            multi_turn=False,
            system_prompt="prompt",
        )
    ]
    dropped = [SourceDrop("s2", "system-prompt-unavailable", number=13)]
    fingerprint = SourceFingerprint(
        cohort_id="c",
        cohort_version_id="cv",
        evaluation_name="quality",
        evaluator_id="eid",
        evaluator_version=3,
        agent_id="a",
    )
    meta = SourceMeta(
        cohort_name="support-production",
        traces_mapped=1,
        sessions_selected=2,
        evaluator_name="quality",
    )
    write_snapshot(layout, fingerprint=fingerprint, meta=meta, mapped=mapped, dropped=dropped)

    assert snapshot_exists(layout, "cv")
    stored = load_fingerprint(layout, "cv")
    assert stored.mapping_version == 1
    assert stored.source == "kitaru"
    assert fingerprints_compatible(
        stored, {"evaluation_name": "quality", "cohort_version_id": "cv"}
    )
    assert not fingerprints_compatible(
        stored, {"evaluation_name": "other", "cohort_version_id": "cv"}
    )
    drops = load_source_drops(layout, "cv")
    assert drops[0].reason == "system-prompt-unavailable"
    batch = (layout.sources / "kitaru" / "cv" / "batch.jsonl").read_text(encoding="utf-8")
    assert "s1" in batch


def test_source_and_batch_tables_are_not_merged() -> None:
    table = format_source_table(
        sessions_selected=10,
        traces_mapped=8,
        dropped=[
            SourceDrop("a", "system-prompt-unavailable"),
            SourceDrop("b", "judge-rationale-missing"),
        ],
        in_batch=6,
        batch_drops={"prompt-hash-partition": 2},
    )
    assert "Sessions selected" in table
    assert "Traces mapped" in table
    assert "In batch" in table
    assert "system-prompt-unavailable" in table
    assert "prompt-hash-partition" in table
    # The two tables stay labeled separately rather than summed.
    assert table.index("Traces mapped") < table.index("In batch")
