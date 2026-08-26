"""Snapshot reuse for the Kitaru source (ADR 0004)."""

from __future__ import annotations

from pathlib import Path

from tracegrad.canonical import text_hash
from tracegrad.integrations.kitaru.accounting import format_source_table
from tracegrad.integrations.kitaru.client import CohortResolution
from tracegrad.integrations.kitaru.mapping import MappedTrace, SourceDrop
from tracegrad.integrations.kitaru.snapshot import (
    LATEST_POINTER,
    SourceFingerprint,
    SourceMeta,
    find_local_snapshot,
    fingerprints_compatible,
    kitaru_root,
    load_fingerprint,
    load_latest_pointer,
    load_source_drops,
    snapshot_exists,
    write_snapshot,
)
from tracegrad.integrations.kitaru.source import prepare_kitaru_source
from tracegrad.schema import Manifest, Trace
from tracegrad.state import initialize


def _sample_trace() -> Trace:
    return Trace(
        trace_id="s1",
        input="q",
        output="a",
        judge={"score": 0.2, "rationale": "needs a citation in the answer now"},
        prompt_hash=text_hash("prompt"),
    )


def _sample_mapped() -> list[MappedTrace]:
    return [
        MappedTrace(
            trace=_sample_trace(),
            session_number=12,
            evaluator_name="quality",
            evaluator_version=3,
            evaluator_version_id="ev",
            multi_turn=False,
            system_prompt="prompt",
        )
    ]


def _sample_fingerprint(**overrides: object) -> SourceFingerprint:
    payload = dict(
        cohort_id="c",
        cohort_version_id="cv",
        evaluation_name="quality",
        evaluator_id="eid",
        evaluator_version=3,
        agent_id="a",
    )
    payload.update(overrides)
    return SourceFingerprint.model_validate(payload)


def _sample_meta(**overrides: object) -> SourceMeta:
    payload = dict(
        cohort_name="support-production",
        display_version="week-34",
        traces_mapped=1,
        sessions_selected=2,
        evaluator_name="quality",
    )
    payload.update(overrides)
    return SourceMeta.model_validate(payload)


def _manifest() -> Manifest:
    return Manifest(template_file=Path("prompt.md"), judge_fingerprint="quality@3")


def test_snapshot_round_trips_and_is_reusable(tmp_path) -> None:
    layout = initialize(tmp_path)
    mapped = _sample_mapped()
    dropped = [SourceDrop("s2", "system-prompt-unavailable", number=13)]
    fingerprint = _sample_fingerprint()
    meta = _sample_meta()
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
    assert not fingerprints_compatible(
        stored, {"evaluation_name": "quality", "cohort_version_id": "other"}
    )
    drops = load_source_drops(layout, "cv")
    assert drops[0].reason == "system-prompt-unavailable"
    batch = (layout.sources / "kitaru" / "cv" / "batch.jsonl").read_text(encoding="utf-8")
    assert "s1" in batch
    pointer = load_latest_pointer(layout)
    assert pointer is not None
    assert pointer["cohort_version_id"] == "cv"
    assert pointer["cohort_name"] == "support-production"
    assert pointer["evaluation_name"] == "quality"
    assert pointer["display_version"] == "week-34"
    assert find_local_snapshot(
        layout, cohort_name="support-production", evaluation_name="quality"
    ) == "cv"


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


def test_prepare_reuses_latest_pointer_without_the_server(tmp_path) -> None:
    layout = initialize(tmp_path)
    write_snapshot(
        layout,
        fingerprint=_sample_fingerprint(),
        meta=_sample_meta(),
        mapped=_sample_mapped(),
        dropped=[],
    )
    prepared = prepare_kitaru_source(
        project_root=tmp_path,
        manifest=_manifest(),
        cohort_name="support-production",
        evaluation_name="quality",
    )
    assert prepared.refreshed is False
    assert prepared.fingerprint.cohort_version_id == "cv"
    assert prepared.traces_path == layout.sources / "kitaru" / "cv" / "batch.jsonl"


def test_prepare_does_not_follow_moved_remote_latest(tmp_path) -> None:
    layout = initialize(tmp_path)
    write_snapshot(
        layout,
        fingerprint=_sample_fingerprint(),
        meta=_sample_meta(),
        mapped=_sample_mapped(),
        dropped=[],
    )

    class MovingLatest:
        resolved = 0

        async def resolve_cohort(self, name: str, version: str | None = None) -> CohortResolution:
            type(self).resolved += 1
            return CohortResolution(
                cohort_id="c",
                cohort_name=name,
                cohort_version_id="cv-new",
                display_version="week-35",
                version_number=2,
                agent_id="a",
                session_count=1,
            )

        async def close(self) -> None:
            return None

    prepared = prepare_kitaru_source(
        project_root=tmp_path,
        manifest=_manifest(),
        cohort_name="support-production",
        evaluation_name="quality",
        gateway=MovingLatest(),
    )
    assert MovingLatest.resolved == 0
    assert prepared.fingerprint.cohort_version_id == "cv"
    assert prepared.refreshed is False


def test_prepare_reuses_explicit_version_and_display_version(tmp_path) -> None:
    layout = initialize(tmp_path)
    write_snapshot(
        layout,
        fingerprint=_sample_fingerprint(),
        meta=_sample_meta(),
        mapped=_sample_mapped(),
        dropped=[],
    )
    by_id = prepare_kitaru_source(
        project_root=tmp_path,
        manifest=_manifest(),
        cohort_name="support-production",
        evaluation_name="quality",
        cohort_version="cv",
    )
    by_display = prepare_kitaru_source(
        project_root=tmp_path,
        manifest=_manifest(),
        cohort_name="support-production",
        evaluation_name="quality",
        cohort_version="week-34",
    )
    assert by_id.fingerprint.cohort_version_id == "cv"
    assert by_display.fingerprint.cohort_version_id == "cv"
    assert by_id.refreshed is False
    assert by_display.refreshed is False


def test_prepare_refresh_fetches_even_when_latest_exists(tmp_path) -> None:
    layout = initialize(tmp_path)
    write_snapshot(
        layout,
        fingerprint=_sample_fingerprint(),
        meta=_sample_meta(),
        mapped=_sample_mapped(),
        dropped=[],
    )

    class FetchGateway:
        def __init__(self) -> None:
            self.resolved = 0

        async def resolve_cohort(self, name: str, version: str | None = None) -> CohortResolution:
            self.resolved += 1
            return CohortResolution(
                cohort_id="c",
                cohort_name=name,
                cohort_version_id="cv-new",
                display_version="week-35",
                version_number=2,
                agent_id="a",
                session_count=0,
            )

        async def list_sessions(self, cohort_version_id: str) -> list[object]:
            return []

        async def fetch_records(self, sessions: list[object]) -> list[object]:
            return []

        async def evaluator_id(self, name: str) -> str:
            return "eid"

        async def close(self) -> None:
            return None

    gateway = FetchGateway()
    prepared = prepare_kitaru_source(
        project_root=tmp_path,
        manifest=_manifest(),
        cohort_name="support-production",
        evaluation_name="quality",
        refresh=True,
        gateway=gateway,
    )
    assert gateway.resolved == 1
    assert prepared.refreshed is True
    assert prepared.fingerprint.cohort_version_id == "cv-new"
    pointer = load_latest_pointer(layout)
    assert pointer is not None
    assert pointer["cohort_version_id"] == "cv-new"


def test_old_latest_pointer_without_extra_fields_still_reuses(tmp_path) -> None:
    layout = initialize(tmp_path)
    write_snapshot(
        layout,
        fingerprint=_sample_fingerprint(),
        meta=_sample_meta(),
        mapped=_sample_mapped(),
        dropped=[],
    )
    pointer_path = kitaru_root(layout) / LATEST_POINTER
    pointer_path.write_text(
        '{"cohort_version_id": "cv", "batch": "ignored"}',
        encoding="utf-8",
    )
    prepared = prepare_kitaru_source(
        project_root=tmp_path,
        manifest=_manifest(),
        cohort_name="support-production",
        evaluation_name="quality",
    )
    assert prepared.fingerprint.cohort_version_id == "cv"
    assert prepared.refreshed is False
