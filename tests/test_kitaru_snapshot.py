"""Snapshot reuse for the Kitaru source (ADR 0004)."""

from __future__ import annotations

import asyncio
import inspect
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tracegrad.canonical import text_hash
from tracegrad.integrations.kitaru.accounting import format_source_table
from tracegrad.integrations.kitaru.client import CohortResolution, KitaruGateway
from tracegrad.integrations.kitaru.errors import KitaruSourceError
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
    load_meta,
    load_source_drops,
    snapshot_exists,
    snapshot_key,
    write_snapshot,
)
from tracegrad.integrations.kitaru.source import prepare_kitaru_source
from tracegrad.schema import Manifest, Trace
from tracegrad.state import initialize


def _sample_mapped(
    *,
    trace_id: str = "s1",
    evaluator_name: str = "quality",
    evaluator_version: int = 3,
) -> list[MappedTrace]:
    trace = Trace(
        trace_id=trace_id,
        input="q",
        output="a",
        judge={"score": 0.2, "rationale": "needs a citation in the answer now"},
        prompt_hash=text_hash("prompt"),
    )
    return [
        MappedTrace(
            trace=trace,
            session_number=12,
            evaluator_name=evaluator_name,
            evaluator_version=evaluator_version,
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
        version_number=1,
        traces_mapped=1,
        sessions_selected=2,
        evaluator_name="quality",
    )
    payload.update(overrides)
    return SourceMeta.model_validate(payload)


def _manifest(judge_fingerprint: str = "quality@3") -> Manifest:
    return Manifest(template_file=Path("prompt.md"), judge_fingerprint=judge_fingerprint)


def test_snapshot_round_trips_and_is_reusable(tmp_path) -> None:
    layout = initialize(tmp_path)
    mapped = _sample_mapped()
    dropped = [SourceDrop("s2", "system-prompt-unavailable", number=13)]
    fingerprint = _sample_fingerprint()
    meta = _sample_meta()
    write_snapshot(layout, fingerprint=fingerprint, meta=meta, mapped=mapped, dropped=dropped)

    assert snapshot_exists(layout, snapshot_key("cv", "quality"))
    stored = load_fingerprint(layout, snapshot_key("cv", "quality"))
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
    drops = load_source_drops(layout, snapshot_key("cv", "quality"))
    assert drops[0].reason == "system-prompt-unavailable"
    batch = (
        layout.sources / "kitaru" / "cv" / "quality" / "batch.jsonl"
    ).read_text(encoding="utf-8")
    assert "s1" in batch
    pointer = load_latest_pointer(
        layout, cohort_name="support-production", evaluation_name="quality"
    )
    assert pointer is not None
    assert pointer["cohort_version_id"] == "cv"
    assert pointer["snapshot_id"] == snapshot_key("cv", "quality")
    assert pointer["cohort_name"] == "support-production"
    assert pointer["evaluation_name"] == "quality"
    assert pointer["display_version"] == "week-34"
    stored_meta = load_meta(layout, snapshot_key("cv", "quality"))
    assert stored_meta.version_number == 1
    assert pointer["version_number"] == 1
    assert find_local_snapshot(
        layout, cohort_name="support-production", evaluation_name="quality"
    ) == snapshot_key("cv", "quality")
    assert find_local_snapshot(
        layout,
        cohort_name="support-production",
        evaluation_name="quality",
        cohort_version="1",
    ) == snapshot_key("cv", "quality")
    assert (
        find_local_snapshot(
            layout,
            cohort_name="support-production",
            evaluation_name="quality",
            cohort_version="2",
        )
        is None
    )


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
    assert prepared.traces_path == layout.sources / "kitaru" / "cv" / "quality" / "batch.jsonl"


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
    by_number = prepare_kitaru_source(
        project_root=tmp_path,
        manifest=_manifest(),
        cohort_name="support-production",
        evaluation_name="quality",
        cohort_version="1",
    )
    assert by_id.fingerprint.cohort_version_id == "cv"
    assert by_display.fingerprint.cohort_version_id == "cv"
    assert by_number.fingerprint.cohort_version_id == "cv"
    assert by_id.refreshed is False
    assert by_display.refreshed is False
    assert by_number.refreshed is False


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
    pointer = load_latest_pointer(
        layout, cohort_name="support-production", evaluation_name="quality"
    )
    assert pointer is not None
    assert pointer["cohort_version_id"] == "cv-new"
    assert pointer["snapshot_id"] == snapshot_key("cv-new", "quality")


def test_refresh_fingerprint_conflict_does_not_clobber_last_good_snapshot(tmp_path) -> None:
    layout = initialize(tmp_path)
    write_snapshot(
        layout,
        fingerprint=_sample_fingerprint(),
        meta=_sample_meta(),
        mapped=_sample_mapped(),
        dropped=[],
    )
    previous_batch = (
        layout.sources / "kitaru" / "cv" / "quality" / "batch.jsonl"
    ).read_text(encoding="utf-8")

    session = SimpleNamespace(id="0f3a0000-0000-4000-8000-00000000c19d", number=12)
    node = SimpleNamespace(
        index=0,
        parent_index=None,
        secondary_parent_indexes=[],
        node_type="llm_call",
        inputs={"input": "q", "system": "prompt"},
        outputs={"output": "a"},
        input_text_selector="/input",
        output_text_selector="/output",
        system_prompt_selector="/system",
        model="gpt-4.1",
    )
    evaluation = SimpleNamespace(
        id="e1",
        name="quality",
        score=0.2,
        explanation="needs a citation in the answer now",
        passed=False,
        value=None,
        data_type="float",
        evaluator_name="quality",
        evaluator_version=9,
        evaluator_version_id="ev-9",
    )

    class ConflictGateway:
        async def resolve_cohort(self, name: str, version: str | None = None) -> CohortResolution:
            return CohortResolution(
                cohort_id="c",
                cohort_name=name,
                cohort_version_id="cv-new",
                display_version="week-35",
                version_number=2,
                agent_id="a",
                session_count=1,
            )

        async def list_sessions(self, cohort_version_id: str) -> list[object]:
            return [session]

        async def fetch_records(self, sessions: list[object]) -> list[object]:
            return [(sessions[0], [node], [evaluation])]

        async def evaluator_id(self, name: str) -> str:
            return "eid-new"

        async def close(self) -> None:
            return None

    with pytest.raises(Exception, match="judge-fingerprint-conflict"):
        prepare_kitaru_source(
            project_root=tmp_path,
            manifest=_manifest("quality@3"),
            cohort_name="support-production",
            evaluation_name="quality",
            refresh=True,
            gateway=ConflictGateway(),
        )

    pointer = load_latest_pointer(
        layout, cohort_name="support-production", evaluation_name="quality"
    )
    assert pointer is not None
    assert pointer["cohort_version_id"] == "cv"
    assert pointer["snapshot_id"] == snapshot_key("cv", "quality")
    assert snapshot_exists(layout, snapshot_key("cv", "quality"))
    assert not snapshot_exists(layout, snapshot_key("cv-new", "quality"))
    assert (
        layout.sources / "kitaru" / "cv" / "quality" / "batch.jsonl"
    ).read_text(encoding="utf-8") == previous_batch


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


def test_two_evaluations_of_the_same_cohort_do_not_clobber(tmp_path) -> None:
    layout = initialize(tmp_path)
    write_snapshot(
        layout,
        fingerprint=_sample_fingerprint(evaluation_name="quality"),
        meta=_sample_meta(evaluator_name="quality"),
        mapped=_sample_mapped(trace_id="quality-trace"),
        dropped=[],
    )
    write_snapshot(
        layout,
        fingerprint=_sample_fingerprint(
            evaluation_name="safety", evaluator_id="eid-s", evaluator_version=1
        ),
        meta=_sample_meta(evaluator_name="safety"),
        mapped=_sample_mapped(
            trace_id="safety-trace", evaluator_name="safety", evaluator_version=1
        ),
        dropped=[],
    )
    quality_batch = (layout.sources / "kitaru" / "cv" / "quality" / "batch.jsonl").read_text(
        encoding="utf-8"
    )
    safety_batch = (layout.sources / "kitaru" / "cv" / "safety" / "batch.jsonl").read_text(
        encoding="utf-8"
    )
    assert "quality-trace" in quality_batch
    assert "safety-trace" in safety_batch
    assert "safety-trace" not in quality_batch
    assert find_local_snapshot(
        layout, cohort_name="support-production", evaluation_name="quality"
    ) == snapshot_key("cv", "quality")
    assert find_local_snapshot(
        layout, cohort_name="support-production", evaluation_name="safety"
    ) == snapshot_key("cv", "safety")
    quality = prepare_kitaru_source(
        project_root=tmp_path,
        manifest=_manifest("quality@3"),
        cohort_name="support-production",
        evaluation_name="quality",
    )
    safety = prepare_kitaru_source(
        project_root=tmp_path,
        manifest=_manifest("safety@1"),
        cohort_name="support-production",
        evaluation_name="safety",
    )
    assert quality.fingerprint.evaluation_name == "quality"
    assert "quality-trace" in quality.traces_path.read_text(encoding="utf-8")
    assert safety.fingerprint.evaluation_name == "safety"
    assert "safety-trace" in safety.traces_path.read_text(encoding="utf-8")


def test_versionless_rerun_keeps_last_fetched_after_another_cohort(tmp_path) -> None:
    layout = initialize(tmp_path)
    write_snapshot(
        layout,
        fingerprint=_sample_fingerprint(cohort_version_id="aaa-old"),
        meta=_sample_meta(display_version="week-33"),
        mapped=_sample_mapped(trace_id="old"),
        dropped=[],
    )
    write_snapshot(
        layout,
        fingerprint=_sample_fingerprint(cohort_version_id="zzz-new"),
        meta=_sample_meta(display_version="week-34"),
        mapped=_sample_mapped(trace_id="new"),
        dropped=[],
    )
    write_snapshot(
        layout,
        fingerprint=_sample_fingerprint(cohort_id="c2", cohort_version_id="other-cv"),
        meta=_sample_meta(cohort_name="billing-production", display_version="week-1"),
        mapped=_sample_mapped(trace_id="other"),
        dropped=[],
    )
    first = load_latest_pointer(
        layout, cohort_name="support-production", evaluation_name="quality"
    )
    other = load_latest_pointer(
        layout, cohort_name="billing-production", evaluation_name="quality"
    )
    assert first is not None
    assert first["cohort_version_id"] == "zzz-new"
    assert other is not None
    assert other["cohort_version_id"] == "other-cv"
    prepared = prepare_kitaru_source(
        project_root=tmp_path,
        manifest=_manifest(),
        cohort_name="support-production",
        evaluation_name="quality",
    )
    assert prepared.fingerprint.cohort_version_id == "zzz-new"
    assert prepared.refreshed is False
    assert "new" in prepared.traces_path.read_text(encoding="utf-8")


def test_missing_pointer_picks_newest_matching_snapshot(tmp_path) -> None:
    layout = initialize(tmp_path)
    write_snapshot(
        layout,
        fingerprint=_sample_fingerprint(cohort_version_id="aaa-old"),
        meta=_sample_meta(display_version="week-33"),
        mapped=_sample_mapped(trace_id="old"),
        dropped=[],
    )
    write_snapshot(
        layout,
        fingerprint=_sample_fingerprint(cohort_version_id="zzz-new"),
        meta=_sample_meta(display_version="week-34"),
        mapped=_sample_mapped(trace_id="new"),
        dropped=[],
    )
    pointer_path = kitaru_root(layout) / LATEST_POINTER
    pointer_path.write_text('{"entries": []}', encoding="utf-8")
    older = layout.sources / "kitaru" / "aaa-old" / "quality" / "fingerprint.json"
    newer = layout.sources / "kitaru" / "zzz-new" / "quality" / "fingerprint.json"
    os.utime(newer, (1_000_000_000, 1_000_000_000))
    os.utime(older, (2_000_000_000, 2_000_000_000))
    found = find_local_snapshot(
        layout, cohort_name="support-production", evaluation_name="quality"
    )
    assert found == snapshot_key("aaa-old", "quality")


def test_pick_version_matches_id_display_and_number_then_none() -> None:
    versions = [
        SimpleNamespace(
            id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            display_version="week-34",
            version=1,
        ),
        SimpleNamespace(
            id="bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
            display_version="week-35",
            version=2,
        ),
    ]
    pick = KitaruGateway._pick_version
    sentinel = object()
    assert pick(sentinel, versions, 2, None).version == 2
    assert pick(sentinel, versions, 2, "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa").version == 1
    assert pick(sentinel, versions, 2, "week-35").version == 2
    assert pick(sentinel, versions, 2, "1").version == 1
    assert pick(sentinel, versions, 2, "cccccccc-cccc-4ccc-8ccc-cccccccccccc") is None
    assert pick(sentinel, versions, 2, "not-a-version") is None
    assert pick(sentinel, [], 1, None) is None


def test_fetch_records_wraps_gather_failures_as_source_error() -> None:
    class Fake:
        async def session_bundle(self, session_id: str) -> tuple[object, ...]:
            raise TimeoutError(f"404 fetching session {session_id}")

    with pytest.raises(KitaruSourceError, match="source fetch aborted") as caught:
        asyncio.run(KitaruGateway.fetch_records(Fake(), [SimpleNamespace(id="s0")]))
    message = str(caught.value)
    assert "TimeoutError" in message
    assert "batch is not mapped" in message
    assert caught.value.__cause__ is not None


def test_corrupt_snapshot_dir_is_skipped_during_reuse(tmp_path) -> None:
    layout = initialize(tmp_path)
    write_snapshot(
        layout,
        fingerprint=_sample_fingerprint(),
        meta=_sample_meta(),
        mapped=_sample_mapped(),
        dropped=[],
    )
    bad = layout.sources / "kitaru" / "cv-bad" / "quality"
    bad.mkdir(parents=True)
    (bad / "batch.jsonl").write_text("{}\n", encoding="utf-8")
    (bad / "fingerprint.json").write_text("not-json", encoding="utf-8")
    found = find_local_snapshot(
        layout, cohort_name="support-production", evaluation_name="quality"
    )
    assert found == snapshot_key("cv", "quality")

    meta_path = layout.sources / "kitaru" / "cv" / "quality" / "meta.json"
    meta_path.unlink()
    assert (
        find_local_snapshot(
            layout, cohort_name="support-production", evaluation_name="quality"
        )
        is None
    )


def test_source_sdk_errors_are_kitaru_source_errors(tmp_path) -> None:
    resolution = CohortResolution(
        cohort_id="c",
        cohort_name="support-production",
        cohort_version_id="cv-new",
        display_version="week-35",
        version_number=2,
        agent_id="a",
        session_count=0,
    )

    class ResolveBoom:
        async def resolve_cohort(self, name: str, version: str | None = None) -> object:
            raise TimeoutError("resolve 404")

        async def close(self) -> None:
            return None

    class ListBoom:
        async def resolve_cohort(self, name: str, version: str | None = None) -> object:
            return resolution

        async def list_sessions(self, cohort_version_id: str) -> list[object]:
            raise TimeoutError("list sessions 404")

        async def close(self) -> None:
            return None

    class EvaluatorBoom:
        async def resolve_cohort(self, name: str, version: str | None = None) -> object:
            return resolution

        async def list_sessions(self, cohort_version_id: str) -> list[object]:
            return []

        async def fetch_records(self, sessions: list[object]) -> list[object]:
            return []

        async def evaluator_id(self, name: str) -> str:
            raise TimeoutError("evaluator 404")

        async def close(self) -> None:
            return None

    kwargs = dict(
        project_root=tmp_path,
        manifest=_manifest(),
        cohort_name="support-production",
        evaluation_name="quality",
        refresh=True,
    )
    with pytest.raises(KitaruSourceError, match="resolving the cohort"):
        prepare_kitaru_source(**kwargs, gateway=ResolveBoom())
    with pytest.raises(KitaruSourceError, match="listing sessions"):
        prepare_kitaru_source(**kwargs, gateway=ListBoom())
    with pytest.raises(KitaruSourceError, match="looking up the evaluator"):
        prepare_kitaru_source(**kwargs, gateway=EvaluatorBoom())


def test_wait_for_experiment_run_polls_get_not_a_missing_client_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """kitaru 0.22's API client has experiment_runs.get, not wait_for_experiment_run."""

    pytest.importorskip("kitaru")
    from importlib.metadata import version

    from kitaru.client.api_client import KitaruAPIClient
    from kitaru.client.client import KitaruClient
    from kitaru.client.resources.experiment_runs import ExperimentRunsResource

    assert version("kitaru").startswith("0.22")
    assert hasattr(ExperimentRunsResource, "get")
    assert not hasattr(KitaruAPIClient, "wait_for_experiment_run")
    # Wrapping KitaruClient would close the shared API client on close().
    assert "self._api_client.close" in inspect.getsource(KitaruClient.close)

    slept: list[float] = []

    async def fake_sleep(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    class ExperimentRuns:
        def __init__(self) -> None:
            self.calls = 0

        async def get(self, experiment_run_id: object) -> SimpleNamespace:
            self.calls += 1
            status = "running" if self.calls < 2 else "completed"
            return SimpleNamespace(id=experiment_run_id, status=status)

    class ApiClient:
        def __init__(self) -> None:
            self.experiment_runs = ExperimentRuns()
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    api = ApiClient()
    assert not hasattr(api, "wait_for_experiment_run")
    run = asyncio.run(
        KitaruGateway(client=api).wait_for_experiment_run(
            "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
        )
    )
    assert run.status == "completed"
    assert api.experiment_runs.calls == 2
    assert api.closed is False
    assert slept == [2.0]


@pytest.mark.parametrize("status", ["failed", "canceled"])
def test_wait_for_experiment_run_returns_terminal_status(status: str) -> None:
    pytest.importorskip("kitaru")

    class ExperimentRuns:
        async def get(self, experiment_run_id: object) -> SimpleNamespace:
            return SimpleNamespace(id=experiment_run_id, status=status)

    api = SimpleNamespace(experiment_runs=ExperimentRuns())
    run = asyncio.run(
        KitaruGateway(client=api).wait_for_experiment_run(
            "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
        )
    )
    assert run.status == status


def test_wait_for_experiment_run_timeout_is_fail_closed() -> None:
    pytest.importorskip("kitaru")

    class ExperimentRuns:
        async def get(self, experiment_run_id: object) -> SimpleNamespace:
            return SimpleNamespace(id=experiment_run_id, status="running")

    api = SimpleNamespace(experiment_runs=ExperimentRuns())
    with pytest.raises(TimeoutError, match="Timed out waiting for experiment run"):
        asyncio.run(
            KitaruGateway(client=api).wait_for_experiment_run(
                "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                timeout=0.01,
            )
        )
