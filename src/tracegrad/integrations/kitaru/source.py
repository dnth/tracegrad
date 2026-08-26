"""Fetch-and-map a frozen Kitaru cohort onto JSONL (ADR 0004, 0010).

The deterministic core never imports this module.  Callers write a snapshot
and pass the JSONL path to ``ingest_traces``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from tracegrad.schema import Manifest, TemplateEngine
from tracegrad.state import StateLayout, initialize

from .accounting import format_source_table
from .errors import KitaruSourceError
from .mapping import (
    REASON_AMBIGUOUS_EVALUATION,
    REASON_FORMAT_ENGINE_REFUSED,
    REASON_JUDGE_FINGERPRINT_CONFLICT,
    SourceDrop,
    map_batch,
)
from .require import require_kitaru
from .scores import judge_fingerprint_for
from .snapshot import (
    SourceFingerprint,
    SourceMeta,
    batch_path,
    find_local_snapshot,
    load_fingerprint,
    load_meta,
    load_source_drops,
    persist_run_source,
    snapshot_identity_key,
    write_snapshot,
)


@dataclass(frozen=True)
class PreparedSource:
    """What the CLI hands the existing pipeline."""

    traces_path: Path
    fingerprint: SourceFingerprint
    meta: SourceMeta
    source_table: str
    refreshed: bool


def refuse_format_engine(manifest: Manifest) -> None:
    if manifest.engine is TemplateEngine.FORMAT:
        raise KitaruSourceError(
            f"{REASON_FORMAT_ENGINE_REFUSED}: --source kitaru supports "
            'engine="none" only. A format template renders per request, so '
            "hashing recorded prompts would collapse the batch. See ADR 0002."
        )


def check_judge_fingerprint(manifest: Manifest, derived: str) -> None:
    if manifest.judge_fingerprint != derived:
        raise KitaruSourceError(
            f"{REASON_JUDGE_FINGERPRINT_CONFLICT}: manifest judge_fingerprint "
            f"{manifest.judge_fingerprint!r} does not match the evaluator that "
            f"scored this cohort ({derived}). Drift detection only works if it "
            "reads the thing that actually drifts. See ADR 0003."
        )


def _agent_version_counts(sessions: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for session in sessions:
        version = getattr(session, "agent_version_id", None)
        key = str(version) if version is not None else "unspecified"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _single_agent_version(counts: dict[str, int]) -> str | None:
    real = {key: count for key, count in counts.items() if key != "unspecified"}
    if len(real) == 1:
        return next(iter(real))
    return None


async def _fetch_and_map(
    *,
    gateway: Any,
    resolution: Any,
    evaluation_name: str,
) -> tuple[Any, Any, Any, Any]:
    sessions = await gateway.list_sessions(resolution.cohort_version_id)
    records = await gateway.fetch_records(sessions)
    mapping = map_batch(records, evaluation_name)
    if isinstance(mapping, str) and mapping == REASON_AMBIGUOUS_EVALUATION:
        versions: dict[str, int] = {}
        for session, _nodes, evaluations in records:
            from .scores import evaluator_version_of, select_evaluation

            selected = select_evaluation(list(evaluations), evaluation_name)
            if isinstance(selected, str):
                continue
            version = evaluator_version_of(selected)
            key = "unknown" if version is None else str(version)
            versions[key] = versions.get(key, 0) + 1
        breakdown = ", ".join(f"v{ver}: {count}" for ver, count in sorted(versions.items()))
        raise KitaruSourceError(
            f"{REASON_AMBIGUOUS_EVALUATION}: evaluation {evaluation_name!r} "
            f"resolves to more than one evaluator_version across the cohort "
            f"({breakdown}). Refusing to mix. See ADR 0003."
        )
    evaluator_id = await gateway.evaluator_id(
        mapping.evaluator_name or evaluation_name
    )
    fingerprint = SourceFingerprint(
        source="kitaru",
        cohort_id=resolution.cohort_id,
        cohort_version_id=resolution.cohort_version_id,
        evaluation_name=evaluation_name,
        evaluator_id=evaluator_id,
        evaluator_version=int(mapping.evaluator_version or 0),
        agent_id=resolution.agent_id,
    )
    if mapping.mapped and mapping.evaluator_version is None:
        raise KitaruSourceError(
            f"{REASON_AMBIGUOUS_EVALUATION}: mapped traces have no evaluator_version"
        )
    if mapping.mapped:
        fingerprint = fingerprint.model_copy(
            update={"evaluator_version": int(mapping.evaluator_version)}
        )
    counts = _agent_version_counts(sessions)
    meta = SourceMeta(
        cohort_name=resolution.cohort_name,
        display_version=resolution.display_version,
        agent_version_id=_single_agent_version(counts),
        agent_version_counts=counts,
        session_numbers={
            item.trace.trace_id: item.session_number
            for item in mapping.mapped
            if item.session_number is not None
        },
        system_prompts={item.trace.trace_id: item.system_prompt for item in mapping.mapped},
        multi_turn_count=mapping.multi_turn_count,
        sessions_selected=len(sessions),
        traces_mapped=len(mapping.mapped),
        evaluator_name=mapping.evaluator_name or evaluation_name,
    )
    return fingerprint, meta, mapping.mapped, mapping.dropped


def _assembled_source(
    layout: StateLayout,
    *,
    snapshot_id: str,
    fingerprint: SourceFingerprint,
    meta: SourceMeta,
    dropped: Sequence[SourceDrop],
    refreshed: bool,
    manifest: Manifest,
    evaluation_name: str,
    run_id: str | None,
) -> PreparedSource:
    if meta.traces_mapped:
        derived = judge_fingerprint_for(
            meta.evaluator_name or evaluation_name,
            fingerprint.evaluator_version,
        )
        check_judge_fingerprint(manifest, derived)
    if run_id is not None:
        persist_run_source(layout, run_id, fingerprint, meta)
    table = format_source_table(
        sessions_selected=meta.sessions_selected,
        traces_mapped=meta.traces_mapped,
        dropped=dropped,
    )
    if meta.multi_turn_count:
        table += (
            f"\n{meta.multi_turn_count} multi-turn session(s) collapsed to "
            "first-root input and last-root output"
        )
    return PreparedSource(
        traces_path=batch_path(layout, snapshot_id),
        fingerprint=fingerprint,
        meta=meta,
        source_table=table,
        refreshed=refreshed,
    )


def prepare_kitaru_source(
    *,
    project_root: str | Path,
    manifest: Manifest,
    cohort_name: str,
    evaluation_name: str,
    cohort_version: str | None = None,
    refresh: bool = False,
    gateway: Any | None = None,
    run_id: str | None = None,
) -> PreparedSource:
    """Resolve, snapshot, and return the JSONL path ingest already reads.

    Re-runs read the local snapshot (including ``latest.json``) unless
    ``--refresh``. The gateway is constructed only when a fetch is needed so
    a pinned snapshot stays reproducible with the server unreachable
    (ADR 0004 / issue #8).
    """

    refuse_format_engine(manifest)
    layout = initialize(project_root)

    if not refresh:
        local_id = find_local_snapshot(
            layout,
            cohort_name=cohort_name,
            evaluation_name=evaluation_name,
            cohort_version=cohort_version,
        )
        if local_id is not None:
            fingerprint = load_fingerprint(layout, local_id)
            meta = load_meta(layout, local_id)
            dropped = load_source_drops(layout, local_id)
            return _assembled_source(
                layout,
                snapshot_id=local_id,
                fingerprint=fingerprint,
                meta=meta,
                dropped=dropped,
                refreshed=False,
                manifest=manifest,
                evaluation_name=evaluation_name,
                run_id=run_id,
            )

    from .client import run_async

    owns = gateway is None
    if owns:
        require_kitaru()
        from .client import KitaruGateway

        gateway = KitaruGateway()

    async def _run() -> PreparedSource:
        try:
            resolution = await gateway.resolve_cohort(cohort_name, cohort_version)
            fingerprint, meta, mapped, dropped = await _fetch_and_map(
                gateway=gateway,
                resolution=resolution,
                evaluation_name=evaluation_name,
            )
            write_snapshot(
                layout, fingerprint=fingerprint, meta=meta, mapped=mapped, dropped=dropped
            )
            return _assembled_source(
                layout,
                snapshot_id=snapshot_identity_key(fingerprint),
                fingerprint=fingerprint,
                meta=meta,
                dropped=dropped,
                refreshed=True,
                manifest=manifest,
                evaluation_name=evaluation_name,
                run_id=run_id,
            )
        finally:
            if owns:
                await gateway.close()

    return run_async(_run())
