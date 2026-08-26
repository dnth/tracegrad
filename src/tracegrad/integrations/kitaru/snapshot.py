"""On-disk snapshot of a mapped Kitaru cohort (ADR 0004).

The deterministic core never sees this module.  Re-runs read the JSONL ingest
already knows; ``--refresh`` refetches.  No Kitaru SDK import.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field
from pydantic.types import StrictInt, StrictStr

from tracegrad.state import (
    StateLayout,
    atomic_write,
    atomic_write_json,
    initialize,
    validate_run_id,
)

from .mapping import MAPPING_VERSION, MappedTrace, SourceDrop

SOURCE_KIND = "kitaru"
BATCH_FILENAME = "batch.jsonl"
FINGERPRINT_FILENAME = "fingerprint.json"
META_FILENAME = "meta.json"
DROPS_FILENAME = "source-drops.jsonl"
LATEST_POINTER = "latest.json"
RUN_SOURCE_FILENAME = "kitaru-source.json"


class SourceFingerprint(BaseModel):
    """The fingerprint persisted next to the mapped batch."""

    model_config = ConfigDict(extra="forbid")

    source: StrictStr = SOURCE_KIND
    cohort_id: StrictStr
    cohort_version_id: StrictStr
    evaluation_name: StrictStr
    evaluator_id: StrictStr
    evaluator_version: StrictInt
    agent_id: StrictStr
    mapping_version: StrictInt = MAPPING_VERSION


class SourceMeta(BaseModel):
    """Sidecar Phase 2 reuses; not part of the fingerprint contract."""

    model_config = ConfigDict(extra="forbid")

    cohort_name: StrictStr
    display_version: StrictStr | None = None
    agent_version_id: StrictStr | None = None
    agent_version_counts: dict[StrictStr, StrictInt] = Field(default_factory=dict)
    session_numbers: dict[StrictStr, StrictInt] = Field(default_factory=dict)
    system_prompts: dict[StrictStr, StrictStr] = Field(default_factory=dict)
    multi_turn_count: StrictInt = 0
    sessions_selected: StrictInt = 0
    traces_mapped: StrictInt = 0
    evaluator_name: StrictStr | None = None


def kitaru_root(layout: StateLayout) -> Path:
    return layout.sources / SOURCE_KIND


def snapshot_dir(layout: StateLayout, cohort_version_id: str) -> Path:
    return kitaru_root(layout) / cohort_version_id


def snapshot_exists(layout: StateLayout, cohort_version_id: str) -> bool:
    target = snapshot_dir(layout, cohort_version_id)
    return (target / BATCH_FILENAME).is_file() and (target / FINGERPRINT_FILENAME).is_file()


def write_snapshot(
    layout: StateLayout,
    *,
    fingerprint: SourceFingerprint,
    meta: SourceMeta,
    mapped: Sequence[MappedTrace],
    dropped: Sequence[SourceDrop],
) -> Path:
    """Write JSONL + fingerprint + meta under ``.tracegrad/sources/kitaru/``."""

    target = snapshot_dir(layout, fingerprint.cohort_version_id)
    target.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(item.trace.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
        for item in mapped
    ]
    atomic_write(target / BATCH_FILENAME, "\n".join(lines) + ("\n" if lines else ""))
    atomic_write_json(target / FINGERPRINT_FILENAME, fingerprint.model_dump(mode="json"))
    atomic_write_json(target / META_FILENAME, meta.model_dump(mode="json"))
    drop_lines = [
        json.dumps(
            {
                "session_id": drop.session_id,
                "reason": drop.reason,
                "detail": drop.detail,
                "number": drop.number,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for drop in dropped
    ]
    atomic_write(target / DROPS_FILENAME, "\n".join(drop_lines) + ("\n" if drop_lines else ""))
    atomic_write_json(
        kitaru_root(layout) / LATEST_POINTER,
        {
            "cohort_version_id": fingerprint.cohort_version_id,
            "batch": str(target / BATCH_FILENAME),
            "cohort_name": meta.cohort_name,
            "evaluation_name": fingerprint.evaluation_name,
            "display_version": meta.display_version,
        },
    )
    return target


def load_fingerprint(layout: StateLayout, cohort_version_id: str) -> SourceFingerprint:
    path = snapshot_dir(layout, cohort_version_id) / FINGERPRINT_FILENAME
    return SourceFingerprint.model_validate_json(path.read_text(encoding="utf-8"))


def load_meta(layout: StateLayout, cohort_version_id: str) -> SourceMeta:
    path = snapshot_dir(layout, cohort_version_id) / META_FILENAME
    return SourceMeta.model_validate_json(path.read_text(encoding="utf-8"))


def load_source_drops(layout: StateLayout, cohort_version_id: str) -> tuple[SourceDrop, ...]:
    path = snapshot_dir(layout, cohort_version_id) / DROPS_FILENAME
    if not path.exists():
        return ()
    drops: list[SourceDrop] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        drops.append(
            SourceDrop(
                session_id=str(raw["session_id"]),
                reason=str(raw["reason"]),
                detail=str(raw.get("detail") or ""),
                number=raw.get("number") if isinstance(raw.get("number"), int) else None,
            )
        )
    return tuple(drops)


def batch_path(layout: StateLayout, cohort_version_id: str) -> Path:
    return snapshot_dir(layout, cohort_version_id) / BATCH_FILENAME


def persist_run_source(
    layout: StateLayout,
    run_id: str,
    fingerprint: SourceFingerprint,
    meta: SourceMeta,
) -> Path:
    """Copy source identity into the run directory so verify can find it."""

    target = layout.runs / validate_run_id(run_id) / RUN_SOURCE_FILENAME
    atomic_write_json(
        target,
        {
            "fingerprint": fingerprint.model_dump(mode="json"),
            "meta": meta.model_dump(mode="json"),
        },
    )
    return target


def load_run_source(project_root: str | Path | StateLayout, run_id: str) -> dict[str, Any] | None:
    layout = initialize(project_root)
    target = layout.runs / validate_run_id(run_id) / RUN_SOURCE_FILENAME
    if not target.exists():
        return None
    return json.loads(target.read_text(encoding="utf-8"))


def fingerprints_compatible(stored: SourceFingerprint, requested: Mapping[str, Any]) -> bool:
    """Whether a snapshot can be reused for this request without refetching."""

    if stored.source != SOURCE_KIND:
        return False
    if stored.mapping_version != MAPPING_VERSION:
        return False
    if stored.evaluation_name != requested.get("evaluation_name"):
        return False
    requested_id = requested.get("cohort_version_id")
    if requested_id is not None and str(stored.cohort_version_id) != str(requested_id):
        return False
    return True


def load_latest_pointer(layout: StateLayout) -> dict[str, Any] | None:
    """Read ``latest.json`` if a previous source write left one."""

    path = kitaru_root(layout) / LATEST_POINTER
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    return raw


def list_snapshot_ids(layout: StateLayout) -> list[str]:
    root = kitaru_root(layout)
    if not root.is_dir():
        return []
    return [
        child.name
        for child in sorted(root.iterdir())
        if child.is_dir() and snapshot_exists(layout, child.name)
    ]


def snapshot_matches_request(
    layout: StateLayout,
    cohort_version_id: str,
    *,
    cohort_name: str,
    evaluation_name: str,
    cohort_version: str | None = None,
) -> bool:
    """Whether this on-disk snapshot is the one the request asked for."""

    if not snapshot_exists(layout, cohort_version_id):
        return False
    fingerprint = load_fingerprint(layout, cohort_version_id)
    if not fingerprints_compatible(fingerprint, {"evaluation_name": evaluation_name}):
        return False
    meta = load_meta(layout, cohort_version_id)
    if meta.cohort_name != cohort_name:
        return False
    if cohort_version is None:
        return True
    return cohort_version_id == cohort_version or meta.display_version == cohort_version


def find_local_snapshot(
    layout: StateLayout,
    *,
    cohort_name: str,
    evaluation_name: str,
    cohort_version: str | None = None,
) -> str | None:
    """Return a local ``cohort_version_id`` that satisfies this request.

    Prefers ``latest.json`` when it matches so a re-run keeps the pinned
    version even if the server's latest has moved. Does not contact the
    server. ``--refresh`` is the caller's decision not to call this.
    """

    if cohort_version is not None:
        if snapshot_matches_request(
            layout,
            cohort_version,
            cohort_name=cohort_name,
            evaluation_name=evaluation_name,
            cohort_version=cohort_version,
        ):
            return cohort_version
        for snapshot_id in list_snapshot_ids(layout):
            if snapshot_matches_request(
                layout,
                snapshot_id,
                cohort_name=cohort_name,
                evaluation_name=evaluation_name,
                cohort_version=cohort_version,
            ):
                return snapshot_id
        return None

    pointer = load_latest_pointer(layout)
    if pointer is not None:
        latest_id = str(pointer.get("cohort_version_id") or "")
        if latest_id and snapshot_matches_request(
            layout,
            latest_id,
            cohort_name=cohort_name,
            evaluation_name=evaluation_name,
        ):
            return latest_id
    for snapshot_id in list_snapshot_ids(layout):
        if snapshot_matches_request(
            layout,
            snapshot_id,
            cohort_name=cohort_name,
            evaluation_name=evaluation_name,
        ):
            return snapshot_id
    return None
