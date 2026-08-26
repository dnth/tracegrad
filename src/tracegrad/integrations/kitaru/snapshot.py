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
    PathContainmentError,
    StateLayout,
    atomic_write,
    atomic_write_json,
    contained_path,
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


def _path_safe_component(value: str) -> str:
    """A single directory name: no separators, no ``.`` / ``..``."""

    text = str(value).strip()
    if not text:
        raise ValueError("snapshot path component must be non-empty")
    collapsed = text.replace("\\", "/")
    parts = [part.replace("..", "__") for part in collapsed.split("/") if part not in {"", ".", ".."}]
    cleaned = "--".join(parts)
    if not cleaned or Path(cleaned).name != cleaned:
        raise ValueError(f"unsafe snapshot path component: {value!r}")
    return cleaned


def snapshot_key(cohort_version_id: str, evaluation_name: str) -> str:
    """On-disk identity: one frozen cohort version × one evaluation.

    Fingerprint identity includes ``evaluation_name``; keying only by
    ``cohort_version_id`` would let a later ``--kitaru-evaluation`` clobber
    the earlier JSONL.
    """

    return f"{_path_safe_component(cohort_version_id)}/{_path_safe_component(evaluation_name)}"


def snapshot_identity_key(fingerprint: SourceFingerprint) -> str:
    return snapshot_key(fingerprint.cohort_version_id, fingerprint.evaluation_name)


def snapshot_dir(layout: StateLayout, snapshot_id: str) -> Path:
    return contained_path(kitaru_root(layout), snapshot_id)


def _is_snapshot_dir(target: Path) -> bool:
    return (target / BATCH_FILENAME).is_file() and (target / FINGERPRINT_FILENAME).is_file()


def snapshot_exists(layout: StateLayout, snapshot_id: str) -> bool:
    try:
        target = snapshot_dir(layout, snapshot_id)
    except (ValueError, PathContainmentError, OSError):
        return False
    return _is_snapshot_dir(target)


def write_snapshot(
    layout: StateLayout,
    *,
    fingerprint: SourceFingerprint,
    meta: SourceMeta,
    mapped: Sequence[MappedTrace],
    dropped: Sequence[SourceDrop],
) -> Path:
    """Write JSONL + fingerprint + meta under ``.tracegrad/sources/kitaru/``."""

    snapshot_id = snapshot_identity_key(fingerprint)
    target = snapshot_dir(layout, snapshot_id)
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
    _upsert_latest_pointer(
        layout,
        {
            "cohort_name": meta.cohort_name,
            "evaluation_name": fingerprint.evaluation_name,
            "cohort_version_id": fingerprint.cohort_version_id,
            "snapshot_id": snapshot_id,
            "display_version": meta.display_version,
            "batch": str(target / BATCH_FILENAME),
        },
    )
    return target


def load_fingerprint(layout: StateLayout, snapshot_id: str) -> SourceFingerprint:
    path = snapshot_dir(layout, snapshot_id) / FINGERPRINT_FILENAME
    return SourceFingerprint.model_validate_json(path.read_text(encoding="utf-8"))


def load_meta(layout: StateLayout, snapshot_id: str) -> SourceMeta:
    path = snapshot_dir(layout, snapshot_id) / META_FILENAME
    return SourceMeta.model_validate_json(path.read_text(encoding="utf-8"))


def load_source_drops(layout: StateLayout, snapshot_id: str) -> tuple[SourceDrop, ...]:
    path = snapshot_dir(layout, snapshot_id) / DROPS_FILENAME
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


def batch_path(layout: StateLayout, snapshot_id: str) -> Path:
    return snapshot_dir(layout, snapshot_id) / BATCH_FILENAME


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


def _read_pointer_file(layout: StateLayout) -> dict[str, Any] | None:
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


def _pointer_entries(raw: dict[str, Any]) -> list[dict[str, Any]]:
    entries = raw.get("entries")
    if isinstance(entries, list):
        return [item for item in entries if isinstance(item, dict)]
    if isinstance(raw.get("cohort_version_id"), str):
        return [raw]
    return []


def _upsert_latest_pointer(layout: StateLayout, entry: Mapping[str, Any]) -> None:
    raw = _read_pointer_file(layout) or {}
    kept: list[dict[str, Any]] = []
    key = (entry.get("cohort_name"), entry.get("evaluation_name"))
    for existing in _pointer_entries(raw):
        existing_key = (existing.get("cohort_name"), existing.get("evaluation_name"))
        if existing_key == key:
            continue
        kept.append(existing)
    kept.append(dict(entry))
    atomic_write_json(kitaru_root(layout) / LATEST_POINTER, {"entries": kept})


def load_latest_pointer(
    layout: StateLayout,
    *,
    cohort_name: str,
    evaluation_name: str,
) -> dict[str, Any] | None:
    """The last-fetched snapshot for this cohort name + evaluation, if any."""

    raw = _read_pointer_file(layout)
    if raw is None:
        return None
    for entry in _pointer_entries(raw):
        if entry.get("cohort_name") == cohort_name and entry.get("evaluation_name") == evaluation_name:
            return entry
    return None


def list_snapshot_ids(layout: StateLayout) -> list[str]:
    """Relative ids under the Kitaru source root, including nested eval dirs.

    Legacy snapshots keyed only by ``cohort_version_id`` are still listed so
    an older tree remains readable.
    """

    root = kitaru_root(layout)
    if not root.is_dir():
        return []
    found: list[str] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        for grandchild in sorted(child.iterdir()):
            rel = f"{child.name}/{grandchild.name}"
            if grandchild.is_dir() and snapshot_exists(layout, rel):
                found.append(rel)
        if snapshot_exists(layout, child.name):
            found.append(child.name)
    return found


def snapshot_matches_request(
    layout: StateLayout,
    snapshot_id: str,
    *,
    cohort_name: str,
    evaluation_name: str,
    cohort_version: str | None = None,
) -> bool:
    """Whether this on-disk snapshot is the one the request asked for."""

    if not snapshot_exists(layout, snapshot_id):
        return False
    fingerprint = load_fingerprint(layout, snapshot_id)
    if not fingerprints_compatible(fingerprint, {"evaluation_name": evaluation_name}):
        return False
    meta = load_meta(layout, snapshot_id)
    if meta.cohort_name != cohort_name:
        return False
    if cohort_version is None:
        return True
    return (
        fingerprint.cohort_version_id == cohort_version
        or meta.display_version == cohort_version
        or snapshot_id == cohort_version
    )


def _snapshot_mtime(layout: StateLayout, snapshot_id: str) -> float:
    path = snapshot_dir(layout, snapshot_id) / FINGERPRINT_FILENAME
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _newest_matching(layout: StateLayout, snapshot_ids: Sequence[str]) -> str | None:
    if not snapshot_ids:
        return None
    return max(snapshot_ids, key=lambda item: (_snapshot_mtime(layout, item), item))


def _pointer_snapshot_id(entry: Mapping[str, Any], evaluation_name: str) -> str | None:
    snapshot_id = entry.get("snapshot_id")
    if isinstance(snapshot_id, str) and snapshot_id:
        return snapshot_id
    cohort_version_id = entry.get("cohort_version_id")
    if not isinstance(cohort_version_id, str) or not cohort_version_id:
        return None
    return snapshot_key(cohort_version_id, evaluation_name)


def find_local_snapshot(
    layout: StateLayout,
    *,
    cohort_name: str,
    evaluation_name: str,
    cohort_version: str | None = None,
) -> str | None:
    """Return a local snapshot id that satisfies this request.

    Prefers the ``latest.json`` entry for ``(cohort_name, evaluation_name)``
    so a version-less re-run keeps the last fetched version of *this* pair
    even after another cohort was sourced. When that pointer is missing,
    picks the newest matching snapshot rather than the lexicographically
    first directory. Does not contact the server. ``--refresh`` is the
    caller's decision not to call this.
    """

    def matches(snapshot_id: str) -> bool:
        return snapshot_matches_request(
            layout,
            snapshot_id,
            cohort_name=cohort_name,
            evaluation_name=evaluation_name,
            cohort_version=cohort_version,
        )

    if cohort_version is not None:
        candidates = [
            snapshot_key(cohort_version, evaluation_name),
            _path_safe_component(cohort_version),
        ]
        seen: set[str] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            if matches(candidate):
                return candidate
        return _newest_matching(layout, [item for item in list_snapshot_ids(layout) if matches(item)])

    pointer = load_latest_pointer(
        layout, cohort_name=cohort_name, evaluation_name=evaluation_name
    )
    if pointer is not None:
        pointed = _pointer_snapshot_id(pointer, evaluation_name)
        if pointed and matches(pointed):
            return pointed
        legacy = pointer.get("cohort_version_id")
        if isinstance(legacy, str) and legacy and matches(legacy):
            return legacy
    return _newest_matching(layout, [item for item in list_snapshot_ids(layout) if matches(item)])
