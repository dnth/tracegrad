"""Trace batch validation and partitioning.

Ingest is the first deterministic stage: it turns a JSONL export into a single
comparable batch.  Everything it drops is dropped with a named reason, and the
reasons are part of the ingest result rather than a log line, because the run
report has to explain why a trace did not contribute to a rate.

Two partitions are applied for different purposes:

* ``prompt_hash`` — hard.  Rates are only meaningful within one prompt version,
  so only the dominant partition survives.
* ``meta.model`` — advisory.  A mixed-model batch is reported, not truncated;
  the prompt is the thing under study.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from pydantic import ValidationError

from .schema import Manifest, Trace

DEFAULT_MIN_RATIONALE_CHARS = 24
"""Rationales shorter than this carry no attributable signal."""

DEFAULT_CANARY_TOLERANCE = 0.1


class IngestError(ValueError):
    """A batch that cannot be read at all."""


@dataclass(frozen=True)
class DroppedTrace:
    """One trace excluded from the batch, with the reason it was excluded."""

    trace_id: str | None
    reason: str
    detail: str = ""


@dataclass(frozen=True)
class CanaryFailure:
    """A canary trace whose judge score moved beyond tolerance."""

    trace_id: str
    expected: float
    observed: float
    tolerance: float


@dataclass(frozen=True)
class IngestResult:
    """The accepted batch plus everything a report needs to explain it."""

    traces: tuple[Trace, ...]
    prompt_hash: str | None
    dropped: tuple[DroppedTrace, ...] = ()
    prompt_hash_partitions: Mapping[str, int] = field(default_factory=dict)
    model_partitions: Mapping[str, int] = field(default_factory=dict)
    judge_fingerprint: str | None = None
    judge_fingerprint_changed: bool = False
    previous_judge_fingerprint: str | None = None
    canary_failures: tuple[CanaryFailure, ...] = ()

    @property
    def accepted_count(self) -> int:
        return len(self.traces)

    @property
    def is_mixed_model(self) -> bool:
        return len(self.model_partitions) > 1

    @property
    def dropped_reasons(self) -> Counter[str]:
        return Counter(drop.reason for drop in self.dropped)


def read_trace_lines(path: str | Path) -> list[tuple[int, object]]:
    """Parse a JSONL trace export into ``(line_number, value)`` pairs."""

    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise IngestError(f"could not read traces {target}: {exc}") from exc

    parsed: list[tuple[int, object]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            parsed.append((line_number, json.loads(line)))
        except json.JSONDecodeError as exc:
            raise IngestError(f"{target}:{line_number}: malformed JSON: {exc}") from exc
    return parsed


def _validate(
    raw_records: Iterable[tuple[int, object]],
) -> tuple[list[Trace], list[DroppedTrace]]:
    accepted: list[Trace] = []
    dropped: list[DroppedTrace] = []
    seen: set[str] = set()
    for line_number, value in raw_records:
        if not isinstance(value, dict):
            dropped.append(
                DroppedTrace(None, "invalid-schema", f"line {line_number}: not an object")
            )
            continue
        try:
            trace = Trace.model_validate(value)
        except ValidationError as exc:
            trace_id = value.get("trace_id")
            dropped.append(
                DroppedTrace(
                    trace_id if isinstance(trace_id, str) else None,
                    "invalid-schema",
                    f"line {line_number}: {exc.error_count()} validation error(s)",
                )
            )
            continue
        if trace.trace_id in seen:
            dropped.append(
                DroppedTrace(trace.trace_id, "duplicate-trace-id", f"line {line_number}")
            )
            continue
        seen.add(trace.trace_id)
        accepted.append(trace)
    return accepted, dropped


def _rationale_is_usable(rationale: str, minimum_chars: int) -> bool:
    stripped = rationale.strip()
    if len(stripped) < minimum_chars:
        return False
    # A rationale made only of punctuation or digits explains nothing.
    return any(character.isalpha() for character in stripped)


def _dominant(counts: Mapping[str, int]) -> str | None:
    """Pick the most common key, breaking ties by sort order for determinism."""

    if not counts:
        return None
    return min(counts.items(), key=lambda item: (-item[1], item[0]))[0]


def ingest_traces(
    traces: Sequence[Trace] | Iterable[tuple[int, object]] | str | Path,
    manifest: Manifest | None = None,
    *,
    previous_judge_fingerprint: str | None = None,
    canary_scores: Mapping[str, float] | None = None,
    canary_tolerance: float = DEFAULT_CANARY_TOLERANCE,
    min_rationale_chars: int = DEFAULT_MIN_RATIONALE_CHARS,
) -> IngestResult:
    """Validate, filter, and partition a trace batch.

    ``traces`` may be a path to a JSONL export, already-parsed records, or
    already-validated :class:`~tracegrad.schema.Trace` objects.
    """

    if isinstance(traces, (str, Path)):
        candidates, dropped = _validate(read_trace_lines(traces))
    elif all(isinstance(item, Trace) for item in traces):
        candidates, dropped = list(traces), []  # type: ignore[arg-type]
    else:
        candidates, dropped = _validate(traces)  # type: ignore[arg-type]

    kept: list[Trace] = []
    for trace in candidates:
        if not _rationale_is_usable(trace.judge.rationale, min_rationale_chars):
            dropped.append(
                DroppedTrace(
                    trace.trace_id,
                    "rationale-below-quality-floor",
                    f"rationale shorter than {min_rationale_chars} usable characters",
                )
            )
            continue
        kept.append(trace)

    prompt_hash_partitions = Counter(trace.prompt_hash for trace in kept)
    dominant_prompt_hash = _dominant(prompt_hash_partitions)
    in_partition: list[Trace] = []
    for trace in kept:
        if trace.prompt_hash != dominant_prompt_hash:
            dropped.append(
                DroppedTrace(
                    trace.trace_id,
                    "prompt-hash-partition",
                    f"{trace.prompt_hash} is not the dominant {dominant_prompt_hash}",
                )
            )
            continue
        in_partition.append(trace)

    model_partitions = Counter(
        trace.meta.model if trace.meta and trace.meta.model else "unspecified"
        for trace in in_partition
    )

    canary_failures: list[CanaryFailure] = []
    if canary_scores:
        by_id = {trace.trace_id: trace for trace in in_partition}
        for trace_id in sorted(canary_scores):
            trace = by_id.get(trace_id)
            if trace is None:
                continue
            expected = float(canary_scores[trace_id])
            observed = float(trace.judge.score)
            if abs(observed - expected) > canary_tolerance:
                canary_failures.append(
                    CanaryFailure(trace_id, expected, observed, canary_tolerance)
                )

    judge_fingerprint = manifest.judge_fingerprint if manifest else None
    fingerprint_changed = bool(
        judge_fingerprint
        and previous_judge_fingerprint
        and judge_fingerprint != previous_judge_fingerprint
    )

    return IngestResult(
        traces=tuple(in_partition),
        prompt_hash=dominant_prompt_hash,
        dropped=tuple(dropped),
        prompt_hash_partitions=dict(prompt_hash_partitions),
        model_partitions=dict(model_partitions),
        judge_fingerprint=judge_fingerprint,
        judge_fingerprint_changed=fingerprint_changed,
        previous_judge_fingerprint=previous_judge_fingerprint,
        canary_failures=tuple(canary_failures),
    )
