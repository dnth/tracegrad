"""The human gate — the only module in tracegrad that writes the prompt.

Everything upstream is analysis and may be re-run freely.  This module changes a
file the user ships, so it is the conservative one: it re-checks that the
template still hashes to what the proposal was computed against, snapshots the
file before touching it, writes atomically, and records what was accepted so the
next batch can be compared against the right baseline.

If the template changed out of band since the proposal was made, the proposal is
marked stale and refused.  Spans resolved against yesterday's file are not
addresses in today's file.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic.types import StrictStr

from .canonical import text_hash
from .distill import DistilledTrace
from .edits import apply_resolved, resolve_edits
from .gates import GateFlag, GateOutcome
from .inventory import build_inventory
from .schema import AttributionResult, Edit
from .state import (
    PathContainmentError,
    StateLayout,
    append_jsonl,
    atomic_write,
    atomic_write_json,
    contained_path,
    initialize,
    validate_run_id,
)

PROPOSAL_FILENAME = "proposal.json"
APPLIED_LEDGER_FILENAME = "applied.jsonl"


class ApplyError(RuntimeError):
    """The prompt cannot be written safely."""


class StaleProposalError(ApplyError):
    """The template changed since the proposal was computed."""


class EditEvidence(BaseModel):
    """One verbatim quote backing an edit, with the source it came from."""

    model_config = ConfigDict(extra="forbid")

    trace_id: StrictStr | None = None
    quote: StrictStr
    quote_source: StrictStr
    theme: StrictStr


class ProposedEdit(BaseModel):
    """One reviewable edit: what changes, why, and what to watch afterwards."""

    model_config = ConfigDict(extra="forbid")

    edit: Edit
    before: StrictStr = ""
    after: StrictStr = ""
    evidence: list[EditEvidence] = Field(default_factory=list)
    flags: list[StrictStr] = Field(default_factory=list)
    reclassified: bool = False


class Proposal(BaseModel):
    """The persisted output of one run, and the input to ``apply``."""

    model_config = ConfigDict(extra="forbid")

    run_id: StrictStr
    template_file: StrictStr
    base_prompt_hash: StrictStr
    edits: list[ProposedEdit] = Field(default_factory=list)
    rejected: list[dict[str, StrictStr]] = Field(default_factory=list)
    tokens_before: int = 0
    tokens_after: int = 0

    def edit_at(self, index: int) -> ProposedEdit:
        return self.edits[index]


def build_proposal(
    *,
    run_id: str,
    template_file: str | Path,
    prompt: str,
    outcome: GateOutcome,
    attributions: Sequence[AttributionResult] = (),
    distilled: Mapping[str, DistilledTrace] | None = None,
    theme_map: Mapping[str, str] | None = None,
    quotes_per_edit: int = 3,
) -> Proposal:
    """Assemble the reviewable proposal from a gate outcome."""

    records = dict(distilled or {})
    canonical = dict(theme_map or {})
    flags_by_edit: dict[str, list[GateFlag]] = {}
    for flag in outcome.flags:
        flags_by_edit.setdefault(flag.instruction_id, []).append(flag)

    evidence_by_theme: dict[str, list[EditEvidence]] = {}
    for result in attributions:
        record = records.get(result.trace_id or "")
        for entry in (*result.violations, *result.harmful):
            theme = canonical.get(entry.theme_slug, entry.theme_slug)
            # A missing distilled record means the quote CANNOT be verified, so
            # it is not evidence. The gates already treat it that way; the card
            # a human actually reads must not be the one place that trusts it.
            if record is None or entry.quote not in record.quotable(entry.quote_source.value):
                continue
            evidence_by_theme.setdefault(theme, []).append(
                EditEvidence(
                    trace_id=result.trace_id,
                    quote=entry.quote,
                    quote_source=entry.quote_source.value,
                    theme=theme,
                )
            )

    proposed: list[ProposedEdit] = []
    for item in outcome.kept:
        theme = canonical.get(item.edit.covers_theme, item.edit.covers_theme)
        proposed.append(
            ProposedEdit(
                edit=item.edit,
                before=item.anchor.text if item.anchor else "",
                after=item.replacement,
                evidence=evidence_by_theme.get(theme, [])[:quotes_per_edit],
                flags=[flag.flag for flag in flags_by_edit.get(item.edit.instruction_id, [])],
                reclassified=item.edit.instruction_id in outcome.reclassified,
            )
        )

    return Proposal(
        run_id=run_id,
        template_file=str(template_file),
        base_prompt_hash=text_hash(prompt),
        edits=proposed,
        rejected=[
            {
                "instruction_id": rejection.edit.instruction_id,
                "operation": rejection.edit.operation,
                "reason": rejection.reason,
                "detail": rejection.detail,
            }
            for rejection in outcome.rejected
        ],
        tokens_before=outcome.tokens_before,
        tokens_after=outcome.tokens_after,
    )


def proposal_path(project_root: str | Path, run_id: str) -> Path:
    layout = initialize(project_root)
    return layout.runs / validate_run_id(run_id) / PROPOSAL_FILENAME


def save_proposal(project_root: str | Path, proposal: Proposal) -> Path:
    target = proposal_path(project_root, proposal.run_id)
    atomic_write_json(target, proposal.model_dump(mode="json"))
    return target


def load_proposal(project_root: str | Path, run_id: str) -> Proposal:
    target = proposal_path(project_root, run_id)
    try:
        return Proposal.model_validate_json(target.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ApplyError(f"no proposal for run {run_id}: {exc}") from exc
    except ValidationError as exc:
        raise ApplyError(f"proposal for run {run_id} is malformed: {exc}") from exc


def latest_run_id(project_root: str | Path) -> str | None:
    """The most recent run that produced a proposal, by run-id sort order."""

    layout = initialize(project_root)
    candidates = sorted(
        path.parent.name
        for path in layout.runs.glob(f"*/{PROPOSAL_FILENAME}")
    )
    return candidates[-1] if candidates else None


@dataclass(frozen=True)
class ReviewCard:
    """What a human sees for one edit before deciding."""

    index: int
    edit: Edit
    diff: str
    evidence: tuple[EditEvidence, ...]
    flags: tuple[str, ...]
    reclassified: bool

    def render(self) -> str:
        lines = [
            f"[{self.index}] {self.edit.operation} {self.edit.instruction_id}",
            f"    theme: {self.edit.covers_theme}   watch: {self.edit.watch_metric}",
        ]
        if self.reclassified:
            lines.append("    note: reclassified REWRITE -> ADD (it introduces a new instruction)")
        for flag in self.flags:
            lines.append(f"    flag: {flag}")
        lines.append("    diff:")
        lines.extend(f"      {line}" for line in self.diff.splitlines())
        if self.evidence:
            lines.append("    evidence:")
            for item in self.evidence:
                trace = item.trace_id or "?"
                lines.append(f'      [{item.quote_source}] {trace}: "{item.quote.strip()}"')
        else:
            lines.append("    evidence: none survived verification")
        return "\n".join(lines)


def _diff(before: str, after: str) -> str:
    return "\n".join(
        difflib.unified_diff(
            before.splitlines() or [""],
            after.splitlines() or [""],
            fromfile="current",
            tofile="proposed",
            lineterm="",
            n=1,
        )
    )


def review_cards(proposal: Proposal) -> tuple[ReviewCard, ...]:
    """Render one card per surviving edit, in proposal order."""

    return tuple(
        ReviewCard(
            index=index,
            edit=item.edit,
            diff=_diff(item.before, item.after),
            evidence=tuple(item.evidence),
            flags=tuple(item.flags),
            reclassified=item.reclassified,
        )
        for index, item in enumerate(proposal.edits)
    )


@dataclass(frozen=True)
class ApplyResult:
    """The outcome of writing accepted edits back to the template."""

    template_file: Path
    applied_prompt_hash: str
    accepted: tuple[Edit, ...]
    rejected: tuple[Edit, ...]
    snapshot: Path | None
    unchanged: bool = False
    resolution_rejections: tuple[str, ...] = ()


def snapshot_template(project_root: str | Path, run_id: str, template: Path) -> Path:
    """Copy the template before writing, so a revert is always possible."""

    layout = initialize(project_root)
    target = layout.snapshots / f"{validate_run_id(run_id)}-{template.name}"
    atomic_write(target, template.read_text(encoding="utf-8"))
    return target


def candidate_prompt(
    prompt: str,
    proposal: Proposal,
    accepted_indices: Iterable[int],
) -> str:
    """The text that would be written if these indices were accepted.

    Used by verify, the apply gate, and apply_proposal so the ADR 0009 hash
    gate hashes the same bytes that are written.
    """

    selected = sorted({index for index in accepted_indices})
    for index in selected:
        if index < 0 or index >= len(proposal.edits):
            raise ApplyError(f"no such edit index: {index}")
    accepted = [proposal.edits[index].edit for index in selected]
    if not accepted:
        return prompt
    inventory = build_inventory(prompt)
    resolution = resolve_edits(inventory, accepted)
    return apply_resolved(prompt, resolution.resolved)


def apply_proposal(
    project_root: str | Path,
    proposal: Proposal,
    accepted_indices: Iterable[int],
    *,
    base_directory: str | Path = ".",
    force: bool = False,
) -> ApplyResult:
    """Write the accepted edits, after re-checking the base hash.

    Partial acceptance is the normal case: the indices a human picked are
    applied, the rest are recorded as rejected so the memory gate remembers them.
    """

    try:
        template = contained_path(base_directory, proposal.template_file)
    except PathContainmentError as exc:
        raise ApplyError(f"refusing to write outside the project: {exc}") from exc
    try:
        current = template.read_text(encoding="utf-8")
    except OSError as exc:
        raise ApplyError(f"could not read template {template}: {exc}") from exc

    if text_hash(current) != proposal.base_prompt_hash and not force:
        raise StaleProposalError(
            f"{template} changed since the proposal was computed; re-run tracegrad"
        )

    selected = sorted({index for index in accepted_indices})
    # ADR 0009's hash gate is only sound if apply writes the same bytes verify
    # hashed. candidate_prompt is that shared path.
    updated = candidate_prompt(current, proposal, selected)
    selected_set = set(selected)
    accepted = [proposal.edits[index].edit for index in selected]
    rejected = [
        item.edit for index, item in enumerate(proposal.edits) if index not in selected_set
    ]

    if updated == current:
        return ApplyResult(
            template_file=template,
            applied_prompt_hash=text_hash(current),
            accepted=(),
            rejected=tuple(rejected),
            snapshot=None,
            unchanged=True,
        )

    snapshot = snapshot_template(project_root, proposal.run_id, template)
    atomic_write(template, updated)
    applied_hash = text_hash(updated)

    layout = initialize(project_root)
    append_jsonl(
        layout.ledgers / APPLIED_LEDGER_FILENAME,
        {
            "event": "applied",
            "run_id": proposal.run_id,
            "template_file": str(proposal.template_file),
            "base_prompt_hash": proposal.base_prompt_hash,
            "applied_prompt_hash": applied_hash,
            "snapshot": str(snapshot),
            "accepted": [edit.model_dump(mode="json") for edit in accepted],
            "rejected": [edit.model_dump(mode="json") for edit in rejected],
        },
    )

    return ApplyResult(
        template_file=template,
        applied_prompt_hash=applied_hash,
        accepted=tuple(accepted),
        rejected=tuple(rejected),
        snapshot=snapshot,
    )


def revert(
    project_root: str | Path,
    run_id: str,
    *,
    base_directory: str | Path = ".",
    force: bool = False,
) -> Path:
    """Restore the pre-apply snapshot for one run and record the revert.

    Reverting overwrites the template, so it is as careful as applying: if the
    file no longer hashes to what the apply produced, someone has edited it
    since, and restoring the snapshot would silently destroy that work.  The
    current file is snapshotted first either way, so a revert is itself
    reversible.
    """

    layout = initialize(project_root)
    records = [
        record
        for record in _applied_records(layout)
        if record.get("run_id") == run_id and record.get("event") == "applied"
    ]
    if not records:
        raise ApplyError(f"run {run_id} has no applied snapshot to revert")
    record = records[-1]
    snapshot = Path(str(record["snapshot"]))
    try:
        template = contained_path(base_directory, str(record["template_file"]))
    except PathContainmentError as exc:
        raise ApplyError(f"refusing to write outside the project: {exc}") from exc

    try:
        current = template.read_text(encoding="utf-8")
    except OSError as exc:
        raise ApplyError(f"could not read template {template}: {exc}") from exc
    applied_hash = str(record.get("applied_prompt_hash", ""))
    if applied_hash and text_hash(current) != applied_hash and not force:
        raise StaleProposalError(
            f"{template} has changed since run {run_id} was applied; reverting would "
            "discard those edits — re-check the file, then revert with force"
        )
    # Snapshot what is there now, so the revert can itself be undone.
    snapshot_template(project_root, validate_run_id(run_id) + "-prerevert", template)

    try:
        atomic_write(template, snapshot.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ApplyError(f"could not restore {template} from {snapshot}: {exc}") from exc
    append_jsonl(
        layout.ledgers / APPLIED_LEDGER_FILENAME,
        {
            "event": "reverted",
            "run_id": run_id,
            "template_file": str(record["template_file"]),
            "restored_prompt_hash": text_hash(template.read_text(encoding="utf-8")),
        },
    )
    _restore_reverted_gaps(layout, record, run_id)
    _remember_reverted_edits(layout, record, run_id)
    return template


def _remember_reverted_edits(
    layout: StateLayout, record: Mapping[str, object], run_id: str
) -> None:
    """Record the reverted edits so G6 does not re-propose them unopposed.

    A revert is a stronger signal than a rejection — the edit was accepted,
    shipped, and taken back — so the memory gate has to see it.
    """

    from .gates import REJECTION_MEMORY_FILENAME, RejectionMemory

    memory = RejectionMemory(layout.ledgers / REJECTION_MEMORY_FILENAME)
    for raw in record.get("accepted", []) or []:  # type: ignore[union-attr]
        if not isinstance(raw, dict):
            continue
        try:
            memory.record_revert(Edit.model_validate(raw), run_id=run_id)
        except ValidationError:
            continue


def _restore_reverted_gaps(
    layout: StateLayout, record: Mapping[str, object], run_id: str
) -> None:
    """Reopen the gap themes the reverted edits were supposed to close.

    An edit that got retired-by-improved-trend and is now reverted leaves the
    failure it addressed uncovered again.  Leaving the theme retired would hide
    it from every future run.
    """

    from .aggregate import GAP_LEDGER_FILENAME, GapLedger

    gaps = GapLedger(layout.ledgers / GAP_LEDGER_FILENAME)
    known = gaps.state()
    themes: list[str] = []
    for edit in record.get("accepted", []) or []:  # type: ignore[union-attr]
        if not isinstance(edit, dict):
            continue
        for key in ("covers_theme", "watch_metric"):
            theme = edit.get(key)
            if isinstance(theme, str) and theme and theme not in themes:
                themes.append(theme)
    for theme in themes:
        if theme in known:
            gaps.restore(theme, run_id=run_id, reason="revert")


def _applied_records(layout: StateLayout) -> list[dict[str, object]]:
    from .state import load_jsonl

    return load_jsonl(layout.ledgers / APPLIED_LEDGER_FILENAME)


def applied_history(project_root: str | Path) -> list[dict[str, object]]:
    """The full applied/reverted ledger, oldest first."""

    return _applied_records(initialize(project_root))


def current_baseline(project_root: str | Path) -> str | None:
    """The prompt hash the next batch should be compared against."""

    for record in reversed(applied_history(project_root)):
        if record.get("event") == "applied":
            return str(record.get("applied_prompt_hash"))
        if record.get("event") == "reverted":
            return str(record.get("restored_prompt_hash"))
    return None


def mark_stale(project_root: str | Path, run_id: str, reason: str = "out-of-band-edit") -> None:
    """Record that a proposal no longer matches the file it was computed against."""

    layout = initialize(project_root)
    append_jsonl(
        layout.ledgers / APPLIED_LEDGER_FILENAME,
        {"event": "stale", "run_id": run_id, "reason": reason},
    )


def is_stale(
    proposal: Proposal, *, base_directory: str | Path = "."
) -> bool:
    """Whether the template has changed out of band since the proposal."""

    try:
        template = contained_path(base_directory, proposal.template_file)
        return text_hash(template.read_text(encoding="utf-8")) != proposal.base_prompt_hash
    except (OSError, PathContainmentError):
        return True
