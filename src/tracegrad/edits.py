"""The edit engine: resolving proposed edits onto real spans and applying them.

This module is pure and model-free, which is the point.  A proposal is a claim
about a span; this module decides whether that claim still holds against the
current prompt, and if it does, produces the exact bytes.  Nothing here trusts
the text a model returned beyond using it as replacement content.

Application is right to left so that earlier offsets stay valid, and overlapping
edits are rejected rather than merged — two proposals fighting over one span is
a disagreement, not a merge conflict to be resolved silently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from .inventory import Instruction, Inventory, fingerprint_text
from .normalize import N, Normalized, normalized_text  # noqa: F401  (N is the public name)
from .schema import Edit

START_SENTINEL = "START"
END_SENTINELS = frozenset({"END", "APPEND"})

_MARKER = re.compile(r"^(\s*)([-*+•]|\d+[.)]|\(\d+\))(\s+)")

REASON_UNKNOWN_ANCHOR = "orphan-anchor"
REASON_CHECKSUM_MISMATCH = "checksum-mismatch"
REASON_NOT_EDITABLE = "variable-span"
REASON_OVERLAP = "overlapping-span"
REASON_EMPTY_TEXT = "empty-text"
REASON_UNKNOWN_OPERATION = "unknown-operation"


@dataclass(frozen=True)
class Rejection:
    """One edit dropped, with the named reason it was dropped."""

    edit: Edit
    reason: str
    detail: str = ""


@dataclass(frozen=True)
class ResolvedEdit:
    """An edit bound to a concrete byte range of the current prompt."""

    edit: Edit
    start: int
    end: int
    replacement: str
    anchor: Instruction | None

    @property
    def operation(self) -> str:
        return self.edit.operation

    @property
    def is_insertion(self) -> bool:
        return self.start == self.end


@dataclass(frozen=True)
class Resolution:
    """The outcome of resolving a proposal set against one prompt."""

    resolved: tuple[ResolvedEdit, ...]
    rejected: tuple[Rejection, ...]

    @property
    def accepted_edits(self) -> tuple[Edit, ...]:
        return tuple(item.edit for item in self.resolved)


def _checksum_of(instruction_id: str) -> str | None:
    """Recover the content checksum embedded in a lineage-addressed id."""

    parts = instruction_id.split("-")
    if len(parts) != 3 or parts[0] != "i":
        return None
    return parts[1]


def _indentation_and_marker(text: str) -> tuple[str, str]:
    match = _MARKER.match(text)
    if not match:
        leading = text[: len(text) - len(text.lstrip(" \t"))]
        return leading, ""
    indentation, marker, gap = match.groups()
    return indentation, f"{marker}{gap}"


def _sibling_text(anchor: Instruction, text: str) -> str:
    """Format added text as a sibling of ``anchor``, inheriting its shape."""

    indentation, marker = _indentation_and_marker(anchor.text)
    body = text.strip()
    if marker and _MARKER.match(body):
        return f"{indentation}{body}"
    return f"{indentation}{marker}{body}"


def _rewrite_text(anchor: Instruction, text: str) -> str:
    indentation, marker = _indentation_and_marker(anchor.text)
    body = text.strip()
    if _MARKER.match(body):
        return f"{indentation}{body}"
    return f"{indentation}{marker}{body}"


def resolve_edits(
    inventory: Inventory, edits: Sequence[Edit], *, allow_sentinels: bool = True
) -> Resolution:
    """Bind each edit to a span of ``inventory.prompt`` or reject it by name.

    Rejection is always per edit.  A bad proposal never aborts the run — it is
    dropped with a reason a human can read in the review card.
    """

    prompt = inventory.prompt
    candidates: list[ResolvedEdit] = []
    rejected: list[Rejection] = []

    for edit in edits:
        operation = edit.operation
        if operation not in {"ADD", "REWRITE", "DELETE"}:
            rejected.append(Rejection(edit, REASON_UNKNOWN_OPERATION, operation))
            continue
        if operation in {"ADD", "REWRITE"} and not edit.text.strip():
            rejected.append(Rejection(edit, REASON_EMPTY_TEXT))
            continue

        if allow_sentinels and edit.instruction_id == START_SENTINEL:
            if operation != "ADD":
                rejected.append(Rejection(edit, REASON_UNKNOWN_OPERATION, "sentinel requires ADD"))
                continue
            candidates.append(ResolvedEdit(edit, 0, 0, edit.text.strip() + "\n", None))
            continue
        if allow_sentinels and edit.instruction_id in END_SENTINELS:
            if operation != "ADD":
                rejected.append(Rejection(edit, REASON_UNKNOWN_OPERATION, "sentinel requires ADD"))
                continue
            tail = len(prompt)
            prefix = "" if prompt.endswith("\n") or not prompt else "\n"
            candidates.append(ResolvedEdit(edit, tail, tail, prefix + edit.text.strip() + "\n", None))
            continue

        anchor = inventory.get(edit.instruction_id)
        if anchor is None:
            rejected.append(Rejection(edit, REASON_UNKNOWN_ANCHOR, edit.instruction_id))
            continue

        expected_checksum = _checksum_of(edit.instruction_id)
        if expected_checksum and not fingerprint_text(anchor.text).startswith(expected_checksum):
            rejected.append(Rejection(edit, REASON_CHECKSUM_MISMATCH, edit.instruction_id))
            continue
        if not anchor.editable:
            rejected.append(Rejection(edit, REASON_NOT_EDITABLE, anchor.origin))
            continue

        if operation == "ADD":
            insertion_point = anchor.end
            candidates.append(
                ResolvedEdit(
                    edit,
                    insertion_point,
                    insertion_point,
                    "\n" + _sibling_text(anchor, edit.text),
                    anchor,
                )
            )
        elif operation == "REWRITE":
            candidates.append(
                ResolvedEdit(edit, anchor.start, anchor.end, _rewrite_text(anchor, edit.text), anchor)
            )
        else:
            start = anchor.start
            end = anchor.end
            if end < len(prompt) and prompt[end] == "\n":
                end += 1
            elif start > 0 and prompt[start - 1] == "\n":
                start -= 1
            candidates.append(ResolvedEdit(edit, start, end, "", anchor))

    accepted: list[ResolvedEdit] = []
    for candidate in sorted(candidates, key=lambda item: (item.start, item.end, item.edit.instruction_id)):
        if any(_overlaps(candidate, existing) for existing in accepted):
            rejected.append(Rejection(candidate.edit, REASON_OVERLAP, candidate.edit.instruction_id))
            continue
        accepted.append(candidate)

    return Resolution(tuple(accepted), tuple(rejected))


def _overlaps(left: ResolvedEdit, right: ResolvedEdit) -> bool:
    if left.is_insertion and right.is_insertion:
        return left.start == right.start
    if left.is_insertion:
        return right.start < left.start < right.end
    if right.is_insertion:
        return left.start < right.start < left.end
    return left.start < right.end and right.start < left.end


def apply_resolved(prompt: str, resolved: Iterable[ResolvedEdit]) -> str:
    """Apply resolved edits right to left, so earlier offsets stay valid."""

    result = prompt
    for item in sorted(resolved, key=lambda edit: (edit.start, edit.end), reverse=True):
        result = result[: item.start] + item.replacement + result[item.end :]
    return result


def apply_edits(inventory: Inventory, edits: Sequence[Edit]) -> tuple[str, Resolution]:
    """Resolve and apply a proposal set, returning the new prompt and outcomes."""

    resolution = resolve_edits(inventory, edits)
    return apply_resolved(inventory.prompt, resolution.resolved), resolution
