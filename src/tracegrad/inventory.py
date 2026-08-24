"""Segmentation of a rendered prompt into addressable instructions.

An edit is only as stable as the address it targets, so segmentation is
versioned and deterministic.  Bullets and numbered items are the natural unit
when a prompt has them; free prose falls back to abbreviation-safe sentence
splitting.  Every instruction carries a content fingerprint and a ``lineage_id``
that survives reordering and reindentation — reworded text is a new lineage, and
that is deliberate: a rewrite should not inherit another instruction's history.

Spans that came from variable substitution are marked non-editable.  tracegrad
edits the template, and a variable's value is not in the template.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from .canonical import sha256_hex
from .distill import RenderedPrompt
from .normalize import NORMALIZER_VERSION, normalized_text

SEGMENTER_VERSION = 2

_BULLET = re.compile(r"^(\s*)([-*+•]|\d+[.)]|\(\d+\))\s+")
_HEADING = re.compile(r"^\s*(#{1,6}\s+\S|[A-Z][A-Za-z0-9 /-]{0,60}:\s*$)")

_ABBREVIATIONS = frozenset(
    {
        "e.g.",
        "i.e.",
        "etc.",
        "vs.",
        "cf.",
        "approx.",
        "resp.",
        "al.",
        "no.",
        "fig.",
        "mr.",
        "mrs.",
        "ms.",
        "dr.",
        "prof.",
        "sr.",
        "jr.",
        "st.",
        "inc.",
        "ltd.",
        "co.",
        "u.s.",
        "u.k.",
        "a.m.",
        "p.m.",
    }
)
_SENTENCE_BOUNDARY = re.compile(r"[.!?]['\")\]]*(\s+)")
_INITIAL = re.compile(r"(?:^|\s)[A-Z]\.$")


class InventoryError(ValueError):
    """A prompt that cannot be segmented into instructions."""


@dataclass(frozen=True)
class Instruction:
    """One addressable unit of the prompt."""

    instruction_id: str
    lineage_id: str
    text: str
    start: int
    end: int
    kind: str
    ordinal: int
    fingerprint: str
    origin: str
    editable: bool
    segmenter_version: int = SEGMENTER_VERSION

    @property
    def normalized(self) -> str:
        return normalized_text(self.text)


@dataclass(frozen=True)
class Inventory:
    """The segmented prompt: ordered instructions plus lookup by address."""

    prompt: str
    instructions: tuple[Instruction, ...]
    segmenter_version: int = SEGMENTER_VERSION
    normalizer_version: int = NORMALIZER_VERSION

    def __iter__(self) -> Iterable[Instruction]:
        return iter(self.instructions)

    def __len__(self) -> int:
        return len(self.instructions)

    def get(self, instruction_id: str) -> Instruction | None:
        for instruction in self.instructions:
            if instruction.instruction_id == instruction_id:
                return instruction
        return None

    @property
    def editable(self) -> tuple[Instruction, ...]:
        return tuple(item for item in self.instructions if item.editable)


def fingerprint_text(text: str) -> str:
    """Content fingerprint of an instruction, invariant to cosmetic differences."""

    return sha256_hex(f"v{SEGMENTER_VERSION}\x1f{normalized_text(text)}")


def _lineage_id(fingerprint: str, ordinal: int) -> str:
    return f"i-{fingerprint[:12]}-{ordinal:02d}"


def _ends_with_abbreviation(text: str) -> bool:
    tail = text.rstrip()
    if _INITIAL.search(tail):
        return True
    last_token = tail.split()[-1].lower() if tail.split() else ""
    return last_token in _ABBREVIATIONS


def split_sentences(text: str, offset: int = 0) -> list[tuple[int, int]]:
    """Split prose into abbreviation-safe sentence spans, as absolute offsets."""

    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_BOUNDARY.finditer(text):
        boundary = match.end() - len(match.group(1))
        candidate = text[start:boundary]
        if _ends_with_abbreviation(candidate):
            continue
        following = text[match.end() :]
        if following and not (following[0].isupper() or following[0] in "\"'“‘-*•(#"):
            continue
        spans.append((offset + start, offset + boundary))
        start = match.end()
    if text[start:].strip():
        spans.append((offset + start, offset + len(text.rstrip())))
    return spans


def _line_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for line in text.split("\n"):
        spans.append((cursor, cursor + len(line)))
        cursor += len(line) + 1
    return spans


def _block_spans(prompt: str) -> list[tuple[str, int, int]]:
    """Group lines into bullet items, headings, and prose paragraphs."""

    blocks: list[tuple[str, int, int]] = []
    paragraph: list[tuple[int, int]] = []

    def flush_paragraph() -> None:
        if not paragraph:
            return
        blocks.append(("prose", paragraph[0][0], paragraph[-1][1]))
        paragraph.clear()

    for start, end in _line_spans(prompt):
        line = prompt[start:end]
        if not line.strip():
            flush_paragraph()
            continue
        bullet = _BULLET.match(line)
        if bullet:
            flush_paragraph()
            blocks.append(("bullet", start, end))
            continue
        if _HEADING.match(line) and not paragraph:
            blocks.append(("heading", start, end))
            continue
        # A continuation line only belongs to the bullet if nothing else has
        # started since. Without the paragraph check the bullet's span grows
        # past intervening prose, and the flushed paragraph then starts inside
        # it — overlapping, out-of-order spans that break every offset downstream.
        if (
            not paragraph
            and blocks
            and blocks[-1][0] == "bullet"
            and blocks[-1][2] == start - 1
            and line.startswith((" ", "\t"))
        ):
            kind, block_start, _ = blocks.pop()
            blocks.append((kind, block_start, end))
            continue
        paragraph.append((start, end))
    flush_paragraph()
    return blocks


def segment(prompt: str) -> list[tuple[str, int, int]]:
    """Return ``(kind, start, end)`` spans for every instruction in ``prompt``."""

    segments: list[tuple[str, int, int]] = []
    for kind, start, end in _block_spans(prompt):
        if kind in {"bullet", "heading"}:
            segments.append((kind, start, end))
            continue
        for sentence_start, sentence_end in split_sentences(prompt[start:end], start):
            segments.append(("sentence", sentence_start, sentence_end))
    return segments


def build_inventory(
    prompt: str | RenderedPrompt,
    variable_spans: Sequence[object] | None = None,
) -> Inventory:
    """Segment a rendered prompt and assign stable, duplicate-safe lineage ids."""

    if isinstance(prompt, RenderedPrompt):
        rendered: RenderedPrompt | None = prompt
        text = prompt.text
    else:
        rendered = None
        text = prompt

    def is_variable(start: int, end: int) -> bool:
        if rendered is not None:
            return rendered.is_variable(start, end)
        for span in variable_spans or ():
            span_start = getattr(span, "start", None)
            span_end = getattr(span, "end", None)
            if span_start is None:
                span_start, span_end = span  # type: ignore[misc]
            if span_start < end and start < span_end:
                return True
        return False

    instructions: list[Instruction] = []
    ordinals: dict[str, int] = {}
    for kind, start, end in segment(text):
        body = text[start:end]
        if not body.strip():
            continue
        fingerprint = fingerprint_text(body)
        ordinal = ordinals.get(fingerprint, 0)
        ordinals[fingerprint] = ordinal + 1
        lineage_id = _lineage_id(fingerprint, ordinal)
        from_variable = is_variable(start, end)
        instructions.append(
            Instruction(
                instruction_id=lineage_id,
                lineage_id=lineage_id,
                text=body,
                start=start,
                end=end,
                kind=kind,
                ordinal=ordinal,
                fingerprint=fingerprint,
                origin="variable" if from_variable else "template",
                editable=not from_variable,
            )
        )
    return Inventory(prompt=text, instructions=tuple(instructions))
