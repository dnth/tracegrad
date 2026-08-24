"""Text normalization with an offset map back to the original characters.

Span resolution has to survive the cosmetic differences between what a model
echoes back and what the prompt file actually contains: smart quotes, em
dashes, non-breaking spaces, compatibility ligatures, wrapped whitespace.  ``N``
folds all of those, and returns the map that lets a match in normalized space be
applied to the original bytes — because the file, not the normalized view, is
what gets written.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

NORMALIZER_VERSION = 1

_QUOTE_FOLDING = {
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
    "′": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "‟": '"',
    "″": '"',
}
_DASH_FOLDING = {
    "‐": "-",
    "‑": "-",
    "‒": "-",
    "–": "-",
    "—": "-",
    "―": "-",
    "−": "-",
}
_FOLDING = {**_QUOTE_FOLDING, **_DASH_FOLDING}


@dataclass(frozen=True)
class Normalized:
    """Normalized text plus the index of the original character behind each one."""

    text: str
    offsets: tuple[int, ...]
    original_length: int

    def original_span(self, start: int, end: int) -> tuple[int, int]:
        """Map a half-open normalized range back onto the original text."""

        if start < 0 or end > len(self.text) or start > end:
            raise ValueError(f"normalized span out of range: [{start}, {end})")
        if start == end:
            anchor = self.offsets[start] if start < len(self.offsets) else self.original_length
            return anchor, anchor
        first = self.offsets[start]
        last = self.offsets[end - 1]
        return first, last + 1


def N(text: str) -> Normalized:
    """Normalize ``text`` (NFKC, folding, whitespace collapse) with an offset map."""

    characters: list[str] = []
    offsets: list[int] = []
    pending_space = False
    for index, character in enumerate(text):
        if character.isspace():
            pending_space = bool(characters)
            continue
        folded = _FOLDING.get(character, character)
        expanded = unicodedata.normalize("NFKC", folded)
        if not expanded:
            continue
        if pending_space:
            characters.append(" ")
            offsets.append(index)
            pending_space = False
        for produced in expanded:
            characters.append(produced)
            offsets.append(index)
    return Normalized("".join(characters), tuple(offsets), len(text))


def normalized_text(text: str) -> str:
    """Return only the normalized form, for fingerprints and comparisons."""

    return N(text).text
