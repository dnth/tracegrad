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
    """Normalize ``text`` (NFKC, folding, whitespace collapse) with an offset map.

    NFKC is applied to whole runs rather than per character: a combining
    sequence like ``e`` + U+0301 only composes when its characters are seen
    together, and decomposed-versus-precomposed is exactly the cosmetic
    difference this module exists to fold.  Normalizing per character silently
    leaves the two forms unequal, which shows up much later as a quote that
    will not verify or an anchor that will not resolve.
    """

    characters: list[str] = []
    offsets: list[int] = []
    pending_space: int | None = None
    run: list[str] = []
    run_offsets: list[int] = []

    def flush_run() -> None:
        if not run:
            return
        composed = unicodedata.normalize("NFKC", "".join(run))
        # Map each produced character back to a source index. Composition only
        # ever shortens a run, so pair them off in order and let any remainder
        # share the last source index.
        for position, produced in enumerate(composed):
            characters.append(produced)
            offsets.append(run_offsets[min(position, len(run_offsets) - 1)])
        run.clear()
        run_offsets.clear()

    for index, character in enumerate(text):
        if character.isspace():
            flush_run()
            if characters and pending_space is None:
                # Remember where the whitespace run STARTED, so a normalized
                # range covering the collapsed space maps back over the whole
                # run rather than just its last character.
                pending_space = index
            continue
        if pending_space is not None:
            flush_run()
            characters.append(" ")
            # The collapsed space stands for the whitespace run itself, so it
            # carries the index of that run's first character. Anything else
            # makes original_span drop the space from the range it returns.
            offsets.append(pending_space)
            pending_space = None
        folded = _FOLDING.get(character, character)
        run.append(folded)
        run_offsets.append(index)
    flush_run()
    return Normalized("".join(characters), tuple(offsets), len(text))


def normalized_text(text: str) -> str:
    """Return only the normalized form, for fingerprints and comparisons."""

    return N(text).text
