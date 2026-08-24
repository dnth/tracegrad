import pytest

from tracegrad.normalize import N, normalized_text


def test_n_folds_smart_quotes_and_dashes() -> None:
    normalized = N("‘hi’ — “there”")

    assert normalized.text == "'hi' - \"there\""


def test_n_applies_nfkc_compatibility_decomposition() -> None:
    # U+FB01 LATIN SMALL LIGATURE FI decomposes to "fi" under NFKC.
    normalized = N("ﬁle")

    assert normalized.text == "file"


def test_n_collapses_whitespace_runs_to_single_space() -> None:
    normalized = N("a\t\n  b")

    assert normalized.text == "a b"


def test_n_strips_leading_and_trailing_whitespace() -> None:
    normalized = N("   a b   ")

    assert normalized.text == "a b"


def test_n_original_span_matches_hand_computed_offsets() -> None:
    # "a  b": index 0='a', 1=' ', 2=' ', 3='b'.
    # The two interior spaces collapse to one, which stands for the whole run
    # and so carries the index the run STARTS at. Anchoring it at the end
    # instead would make a span covering the space map back to a range with the
    # spaces cut off.
    normalized = N("a  b")

    assert normalized.text == "a b"
    assert normalized.offsets == (0, 1, 3)
    assert normalized.original_length == 4

    # "a" -> normalized [0, 1) -> original [0, 1)
    assert normalized.original_span(0, 1) == (0, 1)
    # " " -> normalized [1, 2) -> original [1, 2), the start of the space run.
    # A span that ENDS on a collapsed space is trimmed to that run's first
    # character; only spans ending on real content need to be exact, and those
    # are (see the round-trip property test below).
    assert normalized.original_span(1, 2) == (1, 2)
    # "b" -> normalized [2, 3) -> original [3, 4)
    assert normalized.original_span(2, 3) == (3, 4)
    # "a b" -> normalized [0, 3) -> original [0, 4)
    assert normalized.original_span(0, 3) == (0, 4)


@pytest.mark.parametrize(
    "text",
    [
        "a  b",
        "Be  \u201ckind\u201d\u2014always.",
        "caf\u00e9 ok",
        "  x\ty  z  ",
        "one   two three",
        "a\n\nb",
    ],
)
def test_every_normalized_span_maps_back_to_text_that_normalizes_to_it(text: str) -> None:
    # The property that matters, rather than the offsets themselves: whatever
    # original range a normalized span maps to must normalize back to that span.
    # This is what lets a match found in normalized space be applied to the file.
    normalized = N(text)

    for start in range(len(normalized.text)):
        for end in range(start + 1, len(normalized.text) + 1):
            first, last = normalized.original_span(start, end)
            expected = normalized.text[start:end]
            # A span ending on a collapsed space maps back to a range whose own
            # normalization drops that trailing space, so compare stripped.
            assert normalized_text(text[first:last]) == expected.strip()


def test_n_original_span_empty_span_uses_anchor() -> None:
    normalized = N("a  b")

    # Empty span in the interior anchors at the offset of that index.
    assert normalized.original_span(1, 1) == (1, 1)
    # Empty span at the very end anchors at original_length.
    assert normalized.original_span(3, 3) == (4, 4)


def test_decomposed_and_precomposed_forms_normalize_alike() -> None:
    # NFKC only composes when a combining sequence is seen whole. Applying it
    # per character leaves these unequal, which surfaces much later as a quote
    # that will not verify.
    assert normalized_text("caf\u0065\u0301") == normalized_text("caf\u00e9")


def test_normalized_text_matches_n_text() -> None:
    text = "  Hello ‘World’  "

    assert normalized_text(text) == N(text).text
