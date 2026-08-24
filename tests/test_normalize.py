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
    # The two interior spaces collapse to a single space anchored at the
    # index of the character that ends the pending-space run.
    normalized = N("a  b")

    assert normalized.text == "a b"
    assert normalized.offsets == (0, 3, 3)
    assert normalized.original_length == 4

    # "a" -> normalized [0, 1) -> original [0, 1)
    assert normalized.original_span(0, 1) == (0, 1)
    # " " -> normalized [1, 2) -> original [3, 4), the offset of 'b'
    assert normalized.original_span(1, 2) == (3, 4)
    # "b" -> normalized [2, 3) -> original [3, 4)
    assert normalized.original_span(2, 3) == (3, 4)
    # "a b" -> normalized [0, 3) -> original [0, 4)
    assert normalized.original_span(0, 3) == (0, 4)


def test_n_original_span_empty_span_uses_anchor() -> None:
    normalized = N("a  b")

    # Empty span in the interior anchors at the offset of that index.
    assert normalized.original_span(1, 1) == (3, 3)
    # Empty span at the very end anchors at original_length.
    assert normalized.original_span(3, 3) == (4, 4)


def test_normalized_text_matches_n_text() -> None:
    text = "  Hello ‘World’  "

    assert normalized_text(text) == N(text).text
