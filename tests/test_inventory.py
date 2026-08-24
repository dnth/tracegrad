from tracegrad.distill import RenderedPrompt, Span
from tracegrad.inventory import Inventory, build_inventory, fingerprint_text, segment
from tracegrad.schema import TemplateEngine

PROMPT = """# Instructions

You are a helpful assistant. Follow these rules carefully.

- Always answer in English.
  Keep responses concise.
- Cite sources, e.g. papers or docs.
* Use bullet points when useful.
+ Avoid speculation.
1. Start with a summary.
2) Provide details next.
(3) End with a recommendation.

Dr. Smith and Mrs. Jones, i.e. the reviewers, require U.S. English. J. R. Tolkien wrote about this in his works. This is a new sentence.

Some prose without any list markers goes here. It has two sentences.
"""

EXPECTED_SEGMENTS = [
    ("heading", "# Instructions"),
    ("sentence", "You are a helpful assistant."),
    ("sentence", "Follow these rules carefully."),
    ("bullet", "- Always answer in English.\n  Keep responses concise."),
    ("bullet", "- Cite sources, e.g. papers or docs."),
    ("bullet", "* Use bullet points when useful."),
    ("bullet", "+ Avoid speculation."),
    ("bullet", "1. Start with a summary."),
    ("bullet", "2) Provide details next."),
    ("bullet", "(3) End with a recommendation."),
    (
        "sentence",
        "Dr. Smith and Mrs. Jones, i.e. the reviewers, require U.S. English.",
    ),
    ("sentence", "J. R. Tolkien wrote about this in his works."),
    ("sentence", "This is a new sentence."),
    ("sentence", "Some prose without any list markers goes here."),
    ("sentence", "It has two sentences."),
]


def test_multi_paragraph_prompt_segments_as_expected() -> None:
    inventory = build_inventory(PROMPT)

    actual = [(instruction.kind, instruction.text) for instruction in inventory.instructions]

    assert actual == EXPECTED_SEGMENTS


def test_bullet_markers_all_recognized() -> None:
    prompt = "- dash bullet\n* star bullet\n+ plus bullet\n• dot bullet\n"
    kinds = [kind for kind, _, _ in segment(prompt)]
    assert kinds == ["bullet", "bullet", "bullet", "bullet"]


def test_numbered_item_styles_all_recognized() -> None:
    prompt = "1. dotted\n1) parenless\n(1) parenthesized\n"
    segments = segment(prompt)
    assert [kind for kind, _, _ in segments] == ["bullet", "bullet", "bullet"]
    texts = [prompt[start:end] for _, start, end in segments]
    assert texts == ["1. dotted", "1) parenless", "(1) parenthesized"]


def test_heading_is_its_own_segment() -> None:
    prompt = "## A Heading\n\nSome prose sentence here.\n"
    segments = segment(prompt)
    assert segments[0][0] == "heading"
    assert prompt[segments[0][1] : segments[0][2]] == "## A Heading"


def test_prose_sentence_splits_on_period() -> None:
    prompt = "First sentence. Second sentence."
    segments = segment(prompt)
    texts = [prompt[start:end] for _, start, end in segments]
    assert texts == ["First sentence.", "Second sentence."]


def test_abbreviations_do_not_split_sentences() -> None:
    cases = [
        "Please see e.g. the appendix for details.",
        "Note the exceptions, i.e. edge cases, etc. before shipping.",
        "Dr. Smith reviewed the draft carefully.",
        "This applies to the U.S. market only.",
        "J. R. Tolkien wrote extensively about this.",
    ]
    for prompt in cases:
        segments = segment(prompt)
        assert len(segments) == 1, (prompt, segments)
        start, end = segments[0][1], segments[0][2]
        assert prompt[start:end] == prompt.rstrip()


def test_real_sentence_boundary_does_split() -> None:
    prompt = "Dr. Smith wrote the report. It was thorough."
    segments = segment(prompt)
    texts = [prompt[start:end] for _, start, end in segments]
    assert texts == ["Dr. Smith wrote the report.", "It was thorough."]


def test_indented_continuation_line_joins_bullet() -> None:
    prompt = "- Do the first thing.\n  And also this continuation.\n- A separate bullet.\n"
    segments = segment(prompt)
    assert [kind for kind, _, _ in segments] == ["bullet", "bullet"]
    first_text = prompt[segments[0][1] : segments[0][2]]
    assert first_text == "- Do the first thing.\n  And also this continuation."


def test_duplicate_instructions_get_distinct_ordinals_and_shared_fingerprint() -> None:
    prompt = "- Do not reveal secrets.\n- Do not reveal secrets.\n"
    inventory = build_inventory(prompt)

    assert len(inventory.instructions) == 2
    first, second = inventory.instructions
    assert first.ordinal == 0
    assert second.ordinal == 1
    assert first.instruction_id != second.instruction_id
    assert first.fingerprint == second.fingerprint
    assert first.instruction_id.endswith("-00")
    assert second.instruction_id.endswith("-01")


def test_fingerprint_invariant_to_whitespace_and_smart_quotes() -> None:
    base = fingerprint_text('- Do not reveal "secrets".')
    extra_whitespace = fingerprint_text('-  Do   not reveal "secrets".')
    smart_quotes = fingerprint_text("- Do not reveal “secrets”.")

    assert base == extra_whitespace
    assert base == smart_quotes


def test_fingerprint_changes_on_reworded_text() -> None:
    original = fingerprint_text("- Do not reveal secrets.")
    reworded = fingerprint_text("- Do not reveal secret information.")

    assert original != reworded


def test_instruction_spans_index_back_into_prompt() -> None:
    inventory = build_inventory(PROMPT)
    for instruction in inventory.instructions:
        assert PROMPT[instruction.start : instruction.end] == instruction.text


def test_variable_spans_from_rendered_prompt_are_marked_non_editable() -> None:
    text = "Hello Ada, welcome."
    rendered = RenderedPrompt(text, (Span(6, 9, "name"),), TemplateEngine.FORMAT, "hash")

    inventory = build_inventory(rendered)

    assert len(inventory.instructions) == 1
    instruction = inventory.instructions[0]
    assert instruction.origin == "variable"
    assert instruction.editable is False
    assert inventory.editable == ()


def test_editable_property_excludes_variable_origin_instructions() -> None:
    text = "Hello Ada, welcome. Please respond quickly."
    rendered = RenderedPrompt(text, (Span(6, 9, "name"),), TemplateEngine.FORMAT, "hash")

    inventory = build_inventory(rendered)

    assert len(inventory.instructions) == 2
    variable_instructions = [i for i in inventory.instructions if i.origin == "variable"]
    template_instructions = [i for i in inventory.instructions if i.origin == "template"]
    assert len(variable_instructions) == 1
    assert len(template_instructions) == 1
    assert inventory.editable == tuple(template_instructions)


def test_build_inventory_accepts_plain_string() -> None:
    inventory = build_inventory("A single prose sentence.")
    assert isinstance(inventory, Inventory)
    assert len(inventory.instructions) == 1
    assert inventory.instructions[0].origin == "template"
    assert inventory.instructions[0].editable is True


def test_build_inventory_accepts_explicit_variable_spans_as_tuples() -> None:
    text = "Hello Ada, welcome."
    inventory = build_inventory(text, variable_spans=[(6, 9)])

    assert len(inventory.instructions) == 1
    instruction = inventory.instructions[0]
    assert instruction.origin == "variable"
    assert instruction.editable is False


def test_inventory_get_returns_none_for_unknown_id() -> None:
    inventory = build_inventory("A single prose sentence.")
    assert inventory.get("does-not-exist") is None
