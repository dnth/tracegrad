from pathlib import Path

import pytest

from tracegrad.canonical import text_hash
from tracegrad.distill import (
    DEFAULT_MAX_INPUT_CHARS,
    DistillConfig,
    DistilledTrace,
    DistillError,
    _Redactor,
    distill_trace,
    normalize_whitespace,
    reduce_text,
    render_template,
    store_batch,
    store_distilled,
)
from tracegrad.schema import Judge, TemplateEngine, Trace


def _distilled(
    trace_id: str = "trace-1",
    input_text: str = "the input",
    output: str = "the output",
    score: float = 0.9,
    rationale: str = "the rationale",
    prompt_hash: str = "sha256:p1",
    model: str | None = "gpt-4",
    distill_config_hash: str = "sha256:cfg",
) -> DistilledTrace:
    return DistilledTrace(
        trace_id=trace_id,
        input=input_text,
        output=output,
        score=score,
        rationale=rationale,
        prompt_hash=prompt_hash,
        model=model,
        distill_config_hash=distill_config_hash,
        redactions={},
    )


# --- render_template ----------------------------------------------------------


def test_render_template_engine_none_is_verbatim() -> None:
    template = "Hello {name}, this is literal."

    rendered = render_template(template, TemplateEngine.NONE)

    assert rendered.text == template
    assert rendered.variable_spans == ()
    assert rendered.engine is TemplateEngine.NONE
    assert rendered.template_hash == text_hash(template)
    assert rendered.prompt_hash == text_hash(template)


def test_render_template_engine_format_tracks_variable_spans() -> None:
    template = "Hello {name}, you are {age}."

    rendered = render_template(
        template, TemplateEngine.FORMAT, variables={"name": "Ada", "age": "30"}
    )

    assert rendered.text == "Hello Ada, you are 30."
    assert rendered.text[6:9] == "Ada"
    assert rendered.text[19:21] == "30"
    assert [(span.start, span.end, span.name) for span in rendered.variable_spans] == [
        (6, 9, "name"),
        (19, 21, "age"),
    ]


def test_render_template_is_variable_detects_overlap_and_boundaries() -> None:
    template = "Hello {name}, you are {age}."
    rendered = render_template(
        template, TemplateEngine.FORMAT, variables={"name": "Ada", "age": "30"}
    )

    assert rendered.is_variable(6, 9) is True  # exact span
    assert rendered.is_variable(8, 10) is True  # overlaps into literal
    assert rendered.is_variable(0, 3) is False  # entirely literal
    assert rendered.is_variable(9, 10) is False  # adjacent, not overlapping


def test_render_template_raises_on_undeclared_variable() -> None:
    with pytest.raises(DistillError, match="not declared"):
        render_template("Hi {name}", TemplateEngine.FORMAT, variables={})


def test_render_template_raises_on_positional_field() -> None:
    with pytest.raises(DistillError, match="positional"):
        render_template("Hi {0}", TemplateEngine.FORMAT, variables={"0": "x"})


def test_render_template_raises_on_format_spec() -> None:
    with pytest.raises(DistillError, match="must be plain"):
        render_template("Hi {name:>10}", TemplateEngine.FORMAT, variables={"name": "x"})


def test_render_template_raises_on_unsupported_engine() -> None:
    with pytest.raises(DistillError, match="unsupported template engine"):
        render_template("Hi", "jinja-basic")  # type: ignore[arg-type]


# --- normalize_whitespace ------------------------------------------------------


def test_normalize_whitespace_folds_crlf() -> None:
    assert normalize_whitespace("a\r\nb\r\n") == "a\nb"


def test_normalize_whitespace_strips_trailing_spaces() -> None:
    assert normalize_whitespace("a   \nb  ") == "a\nb"


def test_normalize_whitespace_collapses_three_or_more_blank_lines() -> None:
    assert normalize_whitespace("a\n\n\n\nb") == "a\n\n\nb"


def test_normalize_whitespace_keeps_up_to_two_blank_lines() -> None:
    assert normalize_whitespace("a\n\nb") == "a\n\nb"


# --- reduce_text -----------------------------------------------------------------


def test_reduce_text_is_noop_under_limit() -> None:
    assert reduce_text("short text", 100) == "short text"


def test_reduce_text_is_noop_when_max_chars_non_positive() -> None:
    assert reduce_text("hello", 0) == "hello"
    assert reduce_text("hello", -5) == "hello"


def test_reduce_text_keeps_head_and_tail() -> None:
    text = "0123456789" * 5  # 50 characters
    max_chars = 10

    result = reduce_text(text, max_chars)

    head_chars = (max_chars * 2) // 3  # 6
    tail_chars = max_chars - head_chars  # 4
    assert result.startswith(text[:head_chars])
    assert result.endswith(text[-tail_chars:])
    assert "characters elided" in result
    assert str(len(text) - head_chars - tail_chars) in result


# --- redaction -------------------------------------------------------------------


def test_redactor_placeholders_are_stable_across_repeats() -> None:
    redactor = _Redactor()
    text = "contact a@example.com or a@example.com again"

    result = redactor.redact(text)

    assert result.count("[REDACTED:EMAIL:1]") == 2
    assert "a@example.com" not in result


def test_redactor_numbers_in_first_appearance_order() -> None:
    redactor = _Redactor()
    text = "first b@example.com then a@example.com then b@example.com"

    result = redactor.redact(text)

    assert "[REDACTED:EMAIL:1]" in result
    assert "[REDACTED:EMAIL:2]" in result
    assert result.index("[REDACTED:EMAIL:1]") < result.index("[REDACTED:EMAIL:2]")
    assert redactor.mapping["[REDACTED:EMAIL:1]"] == "EMAIL"
    assert redactor.mapping["[REDACTED:EMAIL:2]"] == "EMAIL"


def test_redactor_counts_each_kind_independently() -> None:
    redactor = _Redactor()
    text = (
        "email a@example.com, url https://example.com/x, "
        "uuid 123e4567-e89b-12d3-a456-426614174000, "
        "key sk-ABCDEFGHIJKLMNOPQRSTUV"
    )

    result = redactor.redact(text)

    assert "[REDACTED:EMAIL:1]" in result
    assert "[REDACTED:URL:1]" in result
    assert "[REDACTED:UUID:1]" in result
    assert "[REDACTED:KEY:1]" in result


# --- distill_config_hash -----------------------------------------------------------


def test_distill_config_hash_stable_for_equivalent_config() -> None:
    assert DistillConfig().config_hash == DistillConfig().config_hash


def test_distill_config_hash_changes_with_max_input_chars() -> None:
    base = DistillConfig()
    changed = DistillConfig(max_input_chars=base.max_input_chars + 1)

    assert base.config_hash != changed.config_hash


def test_distill_config_hash_changes_with_redact_flag() -> None:
    base = DistillConfig(redact=True)
    changed = DistillConfig(redact=False)

    assert base.config_hash != changed.config_hash


# --- DistilledTrace.quotable --------------------------------------------------------


def test_distilled_trace_quotable_output() -> None:
    distilled = _distilled(output="the output text")

    assert distilled.quotable("output") == "the output text"


def test_distilled_trace_quotable_distilled_joins_fields() -> None:
    distilled = _distilled(input_text="in", output="out", rationale="why")

    assert distilled.quotable("distilled") == "in\nout\nwhy"


def test_distilled_trace_quotable_unknown_source_raises() -> None:
    distilled = _distilled()

    with pytest.raises(DistillError, match="unknown quote source"):
        distilled.quotable("bogus")


# --- content_address ------------------------------------------------------------------


def test_content_address_is_stable_for_equivalent_records() -> None:
    first = _distilled()
    second = _distilled()

    assert first.content_address == second.content_address


def test_content_address_changes_with_content() -> None:
    first = _distilled(output="a")
    second = _distilled(output="b")

    assert first.content_address != second.content_address


# --- distill_trace end-to-end ----------------------------------------------------------


def test_distill_trace_reduces_and_redacts() -> None:
    trace = Trace(
        trace_id="trace-1",
        input="write to a@example.com please",
        output="ok, emailing a@example.com now",
        judge=Judge(score=0.8, rationale="reasonable rationale text"),
        prompt_hash="sha256:p1",
        meta=None,
    )

    distilled = distill_trace(trace)

    assert "a@example.com" not in distilled.input
    assert "a@example.com" not in distilled.output
    assert distilled.redactions
    assert distilled.distill_config_hash == DistillConfig().config_hash
    assert distilled.model is None


def test_distill_trace_without_redaction_leaves_secrets() -> None:
    trace = Trace(
        trace_id="trace-1",
        input="write to a@example.com please",
        output="ok",
        judge=Judge(score=0.8, rationale="reasonable rationale text"),
        prompt_hash="sha256:p1",
    )

    distilled = distill_trace(trace, DistillConfig(redact=False))

    assert "a@example.com" in distilled.input
    assert distilled.redactions == {}


# --- store_distilled / store_batch --------------------------------------------------------


def test_store_distilled_writes_content_addressed_path(tmp_path: Path) -> None:
    distilled = _distilled(trace_id="t1")

    path = store_distilled(tmp_path, distilled)

    assert path.exists()
    assert distilled.content_address.split(":", 1)[-1] in str(path)


def test_store_distilled_is_idempotent(tmp_path: Path) -> None:
    distilled = _distilled(trace_id="t1")

    path = store_distilled(tmp_path, distilled)
    sentinel = "not the real content"
    path.write_text(sentinel, encoding="utf-8")

    path_again = store_distilled(tmp_path, distilled)

    assert path_again == path
    assert path.read_text(encoding="utf-8") == sentinel  # second call did not rewrite


def test_store_batch_maps_trace_id_to_path(tmp_path: Path) -> None:
    batch = (_distilled(trace_id="t1"), _distilled(trace_id="t2", output="different"))

    result = store_batch(tmp_path, batch)

    assert set(result) == {"t1", "t2"}
    assert result["t1"].exists()
    assert result["t2"].exists()
    assert result["t1"] != result["t2"]


def test_default_max_input_chars_is_positive() -> None:
    assert DEFAULT_MAX_INPUT_CHARS > 0
