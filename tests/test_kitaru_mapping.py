"""Kitaru session → Trace mapping (issue #8 definition of done).

These tests do not import the Kitaru SDK.  Nodes and evaluations are
duck-typed so core-only CI stays green without the extra.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tracegrad.canonical import text_hash
from tracegrad.integrations.kitaru.graph import is_root_llm_node, root_llm_nodes
from tracegrad.integrations.kitaru.mapping import (
    REASON_FORMAT_ENGINE_REFUSED,
    REASON_INPUT_UNAVAILABLE,
    REASON_MULTIPLE_SYSTEM_PROMPTS,
    REASON_OUTPUT_UNAVAILABLE,
    REASON_ROOT_LLM_UNAVAILABLE,
    REASON_SYSTEM_PROMPT_UNAVAILABLE,
    MappedTrace,
    SourceDrop,
    map_batch,
    map_session,
)
from tracegrad.integrations.kitaru.pointer import resolve_pointer, resolve_text_selector
from tracegrad.integrations.kitaru.scores import (
    REASON_AMBIGUOUS_EVALUATION,
    REASON_RATIONALE_MISSING,
    REASON_SCORE_OUT_OF_RANGE,
    REASON_SCORE_UNAVAILABLE,
    REASON_SCORE_UNSUPPORTED,
    judge_fingerprint_for,
    map_judge,
    select_evaluation,
)
from tracegrad.integrations.kitaru.source import check_judge_fingerprint, refuse_format_engine
from tracegrad.schema import Manifest, TemplateEngine

SYSTEM = "You are the root agent."
RATIONALE = "The agent skipped the citation the customer asked for."


def _node(
    index: int,
    *,
    node_type: str = "llm_call",
    parent: int | None = None,
    secondary: list[int] | None = None,
    system: str | None = SYSTEM,
    input_text: str = "user question",
    output_text: str = "model answer",
    model: str = "gpt-4.1",
    in_sel: str | None = "/input",
    out_sel: str | None = "/output",
    sys_sel: str | None = "/system",
) -> SimpleNamespace:
    inputs: dict[str, object] = {"input": input_text}
    if system is not None:
        inputs["system"] = system
    return SimpleNamespace(
        index=index,
        parent_index=parent,
        secondary_parent_indexes=list(secondary or []),
        node_type=node_type,
        inputs=inputs,
        outputs={"output": output_text},
        input_text_selector=in_sel,
        output_text_selector=out_sel,
        system_prompt_selector=sys_sel,
        model=model,
    )


def _eval(
    *,
    name: str = "quality",
    score: object = 0.25,
    explanation: str | None = RATIONALE,
    passed: bool | None = False,
    data_type: str = "float",
    version: int | None = 3,
    value: str | None = None,
    eval_id: str = "e1",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=eval_id,
        name=name,
        score=score,
        explanation=explanation,
        passed=passed,
        value=value,
        data_type=data_type,
        evaluator_name="quality",
        evaluator_version=version,
        evaluator_version_id="ev-3",
    )


def _session(session_id: str = "0f3a0000-0000-4000-8000-00000000c19d", number: int = 4811):
    return SimpleNamespace(id=session_id, number=number)


def test_json_pointer_resolves_and_unescapes() -> None:
    document = {"a": [{"b~c": "ok"}], "x/y": "slash"}
    assert resolve_pointer(document, "") is document
    assert resolve_pointer(document, "/a/0/b~0c") == "ok"


def test_selector_failure_is_none_not_a_guess() -> None:
    assert resolve_text_selector({"n": 1}, "/n") is None
    assert resolve_text_selector({"n": "s"}, None) is None
    assert resolve_text_selector({"n": "s"}, "/missing") is None


def test_session_maps_to_a_trace() -> None:
    result = map_session(
        _session(),
        [_node(0)],
        [_eval()],
        "quality",
    )
    assert isinstance(result, MappedTrace)
    assert result.trace.trace_id == "0f3a0000-0000-4000-8000-00000000c19d"
    assert result.trace.input == "user question"
    assert result.trace.output == "model answer"
    assert result.trace.prompt_hash == text_hash(SYSTEM)
    assert result.trace.judge.score == 0.25
    assert result.trace.meta is not None
    assert result.trace.meta.model == "gpt-4.1"
    assert result.session_number == 4811
    assert result.evaluator_version == 3


def test_single_system_prompt_is_accepted() -> None:
    result = map_session(_session(), [_node(0), _node(1, parent=0, system=SYSTEM)], [_eval()], "quality")
    assert isinstance(result, MappedTrace)
    assert result.system_prompt == SYSTEM


def test_missing_system_prompt_drops() -> None:
    node = _node(0, sys_sel="/missing")
    result = map_session(_session(), [node], [_eval()], "quality")
    assert isinstance(result, SourceDrop)
    assert result.reason == REASON_SYSTEM_PROMPT_UNAVAILABLE


def test_empty_system_prompt_is_not_guessed() -> None:
    node = _node(0, system="")
    result = map_session(_session(), [node], [_eval()], "quality")
    assert isinstance(result, SourceDrop)
    assert result.reason == REASON_SYSTEM_PROMPT_UNAVAILABLE


def test_multiple_system_prompts_drop() -> None:
    nodes = [_node(0, system="prompt A"), _node(1, parent=0, system="prompt B")]
    result = map_session(_session(), nodes, [_eval()], "quality")
    assert isinstance(result, SourceDrop)
    assert result.reason == REASON_MULTIPLE_SYSTEM_PROMPTS


def test_selector_resolution_failures_drop_by_name() -> None:
    missing_input = map_session(
        _session(), [_node(0, in_sel="/nope")], [_eval()], "quality"
    )
    missing_output = map_session(
        _session(), [_node(0, out_sel="/nope")], [_eval()], "quality"
    )
    assert isinstance(missing_input, SourceDrop)
    assert missing_input.reason == REASON_INPUT_UNAVAILABLE
    assert isinstance(missing_output, SourceDrop)
    assert missing_output.reason == REASON_OUTPUT_UNAVAILABLE


def test_root_vs_subagent_including_secondary_parent() -> None:
    # 0 root llm, 1 subagent, 2 llm under subagent, 3 root llm,
    # 4 llm with parents 3 (root) and 2 (subagent descendant) — not root.
    nodes = [
        _node(0),
        _node(1, node_type="subagent_call", parent=0, system="subagent prompt"),
        _node(2, parent=1, system="subagent prompt", output_text="subagent said"),
        _node(3, parent=0, input_text="second turn", output_text="last answer"),
        _node(4, parent=3, secondary=[2], system="should not be used"),
        _node(5, node_type="tool_call", parent=3, output_text="TOOL OUTPUT"),
    ]
    by_index = {node.index: node for node in nodes}
    assert is_root_llm_node(nodes[0], by_index)
    assert not is_root_llm_node(nodes[1], by_index)
    assert not is_root_llm_node(nodes[2], by_index)
    assert is_root_llm_node(nodes[3], by_index)
    assert not is_root_llm_node(nodes[4], by_index)
    roots = root_llm_nodes(nodes)
    assert [n.index for n in roots] == [0, 3]

    result = map_session(_session(), nodes, [_eval()], "quality")
    assert isinstance(result, MappedTrace)
    assert result.trace.input == "user question"
    assert result.trace.output == "last answer"
    assert result.trace.output != "TOOL OUTPUT"
    assert result.trace.output != "subagent said"
    assert result.system_prompt == SYSTEM
    assert result.multi_turn is True


def test_tool_output_cannot_become_trace_output() -> None:
    nodes = [
        _node(0, output_text="llm answer"),
        _node(1, node_type="tool_call", parent=0, output_text="secret tool payload"),
    ]
    result = map_session(_session(), nodes, [_eval()], "quality")
    assert isinstance(result, MappedTrace)
    assert result.trace.output == "llm answer"


def test_no_root_llm_drops() -> None:
    nodes = [_node(0, node_type="subagent_call"), _node(1, parent=0)]
    result = map_session(_session(), nodes, [_eval()], "quality")
    assert isinstance(result, SourceDrop)
    assert result.reason == REASON_ROOT_LLM_UNAVAILABLE


def test_float_bool_passed_out_of_range_and_categorical_mapping() -> None:
    assert map_judge(_eval(score=0.4, data_type="float")).score == 0.4  # type: ignore[union-attr]
    bool_judge = map_judge(_eval(score=True, data_type="bool", passed=None))
    assert bool_judge.score == 1.0  # type: ignore[union-attr]
    passed_only = map_judge(_eval(score=None, passed=False, data_type="bool"))
    assert passed_only.score == 0.0  # type: ignore[union-attr]
    assert map_judge(_eval(score=1.5, data_type="float")) == REASON_SCORE_OUT_OF_RANGE
    assert map_judge(_eval(score=-0.1, data_type="float")) == REASON_SCORE_OUT_OF_RANGE
    assert (
        map_judge(_eval(score=None, value="bad", data_type="str", passed=None))
        == REASON_SCORE_UNSUPPORTED
    )
    assert (
        map_judge(_eval(score=0.2, value="cat", data_type="categorical"))
        == REASON_SCORE_UNSUPPORTED
    )


def test_missing_rationale_drops() -> None:
    assert map_judge(_eval(explanation=None)) == REASON_RATIONALE_MISSING
    assert map_judge(_eval(explanation="   ")) == REASON_RATIONALE_MISSING


def test_missing_score_drops() -> None:
    assert (
        map_judge(_eval(score=None, passed=None, data_type="float"))
        == REASON_SCORE_UNAVAILABLE
    )


def test_ambiguous_evaluator_version_on_one_session() -> None:
    selected = select_evaluation(
        [_eval(version=2, eval_id="a"), _eval(version=3, eval_id="b")],
        "quality",
    )
    assert selected == REASON_AMBIGUOUS_EVALUATION


def test_ambiguous_evaluator_version_across_the_cohort_refuses() -> None:
    records = [
        (_session("s1"), [_node(0)], [_eval(version=2)]),
        (_session("s2", number=2), [_node(0)], [_eval(version=3)]),
    ]
    result = map_batch(records, "quality")
    assert result == REASON_AMBIGUOUS_EVALUATION


def test_format_engine_is_refused_with_a_named_error(tmp_path) -> None:
    manifest = Manifest(
        template_file=tmp_path / "prompt.md",
        engine=TemplateEngine.FORMAT,
        judge_fingerprint="quality@3",
    )
    with pytest.raises(Exception, match=REASON_FORMAT_ENGINE_REFUSED):
        refuse_format_engine(manifest)


def test_conflicting_judge_fingerprint_is_an_error(tmp_path) -> None:
    manifest = Manifest(
        template_file=tmp_path / "prompt.md",
        engine=TemplateEngine.NONE,
        judge_fingerprint="other-judge",
    )
    with pytest.raises(Exception, match="judge-fingerprint-conflict"):
        check_judge_fingerprint(manifest, judge_fingerprint_for("quality", 3))


def test_matching_derived_fingerprint_is_accepted(tmp_path) -> None:
    derived = judge_fingerprint_for("quality", 3)
    manifest = Manifest(
        template_file=tmp_path / "prompt.md",
        engine="none",
        judge_fingerprint=derived,
    )
    check_judge_fingerprint(manifest, derived)


def test_source_and_batch_reasons_stay_kebab_case() -> None:
    drop = map_session(_session(), [_node(0, sys_sel=None)], [_eval()], "quality")
    assert isinstance(drop, SourceDrop)
    assert "-" in drop.reason
    assert drop.reason == drop.reason.lower()


def test_evaluator_id_looks_up_mapped_evaluator_name_not_the_cli_flag() -> None:
    from tracegrad.integrations.kitaru.client import CohortResolution
    from tracegrad.integrations.kitaru.source import _fetch_and_map

    looked_up: list[str] = []
    evaluation = _eval()
    evaluation.evaluator_name = "quality-judge"

    class Gateway:
        async def list_sessions(self, cohort_version_id: str) -> list[object]:
            return [_session()]

        async def fetch_records(self, sessions: list[object]) -> list[object]:
            return [(sessions[0], [_node(0)], [evaluation])]

        async def evaluator_id(self, name: str) -> str:
            looked_up.append(name)
            if name != "quality-judge":
                raise LookupError(f"kitaru evaluator {name!r} was not found")
            return "eid-judge"

    fingerprint, meta, mapped, _dropped = asyncio.run(
        _fetch_and_map(
            gateway=Gateway(),
            resolution=CohortResolution(
                cohort_id="c",
                cohort_name="support-production",
                cohort_version_id="cv",
                display_version="week-34",
                version_number=1,
                agent_id="a",
                session_count=1,
            ),
            evaluation_name="quality",
        )
    )
    assert looked_up == ["quality-judge"]
    assert fingerprint.evaluator_id == "eid-judge"
    assert fingerprint.evaluation_name == "quality"
    assert meta.evaluator_name == "quality-judge"
    assert mapped
