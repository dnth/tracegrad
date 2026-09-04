"""Session → ``Trace`` mapping (ADR 0002, 0003, 0005).

Kitaru sessions are duck-typed.  This module does not import the Kitaru SDK, so
core-only tests can exercise every mapping rule with plain objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from tracegrad.canonical import text_hash
from tracegrad.schema import Trace, TraceMeta

from .graph import llm_nodes, node_index, root_llm_nodes
from .pointer import resolve_text_selector
from .scores import (
    REASON_AMBIGUOUS_EVALUATION,
    evaluator_name_of,
    evaluator_version_id_of,
    evaluator_version_of,
    map_judge,
    select_evaluation,
)

MAPPING_VERSION = 1

REASON_SYSTEM_PROMPT_UNAVAILABLE = "system-prompt-unavailable"
REASON_MULTIPLE_SYSTEM_PROMPTS = "multiple-system-prompts"
REASON_OUTPUT_UNAVAILABLE = "output-unavailable"
REASON_INPUT_UNAVAILABLE = "input-unavailable"
REASON_ROOT_LLM_UNAVAILABLE = "root-llm-unavailable"
REASON_FORMAT_ENGINE_REFUSED = "format-engine-refused"
REASON_JUDGE_FINGERPRINT_CONFLICT = "judge-fingerprint-conflict"

SOURCE_DROP_REASONS = (
    REASON_SYSTEM_PROMPT_UNAVAILABLE,
    REASON_MULTIPLE_SYSTEM_PROMPTS,
    REASON_OUTPUT_UNAVAILABLE,
    REASON_INPUT_UNAVAILABLE,
    REASON_ROOT_LLM_UNAVAILABLE,
    REASON_AMBIGUOUS_EVALUATION,
    "judge-rationale-missing",
    "judge-score-out-of-range",
    "judge-score-unsupported",
    "judge-score-unavailable",
)


@dataclass(frozen=True)
class SourceDrop:
    """A Session that could not become a Trace, with the kebab-case reason."""

    session_id: str
    reason: str
    detail: str = ""
    number: int | None = None


@dataclass(frozen=True)
class MappedTrace:
    """A Trace plus the Kitaru metadata Phase 2 reuses."""

    trace: Trace
    session_number: int | None
    evaluator_name: str
    evaluator_version: int
    evaluator_version_id: str | None
    multi_turn: bool
    system_prompt: str


def _session_id(session: Any) -> str:
    value = getattr(session, "id", None)
    if value is None:
        raise TypeError("session is missing id")
    return str(value)


def _session_number(session: Any) -> int | None:
    value = getattr(session, "number", None)
    return int(value) if isinstance(value, int) else None


def _extract_system_prompts(roots: Sequence[Any]) -> list[str] | str:
    """Unique system prompts from root LLM nodes, or a drop reason."""

    values: list[str] = []
    for node in roots:
        prompt = resolve_text_selector(
            getattr(node, "inputs", None),
            getattr(node, "system_prompt_selector", None),
        )
        if prompt is None or prompt == "":
            continue
        if prompt not in values:
            values.append(prompt)
    if not values:
        return REASON_SYSTEM_PROMPT_UNAVAILABLE
    if len(values) > 1:
        return REASON_MULTIPLE_SYSTEM_PROMPTS
    return values


def _root_model(roots: Sequence[Any]) -> str | None:
    for node in roots:
        model = getattr(node, "model", None)
        if isinstance(model, str) and model:
            return model
    return None


def map_session(
    session: Any,
    nodes: Sequence[Any],
    evaluations: Sequence[Any],
    evaluation_name: str,
) -> MappedTrace | SourceDrop:
    """Map one Kitaru session onto a ``Trace``, or drop it by name.

    Never guesses a system prompt.  Never reads session-level inputs/outputs.
    Never lets a tool output or a subagent prompt become the artifact.
    """

    session_id = _session_id(session)
    number = _session_number(session)
    roots = root_llm_nodes(nodes)
    if not roots:
        return SourceDrop(
            session_id,
            REASON_ROOT_LLM_UNAVAILABLE,
            "no llm_call node is free of subagent ancestry",
            number,
        )

    prompts = _extract_system_prompts(roots)
    if isinstance(prompts, str):
        detail = (
            "system_prompt_selector did not resolve to a string on any root LLM node"
            if prompts == REASON_SYSTEM_PROMPT_UNAVAILABLE
            else "root LLM nodes recorded more than one distinct system prompt"
        )
        return SourceDrop(session_id, prompts, detail, number)
    system_prompt = prompts[0]

    first, last = roots[0], roots[-1]
    mapped_input = resolve_text_selector(
        getattr(first, "inputs", None),
        getattr(first, "input_text_selector", None),
    )
    if mapped_input is None:
        return SourceDrop(
            session_id,
            REASON_INPUT_UNAVAILABLE,
            "input_text_selector did not resolve to a string on the first root LLM node",
            number,
        )
    mapped_output = resolve_text_selector(
        getattr(last, "outputs", None),
        getattr(last, "output_text_selector", None),
    )
    if mapped_output is None:
        return SourceDrop(
            session_id,
            REASON_OUTPUT_UNAVAILABLE,
            "output_text_selector did not resolve to a string on the last root LLM node",
            number,
        )

    selected = select_evaluation(list(evaluations), evaluation_name)
    if isinstance(selected, str):
        return SourceDrop(session_id, selected, f"evaluation {evaluation_name!r}", number)

    judge = map_judge(selected)
    if isinstance(judge, str):
        return SourceDrop(session_id, judge, f"evaluation {evaluation_name!r}", number)

    name = evaluator_name_of(selected) or evaluation_name
    version = evaluator_version_of(selected)
    if version is None:
        return SourceDrop(
            session_id,
            REASON_AMBIGUOUS_EVALUATION,
            "evaluation has no evaluator_version",
            number,
        )

    model = _root_model(roots)
    multi_turn = len(roots) > 1 or len(llm_nodes(nodes)) > 1
    trace = Trace(
        trace_id=session_id,
        input=mapped_input,
        output=mapped_output,
        judge=judge,
        prompt_hash=text_hash(system_prompt),
        meta=TraceMeta(model=model) if model is not None else None,
    )
    return MappedTrace(
        trace=trace,
        session_number=number,
        evaluator_name=name,
        evaluator_version=version,
        evaluator_version_id=evaluator_version_id_of(selected),
        multi_turn=multi_turn,
        system_prompt=system_prompt,
    )


def extract_system_prompt(node: Any) -> str | None:
    """The recorded system prompt on one node, or ``None`` if unresolved."""

    return resolve_text_selector(
        getattr(node, "inputs", None),
        getattr(node, "system_prompt_selector", None),
    )


def counterpart_prompt(nodes: Sequence[Any], index: int) -> str | None:
    """System prompt of the LLM node at ``index``, if any."""

    for node in nodes:
        if node_index(node) == index:
            return extract_system_prompt(node)
    return None


@dataclass(frozen=True)
class BatchMapping:
    """The mapped traces of one cohort, plus source drops, never mixed."""

    mapped: tuple[MappedTrace, ...]
    dropped: tuple[SourceDrop, ...]
    evaluator_name: str | None
    evaluator_version: int | None
    evaluator_version_id: str | None
    multi_turn_count: int


def map_batch(
    records: Sequence[tuple[Any, Sequence[Any], Sequence[Any]]],
    evaluation_name: str,
) -> BatchMapping | str:
    """Map every session.  Mixed evaluator versions refuse the batch.

    Returns a :class:`BatchMapping`, or ``ambiguous-evaluation`` when the
    successfully mapped traces do not share one evaluator version.
    """

    mapped: list[MappedTrace] = []
    dropped: list[SourceDrop] = []
    for session, nodes, evaluations in records:
        result = map_session(session, nodes, evaluations, evaluation_name)
        if isinstance(result, SourceDrop):
            dropped.append(result)
        else:
            mapped.append(result)

    versions = {(item.evaluator_name, item.evaluator_version) for item in mapped}
    if len(versions) > 1:
        return REASON_AMBIGUOUS_EVALUATION

    first = mapped[0] if mapped else None
    return BatchMapping(
        mapped=tuple(mapped),
        dropped=tuple(dropped),
        evaluator_name=first.evaluator_name if first else None,
        evaluator_version=first.evaluator_version if first else None,
        evaluator_version_id=first.evaluator_version_id if first else None,
        multi_turn_count=sum(1 for item in mapped if item.multi_turn),
    )
