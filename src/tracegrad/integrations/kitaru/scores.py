"""Kitaru evaluation → tracegrad ``Judge`` mapping (ADR 0003).

Scores are accepted, never rescaled.  ``schema.Score`` is ``[0, 1]``; anything
outside that, or non-numeric, drops by name.  This module does not import the
Kitaru SDK; evaluations are duck-typed.
"""

from __future__ import annotations

from typing import Any

from tracegrad.schema import Judge

REASON_RATIONALE_MISSING = "judge-rationale-missing"
REASON_SCORE_OUT_OF_RANGE = "judge-score-out-of-range"
REASON_SCORE_UNSUPPORTED = "judge-score-unsupported"
REASON_SCORE_UNAVAILABLE = "judge-score-unavailable"
REASON_AMBIGUOUS_EVALUATION = "ambiguous-evaluation"

CATEGORICAL = "categorical"
STRING = "str"
BOOL = "bool"
FLOAT = "float"


def _data_type(evaluation: Any) -> str:
    raw = getattr(evaluation, "data_type", None)
    value = getattr(raw, "value", raw)
    if value:
        return str(value)
    score = getattr(evaluation, "score", None)
    label = getattr(evaluation, "value", None)
    if score is not None and label is not None:
        return CATEGORICAL
    if score is None and label is not None:
        return STRING
    if isinstance(score, bool):
        return BOOL
    if score is not None:
        return FLOAT
    return ""


def _rationale(evaluation: Any) -> str | None:
    explanation = getattr(evaluation, "explanation", None)
    if not isinstance(explanation, str):
        return None
    stripped = explanation.strip()
    return stripped or None


def map_score(evaluation: Any) -> float | str:
    """Return a ``[0, 1]`` score, or the kebab-case drop reason."""

    data_type = _data_type(evaluation)
    if data_type in {STRING, CATEGORICAL}:
        return REASON_SCORE_UNSUPPORTED

    score = getattr(evaluation, "score", None)
    passed = getattr(evaluation, "passed", None)

    if isinstance(score, bool):
        return 1.0 if score else 0.0
    if score is None and isinstance(passed, bool):
        return 1.0 if passed else 0.0
    if isinstance(score, (int, float)):
        value = float(score)
        if 0.0 <= value <= 1.0:
            return value
        return REASON_SCORE_OUT_OF_RANGE
    return REASON_SCORE_UNAVAILABLE


def map_judge(evaluation: Any) -> Judge | str:
    """Map one evaluation onto a ``Judge``, or return a drop reason."""

    rationale = _rationale(evaluation)
    if rationale is None:
        return REASON_RATIONALE_MISSING
    mapped = map_score(evaluation)
    if isinstance(mapped, str):
        return mapped
    return Judge(score=mapped, rationale=rationale)


def evaluator_name_of(evaluation: Any) -> str | None:
    name = getattr(evaluation, "evaluator_name", None)
    if isinstance(name, str) and name:
        return name
    fallback = getattr(evaluation, "name", None)
    return fallback if isinstance(fallback, str) and fallback else None


def evaluator_version_of(evaluation: Any) -> int | None:
    version = getattr(evaluation, "evaluator_version", None)
    return int(version) if isinstance(version, int) else None


def evaluator_version_id_of(evaluation: Any) -> str | None:
    value = getattr(evaluation, "evaluator_version_id", None)
    return str(value) if value is not None else None


def select_evaluation(
    evaluations: list[Any] | tuple[Any, ...],
    evaluation_name: str,
) -> Any | str:
    """Pick the evaluation named ``evaluation_name``, or a drop reason.

    Several rows with the same evaluator version are collapsed
    deterministically (sorted by id).  Differing versions on one session are
    ``ambiguous-evaluation``.
    """

    matching = [
        item
        for item in evaluations
        if getattr(item, "name", None) == evaluation_name
    ]
    if not matching:
        return REASON_SCORE_UNAVAILABLE

    versions = {evaluator_version_of(item) for item in matching}
    if len(versions) > 1:
        return REASON_AMBIGUOUS_EVALUATION

    matching.sort(key=lambda item: str(getattr(item, "id", "")))
    return matching[0]


def judge_fingerprint_for(evaluator_name: str, evaluator_version: int) -> str:
    """The fingerprint derived from the evaluator that actually scored the batch."""

    return f"{evaluator_name}@{evaluator_version}"
