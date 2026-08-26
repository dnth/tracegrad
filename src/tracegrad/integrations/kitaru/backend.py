"""Kitaru ``VerificationBackend`` (issue #9, ADR 0001 / 0006 / 0007).

SDK imports stay in this module.  The orchestrator in ``tracegrad.verify``
never imports Kitaru.
"""

from __future__ import annotations

import uuid
from typing import Any

from tracegrad.verify import (
    DIVERGENCE_EVALUATOR_VERSION,
    DIVERGENCE_HISTORY,
    DIVERGENCE_SCOPE,
    DIVERGENCE_SCORE,
    DIVERGENCE_SELECT,
    Divergence,
    ReplayFailure,
    SubmittedVerification,
    VerificationRequest,
    VerificationResult,
    verification_fingerprint_for,
)

from .client import KitaruGateway, run_async, worker_covers_agent_version
from .errors import KitaruVerifyError
from .graph import index_nodes, is_root_llm_node, llm_nodes, node_index
from .mapping import extract_system_prompt
from .policy import asserts_no_passthrough, recorded_history_policy
from .require import require_kitaru
from .scores import evaluator_version_of, map_score, select_evaluation


def _history_tool_policy() -> Any:
    """The only tool policy the tracegrad path can construct."""

    require_kitaru()
    spec = recorded_history_policy()
    asserts_no_passthrough(spec)
    from kitaru.api_models.v1.replay_config import (
        HistoryConfig,
        HistoryScope,
        ToolPolicy,
        ToolPolicyOnMiss,
    )

    policy = ToolPolicy(
        default=HistoryConfig(
            scope=HistoryScope.COHORT_VERSION,
            on_miss=ToolPolicyOnMiss.FAIL,
        )
    )
    # Belt: never let a default or per-tool override sneak passthrough in.
    default = policy.default
    on_miss = getattr(default, "on_miss", None)
    if str(getattr(on_miss, "value", on_miss)).lower() == "passthrough":
        raise KitaruVerifyError("passthrough tool policy is not reachable through tracegrad")
    if getattr(policy, "tools", None):
        raise KitaruVerifyError("per-tool policy overrides are not reachable through tracegrad")
    return policy


def _override(system_prompt: str) -> Any:
    from kitaru.api_models.v1.replay_config import ReplayOverride

    return ReplayOverride(system_prompt=system_prompt)


def _evaluator_config(request: VerificationRequest) -> Any:
    from kitaru.api_models.v1.replay_config import EvaluatorConfig

    return EvaluatorConfig(
        evaluator=request.evaluator_name,
        version=int(request.evaluator_version),
    )


def is_tool_history_miss(error: object | None) -> bool:
    """Whether a replay error is a recorded-history miss (issue #9).

    Kitaru adapters raise ``ToolPolicyMissError`` with
    ``No history result for tool '…'``. Invented needles never match those
    messages, so the miss would otherwise land in ``replay_failures`` instead
    of ``TOOL_HISTORY_MISS`` / incomparable.
    """

    if error is None:
        return False
    names = {type(error).__name__, type(error).__qualname__.rsplit(".", 1)[-1]}
    if "ToolPolicyMissError" in names:
        return True
    text = str(error).strip()
    if not text:
        return False
    lowered = text.lower()
    return "no history result for tool" in lowered or "toolpolicymisserror" in lowered


def mixed_agent_version_message(counts: dict[str, int]) -> str:
    parts = [f"{version}: {count} session(s)" for version, count in sorted(counts.items())]
    return (
        "mixed-agent-version-cohort refused: every session in the cohort "
        "version must share one agent_version_id "
        f"({'; '.join(parts)}). A report that says the prompt caused this "
        "when the agent code also moved is worse. See ADR 0007."
    )


def assert_override_scope(
    baseline_nodes: list[Any] | tuple[Any, ...],
    result_nodes: list[Any] | tuple[Any, ...],
    candidate_prompt: str,
) -> str | None:
    """Return a detail string when the override did not land on root nodes only."""

    base_by = index_nodes(baseline_nodes)
    result_by = index_nodes(result_nodes)
    for node in llm_nodes(result_nodes):
        recorded = extract_system_prompt(node)
        if is_root_llm_node(node, result_by):
            if recorded != candidate_prompt:
                return (
                    f"root llm node {node_index(node)} did not carry the candidate prompt"
                )
            continue
        counterpart = base_by.get(node_index(node))
        expected = extract_system_prompt(counterpart) if counterpart is not None else None
        if recorded != expected:
            return (
                f"non-root llm node {node_index(node)} did not keep its baseline prompt"
            )
    return None


def classify_replay_session(
    *,
    session_id: str,
    number: int | None,
    baseline_eval: Any,
    candidate_eval: Any,
    requested_evaluator_version: int,
) -> str | Divergence:
    """Bucket one replay: improved/regressed/unchanged, or a typed divergence.

    ``select_evaluation`` drop reasons, evaluator-version mismatch, and an
    unclassified score are incomparable — they go in ``diverged``, never
    ``replay_failures``, and never vanish from the per-session buckets.
    """

    if isinstance(baseline_eval, str) or isinstance(candidate_eval, str):
        parts: list[str] = []
        if isinstance(baseline_eval, str):
            parts.append(f"baseline: {baseline_eval}")
        if isinstance(candidate_eval, str):
            parts.append(f"candidate: {candidate_eval}")
        return Divergence(
            session_id=session_id,
            kind=DIVERGENCE_SELECT,
            detail="; ".join(parts),
            number=number,
        )
    baseline_version = evaluator_version_of(baseline_eval)
    candidate_version = evaluator_version_of(candidate_eval)
    if (
        baseline_version != requested_evaluator_version
        or candidate_version != requested_evaluator_version
    ):
        return Divergence(
            session_id=session_id,
            kind=DIVERGENCE_EVALUATOR_VERSION,
            detail=(
                f"requested evaluator_version {requested_evaluator_version}; "
                f"baseline={baseline_version}; candidate={candidate_version}"
            ),
            number=number,
        )
    verdict = classify_scores(baseline_eval, candidate_eval)
    if verdict is None:
        return Divergence(
            session_id=session_id,
            kind=DIVERGENCE_SCORE,
            detail="scores could not be classified as improved, regressed, or unchanged",
            number=number,
        )
    return verdict


def classify_scores(
    baseline: Any | None, candidate: Any | None
) -> str | None:
    """Return improved / regressed / unchanged, or None if incomparable."""

    if baseline is None or candidate is None:
        return None
    base_passed = getattr(baseline, "passed", None)
    cand_passed = getattr(candidate, "passed", None)
    if isinstance(base_passed, bool) and isinstance(cand_passed, bool):
        if (not base_passed) and cand_passed:
            return "improved"
        if base_passed and (not cand_passed):
            return "regressed"
        if base_passed == cand_passed:
            # Fall through to score for a finer signal when both exist.
            pass
        else:
            return "unchanged"
    base_score = map_score(baseline)
    cand_score = map_score(candidate)
    if isinstance(base_score, str) or isinstance(cand_score, str):
        if isinstance(base_passed, bool) and isinstance(cand_passed, bool):
            return "unchanged"
        return None
    if cand_score > base_score:
        return "improved"
    if cand_score < base_score:
        return "regressed"
    return "unchanged"


class KitaruVerificationBackend:
    """Submit and collect a Kitaru experiment run for one candidate prompt."""

    name = "kitaru"

    def __init__(self, gateway: KitaruGateway | None = None) -> None:
        require_kitaru()
        self._gateway = gateway
        self._owns = gateway is None

    def _gw(self) -> KitaruGateway:
        if self._gateway is None:
            self._gateway = KitaruGateway()
        return self._gateway

    def preflight(self, request: VerificationRequest) -> None:
        run_async(self._preflight(request))

    async def _preflight(self, request: VerificationRequest) -> None:
        gateway = self._gw()
        try:
            await gateway.server_info()
        except Exception as exc:
            raise KitaruVerifyError(
                "kitaru server is not reachable. Start the server and run "
                "`kitaru login` first; tracegrad does not host verification "
                "(ADR 0001)."
            ) from exc
        try:
            await gateway.get_cohort_version(request.cohort_version_id)
        except Exception as exc:
            raise KitaruVerifyError(
                f"cohort version {request.cohort_version_id} did not resolve"
            ) from exc
        try:
            await gateway.get_agent_version(request.agent_version_id)
        except Exception as exc:
            raise KitaruVerifyError(
                f"agent version {request.agent_version_id} did not resolve"
            ) from exc
        counts = request.agent_version_counts
        distinct = {key: count for key, count in counts.items() if key != "unspecified"}
        if "unspecified" in counts or len(distinct) != 1:
            raise KitaruVerifyError(mixed_agent_version_message(counts or {"unspecified": 0}))
        only = next(iter(distinct))
        if only != request.agent_version_id:
            raise KitaruVerifyError(mixed_agent_version_message(counts))
        workers = await gateway.list_live_workers()
        if not any(worker_covers_agent_version(worker, request.agent_version_id) for worker in workers):
            raise KitaruVerifyError(
                "no live worker is polling for agent version "
                f"{request.agent_version_id}. Start a worker in the virtualenv "
                "where the agent runs; tracegrad does not host workers (ADR 0001)."
            )

    def submit(self, request: VerificationRequest) -> SubmittedVerification:
        return run_async(self._submit(request))

    async def _submit(self, request: VerificationRequest) -> SubmittedVerification:
        from kitaru.api_models.v1.experiment import ExperimentCreateRequest
        from kitaru.api_models.v1.experiment_run import ExperimentRunCreateRequest

        gateway = self._gw()
        digest = request.candidate_prompt_hash.removeprefix("sha256:")[:8]
        experiment = await gateway.create_experiment(
            ExperimentCreateRequest(
                name=f"tracegrad-{request.run_id}-{digest}",
                description=f"tracegrad verification of {request.run_id}",
                agent_id=uuid.UUID(request.agent_id),
                override=_override(request.candidate_prompt),
                tool_policy=_history_tool_policy(),
                evaluators=[_evaluator_config(request)],
            )
        )
        run = await gateway.start_run(
            str(experiment.id),
            ExperimentRunCreateRequest(
                cohort_version_id=uuid.UUID(request.cohort_version_id),
                agent_version_id=uuid.UUID(request.agent_version_id),
                evaluate_baselines=True,
            ),
        )
        return SubmittedVerification(
            experiment_id=str(experiment.id),
            experiment_run_id=str(run.id),
        )

    def collect(
        self, request: VerificationRequest, submitted: SubmittedVerification
    ) -> VerificationResult:
        return run_async(self._collect(request, submitted))

    async def _collect(
        self, request: VerificationRequest, submitted: SubmittedVerification
    ) -> VerificationResult:
        gateway = self._gw()
        run = await gateway.wait_for_experiment_run(submitted.experiment_run_id)
        status = str(getattr(getattr(run, "status", None), "value", getattr(run, "status", "completed")))
        replays = await gateway.list_replays(submitted.experiment_run_id)
        improved: list[str] = []
        regressed: list[str] = []
        unchanged: list[str] = []
        diverged: list[Divergence] = []
        failures: list[ReplayFailure] = []

        for replay in replays:
            session_id = str(replay.baseline_session_id)
            number = request.session_numbers.get(session_id)
            replay_status = str(
                getattr(getattr(replay, "status", None), "value", getattr(replay, "status", ""))
            )
            error = getattr(replay, "error", None)
            if replay_status == "failed" or (error and not getattr(replay, "result_session_id", None)):
                if is_tool_history_miss(error):
                    diverged.append(
                        Divergence(
                            session_id=session_id,
                            kind=DIVERGENCE_HISTORY,
                            detail=str(error or "tool history miss"),
                            number=number,
                        )
                    )
                else:
                    failures.append(
                        ReplayFailure(
                            session_id=session_id,
                            error=str(error or replay_status or "replay failed"),
                            number=number,
                        )
                    )
                continue
            result_id = getattr(replay, "result_session_id", None)
            if result_id is None:
                failures.append(
                    ReplayFailure(session_id=session_id, error="missing result session", number=number)
                )
                continue
            baseline_nodes = await gateway.session_nodes(str(replay.baseline_session_id))
            result_nodes = await gateway.session_nodes(str(result_id))
            scope = assert_override_scope(
                baseline_nodes, result_nodes, request.candidate_prompt
            )
            if scope:
                diverged.append(
                    Divergence(
                        session_id=session_id,
                        kind=DIVERGENCE_SCOPE,
                        detail=scope,
                        number=number,
                    )
                )
                continue
            baseline_eval = select_evaluation(
                await gateway.evaluations_for(str(replay.baseline_session_id)),
                request.evaluation_name,
            )
            candidate_eval = select_evaluation(
                await gateway.evaluations_for(str(result_id)),
                request.evaluation_name,
            )
            outcome = classify_replay_session(
                session_id=session_id,
                number=number,
                baseline_eval=baseline_eval,
                candidate_eval=candidate_eval,
                requested_evaluator_version=int(request.evaluator_version),
            )
            if isinstance(outcome, Divergence):
                diverged.append(outcome)
                continue
            if outcome == "improved":
                improved.append(session_id)
            elif outcome == "regressed":
                regressed.append(session_id)
            elif outcome == "unchanged":
                unchanged.append(session_id)

        # Headline numbers from /api/v1/ui/experiment-runs/{id}/evaluation-aggregates
        # so they match the Kitaru UI. That namespace is UI-support, not an obvious
        # third-party contract; the <0.23 pin contains it.
        aggregates = await gateway.evaluation_aggregates(submitted.experiment_run_id)
        baseline_stats, candidate_stats = _pick_aggregate(aggregates, request.evaluation_name)
        run_status = "failed" if status == "failed" else (
            "partial" if failures or status != "completed" else "completed"
        )
        result = VerificationResult(
            status=run_status,  # type: ignore[arg-type]
            baseline_count=int((baseline_stats or {}).get("count") or 0),
            candidate_count=int((candidate_stats or {}).get("count") or 0),
            baseline_mean_score=_maybe_float((baseline_stats or {}).get("mean")),
            candidate_mean_score=_maybe_float((candidate_stats or {}).get("mean")),
            baseline_pass_rate=_maybe_float((baseline_stats or {}).get("pass_rate")),
            candidate_pass_rate=_maybe_float((candidate_stats or {}).get("pass_rate")),
            improved_sessions=improved,
            regressed_sessions=regressed,
            unchanged_sessions=unchanged,
            diverged_sessions=diverged,
            replay_failures=failures,
            cohort_version_id=request.cohort_version_id,
            agent_version_id=request.agent_version_id,
            evaluator_version=str(request.evaluator_version),
            baseline_prompt_hash=request.baseline_prompt_hash,
            candidate_prompt_hash=request.candidate_prompt_hash,
            verification_fingerprint=verification_fingerprint_for(request),
            experiment_run_id=submitted.experiment_run_id,
        )
        if self._owns and self._gateway is not None:
            await self._gateway.close()
            self._gateway = None
        return result


def _maybe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _pick_aggregate(
    payloads: list[Any], evaluation_name: str
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    for item in payloads:
        raw = item if isinstance(item, dict) else (
            item.model_dump(mode="json") if hasattr(item, "model_dump") else None
        )
        if not isinstance(raw, dict):
            continue
        if raw.get("name") != evaluation_name:
            continue
        baseline = raw.get("baseline")
        result = raw.get("result")
        return (
            baseline if isinstance(baseline, dict) else None,
            result if isinstance(result, dict) else None,
        )
    return None, None
