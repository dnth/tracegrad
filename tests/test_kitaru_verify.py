"""Phase 2 verification: gate, resume, policy, override scope (issue #9)."""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tracegrad.apply import (
    Proposal,
    ProposedEdit,
    StaleProposalError,
    applied_history,
    apply_proposal,
    candidate_prompt,
    save_proposal,
)
from tracegrad.canonical import text_hash
from tracegrad.edits import resolve_edits
from tracegrad.integrations.kitaru.backend import (
    KitaruVerificationBackend,
    assert_override_scope,
    classify_replay_session,
    classify_scores,
    is_tool_history_miss,
    mixed_agent_version_message,
)
from tracegrad.integrations.kitaru.client import FETCH_JOBS, worker_covers_agent_version
from tracegrad.integrations.kitaru.errors import KitaruVerifyError
from tracegrad.integrations.kitaru.policy import (
    RECORDED_HISTORY_POLICY,
    asserts_no_passthrough,
    recorded_history_policy,
)
from tracegrad.integrations.kitaru.scores import (
    REASON_AMBIGUOUS_EVALUATION,
    REASON_SCORE_UNAVAILABLE,
)
from tracegrad.integrations.kitaru.snapshot import RUN_SOURCE_FILENAME
from tracegrad.inventory import build_inventory
from tracegrad.schema import Edit
from tracegrad.state import atomic_write_json, initialize
from tracegrad.verify import (
    DIVERGENCE_EVALUATOR_VERSION,
    DIVERGENCE_HISTORY,
    DIVERGENCE_SCORE,
    DIVERGENCE_SELECT,
    Divergence,
    ReplayFailure,
    SubmittedVerification,
    VerificationRequest,
    VerificationResult,
    VerifyError,
    backend_is_configured,
    build_request,
    format_verification_report,
    load_run_source_payload,
    load_verification_state,
    matching_verification,
    refuse_ungated_apply,
    run_verification,
    verification_id_for,
)

PROMPT = "Rules:\n- Be concise.\n- Cite the doc.\n"
CANDIDATE = "Rules:\n- Be concise.\n- Always cite the doc.\n"


def _proposal(project: Path, prompt: str = PROMPT) -> Proposal:
    (project / "prompt.md").write_text(prompt, encoding="utf-8")
    inventory = build_inventory(prompt)
    resolution = resolve_edits(
        inventory,
        [
            Edit(
                instruction_id=inventory.instructions[-1].instruction_id,
                operation="REWRITE",
                text="Always cite the doc.",
                covers_theme="missing-citation",
                watch_metric="missing-citation",
            )
        ],
    )
    proposal = Proposal(
        run_id="run-0001",
        template_file="prompt.md",
        base_prompt_hash=text_hash(prompt),
        edits=[
            ProposedEdit(
                edit=item.edit,
                before=item.anchor.text if item.anchor else "",
                after=item.replacement,
            )
            for item in resolution.resolved
        ],
    )
    save_proposal(project, proposal)
    return proposal


def _multi_edit_proposal(project: Path, prompt: str = PROMPT) -> Proposal:
    (project / "prompt.md").write_text(prompt, encoding="utf-8")
    inventory = build_inventory(prompt)
    first, second = inventory.instructions[1], inventory.instructions[2]
    resolution = resolve_edits(
        inventory,
        [
            Edit(
                instruction_id=first.instruction_id,
                operation="REWRITE",
                text="Be brief.",
                covers_theme="verbosity",
                watch_metric="verbosity",
            ),
            Edit(
                instruction_id=second.instruction_id,
                operation="REWRITE",
                text="Always cite the doc.",
                covers_theme="missing-citation",
                watch_metric="missing-citation",
            ),
        ],
    )
    proposal = Proposal(
        run_id="run-0001",
        template_file="prompt.md",
        base_prompt_hash=text_hash(prompt),
        edits=[
            ProposedEdit(
                edit=item.edit,
                before=item.anchor.text if item.anchor else "",
                after=item.replacement,
            )
            for item in resolution.resolved
        ],
    )
    save_proposal(project, proposal)
    return proposal


def _source_sidecar(project: Path, run_id: str = "run-0001") -> None:
    layout = initialize(project)
    atomic_write_json(
        layout.runs / run_id / RUN_SOURCE_FILENAME,
        {
            "fingerprint": {
                "source": "kitaru",
                "cohort_id": "c1",
                "cohort_version_id": "cv1",
                "evaluation_name": "quality",
                "evaluator_id": "ev",
                "evaluator_version": 3,
                "agent_id": "a1",
                "mapping_version": 1,
            },
            "meta": {
                "cohort_name": "support-production",
                "display_version": "week-34",
                "agent_version_id": "av1",
                "agent_version_counts": {"av1": 2},
                "session_numbers": {},
                "system_prompts": {},
                "multi_turn_count": 0,
                "sessions_selected": 2,
                "traces_mapped": 2,
                "evaluator_name": "quality",
            },
        },
    )


def _request(**overrides: object) -> VerificationRequest:
    payload = dict(
        run_id="run-0001",
        proposal_id="run-0001",
        candidate_prompt=CANDIDATE,
        candidate_prompt_hash=text_hash(CANDIDATE),
        baseline_prompt_hash=text_hash(PROMPT),
        cohort_id="c1",
        cohort_version_id="cv1",
        cohort_name="support-production",
        display_version="week-34",
        evaluation_name="quality",
        evaluator_id="ev",
        evaluator_version=3,
        evaluator_name="quality",
        agent_id="a1",
        agent_version_id="av1",
    )
    payload.update(overrides)
    return VerificationRequest.model_validate(payload)


class FakeBackend:
    name = "fake"

    def __init__(self) -> None:
        self.preflighted = 0
        self.submitted = 0
        self.collected = 0

    def preflight(self, request: VerificationRequest) -> None:
        self.preflighted += 1

    def submit(self, request: VerificationRequest) -> SubmittedVerification:
        self.submitted += 1
        return SubmittedVerification(experiment_id="exp-1", experiment_run_id="erun-1")

    def collect(
        self, request: VerificationRequest, submitted: SubmittedVerification
    ) -> VerificationResult:
        self.collected += 1
        return VerificationResult(
            status="completed",
            baseline_count=2,
            candidate_count=2,
            baseline_mean_score=0.5,
            candidate_mean_score=0.7,
            baseline_pass_rate=0.5,
            candidate_pass_rate=0.7,
            improved_sessions=["s1"],
            regressed_sessions=[],
            unchanged_sessions=["s2"],
            diverged_sessions=[],
            replay_failures=[],
            cohort_version_id=request.cohort_version_id,
            agent_version_id=request.agent_version_id,
            evaluator_version=str(request.evaluator_version),
            baseline_prompt_hash=request.baseline_prompt_hash,
            candidate_prompt_hash=request.candidate_prompt_hash,
            verification_fingerprint="fp",
            experiment_run_id=submitted.experiment_run_id,
        )


def test_recorded_history_policy_is_the_only_policy() -> None:
    policy = recorded_history_policy()
    asserts_no_passthrough(policy)
    assert policy["scope"] == "cohort_version"
    assert policy["on_miss"] == "fail"
    assert policy["type"] == "history"
    assert "passthrough" not in json.dumps(RECORDED_HISTORY_POLICY)


def test_passthrough_is_rejected() -> None:
    with pytest.raises(ValueError, match="fail"):
        asserts_no_passthrough({"type": "history", "scope": "cohort_version", "on_miss": "passthrough"})


def test_tool_history_miss_is_typed() -> None:
    assert is_tool_history_miss("No history result for tool 'search_account'")
    assert is_tool_history_miss("ToolPolicyMissError: No history result for tool 'lookup'")

    class ToolPolicyMissError(Exception):
        pass

    assert is_tool_history_miss(ToolPolicyMissError("No history result for tool 'x'"))
    assert not is_tool_history_miss("evaluator crashed")
    assert not is_tool_history_miss("no recorded call matched under on_miss=fail")
    assert not is_tool_history_miss("TOOL_HISTORY_MISS: search_account")


def test_override_scope_divergence_on_non_root() -> None:
    candidate = "NEW PROMPT"
    baseline = [
        SimpleNamespace(
            index=0,
            parent_index=None,
            secondary_parent_indexes=[],
            node_type="llm_call",
            inputs={"system": "ROOT"},
            system_prompt_selector="/system",
        ),
        SimpleNamespace(
            index=1,
            parent_index=None,
            secondary_parent_indexes=[],
            node_type="subagent_call",
            inputs={},
            system_prompt_selector=None,
        ),
        SimpleNamespace(
            index=2,
            parent_index=1,
            secondary_parent_indexes=[],
            node_type="llm_call",
            inputs={"system": "SUB"},
            system_prompt_selector="/system",
        ),
    ]
    result_ok = [
        SimpleNamespace(
            index=0,
            parent_index=None,
            secondary_parent_indexes=[],
            node_type="llm_call",
            inputs={"system": candidate},
            system_prompt_selector="/system",
        ),
        SimpleNamespace(
            index=1,
            parent_index=None,
            secondary_parent_indexes=[],
            node_type="subagent_call",
            inputs={},
            system_prompt_selector=None,
        ),
        SimpleNamespace(
            index=2,
            parent_index=1,
            secondary_parent_indexes=[],
            node_type="llm_call",
            inputs={"system": "SUB"},
            system_prompt_selector="/system",
        ),
    ]
    result_bad = [
        SimpleNamespace(
            index=0,
            parent_index=None,
            secondary_parent_indexes=[],
            node_type="llm_call",
            inputs={"system": candidate},
            system_prompt_selector="/system",
        ),
        SimpleNamespace(
            index=1,
            parent_index=None,
            secondary_parent_indexes=[],
            node_type="subagent_call",
            inputs={},
            system_prompt_selector=None,
        ),
        SimpleNamespace(
            index=2,
            parent_index=1,
            secondary_parent_indexes=[],
            node_type="llm_call",
            inputs={"system": candidate},
            system_prompt_selector="/system",
        ),
    ]
    assert assert_override_scope(baseline, result_ok, candidate) is None
    detail = assert_override_scope(baseline, result_bad, candidate)
    assert detail is not None
    assert "non-root" in detail


def test_override_scope_diverges_when_result_has_no_root_llm() -> None:
    candidate = "NEW PROMPT"
    baseline = [
        SimpleNamespace(
            index=0,
            parent_index=None,
            secondary_parent_indexes=[],
            node_type="llm_call",
            inputs={"system": "ROOT"},
            system_prompt_selector="/system",
        ),
    ]
    empty_detail = assert_override_scope(baseline, (), candidate)
    assert empty_detail is not None
    assert "no root llm node carrying the candidate" in empty_detail

    tool_only = [
        SimpleNamespace(
            index=0,
            parent_index=None,
            secondary_parent_indexes=[],
            node_type="tool_call",
            inputs={},
            system_prompt_selector=None,
        ),
    ]
    tool_detail = assert_override_scope(baseline, tool_only, candidate)
    assert tool_detail is not None
    assert "no root llm node carrying the candidate" in tool_detail

    non_root_only = [
        SimpleNamespace(
            index=0,
            parent_index=None,
            secondary_parent_indexes=[],
            node_type="subagent_call",
            inputs={},
            system_prompt_selector=None,
        ),
        SimpleNamespace(
            index=1,
            parent_index=0,
            secondary_parent_indexes=[],
            node_type="llm_call",
            inputs={"system": candidate},
            system_prompt_selector="/system",
        ),
    ]
    nested_detail = assert_override_scope(baseline, non_root_only, candidate)
    assert nested_detail is not None
    assert "no root llm node carrying the candidate" in nested_detail


def test_mixed_agent_version_message_includes_a_breakdown() -> None:
    message = mixed_agent_version_message({"av-1": 12, "av-2": 3})
    assert "av-1: 12 session(s)" in message
    assert "av-2: 3 session(s)" in message


def test_stale_workers_do_not_cover_an_agent_version() -> None:
    claim = SimpleNamespace(kind="agent", agent_version_id="av1")
    stale = SimpleNamespace(live=False, scope=SimpleNamespace(claims=[claim]))
    live = SimpleNamespace(live=True, scope=SimpleNamespace(claims=[claim]))
    assert worker_covers_agent_version(stale, "av1") is False
    assert worker_covers_agent_version(live, "av1") is True


def test_preflight_wraps_list_live_workers_errors() -> None:
    class Gateway:
        async def server_info(self) -> str:
            return "ok"

        async def get_cohort_version(self, cohort_version_id: str) -> object:
            return SimpleNamespace(id=cohort_version_id)

        async def get_agent_version(self, agent_version_id: str) -> object:
            return SimpleNamespace(id=agent_version_id)

        async def list_live_workers(self) -> list[object]:
            raise RuntimeError("connection reset")

    backend = KitaruVerificationBackend(gateway=Gateway())
    with pytest.raises(KitaruVerifyError, match="could not list workers"):
        backend.preflight(_request(agent_version_counts={"av1": 2}))


def test_preflight_ignores_stale_workers_when_checking_coverage() -> None:
    claim = SimpleNamespace(kind="agent", agent_version_id="av1")

    class Gateway:
        async def server_info(self) -> str:
            return "ok"

        async def get_cohort_version(self, cohort_version_id: str) -> object:
            return SimpleNamespace(id=cohort_version_id)

        async def get_agent_version(self, agent_version_id: str) -> object:
            return SimpleNamespace(id=agent_version_id)

        async def list_live_workers(self) -> list[object]:
            return [SimpleNamespace(live=False, scope=SimpleNamespace(claims=[claim]))]

    backend = KitaruVerificationBackend(gateway=Gateway())
    with pytest.raises(KitaruVerifyError, match="no live worker is polling"):
        backend.preflight(_request(agent_version_counts={"av1": 2}))


def test_classify_fail_to_pass_is_improved() -> None:
    baseline = SimpleNamespace(score=0.0, passed=False, data_type="float", value=None)
    candidate = SimpleNamespace(score=1.0, passed=True, data_type="float", value=None)
    assert classify_scores(baseline, candidate) == "improved"
    assert classify_scores(candidate, baseline) == "regressed"
    assert classify_scores(candidate, candidate) == "unchanged"


def _eval(*, version: int = 3, score: float | None = 0.5, passed: bool | None = True) -> SimpleNamespace:
    return SimpleNamespace(
        evaluator_version=version,
        score=score,
        passed=passed,
        data_type="float",
        value=None,
    )


def test_select_evaluation_failure_is_select_evaluation_failed() -> None:
    outcome = classify_replay_session(
        session_id="s-select",
        number=4,
        baseline_eval=REASON_SCORE_UNAVAILABLE,
        candidate_eval=_eval(),
        requested_evaluator_version=3,
    )
    assert isinstance(outcome, Divergence)
    assert outcome.kind == DIVERGENCE_SELECT
    assert outcome.kind == "SELECT_EVALUATION_FAILED"
    assert "baseline: judge-score-unavailable" in outcome.detail
    assert outcome.session_id == "s-select"
    assert outcome.number == 4

    both = classify_replay_session(
        session_id="s-both",
        number=None,
        baseline_eval=REASON_SCORE_UNAVAILABLE,
        candidate_eval=REASON_AMBIGUOUS_EVALUATION,
        requested_evaluator_version=3,
    )
    assert isinstance(both, Divergence)
    assert both.kind == "SELECT_EVALUATION_FAILED"
    assert "baseline: judge-score-unavailable" in both.detail
    assert "candidate: ambiguous-evaluation" in both.detail


def test_evaluator_version_mismatch_is_typed() -> None:
    outcome = classify_replay_session(
        session_id="s-ver",
        number=2,
        baseline_eval=_eval(version=2),
        candidate_eval=_eval(version=3),
        requested_evaluator_version=3,
    )
    assert isinstance(outcome, Divergence)
    assert outcome.kind == DIVERGENCE_EVALUATOR_VERSION
    assert outcome.kind == "EVALUATOR_VERSION_MISMATCH"
    assert "requested evaluator_version 3" in outcome.detail
    assert "baseline=2" in outcome.detail
    assert "candidate=3" in outcome.detail


def test_unclassified_scores_are_score_unclassified() -> None:
    unclassifiable = SimpleNamespace(
        evaluator_version=3,
        score=None,
        passed=None,
        data_type="float",
        value=None,
    )
    assert classify_scores(unclassifiable, unclassifiable) is None
    outcome = classify_replay_session(
        session_id="s-score",
        number=9,
        baseline_eval=unclassifiable,
        candidate_eval=unclassifiable,
        requested_evaluator_version=3,
    )
    assert isinstance(outcome, Divergence)
    assert outcome.kind == DIVERGENCE_SCORE
    assert outcome.kind == "SCORE_UNCLASSIFIED"


def test_replay_session_still_classifies_comparable_scores() -> None:
    outcome = classify_replay_session(
        session_id="s-ok",
        number=1,
        baseline_eval=_eval(score=0.2, passed=False),
        candidate_eval=_eval(score=0.9, passed=True),
        requested_evaluator_version=3,
    )
    assert outcome == "improved"


def test_new_divergence_kinds_stay_in_per_session_buckets(tmp_path: Path) -> None:
    request = _request()
    result = VerificationResult(
        status="completed",
        baseline_count=4,
        candidate_count=4,
        baseline_mean_score=0.4,
        candidate_mean_score=0.4,
        improved_sessions=["s-improved"],
        diverged_sessions=[
            Divergence(
                session_id="s-select",
                kind="SELECT_EVALUATION_FAILED",
                detail="baseline: judge-score-unavailable",
                number=1,
            ),
            Divergence(
                session_id="s-ver",
                kind="EVALUATOR_VERSION_MISMATCH",
                detail="requested evaluator_version 3; baseline=2; candidate=3",
                number=2,
            ),
            Divergence(
                session_id="s-score",
                kind="SCORE_UNCLASSIFIED",
                detail="scores could not be classified as improved, regressed, or unchanged",
                number=3,
            ),
        ],
        replay_failures=[],
        cohort_version_id=request.cohort_version_id,
        agent_version_id=request.agent_version_id,
        evaluator_version=str(request.evaluator_version),
        baseline_prompt_hash=request.baseline_prompt_hash,
        candidate_prompt_hash=request.candidate_prompt_hash,
        verification_fingerprint="fp",
        experiment_run_id="erun-1",
    )

    class DivergedBackend(FakeBackend):
        def collect(
            self, request: VerificationRequest, submitted: SubmittedVerification
        ) -> VerificationResult:
            self.collected += 1
            return result

    stored = run_verification(tmp_path, request, DivergedBackend())
    assert stored.replay_failures == []
    assert {item.kind for item in stored.diverged_sessions} == {
        "SELECT_EVALUATION_FAILED",
        "EVALUATOR_VERSION_MISMATCH",
        "SCORE_UNCLASSIFIED",
    }
    matched = matching_verification(tmp_path, request.candidate_prompt_hash)
    assert matched is not None
    assert matched.result is not None
    assert matched.per_session["s-select"] == "SELECT_EVALUATION_FAILED"
    assert matched.per_session["s-ver"] == "EVALUATOR_VERSION_MISMATCH"
    assert matched.per_session["s-score"] == "SCORE_UNCLASSIFIED"
    assert matched.per_session["s-improved"] == "improved"
    report = format_verification_report(stored, request)
    assert "SELECT_EVALUATION_FAILED" in report
    assert "EVALUATOR_VERSION_MISMATCH" in report
    assert "SCORE_UNCLASSIFIED" in report
    assert "judge-score-unavailable" in report
    assert stored.baseline_count == 4
    assert stored.candidate_mean_score == 0.4


def test_interrupted_verify_does_not_duplicate_the_experiment(tmp_path: Path) -> None:
    backend = FakeBackend()
    request = _request()
    first = run_verification(tmp_path, request, backend)
    assert backend.submitted == 1
    second = run_verification(tmp_path, request, backend)
    assert backend.submitted == 1
    assert backend.collected == 2
    assert first.experiment_run_id == second.experiment_run_id == "erun-1"
    assert matching_verification(tmp_path, request.candidate_prompt_hash) is not None


def test_apply_is_gated_on_a_matching_hash(tmp_path: Path) -> None:
    proposal = _proposal(tmp_path)
    _source_sidecar(tmp_path)
    about_to_write = candidate_prompt(
        PROMPT, proposal, range(len(proposal.edits))
    )
    digest = text_hash(about_to_write)
    with pytest.raises(VerifyError, match="hash-matching verification"):
        refuse_ungated_apply(
            tmp_path, run_id="run-0001", candidate_prompt_hash=digest, force=False
        )
    refuse_ungated_apply(
        tmp_path, run_id="run-0001", candidate_prompt_hash=digest, force=True
    )


def test_partial_apply_after_full_verify_needs_force_or_all(tmp_path: Path) -> None:
    proposal = _multi_edit_proposal(tmp_path)
    _source_sidecar(tmp_path)
    full = candidate_prompt(PROMPT, proposal, range(len(proposal.edits)))
    subset = candidate_prompt(PROMPT, proposal, [1])
    assert text_hash(full) != text_hash(subset)
    run_verification(
        tmp_path,
        _request(candidate_prompt=full, candidate_prompt_hash=text_hash(full)),
        FakeBackend(),
    )
    refuse_ungated_apply(
        tmp_path, run_id="run-0001", candidate_prompt_hash=text_hash(full), force=False
    )
    with pytest.raises(VerifyError, match="apply --all") as caught:
        refuse_ungated_apply(
            tmp_path,
            run_id="run-0001",
            candidate_prompt_hash=text_hash(subset),
            force=False,
        )
    message = str(caught.value)
    assert "--force" in message
    assert "re-running verify cannot ungate" in message
    assert "Run `tracegrad verify --backend kitaru` first" not in message


def test_apply_gate_does_not_affect_core_only_runs(tmp_path: Path) -> None:
    proposal = _proposal(tmp_path)
    about_to_write = candidate_prompt(PROMPT, proposal, [0])
    refuse_ungated_apply(
        tmp_path,
        run_id="run-0001",
        candidate_prompt_hash=text_hash(about_to_write),
        force=False,
    )
    result = apply_proposal(tmp_path, proposal, [0], base_directory=tmp_path)
    assert "Always cite the doc." in (tmp_path / "prompt.md").read_text(encoding="utf-8")
    assert result.accepted


def test_hand_edit_after_verify_misses_the_gate(tmp_path: Path) -> None:
    proposal = _proposal(tmp_path)
    _source_sidecar(tmp_path)
    backend = FakeBackend()
    written = candidate_prompt(PROMPT, proposal, [0])
    request = _request(
        candidate_prompt=written,
        candidate_prompt_hash=text_hash(written),
    )
    run_verification(tmp_path, request, backend)
    refuse_ungated_apply(
        tmp_path, run_id="run-0001", candidate_prompt_hash=text_hash(written), force=False
    )
    with pytest.raises(VerifyError, match="hash-matching"):
        refuse_ungated_apply(
            tmp_path,
            run_id="run-0001",
            candidate_prompt_hash=text_hash(written + "\n# hand edit\n"),
            force=False,
        )


def test_apply_gate_requires_a_stored_result(tmp_path: Path) -> None:
    """A submit that never stored result must not ungate apply."""

    proposal = _proposal(tmp_path)
    _source_sidecar(tmp_path)
    written = candidate_prompt(PROMPT, proposal, [0])
    digest = text_hash(written)

    class SubmitOnlyBackend(FakeBackend):
        def collect(
            self, request: VerificationRequest, submitted: SubmittedVerification
        ) -> VerificationResult:
            raise RuntimeError("collect exploded")

    request = _request(candidate_prompt=written, candidate_prompt_hash=digest)
    with pytest.raises(RuntimeError, match="collect exploded"):
        run_verification(tmp_path, request, SubmitOnlyBackend())
    assert matching_verification(tmp_path, digest) is None
    with pytest.raises(VerifyError, match="hash-matching"):
        refuse_ungated_apply(
            tmp_path, run_id="run-0001", candidate_prompt_hash=digest, force=False
        )


def test_collect_fetch_error_does_not_persist_result_or_ungate_apply(tmp_path: Path) -> None:
    """A session_nodes/evaluations 404 or timeout aborts collect fail-closed."""

    proposal = _proposal(tmp_path)
    _source_sidecar(tmp_path)
    written = candidate_prompt(PROMPT, proposal, [0])
    digest = text_hash(written)
    request = _request(candidate_prompt=written, candidate_prompt_hash=digest)

    def _root(system: str) -> SimpleNamespace:
        return SimpleNamespace(
            index=0,
            parent_index=None,
            secondary_parent_indexes=[],
            node_type="llm_call",
            inputs={"system": system},
            system_prompt_selector="/system",
        )

    class FetchErrorGateway:
        async def wait_for_experiment_run(
            self, run_id: str, timeout: float | None = None
        ) -> object:
            return SimpleNamespace(status="completed")

        async def list_replays(self, experiment_run_id: str) -> list[object]:
            return [
                SimpleNamespace(
                    baseline_session_id="s0",
                    result_session_id="r0",
                    status="completed",
                    error=None,
                ),
                SimpleNamespace(
                    baseline_session_id="s1",
                    result_session_id="r1",
                    status="completed",
                    error=None,
                ),
            ]

        async def session_nodes(self, session_id: str) -> tuple[object, ...]:
            system = CANDIDATE if session_id.startswith("r") else PROMPT
            return (_root(system),)

        async def evaluations_for(self, session_id: str) -> list[object]:
            if session_id == "r1":
                raise TimeoutError("404/timeout fetching evaluations for r1")
            passed = session_id.startswith("r")
            return [
                SimpleNamespace(
                    name="quality",
                    evaluator_version=3,
                    passed=passed,
                    score=0.9 if passed else 0.2,
                    id=session_id,
                )
            ]

        async def evaluation_aggregates(self, experiment_run_id: str) -> list[object]:
            return [
                {
                    "name": "quality",
                    "baseline": {"count": 2, "mean": 0.2, "pass_rate": 0.0},
                    "result": {"count": 2, "mean": 0.9, "pass_rate": 1.0},
                }
            ]

    class FetchErrorBackend:
        name = "kitaru"

        def preflight(self, request: VerificationRequest) -> None:
            return None

        def submit(self, request: VerificationRequest) -> SubmittedVerification:
            return SubmittedVerification(experiment_id="exp-1", experiment_run_id="erun-1")

        def collect(
            self, request: VerificationRequest, submitted: SubmittedVerification
        ) -> VerificationResult:
            return KitaruVerificationBackend(gateway=FetchErrorGateway()).collect(
                request, submitted
            )

    with pytest.raises(KitaruVerifyError, match="collect aborted") as caught:
        run_verification(tmp_path, request, FetchErrorBackend())
    message = str(caught.value)
    assert "Apply stays gated" in message
    assert "Resume redoes collect" in message
    assert "--force" in message
    assert "TimeoutError" in message
    assert caught.value.__cause__ is not None

    assert matching_verification(tmp_path, digest) is None
    state = load_verification_state(
        initialize(tmp_path), verification_id_for("run-0001", digest)
    )
    assert state is not None
    assert state.experiment_run_id == "erun-1"
    assert state.result is None
    with pytest.raises(VerifyError, match="hash-matching"):
        refuse_ungated_apply(
            tmp_path, run_id="run-0001", candidate_prompt_hash=digest, force=False
        )


def _collect_sdk_error_gateway(broken: str) -> object:
    def _root(system: str) -> SimpleNamespace:
        return SimpleNamespace(
            index=0,
            parent_index=None,
            secondary_parent_indexes=[],
            node_type="llm_call",
            inputs={"system": system},
            system_prompt_selector="/system",
        )

    class Gateway:
        async def wait_for_experiment_run(
            self, run_id: str, timeout: float | None = None
        ) -> object:
            if broken == "wait":
                raise TimeoutError("wait timed out")
            return SimpleNamespace(status="completed")

        async def list_replays(self, experiment_run_id: str) -> list[object]:
            if broken == "replays":
                raise RuntimeError("APIError listing replays")
            return [
                SimpleNamespace(
                    baseline_session_id="s0",
                    result_session_id="r0",
                    status="completed",
                    error=None,
                )
            ]

        async def session_nodes(self, session_id: str) -> tuple[object, ...]:
            system = CANDIDATE if session_id.startswith("r") else PROMPT
            return (_root(system),)

        async def evaluations_for(self, session_id: str) -> list[object]:
            passed = session_id.startswith("r")
            return [
                SimpleNamespace(
                    name="quality",
                    evaluator_version=3,
                    passed=passed,
                    score=0.9 if passed else 0.2,
                    id=session_id,
                )
            ]

        async def evaluation_aggregates(self, experiment_run_id: str) -> list[object]:
            if broken == "aggregates":
                raise TimeoutError("aggregates 404")
            return [
                {
                    "name": "quality",
                    "baseline": {"count": 1, "mean": 0.2, "pass_rate": 0.0},
                    "result": {"count": 1, "mean": 0.9, "pass_rate": 1.0},
                }
            ]

    return Gateway()


@pytest.mark.parametrize("broken", ["wait", "replays", "aggregates"])
def test_collect_sdk_errors_do_not_persist_result_or_ungate_apply(
    tmp_path: Path, broken: str
) -> None:
    proposal = _proposal(tmp_path)
    _source_sidecar(tmp_path)
    written = candidate_prompt(PROMPT, proposal, [0])
    digest = text_hash(written)
    request = _request(candidate_prompt=written, candidate_prompt_hash=digest)

    class Backend:
        name = "kitaru"

        def preflight(self, request: VerificationRequest) -> None:
            return None

        def submit(self, request: VerificationRequest) -> SubmittedVerification:
            return SubmittedVerification(experiment_id="exp-1", experiment_run_id="erun-1")

        def collect(
            self, request: VerificationRequest, submitted: SubmittedVerification
        ) -> VerificationResult:
            return KitaruVerificationBackend(
                gateway=_collect_sdk_error_gateway(broken)
            ).collect(request, submitted)

    with pytest.raises(KitaruVerifyError, match="collect aborted") as caught:
        run_verification(tmp_path, request, Backend())
    assert "Apply stays gated" in str(caught.value)
    assert matching_verification(tmp_path, digest) is None
    state = load_verification_state(
        initialize(tmp_path), verification_id_for("run-0001", digest)
    )
    assert state is not None
    assert state.experiment_run_id == "erun-1"
    assert state.result is None


def test_submit_sdk_errors_are_kitaru_verify_errors() -> None:
    pytest.importorskip("kitaru")

    class Gateway:
        async def create_experiment(self, request: object) -> object:
            raise RuntimeError("APIError creating experiment")

        async def start_run(self, experiment_id: str, request: object) -> object:
            raise AssertionError("start_run must not run after create_experiment fails")

    backend = KitaruVerificationBackend(gateway=Gateway())
    request = _request(
        agent_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        agent_version_id="bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
        cohort_version_id="cccccccc-cccc-4ccc-8ccc-cccccccccccc",
    )
    with pytest.raises(KitaruVerifyError, match="submit aborted") as caught:
        backend.submit(request)
    assert "creating the experiment" in str(caught.value)
    assert "Apply stays gated" in str(caught.value)


def test_apply_gate_accepts_a_finished_failed_report(tmp_path: Path) -> None:
    """A finished REVIEW/FAILED report still allows apply; do not require completed."""

    proposal = _proposal(tmp_path)
    _source_sidecar(tmp_path)
    written = candidate_prompt(PROMPT, proposal, [0])
    digest = text_hash(written)

    class FailedBackend(FakeBackend):
        def collect(
            self, request: VerificationRequest, submitted: SubmittedVerification
        ) -> VerificationResult:
            result = super().collect(request, submitted)
            return result.model_copy(update={"status": "failed"})

    request = _request(candidate_prompt=written, candidate_prompt_hash=digest)
    run_verification(tmp_path, request, FailedBackend())
    matched = matching_verification(tmp_path, digest)
    assert matched is not None
    assert matched.result is not None
    assert matched.result.status == "failed"
    refuse_ungated_apply(
        tmp_path, run_id="run-0001", candidate_prompt_hash=digest, force=False
    )


def test_verification_id_is_path_safe() -> None:
    vid = verification_id_for("run-0001", "sha256:abcdef1234567890")
    assert vid.startswith("verify-run-0001-")
    assert "/" not in vid


def test_build_request_refuses_a_stale_proposal_without_submitting(tmp_path: Path) -> None:
    proposal = _proposal(tmp_path)
    _source_sidecar(tmp_path)
    source = load_run_source_payload(tmp_path, "run-0001")
    assert source is not None
    (tmp_path / "prompt.md").write_text(PROMPT + "- Edited by hand.\n", encoding="utf-8")
    backend = FakeBackend()
    with pytest.raises(StaleProposalError, match="stale"):
        build_request(
            project_root=tmp_path,
            run_id="run-0001",
            proposal=proposal,
            base_directory=tmp_path,
            source=source,
        )
    assert backend.submitted == 0
    assert backend.preflighted == 0
    layout = initialize(tmp_path)
    assert list(layout.verification.glob("*")) == []


def test_build_request_wraps_unreadable_template_as_verify_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proposal = _proposal(tmp_path)
    _source_sidecar(tmp_path)
    source = load_run_source_payload(tmp_path, "run-0001")
    assert source is not None
    (tmp_path / "prompt.md").unlink()
    monkeypatch.setattr("tracegrad.verify.is_stale", lambda *args, **kwargs: False)
    with pytest.raises(VerifyError, match="could not read template"):
        build_request(
            project_root=tmp_path,
            run_id="run-0001",
            proposal=proposal,
            base_directory=tmp_path,
            source=source,
        )


def test_corrupt_run_source_sidecar_does_not_traceback_or_ungate(tmp_path: Path) -> None:
    proposal = _proposal(tmp_path)
    _source_sidecar(tmp_path)
    sidecar = tmp_path / ".tracegrad" / "runs" / "run-0001" / "kitaru-source.json"
    sidecar.write_text("not-json", encoding="utf-8")

    assert load_run_source_payload(tmp_path, "run-0001") is None
    assert backend_is_configured(tmp_path, "run-0001") is True
    with pytest.raises(VerifyError, match="hash-matching"):
        refuse_ungated_apply(
            tmp_path,
            run_id="run-0001",
            candidate_prompt_hash=text_hash(candidate_prompt(PROMPT, proposal, [0])),
            force=False,
        )


def test_incomplete_run_source_sidecar_is_rejected_and_stays_gated(tmp_path: Path) -> None:
    proposal = _proposal(tmp_path)
    _source_sidecar(tmp_path)
    sidecar = tmp_path / ".tracegrad" / "runs" / "run-0001" / "kitaru-source.json"
    digest = text_hash(candidate_prompt(PROMPT, proposal, [0]))

    for payload in ({}, {"fingerprint": {}, "meta": {}}, {"fingerprint": {"cohort_id": "c1"}, "meta": {}}):
        sidecar.write_text(json.dumps(payload), encoding="utf-8")
        assert load_run_source_payload(tmp_path, "run-0001") is None
        assert backend_is_configured(tmp_path, "run-0001") is True
        with pytest.raises(VerifyError, match="hash-matching"):
            refuse_ungated_apply(
                tmp_path, run_id="run-0001", candidate_prompt_hash=digest, force=False
            )

    sidecar.write_text(json.dumps({"fingerprint": "nope", "meta": {}}), encoding="utf-8")
    assert load_run_source_payload(tmp_path, "run-0001") is None
    with pytest.raises(VerifyError, match="usable Kitaru sidecar"):
        build_request(
            project_root=tmp_path,
            run_id="run-0001",
            proposal=proposal,
            base_directory=tmp_path,
            source={"fingerprint": "nope", "meta": {}},
        )


def test_verify_cli_refuses_a_stale_proposal_before_kitaru(tmp_path: Path) -> None:
    from tracegrad import cli

    _proposal(tmp_path)
    _source_sidecar(tmp_path)
    (tmp_path / "prompt.md").write_text(PROMPT + "- Edited by hand.\n", encoding="utf-8")
    stream = io.StringIO()
    code = cli.main(
        [
            "verify",
            "--backend",
            "kitaru",
            "--run",
            "run-0001",
            "--project-root",
            str(tmp_path),
            "--base-directory",
            str(tmp_path),
        ],
        out=stream,
    )
    assert code == 1
    output = stream.getvalue()
    assert "stale" in output
    assert "re-run tracegrad" in output
    assert "prompt.md changed since run run-0001" in output
    layout = initialize(tmp_path)
    assert list(layout.verification.glob("*")) == []
    assert any(record.get("event") == "stale" for record in applied_history(tmp_path))


def test_report_lists_replay_failures_next_to_divergence() -> None:
    request = _request(session_numbers={"crash-session": 7})
    result = VerificationResult(
        status="partial",
        baseline_count=2,
        candidate_count=1,
        improved_sessions=[],
        regressed_sessions=[],
        unchanged_sessions=[],
        diverged_sessions=[
            Divergence(
                session_id="miss-session",
                kind="TOOL_HISTORY_MISS",
                detail="No history result for tool 'search'",
                number=3,
            )
        ],
        replay_failures=[
            ReplayFailure(
                session_id="crash-session",
                error="worker process exited 1",
                number=7,
            )
        ],
        cohort_version_id=request.cohort_version_id,
        agent_version_id=request.agent_version_id,
        evaluator_version=str(request.evaluator_version),
        baseline_prompt_hash=request.baseline_prompt_hash,
        candidate_prompt_hash=request.candidate_prompt_hash,
        verification_fingerprint="fp",
        experiment_run_id="erun-1",
    )
    report = format_verification_report(result, request)
    assert "Replay failures           1" in report
    assert "Diverged                  1" in report
    assert "Replay failures" in report.split("Divergence", 1)[1]
    assert "#7   worker process exited 1" in report
    assert "TOOL_HISTORY_MISS" in report


def test_second_run_async_does_not_reuse_a_closed_loop_client(monkeypatch: pytest.MonkeyPatch) -> None:
    from tracegrad.integrations.kitaru import backend as backend_mod

    created: list[object] = []

    class LoopBoundGateway:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self._loop: asyncio.AbstractEventLoop | None = None
            self._closed = False
            created.append(self)

        async def probe(self) -> str:
            loop = asyncio.get_running_loop()
            if self._closed or (
                self._loop is not None and (self._loop is not loop or self._loop.is_closed())
            ):
                raise RuntimeError("reused client bound to a closed loop")
            self._loop = loop
            return "ok"

        async def close(self) -> None:
            self._closed = True

    monkeypatch.setattr(backend_mod, "require_kitaru", lambda: None)
    monkeypatch.setattr(backend_mod, "KitaruGateway", LoopBoundGateway)

    backend = backend_mod.KitaruVerificationBackend()

    async def probe() -> str:
        return await backend._gw().probe()

    assert backend._run_async(probe()) == "ok"
    assert backend._run_async(probe()) == "ok"
    assert len(created) == 2
    assert created[0] is not created[1]
    assert created[0]._closed is True
    assert created[1]._closed is True


def test_collect_fetches_replay_payloads_with_bounded_concurrency() -> None:
    n_success = FETCH_JOBS + 4
    fetched: list[str] = []

    class CountingGateway:
        def __init__(self) -> None:
            self.http_in_flight = 0
            self.max_http_in_flight = 0
            self._replay_counts: dict[str, int] = {}
            self.max_replay_in_flight = 0
            self._lock = asyncio.Lock()

        def _replay_key(self, session_id: str) -> str:
            if session_id.startswith("r") and session_id[1:].isdigit():
                return f"s{session_id[1:]}"
            return session_id

        async def _track(self, session_id: str) -> None:
            if session_id in {"fail-session", "hist-session"}:
                raise AssertionError(f"failed replay {session_id} must not be fetched")
            key = self._replay_key(session_id)
            async with self._lock:
                self.http_in_flight += 1
                self.max_http_in_flight = max(self.max_http_in_flight, self.http_in_flight)
                self._replay_counts[key] = self._replay_counts.get(key, 0) + 1
                self.max_replay_in_flight = max(
                    self.max_replay_in_flight, len(self._replay_counts)
                )
                fetched.append(session_id)
            try:
                await asyncio.sleep(0.05)
            finally:
                async with self._lock:
                    self.http_in_flight -= 1
                    self._replay_counts[key] -= 1
                    if self._replay_counts[key] == 0:
                        del self._replay_counts[key]

        async def wait_for_experiment_run(
            self, run_id: str, timeout: float | None = None
        ) -> object:
            return SimpleNamespace(status="completed")

        async def list_replays(self, experiment_run_id: str) -> list[object]:
            rows: list[object] = [
                SimpleNamespace(
                    baseline_session_id="fail-session",
                    result_session_id=None,
                    status="failed",
                    error="worker crashed",
                ),
                SimpleNamespace(
                    baseline_session_id="hist-session",
                    result_session_id=None,
                    status="failed",
                    error="No history result for tool 'search'",
                ),
            ]
            for index in range(n_success):
                rows.append(
                    SimpleNamespace(
                        baseline_session_id=f"s{index}",
                        result_session_id=f"r{index}",
                        status="completed",
                        error=None,
                    )
                )
            return rows

        async def session_nodes(self, session_id: str) -> tuple[object, ...]:
            await self._track(session_id)
            system = CANDIDATE if session_id.startswith("r") else PROMPT
            return (
                SimpleNamespace(
                    index=0,
                    parent_index=None,
                    secondary_parent_indexes=[],
                    node_type="llm_call",
                    inputs={"system": system},
                    system_prompt_selector="/system",
                ),
            )

        async def evaluations_for(self, session_id: str) -> list[object]:
            await self._track(session_id)
            passed = session_id.startswith("r")
            return [
                SimpleNamespace(
                    name="quality",
                    evaluator_version=3,
                    passed=passed,
                    score=0.9 if passed else 0.2,
                    id=session_id,
                )
            ]

        async def evaluation_aggregates(self, experiment_run_id: str) -> list[object]:
            return [
                {
                    "name": "quality",
                    "baseline": {"count": n_success, "mean": 0.2, "pass_rate": 0.0},
                    "result": {"count": n_success, "mean": 0.9, "pass_rate": 1.0},
                }
            ]

    gateway = CountingGateway()
    backend = KitaruVerificationBackend(gateway=gateway)
    request = _request(session_numbers={f"s{index}": index for index in range(n_success)})
    result = backend.collect(
        request, SubmittedVerification(experiment_id="exp-1", experiment_run_id="erun-1")
    )

    assert gateway.max_replay_in_flight > 1
    assert gateway.max_replay_in_flight <= FETCH_JOBS
    assert gateway.max_http_in_flight > 1
    assert result.improved_sessions == [f"s{index}" for index in range(n_success)]
    assert result.replay_failures == [
        ReplayFailure(session_id="fail-session", error="worker crashed", number=None)
    ]
    assert result.diverged_sessions[0].kind == DIVERGENCE_HISTORY
    assert result.diverged_sessions[0].session_id == "hist-session"
    assert "fail-session" not in fetched
    assert "hist-session" not in fetched
    assert result.baseline_count == n_success
    assert result.candidate_count == n_success
