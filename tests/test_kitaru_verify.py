"""Phase 2 verification: gate, resume, policy, override scope (issue #9)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tracegrad.apply import (
    Proposal,
    ProposedEdit,
    apply_proposal,
    candidate_prompt,
    save_proposal,
)
from tracegrad.canonical import text_hash
from tracegrad.edits import resolve_edits
from tracegrad.integrations.kitaru.backend import (
    assert_override_scope,
    classify_scores,
    is_tool_history_miss,
    mixed_agent_version_message,
)
from tracegrad.integrations.kitaru.policy import (
    RECORDED_HISTORY_POLICY,
    asserts_no_passthrough,
    recorded_history_policy,
)
from tracegrad.integrations.kitaru.snapshot import RUN_SOURCE_FILENAME
from tracegrad.inventory import build_inventory
from tracegrad.schema import Edit
from tracegrad.state import atomic_write_json, initialize
from tracegrad.verify import (
    SubmittedVerification,
    VerificationRequest,
    VerificationResult,
    VerifyError,
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


def test_mixed_agent_version_message_includes_a_breakdown() -> None:
    message = mixed_agent_version_message({"av-1": 12, "av-2": 3})
    assert "av-1: 12 session(s)" in message
    assert "av-2: 3 session(s)" in message


def test_classify_fail_to_pass_is_improved() -> None:
    baseline = SimpleNamespace(score=0.0, passed=False, data_type="float", value=None)
    candidate = SimpleNamespace(score=1.0, passed=True, data_type="float", value=None)
    assert classify_scores(baseline, candidate) == "improved"
    assert classify_scores(candidate, baseline) == "regressed"
    assert classify_scores(candidate, candidate) == "unchanged"


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
