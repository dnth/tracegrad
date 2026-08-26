"""Replay verification orchestrator.

Holds a :class:`~tracegrad.ports.VerificationBackend` without becoming
backend-aware (ADR 0010).  Persist / resume lives here so an interrupted
verify never duplicates the experiment.  Apply-gating on
``candidate_prompt_hash`` lives here so the core of ``apply`` stays a writer,
not a Kitaru client.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.types import StrictFloat, StrictInt, StrictStr

from .apply import Proposal, StaleProposalError, candidate_prompt, is_stale, load_proposal
from .canonical import content_hash, text_hash
from .ports import VerificationBackend
from .state import (
    StateLayout,
    atomic_write_json,
    contained_path,
    initialize,
    validate_run_id,
)

RUN_SOURCE_FILENAME = "kitaru-source.json"

DIVERGENCE_HISTORY = "TOOL_HISTORY_MISS"
DIVERGENCE_SCOPE = "OVERRIDE_SCOPE_DIVERGENCE"
DIVERGENCE_EVALUATOR_VERSION = "EVALUATOR_VERSION_MISMATCH"
DIVERGENCE_SELECT = "SELECT_EVALUATION_FAILED"
DIVERGENCE_SCORE = "SCORE_UNCLASSIFIED"
VERIFICATION_FILENAME = "state.json"

DivergenceKind = Literal[
    "TOOL_HISTORY_MISS",
    "OVERRIDE_SCOPE_DIVERGENCE",
    "EVALUATOR_VERSION_MISMATCH",
    "SELECT_EVALUATION_FAILED",
    "SCORE_UNCLASSIFIED",
]


class VerifyError(ValueError):
    """Verification cannot start or cannot gate apply."""


class Divergence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: StrictStr
    kind: DivergenceKind
    detail: StrictStr = ""
    number: StrictInt | None = None


class ReplayFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: StrictStr
    error: StrictStr
    number: StrictInt | None = None


class VerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["completed", "partial", "failed"]
    baseline_count: StrictInt = 0
    candidate_count: StrictInt = 0
    baseline_mean_score: StrictFloat | None = None
    candidate_mean_score: StrictFloat | None = None
    baseline_pass_rate: StrictFloat | None = None
    candidate_pass_rate: StrictFloat | None = None
    improved_sessions: list[StrictStr] = Field(default_factory=list)
    regressed_sessions: list[StrictStr] = Field(default_factory=list)
    unchanged_sessions: list[StrictStr] = Field(default_factory=list)
    diverged_sessions: list[Divergence] = Field(default_factory=list)
    replay_failures: list[ReplayFailure] = Field(default_factory=list)
    cohort_version_id: StrictStr
    agent_version_id: StrictStr
    evaluator_version: StrictStr
    baseline_prompt_hash: StrictStr
    candidate_prompt_hash: StrictStr
    verification_fingerprint: StrictStr
    experiment_run_id: StrictStr


class SubmittedVerification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: StrictStr
    experiment_run_id: StrictStr


class VerificationRequest(BaseModel):
    """What a backend needs, all sourced from the originating run's snapshot."""

    model_config = ConfigDict(extra="forbid")

    run_id: StrictStr
    proposal_id: StrictStr
    candidate_prompt: StrictStr
    candidate_prompt_hash: StrictStr
    baseline_prompt_hash: StrictStr
    cohort_id: StrictStr
    cohort_version_id: StrictStr
    cohort_name: StrictStr
    display_version: StrictStr | None = None
    evaluation_name: StrictStr
    evaluator_id: StrictStr
    evaluator_version: StrictInt
    evaluator_name: StrictStr
    agent_id: StrictStr
    agent_version_id: StrictStr
    agent_version_counts: dict[StrictStr, StrictInt] = Field(default_factory=dict)
    system_prompts: dict[StrictStr, StrictStr] = Field(default_factory=dict)
    session_numbers: dict[StrictStr, StrictInt] = Field(default_factory=dict)


class VerificationState(BaseModel):
    """Persisted verification record under ``.tracegrad/verification/``."""

    model_config = ConfigDict(extra="forbid")

    verification_id: StrictStr
    run_id: StrictStr
    proposal_id: StrictStr
    experiment_id: StrictStr | None = None
    experiment_run_id: StrictStr | None = None
    cohort_version_id: StrictStr
    evaluator_version: StrictStr
    agent_version_id: StrictStr
    baseline_prompt_hash: StrictStr
    candidate_prompt_hash: StrictStr
    tool_policy: dict[str, str] = Field(default_factory=dict)
    per_session: dict[str, str] = Field(default_factory=dict)
    result: VerificationResult | None = None
    verification_fingerprint: StrictStr | None = None


def verification_id_for(run_id: str, candidate_prompt_hash: str) -> str:
    digest = candidate_prompt_hash.removeprefix("sha256:")[:12]
    return f"verify-{validate_run_id(run_id)}-{digest}"


def verification_dir(layout: StateLayout, verification_id: str) -> Path:
    return layout.verification / verification_id


def verification_path(layout: StateLayout, verification_id: str) -> Path:
    return verification_dir(layout, verification_id) / VERIFICATION_FILENAME


def save_verification_state(layout: StateLayout, state: VerificationState) -> Path:
    target = verification_path(layout, state.verification_id)
    atomic_write_json(target, state.model_dump(mode="json"))
    return target


def load_verification_state(layout: StateLayout, verification_id: str) -> VerificationState | None:
    target = verification_path(layout, verification_id)
    if not target.exists():
        return None
    return VerificationState.model_validate_json(target.read_text(encoding="utf-8"))


def list_verification_states(layout: StateLayout) -> list[VerificationState]:
    records: list[VerificationState] = []
    for path in sorted(layout.verification.glob(f"*/{VERIFICATION_FILENAME}")):
        try:
            records.append(VerificationState.model_validate_json(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return records


def matching_verification(
    project_root: str | Path,
    candidate_prompt_hash: str,
) -> VerificationState | None:
    """A persisted verification of exactly this candidate text, if any.

    Requires a stored ``result``. A submit that never collected (``result``
    is ``None``) must not ungate apply. A finished REVIEW/FAILED report
    still matches; status is not required to be ``completed``.
    """

    layout = initialize(project_root)
    for state in list_verification_states(layout):
        if (
            state.candidate_prompt_hash == candidate_prompt_hash
            and state.experiment_run_id
            and state.result is not None
        ):
            return state
    return None


def load_run_source_payload(project_root: str | Path, run_id: str) -> dict[str, Any] | None:
    """Read the originating-run source sidecar, if the run had a backend."""

    layout = initialize(project_root)
    target = layout.runs / validate_run_id(run_id) / RUN_SOURCE_FILENAME
    if not target.exists():
        return None
    import json

    return json.loads(target.read_text(encoding="utf-8"))


def backend_is_configured(project_root: str | Path, run_id: str) -> bool:
    """Whether this run originated from a Kitaru source (ADR 0009)."""

    return load_run_source_payload(project_root, run_id) is not None


def refuse_ungated_apply(
    project_root: str | Path,
    *,
    run_id: str,
    candidate_prompt_hash: str,
    force: bool,
) -> None:
    """Refuse apply when a backend is configured and no matching verify exists."""

    if force or not backend_is_configured(project_root, run_id):
        return
    if matching_verification(project_root, candidate_prompt_hash) is None:
        raise VerifyError(
            "apply is gated on a hash-matching verification for this candidate. "
            "Run `tracegrad verify --backend kitaru` first, or pass --force "
            "to override. Matching the hash (not the run id) is what makes the "
            "gate real: verify, hand-edit, and apply notices the text was never "
            "verified. See ADR 0009."
        )


def build_request(
    *,
    project_root: str | Path,
    run_id: str,
    proposal: Proposal,
    base_directory: str | Path = ".",
    source: dict[str, Any],
) -> VerificationRequest:
    if is_stale(proposal, base_directory=base_directory):
        raise StaleProposalError(
            f"{proposal.template_file} changed since run {run_id}; "
            "the proposal is stale — re-run tracegrad"
        )
    template = contained_path(base_directory, proposal.template_file)
    try:
        current = template.read_text(encoding="utf-8")
    except OSError as exc:
        raise VerifyError(f"could not read template {template}: {exc}") from exc
    candidate = candidate_prompt(
        current, proposal, range(len(proposal.edits))
    )
    fingerprint = source["fingerprint"]
    meta = source["meta"]
    agent_version_id = meta.get("agent_version_id")
    if not agent_version_id:
        raise VerifyError(
            "verification requires a single agent_version_id on the originating "
            "cohort; this run's source metadata does not have one. See ADR 0007."
        )
    return VerificationRequest(
        run_id=run_id,
        proposal_id=proposal.run_id,
        candidate_prompt=candidate,
        candidate_prompt_hash=text_hash(candidate),
        baseline_prompt_hash=proposal.base_prompt_hash,
        cohort_id=str(fingerprint["cohort_id"]),
        cohort_version_id=str(fingerprint["cohort_version_id"]),
        cohort_name=str(meta.get("cohort_name") or ""),
        display_version=meta.get("display_version"),
        evaluation_name=str(fingerprint["evaluation_name"]),
        evaluator_id=str(fingerprint["evaluator_id"]),
        evaluator_version=int(fingerprint["evaluator_version"]),
        evaluator_name=str(meta.get("evaluator_name") or fingerprint["evaluation_name"]),
        agent_id=str(fingerprint["agent_id"]),
        agent_version_id=str(agent_version_id),
        agent_version_counts=dict(meta.get("agent_version_counts") or {}),
        system_prompts=dict(meta.get("system_prompts") or {}),
        session_numbers={
            key: int(value) for key, value in (meta.get("session_numbers") or {}).items()
        },
    )


def verification_fingerprint_for(request: VerificationRequest) -> str:
    return content_hash(
        {
            "cohort_version_id": request.cohort_version_id,
            "agent_version_id": request.agent_version_id,
            "evaluator_version": request.evaluator_version,
            "evaluation_name": request.evaluation_name,
            "baseline_prompt_hash": request.baseline_prompt_hash,
            "candidate_prompt_hash": request.candidate_prompt_hash,
            "tool_policy": {
                "type": "history",
                "scope": "cohort_version",
                "on_miss": "fail",
            },
        }
    )


def run_verification(
    project_root: str | Path,
    request: VerificationRequest,
    backend: VerificationBackend,
) -> VerificationResult:
    """Submit (or resume) verification and persist the result."""

    layout = initialize(project_root)
    vid = verification_id_for(request.run_id, request.candidate_prompt_hash)
    state = load_verification_state(layout, vid) or VerificationState(
        verification_id=vid,
        run_id=request.run_id,
        proposal_id=request.proposal_id,
        cohort_version_id=request.cohort_version_id,
        evaluator_version=str(request.evaluator_version),
        agent_version_id=request.agent_version_id,
        baseline_prompt_hash=request.baseline_prompt_hash,
        candidate_prompt_hash=request.candidate_prompt_hash,
        tool_policy={
            "type": "history",
            "scope": "cohort_version",
            "on_miss": "fail",
        },
        verification_fingerprint=verification_fingerprint_for(request),
    )
    save_verification_state(layout, state)

    if state.experiment_run_id is None:
        backend.preflight(request)
        submitted = backend.submit(request)
        if not isinstance(submitted, SubmittedVerification):
            submitted = SubmittedVerification.model_validate(
                submitted if isinstance(submitted, dict) else submitted.__dict__
            )
        state = state.model_copy(
            update={
                "experiment_id": submitted.experiment_id,
                "experiment_run_id": submitted.experiment_run_id,
            }
        )
        save_verification_state(layout, state)

    submitted = SubmittedVerification(
        experiment_id=state.experiment_id or "",
        experiment_run_id=state.experiment_run_id or "",
    )
    result = backend.collect(request, submitted)
    if not isinstance(result, VerificationResult):
        result = VerificationResult.model_validate(result)
    state = state.model_copy(
        update={
            "result": result,
            "verification_fingerprint": result.verification_fingerprint,
            "per_session": {
                **{sid: "improved" for sid in result.improved_sessions},
                **{sid: "regressed" for sid in result.regressed_sessions},
                **{sid: "unchanged" for sid in result.unchanged_sessions},
                **{item.session_id: item.kind for item in result.diverged_sessions},
            },
        }
    )
    save_verification_state(layout, state)
    return result


def format_verification_report(
    result: VerificationResult,
    request: VerificationRequest,
    *,
    server_url: str = "",
) -> str:
    """Human-readable summary.  Never prints SHIP (issue #9)."""

    label = request.cohort_name
    if request.display_version:
        label = f"{request.cohort_name}/{request.display_version}"
    sessions = (
        result.baseline_count
        or len(result.improved_sessions)
        + len(result.regressed_sessions)
        + len(result.unchanged_sessions)
        + len(result.diverged_sessions)
        + len(result.replay_failures)
    )
    def _pct(value: float | None) -> str:
        return "     n/a" if value is None else f"{value * 100:8.1f}%"

    def _num(value: float | None) -> str:
        return "     n/a" if value is None else f"{value:8.3f}"

    lines = [
        "tracegrad verification",
        "────────────────────────────────",
        f"Cohort: {label}",
        f"Sessions: {sessions}",
        "",
        "                    Baseline   Candidate",
        f"Mean score          {_num(result.baseline_mean_score)}   {_num(result.candidate_mean_score)}",
        f"Pass rate           {_pct(result.baseline_pass_rate)}   {_pct(result.candidate_pass_rate)}",
        "",
        f"Improved           {len(result.improved_sessions):8d}",
        f"Regressed          {len(result.regressed_sessions):8d}",
        f"Unchanged          {len(result.unchanged_sessions):8d}",
        f"Diverged           {len(result.diverged_sessions):8d}",
        f"Replay failures    {len(result.replay_failures):8d}",
    ]
    if result.regressed_sessions:
        lines.append("")
        lines.append("Regressions")
        for session_id in result.regressed_sessions:
            number = request.session_numbers.get(session_id)
            prefix = f"#{number}" if number is not None else session_id[:8]
            short = _short_id(session_id)
            lines.append(f"{prefix}   session {short}")
    if result.diverged_sessions:
        lines.append("")
        lines.append("Divergence")
        for item in result.diverged_sessions:
            number = item.number or request.session_numbers.get(item.session_id)
            prefix = f"#{number}" if number is not None else item.session_id[:8]
            lines.append(f"{prefix}   {item.kind}  {item.detail}".rstrip())
    if result.replay_failures:
        lines.append("")
        lines.append("Replay failures")
        for item in result.replay_failures:
            number = item.number or request.session_numbers.get(item.session_id)
            prefix = f"#{number}" if number is not None else item.session_id[:8]
            lines.append(f"{prefix}   {item.error}".rstrip())
    lines.append("")
    lines.append(f"Experiment run: {_short_id(result.experiment_run_id)}")
    if server_url:
        lines.append(f"Kitaru: {server_url}")
    verdict = "FAILED" if result.status == "failed" else "REVIEW"
    lines.append(f"Verdict: {verdict}")
    return "\n".join(lines)


def _short_id(value: str) -> str:
    text = value.removeprefix("sha256:")
    if len(text) <= 12:
        return text
    return f"{text[:4]}…{text[-4:]}"


def load_proposal_for_verify(project_root: str | Path, run_id: str) -> Proposal:
    return load_proposal(project_root, run_id)
