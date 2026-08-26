"""Pipeline wiring: the parts that only exist once the stages are connected.

Each of these covers a gap where a correct library function was reachable only
through the orchestrator — a gate fed the wrong number, a ledger with no caller.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from test_e2e import _attribution_response, _synthesis_response

from tracegrad.aggregate import GAP_LEDGER_FILENAME, GAP_RETIRED, THEME_HISTORY_FILENAME, GapLedger
from tracegrad.apply import apply_proposal, load_proposal, revert
from tracegrad.gates import REJECTION_MEMORY_FILENAME, RejectionMemory
from tracegrad.ingest import ingest_traces
from tracegrad.llm import FakeBackend
from tracegrad.pipeline import load_manifest, run_pipeline, verdict_history
from tracegrad.state import initialize, load_jsonl

EXAMPLE = Path(__file__).resolve().parents[1] / "example"


def _project(tmp_path: Path) -> Path:
    shutil.copytree(EXAMPLE, tmp_path / "example")
    return tmp_path


def _run(project: Path, run_id: str = "run-0001", session: str = "session-1"):
    return run_pipeline(
        project / "example" / "batch.jsonl",
        project / "example" / "manifest.json",
        run_id=run_id,
        project_root=project,
        base_directory=project / "example",
        attribution_backend=FakeBackend(handler=_attribution_response, name="fake-attribution"),
        synthesis_backend=FakeBackend(handler=_synthesis_response, name="fake-synthesis"),
        session_id=session,
    )


def test_theme_history_is_recorded_for_the_memory_gate(tmp_path: Path) -> None:
    project = _project(tmp_path)

    _run(project)

    records = load_jsonl(initialize(project).ledgers / THEME_HISTORY_FILENAME)
    assert records, "the memory gate has nothing to read without this ledger"
    assert {record["session_id"] for record in records} == {"session-1"}


def test_a_rejected_edit_is_not_re_proposed_from_one_session(tmp_path: Path) -> None:
    # The whole point of G6: one batch showing the same theme in many traces is
    # one observation, so a human's rejection stands.
    project = _project(tmp_path)
    first = _run(project)
    assert first.proposal is not None and first.proposal.edits

    memory = RejectionMemory(initialize(project).ledgers / REJECTION_MEMORY_FILENAME)
    for edit in first.proposal.edits:
        memory.record_rejection(edit.edit, run_id="run-0001")

    second = _run(project, run_id="run-0002", session="session-1")

    assert second.proposal is not None
    assert second.proposal.edits == []
    reasons = {entry["reason"] for entry in second.proposal.rejected}
    assert "G6-remembered-rejection" in reasons


def test_the_same_failure_in_a_second_session_clears_the_bar(tmp_path: Path) -> None:
    project = _project(tmp_path)
    first = _run(project)
    assert first.proposal is not None
    memory = RejectionMemory(initialize(project).ledgers / REJECTION_MEMORY_FILENAME)
    for edit in first.proposal.edits:
        memory.record_rejection(edit.edit, run_id="run-0001")

    second = _run(project, run_id="run-0002", session="session-2")

    assert second.proposal is not None
    assert second.proposal.edits, "a second independent session is new evidence"


def test_a_changed_instrument_suppresses_the_trend_comparison(tmp_path: Path) -> None:
    project = _project(tmp_path)
    _run(project)

    result = run_pipeline(
        project / "example" / "batch.jsonl",
        project / "example" / "manifest.json",
        run_id="run-0002",
        project_root=project,
        base_directory=project / "example",
        # A different model is a different measuring instrument.
        attribution_backend=FakeBackend(handler=_attribution_response, name="other-model"),
        synthesis_backend=FakeBackend(handler=_synthesis_response, name="fake-synthesis"),
        session_id="session-2",
    )

    assert result.trends is None
    assert any("instrument changed" in warning for warning in result.warnings)


def test_verdicts_are_recorded_so_hysteresis_has_a_history(tmp_path: Path) -> None:
    project = _project(tmp_path)
    _run(project)

    batch = project / "example" / "batch.jsonl"
    renamed = []
    for line in batch.read_text(encoding="utf-8").splitlines():
        trace = json.loads(line)
        trace["trace_id"] = f"n-{trace['trace_id']}"
        renamed.append(json.dumps(trace))
    batch.write_text("\n".join(renamed) + "\n", encoding="utf-8")

    _run(project, run_id="run-0002", session="session-2")

    assert verdict_history(project), "no history means hysteresis can never fire"


def test_reverting_an_applied_run_restores_its_gap_themes(tmp_path: Path) -> None:
    project = _project(tmp_path)
    result = _run(project)
    assert result.proposal is not None and result.proposal.edits

    theme = result.proposal.edits[0].edit.covers_theme
    gaps = GapLedger(initialize(project).ledgers / GAP_LEDGER_FILENAME)
    gaps.record_observation(theme, run_id="run-0001", session_id="session-1")
    gaps.retire(theme, run_id="run-0001", reason="improved-trend")
    assert gaps.state()[theme].status == GAP_RETIRED

    proposal = load_proposal(project, "run-0001")
    apply_proposal(project, proposal, [0], base_directory=project / "example")
    revert(project, "run-0001", base_directory=project / "example")

    # The edit that justified retiring the theme is gone, so the theme is open.
    assert gaps.state()[theme].status != GAP_RETIRED


def test_manifest_canary_scores_reach_ingest(tmp_path: Path) -> None:
    project = _project(tmp_path)
    manifest_path = project / "example" / "manifest.json"
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_data["canary_scores"] = {"t-003": 0.0}
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
    manifest = load_manifest(manifest_path)

    ingested = ingest_traces(
        project / "example" / "batch.jsonl", manifest, canary_scores=manifest.canary_scores
    )

    # t-003 is scored 1.0 in the fixture; a canary expecting 0.0 must be flagged.
    assert [failure.trace_id for failure in ingested.canary_failures] == ["t-003"]


def test_a_drifting_judge_canary_warns_the_run(tmp_path: Path) -> None:
    project = _project(tmp_path)
    manifest_path = project / "example" / "manifest.json"
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_data["canary_scores"] = {"t-003": 0.0}
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

    result = _run(project)

    assert any("canary" in warning for warning in result.warnings)


def test_jobs_does_not_change_what_a_run_produces(tmp_path: Path) -> None:
    sequential_project = _project(tmp_path / "sequential")
    concurrent_project = _project(tmp_path / "concurrent")

    sequential = _run(sequential_project)
    concurrent = run_pipeline(
        concurrent_project / "example" / "batch.jsonl",
        concurrent_project / "example" / "manifest.json",
        run_id="run-0001",
        project_root=concurrent_project,
        base_directory=concurrent_project / "example",
        attribution_backend=FakeBackend(handler=_attribution_response, name="fake-attribution"),
        synthesis_backend=FakeBackend(handler=_synthesis_response, name="fake-synthesis"),
        session_id="session-1",
        jobs=4,
    )

    assert [theme.theme for theme in concurrent.aggregation.themes] == [
        theme.theme for theme in sequential.aggregation.themes
    ]
    assert [theme.numerator for theme in concurrent.aggregation.themes] == [
        theme.numerator for theme in sequential.aggregation.themes
    ]


def test_the_rc_is_read_from_the_project_root_not_the_base_directory(tmp_path: Path) -> None:
    # The rc is documented at the project root; base_directory is a subdirectory.
    # If the config lookup regresses to base_directory, neverDelete never loads
    # and this DELETE sails through the gates.
    project = _project(tmp_path)
    (project / ".tracegradrc").write_text('neverDelete = ["help-centre"]\n', encoding="utf-8")

    def delete_synthesis(system: str, user: str) -> str:
        catalogue = dict(
            re.findall(r"^(i-[0-9a-f]{12}-\d\d): (.*?)(?:\s+\[non-editable.*)?$", user, re.M)
        )
        target = next(item for item, text in catalogue.items() if "Cite the help-centre" in text)
        return json.dumps(
            {
                "edits": [
                    {
                        "instruction_id": target,
                        "operation": "DELETE",
                        "text": "",
                        "covers_theme": "missing-citation",
                        "watch_metric": "missing-citation",
                    }
                ],
                "reasoning": "drop the citation instruction entirely",
            }
        )

    result = run_pipeline(
        project / "example" / "batch.jsonl",
        project / "example" / "manifest.json",
        run_id="run-0001",
        project_root=project,
        base_directory=project / "example",
        attribution_backend=FakeBackend(handler=_attribution_response, name="fake-attribution"),
        synthesis_backend=FakeBackend(handler=delete_synthesis, name="fake-synthesis"),
        session_id="session-1",
    )

    assert result.proposal is not None
    assert result.proposal.edits == []
    assert any(
        entry["reason"] == "G7-variable-span" and "neverDelete" in entry["detail"]
        for entry in result.proposal.rejected
    )
