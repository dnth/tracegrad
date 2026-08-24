"""End-to-end: the example batch through analysis, review, apply, and revert.

These tests drive the real pipeline with scripted backends, so they cover the
wiring the unit tests deliberately leave alone: stage order, the attribution
cache surviving a killed run, partial acceptance, and byte-exact output.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from tracegrad.apply import apply_proposal, load_proposal, revert, review_cards
from tracegrad.attribute import CoverageError
from tracegrad.llm import FakeBackend, LLMError
from tracegrad.pipeline import estimate_run, run_pipeline
from tracegrad.state import load_resume_state

EXAMPLE = Path(__file__).resolve().parents[1] / "example"


def _output_of(user: str) -> str:
    match = re.search(r"OUTPUT:\n(.*?)\n\nJUDGE SCORE", user, re.S)
    return match.group(1) if match else ""


def _instruction_ids(system: str) -> list[tuple[str, str]]:
    return re.findall(r"^(i-[0-9a-f]{12}-\d\d): (.*)$", system, re.M)


def _attribution_response(system: str, user: str) -> str:
    """Attribute from the judge rationale, quoting verbatim from the output."""

    rationale = user.split("JUDGE RATIONALE:\n")[-1].lower()
    output = _output_of(user)
    quote = output.split(". ")[0][:60]
    catalogue = _instruction_ids(system)

    def anchor(fragment: str) -> str | None:
        return next((item for item, text in catalogue if fragment in text), None)

    violations = []
    if "no help-centre article" in rationale or "no article cited" in rationale:
        violations.append(("missing-citation", anchor("Cite the help-centre")))
    if "refund" in rationale and "promis" in rationale:
        violations.append(("refund-promised", anchor("Never promise a refund")))
    if "jargon" in rationale:
        violations.append(("jargon-tone", anchor("warm, plain tone")))
    if "sentence" in rationale and "over" in rationale:
        violations.append(("over-length", anchor("at most three sentences")))

    return json.dumps(
        {
            "violations": [
                {"instruction_id": anchor_id, "theme_slug": theme, "quote": quote}
                for theme, anchor_id in violations
                if quote
            ],
            "harmful": [],
        }
    )


def _synthesis_response(system: str, user: str) -> str:
    catalogue = dict(
        re.findall(r"^(i-[0-9a-f]{12}-\d\d): (.*?)(?:\s+\[non-editable.*)?$", user, re.M)
    )
    target = next(
        (item for item, text in catalogue.items() if "Cite the help-centre" in text), None
    )
    return json.dumps(
        {
            "edits": [
                {
                    "instruction_id": target,
                    "operation": "REWRITE",
                    "text": "Cite the help-centre article you used, by its title, in every answer.",
                    "covers_theme": "missing-citation",
                    "watch_metric": "missing-citation",
                }
            ],
            "reasoning": "missing citations are the largest theme in the batch",
        }
    )


def _project(tmp_path: Path) -> Path:
    shutil.copytree(EXAMPLE, tmp_path / "example")
    return tmp_path


def _run(project: Path, run_id: str = "run-0001", **overrides: object):
    return run_pipeline(
        project / "example" / "batch.jsonl",
        project / "example" / "manifest.json",
        run_id=run_id,
        project_root=project,
        base_directory=project / "example",
        attribution_backend=overrides.get(
            "attribution", FakeBackend(handler=_attribution_response, name="fake-attribution")
        ),
        synthesis_backend=overrides.get(
            "synthesis", FakeBackend(handler=_synthesis_response, name="fake-synthesis")
        ),
        session_id=str(overrides.get("session", "session-1")),
    )


def test_estimate_contacts_no_model(tmp_path: Path) -> None:
    project = _project(tmp_path)

    estimate = estimate_run(
        project / "example" / "batch.jsonl",
        project / "example" / "manifest.json",
        project_root=project,
        base_directory=project / "example",
    )

    assert estimate.traces == 12
    assert estimate.attribution_calls == 12
    assert estimate.synthesis_calls == 1
    assert estimate.prompt_tokens > 0
    assert "12 traces" in estimate.render()


def test_the_example_batch_produces_a_report_and_a_proposal(tmp_path: Path) -> None:
    project = _project(tmp_path)

    result = _run(project)

    # One trace in the fixture has a rationale below the quality floor.
    assert len(result.ingest.traces) == 12
    assert [drop.reason for drop in result.ingest.dropped] == [
        "rationale-below-quality-floor"
    ]
    assert result.attribution is not None and result.attribution.coverage == 1.0

    themes = {theme.theme: theme for theme in result.aggregation.themes}
    assert themes["missing-citation"].numerator == 7
    assert themes["missing-citation"].denominator == 12
    assert not themes["missing-citation"].is_gap

    assert result.proposal is not None
    assert len(result.proposal.edits) == 1
    card = review_cards(result.proposal)[0]
    assert card.edit.covers_theme == "missing-citation"
    assert card.evidence, "an edit must reach review with verified evidence"

    assert (project / ".tracegrad" / "reports" / "run-0001.json").is_file()
    assert load_resume_state(project, "run-0001") == {
        "run_id": "run-0001",
        "stage": "complete",
    }


def test_a_killed_run_resumes_without_paying_for_attribution_twice(tmp_path: Path) -> None:
    project = _project(tmp_path)
    calls = {"count": 0}

    def dies_after_five(system: str, user: str) -> str:
        calls["count"] += 1
        if calls["count"] > 5:
            raise LLMError("connection lost")
        return _attribution_response(system, user)

    with pytest.raises(CoverageError):
        _run(project, attribution=FakeBackend(handler=dies_after_five))

    survivor = FakeBackend(handler=_attribution_response)
    result = _run(project, attribution=survivor)

    # The five attributions paid for before the kill are served from the cache;
    # only the remaining seven traces reach the backend again.  (The blinded
    # health sample also calls the backend, so count attribution calls by shape.)
    attribution_calls = [
        call for call in survivor.calls if call[0].startswith("SYSTEM PROMPT INSTRUCTIONS:")
    ]
    assert result.attribution is not None
    assert result.attribution.cache_hits == 5
    assert len(attribution_calls) == 7
    assert result.proposal is not None and len(result.proposal.edits) == 1


def test_partial_acceptance_writes_a_byte_exact_prompt(tmp_path: Path) -> None:
    project = _project(tmp_path)
    template = project / "example" / "prompt.md"
    original = template.read_text(encoding="utf-8")

    _run(project)
    proposal = load_proposal(project, "run-0001")
    outcome = apply_proposal(
        project, proposal, [0], base_directory=project / "example"
    )

    expected = original.replace(
        "- Cite the help-centre article you used, by its title.",
        "- Cite the help-centre article you used, by its title, in every answer.",
    )
    assert template.read_text(encoding="utf-8") == expected
    assert outcome.applied_prompt_hash != proposal.base_prompt_hash

    revert(project, "run-0001", base_directory=project / "example")
    assert template.read_text(encoding="utf-8") == original


def test_a_second_batch_reports_trends_against_the_first(tmp_path: Path) -> None:
    project = _project(tmp_path)
    _run(project, run_id="run-0001")

    # A real second batch is new traces, not the same ones again — the same
    # traces would be served from the attribution cache and measure nothing.
    batch = project / "example" / "batch.jsonl"
    follow_up = []
    for line in batch.read_text(encoding="utf-8").splitlines():
        trace = json.loads(line)
        trace["trace_id"] = f"n-{trace['trace_id']}"
        trace["judge"] = {
            "score": 1.0,
            "rationale": "Concise, warm, and cites the help-centre article by title.",
        }
        follow_up.append(json.dumps(trace))
    batch.write_text("\n".join(follow_up) + "\n", encoding="utf-8")

    result = _run(
        project,
        run_id="run-0002",
        attribution=FakeBackend(handler=_attribution_response, name="fake-attribution"),
        session="session-2",
    )

    assert result.trends is not None
    citation = result.trends.by_theme("missing-citation")
    assert citation is not None
    assert citation.before.numerator == 7
    assert citation.after.numerator == 0
    assert citation.detectable_effect > 0


def test_a_run_that_proposes_nothing_is_a_valid_outcome(tmp_path: Path) -> None:
    project = _project(tmp_path)

    result = _run(
        project,
        synthesis=FakeBackend(handler=lambda system, user: json.dumps({"edits": []})),
    )

    assert result.proposal is not None
    assert result.proposal.edits == []
    assert result.synthesis is not None and result.synthesis.proposed_nothing
