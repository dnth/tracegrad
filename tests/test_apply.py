from __future__ import annotations

from pathlib import Path

import pytest

from tracegrad.apply import (
    ApplyError,
    StaleProposalError,
    applied_history,
    apply_proposal,
    build_proposal,
    current_baseline,
    is_stale,
    latest_run_id,
    load_proposal,
    revert,
    review_cards,
    save_proposal,
)
from tracegrad.canonical import text_hash
from tracegrad.distill import DistillConfig, distill_trace
from tracegrad.edits import resolve_edits
from tracegrad.gates import GateFlag, GateOutcome
from tracegrad.inventory import build_inventory
from tracegrad.schema import AttributionResult, Edit, Trace

PROMPT = "Rules:\n- Be concise.\n- Cite the doc.\n"


def _template(tmp_path: Path, text: str = PROMPT) -> Path:
    target = tmp_path / "prompt.md"
    target.write_text(text, encoding="utf-8")
    return target


def _edit(instruction_id: str, text: str = "Always cite the doc.") -> Edit:
    return Edit(
        instruction_id=instruction_id,
        operation="REWRITE",
        text=text,
        covers_theme="missing-citation",
        watch_metric="missing-citation",
    )


def _outcome(prompt: str = PROMPT) -> tuple[GateOutcome, str]:
    inventory = build_inventory(prompt)
    target = inventory.instructions[-1].instruction_id
    resolution = resolve_edits(inventory, [_edit(target)])
    return GateOutcome(kept=resolution.resolved, rejected=(), tokens_before=10, tokens_after=12), target


def _distilled() -> dict[str, object]:
    trace = Trace(
        trace_id="t-1",
        input="where is it documented",
        output="It is in the manual somewhere.",
        judge={"score": 0.2, "rationale": "no citation given at all in the answer"},
        prompt_hash="sha256:abc",
    )
    return {"t-1": distill_trace(trace, DistillConfig())}


def _attributions() -> list[AttributionResult]:
    return [
        AttributionResult(
            trace_id="t-1",
            violations=[
                {
                    "instruction_id": None,
                    "theme_slug": "missing-citation",
                    "quote": "It is in the manual somewhere.",
                    "quote_source": "output",
                }
            ],
        )
    ]


def test_build_proposal_carries_diff_evidence_and_flags(tmp_path: Path) -> None:
    outcome, target = _outcome()
    outcome = GateOutcome(
        kept=outcome.kept,
        rejected=(),
        flags=(GateFlag(target, "negation-window", "never"),),
        reclassified=(target,),
    )

    proposal = build_proposal(
        run_id="run-0001",
        template_file="prompt.md",
        prompt=PROMPT,
        outcome=outcome,
        attributions=_attributions(),
        distilled=_distilled(),
    )

    assert proposal.base_prompt_hash == text_hash(PROMPT)
    assert len(proposal.edits) == 1
    card = review_cards(proposal)[0]
    assert "Always cite the doc." in card.diff
    assert card.evidence[0].quote == "It is in the manual somewhere."
    assert card.flags == ("negation-window",)
    assert card.reclassified is True
    assert "evidence:" in card.render()


def test_confabulated_evidence_never_reaches_a_review_card() -> None:
    outcome, _ = _outcome()
    attributions = [
        AttributionResult(
            trace_id="t-1",
            violations=[
                {
                    "instruction_id": None,
                    "theme_slug": "missing-citation",
                    "quote": "a sentence the model invented",
                    "quote_source": "output",
                }
            ],
        )
    ]

    proposal = build_proposal(
        run_id="run-0001",
        template_file="prompt.md",
        prompt=PROMPT,
        outcome=outcome,
        attributions=attributions,
        distilled=_distilled(),
    )

    assert proposal.edits[0].evidence == []


def test_proposal_round_trips_through_disk(tmp_path: Path) -> None:
    outcome, _ = _outcome()
    proposal = build_proposal(
        run_id="run-0001", template_file="prompt.md", prompt=PROMPT, outcome=outcome
    )

    save_proposal(tmp_path, proposal)

    assert latest_run_id(tmp_path) == "run-0001"
    assert load_proposal(tmp_path, "run-0001") == proposal


def test_loading_an_unknown_run_is_an_apply_error(tmp_path: Path) -> None:
    with pytest.raises(ApplyError):
        load_proposal(tmp_path, "run-9999")


def test_apply_writes_the_template_and_records_the_baseline(tmp_path: Path) -> None:
    template = _template(tmp_path)
    outcome, _ = _outcome()
    proposal = build_proposal(
        run_id="run-0001", template_file="prompt.md", prompt=PROMPT, outcome=outcome
    )

    result = apply_proposal(tmp_path, proposal, [0], base_directory=tmp_path)

    assert "Always cite the doc." in template.read_text(encoding="utf-8")
    assert result.applied_prompt_hash == text_hash(template.read_text(encoding="utf-8"))
    assert result.snapshot is not None and result.snapshot.exists()
    assert current_baseline(tmp_path) == result.applied_prompt_hash
    assert applied_history(tmp_path)[-1]["event"] == "applied"


def test_partial_acceptance_applies_only_the_chosen_edits(tmp_path: Path) -> None:
    _template(tmp_path)
    inventory = build_inventory(PROMPT)
    first, second = inventory.instructions[1], inventory.instructions[2]
    resolution = resolve_edits(
        inventory,
        [
            _edit(first.instruction_id, "Be brief."),
            _edit(second.instruction_id, "Always cite the doc."),
        ],
    )
    outcome = GateOutcome(kept=resolution.resolved, rejected=())
    proposal = build_proposal(
        run_id="run-0001", template_file="prompt.md", prompt=PROMPT, outcome=outcome
    )

    result = apply_proposal(tmp_path, proposal, [1], base_directory=tmp_path)

    written = (tmp_path / "prompt.md").read_text(encoding="utf-8")
    assert "Always cite the doc." in written
    assert "Be concise." in written
    assert len(result.accepted) == 1
    assert len(result.rejected) == 1


def test_accepting_nothing_leaves_the_template_untouched(tmp_path: Path) -> None:
    template = _template(tmp_path)
    outcome, _ = _outcome()
    proposal = build_proposal(
        run_id="run-0001", template_file="prompt.md", prompt=PROMPT, outcome=outcome
    )

    result = apply_proposal(tmp_path, proposal, [], base_directory=tmp_path)

    assert result.unchanged is True
    assert template.read_text(encoding="utf-8") == PROMPT
    assert applied_history(tmp_path) == []


def test_an_out_of_band_edit_makes_the_proposal_stale(tmp_path: Path) -> None:
    template = _template(tmp_path)
    outcome, _ = _outcome()
    proposal = build_proposal(
        run_id="run-0001", template_file="prompt.md", prompt=PROMPT, outcome=outcome
    )
    template.write_text(PROMPT + "- Someone edited this by hand.\n", encoding="utf-8")

    assert is_stale(proposal, base_directory=tmp_path) is True
    with pytest.raises(StaleProposalError):
        apply_proposal(tmp_path, proposal, [0], base_directory=tmp_path)


def test_force_overrides_the_stale_check(tmp_path: Path) -> None:
    template = _template(tmp_path)
    outcome, _ = _outcome()
    proposal = build_proposal(
        run_id="run-0001", template_file="prompt.md", prompt=PROMPT, outcome=outcome
    )
    template.write_text(PROMPT + "- Hand edited.\n", encoding="utf-8")

    result = apply_proposal(tmp_path, proposal, [0], base_directory=tmp_path, force=True)

    assert result.unchanged is False


def test_an_unknown_edit_index_is_refused(tmp_path: Path) -> None:
    _template(tmp_path)
    outcome, _ = _outcome()
    proposal = build_proposal(
        run_id="run-0001", template_file="prompt.md", prompt=PROMPT, outcome=outcome
    )

    with pytest.raises(ApplyError):
        apply_proposal(tmp_path, proposal, [7], base_directory=tmp_path)


def test_revert_restores_the_snapshot_byte_for_byte(tmp_path: Path) -> None:
    template = _template(tmp_path)
    outcome, _ = _outcome()
    proposal = build_proposal(
        run_id="run-0001", template_file="prompt.md", prompt=PROMPT, outcome=outcome
    )
    apply_proposal(tmp_path, proposal, [0], base_directory=tmp_path)
    assert template.read_text(encoding="utf-8") != PROMPT

    revert(tmp_path, "run-0001", base_directory=tmp_path)

    assert template.read_text(encoding="utf-8") == PROMPT
    assert current_baseline(tmp_path) == text_hash(PROMPT)


def test_reverting_a_run_that_never_applied_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ApplyError):
        revert(tmp_path, "run-0001", base_directory=tmp_path)
