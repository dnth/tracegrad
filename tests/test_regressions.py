"""Regressions from the code review, each pinned to the behaviour it broke.

Every test here failed before its fix.  The comments say what went wrong rather
than what the code does, because the failure mode is the part worth keeping.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from tracegrad import cli
from tracegrad.aggregate import ThemeHistory, ThemeStat
from tracegrad.apply import build_proposal, review_cards
from tracegrad.attribute import build_instrument
from tracegrad.distill import DistillConfig, distill_trace
from tracegrad.edits import apply_edits, resolve_edits
from tracegrad.gates import GateOutcome, gate_accounting, gate_budget, run_gates
from tracegrad.ingest import ingest_traces
from tracegrad.inventory import build_inventory, segment
from tracegrad.llm import FakeBackend
from tracegrad.schema import AttributionResult, Cluster, Edit, Trace
from tracegrad.trends import Proportion, compare, detectable_effect

BULLETS = "- One two three four five six seven eight nine ten.\n- Redundant filler here.\n"


def _edit(instruction_id: str, operation: str, text: str, theme: str = "t") -> Edit:
    return Edit(
        instruction_id=instruction_id,
        operation=operation,
        text=text,
        covers_theme=theme,
        watch_metric=theme,
    )


def _trace(trace_id: str = "t-1", output: str = "an output") -> Trace:
    return Trace(
        trace_id=trace_id,
        input="a question worth asking",
        output=output,
        judge={"score": 0.3, "rationale": "a rationale long enough to be usable"},
        prompt_hash="sha256:abc",
    )


def test_an_over_ceiling_prompt_can_still_accept_a_paid_for_addition() -> None:
    # The budget rule at the ceiling is zero-sum, not zero-additions. The old
    # limit was min(ceiling, before), which is just `ceiling`, so anyone already
    # over their ceiling could never accept an addition again — not even one
    # paired with a deletion that made the set net-negative.
    inventory = build_inventory(BULLETS)
    resolution = resolve_edits(
        inventory,
        [
            _edit("END", "ADD", "Be brief."),
            _edit(inventory.instructions[1].instruction_id, "DELETE", ""),
        ],
    )

    kept, rejected, before, after = gate_budget(BULLETS, resolution.resolved, ceiling=5)

    assert len(kept) == 2
    assert rejected == ()
    assert after < before


def test_a_lone_addition_still_cannot_grow_an_over_ceiling_prompt() -> None:
    inventory = build_inventory(BULLETS)
    resolution = resolve_edits(inventory, [_edit("END", "ADD", "Be brief.")])

    kept, rejected, before, after = gate_budget(BULLETS, resolution.resolved, ceiling=5)

    assert kept == ()
    assert rejected and after == before


def test_deleting_one_of_two_identical_instructions_is_possible() -> None:
    # G2 tested for presence of the deleted text anywhere in the result, but
    # duplicate instruction text is supported on purpose, so de-duplicating a
    # repeated instruction was impossible.
    prompt = "- Cite sources.\n- Be brief.\n- Cite sources.\n"
    inventory = build_inventory(prompt)
    resolution = resolve_edits(
        inventory, [_edit(inventory.instructions[0].instruction_id, "DELETE", "")]
    )

    kept, rejected = gate_accounting(prompt, resolution.resolved)

    assert len(kept) == 1
    assert rejected == ()


def test_a_rewrite_that_tightens_while_adding_a_clause_survives() -> None:
    # G3 reclassified this to ADD and G2 then rejected it as "ADD removed
    # content" — so the commonest real edit shape could never be proposed.
    prompt = "- Always cite your sources whenever you make any factual claim in your text.\n"
    inventory = build_inventory(prompt)
    resolution = resolve_edits(
        inventory,
        [
            _edit(
                inventory.instructions[0].instruction_id,
                "REWRITE",
                "Cite sources. Never speculate.",
            )
        ],
    )
    attributions = [
        AttributionResult(
            trace_id="t-1",
            violations=[
                {
                    "instruction_id": None,
                    "theme_slug": "t",
                    "quote": "an output",
                    "quote_source": "output",
                }
            ],
        )
    ]
    distilled = {"t-1": distill_trace(_trace(), DistillConfig())}

    outcome = run_gates(
        resolution, inventory, attributions=attributions, distilled=distilled
    )

    assert len(outcome.kept) == 1
    assert "drops-most-of-the-instruction" in [flag.flag for flag in outcome.flags]


def test_two_additions_after_one_anchor_are_both_applied() -> None:
    # Adding two sibling bullets under one instruction is a proposal, not a
    # conflict; one of them used to be silently dropped as an overlap.
    prompt = "Rules:\n- Be concise.\n"
    inventory = build_inventory(prompt)
    anchor = inventory.instructions[1].instruction_id

    updated, resolution = apply_edits(
        inventory,
        [_edit(anchor, "ADD", "Cite the doc."), _edit(anchor, "ADD", "Never guess.")],
    )

    assert resolution.rejected == ()
    assert "Cite the doc." in updated
    assert "Never guess." in updated


def test_ingest_accepts_a_generator_of_traces() -> None:
    # `all(...)` consumed the iterator, so a generator ingested as zero traces
    # with zero named drop reasons — a silent empty batch.
    result = ingest_traces(trace for trace in [_trace()])

    assert result.accepted_count == 1
    assert result.dropped == ()


def test_a_run_id_alone_is_not_new_evidence_for_the_memory_gate(tmp_path: Path) -> None:
    # Run ids are distinct by construction, so counting them cleared the
    # re-proposal bar every run with identical evidence behind it.
    history = ThemeHistory(tmp_path / "themes.jsonl")
    themes = (ThemeStat(theme="x", numerator=3, denominator=10),)

    history.record(themes, run_id="run-0001")
    history.record(themes, run_id="run-0002")

    assert history.distinct_sources() == {}


def test_a_review_card_never_shows_an_unverifiable_quote() -> None:
    # The gates treat a missing distilled record as unverified; the card a human
    # reads was the one place that trusted it.
    inventory = build_inventory("Rules:\n- Be concise.\n")
    resolution = resolve_edits(
        inventory, [_edit(inventory.instructions[1].instruction_id, "REWRITE", "Be brief.")]
    )
    attributions = [
        AttributionResult(
            trace_id="missing-from-the-store",
            violations=[
                {
                    "instruction_id": None,
                    "theme_slug": "t",
                    "quote": "a quote nobody can check",
                    "quote_source": "output",
                }
            ],
        )
    ]

    proposal = build_proposal(
        run_id="run-0001",
        template_file="prompt.md",
        prompt=inventory.prompt,
        outcome=GateOutcome(kept=resolution.resolved, rejected=()),
        attributions=attributions,
        distilled={},
    )

    assert proposal.edits[0].evidence == []
    assert "none survived verification" in review_cards(proposal)[0].render()


def test_apply_with_no_decision_does_not_record_rejections(tmp_path: Path) -> None:
    # A piped invocation with no --accept/--all recorded every proposed edit as
    # human-rejected, permanently poisoning G6 for that proposal.
    template = tmp_path / "prompt.md"
    template.write_text("Rules:\n- Be concise.\n", encoding="utf-8")
    inventory = build_inventory(template.read_text(encoding="utf-8"))
    resolution = resolve_edits(
        inventory, [_edit(inventory.instructions[1].instruction_id, "REWRITE", "Be brief.")]
    )
    from tracegrad.apply import save_proposal

    save_proposal(
        tmp_path,
        build_proposal(
            run_id="run-0001",
            template_file="prompt.md",
            prompt=inventory.prompt,
            outcome=GateOutcome(kept=resolution.resolved, rejected=()),
        ),
    )

    stream = io.StringIO()
    code = cli.main(
        ["apply", "--project-root", str(tmp_path), "--base-directory", str(tmp_path)],
        out=stream,
    )

    assert code == 1
    assert not (tmp_path / ".tracegrad" / "ledgers" / "rejections.jsonl").exists()


def test_apply_with_an_explicitly_empty_selection_does_record_the_rejection(
    tmp_path: Path,
) -> None:
    # The other half: choosing nothing IS a decision, and must be remembered.
    template = tmp_path / "prompt.md"
    template.write_text("Rules:\n- Be concise.\n", encoding="utf-8")
    inventory = build_inventory(template.read_text(encoding="utf-8"))
    resolution = resolve_edits(
        inventory, [_edit(inventory.instructions[1].instruction_id, "REWRITE", "Be brief.")]
    )
    from tracegrad.apply import save_proposal

    save_proposal(
        tmp_path,
        build_proposal(
            run_id="run-0001",
            template_file="prompt.md",
            prompt=inventory.prompt,
            outcome=GateOutcome(kept=resolution.resolved, rejected=()),
        ),
    )

    cli.main(
        [
            "apply",
            "--accept",
            "",
            "--project-root",
            str(tmp_path),
            "--base-directory",
            str(tmp_path),
        ],
        out=io.StringIO(),
    )

    assert (tmp_path / ".tracegrad" / "ledgers" / "rejections.jsonl").is_file()


def test_revert_refuses_to_discard_edits_made_after_the_apply(tmp_path: Path) -> None:
    # revert overwrote the template with no staleness check, destroying any
    # manual edit made since the apply, unrecoverably.
    from tracegrad.apply import StaleProposalError, apply_proposal, revert, save_proposal

    template = tmp_path / "prompt.md"
    original = "Rules:\n- Be concise.\n"
    template.write_text(original, encoding="utf-8")
    inventory = build_inventory(original)
    resolution = resolve_edits(
        inventory, [_edit(inventory.instructions[1].instruction_id, "REWRITE", "Be brief.")]
    )
    proposal = build_proposal(
        run_id="run-0001",
        template_file="prompt.md",
        prompt=original,
        outcome=GateOutcome(kept=resolution.resolved, rejected=()),
    )
    save_proposal(tmp_path, proposal)
    apply_proposal(tmp_path, proposal, [0], base_directory=tmp_path)

    template.write_text(
        template.read_text(encoding="utf-8") + "- Hand written since.\n", encoding="utf-8"
    )

    with pytest.raises(StaleProposalError):
        revert(tmp_path, "run-0001", base_directory=tmp_path)
    assert "Hand written since." in template.read_text(encoding="utf-8")

    revert(tmp_path, "run-0001", base_directory=tmp_path, force=True)
    assert template.read_text(encoding="utf-8") == original


def test_a_revert_is_remembered_by_the_memory_gate(tmp_path: Path) -> None:
    from tracegrad.apply import apply_proposal, revert, save_proposal
    from tracegrad.gates import REJECTION_MEMORY_FILENAME, RejectionMemory

    template = tmp_path / "prompt.md"
    template.write_text("Rules:\n- Be concise.\n", encoding="utf-8")
    inventory = build_inventory(template.read_text(encoding="utf-8"))
    resolution = resolve_edits(
        inventory, [_edit(inventory.instructions[1].instruction_id, "REWRITE", "Be brief.")]
    )
    proposal = build_proposal(
        run_id="run-0001",
        template_file="prompt.md",
        prompt=inventory.prompt,
        outcome=GateOutcome(kept=resolution.resolved, rejected=()),
    )
    save_proposal(tmp_path, proposal)
    apply_proposal(tmp_path, proposal, [0], base_directory=tmp_path)
    revert(tmp_path, "run-0001", base_directory=tmp_path)

    memory = RejectionMemory(
        (tmp_path / ".tracegrad" / "ledgers" / REJECTION_MEMORY_FILENAME)
    )
    assert sum(memory.counts().values()) >= 1


def test_a_continuation_line_after_prose_does_not_swallow_it() -> None:
    # The bullet span grew past intervening prose and the paragraph then began
    # inside it — overlapping, out-of-order spans that corrupt every offset.
    prompt = "- a bullet\nunindented prose line\n  indented continuation\n"

    spans = [(start, end) for _, start, end in segment(prompt)]

    assert all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))


def test_every_instruction_span_indexes_back_into_the_prompt() -> None:
    prompt = "Rules:\n- one\n  more of one\n\nProse here. And more.\n"

    inventory = build_inventory(prompt)

    for instruction in inventory.instructions:
        assert prompt[instruction.start : instruction.end] == instruction.text


def test_a_theme_new_in_the_second_batch_can_trip_the_guardrail() -> None:
    # before_size fell back to 0, so a brand-new regression short-circuited to
    # p=1.0 and the guardrail could never fire on it.
    report = compare(
        [Cluster(theme="tone", numerator=10, denominator=200)],
        [
            Cluster(theme="tone", numerator=2, denominator=200),
            Cluster(theme="brand-new-failure", numerator=60, denominator=200),
        ],
        targeted=["tone"],
    )

    assert "brand-new-failure" in report.regressed_elsewhere


def test_detectable_effect_uses_the_power_it_is_given() -> None:
    before = Proportion(20, 200)
    after = Proportion(20, 200)

    assert detectable_effect(before, after, power=0.95) > detectable_effect(
        before, after, power=0.8
    )


def test_the_blinded_health_sample_is_skipped_on_a_fully_cached_run(
    tmp_path: Path,
) -> None:
    # The sample is uncached by design, so a re-run that hit the cache for every
    # trace still paid for five model calls to re-measure an unchanged number.
    from tracegrad.attribute import attribute_batch

    inventory = build_inventory("- Be concise.\n")
    distilled = [distill_trace(_trace(f"t-{n}"), DistillConfig()) for n in range(3)]

    def respond(system: str, user: str) -> str:
        return '{"violations": [], "harmful": []}'

    attribute_batch(
        distilled, inventory, FakeBackend(handler=respond), project_root=tmp_path
    )
    second = FakeBackend(handler=respond)
    run = attribute_batch(distilled, inventory, second, project_root=tmp_path)

    assert run.cache_hits == 3
    assert second.calls == []


def test_a_secret_straddling_the_truncation_boundary_is_still_redacted() -> None:
    # Redaction ran after truncation, so a cut secret left an unmatched
    # fragment in the text sent to the model.
    email = "someone.longname@corporate.example"
    padding = "x" * 90
    trace = _trace(output=f"{padding} {email} {padding}")

    distilled = distill_trace(trace, DistillConfig(max_output_chars=100))

    assert "corporate.example" not in distilled.output
    assert "someone.longname" not in distilled.output


def test_the_instrument_is_stable_across_equal_inputs() -> None:
    inventory = build_inventory("- Be concise.\n")
    backend = FakeBackend(responses=["{}"])

    first = build_instrument(backend, inventory, "config-hash")
    second = build_instrument(backend, inventory, "config-hash")

    assert first.fingerprint == second.fingerprint
    assert first.measurement_fingerprint == second.measurement_fingerprint


def test_the_measurement_fingerprint_ignores_the_prompt_it_measured() -> None:
    # Applying an edit changes the prompt on purpose. If that changed the
    # comparability fingerprint, trends would be suppressed forever after the
    # first accepted edit — the exact comparison the tool exists for.
    backend = FakeBackend(responses=["{}"])
    before = build_instrument(backend, build_inventory("- Be concise.\n"), "config-hash")
    after = build_instrument(backend, build_inventory("- Be very concise.\n"), "config-hash")

    assert before.measurement_fingerprint == after.measurement_fingerprint
    assert before.fingerprint != after.fingerprint
