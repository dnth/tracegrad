"""Synthesis — the second and last module allowed to call a model.

Synthesis is deliberately thin.  It assembles a prompt from evidence the
deterministic core already computed, asks for edits, and hands the answer
straight to the gates.  It has no opinion the gates cannot check.

When a gate drops a proposal, synthesis re-prompts once or twice with the named
reason, because a model that is told "your quote did not verify" often fixes it.
After that the proposal is written to an autopsy file rather than retried
forever — a proposal that cannot survive three attempts is evidence about the
batch, not a prompt-engineering problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from pydantic import ValidationError

from .aggregate import Aggregation, ThemeStat
from .config import TracegradConfig
from .distill import DistilledTrace
from .edits import resolve_edits
from .gates import GateConfig, GateOutcome, RejectionMemory, run_gates
from .inventory import Inventory
from .llm import SYNTHESIS_TIER, Backend, LLMError, parse_json_response, resolve_backend
from .schema import AttributionResult, Edit
from .state import atomic_write_json, initialize

SYNTHESIS_PROMPT_VERSION = 1
MAX_REPROMPTS = 2
DEFAULT_EVIDENCE_QUOTES = 3

SYSTEM_PROMPT = """\
You propose edits to a system prompt, from measured failure evidence.

You are given the numbered instructions of the current system prompt and a table \
of failure themes with counts over the whole batch.

Return JSON only, matching:
{"edits": [{"instruction_id": "<id | START | END>", "operation": "ADD|REWRITE|DELETE",
            "text": "<the new text, empty for DELETE>", "covers_theme": "<theme slug>",
            "watch_metric": "<the theme slug to watch next batch>"}],
 "reasoning": "<one short paragraph>"}

Rules:
- Propose at most five edits. Fewer is better. An empty list is a valid and \
common answer: propose nothing when the evidence does not support a change.
- Every edit must name the theme it covers, and that theme must appear in the \
evidence table.
- REWRITE replaces one instruction. ADD inserts a sibling after the anchor \
instruction; use START or END to add at the edges. DELETE removes one instruction.
- Do not restate an instruction that already exists. Do not edit instructions \
marked non-editable.
- Prefer rewriting an existing instruction over adding a new one. The prompt is \
under a token budget and additions must earn their place."""


def resolve_synthesis_backend(
    config: TracegradConfig,
    *,
    override: Backend | None = None,
    on_fallback: Callable[[str], None] | None = None,
) -> Backend:
    """Resolve the synthesis-tier backend, keeping ``llm`` behind this module."""

    return resolve_backend(
        config, SYNTHESIS_TIER, override=override, on_fallback=on_fallback
    )


class SynthesisError(RuntimeError):
    """Synthesis could not produce a usable proposal set."""


@dataclass(frozen=True)
class SynthesisResult:
    """What synthesis produced, and what the gates did to it."""

    outcome: GateOutcome
    proposed: tuple[Edit, ...] = ()
    rounds: int = 1
    reasoning: str = ""
    autopsy_path: Path | None = None

    @property
    def edits(self) -> tuple[Edit, ...]:
        return self.outcome.accepted_edits

    @property
    def proposed_nothing(self) -> bool:
        return not self.proposed

    @property
    def new_prompt_required(self) -> bool:
        return bool(self.outcome.kept)


def render_instructions(inventory: Inventory) -> str:
    lines = []
    for instruction in inventory.instructions:
        suffix = "" if instruction.editable else "  [non-editable: variable]"
        lines.append(f"{instruction.instruction_id}: {instruction.text.strip()}{suffix}")
    return "\n".join(lines)


def render_evidence(
    aggregation: Aggregation,
    attributions: Sequence[AttributionResult],
    distilled: Mapping[str, DistilledTrace],
    *,
    quotes_per_theme: int = DEFAULT_EVIDENCE_QUOTES,
) -> str:
    """Render the theme table with a few verbatim quotes each.

    Only quotes that already verify against the distilled store are shown, so
    the model never sees — and cannot echo back — a quote the gates will reject.
    """

    quotes: dict[str, list[str]] = {}
    for result in attributions:
        record = distilled.get(result.trace_id or "")
        for entry in (*result.violations, *result.harmful):
            theme = aggregation.theme_map.get(entry.theme_slug, entry.theme_slug)
            bucket = quotes.setdefault(theme, [])
            if len(bucket) >= quotes_per_theme:
                continue
            if record is not None and entry.quote in record.quotable(entry.quote_source.value):
                bucket.append(f'"{entry.quote.strip()}" ({entry.quote_source.value})')

    blocks: list[str] = []
    for theme in sorted(aggregation.themes, key=lambda item: (-item.numerator, item.theme)):
        header = (
            f"- {theme.theme}: {theme.numerator}/{theme.denominator} traces"
            f" ({theme.numerator / theme.denominator:.1%})"
            if theme.denominator
            else f"- {theme.theme}: {theme.numerator} traces"
        )
        if theme.instruction_ids:
            header += f"  attributed to: {', '.join(theme.instruction_ids)}"
        else:
            header += "  no instruction covers this (gap)"
        blocks.append(header)
        for quote in quotes.get(theme.theme, []):
            blocks.append(f"    {quote}")
    return "\n".join(blocks) if blocks else "(no failure themes were attributed)"


def _parse_edits(text: str) -> tuple[list[Edit], str]:
    payload = parse_json_response(text)
    if not isinstance(payload, dict):
        raise SynthesisError("synthesis response was not an object")
    raw_edits = payload.get("edits")
    if raw_edits is None:
        raw_edits = []
    if not isinstance(raw_edits, list):
        raise SynthesisError("synthesis field 'edits' was not a list")
    edits: list[Edit] = []
    for item in raw_edits:
        if not isinstance(item, dict):
            continue
        try:
            edits.append(
                Edit.model_validate(
                    {
                        "instruction_id": item.get("instruction_id") or "",
                        "operation": (item.get("operation") or "REWRITE").upper(),
                        "text": item.get("text") or "",
                        "covers_theme": item.get("covers_theme") or item.get("theme") or "",
                        "watch_metric": item.get("watch_metric")
                        or item.get("covers_theme")
                        or item.get("theme")
                        or "",
                    }
                )
            )
        except ValidationError:
            continue
    reasoning = payload.get("reasoning")
    return edits, reasoning if isinstance(reasoning, str) else ""


def _rejection_feedback(outcome: GateOutcome) -> str:
    lines = [
        f"- {rejection.edit.instruction_id} ({rejection.edit.operation}): "
        f"{rejection.reason}{' — ' + rejection.detail if rejection.detail else ''}"
        for rejection in outcome.rejected
    ]
    return "\n".join(lines)


def synthesize(
    inventory: Inventory,
    aggregation: Aggregation,
    backend: Backend,
    *,
    attributions: Sequence[AttributionResult] = (),
    distilled: Mapping[str, DistilledTrace] | None = None,
    memory: RejectionMemory | None = None,
    support: Mapping[str, int] | None = None,
    config: GateConfig | None = None,
    project_root: str | Path | None = None,
    run_id: str = "run",
    max_reprompts: int = MAX_REPROMPTS,
    eligible_gaps: Sequence[ThemeStat] = (),
) -> SynthesisResult:
    """Ask for edits, gate them, and re-prompt on gate rejections up to twice."""

    records = dict(distilled or {})
    instructions_block = render_instructions(inventory)
    evidence_block = render_evidence(aggregation, attributions, records)
    gaps_block = (
        "\n".join(f"- {gap.theme}" for gap in eligible_gaps)
        if eligible_gaps
        else "(none have graduated yet — do not propose additions for ungraduated gaps)"
    )
    base_user = (
        f"CURRENT SYSTEM PROMPT INSTRUCTIONS:\n{instructions_block}\n\n"
        f"FAILURE EVIDENCE (batch of {aggregation.denominator} traces):\n{evidence_block}\n\n"
        f"GAPS ELIGIBLE FOR A NEW INSTRUCTION:\n{gaps_block}"
    )

    all_proposed: list[Edit] = []
    feedback = ""
    outcome: GateOutcome | None = None
    reasoning = ""
    attempts = 0

    while attempts <= max_reprompts:
        attempts += 1
        user = base_user if not feedback else (
            f"{base_user}\n\nYOUR PREVIOUS PROPOSAL WAS REJECTED BY THE GATES:\n{feedback}\n"
            "Propose again, fixing those reasons, or return an empty edit list."
        )
        try:
            completion = backend.complete(SYSTEM_PROMPT, user)
            # Parsing lives inside the same guard: a response with no JSON at all
            # raises LLMError from parse_json_response, and that is a synthesis
            # failure to every caller, not a transport failure.
            edits, reasoning = _parse_edits(completion.text)
        except LLMError as exc:
            raise SynthesisError(f"synthesis backend failed: {exc}") from exc
        all_proposed.extend(edits)
        if not edits:
            outcome = GateOutcome(kept=(), rejected=())
            break
        resolution = resolve_edits(inventory, edits)
        outcome = run_gates(
            resolution,
            inventory,
            attributions=attributions,
            distilled=records,
            aggregation=aggregation,
            memory=memory,
            support=support,
            config=config,
        )
        if outcome.kept or not outcome.rejected:
            break
        feedback = _rejection_feedback(outcome)

    assert outcome is not None
    autopsy_path = None
    if outcome.rejected and project_root is not None:
        autopsy_path = _dump_autopsy(project_root, run_id, all_proposed, outcome, reasoning)

    return SynthesisResult(
        outcome=outcome,
        proposed=tuple(all_proposed),
        rounds=attempts,
        reasoning=reasoning,
        autopsy_path=autopsy_path,
    )


def _dump_autopsy(
    project_root: str | Path,
    run_id: str,
    proposed: Sequence[Edit],
    outcome: GateOutcome,
    reasoning: str,
) -> Path:
    """Persist every dropped proposal with its reason, for later reading.

    A run that proposes nothing is a fine outcome, but "nothing survived the
    gates" and "the model proposed nothing" are different facts, and the
    difference is worth keeping on disk.
    """

    layout = initialize(project_root)
    target = layout.runs / run_id / "autopsy.json"
    atomic_write_json(
        target,
        {
            "run_id": run_id,
            "reasoning": reasoning,
            "proposed": [edit.model_dump(mode="json") for edit in proposed],
            "rejected": [
                {
                    "instruction_id": rejection.edit.instruction_id,
                    "operation": rejection.edit.operation,
                    "text": rejection.edit.text,
                    "reason": rejection.reason,
                    "detail": rejection.detail,
                }
                for rejection in outcome.rejected
            ],
            "reclassified": list(outcome.reclassified),
            "flags": [
                {"instruction_id": flag.instruction_id, "flag": flag.flag, "detail": flag.detail}
                for flag in outcome.flags
            ],
        },
    )
    return target
