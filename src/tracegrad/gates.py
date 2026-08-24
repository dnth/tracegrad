"""The mechanical gates every proposed edit must survive.

Each gate is a pure function over a proposal set and the persisted evidence.
They are numbered to match the spec and are individually testable, because the
gates are the product: they are what makes a proposal trustworthy without a
human re-reading every trace.

Two rules hold across all of them:

* **Per-edit drop with a named reason.**  A gate never aborts a run.  A bad
  proposal is dropped and the reason appears on the review card.
* **Nothing is taken on the model's word.**  Quotes are substring-checked
  against the persisted distilled record, accounting is recomputed by re-diffing
  the actual result, and a REWRITE that smuggles in a new instruction is
  reclassified as the addition it really is.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .aggregate import Aggregation
from .distill import DistilledTrace
from .edits import Rejection, Resolution, ResolvedEdit, apply_resolved
from .inventory import Inventory
from .normalize import normalized_text
from .schema import AttributionEntry, AttributionResult, Edit  # noqa: F401
from .state import append_jsonl, load_jsonl

DEFAULT_EDIT_CAP = 5
DEFAULT_DISTINCT_SESSION_BAR = 2
NEGATION_WINDOW_TOKENS = 5
REJECTION_MEMORY_FILENAME = "rejections.jsonl"

REASON_EDIT_CAP = "G1-edit-cap"
REASON_ACCOUNTING = "G2-accounting-mismatch"
REASON_UNVERIFIED_QUOTE = "G4-unverified-quote"
REASON_NO_EVIDENCE = "G4-no-evidence"
REASON_BUDGET = "G5-budget-ceiling"
REASON_REMEMBERED_REJECTION = "G6-remembered-rejection"
REASON_VARIABLE_SPAN = "G7-variable-span"
REASON_DUPLICATE_ADD = "G8-duplicate-add"

_TOKEN = re.compile(r"\w+|[^\w\s]")
_CLAUSE = re.compile(r"[.;!?\n]+")
_NEGATIONS = frozenset({"not", "never", "no", "without", "don't", "dont", "avoid", "cannot", "n't"})
_IMPERATIVE_HINTS = frozenset(
    {
        "always",
        "never",
        "must",
        "should",
        "do",
        "don't",
        "dont",
        "avoid",
        "use",
        "cite",
        "include",
        "prefer",
        "keep",
        "write",
        "answer",
        "respond",
        "ensure",
        "limit",
        "refuse",
    }
)


def measure_tokens(text: str) -> int:
    """A deterministic token proxy: words and standalone punctuation.

    tracegrad never calls a tokenizer service, so the budget is measured with a
    stable proxy.  It is used only for relative comparison against a ceiling
    measured the same way.
    """

    return len(_TOKEN.findall(text))


@dataclass(frozen=True)
class GateFlag:
    """A non-fatal warning attached to an edit for the review card."""

    instruction_id: str
    flag: str
    detail: str = ""


@dataclass(frozen=True)
class GateConfig:
    """Tunable limits for one gate run."""

    edit_cap: int = DEFAULT_EDIT_CAP
    token_ceiling: int | None = None
    never_delete: tuple[str, ...] = ()
    distinct_session_bar: int = DEFAULT_DISTINCT_SESSION_BAR


@dataclass(frozen=True)
class GateOutcome:
    """What survived the gates, what did not, and what a human should see."""

    kept: tuple[ResolvedEdit, ...]
    rejected: tuple[Rejection, ...]
    flags: tuple[GateFlag, ...] = ()
    reclassified: tuple[str, ...] = ()
    tokens_before: int = 0
    tokens_after: int = 0

    @property
    def accepted_edits(self) -> tuple[Edit, ...]:
        return tuple(item.edit for item in self.kept)


@dataclass
class _Working:
    kept: list[ResolvedEdit]
    rejected: list[Rejection] = field(default_factory=list)
    flags: list[GateFlag] = field(default_factory=list)
    reclassified: list[str] = field(default_factory=list)

    def drop(self, item: ResolvedEdit, reason: str, detail: str = "") -> None:
        self.kept = [kept for kept in self.kept if kept is not item]
        self.rejected.append(Rejection(item.edit, reason, detail))


# ---------------------------------------------------------------- G1: edit cap


def gate_edit_cap(
    resolved: Sequence[ResolvedEdit],
    cap: int = DEFAULT_EDIT_CAP,
    *,
    rank: Mapping[str, int] | None = None,
    theme_map: Mapping[str, str] | None = None,
) -> tuple[tuple[ResolvedEdit, ...], tuple[Rejection, ...]]:
    """G1 — keep at most ``cap`` edits, strongest evidence first.

    ``rank`` is keyed by canonical theme, but a model proposes whatever slug it
    likes, so the lookup goes through ``theme_map`` first.
    """

    if len(resolved) <= cap:
        return tuple(resolved), ()
    weights = rank or {}
    canonical = theme_map or {}
    ordered = sorted(
        resolved,
        key=lambda item: (
            -weights.get(canonical.get(item.edit.covers_theme, item.edit.covers_theme), 0),
            item.edit.instruction_id,
        ),
    )
    kept = ordered[:cap]
    dropped = tuple(
        Rejection(item.edit, REASON_EDIT_CAP, f"beyond the cap of {cap}") for item in ordered[cap:]
    )
    kept_in_order = tuple(item for item in resolved if item in kept)
    return kept_in_order, dropped


# ------------------------------------------------------- G2: accounting re-diff


DROPPED_CLAUSE_TOLERANCE = 0.5
"""How much of an instruction a REWRITE may silently drop before it is a DELETE."""


def _dropped_clauses(original: str, replacement: str, *, overlap: float = 0.5) -> list[str]:
    """Clauses of ``original`` with no meaningful counterpart in ``replacement``."""

    replacement_clauses = [_clause_tokens(clause) for clause in _clauses(replacement)]
    dropped: list[str] = []
    for clause in _clauses(original):
        tokens = _clause_tokens(clause)
        if not tokens:
            continue
        best = max(
            (len(tokens & candidate) / len(tokens) for candidate in replacement_clauses),
            default=0.0,
        )
        if best < overlap:
            dropped.append(clause)
    return dropped


def shrinking_rewrites(resolved: Sequence[ResolvedEdit]) -> tuple[GateFlag, ...]:
    """Flag rewrites that drop most of the instruction they replace.

    Advisory, for the review card: the human sees "this rewrite removes most of
    what it replaces" next to the diff, and decides.
    """

    flags: list[GateFlag] = []
    for item in resolved:
        if item.operation != "REWRITE" or item.anchor is None:
            continue
        original = _clauses(item.anchor.text)
        dropped = _dropped_clauses(item.anchor.text, item.replacement)
        if original and len(dropped) / len(original) > DROPPED_CLAUSE_TOLERANCE:
            flags.append(
                GateFlag(
                    item.edit.instruction_id,
                    "drops-most-of-the-instruction",
                    f"{len(dropped)} of {len(original)} clauses have no counterpart",
                )
            )
    return tuple(flags)


def gate_accounting(
    prompt: str, resolved: Sequence[ResolvedEdit]
) -> tuple[tuple[ResolvedEdit, ...], tuple[Rejection, ...]]:
    """G2 — recompute what each edit actually did and reject false accounting.

    The declared operation is never trusted.  Each edit is re-applied on its own
    and the result is re-diffed against the original: a no-op, a DELETE that
    leaves its text in place, or an ADD that shrinks the prompt are all dropped.

    A rewrite that quietly drops most of the instruction it replaces is a
    deletion wearing a rewrite's label, but it is not rejected here: tightening
    a verbose instruction looks identical by token overlap, and that is the
    commonest legitimate edit there is.  It is flagged for the review card
    instead, and ``neverDelete`` is enforced against rewrites in G7 so protected
    text still cannot be dropped by relabelling the operation.
    """

    kept: list[ResolvedEdit] = []
    rejected: list[Rejection] = []
    before_tokens = measure_tokens(prompt)
    for item in resolved:
        after = apply_resolved(prompt, [item])
        after_tokens = measure_tokens(after)
        # Duplicate instruction text is supported on purpose (inventory gives
        # each copy its own ordinal), so a DELETE is checked by occurrence
        # count, not by presence — otherwise de-duplicating is impossible.
        removed = normalized_text(item.anchor.text) if item.anchor else ""
        still_present = bool(removed) and normalized_text(after).count(
            removed
        ) >= normalized_text(prompt).count(removed)
        if after == prompt:
            rejected.append(Rejection(item.edit, REASON_ACCOUNTING, "edit is a no-op"))
            continue
        if item.operation == "DELETE" and still_present:
            rejected.append(Rejection(item.edit, REASON_ACCOUNTING, "DELETE left the text in place"))
            continue
        if item.operation == "ADD" and after_tokens < before_tokens:
            rejected.append(Rejection(item.edit, REASON_ACCOUNTING, "ADD removed content"))
            continue
        kept.append(item)
    return tuple(kept), tuple(rejected)


# ------------------------------------------ G3: REWRITE -> ADD reclassification


def _clauses(text: str) -> list[str]:
    return [clause.strip() for clause in _CLAUSE.split(text) if clause.strip()]


def _clause_tokens(clause: str) -> set[str]:
    return {token.lower() for token in _TOKEN.findall(clause) if token.isalnum()}


def _is_imperative(clause: str) -> bool:
    tokens = [token.lower() for token in _TOKEN.findall(clause) if token.isalpha()]
    return bool(tokens) and bool(_IMPERATIVE_HINTS.intersection(tokens))


def introduces_new_clause(original: str, replacement: str, *, overlap: float = 0.5) -> bool:
    """Whether ``replacement`` adds an imperative clause absent from ``original``."""

    original_clauses = [_clause_tokens(clause) for clause in _clauses(original)]
    for clause in _clauses(replacement):
        if not _is_imperative(clause):
            continue
        tokens = _clause_tokens(clause)
        if not tokens:
            continue
        best = max(
            (len(tokens & existing) / len(tokens) for existing in original_clauses),
            default=0.0,
        )
        if best < overlap:
            return True
    return False


def gate_reclassify(
    resolved: Sequence[ResolvedEdit],
) -> tuple[tuple[ResolvedEdit, ...], tuple[str, ...]]:
    """G3 — a REWRITE that smuggles in a new instruction is really an ADD.

    Reclassification does not drop the edit; it makes the edit cap and the token
    budget count it honestly.
    """

    updated: list[ResolvedEdit] = []
    reclassified: list[str] = []
    for item in resolved:
        if item.operation != "REWRITE" or item.anchor is None:
            updated.append(item)
            continue
        if introduces_new_clause(item.anchor.text, item.replacement):
            updated.append(replace(item, edit=item.edit.model_copy(update={"operation": "ADD"})))
            reclassified.append(item.edit.instruction_id)
            continue
        updated.append(item)
    return tuple(updated), tuple(reclassified)


# ------------------------------------------------------ G4: source-checked quotes


def verify_quote(entry: AttributionEntry, distilled: DistilledTrace) -> bool:
    """Substring-verify one quote against the field its source declares."""

    haystack = distilled.quotable(entry.quote_source.value)
    if entry.quote in haystack:
        return True
    return normalized_text(entry.quote) in normalized_text(haystack)


def negation_window_flag(entry: AttributionEntry, distilled: DistilledTrace) -> bool:
    """Whether a negation immediately precedes the quote, inverting its sense."""

    haystack = distilled.quotable(entry.quote_source.value)
    position = haystack.find(entry.quote)
    if position < 0:
        normalized_haystack = normalized_text(haystack)
        position = normalized_haystack.find(normalized_text(entry.quote))
        if position < 0:
            return False
        haystack = normalized_haystack
    preceding = _TOKEN.findall(haystack[:position])[-NEGATION_WINDOW_TOKENS:]
    return any(token.lower() in _NEGATIONS for token in preceding)


def gate_evidence(
    resolved: Sequence[ResolvedEdit],
    attributions: Sequence[AttributionResult],
    distilled: Mapping[str, DistilledTrace],
    aggregation: Aggregation | None = None,
) -> tuple[tuple[ResolvedEdit, ...], tuple[Rejection, ...], tuple[GateFlag, ...]]:
    """G4 — every edit needs at least one quote that verifies against the store.

    A quote that does not appear in the persisted distilled record is not
    weak evidence; it is confabulation, and the edit goes with it.
    """

    theme_map = dict(aggregation.theme_map) if aggregation else {}

    def canonical(slug: str) -> str:
        return theme_map.get(slug, slug)

    entries_by_theme: dict[str, list[tuple[str | None, AttributionEntry]]] = {}
    for result in attributions:
        for entry in (*result.violations, *result.harmful):
            entries_by_theme.setdefault(canonical(entry.theme_slug), []).append(
                (result.trace_id, entry)
            )

    kept: list[ResolvedEdit] = []
    rejected: list[Rejection] = []
    flags: list[GateFlag] = []
    for item in resolved:
        theme = canonical(item.edit.covers_theme)
        entries = entries_by_theme.get(theme, [])
        if not entries:
            rejected.append(Rejection(item.edit, REASON_NO_EVIDENCE, theme))
            continue
        verified = 0
        for trace_id, entry in entries:
            record = distilled.get(trace_id) if trace_id else None
            if record is None:
                continue
            if not verify_quote(entry, record):
                continue
            verified += 1
            if negation_window_flag(entry, record):
                flags.append(
                    GateFlag(item.edit.instruction_id, "negation-window", entry.quote[:80])
                )
        if verified == 0:
            rejected.append(
                Rejection(item.edit, REASON_UNVERIFIED_QUOTE, f"no quote verified for {theme}")
            )
            continue
        kept.append(item)
    return tuple(kept), tuple(rejected), tuple(flags)


# -------------------------------------------------------------- G5: token budget


def gate_budget(
    prompt: str,
    resolved: Sequence[ResolvedEdit],
    ceiling: int | None,
) -> tuple[tuple[ResolvedEdit, ...], tuple[Rejection, ...], int, int]:
    """G5 — measure the budget on the template; at the ceiling, additions are zero-sum.

    Below the ceiling an addition is free to grow the prompt.  At or above it,
    the set as a whole may not grow: additions are dropped, largest first, until
    the result fits.
    """

    before = measure_tokens(prompt)
    kept = list(resolved)
    rejected: list[Rejection] = []
    if ceiling is None:
        return tuple(kept), (), before, measure_tokens(apply_resolved(prompt, kept))

    # At or above the ceiling the rule is zero-sum, not zero-additions: the set
    # may not grow. An addition paired with a deletion that pays for it is
    # exactly what the budget is meant to encourage, so the bar is the current
    # size, never a ceiling the prompt has already passed.
    limit = max(ceiling, before)
    while kept:
        after = measure_tokens(apply_resolved(prompt, kept))
        if after <= limit:
            break
        additions = [item for item in kept if item.operation == "ADD"]
        if not additions:
            break
        largest = max(
            additions, key=lambda item: (measure_tokens(item.replacement), item.edit.instruction_id)
        )
        kept.remove(largest)
        rejected.append(
            Rejection(
                largest.edit,
                REASON_BUDGET,
                f"prompt would reach {after} tokens against a ceiling of {limit}",
            )
        )
    return tuple(kept), tuple(rejected), before, measure_tokens(apply_resolved(prompt, kept))


# ------------------------------------------------- G6: rejection / revert memory


class RejectionMemory:
    """Append-only memory of what a human already turned down.

    A rejected proposal may return, but only carrying evidence from more
    distinct sessions than the bar — otherwise the tool re-asks the same
    question every batch, which is how a human stops reading review cards.
    """

    def __init__(self, path: str | Path, *, distinct_session_bar: int = DEFAULT_DISTINCT_SESSION_BAR):
        self.path = Path(path)
        self.distinct_session_bar = distinct_session_bar

    @staticmethod
    def key(edit: Edit) -> str:
        return f"{edit.instruction_id}\x1f{edit.operation}\x1f{normalized_text(edit.text)}"

    def record_rejection(self, edit: Edit, *, run_id: str, reason: str = "human-rejected") -> None:
        append_jsonl(
            self.path,
            {"event": "rejected", "key": self.key(edit), "run_id": run_id, "reason": reason},
        )

    def record_revert(self, edit: Edit, *, run_id: str) -> None:
        append_jsonl(
            self.path, {"event": "reverted", "key": self.key(edit), "run_id": run_id}
        )

    def counts(self) -> Counter[str]:
        counts: Counter[str] = Counter()
        for record in load_jsonl(self.path):
            if record.get("event") in {"rejected", "reverted"}:
                counts[str(record.get("key", ""))] += 1
        return counts

    def blocks(self, edit: Edit, *, distinct_sessions: int) -> bool:
        """Whether this edit was turned down before without new support since."""

        if self.counts().get(self.key(edit), 0) == 0:
            return False
        return distinct_sessions < self.distinct_session_bar


def gate_memory(
    resolved: Sequence[ResolvedEdit],
    memory: RejectionMemory | None,
    support: Mapping[str, int] | None = None,
    theme_map: Mapping[str, str] | None = None,
) -> tuple[tuple[ResolvedEdit, ...], tuple[Rejection, ...]]:
    """G6 — drop edits a human already rejected, unless new sessions back them.

    ``support`` is keyed by canonical theme; the edit names whatever slug the
    model used, so it is canonicalized before the lookup.  Without that, an
    alias reads as zero support and blocks a re-proposal forever.
    """

    if memory is None:
        return tuple(resolved), ()
    canonical = theme_map or {}
    kept: list[ResolvedEdit] = []
    rejected: list[Rejection] = []
    for item in resolved:
        theme = canonical.get(item.edit.covers_theme, item.edit.covers_theme)
        distinct = (support or {}).get(theme, 0)
        if memory.blocks(item.edit, distinct_sessions=distinct):
            rejected.append(
                Rejection(
                    item.edit,
                    REASON_REMEMBERED_REJECTION,
                    f"rejected before; {distinct} distinct sessions is below the bar",
                )
            )
            continue
        kept.append(item)
    return tuple(kept), tuple(rejected)


# ----------------------------------------------------------- G7: variable spans


def gate_variable_spans(
    resolved: Sequence[ResolvedEdit], inventory: Inventory, never_delete: Iterable[str] = ()
) -> tuple[tuple[ResolvedEdit, ...], tuple[Rejection, ...]]:
    """G7 — variable-origin spans and ``neverDelete`` matches are untouchable."""

    patterns = tuple(never_delete)
    kept: list[ResolvedEdit] = []
    rejected: list[Rejection] = []
    for item in resolved:
        anchor = item.anchor
        if anchor is not None and not anchor.editable:
            rejected.append(Rejection(item.edit, REASON_VARIABLE_SPAN, anchor.origin))
            continue
        if anchor is not None and item.operation in {"DELETE", "REWRITE"}:
            # A REWRITE that drops protected text is a delete by another name,
            # so neverDelete is checked against what actually survives.
            protected = next(
                (
                    pattern
                    for pattern in patterns
                    if pattern
                    and pattern in anchor.text
                    and pattern not in item.replacement
                ),
                None,
            )
            if protected:
                rejected.append(
                    Rejection(item.edit, REASON_VARIABLE_SPAN, f"neverDelete: {protected}")
                )
                continue
        kept.append(item)
    return tuple(kept), tuple(rejected)


# ------------------------------------------------------------ G8: duplicate ADDs


def gate_duplicate_adds(
    resolved: Sequence[ResolvedEdit], inventory: Inventory
) -> tuple[tuple[ResolvedEdit, ...], tuple[Rejection, ...]]:
    """G8 — an addition that already exists in the prompt or in the set is dropped."""

    existing = {instruction.normalized for instruction in inventory.instructions}
    kept: list[ResolvedEdit] = []
    rejected: list[Rejection] = []
    for item in resolved:
        if item.operation != "ADD":
            kept.append(item)
            continue
        candidate = normalized_text(item.replacement)
        if candidate in existing:
            rejected.append(Rejection(item.edit, REASON_DUPLICATE_ADD, candidate[:80]))
            continue
        existing.add(candidate)
        kept.append(item)
    return tuple(kept), tuple(rejected)


# ------------------------------------------------------------------ the pipeline


def run_gates(
    resolution: Resolution,
    inventory: Inventory,
    *,
    attributions: Sequence[AttributionResult] = (),
    distilled: Mapping[str, DistilledTrace] | None = None,
    aggregation: Aggregation | None = None,
    memory: RejectionMemory | None = None,
    support: Mapping[str, int] | None = None,
    config: GateConfig | None = None,
) -> GateOutcome:
    """Run G1–G8 in order and return what survived, with every reason recorded."""

    settings = config or GateConfig()
    prompt = inventory.prompt
    working = _Working(kept=list(resolution.resolved))
    working.rejected.extend(resolution.rejected)

    kept, rejected = gate_variable_spans(working.kept, inventory, settings.never_delete)
    working.kept = list(kept)
    working.rejected.extend(rejected)

    # Accounting runs first, judging the operation the model declared.  Running
    # it after reclassification would apply the ADD-only "did not shrink" check
    # to a REWRITE that legitimately tightens an instruction while adding a
    # clause — the most common real edit shape there is.
    kept, rejected = gate_accounting(prompt, working.kept)
    working.kept = list(kept)
    working.rejected.extend(rejected)
    working.flags.extend(shrinking_rewrites(working.kept))

    kept, reclassified = gate_reclassify(working.kept)
    working.kept = list(kept)
    working.reclassified.extend(reclassified)

    kept, rejected, flags = gate_evidence(
        working.kept, attributions, distilled or {}, aggregation
    )
    working.kept = list(kept)
    working.rejected.extend(rejected)
    working.flags.extend(flags)

    theme_map = dict(aggregation.theme_map) if aggregation else {}
    kept, rejected = gate_memory(working.kept, memory, support, theme_map)
    working.kept = list(kept)
    working.rejected.extend(rejected)

    kept, rejected = gate_duplicate_adds(working.kept, inventory)
    working.kept = list(kept)
    working.rejected.extend(rejected)

    rank = {
        theme.theme: theme.numerator for theme in (aggregation.themes if aggregation else ())
    }
    kept, rejected = gate_edit_cap(
        working.kept, settings.edit_cap, rank=rank, theme_map=theme_map
    )
    working.kept = list(kept)
    working.rejected.extend(rejected)

    kept, rejected, before, after = gate_budget(prompt, working.kept, settings.token_ceiling)
    working.kept = list(kept)
    working.rejected.extend(rejected)

    return GateOutcome(
        kept=tuple(working.kept),
        rejected=tuple(working.rejected),
        flags=tuple(working.flags),
        reclassified=tuple(working.reclassified),
        tokens_before=before,
        tokens_after=after,
    )
