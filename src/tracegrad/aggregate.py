"""Aggregation of per-trace attributions into rates, themes, and the gap ledger.

Three rules shape this module:

* **Denominators are the whole batch.**  A theme seen in 3 of 200 traces is a
  3/200 problem, not a 3/5 problem, even if attribution only spoke about 5.
  Rates that quietly drop the silent majority are how prompt bloat happens.
* **Themes are unified once, mechanically.**  Exact slug first, then
  single-link Jaccard over slug tokens in sorted order, so the canonical
  vocabulary does not depend on trace order or on a model's mood.
* **A missing instruction has to earn its way in.**  Gaps accumulate in an
  append-only ledger and graduate only after appearing in at least two distinct
  runs or sessions.  Retirement is never automatic on a bad batch: it takes an
  improved trend or a human.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .schema import AttributionResult, Cluster
from .state import append_jsonl, load_jsonl

DEFAULT_JACCARD_THRESHOLD = 0.6
GRADUATION_DISTINCT_SOURCES = 2
GAP_LEDGER_FILENAME = "gaps.jsonl"
THEME_HISTORY_FILENAME = "themes.jsonl"

GAP_OBSERVED = "observed"
GAP_GRADUATED = "graduated"
GAP_RETIRED = "retired"


def _tokens(slug: str) -> frozenset[str]:
    return frozenset(part for part in slug.replace("_", "-").split("-") if part)


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union)


def unify_themes(
    slugs: Iterable[str], threshold: float = DEFAULT_JACCARD_THRESHOLD
) -> dict[str, str]:
    """Map every observed slug onto a canonical theme slug.

    Exact matches unify first.  The remaining slugs are single-link clustered by
    Jaccard similarity over their tokens, walked in sorted order so the result
    depends only on the set of slugs.
    """

    unique = sorted(set(slugs))
    parent: dict[str, str] = {slug: slug for slug in unique}

    def find(slug: str) -> str:
        while parent[slug] != slug:
            parent[slug] = parent[parent[slug]]
            slug = parent[slug]
        return slug

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        winner, loser = sorted((left_root, right_root))
        parent[loser] = winner

    token_sets = {slug: _tokens(slug) for slug in unique}
    for index, slug in enumerate(unique):
        for other in unique[index + 1 :]:
            if _jaccard(token_sets[slug], token_sets[other]) >= threshold:
                union(slug, other)

    return {slug: find(slug) for slug in unique}


@dataclass(frozen=True)
class InstructionStat:
    """Per-instruction failure counts over one batch."""

    instruction_id: str
    violations: int
    harmful: int
    denominator: int
    themes: tuple[str, ...] = ()

    @property
    def violation_rate(self) -> float:
        return self.violations / self.denominator if self.denominator else 0.0


@dataclass(frozen=True)
class ThemeStat:
    """Per-theme counts over one batch, with the slugs that were folded in."""

    theme: str
    numerator: int
    denominator: int
    aliases: tuple[str, ...] = ()
    instruction_ids: tuple[str, ...] = ()
    trace_ids: tuple[str, ...] = ()

    @property
    def is_gap(self) -> bool:
        return not self.instruction_ids

    def to_cluster(self) -> Cluster:
        return Cluster(theme=self.theme, numerator=self.numerator, denominator=self.denominator)


@dataclass(frozen=True)
class Aggregation:
    """Everything synthesis is allowed to reason from."""

    denominator: int
    themes: tuple[ThemeStat, ...]
    instructions: tuple[InstructionStat, ...]
    theme_map: Mapping[str, str] = field(default_factory=dict)

    @property
    def gaps(self) -> tuple[ThemeStat, ...]:
        return tuple(theme for theme in self.themes if theme.is_gap)

    @property
    def clusters(self) -> tuple[Cluster, ...]:
        return tuple(theme.to_cluster() for theme in self.themes)

    def theme(self, slug: str) -> ThemeStat | None:
        canonical = self.theme_map.get(slug, slug)
        for stat in self.themes:
            if stat.theme == canonical:
                return stat
        return None


def aggregate(
    attributions: Sequence[AttributionResult],
    denominator: int | None = None,
    *,
    threshold: float = DEFAULT_JACCARD_THRESHOLD,
) -> Aggregation:
    """Fold per-trace attributions into batch rates over a unified theme space.

    ``denominator`` is the size of the whole batch; it defaults to the number of
    attribution results, which is only correct when every trace was attributed.
    """

    batch_size = len(attributions) if denominator is None else denominator
    all_slugs = [
        entry.theme_slug
        for result in attributions
        for entry in (*result.violations, *result.harmful)
    ]
    theme_map = unify_themes(all_slugs, threshold)

    theme_traces: dict[str, set[str]] = defaultdict(set)
    theme_aliases: dict[str, set[str]] = defaultdict(set)
    theme_instructions: dict[str, set[str]] = defaultdict(set)
    violations_by_instruction: dict[str, set[str]] = defaultdict(set)
    harmful_by_instruction: dict[str, set[str]] = defaultdict(set)
    instruction_themes: dict[str, set[str]] = defaultdict(set)

    for position, result in enumerate(attributions):
        trace_key = result.trace_id or f"#{position}"
        for entry in result.violations:
            canonical = theme_map[entry.theme_slug]
            theme_traces[canonical].add(trace_key)
            theme_aliases[canonical].add(entry.theme_slug)
            if entry.instruction_id:
                theme_instructions[canonical].add(entry.instruction_id)
                violations_by_instruction[entry.instruction_id].add(trace_key)
                instruction_themes[entry.instruction_id].add(canonical)
        for entry in result.harmful:
            canonical = theme_map[entry.theme_slug]
            theme_traces[canonical].add(trace_key)
            theme_aliases[canonical].add(entry.theme_slug)
            if entry.instruction_id:
                theme_instructions[canonical].add(entry.instruction_id)
                harmful_by_instruction[entry.instruction_id].add(trace_key)
                instruction_themes[entry.instruction_id].add(canonical)

    themes = tuple(
        ThemeStat(
            theme=canonical,
            numerator=len(theme_traces[canonical]),
            denominator=batch_size,
            aliases=tuple(sorted(theme_aliases[canonical])),
            instruction_ids=tuple(sorted(theme_instructions[canonical])),
            trace_ids=tuple(sorted(theme_traces[canonical])),
        )
        for canonical in sorted(theme_traces)
    )

    instruction_ids = sorted(set(violations_by_instruction) | set(harmful_by_instruction))
    instructions = tuple(
        InstructionStat(
            instruction_id=instruction_id,
            violations=len(violations_by_instruction[instruction_id]),
            harmful=len(harmful_by_instruction[instruction_id]),
            denominator=batch_size,
            themes=tuple(sorted(instruction_themes[instruction_id])),
        )
        for instruction_id in instruction_ids
    )

    return Aggregation(
        denominator=batch_size,
        themes=themes,
        instructions=instructions,
        theme_map=theme_map,
    )


@dataclass(frozen=True)
class GapState:
    """The folded state of one theme in the gap ledger."""

    theme: str
    status: str
    runs: tuple[str, ...]
    sessions: tuple[str, ...]
    observations: int

    @property
    def distinct_sources(self) -> int:
        return max(len(self.runs), len(self.sessions))

    @property
    def is_graduated(self) -> bool:
        return self.status == GAP_GRADUATED

    def can_graduate(self, required: int = GRADUATION_DISTINCT_SOURCES) -> bool:
        return self.status != GAP_RETIRED and self.distinct_sources >= required


class GapLedger:
    """An append-only record of missing-instruction themes across runs.

    The ledger is events, never mutable state: ``observed`` entries accumulate,
    ``graduated`` marks a theme eligible to become a proposed addition, and
    ``retired`` is written only when a trend improved or a human resolved it.
    A ``revert`` restores the theme, because the reason it was retired is gone.
    """

    def __init__(self, path: str | Path, *, required_sources: int = GRADUATION_DISTINCT_SOURCES):
        self.path = Path(path)
        self.required_sources = required_sources

    def record_observation(
        self, theme: str, *, run_id: str, session_id: str | None = None, count: int = 1
    ) -> None:
        append_jsonl(
            self.path,
            {
                "event": GAP_OBSERVED,
                "theme": theme,
                "run_id": run_id,
                "session_id": session_id,
                "count": count,
            },
        )

    def record_observations(
        self,
        themes: Iterable[ThemeStat],
        *,
        run_id: str,
        session_id: str | None = None,
    ) -> None:
        for theme in themes:
            self.record_observation(
                theme.theme, run_id=run_id, session_id=session_id, count=theme.numerator
            )

    def graduate(self, theme: str, *, run_id: str) -> None:
        append_jsonl(self.path, {"event": GAP_GRADUATED, "theme": theme, "run_id": run_id})

    def retire(self, theme: str, *, run_id: str, reason: str) -> None:
        """Retire a theme.  Only an improved trend or a human resolve may do this."""

        if reason not in {"improved-trend", "human-resolved"}:
            raise ValueError(f"gap retirement requires a justified reason, got {reason!r}")
        append_jsonl(
            self.path,
            {"event": GAP_RETIRED, "theme": theme, "run_id": run_id, "reason": reason},
        )

    def restore(self, theme: str, *, run_id: str, reason: str = "revert") -> None:
        append_jsonl(
            self.path,
            {"event": "restored", "theme": theme, "run_id": run_id, "reason": reason},
        )

    def state(self) -> dict[str, GapState]:
        """Fold the event log into current per-theme state."""

        runs: dict[str, list[str]] = defaultdict(list)
        sessions: dict[str, list[str]] = defaultdict(list)
        status: dict[str, str] = {}
        observations: Counter[str] = Counter()

        for record in load_jsonl(self.path):
            theme = str(record.get("theme", ""))
            if not theme:
                continue
            event = str(record.get("event", ""))
            if event == GAP_OBSERVED:
                # An observation never un-retires a theme.  Retirement is a
                # decision — an improved trend or a human — and seeing the theme
                # again is exactly what a retired theme is expected to do for a
                # while.  Only restore() or a human resolve reopens it.
                status.setdefault(theme, GAP_OBSERVED)
                observations[theme] += int(record.get("count", 1) or 0)
                run_id = record.get("run_id")
                if isinstance(run_id, str) and run_id not in runs[theme]:
                    runs[theme].append(run_id)
                session_id = record.get("session_id")
                if isinstance(session_id, str) and session_id not in sessions[theme]:
                    sessions[theme].append(session_id)
            elif event in {GAP_GRADUATED, GAP_RETIRED}:
                status[theme] = event
            elif event == "restored":
                status[theme] = GAP_OBSERVED

        return {
            theme: GapState(
                theme=theme,
                status=state,
                runs=tuple(runs[theme]),
                sessions=tuple(sessions[theme]),
                observations=observations[theme],
            )
            for theme, state in sorted(status.items())
        }

    def eligible(self) -> tuple[GapState, ...]:
        """Themes that may become proposed additions in this run."""

        return tuple(
            gap
            for gap in self.state().values()
            if gap.is_graduated or gap.can_graduate(self.required_sources)
        )


class ThemeHistory:
    """How many distinct sessions or runs have ever shown each theme.

    The rejection-memory gate needs this and nothing else does: re-proposing an
    edit a human already turned down is only justified by the failure showing up
    somewhere new.  Counting traces inside one batch would clear that bar on
    every run, which is the same as not having the gate at all — so the unit
    here is the session (falling back to the run when the caller has no session
    id to give).
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def record(
        self,
        themes: Iterable[ThemeStat],
        *,
        run_id: str,
        session_id: str | None = None,
    ) -> None:
        for theme in themes:
            if theme.numerator <= 0:
                continue
            append_jsonl(
                self.path,
                {
                    "theme": theme.theme,
                    "run_id": run_id,
                    "session_id": session_id,
                    "traces": theme.numerator,
                },
            )

    def sources(self) -> dict[str, set[str]]:
        """Per theme, the distinct sources that have shown it."""

        seen: dict[str, set[str]] = defaultdict(set)
        for record in load_jsonl(self.path):
            theme = record.get("theme")
            if not isinstance(theme, str) or not theme:
                continue
            session_id = record.get("session_id")
            run_id = record.get("run_id")
            source = session_id if isinstance(session_id, str) and session_id else run_id
            if isinstance(source, str) and source:
                seen[theme].add(source)
        return dict(seen)

    def distinct_sources(self) -> dict[str, int]:
        """Per theme, the number of distinct sessions (or runs) it has appeared in."""

        return {theme: len(sources) for theme, sources in self.sources().items()}
