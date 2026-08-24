import random
from pathlib import Path

import pytest

from tracegrad.aggregate import (
    GAP_GRADUATED,
    GAP_OBSERVED,
    GAP_RETIRED,
    GapLedger,
    ThemeHistory,
    ThemeStat,
    aggregate,
    unify_themes,
)
from tracegrad.schema import AttributionResult, Cluster


def _violation(instruction_id: str, theme_slug: str) -> dict[str, str]:
    return {
        "instruction_id": instruction_id,
        "theme_slug": theme_slug,
        "quote": "some quote",
        "quote_source": "output",
    }


def _harmful(theme_slug: str, instruction_id: str | None = None) -> dict[str, str | None]:
    return {
        "instruction_id": instruction_id,
        "theme_slug": theme_slug,
        "quote": "some harmful quote",
        "quote_source": "output",
    }


def test_unify_themes_exact_match() -> None:
    mapping = unify_themes(["missing-context", "missing-context", "unsafe-advice"])

    assert mapping["missing-context"] == "missing-context"
    assert mapping["unsafe-advice"] == "unsafe-advice"


def test_unify_themes_unifies_above_jaccard_threshold() -> None:
    mapping = unify_themes(["cite-sources", "cite-sources-properly"])

    assert mapping["cite-sources"] == mapping["cite-sources-properly"]


def test_unify_themes_leaves_below_threshold_slugs_separate() -> None:
    mapping = unify_themes(["add-code-comments", "add-inline-comments"])

    assert mapping["add-code-comments"] != mapping["add-inline-comments"]


def test_unify_themes_canonical_representative_is_order_independent() -> None:
    slugs = ["cite-sources", "cite-sources-properly", "add-code-comments", "add-inline-comments"]

    baseline = unify_themes(slugs)

    shuffled = slugs[:]
    random.shuffle(shuffled)
    shuffled_mapping = unify_themes(shuffled)

    assert shuffled_mapping == baseline


def test_aggregate_denominator_is_the_whole_batch_not_attributed_count() -> None:
    attributions = [
        AttributionResult(trace_id="t1", violations=[_violation("i-1", "missing-context")]),
    ]

    aggregation = aggregate(attributions, denominator=200)

    assert aggregation.denominator == 200
    theme = aggregation.theme("missing-context")
    assert theme is not None
    assert theme.denominator == 200
    instruction = aggregation.instructions[0]
    assert instruction.denominator == 200


def test_aggregate_defaults_denominator_to_number_of_attributions() -> None:
    attributions = [
        AttributionResult(trace_id="t1", violations=[_violation("i-1", "missing-context")]),
        AttributionResult(trace_id="t2"),
    ]

    aggregation = aggregate(attributions)

    assert aggregation.denominator == 2


def test_trace_counted_once_per_theme_with_two_entries() -> None:
    attributions = [
        AttributionResult(
            trace_id="t1",
            violations=[
                _violation("i-1", "missing-context"),
                _violation("i-2", "missing-context"),
            ],
        ),
    ]

    aggregation = aggregate(attributions, denominator=10)

    theme = aggregation.theme("missing-context")
    assert theme is not None
    assert theme.numerator == 1
    assert theme.trace_ids == ("t1",)
    assert theme.instruction_ids == ("i-1", "i-2")


def test_instruction_stat_counts_violations_and_harmful_and_rate() -> None:
    attributions = [
        AttributionResult(trace_id="t1", violations=[_violation("i-1", "missing-context")]),
        AttributionResult(trace_id="t2", violations=[_violation("i-1", "lacks-detail")]),
        AttributionResult(trace_id="t3", harmful=[_harmful("unsafe-advice", "i-1")]),
    ]

    aggregation = aggregate(attributions, denominator=10)

    instruction = next(i for i in aggregation.instructions if i.instruction_id == "i-1")
    assert instruction.violations == 2
    assert instruction.harmful == 1
    assert instruction.denominator == 10
    assert instruction.violation_rate == 0.2


def test_theme_stat_is_gap_only_when_no_instruction_attributed() -> None:
    attributions = [
        AttributionResult(trace_id="t1", violations=[_violation("i-1", "missing-context")]),
        AttributionResult(trace_id="t2", harmful=[_harmful("unsafe-advice")]),
    ]

    aggregation = aggregate(attributions, denominator=10)

    attributed_theme = aggregation.theme("missing-context")
    gap_theme = aggregation.theme("unsafe-advice")
    assert attributed_theme is not None and attributed_theme.is_gap is False
    assert gap_theme is not None and gap_theme.is_gap is True
    assert aggregation.gaps == (gap_theme,)


def test_aggregation_clusters_round_trip_to_schema_cluster() -> None:
    attributions = [
        AttributionResult(trace_id="t1", violations=[_violation("i-1", "missing-context")]),
    ]

    aggregation = aggregate(attributions, denominator=10)

    assert all(isinstance(cluster, Cluster) for cluster in aggregation.clusters)
    cluster = next(c for c in aggregation.clusters if c.theme == "missing-context")
    assert cluster.numerator == 1
    assert cluster.denominator == 10


def test_gap_ledger_record_observation_across_runs(tmp_path: Path) -> None:
    ledger = GapLedger(tmp_path / "gaps.jsonl")

    ledger.record_observation("missing-context", run_id="run-1")
    state = ledger.state()["missing-context"]
    assert state.status == GAP_OBSERVED
    assert state.runs == ("run-1",)
    assert state.can_graduate() is False

    ledger.record_observation("missing-context", run_id="run-2")
    state = ledger.state()["missing-context"]
    assert state.runs == ("run-1", "run-2")
    assert state.can_graduate() is True


def test_gap_ledger_can_graduate_via_distinct_sessions(tmp_path: Path) -> None:
    ledger = GapLedger(tmp_path / "gaps.jsonl")

    ledger.record_observation("missing-context", run_id="run-1", session_id="session-a")
    ledger.record_observation("missing-context", run_id="run-1", session_id="session-b")

    state = ledger.state()["missing-context"]
    assert state.runs == ("run-1",)
    assert state.sessions == ("session-a", "session-b")
    assert state.can_graduate() is True


def test_gap_ledger_graduate_sets_status(tmp_path: Path) -> None:
    ledger = GapLedger(tmp_path / "gaps.jsonl")

    ledger.record_observation("missing-context", run_id="run-1")
    ledger.record_observation("missing-context", run_id="run-2")
    ledger.graduate("missing-context", run_id="run-2")

    state = ledger.state()["missing-context"]
    assert state.status == GAP_GRADUATED
    assert state.is_graduated is True


def test_gap_ledger_retire_rejects_unjustified_reason(tmp_path: Path) -> None:
    ledger = GapLedger(tmp_path / "gaps.jsonl")
    ledger.record_observation("missing-context", run_id="run-1")

    with pytest.raises(ValueError, match="justified reason"):
        ledger.retire("missing-context", run_id="run-1", reason="bad-batch")


@pytest.mark.parametrize("reason", ["improved-trend", "human-resolved"])
def test_gap_ledger_retire_accepts_justified_reasons(tmp_path: Path, reason: str) -> None:
    ledger = GapLedger(tmp_path / "gaps.jsonl")
    ledger.record_observation("missing-context", run_id="run-1")

    ledger.retire("missing-context", run_id="run-1", reason=reason)

    state = ledger.state()["missing-context"]
    assert state.status == GAP_RETIRED


def test_gap_ledger_restore_reopens_retired_theme(tmp_path: Path) -> None:
    ledger = GapLedger(tmp_path / "gaps.jsonl")
    ledger.record_observation("missing-context", run_id="run-1")
    ledger.retire("missing-context", run_id="run-1", reason="human-resolved")

    ledger.restore("missing-context", run_id="run-2")

    state = ledger.state()["missing-context"]
    assert state.status == GAP_OBSERVED


def test_a_later_observation_does_not_reopen_a_retired_theme(tmp_path: Path) -> None:
    # Retirement is a decision, and a retired theme is expected to keep showing
    # up for a while. Only restore() or a human resolve reopens it — otherwise
    # "retired" would mean nothing after the very next batch.
    ledger = GapLedger(tmp_path / "gaps.jsonl")
    ledger.record_observation("missing-context", run_id="run-1")
    ledger.retire("missing-context", run_id="run-1", reason="human-resolved")

    ledger.record_observation("missing-context", run_id="run-2")

    state = ledger.state()["missing-context"]
    assert state.status == GAP_RETIRED
    # The observation is still counted; it just does not change the decision.
    assert state.observations == 2


def test_gap_ledger_eligible_returns_graduated_and_graduation_ready(tmp_path: Path) -> None:
    ledger = GapLedger(tmp_path / "gaps.jsonl")

    ledger.record_observation("ready-theme", run_id="run-1")
    ledger.record_observation("ready-theme", run_id="run-2")

    ledger.record_observation("graduated-theme", run_id="run-1")
    ledger.record_observation("graduated-theme", run_id="run-2")
    ledger.graduate("graduated-theme", run_id="run-2")

    ledger.record_observation("not-ready-theme", run_id="run-1")

    eligible_themes = {gap.theme for gap in ledger.eligible()}
    assert eligible_themes == {"ready-theme", "graduated-theme"}


def test_theme_history_counts_distinct_sessions_not_traces(tmp_path: Path) -> None:
    # This is the number G6 reads. Counting traces inside one batch would clear
    # the re-proposal bar on every run, which is the same as no gate at all.
    history = ThemeHistory(tmp_path / "themes.jsonl")
    themes = (ThemeStat(theme="missing-citation", numerator=9, denominator=20),)

    history.record(themes, run_id="run-1", session_id="session-a")

    assert history.distinct_sources() == {"missing-citation": 1}

    history.record(themes, run_id="run-2", session_id="session-b")

    assert history.distinct_sources() == {"missing-citation": 2}


def test_theme_history_falls_back_to_runs_without_a_session_id(tmp_path: Path) -> None:
    history = ThemeHistory(tmp_path / "themes.jsonl")
    themes = (ThemeStat(theme="jargon-tone", numerator=3, denominator=20),)

    history.record(themes, run_id="run-1")
    history.record(themes, run_id="run-1")
    history.record(themes, run_id="run-2")

    assert history.distinct_sources() == {"jargon-tone": 2}


def test_theme_history_ignores_themes_with_no_observations(tmp_path: Path) -> None:
    history = ThemeHistory(tmp_path / "themes.jsonl")

    history.record(
        (ThemeStat(theme="quiet", numerator=0, denominator=20),),
        run_id="run-1",
        session_id="session-a",
    )

    assert history.distinct_sources() == {}
