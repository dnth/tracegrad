import pytest

from tracegrad.schema import Cluster, Verdict
from tracegrad.trends import (
    Proportion,
    compare,
    convergence,
    detectable_effect,
    difference_interval,
    evaluate_theme,
    format_trend,
    hysteresis,
    two_proportion_z,
)


def test_two_proportion_z_detects_clear_improvement() -> None:
    before = Proportion(40, 200)
    after = Proportion(10, 200)

    z, p = two_proportion_z(before, after)

    # pooled p = (40 + 10) / (200 + 200) = 50 / 400 = 0.125
    # SE = sqrt(0.125 * 0.875 * (1/200 + 1/200)) = sqrt(0.125 * 0.875 * 0.01)
    #    = sqrt(0.00109375) ≈ 0.033072
    # z = (0.05 - 0.2) / 0.033072 ≈ -4.5356
    assert z == pytest.approx(-4.535573676110727, abs=1e-6)
    assert p < 0.001


def test_two_proportion_z_no_change_returns_p_one() -> None:
    before = Proportion(20, 100)
    after = Proportion(20, 100)

    z, p = two_proportion_z(before, after)

    assert z == 0.0
    assert p == 1.0


def test_difference_interval_matches_hand_arithmetic() -> None:
    before = Proportion(40, 200)
    after = Proportion(10, 200)

    low, high = difference_interval(before, after)

    # unpooled variance = 0.2*0.8/200 + 0.05*0.95/200 = 0.0008 + 0.0002375 = 0.0010375
    # margin = 1.959963984540054 * sqrt(0.0010375) ≈ 0.063131
    # difference = 0.05 - 0.2 = -0.15
    # CI = (-0.15 - 0.063131, -0.15 + 0.063131) ≈ (-0.213131, -0.086869)
    assert low == pytest.approx(-0.21313092369409906, abs=1e-3)
    assert high == pytest.approx(-0.08686907630590099, abs=1e-3)


def test_detectable_effect_is_larger_for_small_batches() -> None:
    before = Proportion(10, 100)
    small_after = Proportion(8, 40)
    large_after = Proportion(8, 4000)

    small_floor = detectable_effect(before, small_after)
    large_floor = detectable_effect(before, large_after)

    assert small_floor > large_floor


def test_evaluate_theme_improved() -> None:
    result = evaluate_theme("tone", Proportion(40, 200), Proportion(10, 200))

    assert result.verdict is Verdict.IMPROVED
    assert result.is_significant


def test_evaluate_theme_regressed() -> None:
    result = evaluate_theme("tone", Proportion(10, 200), Proportion(40, 200))

    assert result.verdict is Verdict.REGRESSED
    assert result.is_significant


def test_evaluate_theme_eliminated_requires_zero_after_and_nonzero_before() -> None:
    result = evaluate_theme("tone", Proportion(40, 200), Proportion(0, 200))

    assert result.verdict is Verdict.ELIMINATED
    assert result.needs_reattribution


def test_evaluate_theme_no_signal_when_significant_but_below_min_effect() -> None:
    # A large batch can make a tiny, practically-irrelevant difference
    # statistically significant; min_effect keeps that from being actionable.
    result = evaluate_theme(
        "tone", Proportion(1000, 10000), Proportion(880, 10000), min_effect=0.5
    )

    assert result.is_significant
    assert abs(result.difference) < result.min_effect
    assert result.verdict is Verdict.NO_SIGNAL


def test_evaluate_theme_no_signal_when_batch_too_small() -> None:
    result = evaluate_theme("tone", Proportion(2, 10), Proportion(0, 10))

    assert result.verdict is Verdict.NO_SIGNAL
    assert not result.is_significant


def test_compare_over_union_of_themes_missing_treated_as_zero() -> None:
    before = [Cluster(theme="tone", numerator=40, denominator=200)]
    after = [
        Cluster(theme="tone", numerator=10, denominator=200),
        Cluster(theme="citations", numerator=30, denominator=200),
    ]

    report = compare(before, after)

    themes = {result.theme for result in report.results}
    assert themes == {"tone", "citations"}
    citations = report.by_theme("citations")
    assert citations is not None
    assert citations.before.numerator == 0
    assert citations.before.denominator == 200


def test_compare_targeted_vs_guardrail_regressed_elsewhere() -> None:
    before = [
        Cluster(theme="tone", numerator=10, denominator=200),
        Cluster(theme="citations", numerator=10, denominator=200),
    ]
    after = [
        Cluster(theme="tone", numerator=40, denominator=200),
        Cluster(theme="citations", numerator=40, denominator=200),
    ]

    report = compare(before, after, targeted=["tone"])

    assert report.by_theme("tone").verdict is Verdict.REGRESSED
    assert report.by_theme("citations").verdict is Verdict.REGRESSED
    assert report.regressed_elsewhere == ("citations",)


def test_hysteresis_one_regression_is_not_enough() -> None:
    history = [Verdict.NO_SIGNAL, Verdict.REGRESSED]

    assert hysteresis(history) is False


def test_hysteresis_two_consecutive_regressions_trigger() -> None:
    history = [Verdict.NO_SIGNAL, Verdict.REGRESSED, Verdict.REGRESSED]

    assert hysteresis(history) is True


def test_hysteresis_non_regression_breaks_the_streak() -> None:
    history = [Verdict.REGRESSED, Verdict.IMPROVED, Verdict.REGRESSED]

    assert hysteresis(history) is False


def test_convergence_counts_trailing_zero_proposal_int_runs() -> None:
    converged, streak = convergence([3, 0, 0], required_runs=2)

    assert converged is True
    assert streak == 2


def test_convergence_counts_trailing_zero_proposal_mapping_runs() -> None:
    history = [{"proposed": 2}, {"proposed": 0}, {"proposed": 0}]

    converged, streak = convergence(history, required_runs=2)

    assert converged is True
    assert streak == 2


def test_convergence_not_converged_when_streak_too_short() -> None:
    converged, streak = convergence([0, 3, 0], required_runs=2)

    assert converged is False
    assert streak == 1


def test_format_trend_contains_counts_ci_and_verdict() -> None:
    result = evaluate_theme("tone", Proportion(40, 200), Proportion(10, 200))

    line = format_trend(result)

    assert "40/200" in line
    assert "10/200" in line
    assert result.verdict.value in line
    low, high = result.confidence_interval
    assert f"{low:+.1%}" in line
    assert f"{high:+.1%}" in line
