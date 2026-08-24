"""Cross-batch trend statistics — advisory, with error bars, never automatic.

tracegrad does not revert an edit because a rate moved.  It reports whether the
move is distinguishable from noise, how large it might really be, and how large
an effect this batch size could even have detected.  A "no-signal" verdict on a
40-trace batch usually means the batch was small, not that the edit did nothing,
and the detectable-effect floor is printed so that is visible instead of implied.

The statistics are deliberately plain: a two-proportion z-test with a normal
confidence interval on the difference.  No scipy, no simulation, no
randomness — the same two batches always produce the same verdict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .schema import Cluster, Verdict

DEFAULT_ALPHA = 0.05
DEFAULT_MIN_EFFECT = 0.05
DEFAULT_POWER = 0.8
CONSECUTIVE_REGRESSIONS_FOR_HYSTERESIS = 2

_Z_TWO_SIDED_95 = 1.959963984540054
_Z_POWER_80 = 0.8416212335729143


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _z_two_sided(alpha: float) -> float:
    if abs(alpha - DEFAULT_ALPHA) < 1e-12:
        return _Z_TWO_SIDED_95
    # Inverse normal by bisection: exact enough for a reported interval, and
    # dependency-free.
    low, high = 0.0, 10.0
    target = 1.0 - alpha / 2.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if _normal_cdf(middle) < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


@dataclass(frozen=True)
class Proportion:
    """One rate as counted, never as a bare float."""

    numerator: int
    denominator: int

    @property
    def rate(self) -> float:
        return self.numerator / self.denominator if self.denominator else 0.0

    @classmethod
    def from_cluster(cls, cluster: Cluster) -> "Proportion":
        return cls(cluster.numerator, cluster.denominator)


@dataclass(frozen=True)
class TrendResult:
    """The full comparison of one theme across two batches."""

    theme: str
    before: Proportion
    after: Proportion
    difference: float
    confidence_interval: tuple[float, float]
    p_value: float
    verdict: Verdict
    detectable_effect: float
    alpha: float = DEFAULT_ALPHA
    min_effect: float = DEFAULT_MIN_EFFECT

    @property
    def is_significant(self) -> bool:
        return self.p_value < self.alpha

    @property
    def needs_reattribution(self) -> bool:
        """Whether this theme's traces must be re-attributed with one instrument."""

        return self.verdict in {Verdict.IMPROVED, Verdict.REGRESSED, Verdict.ELIMINATED}


def two_proportion_z(before: Proportion, after: Proportion) -> tuple[float, float]:
    """Return ``(z, p)`` for the pooled two-proportion test."""

    if not before.denominator or not after.denominator:
        return 0.0, 1.0
    pooled = (before.numerator + after.numerator) / (before.denominator + after.denominator)
    if pooled in (0.0, 1.0):
        return 0.0, 1.0
    standard_error = math.sqrt(
        pooled * (1 - pooled) * (1 / before.denominator + 1 / after.denominator)
    )
    if standard_error == 0.0:
        return 0.0, 1.0
    z = (after.rate - before.rate) / standard_error
    p_value = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return z, p_value


def difference_interval(
    before: Proportion, after: Proportion, alpha: float = DEFAULT_ALPHA
) -> tuple[float, float]:
    """Unpooled normal confidence interval on ``after - before``."""

    if not before.denominator or not after.denominator:
        return (0.0, 0.0)
    variance = (
        before.rate * (1 - before.rate) / before.denominator
        + after.rate * (1 - after.rate) / after.denominator
    )
    margin = _z_two_sided(alpha) * math.sqrt(variance)
    difference = after.rate - before.rate
    return (difference - margin, difference + margin)


def detectable_effect(
    before: Proportion, after: Proportion, alpha: float = DEFAULT_ALPHA, power: float = DEFAULT_POWER
) -> float:
    """The smallest difference these two batch sizes could reliably detect.

    Reported alongside every verdict so that "no-signal" can be read as "this
    batch was too small to tell" when that is what it means.
    """

    if not before.denominator or not after.denominator:
        return 1.0
    baseline = before.rate if 0.0 < before.rate < 1.0 else 0.5
    variance = baseline * (1 - baseline) * (1 / before.denominator + 1 / after.denominator)
    return min(1.0, (_z_two_sided(alpha) + _Z_POWER_80) * math.sqrt(variance))


def evaluate_theme(
    theme: str,
    before: Proportion,
    after: Proportion,
    *,
    alpha: float = DEFAULT_ALPHA,
    min_effect: float = DEFAULT_MIN_EFFECT,
) -> TrendResult:
    """Compare one theme across batches and assign a verdict.

    ``eliminated`` requires the theme to be gone, not merely smaller, and still
    requires a batch large enough to have seen it: zero out of three traces is
    not elimination.
    """

    _, p_value = two_proportion_z(before, after)
    interval = difference_interval(before, after, alpha)
    difference = after.rate - before.rate
    floor = detectable_effect(before, after, alpha)

    if after.numerator == 0 and before.numerator > 0 and p_value < alpha:
        verdict = Verdict.ELIMINATED
    elif p_value < alpha and abs(difference) >= min_effect:
        verdict = Verdict.IMPROVED if difference < 0 else Verdict.REGRESSED
    else:
        verdict = Verdict.NO_SIGNAL

    return TrendResult(
        theme=theme,
        before=before,
        after=after,
        difference=difference,
        confidence_interval=interval,
        p_value=p_value,
        verdict=verdict,
        detectable_effect=floor,
        alpha=alpha,
        min_effect=min_effect,
    )


def _cluster_map(clusters: Iterable[Cluster]) -> dict[str, Proportion]:
    return {cluster.theme: Proportion.from_cluster(cluster) for cluster in clusters}


@dataclass(frozen=True)
class TrendReport:
    """Every theme's trend, plus the guardrail read across all of them."""

    results: tuple[TrendResult, ...]
    regressed_elsewhere: tuple[str, ...] = ()
    converged: bool = False
    convergence_runs: int = 0

    def by_theme(self, theme: str) -> TrendResult | None:
        for result in self.results:
            if result.theme == theme:
                return result
        return None

    @property
    def improved(self) -> tuple[TrendResult, ...]:
        return tuple(r for r in self.results if r.verdict is Verdict.IMPROVED)

    @property
    def regressed(self) -> tuple[TrendResult, ...]:
        return tuple(r for r in self.results if r.verdict is Verdict.REGRESSED)

    @property
    def eliminated(self) -> tuple[TrendResult, ...]:
        return tuple(r for r in self.results if r.verdict is Verdict.ELIMINATED)

    @property
    def reattribution_themes(self) -> tuple[str, ...]:
        return tuple(r.theme for r in self.results if r.needs_reattribution)


def compare(
    before: Sequence[Cluster],
    after: Sequence[Cluster],
    *,
    targeted: Iterable[str] = (),
    alpha: float = DEFAULT_ALPHA,
    min_effect: float = DEFAULT_MIN_EFFECT,
) -> TrendReport:
    """Compare two batches theme by theme, over the union of their themes.

    ``targeted`` names the themes the accepted edits were meant to move.  Any
    *other* theme that regressed is reported as a guardrail breach: an edit that
    fixes its own theme while breaking two others has not helped.
    """

    before_map = _cluster_map(before)
    after_map = _cluster_map(after)
    before_size = max((p.denominator for p in before_map.values()), default=0)
    after_size = max((p.denominator for p in after_map.values()), default=0)

    results = []
    for theme in sorted(set(before_map) | set(after_map)):
        results.append(
            evaluate_theme(
                theme,
                before_map.get(theme, Proportion(0, before_size)),
                after_map.get(theme, Proportion(0, after_size)),
                alpha=alpha,
                min_effect=min_effect,
            )
        )

    targeted_set = set(targeted)
    guardrail = tuple(
        result.theme
        for result in results
        if result.verdict is Verdict.REGRESSED and result.theme not in targeted_set
    )
    return TrendReport(results=tuple(results), regressed_elsewhere=guardrail)


def hysteresis(
    history: Sequence[Verdict], required: int = CONSECUTIVE_REGRESSIONS_FOR_HYSTERESIS
) -> bool:
    """Whether a theme has regressed on enough consecutive runs to be actionable.

    One bad batch is a batch.  Two in a row is a signal — and even then it only
    surfaces the theme to a human; tracegrad still never reverts on its own.
    """

    streak = 0
    for verdict in reversed(list(history)):
        if verdict is Verdict.REGRESSED:
            streak += 1
            if streak >= required:
                return True
            continue
        break
    return False


def convergence(
    history: Sequence[Mapping[str, object]] | Sequence[int], required_runs: int = 2
) -> tuple[bool, int]:
    """Whether recent runs proposed nothing, and how many in a row.

    Convergence is the honest end state of this tool: the traces stop suggesting
    changes.  ``required_runs`` comes from ``convergenceRuns`` in the rc file.
    """

    streak = 0
    for entry in reversed(list(history)):
        count = entry if isinstance(entry, int) else int(entry.get("proposed", 0) or 0)
        if count == 0:
            streak += 1
            continue
        break
    return streak >= required_runs, streak


def format_trend(result: TrendResult) -> str:
    """One human-readable line per theme, counts first, then the interval."""

    low, high = result.confidence_interval
    return (
        f"{result.theme}: {result.before.numerator}/{result.before.denominator} "
        f"({result.before.rate:.1%}) -> {result.after.numerator}/{result.after.denominator} "
        f"({result.after.rate:.1%})  diff {result.difference:+.1%} "
        f"[95% CI {low:+.1%}, {high:+.1%}]  p={result.p_value:.3f}  "
        f"{result.verdict.value} (detectable >= {result.detectable_effect:.1%})"
    )
