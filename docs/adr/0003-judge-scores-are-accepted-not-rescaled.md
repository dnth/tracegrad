# 0003. Judge scores are accepted, not rescaled

## Status

Accepted

## Context

`schema.Score` is bounded `[0, 1]`. Kitaru evaluations carry a `score` that may
be a bool, a float of unknown range, a string, or a categorical pair of score
plus value, plus an optional `passed` flag and an `explanation`.

Rescaling an out-of-range float against an assumed range would invent a judge
the user did not run. A `--kitaru-score-range` flag would be a declared
normalization, which is a product decision of its own.

tracegrad attribution is built on the rationale, not just the score. A missing
`explanation` is not a score of zero; it is a session that cannot be attributed.

## Decision

Map Kitaru evaluations onto `Judge` by name, never by rescaling:

| Kitaru | tracegrad |
|---|---|
| `bool` score, or `passed` flag | 1.0 / 0.0 |
| `float` in [0, 1] | as-is |
| `float` outside [0, 1] | drop `judge-score-out-of-range` |
| `str` / `categorical` | drop `judge-score-unsupported` |
| missing `explanation` | drop `judge-rationale-missing` |

`judge_fingerprint` is derived from the resolved `evaluator_name` +
`evaluator_version` and overrides the manifest value. A conflicting manifest
value is an error, not a tiebreak. Drift detection is only worth having if it
reads the thing that actually drifts.

One evaluation name resolving to more than one `evaluator_version` across the
cohort is `ambiguous-evaluation`: refuse rather than mix.

A declared `--kitaru-score-range` is deferred, not rejected.

## Consequences

- Out-of-range and non-numeric evaluations never become a `Trace`.
- Core ingest still sees only `[0, 1]` scores.
- Changing the Kitaru evaluator version changes the fingerprint, so trends
  across that change are not comparable.
