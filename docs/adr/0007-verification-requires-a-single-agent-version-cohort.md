# 0007. Verification requires a single-agent-version cohort

## Status

Accepted

## Context

`ExperimentRunCreateRequest.agent_version_id` applies to the whole run, while
`evaluate_baselines=True` scores the *stored* baselines under whatever version
each recorded. A mixed-agent-version cohort silently replays sessions under
code they never ran on.

A report that says "the prompt caused this" when the agent code also moved is
worse than refusing the cohort.

The same confounder exists on the judge: comparing a candidate scored under
evaluator version 3 against stored baseline scores from version 2 is not a
prompt comparison.

Kitaru can derive a single-agent-version cohort version via `remove_session_ids`.
Doing that automatically would hide the population change from the user.

## Decision

Every session in the cohort version must share one `agent_version_id`. A mixed
cohort is refused, with the version breakdown and session count per version.

This will refuse some real cohorts. That is acceptable.

Baseline and candidate always use the same evaluator version. A comparison
against stale stored judge output from a different evaluator version is marked
incomparable.

Deriving a single-agent-version cohort version via `remove_session_ids` is
deferred, not rejected.

## Consequences

- Phase 2 will not start an experiment run against a mixed cohort.
- Evaluator version is pinned from the Phase 1 source fingerprint, not
  re-resolved to `latest`.
- Users who need a mixed population must build a new cohort version themselves.
