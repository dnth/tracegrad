# 0009. Apply is gated on a matching verification

## Status

Accepted

## Context

With a replay backend, the point of `verify` is that a candidate is replayed
against the same frozen cohort before it can be applied. Gating `apply` on
"a verify ran for this run id" is not enough: the user can verify, then
hand-edit the proposal or accept a subset of edits, and the text that is about
to be written was never replayed.

Core-only users have no backend. Blocking `apply` for them would break the
standalone workflow the umbrella issue guarantees.

Renaming `apply` to `approve` was considered so that "approve for testing"
and "apply for real" could be different verbs. That is a broader CLI change
and is not required to make the gate real.

## Decision

With a backend configured — meaning the originating run persisted Kitaru source
metadata — `apply` refuses unless a persisted verification exists whose
`candidate_prompt_hash` equals the hash of what is about to be written.
`--force` overrides.

Matching on the hash rather than the run id is what makes the gate real:
verify, hand-edit, and the gate correctly notices the text was never verified.

Core-only users are unaffected: JSONL runs have no Kitaru source metadata, so
`apply` behaves as it does today.

An `approve` / `apply` rename is deferred, not rejected.

## Consequences

- `apply --all` after a full-proposal verify is the matching happy path.
- Partial acceptance, hand-edits, and a stale candidate all refuse unless
  `--force`.
- `run`, `apply` (core-only), and `trends` keep working without a backend.
- `tracegrad verify` with no backend prints an actionable message and exits
  non-zero. A verify that exits 0 having done nothing would read as *verified*
  to every downstream CI step.
