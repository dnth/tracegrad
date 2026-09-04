# 0008. Inspection is handed off by identifier

## Status

Accepted

## Context

The Kitaru API exposes **no** UI URL for an experiment, run, session, or replay.
`client/dashboard_urls.py` has only `get_investigation_review_url`.
Reconstructing undocumented dashboard routes would break silently on the next
Kitaru UI change and would duplicate Kitaru's execution viewer inside
tracegrad, which the product boundary forbids.

`--open` and `tracegrad inspect` were in an earlier draft. They cannot be
honestly implemented against this API.

## Decision

Inspection hands off the supported dashboard base plus real identifiers.
`--open` / `tracegrad inspect` are **not** built.

Identifiers (experiment id, experiment-run id, session ids, replay ids) are
persisted on the verification record, so wiring a URL helper later is small.

tracegrad builds no competing execution viewer. Proposal approval and
application stay in tracegrad, not the Kitaru UI.

An upstream request for experiment / session / compare URL helpers is a
Kitaru-side follow-up, not a tracegrad feature.

## Consequences

- The verification report prints identifiers and the configured Kitaru server
  URL; the user inspects in Kitaru's UI.
- No new CLI commands for browsing sessions.
- A later URL helper can read the persisted identifiers without a schema break.
