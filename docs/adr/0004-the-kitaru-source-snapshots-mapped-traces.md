# 0004. The Kitaru source snapshots mapped traces

## Status

Accepted

## Context

`--source kitaru` has to turn a frozen Kitaru cohort into the JSONL the existing
pipeline already reads. If the fetch lived inside ingest, the deterministic core
would learn Kitaru exists, and a run would be unreproducible the moment the
server was unreachable.

Cohort names resolve to a moving "latest" version. Re-resolving `latest` inside
one run would mix two populations in one batch.

## Decision

`--source kitaru` is a fetch-and-map step that writes the JSONL the pipeline
already reads. The deterministic core does not learn Kitaru exists.

The mapped `Trace` objects are written to `.tracegrad/` as JSONL alongside the
source fingerprint, before ingest. Re-runs read the snapshot; `--refresh`
refetches and rewrites.

Prefer immutable cohort versions. Given only a cohort name: resolve the current
version once, persist the immutable `cohort_version_id`, use it for the whole
run, and include it in the source fingerprint. Never re-resolve `latest` within
a run.

The source fingerprint is:

```json
{
  "source": "kitaru",
  "cohort_id": "...",
  "cohort_version_id": "...",
  "evaluation_name": "quality",
  "evaluator_id": "...",
  "evaluator_version": 3,
  "agent_id": "...",
  "mapping_version": 1
}
```

A run is reproducible from the snapshot with the server unreachable.

## Consequences

- `ingest.py` does not change for Kitaru (see ADR 0010).
- Phase 2 reuses the cohort, evaluator, and agent metadata this snapshot
  persists.
- `--traces` and `--source kitaru` are mutually exclusive: one batch, one
  origin.
