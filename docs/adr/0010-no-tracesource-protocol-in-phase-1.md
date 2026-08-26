# 0010. No TraceSource protocol in Phase 1

## Status

Accepted

## Context

An earlier draft introduced a `TraceSource` protocol and a `sources/` package,
and proposed splitting parsing from validation in `ingest.py`, so Kitaru and
JSONL could be two in-process sources behind one interface.

With ADR 0004 there is no second in-process source to abstract over: Kitaru is
a fetch-and-map that writes the JSONL ingest already reads.
`ingest_traces` already accepts `Sequence[Trace]`, a path, or parsed records,
so the parsing/validation seam already exists.

A protocol with one implementation is an extra indirection the deterministic
core would have to import, which is how Kitaru would leak across the boundary.

`ports.py` exists so the orchestrator can hold a backend without becoming
backend-aware. That is the right place for `VerificationBackend` in Phase 2.
It is the wrong place for a trace source that does not enter the core.

## Decision

No `TraceSource` protocol. No `sources/` package. No change to `ingest.py` for
Kitaru.

All Kitaru SDK imports live under `src/tracegrad/integrations/kitaru/`.
`import tracegrad` never requires Kitaru. Using a Kitaru path without the extra
returns an actionable install message rather than an `ImportError`.

`VerificationBackend` goes in `ports.py`. The Kitaru implementation lives under
`src/tracegrad/integrations/kitaru/`.

## Consequences

- The core pipeline, ingest, and schema stay Kitaru-ignorant.
- Adding a second observability backend later is a new fetch-and-map, not a
  new core protocol.
- Phase 2 can hold a backend through `ports.py` without teaching `pipeline.py`
  about Kitaru.
