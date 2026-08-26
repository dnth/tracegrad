# 0001. tracegrad submits verification; it does not host it

## Status

Accepted

## Context

Kitaru replay re-executes the user's real agent code. `ExperimentRunCreateRequest`
requires an `agent_version_id`, and `AgentVersionResponse` carries a `RunSpec`
with a shell command. Kitaru's README is explicit that replays run "on workers
in your environment: your virtualenv, your credentials, your network."

tracegrad is installed via `uv tool install`, into an isolated virtualenv, which
by construction is not the virtualenv the agent runs in. The optional extra
`tracegrad[kitaru]` installs a client library. It does not install a worker, a
server, Docker, or the user's agent.

Four capabilities an earlier draft assumed do not exist in the Kitaru 0.22 API,
and treating the extra as "gaining replay verification" would hide the real
prerequisites.

## Decision

tracegrad submits verification; it does not host it.

Phase 2 requires the user to already have:

1. a running Kitaru server (FastAPI + Postgres, via Docker),
2. a worker process running in the virtualenv where their agent code lives,
3. their agent instrumented with a Kitaru adapter and registered as an agent
   version.

`tracegrad verify` preflights those before spending anything: probe the server,
confirm a live worker claims this agent version (workers report `last_seen_at`),
and confirm the agent version and cohort version resolve. A replay experiment
is paid and slow; "no worker is polling" should surface in milliseconds.

tracegrad never hosts a worker and does not try to.

## Consequences

- The extra is a client pin (`kitaru>=0.22,<0.23`), not a verification runtime.
- Core-only users keep a complete JSONL workflow with no Kitaru package, server,
  or worker.
- Documentation must not claim that `tracegrad[kitaru]` "gains replay
  verification" by itself.
- Preflight cannot check whether an adapter honours a system-prompt override;
  that gap is closed by ADR 0006's post-replay assertion, not by prediction.
