# CONTEXT

Shared vocabulary for the Kitaru integration. Product boundary and roadmap
consequences live in GitHub issue #7; Phase 1 is #8; Phase 2 is #9. Decisions
are recorded in [`docs/adr/`](docs/adr/).

## Product

**tracegrad** is the evidence-gated optimization engine. It owns the prompt/text
artifact, artifact lineage and hash, deterministic distillation, attribution,
failure-theme aggregation, evidence verification, edit synthesis, token-budget
discipline, rejection memory, human approval, standalone JSONL source,
post-deployment trends, and the verification summary / decision UX.

**Kitaru** owns provider-specific trace ingestion, trace normalization, the
session graph, execution storage, evaluators and judge execution, cohorts,
replay, the worker runtime, tool-history replay, and execution / replay
visualization.

tracegrad is not an observability or replay platform. Kitaru is not a mandatory
core dependency. tracegrad does not duplicate Kitaru's execution UI.

```text
without Kitaru:  proposal → deploy → next batch → trends
with Kitaru:     proposal → replay verify → deploy → trends
```

`trends` stays in core. For core-only users it is verification after
deployment, via the next evaluated batch. For Kitaru users it is post-deployment
confirmation that replay-verified improvements persist in real traffic.

## Install modes

**Core-only.** `uv tool install tracegrad`. No Kitaru package, server, worker,
or observability integration. `tracegrad run --traces …`, `apply`, and `trends`
are unchanged. `import tracegrad` never requires Kitaru.

**tracegrad + Kitaru.** `uv tool install "tracegrad[kitaru]"`, extra pinned
`kitaru>=0.22,<0.23`. The extra installs a **client**. It does not install
verification (ADR 0001). Credentials and server URL come from Kitaru's own
config (`kitaru login`). tracegrad stores no Kitaru secrets. `.tracegradrc`
holds only non-secret selection (cohort name, evaluation name); CLI flags
override.

## Source vs batch (Phase 1)

`--source kitaru` is a **fetch-and-map** that writes the JSONL the existing
pipeline already reads (ADR 0004). `--traces` and `--source kitaru` are
mutually exclusive. The deterministic core does not learn Kitaru exists. There
is no `TraceSource` protocol, no `sources/` package, and no `ingest.py` rewrite
(ADR 0010). All Kitaru SDK code lives under `src/tracegrad/integrations/kitaru/`.

**Session** — a Kitaru recording (`SessionResponse`). Durable id is the UUID;
`number` is display-only (`#4811`).

**Trace** — a tracegrad `Trace`. One session maps to at most one trace.

**Root LLM node** — an `llm_call` with no `subagent_call` anywhere in its
ancestry, following `parent_index` **and** `secondary_parent_indexes`. The
graph is a DAG. A node reachable from a subagent is not root even when one of
its parents is (ADR 0005). Subagent prompts and tool outputs cannot become the
artifact or `Trace.output`. Never guess a missing system prompt.

**Source drop** — a Session could not become a Trace. Named kebab-case reasons
(`system-prompt-unavailable`, `multiple-system-prompts`,
`judge-rationale-missing`, `judge-score-out-of-range`,
`judge-score-unsupported`, `judge-score-unavailable`, `output-unavailable`,
`input-unavailable`, `ambiguous-evaluation`, …).

**Batch drop** — a Trace is not part of this Batch (the four reasons
`ingest.py` already uses, including `prompt-hash-partition` and
`rationale-below-quality-floor`).

Source drops and batch drops are two tables, never merged. Merging them lets a
mapping bug hide behind a legitimate partition.

**Snapshot** — mapped JSONL plus source fingerprint under `.tracegrad/`, written
before ingest. Re-runs read it; `--refresh` refetches. Cohort version is
resolved **once** per run (`cohort_version_id` in the fingerprint). `engine=format`
manifests are refused with a named error (ADR 0002).

**Judge mapping** — ADR 0003. `Score` stays `[0, 1]`; out-of-range and
non-numeric evaluations drop by name rather than being rescaled.
`judge_fingerprint` is derived from `evaluator_name` + `evaluator_version`. A
conflicting manifest value is an error.

## Verification (Phase 2)

`tracegrad verify --backend kitaru --run <id>` reuses the cohort, evaluator,
and agent metadata Phase 1 persisted. `VerificationBackend` lives in
`ports.py`; the implementation lives under `integrations/kitaru/`.

**No backend.** `tracegrad verify` prints an actionable message and exits
non-zero. That does not block `run` / `apply` / `trends`. A verify that exits 0
having done nothing would read as *verified* in CI.

**Tool policy (hard invariant).** Every tracegrad-created replay sets
`HistoryConfig(scope=COHORT_VERSION, on_miss=FAIL)`. No passthrough through the
tracegrad path. A novel call becomes `TOOL_HISTORY_MISS`, not a live production
side effect.

**Override scope (hard invariant).** Only the root LLM system prompt is
overridden (`ReplayOverride.system_prompt`). After replay, assert root LLM
nodes carry the candidate and non-root LLM nodes carry their baseline
counterpart. Violations are `OVERRIDE_SCOPE_DIVERGENCE`. A failed `select_evaluation`
is `SELECT_EVALUATION_FAILED` (drop reason in detail). A mismatched
`evaluator_version` is `EVALUATOR_VERSION_MISMATCH`. Scores that cannot
be classified are `SCORE_UNCLASSIFIED`. All of these are **incomparable**
— not improved, not regressed — and stay in the per-session buckets.
Headline aggregates still come from Kitaru (ADR 0006).

**Cohort constraint.** Mixed-agent-version cohorts are refused with a
per-version breakdown (ADR 0007). Baseline and candidate use the same evaluator
version.

**Apply gate.** With a backend configured (the originating run persisted Kitaru
source metadata), `apply` refuses unless a persisted verification exists whose
`candidate_prompt_hash` equals the hash of what is about to be written.
`--force` overrides. Core-only users are unaffected (ADR 0009).

**Persistence / resume.** `.tracegrad/verification/<verification-id>.json`.
Persist the Kitaru run id immediately after creation. An interrupted
verification resumes and watches the existing experiment run rather than
creating a duplicate.

**Inspection.** Hand off the dashboard base plus real identifiers. Do not build
`--open` or `tracegrad inspect` (ADR 0008).

**Verdict.** The verification report never prints `SHIP`, and never applies or
reverts on its own. The human decides.

**Aggregates.** Headline numbers come from
`/api/v1/ui/experiment-runs/{id}/evaluation-aggregates` so they match the
Kitaru UI. That `/api/v1/ui/` namespace is a UI-support contract, contained by
the `<0.23` pin.

## Do not build

Langfuse / LangSmith / Braintrust importers. A tracegrad-native replay engine.
Worker infrastructure. Tool mocking or a history engine. Cohort storage. Replay
experiment orchestration. A full eval runner. A competing trace viewer.
`--open` / `inspect`. An `approve`/`apply` rename (deferred, ADR 0009).
Multi-artifact editing (README "Beyond system prompts" is unaffected).
