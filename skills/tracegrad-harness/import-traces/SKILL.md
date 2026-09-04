---
name: import-traces
description: Pull the latest batch from the user store into Tracegrad JSONL via a sidecar adapt-in. Use when asked to import traces, prepare a batch, or feed Tracegrad without changing the user pipeline.
---

# Import traces

Map the user's existing trace store into Tracegrad JSONL. The user pipeline
stays unchanged. The adapter lives **beside** their repo, not in Tracegrad
core. Do not add vendor integrations under `src/tracegrad/`.

## When to use

- A new eval or production batch is ready and needs to enter Tracegrad.
- The user asks to import, adapt-in, or "pull latest traces".
- `next-batch` needs a JSONL path before `tracegrad run`.

Do not use this skill to call `tracegrad apply`, edit the prompt, or invent
fields the store does not have.

## JSONL contract

One JSON object per line. Extra keys are rejected (`extra: forbid`). Required:

```json
{
  "trace_id": "t-001",
  "input": "user or task text",
  "output": "model response",
  "judge": {"score": 0.4, "rationale": "why this score, in words"},
  "prompt_hash": "sha256:… of the prompt version that produced this trace"
}
```

| Field | Rules |
| --- | --- |
| `trace_id` | Non-empty string. Unique within the file. |
| `input` / `output` | Strings (may be empty, but output is what violations quote). |
| `judge.score` | Number in `[0.0, 1.0]`. |
| `judge.rationale` | Non-empty string. Ingest drops rationales shorter than 24 usable characters (`rationale-below-quality-floor`). A score without a real rationale is not a batch. |
| `prompt_hash` | Non-empty string identifying the prompt version. Mixed hashes: only the dominant partition is kept. |
| `meta` | Optional. Only `meta.model` is allowed. Mixed models are reported, not dropped. |

Do not emit `null` for required strings. Do not put the judge score in a
different shape (`label`, `pass`, nested vendor blobs). Map those in the
sidecar.

A manifest is not part of the JSONL. It is a separate JSON file
(`template_file`, `engine`, `judge_fingerprint`, …) passed to `tracegrad run
--manifest`. Confirm it exists; do not rewrite the user's stack to produce it.

## Thin adapt-in

1. Copy [`../examples/sidecar-adapt-in.py`](../examples/sidecar-adapt-in.py) next
   to the user's repo (or edit their existing adapter).
2. Fill the field map: vendor id → `trace_id`, prompt/response → `input`/`output`,
   judge score + rationale, prompt version → `prompt_hash`.
3. Write JSONL. Leave Tracegrad unaware of the source format.

```sh
python sidecar-adapt-in.py --source /path/to/user-export --out batch.jsonl
```

## Steps

1. Confirm `tracegrad init` has been run in the project (creates `.tracegrad/`):

   ```sh
   tracegrad init
   ```

2. Locate the latest user-store export. Do not scrape production APIs unless
   the user named the source.
3. Run the sidecar adapt-in. Do not import traces by hand-writing JSONL unless
   the batch is tiny and the user asked.
4. Sanity-check: line count, unique `trace_id`, rationale length, single
   dominant `prompt_hash`.
5. Hand the JSONL path to `propose-edits` / `next-batch`. Stop. This skill
   does not run analysis.

## Inputs / outputs

**Inputs**

- Path to the user store or export (file, directory, or command the user named).
- Sidecar script path (default: a copy of `examples/sidecar-adapt-in.py`).
- Destination JSONL path (for example `batch.jsonl`).
- Optional: `--project-root` if `.tracegrad/` is not cwd.

**Outputs**

- JSONL file matching the contract above.
- A short note: how many lines written, which source, which `prompt_hash` values
  were seen. No prompt writes.

## Failure modes

| Failure | What to do |
| --- | --- |
| No adapter and unknown store format | Stop and ask. Do not invent a vendor integration in core. |
| Missing rationale or score not in `[0, 1]` | Fix the map or drop the row in the adapter; do not pad fake rationales. |
| Duplicate `trace_id` | Deduplicate in the adapter; ingest will drop later copies as `duplicate-trace-id`. |
| Several `prompt_hash` values | Warn: ingest keeps only the dominant partition. Split batches if the user wants both versions. |
| Empty export | Stop. Do not call `tracegrad run` on an empty file. |
| Temptation to "just add a connector to src/tracegrad" | Refuse. Sidecar only. |

## Exact CLI

```sh
tracegrad init
# Adapt-in is not a tracegrad subcommand. Then, later:
tracegrad run --traces batch.jsonl --manifest manifest.json --estimate
```

`tracegrad init` is the only Tracegrad command this skill should run. Import
does not call `tracegrad run`, `attribute`, `propose`, `apply`, `status`, or
`trends`.
