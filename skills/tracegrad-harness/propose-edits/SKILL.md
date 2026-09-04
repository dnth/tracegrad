---
name: propose-edits
description: Estimate then run Tracegrad analysis so edit cards land under .tracegrad/. Use when asked to propose prompt edits, run a batch, or attribute+propose. Never apply.
---

# Propose edits

Run analysis and stop with review cards on disk. **Do not apply.**
`tracegrad run` never writes the prompt; only `tracegrad apply` does.

## When to use

- JSONL + manifest are ready and the user wants proposals.
- After `import-traces`, before `review-edits`.
- The user asks to estimate cost, run the pipeline, or use the staged
  attribute → propose path.

Do not use this skill to accept edits, pass `--accept` / `--all`, or copy the
template anywhere.

## Steps (default path)

1. Confirm project state exists:

   ```sh
   tracegrad init
   ```

2. Cost preview. No model is contacted:

   ```sh
   tracegrad run --traces batch.jsonl --manifest manifest.json --estimate
   ```

   Optional location flags (same for every command below):

   ```sh
   tracegrad run --traces batch.jsonl --manifest manifest.json \
     --project-root . --base-directory . --estimate
   ```

3. If the estimate is acceptable to the user (or they already approved spend),
   run the full pipeline:

   ```sh
   tracegrad run --traces batch.jsonl --manifest manifest.json
   ```

   Useful flags:

   ```sh
   tracegrad run --traces batch.jsonl --manifest manifest.json --jobs 8
   tracegrad run --traces batch.jsonl --manifest manifest.json --token-ceiling 4000
   tracegrad run --traces batch.jsonl --manifest manifest.json --run-id run-0001
   ```

4. Stop. Cards print to stdout. The proposal is
   `.tracegrad/runs/<run-id>/proposal.json`. Hand off to `review-edits`.

## Steps (staged path)

Use when attribution should be paid once and synthesis retried:

```sh
tracegrad attribute --traces batch.jsonl --manifest manifest.json
tracegrad propose --traces batch.jsonl --manifest manifest.json
```

`propose` is `run` against cached attributions (one synthesis call, not
another pass over the batch). It still does not write the prompt.

`tracegrad propose --traces batch.jsonl --manifest manifest.json --estimate`
is valid and is the same preview as `run --estimate`.

## Inputs / outputs

**Inputs**

- `--traces` — JSONL from `import-traces`.
- `--manifest` — JSON with `template_file`, `engine` (`none` or `format`),
  `judge_fingerprint`.
- Optional: `--project-root`, `--base-directory`, `--jobs`, `--token-ceiling`,
  `--run-id`, `--session-id`.
- Model access: `.tracegradrc` harness presets, or env key for the openai
  provider. Do not silently switch providers.

**Outputs (on disk under `.tracegrad/`)**

| Path | What |
| --- | --- |
| `runs/<run-id>/proposal.json` | Proposed edits, diffs, evidence, token counts |
| `runs/<run-id>/` | Resume checkpoint; autopsy of dropped proposals |
| `reports/*.json` | Theme counts for later `trends` |
| `distilled/` | Content-addressed traces quotes must match |
| `ledgers/` | Runs, gaps, rejections (append-only) |

Stdout from `run` / `propose` already prints one card per surviving edit
(`[index] OPERATION instruction_id`, theme, diff, quotes). "No edits
proposed" is a valid outcome.

## Failure modes

| Failure | What to do |
| --- | --- |
| Estimate looks expensive | Stop and ask before `run` without `--estimate`. |
| Missing API key / harness binary | Report the error. Do not disable gates or invent a backend. |
| Ingest drops most traces | Show dropped reasons (`rationale-below-quality-floor`, `invalid-schema`, `prompt-hash-partition`). Fix JSONL; do not lower the floor in core. |
| Stale or missing manifest / template | Stop. `--base-directory` must be where `template_file` resolves. |
| Proposal empty | Valid. Do not force synthesis. Proceed to `status` if asked. |
| Urge to "just apply the obvious ones" | Refuse. That is `review-edits`, and default is stop/ask. |

## Exact CLI

```sh
tracegrad init
tracegrad run --traces batch.jsonl --manifest manifest.json --estimate
tracegrad run --traces batch.jsonl --manifest manifest.json
tracegrad attribute --traces batch.jsonl --manifest manifest.json
tracegrad propose --traces batch.jsonl --manifest manifest.json
# Not in this skill:
# tracegrad apply
# tracegrad apply --accept …
# tracegrad apply --all
```
