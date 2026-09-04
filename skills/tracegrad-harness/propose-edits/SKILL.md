---
name: propose-edits
description: "Propose cards: estimate then run, or attribute then propose. Stop on disk. Not apply."
---

# Propose edits

Run analysis and stop with review cards on disk. Apply is [`../review-edits/SKILL.md`](../review-edits/SKILL.md). `tracegrad run` does not write the prompt.

Flags: `tracegrad run --help`. Manifest: repo README (*The manifest*).

## Steps

1. **Project state.** Run `tracegrad init` if `.tracegrad/` is missing.

   Done: `.tracegrad/` exists.

2. **Estimate.** No model is contacted:

   ```sh
   tracegrad run --traces batch.jsonl --manifest manifest.json --estimate
   ```

   Done: CLI exit 0 and the user has approved spend (or already had). If spend is unclear, stop after this step and ask — do not start the paid run.

3. **Paid analysis.** Default:

   ```sh
   tracegrad run --traces batch.jsonl --manifest manifest.json
   ```

   Staged (pay attribution once, retry synthesis):

   ```sh
   tracegrad attribute --traces batch.jsonl --manifest manifest.json
   tracegrad propose --traces batch.jsonl --manifest manifest.json
   ```

   `propose` is `run` against cached attributions (one synthesis call). `tracegrad propose … --estimate` is the same preview as `run --estimate`. `--base-directory` must be where `template_file` resolves. Missing API key or harness binary → report the error; do not disable gates, invent a backend, or silently switch providers. Ingest dropping most traces → show the named reasons and fix JSONL via `import-traces`; do not lower the rationale floor in core.

   Done: CLI exit 0 and `.tracegrad/runs/<run-id>/proposal.json` exists. Cards print to stdout (`[index] OPERATION instruction_id`). Empty proposal is valid — do not force synthesis.

4. **Handoff.** Stop. Point `review-edits` at that run. Do not call `tracegrad apply`.

   Done: proposal path is in the reply; no apply was invoked.
