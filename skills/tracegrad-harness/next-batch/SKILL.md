---
name: next-batch
description: "Batch: one loop of import → estimate → propose → review-edits → adapt-out → status/trends."
---

# Next batch

Thin conductor. Follow the other skills; do not invent a parallel pipeline. If the user asked for only one step, invoke that skill instead.

Apply lives in [`../review-edits/SKILL.md`](../review-edits/SKILL.md) (policy gate + attended `--accept`). `tracegrad run` does not write the prompt.

## Steps

1. **Import.** Follow [`../import-traces/SKILL.md`](../import-traces/SKILL.md).

   Done: that skill's JSONL exists with N > 0. Invalid or empty JSONL → stop before `run`.

2. **Estimate + propose.** Follow [`../propose-edits/SKILL.md`](../propose-edits/SKILL.md).

   Done: `.tracegrad/runs/<run-id>/proposal.json` exists (empty proposal is valid). Unclear spend or missing model → stop and ask.

3. **Review.** Follow [`../review-edits/SKILL.md`](../review-edits/SKILL.md).

   Done: that skill's apply step completed with `--accept`, **or** it stopped at ask and the prompt is unchanged. Stale proposal → re-propose; do not `--force`.

4. **Export.** Only if apply actually wrote the template. Follow [`../export-prompt/SKILL.md`](../export-prompt/SKILL.md). Skip if apply did not run, or if the user path is already the manifest file.

   Done: adapt-out ran, was a same-path no-op, or was skipped because apply did not write. Unknown destination after a real apply → ask where to copy; do not revert.

5. **Status / trends.**

   ```sh
   tracegrad status
   tracegrad trends
   ```

   `--manifest` and other flags: `tracegrad status --help` / `tracegrad trends --help`. Trends need at least two runs. Advisory only — do not auto-revert. A changed `judge_fingerprint` makes batches incomparable; report that instead of forcing a delta. Empty proposal → skip apply/export; still run these if two reports exist.

   Done: both CLIs exit 0, or `trends` stated the batches are not comparable.
