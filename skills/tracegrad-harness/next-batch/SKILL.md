---
name: next-batch
description: Conduct one Tracegrad loop — import, estimate, propose, review, maybe export, then status/trends. Unattended apply only if a policy file exists and permits it; otherwise stop at review.
---

# Next batch

Thin conductor over the other skills. Follow them; do not invent a parallel
pipeline. Unattended apply is **off** unless a policy file is present **and**
permits it. Otherwise stop at review and ask.

`tracegrad run` never writes the prompt. Only `tracegrad apply` does, and only
after human or policy accept.

## When to use

- The user asks to run the next batch, close the loop, or "do a Tracegrad
  pass".
- A new export is ready and they want import through trends in one go.

Do not use when they only asked for one step (import, estimate, review).
Invoke that skill instead.

## Steps

1. **Import** — follow [`../import-traces/SKILL.md`](../import-traces/SKILL.md).
   Sidecar adapt-in → JSONL. User pipeline unchanged.

   ```sh
   tracegrad init
   python sidecar-adapt-in.py --source /path/to/user-export --out batch.jsonl
   ```

2. **Estimate** — follow [`../propose-edits/SKILL.md`](../propose-edits/SKILL.md).

   ```sh
   tracegrad run --traces batch.jsonl --manifest manifest.json --estimate
   ```

   If spend is unclear, stop and ask before the paid run.

3. **Propose** — same skill. Default:

   ```sh
   tracegrad run --traces batch.jsonl --manifest manifest.json
   ```

   Staged alternative:

   ```sh
   tracegrad attribute --traces batch.jsonl --manifest manifest.json
   tracegrad propose --traces batch.jsonl --manifest manifest.json
   ```

   Stop with cards on disk under `.tracegrad/runs/<run-id>/`. Do not apply here.

4. **Review** — follow [`../review-edits/SKILL.md`](../review-edits/SKILL.md).

   - Policy file missing, `unattended_apply` not true, `accept` empty, or any
     rule ambiguous → **stop and ask**. Show the cards. Do not apply.
   - Policy present **and** permits a specific `--accept` list → apply only
     those indices.

   ```sh
   tracegrad apply --accept 0,2
   ```

   Never invent accepts. Never `--all` as a shortcut.

5. **Export** — only if apply actually wrote the template. Follow
   [`../export-prompt/SKILL.md`](../export-prompt/SKILL.md).

   ```sh
   python sidecar-adapt-out.py --from prompt.md --to /user/path/prompt.md
   ```

   Skip if apply was not run, or if the user path is already the manifest file.

6. **Status / trends**

   ```sh
   tracegrad status
   tracegrad status --manifest manifest.json
   tracegrad trends
   ```

   Trends need at least two runs. Advisory only — never auto-revert.

## Unattended apply (strict)

Apply without a human in the loop **only** when all of:

1. A policy file exists (see [`../examples/policy.commented.toml`](../examples/policy.commented.toml)).
2. `unattended_apply = true` (default in the example is **false**).
3. `accept` names real card indices; the agent does not fill gaps.
4. `allow_delete`, `neverDelete`, and `token_ceiling` all pass.

Otherwise the conductor **stops at review**.

## Inputs / outputs

**Inputs**

- User-store location, sidecar paths, JSONL destination, manifest,
  `--project-root` / `--base-directory`.
- Optional policy file. Absence is a valid input: it means stop at review.

**Outputs**

- JSONL batch, `.tracegrad/` proposal (always if run succeeded).
- Applied template + adapt-out copy **only** if review allowed apply.
- Status / trend text. No autonomous revert.

## Failure modes

| Failure | What to do |
| --- | --- |
| Import produced empty/invalid JSONL | Stop before `run`. |
| Estimate too large / no model configured | Stop and ask. |
| No edits proposed | Skip apply/export; still run `status` / `trends` if two reports exist. |
| Policy would apply but proposal is stale | Stop. Re-propose; do not `--force`. |
| Review blocked | Leave the prompt untouched. Report cards and wait. |
| Export destination unknown after apply | Prompt was still written by apply; ask where to copy. Do not revert. |
| Second batch with a changed judge fingerprint | Trends may be incomparable; report that instead of forcing a delta. |

## Exact CLI (full sequence)

```sh
tracegrad init
# sidecar adapt-in → batch.jsonl
tracegrad run --traces batch.jsonl --manifest manifest.json --estimate
tracegrad run --traces batch.jsonl --manifest manifest.json
# stop here unless policy/human named indices
tracegrad apply --accept 0,2
# sidecar adapt-out if the user path differs
tracegrad status --manifest manifest.json
tracegrad trends
```

Staged variant replaces the paid `run` with:

```sh
tracegrad attribute --traces batch.jsonl --manifest manifest.json
tracegrad propose --traces batch.jsonl --manifest manifest.json
```
