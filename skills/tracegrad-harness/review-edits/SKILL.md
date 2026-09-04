---
name: review-edits
description: Show Tracegrad review cards and stop to ask the human. Apply only when a policy file clearly allows it. Use when asked to review, accept, or apply proposed edits. Never invent --accept indices.
---

# Review edits

Show the cards from the latest proposal. **Default: stop and ask.** Fully
unattended apply is off unless a policy file is present **and** permits it.

Only `tracegrad apply` writes the prompt. Do not edit the template by hand to
"save a step". Do not pass `--all` unless a human-named policy lists every
index.

## When to use

- After `propose-edits`, when cards exist under `.tracegrad/`.
- The user asks to review, accept, reject, or apply.
- `next-batch` reaches the review step.

Do not use this skill to re-run attribution or to export; those are other
skills. Do not apply if policy is missing or ambiguous.

## Steps

1. Find the proposal. Latest run id is the sorted name under
   `.tracegrad/runs/*/proposal.json`. Or pass `--run-id` the user named.

2. Show every card to the human: index, `ADD`/`REWRITE`/`DELETE`, instruction
   id, theme, flags, unified diff, verbatim quotes. Prefer the cards already
   printed by `tracegrad run`. Re-read `proposal.json` if the terminal scroll
   is gone. Do not call `apply --all` in order to "see" the result.

3. **Default path — stop and ask.** List the indices and wait. A human types
   which cards to accept. Then:

   ```sh
   tracegrad apply
   ```

   Interactive TTY: one `[y/N]` per card. Non-TTY without `--accept`/`--all`:
   nothing is applied (exit 1). That is correct, not a prompt to guess.

   After a human names indices:

   ```sh
   tracegrad apply --accept 0,2
   ```

4. **Policy path — optional, gated.** Read the project policy file (see
   [`../examples/policy.commented.toml`](../examples/policy.commented.toml)).
   Apply **only** if every condition holds:

   - File exists and parses.
   - `unattended_apply` is explicitly `true`.
   - `accept` is a non-empty list of integer indices that exist on this
     proposal. **Never invent or extend that list.**
   - Each selected edit is allowed: not `DELETE` unless `allow_delete = true`;
     instruction id not in policy `neverDelete`; proposal `tokens_after`
     does not exceed policy `token_ceiling` when that key is set.
   - Indices still match the current proposal (same `run_id`, template hash
     not stale).

   Then, and only then:

   ```sh
   tracegrad apply --accept 0,2
   ```

   Use `--run-id` when the policy or user named a run:

   ```sh
   tracegrad apply --run-id run-0001 --accept 0,2
   ```

5. If any check fails or is ambiguous: **stop and ask**. Do not fall back to
   `--all`. Do not apply a subset the policy did not name.

## Policy vs `.tracegradrc`

| File | Who reads it |
| --- | --- |
| `.tracegradrc` | Tracegrad core (`neverDelete`, coverage, harness presets) |
| apply policy (this skill) | The harness agent, before it may run `tracegrad apply` |

Core `neverDelete` already drops protected deletes at synthesize time. The
policy `neverDelete` is a second, agent-side refuse-to-apply list. It does
not replace the rc file.

## Inputs / outputs

**Inputs**

- `.tracegrad/runs/<run-id>/proposal.json`
- Optional policy file (path the user named, else
  `tracegrad-apply-policy.toml` in the project root if present)
- Optional `--run-id`, `--project-root`, `--base-directory`

**Outputs**

- Review text for the human (always).
- If apply ran: Tracegrad prints accepted count, new prompt hash, snapshot
  path under `.tracegrad/snapshots/`.
- If apply did not run: a clear stop/ask message. Prompt file unchanged.

Revert is a separate, explicit user request:

```sh
tracegrad apply --revert
tracegrad apply --revert --force
```

Do not revert as part of ordinary review.

## Failure modes

| Failure | What to do |
| --- | --- |
| No proposal | Tell the user to run `propose-edits` first. `tracegrad apply` exits 1. |
| Policy missing / `unattended_apply` false / omitted | Stop and ask. |
| `accept` empty, omitted, or contains unknown indices | Stop and ask. Never invent. |
| Selected edit is `DELETE` and `allow_delete` is not true | Skip apply; stop and ask. |
| Instruction id in policy `neverDelete` | Do not include it in `--accept`. If that empties the list, stop. |
| `tokens_after` over `token_ceiling` | Stop and ask. |
| "template changed … proposal is stale" | Do not `--force` apply. Re-run `propose-edits`. |
| Non-TTY, no `--accept` | Nothing applied. Ask the human; do not switch to `--all`. |
| `--all` looks convenient | Forbidden unless the policy's `accept` list is exactly every index a human already approved. Prefer `--accept` with that list. |

## Exact CLI

```sh
tracegrad apply
tracegrad apply --accept 0,2
tracegrad apply --run-id run-0001 --accept 0
tracegrad apply --project-root . --base-directory . --accept 0
tracegrad apply --revert
# Do not use unless a human-named policy lists every index:
# tracegrad apply --all
```

Related, not this skill's job:

```sh
tracegrad status
tracegrad trends
```
