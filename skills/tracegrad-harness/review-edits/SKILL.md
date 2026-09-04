---
name: review-edits
description: Show Tracegrad review cards and stop to ask the human. Apply only with tracegrad apply --accept after a human names indices, or when a policy file lists them. Never answer interactive y/N. Never invent --accept indices. Never use bare apply.
---

# Review edits

Show the cards from the latest proposal. **Default: stop and ask.** Fully
unattended apply is off unless a policy file is present **and** permits it.
Attended apply is allowed after the human names card indices.

Only `tracegrad apply --accept <indices>` writes the prompt on the harness
path. Do not edit the template by hand to "save a step". Do not pass `--all`
unless a human-named policy lists every index. **Do not run bare
`tracegrad apply`.** That form is interactive `[y/N]` per card; a TTY agent
can answer those prompts itself. That is not a human-named accept.

## When to use

- After `propose-edits`, when cards exist under `.tracegrad/`.
- The user asks to review, accept, reject, or apply.
- `next-batch` reaches the review step.

Do not use this skill to re-run attribution or to export; those are other
skills. Do not apply unattended if policy is missing or ambiguous. Attended
apply still requires explicit human-named `--accept` indices.

## Steps

1. Find the proposal. Latest run id is the sorted name under
   `.tracegrad/runs/*/proposal.json`. Or pass `--run-id` the user named.

2. Show every card to the human: index, `ADD`/`REWRITE`/`DELETE`, instruction
   id, theme, flags, unified diff, verbatim quotes. Prefer the cards already
   printed by `tracegrad run`. Re-read `proposal.json` if the terminal scroll
   is gone. Do not call `apply --all` in order to "see" the result.

3. **Default path — stop and ask.** List the card indices and wait. Do not
   call `tracegrad apply` yet. Do not answer `[y/N]`. A human must type which
   cards to accept (for example `0` and `2`). Only after those indices exist
   in this conversation, substitute them and run:

   ```sh
   tracegrad apply --accept <HUMAN_OR_POLICY_INDICES>
   ```

   The placeholder is not an example list. Replace it with the integers the
   human typed (or, on the policy path below, the integers the file listed).
   **Forbidden for the harness agent:** bare `tracegrad apply`, answering
   interactive `[y/N]`, piping `yes`/`y` into apply, or using `--all` to skip
   naming indices. Non-TTY without `--accept`/`--all` applies nothing (exit
   1); that is correct, not a prompt to guess or to switch to a TTY.

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
   tracegrad apply --accept <HUMAN_OR_POLICY_INDICES>
   ```

   Use `--run-id` when the policy or user named a run:

   ```sh
   tracegrad apply --run-id run-0001 --accept <HUMAN_OR_POLICY_INDICES>
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
| Policy missing / `unattended_apply` false / omitted | Do not apply unattended. Stop and ask. Attended `--accept` is allowed if the human named indices. |
| `accept` empty, omitted, or contains unknown indices | Stop and ask. Never invent. |
| Selected edit is `DELETE` and `allow_delete` is not true | Skip apply; stop and ask. |
| Instruction id in policy `neverDelete` | Do not include it in `--accept`. If that empties the list, stop. |
| `tokens_after` over `token_ceiling` | Stop and ask. |
| "template changed … proposal is stale" | Do not `--force` apply. Re-run `propose-edits`. |
| Non-TTY, no `--accept` | Nothing applied. Ask the human; do not switch to `--all`. |
| TTY offers `[y/N]` / bare `tracegrad apply` | Do not answer. Stop and collect indices, then `--accept` only. |
| `--all` looks convenient | Forbidden unless the policy's `accept` list is exactly every index a human already approved. Prefer `--accept` with that list. |

## Exact CLI

```sh
# Harness path — only after human-typed or policy-listed indices replace the placeholder:
tracegrad apply --accept <HUMAN_OR_POLICY_INDICES>
tracegrad apply --run-id run-0001 --accept <HUMAN_OR_POLICY_INDICES>
tracegrad apply --project-root . --base-directory . --accept <HUMAN_OR_POLICY_INDICES>
# Explicit user request to revert only:
tracegrad apply --revert
# Forbidden for the harness agent:
# tracegrad apply
# (do not answer interactive [y/N])
# tracegrad apply --all
```

Related, not this skill's job:

```sh
tracegrad status
tracegrad trends
```
