---
name: review-edits
description: "Policy gate: show cards, then tracegrad apply --accept <HUMAN_OR_POLICY_INDICES> after a human or policy names those indices."
---

# Review edits

Show the cards from the latest proposal. Default: stop and ask.

## Apply gate

The prompt is written only by:

```sh
tracegrad apply --accept <HUMAN_OR_POLICY_INDICES>
```

after those indices exist (human-typed this turn, or policy `accept` after every check below passes). Replace the placeholder with those integers — it is not a list. Join them with commas (`tracegrad apply --help` for `--run-id` and location flags).

Pair with that path: do not run bare `tracegrad apply` (interactive `[y/N]` per card — do not answer it, do not pipe `yes`). Do not pass `--all`. Do not invent or extend the index list. Do not edit the template by hand. Non-TTY without `--accept` applies nothing (exit 1) — collect indices and retry with `--accept`.

Core does **not** load the apply policy. Core still only reads `.tracegradrc` (`neverDelete`, coverage, harness presets). The policy is an agent-side gate on whether this skill may invoke apply. Path: the file the user named, else `tracegrad-apply-policy.toml` at the project root if present. Shape: [`../examples/policy.commented.toml`](../examples/policy.commented.toml). `accept` is a TOML integer array; pass the same integers as comma-separated `--accept`.

Unattended apply runs only when **every** check holds:

- File exists and parses.
- `unattended_apply` is explicitly `true` (example default is **false**).
- `accept` is a non-empty list of integer indices that exist on **this** proposal. Never invent or extend that list.
- Each selected edit is allowed: not `DELETE` unless `allow_delete = true`; instruction id not in policy `neverDelete` (this list complements rc `neverDelete`; it does not replace it).
- If `token_ceiling` is set, proposal `tokens_after` does not exceed it.
- Indices still match this proposal (policy `run_id` if set; template hash not stale).

Any miss or ambiguity → stop and ask. A human may still name indices for attended `--accept`. Policy missing or `unattended_apply` false blocks only the unattended path.

## Steps

1. **Locate the proposal.** Latest run id is the sorted name under `.tracegrad/runs/*/proposal.json`, or the `--run-id` the user named.

   Done: that `proposal.json` exists. If none, send the user to `propose-edits` (`tracegrad apply` would exit 1).

2. **Show every card.** Index, `ADD`/`REWRITE`/`DELETE`, instruction id, theme, flags, unified diff, verbatim quotes. Prefer stdout from `tracegrad run`. Re-read `proposal.json` if scrollback is gone. Show cards by reading the proposal, not by applying.

   Done: the human has seen every index on this proposal.

3. **Collect indices.**

   - **Attended (default):** list the indices and wait. After the human types which cards to accept, those integers are the list.
   - **Unattended:** read the policy file and run the checks above. The list is policy `accept` unchanged.

   Done: an explicit integer list exists in this turn (human-typed or policy-listed). If not, stop and ask — prompt file unchanged.

4. **Apply** with that list only:

   ```sh
   tracegrad apply --accept <HUMAN_OR_POLICY_INDICES>
   ```

   Stale template (`template changed … proposal is stale`) → do not `--force` apply; re-run `propose-edits`. A policy `neverDelete` hit → drop that index; if the list becomes empty, stop.

   Done: CLI exit 0 and stdout reports accepted count, new prompt hash, and snapshot under `.tracegrad/snapshots/`; or apply was not invoked (stale proposal / empty list) and the prompt file is unchanged.

## Revert

Only on an explicit user request to restore the pre-apply snapshot:

```sh
tracegrad apply --revert
```

`--force` only when they asked to revert despite a later template change. Not part of ordinary review.
