# Tracegrad harness skill pack

A verb-first skill pack so Claude Code or Pi can drive Tracegrad from
**outside** the Python package. Core stays lean and harness-driven. This
directory is documentation and examples, not a product surface and not a
dependency.

Kitaru is not in core. Do not add Kitaru code, deps, or docs from this pack.
Adapters live beside the user's repo (sidecar), never under `src/tracegrad/`.

## The loop

1. **Import traces.** A sidecar adapt-in maps the user's existing store into
   Tracegrad JSONL. Their pipeline does not change. Tracegrad never learns the
   stack.
2. **Propose edits.** `tracegrad run --estimate`, then `tracegrad run`. Analysis
   writes cards and a proposal under `.tracegrad/`. It does **not** write the
   prompt.
3. **Review edits.** Default: stop and ask the human. Apply only when a local
   policy file is present **and** clearly permits it. Missing or ambiguous
   policy → stop and ask. Never invent `--accept` indices.
4. **Export prompt.** After a successful `tracegrad apply`, a sidecar adapt-out
   copies the written template back to the path the user's app actually loads
   (no-op if that path already *is* the manifest `template_file`).
5. **Next batch.** Thin conductor: import → estimate → propose → review →
   export → `tracegrad status` / `tracegrad trends`. Unattended apply stays
   off unless the policy file both exists and permits it.

The existing gate is unchanged: `tracegrad run` never writes the prompt; only
`tracegrad apply` does, and only after a human or an explicit policy accept.

## Skills

| Skill | Folder | Does |
| --- | --- | --- |
| import traces | [`import-traces/`](import-traces/SKILL.md) | Adapt-in → JSONL |
| propose edits | [`propose-edits/`](propose-edits/SKILL.md) | Estimate + run (or attribute + propose); stop with cards |
| review edits | [`review-edits/`](review-edits/SKILL.md) | Show cards; apply only if policy allows |
| export prompt | [`export-prompt/`](export-prompt/SKILL.md) | Adapt-out after apply |
| next batch | [`next-batch/`](next-batch/SKILL.md) | Conduct the loop; stop at review unless policy permits apply |

## Policy file

The harness agent may read a project-local policy file (suggested name:
`tracegrad-apply-policy.toml` at the project root, or a path the user names).
**Tracegrad core does not load this file.** `.tracegradrc` remains the only
config core reads (`neverDelete`, coverage, harness presets). The policy file
is an extra, agent-side gate on whether the skill may invoke `tracegrad apply`.

Shape (see [`examples/policy.commented.toml`](examples/policy.commented.toml)):

- `unattended_apply` — default **false**. Off means stop and ask.
- `accept` — comma-style list of card indices for `tracegrad apply --accept`.
  Empty, omitted, or guessed → do not apply.
- `allow_delete` — default **false**. Skip or refuse `DELETE` edits.
- `neverDelete` — instruction ids the agent must not apply, even if they
  survived core gates. Complements `.tracegradrc`; does not replace it.
- `token_ceiling` — if the proposal's `tokens_after` would exceed this, stop
  and ask rather than apply.

If the file is missing, `unattended_apply` is false, `accept` is empty, or any
rule is ambiguous: **stop and ask**. Never pass `--all` unless the policy
explicitly lists every index a human already named.

## Adapters stay outside core

Copy the stubs in [`examples/`](examples/) next to the user's repo:

- `sidecar-adapt-in.py` — foreign traces → JSONL contract
- `sidecar-adapt-out.py` — applied template → user path

Do not vendor these into `src/tracegrad/`. Do not add a Kitaru (or any other
eval-stack) integration to the package to make import/export "just work".
