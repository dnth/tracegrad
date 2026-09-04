# Tracegrad harness skill pack

Harness loop for Claude Code / Pi. Core stays lean. Sidecars live beside the user repo, never under `src/tracegrad/`. No Kitaru in this pack.

`tracegrad run` does not write the prompt. Harness apply is only `tracegrad apply --accept <HUMAN_OR_POLICY_INDICES>` after a human or policy names those indices — procedure in [review-edits](review-edits/SKILL.md). Never invent accepts.

CLI flags: `tracegrad <cmd> --help`. Manifest, `.tracegradrc`, and project state: repo README.

## Loop

1. [import-traces](import-traces/SKILL.md) — adapt-in a user-named export → JSONL
2. [propose-edits](propose-edits/SKILL.md) — estimate, then run (or attribute + propose); cards on disk
3. [review-edits](review-edits/SKILL.md) — show cards; `--accept` after human- or policy-named indices
4. [export-prompt](export-prompt/SKILL.md) — adapt-out after apply wrote the template
5. [next-batch](next-batch/SKILL.md) — conductor over 1–4, then `status` / `trends`

## Reach

| Situation | Skill |
| --- | --- |
| import traces, adapt-in, JSONL from a store | [import-traces](import-traces/SKILL.md) |
| estimate, propose, run, attribute | [propose-edits](propose-edits/SKILL.md) |
| cards, review, accept, apply, policy | [review-edits](review-edits/SKILL.md) |
| export / adapt-out the applied prompt | [export-prompt](export-prompt/SKILL.md) |
| next batch, close the loop | [next-batch](next-batch/SKILL.md) |

JSONL ingest rules: [import-traces/jsonl-contract.md](import-traces/jsonl-contract.md).

## Sidecars and policy

Copy [examples/](examples/) beside the user repo:

- `sidecar-adapt-in.py` — `FIELD_MAP` = Tracegrad field → foreign path
- `sidecar-adapt-out.py` — applied template → user path
- `policy.commented.toml` — agent-side apply gate (unattended apply off by default; `accept` is a TOML integer array). Core does not load this file.
