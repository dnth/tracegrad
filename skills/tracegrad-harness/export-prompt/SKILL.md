---
name: export-prompt
description: After a successful tracegrad apply, copy the written template back to the user path via a sidecar adapt-out. Use when asked to export, sync, or deploy the applied prompt. Does not apply.
---

# Export prompt

Sidecar adapt-out: after `tracegrad apply` has already written the template,
copy or write that file to the path the user's app actually loads.

This skill does **not** apply edits. If apply has not happened, stop and send
the user to `review-edits`.

## When to use

- Apply succeeded and the production prompt is a *different* path than the
  manifest `template_file`.
- The user asks to export, sync, or copy the applied prompt back.
- `next-batch` reaches the export step after a permitted apply.

Skip (successful no-op) when the manifest path **is** the user path. Do not
use this skill to bypass the apply gate by writing a "proposed" prompt.

## Sidecar adapt-out

Tracegrad writes the file named in the manifest, resolved against
`--base-directory`. The adapter lives beside the user repo. Copy
[`../examples/sidecar-adapt-out.py`](../examples/sidecar-adapt-out.py) and
point `--from` at the applied template and `--to` at the user path.

```sh
python sidecar-adapt-out.py \
  --from path/from/manifest/prompt.md \
  --to /path/the/user/app/loads/prompt.md
```

Do not add an export command to Tracegrad core. Do not vendor the user's
prompt store into `src/tracegrad/`.

## Steps

1. Confirm apply already happened for this run: new prompt hash on stdout, or
   an entry in `.tracegrad/ledgers/applied.jsonl`, plus a snapshot under
   `.tracegrad/snapshots/`. If none, **stop** — export has nothing safe to
   copy.

2. Resolve the source path: manifest `template_file` + `--base-directory`.

3. Resolve the destination: only a path the user named (config, flag, or
   existing sidecar defaults). Do not guess a production path.

4. Run the sidecar. Overwrite only that destination.

5. Report source, destination, and that core was not modified.

## Inputs / outputs

**Inputs**

- Applied template path (manifest `template_file`).
- User destination path.
- Sidecar script (example: `examples/sidecar-adapt-out.py`).
- Optional: `--project-root` / `--base-directory` used during apply, so the
  same file is found.

**Outputs**

- User-path file updated to match the applied template (or no-op if identical
  path / identical bytes).
- No Tracegrad state writes. No second apply.

## Failure modes

| Failure | What to do |
| --- | --- |
| No apply yet | Stop. Do not copy a pre-apply template and call it exported. |
| Stale proposal was refused | Nothing to export. Re-run propose + review. |
| Destination unknown | Stop and ask. |
| Destination outside what the user named | Refuse. |
| Apply succeeded but template hash does not match apply output | Stop; do not overwrite the user path with a file that may have been edited out of band. |
| Urge to `tracegrad apply` "so there is something to export" | Refuse unless `review-edits` policy/human already allowed it. |

## Exact CLI

Export is not a Tracegrad subcommand. Apply (already done, other skill):

```sh
tracegrad apply --accept 0,2
```

Verify after export, if useful:

```sh
tracegrad status --manifest manifest.json
```

Do not run:

```sh
tracegrad run
tracegrad apply --all
tracegrad apply --revert
```

from this skill.
