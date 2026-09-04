---
name: export-prompt
description: "Adapt-out the applied template to a user-named path. Invoke for export after apply. Not apply."
---

# Export prompt

Sidecar adapt-out: copy the template `tracegrad apply` already wrote to the path the user's app loads. Apply is [`../review-edits/SKILL.md`](../review-edits/SKILL.md). Export is not a Tracegrad subcommand.

Skip (successful no-op) when the manifest `template_file` **is** the user path. Do not copy a proposed, unapplied template.

## Steps

1. **Confirm apply for this run.** Look for a new prompt hash on apply stdout, a new line in `.tracegrad/ledgers/applied.jsonl`, and a snapshot under `.tracegrad/snapshots/`.

   Done: at least one of those exists. If none, stop and send the user to `review-edits`.

2. **Source.** Resolve manifest `template_file` against the same `--base-directory` used at apply (`tracegrad apply --help`). The source is the path apply printed (`applied … to <template>`).

   Done: that file exists. If it is missing or was edited after apply, stop — do not copy an out-of-band file.

3. **Destination.** Use only a path the user named (config, flag, or existing sidecar default). Do not guess a production path.

   Done: destination path is explicit.

4. **Adapt-out.** Copy [`../examples/sidecar-adapt-out.py`](../examples/sidecar-adapt-out.py) beside the user repo if needed, then:

   ```sh
   python sidecar-adapt-out.py \
     --from path/from/manifest/prompt.md \
     --to /path/the/user/app/loads/prompt.md
   ```

   Overwrite only that destination. Same resolved `--from` and `--to` is a no-op (exit 0). Do not vendor a prompt store into `src/tracegrad/`.

   Done: sidecar exit 0, and either the destination bytes match the source or the adapter printed the same-path no-op. Report source, destination, and that core was not modified.
