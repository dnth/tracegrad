---
name: import-traces
description: "Adapt-in a user-named store export to Tracegrad JSONL. Invoke for import traces."
---

# Import traces

JSONL shape and ingest drops: [jsonl-contract.md](jsonl-contract.md). Adapter lives beside the user repo, not under `src/tracegrad/`. This skill does not run analysis or apply.

## Steps

1. **Project state.** Run `tracegrad init` if `.tracegrad/` is missing.

   Done: `.tracegrad/` exists (init exit 0, or the directory was already there).

2. **Source.** Use only the export path, directory, or command the user named. Do not scrape an unnamed production API.

   Done: that source path is recorded.

3. **Sidecar.** Copy [`../examples/sidecar-adapt-in.py`](../examples/sidecar-adapt-in.py) beside the user repo (or edit their existing adapter). Fill `FIELD_MAP` (Tracegrad field → foreign dotted path): vendor id → `trace_id`, prompt/response → `input`/`output`, judge score + rationale, prompt version → `prompt_hash`. Then:

   ```sh
   python sidecar-adapt-in.py --source /path/to/user-export --out batch.jsonl
   ```

   Hand-write JSONL only when the batch is tiny **and** the user asked. Unknown store format → stop and ask; do not add a connector under `src/tracegrad/`.

   Done: sidecar exit 0, `batch.jsonl` exists, stderr reports `wrote N traces` with N > 0.

4. **Sanity.** Check line count, unique `trace_id`, rationale usable-length, and `prompt_hash` values (see [jsonl-contract.md](jsonl-contract.md)). Deduplicate in the adapter. Fix the map rather than pad fake rationales. Several hashes → warn and split batches if the user wants every version.

   Done: a short note lists N, the source, and the `prompt_hash` values seen. Empty export → stop here; do not call `tracegrad run`.

5. **Handoff.** Give the JSONL path to `propose-edits` or `next-batch`.

   Done: JSONL path is in the reply. This skill ran no `tracegrad` command except `init`.
