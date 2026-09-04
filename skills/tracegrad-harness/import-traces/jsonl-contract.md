# Tracegrad JSONL contract

Ingest reads **one JSON object per line**. Extra keys are rejected (`extra: forbid`). A manifest is a separate JSON file for `tracegrad run --manifest`, not a JSONL row.

## Object shape

```json
{
  "trace_id": "t-001",
  "input": "user or task text",
  "output": "model response",
  "judge": {"score": 0.4, "rationale": "why this score, in words"},
  "prompt_hash": "sha256:… of the prompt version that produced this trace"
}
```

| Field | Rule |
| --- | --- |
| `trace_id` | Non-empty string. Unique within the file. Later duplicates drop as `duplicate-trace-id`. |
| `input` / `output` | Strings (may be empty). `output` is what violations quote. |
| `judge.score` | Number in `[0.0, 1.0]`. |
| `judge.rationale` | Non-empty string. Ingest drops rationales with fewer than **24** usable characters (`rationale-below-quality-floor`). Usable = stripped length ≥ 24 **and** at least one letter. A score without a real rationale is not a batch. |
| `prompt_hash` | Non-empty string identifying the prompt version. Mixed hashes: only the dominant partition is kept (`prompt-hash-partition`). |
| `meta` | Optional. Only `meta.model` is allowed. Mixed models are reported, not dropped. |

Do not emit `null` for required strings. Do not put the judge score in a different shape (`label`, `pass`, nested vendor blobs) — map those in the sidecar.

`FIELD_MAP` in the adapt-in sidecar is **Tracegrad field → foreign dotted path** (e.g. `trace_id` → vendor id path, `input`/`output` → prompt/response paths).
