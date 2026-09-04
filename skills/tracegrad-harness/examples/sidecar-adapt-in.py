#!/usr/bin/env python3
"""Sidecar adapt-in: map a foreign trace export to Tracegrad JSONL.

Copy this file next to the user repo. Do not move it into src/tracegrad/.
The user pipeline stays unchanged; this adapter is the only place that
learns the foreign field names. This stub is not a vendor integration.

One JSON object per line; ingest forbids extra keys. Required: trace_id,
input, output, judge.score in [0, 1], judge.rationale, prompt_hash.
Optional meta.model. Ingest drops duplicate trace_ids, rationales under
24 usable characters, and non-dominant prompt_hash.

FIELD_MAP (or --map-json) is Tracegrad field → foreign dotted path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

# Tracegrad field → foreign dotted path. Edit per project.
FIELD_MAP: dict[str, str] = {
    "trace_id": "id",
    "input": "prompt",
    "output": "response",
    "judge.score": "score",
    "judge.rationale": "rationale",
    "prompt_hash": "prompt_hash",
    "meta.model": "model",
}


def _get(record: Mapping[str, Any], path: str) -> Any:
    current: Any = record
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def adapt_record(record: Mapping[str, Any], field_map: Mapping[str, str]) -> dict[str, Any]:
    """Map one foreign object to a Tracegrad trace dict. Drop incomplete rows."""

    def mapped(key: str) -> Any:
        source = field_map.get(key)
        if not source:
            return None
        return _get(record, source)

    trace_id = mapped("trace_id")
    rationale = mapped("judge.rationale")
    prompt_hash = mapped("prompt_hash")
    score = mapped("judge.score")
    if trace_id is None or rationale is None or prompt_hash is None or score is None:
        return {}
    try:
        score_f = float(score)
    except (TypeError, ValueError):
        return {}
    if not 0.0 <= score_f <= 1.0:
        return {}

    out: dict[str, Any] = {
        "trace_id": str(trace_id),
        "input": "" if mapped("input") is None else str(mapped("input")),
        "output": "" if mapped("output") is None else str(mapped("output")),
        "judge": {"score": score_f, "rationale": str(rationale)},
        "prompt_hash": str(prompt_hash),
    }
    model = mapped("meta.model")
    if model:
        out["meta"] = {"model": str(model)}
    return out


def iter_records(path: Path) -> list[Mapping[str, Any]]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise SystemExit(f"{path}: expected a JSON array")
        return [row for row in payload if isinstance(row, Mapping)]
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise SystemExit(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="foreign JSON or JSONL export")
    parser.add_argument("--out", required=True, help="Tracegrad JSONL destination")
    parser.add_argument(
        "--map-json",
        default=None,
        help="optional JSON object overriding FIELD_MAP",
    )
    args = parser.parse_args(argv)

    field_map = dict(FIELD_MAP)
    if args.map_json:
        override = json.loads(Path(args.map_json).read_text(encoding="utf-8"))
        if not isinstance(override, dict):
            print("tracegrad sidecar: --map-json must be an object", file=sys.stderr)
            return 1
        field_map.update({str(k): str(v) for k, v in override.items()})

    written = 0
    skipped = 0
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in iter_records(Path(args.source)):
            adapted = adapt_record(record, field_map)
            if not adapted:
                skipped += 1
                continue
            handle.write(json.dumps(adapted, ensure_ascii=False) + "\n")
            written += 1

    print(f"wrote {written} traces to {destination} ({skipped} skipped)", file=sys.stderr)
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
