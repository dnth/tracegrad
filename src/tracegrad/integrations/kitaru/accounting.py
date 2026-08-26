"""Source-drop vs batch-drop accounting.  The two tables are never merged."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping

from .mapping import SourceDrop


def format_source_table(
    *,
    sessions_selected: int,
    traces_mapped: int,
    dropped: Iterable[SourceDrop],
    in_batch: int | None = None,
    batch_drops: Mapping[str, int] | None = None,
) -> str:
    """Render the two-table drop report from issue #8."""

    reasons = Counter(drop.reason for drop in dropped)
    lines = [
        f"Sessions selected:              {sessions_selected:>6}",
        f"Traces mapped:                  {traces_mapped:>6}",
    ]
    for reason, count in sorted(reasons.items()):
        lines.append(f"  {reason:<32} {count:>4}")
    if in_batch is not None:
        lines.append(f"In batch:                       {in_batch:>6}")
        for reason, count in sorted((batch_drops or {}).items()):
            lines.append(f"  {reason:<32} {count:>4}")
    return "\n".join(lines)
