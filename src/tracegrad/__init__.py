"""tracegrad — evidence-gated system-prompt optimization from traces.

The package is usable as a library as well as a CLI.  The names re-exported
here are the stable surface; everything else is an implementation detail that
may move between versions.
"""

from __future__ import annotations

from .aggregate import Aggregation, GapLedger, ThemeStat, aggregate
from .apply import Proposal, apply_proposal, revert, review_cards
from .config import TracegradConfig, load_config
from .distill import DistilledTrace, distill_batch, render_template
from .edits import apply_edits, resolve_edits
from .gates import GateConfig, GateOutcome, run_gates
from .ingest import IngestResult, ingest_traces
from .inventory import Instruction, Inventory, build_inventory
from .pipeline import RunResult, estimate_run, run_pipeline
from .schema import (
    AttributionResult,
    Cluster,
    Edit,
    Manifest,
    Report,
    StepVerdict,
    Trace,
    Verdict,
)
from .trends import TrendReport, TrendResult, compare

__version__ = "0.1.0"

__all__ = [
    "Aggregation",
    "AttributionResult",
    "Cluster",
    "DistilledTrace",
    "Edit",
    "GapLedger",
    "GateConfig",
    "GateOutcome",
    "IngestResult",
    "Instruction",
    "Inventory",
    "Manifest",
    "Proposal",
    "Report",
    "RunResult",
    "StepVerdict",
    "ThemeStat",
    "Trace",
    "TracegradConfig",
    "TrendReport",
    "TrendResult",
    "Verdict",
    "__version__",
    "aggregate",
    "apply_edits",
    "apply_proposal",
    "build_inventory",
    "compare",
    "distill_batch",
    "estimate_run",
    "ingest_traces",
    "load_config",
    "render_template",
    "resolve_edits",
    "review_cards",
    "revert",
    "run_gates",
    "run_pipeline",
]
