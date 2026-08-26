"""The tracegrad command line.

Commands split along the one boundary that matters: everything except ``apply``
reads and analyses, and ``apply`` writes.  ``run`` is the whole pipeline;
``attribute`` and ``propose`` exist so a slow, paid stage can be run once and
reused — attribution results are cached by instrument, so ``propose`` after
``attribute`` costs one synthesis call, not another pass over the batch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence, TextIO

from .aggregate import GAP_LEDGER_FILENAME, GapLedger, aggregate
from .apply import (
    ApplyError,
    Proposal,
    apply_proposal,
    is_stale,
    latest_run_id,
    load_proposal,
    mark_stale,
    revert,
    review_cards,
)
from .attribute import (
    AttributionError,
    LLMError,
    attribute_batch,
    resolve_attribution_backend,
)
from .config import ConfigError, load_config
from .distill import DistillConfig, DistillError, distill_batch, render_manifest_prompt, store_batch
from .gates import REJECTION_MEMORY_FILENAME, RejectionMemory, measure_tokens
from .ingest import IngestError, ingest_traces
from .inventory import InventoryError, build_inventory
from .pipeline import (
    RUN_LEDGER_FILENAME,
    PipelineError,
    estimate_run,
    load_manifest,
    run_pipeline,
    verdict_history,
)
from .schema import Report
from .state import StateError, initialize, load_jsonl
from .synthesize import SynthesisError
from .trends import compare, convergence, format_trend, hysteresis

DEFAULT_RUN_ID_PREFIX = "run"


def _next_run_id(project_root: str | Path) -> str:
    """Sequential, sortable run ids — no clock, so runs stay reproducible."""

    layout = initialize(project_root)
    existing = [
        path.name
        for path in layout.runs.glob(f"{DEFAULT_RUN_ID_PREFIX}-*")
        if path.is_dir()
    ]
    numbers = []
    for name in existing:
        suffix = name.rsplit("-", 1)[-1]
        if suffix.isdigit():
            numbers.append(int(suffix))
    return f"{DEFAULT_RUN_ID_PREFIX}-{max(numbers, default=0) + 1:04d}"


def _reports(project_root: str | Path) -> list[Report]:
    layout = initialize(project_root)
    reports: list[Report] = []
    for path in sorted(layout.reports.glob("*.json")):
        try:
            reports.append(Report.model_validate_json(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return reports


def command_init(args: argparse.Namespace, out: TextIO) -> int:
    layout = initialize(args.project_root)
    print(f"initialized {layout.root}", file=out)
    print("state is git-ignored by a nested .gitignore", file=out)
    rc_path = Path(args.project_root) / ".tracegradrc"
    if not rc_path.exists():
        print(f"no {rc_path.name} found; defaults apply", file=out)
    return 0


def command_run(args: argparse.Namespace, out: TextIO) -> int:
    if args.estimate:
        estimate = estimate_run(
            args.traces,
            args.manifest,
            project_root=args.project_root,
            base_directory=args.base_directory,
        )
        print(estimate.render(), file=out)
        return 0

    run_id = args.run_id or _next_run_id(args.project_root)
    result = run_pipeline(
        args.traces,
        args.manifest,
        run_id=run_id,
        project_root=args.project_root,
        base_directory=args.base_directory,
        token_ceiling=args.token_ceiling,
        session_id=args.session_id,
        jobs=args.jobs,
    )

    print(f"run {result.run_id}", file=out)
    print(
        f"  {len(result.ingest.traces)} traces analysed, "
        f"{len(result.ingest.dropped)} dropped",
        file=out,
    )
    for reason, count in sorted(result.ingest.dropped_reasons.items()):
        print(f"    {count} x {reason}", file=out)
    if result.attribution:
        rate = result.attribution.health.agreement_rate
        print(
            f"  attribution coverage {result.attribution.coverage:.0%}, "
            f"{result.attribution.cache_hits} cached"
            + (f", blinded agreement {rate:.0%}" if rate is not None else ""),
            file=out,
        )
    for warning in result.warnings:
        print(f"  warning: {warning}", file=out)

    print(f"  themes ({result.aggregation.denominator} traces):", file=out)
    for theme in result.aggregation.themes:
        print(
            f"    {theme.theme}: {theme.numerator}/{theme.denominator}"
            + ("  [gap]" if theme.is_gap else ""),
            file=out,
        )

    proposal = result.proposal
    if proposal is None or not proposal.edits:
        print("  no edits proposed — that is a valid outcome", file=out)
        if result.synthesis and result.synthesis.autopsy_path:
            print(f"  dropped proposals: {result.synthesis.autopsy_path}", file=out)
        return 0

    print(f"  {len(proposal.edits)} edit(s) proposed; review with: tracegrad apply", file=out)
    for card in review_cards(proposal):
        print(card.render(), file=out)
    return 0


def command_attribute(args: argparse.Namespace, out: TextIO) -> int:
    config = load_config(args.project_root)
    manifest = load_manifest(args.manifest)
    rendered = render_manifest_prompt(manifest, args.base_directory)
    inventory = build_inventory(rendered)
    ingested = ingest_traces(args.traces, manifest)
    distill_config = DistillConfig()
    distilled = distill_batch(ingested.traces, distill_config)
    layout = initialize(args.project_root)
    store_batch(layout, distilled)
    backend = resolve_attribution_backend(config)
    run = attribute_batch(
        distilled,
        inventory,
        backend,
        project_root=layout,
        distill_config_hash=distill_config.config_hash,
        min_coverage=float(config.minCoverage),
    )
    aggregation = aggregate(run.results, denominator=len(distilled))
    print(f"attributed {len(run.results)}/{run.denominator} traces", file=out)
    for theme in aggregation.themes:
        print(
            f"  {theme.theme}: {theme.numerator}/{theme.denominator}"
            + ("  [gap]" if theme.is_gap else ""),
            file=out,
        )
    return 0


def command_propose(args: argparse.Namespace, out: TextIO) -> int:
    """Run the pipeline against cached attributions and write a proposal."""

    return command_run(args, out)


def command_trends(args: argparse.Namespace, out: TextIO) -> int:
    config = load_config(args.project_root)
    reports = _reports(args.project_root)
    if len(reports) < 2:
        print("need at least two runs before trends mean anything", file=out)
        return 0
    report = compare(
        reports[-2].clusters,
        reports[-1].clusters,
        min_effect=float(config.minEffect),
    )
    for result in report.results:
        print(format_trend(result), file=out)
    if report.regressed_elsewhere:
        print(
            "guardrail: themes that regressed without being targeted: "
            + ", ".join(report.regressed_elsewhere),
            file=out,
        )
    print("trends are advisory: tracegrad never reverts on its own", file=out)
    return 0


def _selected_indices(
    args: argparse.Namespace, proposal: Proposal, out: TextIO
) -> list[int] | None:
    """The edits the human chose, or ``None`` when they were never asked.

    The distinction matters: an empty selection is a decision the memory gate
    should remember, and "nobody was asked" is not.  Conflating them lets one
    piped invocation mark every proposed edit as human-rejected.
    """

    if args.all:
        return list(range(len(proposal.edits)))
    if args.accept is not None:
        indices = []
        for token in args.accept.split(","):
            token = token.strip()
            if token.isdigit():
                indices.append(int(token))
        return indices
    if not sys.stdin.isatty():
        print(
            "no selection given and stdin is not a terminal; nothing applied.\n"
            "  pass --accept <indices> or --all to decide non-interactively",
            file=out,
        )
        return None
    selected: list[int] = []
    for card in review_cards(proposal):
        print(card.render(), file=out)
        answer = input("accept this edit? [y/N] ").strip().lower()
        if answer in {"y", "yes"}:
            selected.append(card.index)
    return selected


def command_apply(args: argparse.Namespace, out: TextIO) -> int:
    run_id = args.run_id or latest_run_id(args.project_root)
    if run_id is None:
        print("no proposal to apply; run tracegrad run first", file=out)
        return 1

    if args.revert:
        template = revert(
            args.project_root, run_id, base_directory=args.base_directory, force=args.force
        )
        print(f"reverted {template} from the snapshot taken for {run_id}", file=out)
        return 0

    proposal = load_proposal(args.project_root, run_id)
    if is_stale(proposal, base_directory=args.base_directory):
        mark_stale(args.project_root, run_id)
        print(
            f"{proposal.template_file} changed since run {run_id}; "
            "the proposal is stale — re-run tracegrad",
            file=out,
        )
        return 1
    if not proposal.edits:
        print(f"run {run_id} proposed no edits", file=out)
        return 0

    selected = _selected_indices(args, proposal, out)
    if selected is None:
        return 1
    result = apply_proposal(
        args.project_root,
        proposal,
        selected,
        base_directory=args.base_directory,
    )
    # Only a real decision is remembered.
    memory = RejectionMemory(
        initialize(args.project_root).ledgers / REJECTION_MEMORY_FILENAME
    )
    for edit in result.rejected:
        memory.record_rejection(edit, run_id=run_id)

    if result.unchanged:
        print("nothing applied", file=out)
        return 0
    print(
        f"applied {len(result.accepted)} edit(s) to {result.template_file}\n"
        f"  new prompt hash: {result.applied_prompt_hash}\n"
        f"  snapshot: {result.snapshot}",
        file=out,
    )
    return 0


def command_status(args: argparse.Namespace, out: TextIO) -> int:
    layout = initialize(args.project_root)
    config = load_config(args.project_root)
    history = load_jsonl(layout.ledgers / RUN_LEDGER_FILENAME)
    print(f"state: {layout.root}", file=out)
    print(f"runs recorded: {len(history)}", file=out)

    if args.manifest:
        manifest = load_manifest(args.manifest)
        rendered = render_manifest_prompt(manifest, args.base_directory)
        inventory = build_inventory(rendered)
        print(
            f"prompt: {len(inventory)} instructions, "
            f"{measure_tokens(rendered.text)} tokens (proxy count)",
            file=out,
        )

    reports = _reports(args.project_root)
    if len(reports) >= 2:
        report = compare(
            reports[-2].clusters, reports[-1].clusters, min_effect=float(config.minEffect)
        )
        print("trends since the previous run:", file=out)
        for result in report.results:
            print("  " + format_trend(result), file=out)

    gaps = GapLedger(layout.ledgers / GAP_LEDGER_FILENAME).state()
    if gaps:
        print("gap ledger:", file=out)
        for gap in gaps.values():
            print(
                f"  {gap.theme}: {gap.status}, seen in {len(gap.runs)} run(s), "
                f"{gap.observations} observation(s)",
                file=out,
            )

    regressed = [
        theme
        for theme, verdicts in sorted(verdict_history(args.project_root).items())
        if hysteresis(verdicts)
    ]
    if regressed:
        print("regressed on consecutive runs: " + ", ".join(regressed), file=out)
        print("  advisory only — review these yourself; nothing is reverted", file=out)

    converged, streak = convergence(history, required_runs=int(config.convergenceRuns))
    if converged:
        print(f"converged: {streak} consecutive runs proposed nothing", file=out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tracegrad",
        description="Offline, evidence-gated system-prompt optimization from traces.",
    )
    # The same globals are accepted before or after the subcommand, because
    # "tracegrad init --project-root x" is what everyone types first.  The
    # subcommand copies suppress their defaults so they never clobber a value
    # given ahead of the subcommand.
    parser.add_argument("--project-root", default=".", help="where .tracegrad/ lives")
    parser.add_argument(
        "--base-directory", default=".", help="where the manifest's paths resolve from"
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--project-root", default=argparse.SUPPRESS)
    common.add_argument("--base-directory", default=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser(
        "init", parents=[common], help="create the local state directory"
    )
    init_parser.set_defaults(handler=command_init)

    for name, handler, help_text in (
        ("run", command_run, "analyse a batch and propose edits"),
        ("propose", command_propose, "propose edits, reusing cached attributions"),
    ):
        run_parser = subparsers.add_parser(name, parents=[common], help=help_text)
        run_parser.add_argument("--traces", required=True, help="JSONL trace export")
        run_parser.add_argument("--manifest", required=True, help="run manifest JSON")
        run_parser.add_argument("--run-id", default=None)
        run_parser.add_argument("--session-id", default=None)
        run_parser.add_argument(
            "--token-ceiling", type=int, default=None, help="prompt token budget ceiling"
        )
        run_parser.add_argument(
            "--jobs",
            type=int,
            default=None,
            help="attribute this many traces concurrently (default 1)",
        )
        run_parser.add_argument(
            "--estimate", action="store_true", help="preview cost without calling a model"
        )
        run_parser.set_defaults(handler=handler)

    attribute_parser = subparsers.add_parser(
        "attribute", parents=[common], help="attribute a batch only"
    )
    attribute_parser.add_argument("--traces", required=True)
    attribute_parser.add_argument("--manifest", required=True)
    attribute_parser.set_defaults(handler=command_attribute)

    trends_parser = subparsers.add_parser(
        "trends", parents=[common], help="compare the last two runs"
    )
    trends_parser.set_defaults(handler=command_trends)

    apply_parser = subparsers.add_parser(
        "apply", parents=[common], help="review and accept proposed edits"
    )
    apply_parser.add_argument("--run-id", default=None)
    apply_parser.add_argument("--accept", default=None, help="comma-separated edit indices")
    apply_parser.add_argument("--all", action="store_true", help="accept every proposed edit")
    apply_parser.add_argument("--revert", action="store_true", help="restore the snapshot")
    apply_parser.add_argument(
        "--force",
        action="store_true",
        help="revert even though the template changed after it was applied",
    )
    apply_parser.set_defaults(handler=command_apply)

    status_parser = subparsers.add_parser(
        "status", parents=[common], help="budget, trends, and ledgers"
    )
    status_parser.add_argument("--manifest", default=None)
    status_parser.set_defaults(handler=command_status)

    return parser


def main(argv: Sequence[str] | None = None, out: TextIO | None = None) -> int:
    stream = out or sys.stdout
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args, stream))
    except (
        ApplyError,
        AttributionError,
        ConfigError,
        DistillError,
        IngestError,
        InventoryError,
        LLMError,
        PipelineError,
        StateError,
        SynthesisError,
    ) as exc:
        print(f"tracegrad: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
