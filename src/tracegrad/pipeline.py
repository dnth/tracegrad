"""Run orchestration: the fixed order the stages execute in.

The pipeline is the only place that knows the whole sequence, and it holds the
run lock while it executes.  Each completed stage writes a resume checkpoint, so
a killed run picks up where it stopped instead of re-spending the attribution
budget — the attribution cache does the real work there, and the checkpoint just
records what has already been paid for.

The pipeline never writes the prompt.  It produces a proposal; ``apply`` is the
only writer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from pydantic import ValidationError

from .aggregate import (
    GAP_LEDGER_FILENAME,
    GAP_RETIRED,
    THEME_HISTORY_FILENAME,
    Aggregation,
    GapLedger,
    ThemeHistory,
    aggregate,
)
from .apply import Proposal, build_proposal, current_baseline, save_proposal
from .attribute import (
    ATTRIBUTION_TIER,
    AttributionCache,
    AttributionRun,
    EstimateBackend,
    attribute_batch,
    build_instrument,
    resolve_attribution_backend,
)
from .config import TracegradConfig, load_config
from .distill import (
    DistillConfig,
    DistilledTrace,
    RenderedPrompt,
    distill_batch,
    render_manifest_prompt,
    store_batch,
)
from .gates import REJECTION_MEMORY_FILENAME, GateConfig, RejectionMemory, measure_tokens
from .ingest import IngestResult, ingest_traces
from .inventory import Inventory, build_inventory
from .ports import Backend
from .schema import Manifest, Report, Verdict
from .state import (
    StateLayout,
    StateLock,
    atomic_write_json,
    initialize,
    load_jsonl,
    save_resume_state,
)
from .synthesize import SynthesisResult, resolve_synthesis_backend, synthesize
from .trends import TrendReport, compare, convergence, hysteresis

RUN_LEDGER_FILENAME = "runs.jsonl"
VERDICT_LEDGER_FILENAME = "verdicts.jsonl"


class PipelineError(RuntimeError):
    """A run that cannot proceed."""


def load_manifest(path: str | Path) -> Manifest:
    """Read and validate a run manifest."""

    target = Path(path)
    try:
        return Manifest.model_validate_json(target.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PipelineError(f"could not read manifest {target}: {exc}") from exc
    except ValidationError as exc:
        raise PipelineError(f"invalid manifest {target}: {exc}") from exc


@dataclass(frozen=True)
class Estimate:
    """A pre-flight cost preview: calls and a token proxy, no model contacted."""

    traces: int
    attribution_calls: int
    synthesis_calls: int
    cached_calls: int
    prompt_tokens: int

    @property
    def total_calls(self) -> int:
        return self.attribution_calls + self.synthesis_calls

    def render(self) -> str:
        return (
            f"{self.traces} traces in the batch\n"
            f"{self.attribution_calls} attribution call(s), {self.cached_calls} already cached\n"
            f"{self.synthesis_calls} synthesis call(s)\n"
            f"~{self.prompt_tokens} prompt tokens per attribution call (proxy count)"
        )


@dataclass(frozen=True)
class RunResult:
    """Everything one run produced."""

    run_id: str
    ingest: IngestResult
    inventory: Inventory
    distilled: tuple[DistilledTrace, ...]
    attribution: AttributionRun | None
    aggregation: Aggregation
    synthesis: SynthesisResult | None
    proposal: Proposal | None
    report: Report
    trends: TrendReport | None = None
    converged: bool = False
    warnings: tuple[str, ...] = ()

    @property
    def proposed_edit_count(self) -> int:
        return len(self.proposal.edits) if self.proposal else 0


def estimate_run(
    traces_path: str | Path,
    manifest_path: str | Path,
    *,
    project_root: str | Path = ".",
    base_directory: str | Path = ".",
    backend_for_estimate: object = None,
) -> Estimate:
    """Preview a run's cost without contacting a model.

    The cached count assumes the default attribution backend unless
    ``backend_for_estimate`` names another, because a different backend is a
    different instrument and therefore a different cache.
    """

    manifest = load_manifest(manifest_path)
    rendered = render_manifest_prompt(manifest, base_directory)
    ingested = ingest_traces(traces_path, manifest)
    distill_config = DistillConfig()
    distilled = distill_batch(ingested.traces, distill_config)
    layout = initialize(project_root)

    # The cache is keyed by instrument crossed with the trace, so the estimate
    # has to build the same instrument the run would — comparing bare content
    # addresses reports every batch as uncached, telling the user to re-pay for
    # work already done.
    inventory = build_inventory(rendered)
    backend = backend_for_estimate or EstimateBackend()
    instrument = build_instrument(backend, inventory, distill_config.config_hash)
    cache = AttributionCache(layout)
    cached = sum(1 for item in distilled if cache.get(instrument.cache_key(item)) is not None)
    return Estimate(
        traces=len(ingested.traces),
        attribution_calls=max(0, len(ingested.traces) - cached),
        synthesis_calls=1,
        cached_calls=cached,
        prompt_tokens=measure_tokens(rendered.text),
    )


def _previous_report(layout: StateLayout) -> Report | None:
    reports = sorted(layout.reports.glob("*.json"))
    if not reports:
        return None
    try:
        return Report.model_validate_json(reports[-1].read_text(encoding="utf-8"))
    except (OSError, ValidationError):
        return None


def _run_history(layout: StateLayout) -> list[Mapping[str, object]]:
    return load_jsonl(layout.ledgers / RUN_LEDGER_FILENAME)


def run_pipeline(
    traces_path: str | Path,
    manifest_path: str | Path,
    *,
    run_id: str,
    project_root: str | Path = ".",
    base_directory: str | Path = ".",
    config: TracegradConfig | None = None,
    attribution_backend: Backend | None = None,
    synthesis_backend: Backend | None = None,
    token_ceiling: int | None = None,
    session_id: str | None = None,
    jobs: int | None = None,
) -> RunResult:
    """Execute one full analysis run and persist its proposal and report."""

    settings = config or load_config(base_directory)
    layout = initialize(project_root)
    warnings: list[str] = []

    with StateLock(layout.lock):
        manifest = load_manifest(manifest_path)
        rendered: RenderedPrompt = render_manifest_prompt(manifest, base_directory)
        inventory = build_inventory(rendered)
        save_resume_state(layout, run_id, {"run_id": run_id, "stage": "inventory"})

        previous_report = _previous_report(layout)
        ingested = ingest_traces(
            traces_path,
            manifest,
            previous_judge_fingerprint=_previous_judge_fingerprint(layout),
            canary_scores=manifest.canary_scores,
        )
        if not ingested.traces:
            raise PipelineError("no traces survived ingest; nothing to analyse")
        if ingested.judge_fingerprint_changed:
            warnings.append(
                "judge fingerprint changed since the last run: trends across it are not comparable"
            )
        if ingested.is_mixed_model:
            warnings.append(
                "batch mixes response models: " + ", ".join(sorted(ingested.model_partitions))
            )
        for failure in ingested.canary_failures:
            warnings.append(
                f"judge canary {failure.trace_id} scored {failure.observed} "
                f"against an expected {failure.expected}"
            )
        save_resume_state(layout, run_id, {"run_id": run_id, "stage": "ingest"})

        distill_config = DistillConfig()
        distilled = distill_batch(ingested.traces, distill_config)
        store_batch(layout, distilled)
        by_id: dict[str, DistilledTrace] = {item.trace_id: item for item in distilled}
        save_resume_state(layout, run_id, {"run_id": run_id, "stage": "distill"})

        attribution_client = resolve_attribution_backend(
            settings, override=attribution_backend, on_fallback=warnings.append
        )
        attribution_preset = settings.harness_presets.get(ATTRIBUTION_TIER)
        attribution = attribute_batch(
            distilled,
            inventory,
            attribution_client,
            project_root=layout,
            distill_config_hash=distill_config.config_hash,
            min_coverage=float(settings.minCoverage),
            jobs=jobs if jobs is not None else (attribution_preset.jobs if attribution_preset else 1),
        )
        save_resume_state(layout, run_id, {"run_id": run_id, "stage": "attribute"})

        aggregation = aggregate(attribution.results, denominator=len(distilled))

        history_ledger = ThemeHistory(layout.ledgers / THEME_HISTORY_FILENAME)
        history_ledger.record(aggregation.themes, run_id=run_id, session_id=session_id)

        gaps = GapLedger(layout.ledgers / GAP_LEDGER_FILENAME)
        gaps.record_observations(aggregation.gaps, run_id=run_id, session_id=session_id)
        eligible_slugs = {gap.theme for gap in gaps.eligible()}
        for gap in aggregation.gaps:
            state = gaps.state().get(gap.theme)
            if state is not None and state.can_graduate() and not state.is_graduated:
                gaps.graduate(gap.theme, run_id=run_id)
                eligible_slugs.add(gap.theme)
        eligible_gaps = tuple(gap for gap in aggregation.gaps if gap.theme in eligible_slugs)
        save_resume_state(layout, run_id, {"run_id": run_id, "stage": "aggregate"})

        synthesis_client = resolve_synthesis_backend(
            settings, override=synthesis_backend, on_fallback=warnings.append
        )
        memory = RejectionMemory(layout.ledgers / REJECTION_MEMORY_FILENAME)
        gate_config = GateConfig(
            token_ceiling=token_ceiling,
            never_delete=tuple(settings.neverDelete),
        )
        # G6 counts distinct sessions, not traces: a theme touching two traces
        # of one batch is one observation, not two independent ones.
        support = history_ledger.distinct_sources()
        synthesis = synthesize(
            inventory,
            aggregation,
            synthesis_client,
            attributions=attribution.results,
            distilled=by_id,
            memory=memory,
            support=support,
            config=gate_config,
            project_root=layout,
            run_id=run_id,
            eligible_gaps=eligible_gaps,
        )
        save_resume_state(layout, run_id, {"run_id": run_id, "stage": "synthesize"})

        proposal = build_proposal(
            run_id=run_id,
            template_file=manifest.template_file,
            prompt=rendered.text,
            outcome=synthesis.outcome,
            attributions=attribution.results,
            distilled=by_id,
            theme_map=aggregation.theme_map,
        )
        save_proposal(layout, proposal)

        baseline = current_baseline(layout) or rendered.prompt_hash
        report = Report(
            applied_prompt_hash=baseline,
            clusters=list(aggregation.clusters),
            instrument_fingerprint=attribution.instrument.measurement_fingerprint,
        )
        # Re-running the same batch measures nothing new, and writing the report
        # again would leave `trends` comparing a run against itself.
        measured_something_new = previous_report is None or (
            previous_report.clusters != report.clusters
            or previous_report.instrument_fingerprint != report.instrument_fingerprint
        )
        if measured_something_new:
            atomic_write_json(layout.reports / f"{run_id}.json", report.model_dump(mode="json"))
        else:
            warnings.append(
                "this batch measured the same counts as the previous run; "
                "no new report was written"
            )

        trends: TrendReport | None = None
        if previous_report is not None:
            previous_instrument = previous_report.instrument_fingerprint
            if (
                previous_instrument is not None
                and previous_instrument != report.instrument_fingerprint
            ):
                # Two batches measured with different instruments are not
                # comparable.  Say so instead of differencing them anyway.
                warnings.append(
                    "the attribution instrument changed since the previous run: "
                    "trends across it are not comparable and were not computed"
                )
            else:
                targeted = _targeted_themes(layout)
                trends = compare(
                    previous_report.clusters,
                    report.clusters,
                    targeted=targeted,
                    min_effect=float(settings.minEffect),
                )
                _retire_improved_gaps(gaps, trends, run_id)
                _record_verdicts(layout, trends, run_id)
                for theme in _persistently_regressed(layout, settings.convergenceRuns):
                    warnings.append(
                        f"{theme} has regressed on consecutive runs — worth a look "
                        "(tracegrad does not revert on its own)"
                    )

        history = _run_history(layout)
        converged, _ = convergence(
            [*history, {"proposed": len(proposal.edits)}],
            required_runs=int(settings.convergenceRuns),
        )
        _record_run(
            layout, run_id, proposal, attribution, warnings, ingested.judge_fingerprint
        )
        save_resume_state(layout, run_id, {"run_id": run_id, "stage": "complete"})

    return RunResult(
        run_id=run_id,
        ingest=ingested,
        inventory=inventory,
        distilled=tuple(distilled),
        attribution=attribution,
        aggregation=aggregation,
        synthesis=synthesis,
        proposal=proposal,
        report=report,
        trends=trends,
        converged=converged,
        warnings=tuple(warnings),
    )


def _retire_improved_gaps(gaps: GapLedger, trends: TrendReport, run_id: str) -> None:
    """Retire gap themes whose trend improved or eliminated them.

    This is the only automatic retirement path the spec allows, and it is still
    not a revert: retiring a gap only stops tracegrad proposing a new
    instruction for a failure that has stopped happening.
    """

    known = gaps.state()
    for result in (*trends.improved, *trends.eliminated):
        state = known.get(result.theme)
        if state is not None and state.status != GAP_RETIRED:
            gaps.retire(result.theme, run_id=run_id, reason="improved-trend")


def _record_verdicts(layout: StateLayout, trends: TrendReport, run_id: str) -> None:
    """Append this run's per-theme verdicts, the input hysteresis reads."""

    from .state import append_jsonl

    for result in trends.results:
        append_jsonl(
            layout.ledgers / VERDICT_LEDGER_FILENAME,
            {"run_id": run_id, "theme": result.theme, "verdict": result.verdict.value},
        )


def verdict_history(project_root: str | Path | StateLayout) -> dict[str, list[Verdict]]:
    """Per theme, the verdicts it has received, oldest first."""

    layout = initialize(project_root)
    history: dict[str, list[Verdict]] = {}
    for record in load_jsonl(layout.ledgers / VERDICT_LEDGER_FILENAME):
        theme = record.get("theme")
        verdict = record.get("verdict")
        if not isinstance(theme, str) or not isinstance(verdict, str):
            continue
        try:
            history.setdefault(theme, []).append(Verdict(verdict))
        except ValueError:
            continue
    return history


def _persistently_regressed(layout: StateLayout, required: int) -> tuple[str, ...]:
    """Themes that have regressed on enough consecutive runs to be worth raising.

    One bad batch is a batch; the hysteresis bar is what keeps this from crying
    wolf every run.  It still only warns — nothing is reverted automatically.
    """

    return tuple(
        theme
        for theme, verdicts in sorted(verdict_history(layout).items())
        if hysteresis(verdicts, required=max(2, int(required)))
    )


def _previous_judge_fingerprint(layout: StateLayout) -> str | None:
    for record in reversed(_run_history(layout)):
        fingerprint = record.get("judge_fingerprint")
        if isinstance(fingerprint, str):
            return fingerprint
    return None


def _targeted_themes(layout: StateLayout) -> tuple[str, ...]:
    from .apply import applied_history

    themes: list[str] = []
    for record in applied_history(layout):
        if record.get("event") != "applied":
            continue
        for edit in record.get("accepted", []) or []:
            if isinstance(edit, dict):
                metric = edit.get("watch_metric") or edit.get("covers_theme")
                if isinstance(metric, str):
                    themes.append(metric)
    return tuple(dict.fromkeys(themes))


def _record_run(
    layout: StateLayout,
    run_id: str,
    proposal: Proposal,
    attribution: AttributionRun | None,
    warnings: Sequence[str],
    judge_fingerprint: str | None = None,
) -> None:
    from .state import append_jsonl

    append_jsonl(
        layout.ledgers / RUN_LEDGER_FILENAME,
        {
            "run_id": run_id,
            "proposed": len(proposal.edits),
            "judge_fingerprint": judge_fingerprint,
            "coverage": attribution.coverage if attribution else None,
            "cache_hits": attribution.cache_hits if attribution else 0,
            "agreement_rate": attribution.health.agreement_rate if attribution else None,
            "warnings": list(warnings),
        },
    )
