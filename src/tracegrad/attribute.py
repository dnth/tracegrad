"""Attribution — mapping each judge rationale onto the instruction that caused it.

This is the module that spends the model budget, one call per trace.  Three
things make its output usable downstream:

* **A versioned instrument.**  The cache key covers the model, the prompt, the
  segmenter, the normalizer, and the distill config.  Change any of them and the
  cache misses, because a rate computed with two different instruments is not a
  rate.
* **A canonical theme vocabulary fed forward.**  Themes seen earlier in the
  batch are offered to later calls, so the model reuses slugs instead of
  inventing a synonym per trace.  Aggregation still unifies afterwards; this
  just keeps the space small enough to unify well.
* **A coverage floor.**  If too many traces fail to attribute, the run aborts
  rather than reporting a rate over an unknown denominator.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

from pydantic import ValidationError

from .canonical import canonical_json, content_hash
from .config import TracegradConfig
from .distill import DistilledTrace
from .inventory import Inventory
from .llm import (
    ATTRIBUTION_TIER,
    Backend,
    Completion,
    LLMError,
    parse_json_response,
    resolve_backend,
)
from .normalize import NORMALIZER_VERSION
from .schema import AttributionResult, QuoteSource
from .state import atomic_write_json, initialize

ATTRIBUTION_PROMPT_VERSION = 1
DEFAULT_HEALTH_SAMPLE = 5

SYSTEM_PROMPT = """\
You attribute one graded LLM trace to the system-prompt instructions responsible \
for its judged failure.

You are given the numbered instructions of the system prompt, one trace (input, \
output, judge score, judge rationale), and a list of theme slugs already used in \
this batch.

Return JSON only, matching:
{"violations": [{"instruction_id": "<id>", "theme_slug": "<slug>", "quote": "<verbatim from the OUTPUT>"}],
 "harmful": [{"instruction_id": "<id or null>", "theme_slug": "<slug>", "quote": "<verbatim from input, output, or rationale>"}]}

Rules:
- "violations" are instructions the output disobeyed. Every violation quote MUST \
be copied verbatim from the OUTPUT text.
- "harmful" covers instructions that were followed but caused the failure, and \
failures no instruction covers. Use instruction_id null when no instruction covers it.
- Quotes must be exact substrings of the source you name. Never paraphrase. Never \
invent a quote. If you cannot quote it, omit the entry.
- Reuse an existing theme slug when one fits. Slugs are lowercase-hyphenated.
- An empty result is correct when the trace shows no attributable failure."""


def resolve_attribution_backend(
    config: TracegradConfig,
    *,
    override: Backend | None = None,
    on_fallback: Callable[[str], None] | None = None,
) -> Backend:
    """Resolve the attribution-tier backend.

    The determinism boundary means ``llm`` has exactly two importers, so the
    orchestrator asks this module for its backend rather than reaching past it.
    """

    return resolve_backend(
        config, ATTRIBUTION_TIER, override=override, on_fallback=on_fallback
    )


class AttributionError(RuntimeError):
    """Attribution failed in a way that invalidates the batch."""


class CoverageError(AttributionError):
    """Too few traces attributed to report a rate honestly."""


@dataclass(frozen=True)
class Instrument:
    """Every version that can change an attribution, folded into one address."""

    backend: str
    model: str | None
    prompt_version: int = ATTRIBUTION_PROMPT_VERSION
    segmenter_version: int = 0
    normalizer_version: int = NORMALIZER_VERSION
    distill_config_hash: str = ""
    inventory_hash: str = ""
    temperature: float | None = None
    reasoning_effort: str | None = None

    @property
    def measurement_fingerprint(self) -> str:
        """Everything about HOW a batch was measured, excluding WHAT was measured.

        The prompt is supposed to change between batches — that is the point of
        the tool — so the inventory is deliberately not part of this.  Two
        reports sharing this fingerprint were measured the same way and may be
        differenced; two that do not, may not.
        """

        return content_hash(
            {
                "backend": self.backend,
                "model": self.model,
                "prompt_version": self.prompt_version,
                "segmenter_version": self.segmenter_version,
                "normalizer_version": self.normalizer_version,
                "distill_config_hash": self.distill_config_hash,
                "temperature": self.temperature,
                "reasoning_effort": self.reasoning_effort,
            }
        )

    @property
    def fingerprint(self) -> str:
        """The full cache identity: the measurement, plus the prompt measured."""

        return content_hash(
            {
                "measurement": self.measurement_fingerprint,
                "inventory_hash": self.inventory_hash,
            }
        )

    def cache_key(self, distilled: DistilledTrace) -> str:
        return content_hash(
            {"instrument": self.fingerprint, "trace": distilled.content_address}
        )


def build_instrument(
    backend: Backend, inventory: Inventory, distill_config_hash: str, model: str | None = None
) -> Instrument:
    """Describe the exact instrument a batch is about to be measured with."""

    return Instrument(
        backend=getattr(backend, "name", "unknown"),
        model=model or getattr(backend, "model", None),
        temperature=getattr(backend, "temperature", None),
        reasoning_effort=getattr(backend, "reasoning_effort", None),
        segmenter_version=inventory.segmenter_version,
        normalizer_version=inventory.normalizer_version,
        distill_config_hash=distill_config_hash,
        inventory_hash=content_hash(
            [
                {"id": item.instruction_id, "text": item.text}
                for item in inventory.instructions
            ]
        ),
    )


class AttributionCache:
    """A content-addressed cache keyed by instrument and distilled trace."""

    def __init__(self, project_root: str | Path | None):
        self.root: Path | None = None
        if project_root is not None:
            layout = initialize(project_root)
            self.root = layout.root / "cache" / "attribution"
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        assert self.root is not None
        digest = key.split(":", 1)[-1]
        return self.root / digest[:2] / f"{digest}.json"

    def get(self, key: str) -> AttributionResult | None:
        if self.root is None:
            return None
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return AttributionResult.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError):
            return None

    def put(self, key: str, result: AttributionResult) -> None:
        if self.root is None:
            return
        atomic_write_json(self._path(key), result.model_dump(mode="json"))


@dataclass(frozen=True)
class TraceAttribution:
    """One trace's attribution plus how it was obtained."""

    trace_id: str
    result: AttributionResult | None
    cached: bool = False
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.result is not None


@dataclass(frozen=True)
class HealthSample:
    """Blinded re-attribution agreement, reported not enforced."""

    sampled: int = 0
    agreed: int = 0

    @property
    def agreement_rate(self) -> float | None:
        return self.agreed / self.sampled if self.sampled else None


@dataclass(frozen=True)
class AttributionRun:
    """The batch-level outcome of attribution."""

    attributions: tuple[TraceAttribution, ...]
    instrument: Instrument
    denominator: int
    cache_hits: int = 0
    health: HealthSample = field(default_factory=HealthSample)

    @property
    def results(self) -> tuple[AttributionResult, ...]:
        return tuple(item.result for item in self.attributions if item.result is not None)

    @property
    def coverage(self) -> float:
        if not self.denominator:
            return 1.0
        return sum(1 for item in self.attributions if item.succeeded) / self.denominator

    @property
    def failures(self) -> tuple[TraceAttribution, ...]:
        return tuple(item for item in self.attributions if not item.succeeded)


def render_instructions(inventory: Inventory) -> str:
    """Render the instruction table the model attributes against."""

    lines = []
    for instruction in inventory.instructions:
        marker = "" if instruction.editable else "  [variable, not editable]"
        lines.append(f"{instruction.instruction_id}: {instruction.text.strip()}{marker}")
    return "\n".join(lines)


def render_trace(distilled: DistilledTrace) -> str:
    """Render one distilled trace for attribution."""

    return (
        f"INPUT:\n{distilled.input}\n\n"
        f"OUTPUT:\n{distilled.output}\n\n"
        f"JUDGE SCORE: {distilled.score}\n"
        f"JUDGE RATIONALE:\n{distilled.rationale}"
    )


def _parse_result(completion: Completion, trace_id: str) -> AttributionResult:
    payload = parse_json_response(completion.text)
    if not isinstance(payload, dict):
        raise AttributionError("attribution response was not an object")

    def entries(key: str, default_source: QuoteSource) -> list[dict[str, object]]:
        raw = payload.get(key) or []
        if not isinstance(raw, list):
            raise AttributionError(f"attribution field {key!r} was not a list")
        cleaned: list[dict[str, object]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            quote = item.get("quote")
            theme = item.get("theme_slug") or item.get("theme")
            if not isinstance(quote, str) or not quote.strip():
                continue
            if not isinstance(theme, str) or not theme.strip():
                continue
            instruction_id = item.get("instruction_id")
            source = item.get("quote_source")
            cleaned.append(
                {
                    "instruction_id": instruction_id if isinstance(instruction_id, str) else None,
                    "theme_slug": theme.strip(),
                    "quote": quote,
                    "quote_source": source
                    if source in {"output", "distilled"}
                    else default_source.value,
                }
            )
        return cleaned

    violation_entries = entries("violations", QuoteSource.OUTPUT)
    for entry in violation_entries:
        # A violation is a claim about the output; the contract enforces it too.
        entry["quote_source"] = QuoteSource.OUTPUT.value
    try:
        return AttributionResult.model_validate(
            {
                "trace_id": trace_id,
                "violations": violation_entries,
                "harmful": entries("harmful", QuoteSource.DISTILLED),
            }
        )
    except ValidationError as exc:
        raise AttributionError(f"attribution response failed validation: {exc}") from exc


def _vocabulary_line(vocabulary: Sequence[str]) -> str:
    if not vocabulary:
        return "THEME SLUGS ALREADY USED IN THIS BATCH: (none yet)"
    return "THEME SLUGS ALREADY USED IN THIS BATCH: " + ", ".join(sorted(vocabulary))


def attribute_trace(
    distilled: DistilledTrace,
    inventory: Inventory,
    backend: Backend,
    *,
    vocabulary: Sequence[str] = (),
    instructions_block: str | None = None,
    timeout: float | None = None,
) -> AttributionResult:
    """Attribute one trace.  Temperature 0 is set by the backend where supported."""

    block = instructions_block if instructions_block is not None else render_instructions(inventory)
    completion = backend.complete(
        SYSTEM_PROMPT,
        f"{_vocabulary_line(vocabulary)}\n\nTRACE:\n{render_trace(distilled)}",
        cacheable_prefix=f"SYSTEM PROMPT INSTRUCTIONS:\n{block}",
        timeout=timeout,
    )
    return _parse_result(completion, distilled.trace_id)


BLINDED_SYSTEM_PROMPT = """\
You name the failure themes in one graded LLM trace, without reference to any \
system prompt.

Return JSON only: {"themes": ["<lowercase-hyphenated-slug>", ...]}
Name at most three themes. An empty list is correct when the trace shows no failure."""


def _blinded_themes(distilled: DistilledTrace, backend: Backend) -> set[str]:
    completion = backend.complete(BLINDED_SYSTEM_PROMPT, render_trace(distilled))
    payload = parse_json_response(completion.text)
    if isinstance(payload, dict):
        themes = payload.get("themes") or []
        return {str(theme) for theme in themes if isinstance(theme, str)}
    return set()


def attribute_batch(
    distilled: Sequence[DistilledTrace],
    inventory: Inventory,
    backend: Backend,
    *,
    project_root: str | Path | None = None,
    distill_config_hash: str = "",
    min_coverage: float = 0.8,
    health_sample: int = DEFAULT_HEALTH_SAMPLE,
    denominator: int | None = None,
    jobs: int = 1,
) -> AttributionRun:
    """Attribute a whole batch, caching by instrument and enforcing coverage.

    ``jobs`` attributes that many traces concurrently.  Raises
    :class:`CoverageError` when the share of successfully attributed traces
    falls below ``min_coverage`` — a partial batch produces a rate whose
    denominator nobody can defend.
    """

    instrument = build_instrument(backend, inventory, distill_config_hash)
    cache = AttributionCache(project_root)
    block = render_instructions(inventory)
    vocabulary: set[str] = set()
    attributions: list[TraceAttribution] = []
    cache_hits = 0

    def attribute_one(item: DistilledTrace, known: tuple[str, ...]) -> TraceAttribution:
        key = instrument.cache_key(item)
        cached = cache.get(key)
        if cached is not None:
            return TraceAttribution(item.trace_id, cached, cached=True)
        try:
            result = attribute_trace(
                item, inventory, backend, vocabulary=known, instructions_block=block
            )
        except (AttributionError, LLMError) as exc:
            return TraceAttribution(item.trace_id, None, error=str(exc))
        cache.put(key, result)
        return TraceAttribution(item.trace_id, result)

    # Traces are attributed in fixed-size waves rather than one long queue: the
    # theme vocabulary still feeds forward between waves, and the batch order of
    # the results does not depend on which call happened to finish first.
    width = max(1, jobs)
    for start in range(0, len(distilled), width):
        wave = distilled[start : start + width]
        known = tuple(sorted(vocabulary))
        if width == 1:
            done = [attribute_one(item, known) for item in wave]
        else:
            with ThreadPoolExecutor(max_workers=width) as pool:
                done = list(pool.map(lambda item: attribute_one(item, known), wave))
        for attribution in done:
            if attribution.cached:
                cache_hits += 1
            if attribution.result is not None:
                vocabulary.update(_themes_of(attribution.result))
            attributions.append(attribution)

    total = denominator if denominator is not None else len(distilled)
    run = AttributionRun(
        attributions=tuple(attributions),
        instrument=instrument,
        denominator=total,
        cache_hits=cache_hits,
        health=_health(distilled, attributions, backend, health_sample),
    )
    if run.coverage < min_coverage:
        raise CoverageError(
            f"attribution covered {run.coverage:.0%} of {total} traces, "
            f"below the {min_coverage:.0%} floor"
        )
    return run


def _themes_of(result: AttributionResult) -> set[str]:
    return {entry.theme_slug for entry in (*result.violations, *result.harmful)}


def _health(
    distilled: Sequence[DistilledTrace],
    attributions: Sequence[TraceAttribution],
    backend: Backend,
    sample_size: int,
) -> HealthSample:
    """Re-attribute a deterministic sample blind, and report the agreement rate.

    The sample is the first ``sample_size`` traces with a non-empty attribution,
    in batch order — deterministic, so a health number is reproducible.
    """

    if sample_size <= 0:
        return HealthSample()
    by_id = {item.trace_id: item for item in distilled}
    sampled = 0
    agreed = 0
    for attribution in attributions:
        if sampled >= sample_size:
            break
        result = attribution.result
        record = by_id.get(attribution.trace_id)
        if result is None or record is None:
            continue
        themes = _themes_of(result)
        if not themes:
            continue
        try:
            blinded = _blinded_themes(record, backend)
        except (LLMError, AttributionError):
            continue
        sampled += 1
        if themes & blinded:
            agreed += 1
    return HealthSample(sampled=sampled, agreed=agreed)


def attribution_records(run: AttributionRun) -> list[Mapping[str, object]]:
    """Serialize an attribution run for the ledger."""

    return [
        {
            "trace_id": item.trace_id,
            "cached": item.cached,
            "error": item.error,
            "result": item.result.model_dump(mode="json") if item.result else None,
        }
        for item in run.attributions
    ]


def instrument_summary(instrument: Instrument) -> str:
    """A one-line human description of the instrument in use."""

    return canonical_json(
        {
            "backend": instrument.backend,
            "model": instrument.model,
            "fingerprint": instrument.fingerprint[:19],
        }
    )
