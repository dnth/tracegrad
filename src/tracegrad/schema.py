"""Versioned data contracts for tracegrad's JSON and ledger interfaces."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.types import StrictBool, StrictFloat, StrictInt, StrictStr


class _ContractModel(BaseModel):
    """Shared strict behavior for persisted contracts."""

    model_config = ConfigDict(extra="forbid")


class TemplateEngine(StrEnum):
    NONE = "none"
    FORMAT = "format"


class QuoteSource(StrEnum):
    """The persisted artifact from which evidence was quoted."""

    DISTILLED = "distilled"
    OUTPUT = "output"


class Verdict(StrEnum):
    IMPROVED = "improved"
    REGRESSED = "regressed"
    ELIMINATED = "eliminated"
    NO_SIGNAL = "no-signal"


JsonScalar: TypeAlias = StrictBool | StrictInt | StrictFloat | StrictStr | None
Score: TypeAlias = Annotated[StrictFloat | StrictInt, Field(ge=0.0, le=1.0)]


class Judge(_ContractModel):
    score: Score
    rationale: StrictStr = Field(min_length=1)


class TraceMeta(_ContractModel):
    model: StrictStr | None = None


class Manifest(_ContractModel):
    template_file: Path
    engine: TemplateEngine = TemplateEngine.NONE
    vars: dict[StrictStr, StrictStr] = Field(default_factory=dict)
    sampling: dict[StrictStr, JsonScalar] = Field(default_factory=dict)
    judge_fingerprint: StrictStr = Field(min_length=1)


class Trace(_ContractModel):
    trace_id: StrictStr = Field(min_length=1)
    input: StrictStr
    output: StrictStr
    judge: Judge
    prompt_hash: StrictStr = Field(min_length=1)
    meta: TraceMeta | None = None


class AttributionEntry(_ContractModel):
    instruction_id: StrictStr | None = None
    theme_slug: StrictStr = Field(min_length=1)
    quote: StrictStr = Field(min_length=1)
    quote_source: QuoteSource


class AttributionResult(_ContractModel):
    trace_id: StrictStr | None = None
    violations: list[AttributionEntry] = Field(default_factory=list)
    harmful: list[AttributionEntry] = Field(default_factory=list)

    @field_validator("violations")
    @classmethod
    def violations_require_output_evidence(
        cls, entries: list[AttributionEntry]
    ) -> list[AttributionEntry]:
        if any(entry.quote_source is not QuoteSource.OUTPUT for entry in entries):
            raise ValueError("violation evidence must use quote_source='output'")
        return entries


class Edit(_ContractModel):
    instruction_id: StrictStr = Field(min_length=1)
    operation: Annotated[StrictStr, Field(pattern="^(ADD|REWRITE|DELETE)$")] = "REWRITE"
    text: StrictStr = ""
    covers_theme: StrictStr = Field(min_length=1)
    watch_metric: StrictStr = Field(min_length=1)


class Cluster(_ContractModel):
    theme: StrictStr = Field(min_length=1)
    numerator: StrictInt = Field(ge=0)
    denominator: StrictInt = Field(ge=0)

    @field_validator("denominator")
    @classmethod
    def denominator_covers_numerator(cls, value: int, info: object) -> int:
        numerator = info.data.get("numerator")  # type: ignore[union-attr]
        if numerator is not None and value < numerator:
            raise ValueError("denominator must be greater than or equal to numerator")
        return value


class Report(_ContractModel):
    applied_prompt_hash: StrictStr = Field(min_length=1)
    clusters: list[Cluster] = Field(default_factory=list)


class StepVerdict(_ContractModel):
    verdict: Verdict
    theme: StrictStr | None = None
    step: StrictStr | None = None
    reason: StrictStr | None = None
