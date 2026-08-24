"""Deterministic trace distillation and the content-addressed distilled store.

Distillation is the evidence boundary.  Attribution never sees a raw trace: it
sees the distilled record, and every quote is later substring-verified against
exactly this text.  So distillation must be reproducible byte for byte —
no model, no clock, no randomness — and its parameters are folded into
``distill_config_hash`` so a change in reduction or redaction invalidates
downstream caches instead of silently mixing eras.

Redaction uses stable placeholders: the same secret in the same record always
becomes the same ``[REDACTED:EMAIL:1]`` token, so a quote stays quotable and a
human reading two traces can tell whether they saw the same value.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Mapping, Sequence

from .canonical import canonical_json, content_hash, text_hash
from .schema import Manifest, TemplateEngine, Trace
from .state import PathContainmentError, atomic_write, contained_path, initialize

DISTILL_VERSION = 1
REDACTION_VERSION = 1
DEFAULT_MAX_INPUT_CHARS = 2000
DEFAULT_MAX_OUTPUT_CHARS = 4000
DEFAULT_MAX_RATIONALE_CHARS = 2000
ELISION_TEMPLATE = "\n…[{count} characters elided]…\n"


class DistillError(ValueError):
    """A template that cannot be rendered, or a record that cannot be distilled."""


@dataclass(frozen=True)
class Span:
    """A half-open ``[start, end)`` character range of the rendered prompt."""

    start: int
    end: int
    name: str

    def contains(self, start: int, end: int) -> bool:
        return self.start < end and start < self.end


@dataclass(frozen=True)
class RenderedPrompt:
    """A rendered prompt plus the spans that came from variable substitution."""

    text: str
    variable_spans: tuple[Span, ...]
    engine: TemplateEngine
    template_hash: str

    @property
    def prompt_hash(self) -> str:
        return text_hash(self.text)

    def is_variable(self, start: int, end: int) -> bool:
        return any(span.contains(start, end) for span in self.variable_spans)


def render_template(
    template: str,
    engine: TemplateEngine = TemplateEngine.NONE,
    variables: Mapping[str, str] | None = None,
) -> RenderedPrompt:
    """Render ``template`` under the declared engine, tracking variable spans.

    Only the engines shipped in v0.1.0 are accepted: ``none`` renders the
    template verbatim, ``format`` performs positional-free ``str.format``
    substitution.  An engine is declared in the manifest and never guessed.
    """

    values = dict(variables or {})
    template_digest = text_hash(template)

    if engine is TemplateEngine.NONE:
        return RenderedPrompt(template, (), engine, template_digest)

    if engine is not TemplateEngine.FORMAT:
        raise DistillError(f"unsupported template engine: {engine}")

    pieces: list[str] = []
    spans: list[Span] = []
    cursor = 0
    for literal, field_name, format_spec, conversion in Formatter().parse(template):
        pieces.append(literal)
        cursor += len(literal)
        if field_name is None:
            continue
        if field_name == "" or field_name.isdigit():
            raise DistillError("positional template fields are not supported")
        if format_spec or conversion:
            raise DistillError(f"template field '{field_name}' must be plain")
        if field_name not in values:
            raise DistillError(f"template variable is not declared: {field_name}")
        value = values[field_name]
        pieces.append(value)
        spans.append(Span(cursor, cursor + len(value), field_name))
        cursor += len(value)

    return RenderedPrompt("".join(pieces), tuple(spans), engine, template_digest)


def render_manifest_prompt(manifest: Manifest, base_directory: str | Path = ".") -> RenderedPrompt:
    """Read and render the prompt template a manifest points at."""

    try:
        template_path = contained_path(base_directory, manifest.template_file)
    except PathContainmentError as exc:
        raise DistillError(f"manifest template_file escapes the project: {exc}") from exc
    try:
        template = template_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DistillError(f"could not read template {template_path}: {exc}") from exc
    return render_template(template, manifest.engine, manifest.vars)


_REDACTION_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("URL", re.compile(r"https?://[^\s<>\"')]+")),
    ("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    (
        "UUID",
        re.compile(
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
        ),
    ),
    ("KEY", re.compile(r"\b(?:sk|pk|api|token)[-_][A-Za-z0-9_-]{12,}\b")),
    ("PHONE", re.compile(r"(?<![\w.])\+?\d[\d ()-]{8,}\d(?![\w.])")),
)


class _Redactor:
    """Assigns stable, first-appearance-ordered placeholders within one record."""

    def __init__(self) -> None:
        self._assigned: dict[tuple[str, str], str] = {}
        self._counters: dict[str, int] = {}

    def placeholder(self, kind: str, value: str) -> str:
        key = (kind, value)
        existing = self._assigned.get(key)
        if existing is not None:
            return existing
        ordinal = self._counters.get(kind, 0) + 1
        self._counters[kind] = ordinal
        token = f"[REDACTED:{kind}:{ordinal}]"
        self._assigned[key] = token
        return token

    def redact(self, text: str) -> str:
        result = text
        for kind, pattern in _REDACTION_RULES:
            result = pattern.sub(lambda match, kind=kind: self.placeholder(kind, match.group(0)), result)
        return result

    @property
    def mapping(self) -> dict[str, str]:
        return {token: kind for (kind, _), token in self._assigned.items()}


def normalize_whitespace(text: str) -> str:
    """Collapse platform noise without touching the words a quote will use."""

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    stripped = [line.rstrip() for line in lines]
    collapsed: list[str] = []
    blank_run = 0
    for line in stripped:
        if line:
            blank_run = 0
            collapsed.append(line)
            continue
        blank_run += 1
        if blank_run <= 2:
            collapsed.append(line)
    return "\n".join(collapsed).strip("\n")


def reduce_text(text: str, max_chars: int) -> str:
    """Bound ``text`` deterministically, keeping the head and the tail."""

    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head_chars = (max_chars * 2) // 3
    tail_chars = max_chars - head_chars
    elided = len(text) - head_chars - tail_chars
    marker = ELISION_TEMPLATE.format(count=elided)
    return text[:head_chars] + marker + text[len(text) - tail_chars :]


@dataclass(frozen=True)
class DistillConfig:
    """Everything that can change distilled bytes, and therefore cache validity."""

    max_input_chars: int = DEFAULT_MAX_INPUT_CHARS
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS
    max_rationale_chars: int = DEFAULT_MAX_RATIONALE_CHARS
    redact: bool = True

    @property
    def config_hash(self) -> str:
        return content_hash(
            {
                "distill_version": DISTILL_VERSION,
                "redaction_version": REDACTION_VERSION if self.redact else 0,
                "max_input_chars": self.max_input_chars,
                "max_output_chars": self.max_output_chars,
                "max_rationale_chars": self.max_rationale_chars,
                "redact": self.redact,
            }
        )


@dataclass(frozen=True)
class DistilledTrace:
    """The only view of a trace that attribution and evidence checks may use."""

    trace_id: str
    input: str
    output: str
    score: float
    rationale: str
    prompt_hash: str
    model: str | None
    distill_config_hash: str
    redactions: Mapping[str, str]

    def to_record(self) -> dict[str, object]:
        return {
            "trace_id": self.trace_id,
            "input": self.input,
            "output": self.output,
            "score": self.score,
            "rationale": self.rationale,
            "prompt_hash": self.prompt_hash,
            "model": self.model,
            "distill_config_hash": self.distill_config_hash,
            "redactions": dict(self.redactions),
        }

    @property
    def content_address(self) -> str:
        return content_hash(self.to_record())

    def quotable(self, source: str) -> str:
        """Return the field a quote with ``quote_source`` may be verified against."""

        if source == "output":
            return self.output
        if source == "distilled":
            return "\n".join((self.input, self.output, self.rationale))
        raise DistillError(f"unknown quote source: {source}")


def distill_trace(trace: Trace, config: DistillConfig | None = None) -> DistilledTrace:
    """Reduce and redact one trace into its distilled form."""

    settings = config or DistillConfig()
    redactor = _Redactor()

    def prepare(text: str, limit: int) -> str:
        # Redact before reducing: truncating first can cut a secret in half and
        # leave the surviving fragment unmatched by every rule.
        normalized = normalize_whitespace(text)
        redacted = redactor.redact(normalized) if settings.redact else normalized
        return reduce_text(redacted, limit)

    distilled_input = prepare(trace.input, settings.max_input_chars)
    distilled_output = prepare(trace.output, settings.max_output_chars)
    distilled_rationale = prepare(trace.judge.rationale, settings.max_rationale_chars)

    return DistilledTrace(
        trace_id=trace.trace_id,
        input=distilled_input,
        output=distilled_output,
        score=float(trace.judge.score),
        rationale=distilled_rationale,
        prompt_hash=trace.prompt_hash,
        model=trace.meta.model if trace.meta else None,
        distill_config_hash=settings.config_hash,
        redactions=redactor.mapping,
    )


def distill_batch(
    traces: Sequence[Trace], config: DistillConfig | None = None
) -> tuple[DistilledTrace, ...]:
    """Distill a batch, preserving ingest order."""

    settings = config or DistillConfig()
    return tuple(distill_trace(trace, settings) for trace in traces)


def store_path(project_root: str | Path, address: str) -> Path:
    """Return the content-addressed path for one distilled record."""

    digest = address.split(":", 1)[-1]
    if len(digest) < 4 or not all(character in "0123456789abcdef" for character in digest):
        raise DistillError(f"invalid distilled content address: {address}")
    layout = initialize(project_root)
    return layout.distilled / digest[:2] / f"{digest}.json"


def store_distilled(project_root: str | Path, distilled: DistilledTrace) -> Path:
    """Write one distilled record into the content-addressed store."""

    target = store_path(project_root, distilled.content_address)
    if not target.exists():
        atomic_write(target, canonical_json(distilled.to_record()) + "\n")
    return target


def store_batch(
    project_root: str | Path, batch: Sequence[DistilledTrace]
) -> dict[str, Path]:
    """Persist a distilled batch, returning ``trace_id -> path``."""

    return {item.trace_id: store_distilled(project_root, item) for item in batch}
