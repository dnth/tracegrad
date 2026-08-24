"""Local `.tracegradrc` configuration loading.

The rc file is TOML and lives at the project root.  A missing file is valid and
uses the defaults below.  The top-level keys intentionally use the camel-case
names consumed by the CLI contract.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic.types import StrictBool, StrictFloat, StrictInt, StrictStr

DEFAULT_RC_FILENAME = ".tracegradrc"
DEFAULT_MIN_EFFECT = 0.05
DEFAULT_MIN_COVERAGE = 0.8
DEFAULT_CONVERGENCE_RUNS = 2
DEFAULT_ATTRIBUTION_TEMPERATURE = 0.0
"""Attribution is a measurement, so it samples deterministically where it can."""

_ConfigNumber: TypeAlias = StrictFloat | StrictInt


class ConfigError(ValueError):
    """A malformed or invalid tracegrad rc file."""


class HarnessPreset(BaseModel):
    """Configuration for one model or command harness preset."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    provider: StrictStr = "command"
    model: StrictStr | None = None
    temperature: StrictFloat | StrictInt | None = None
    reasoning_effort: StrictStr | None = None
    jobs: StrictInt = Field(default=1, ge=1)
    command: StrictStr | list[StrictStr] | None = None
    env: dict[StrictStr, StrictStr] = Field(default_factory=dict)
    timeoutSeconds: StrictInt | StrictFloat | None = Field(default=None, ge=0)
    enabled: StrictBool = True


class TracegradConfig(BaseModel):
    """Validated project configuration loaded from `.tracegradrc`."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    neverDelete: list[StrictStr] = Field(default_factory=list)
    minEffect: _ConfigNumber = Field(default=DEFAULT_MIN_EFFECT, ge=0)
    minCoverage: _ConfigNumber = Field(default=DEFAULT_MIN_COVERAGE, ge=0, le=1)
    convergenceRuns: StrictInt = Field(default=DEFAULT_CONVERGENCE_RUNS, ge=1)
    harness_presets: dict[StrictStr, HarnessPreset] = Field(
        default_factory=lambda: {
            "attribution": HarnessPreset(provider="openai"),
            "synthesis": HarnessPreset(provider="claude"),
        }
    )


def _resolve_rc_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate / DEFAULT_RC_FILENAME if candidate.is_dir() else candidate


def load_config(path: str | Path = ".") -> TracegradConfig:
    """Load and validate a TOML rc file, or defaults when it is absent.

    ``path`` may be the rc file itself or a project directory.  All parse and
    validation failures become :class:`ConfigError` with the source path.
    """

    rc_path = _resolve_rc_path(path)
    if not rc_path.exists():
        return TracegradConfig()
    if not rc_path.is_file():
        raise ConfigError(f"tracegrad config is not a file: {rc_path}")

    try:
        with rc_path.open("rb") as handle:
            raw = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"could not parse tracegrad config {rc_path}: {exc}") from exc

    try:
        return TracegradConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid tracegrad config {rc_path}: {exc}") from exc
