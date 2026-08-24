"""Local `.tracegradrc` configuration loading.

The rc file is TOML and lives at the project root.  A missing file is valid and
uses the defaults below.  The top-level keys intentionally use the camel-case
names consumed by the CLI contract.
"""

from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic.types import StrictBool, StrictFloat, StrictInt, StrictStr

DEFAULT_RC_FILENAME = ".tracegradrc"
DEFAULT_MIN_EFFECT = 0.05
DEFAULT_MIN_COVERAGE = 0.8
DEFAULT_CONVERGENCE_RUNS = 2

_ConfigNumber: TypeAlias = StrictFloat | StrictInt

class ConfigError(ValueError):
    """A malformed or invalid tracegrad rc file."""


class HarnessPreset(BaseModel):
    """Configuration for one model or command harness preset."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    provider: StrictStr = "command"
    model: StrictStr | None = None
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

    @property
    def harness(self) -> dict[str, HarnessPreset]:
        """Compatibility accessor for the rc's ``harness`` wording."""

        return self.harness_presets


def _resolve_rc_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate / DEFAULT_RC_FILENAME if candidate.is_dir() else candidate


def _normalize_harness_key(data: dict[str, object]) -> dict[str, object]:
    """Accept the documented name and the concise ``harness`` spelling."""

    normalized = copy.deepcopy(data)
    if "harness" in normalized and "harness_presets" not in normalized:
        harness = normalized.pop("harness")
        if isinstance(harness, dict) and set(harness) == {"presets"}:
            harness = harness["presets"]
        normalized["harness_presets"] = harness
    return normalized


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
        return TracegradConfig.model_validate(_normalize_harness_key(raw))
    except ValidationError as exc:
        raise ConfigError(f"invalid tracegrad config {rc_path}: {exc}") from exc


def load_rc(root: str | Path = ".") -> TracegradConfig:
    """Load ``.tracegradrc`` from a project root."""

    return load_config(root)


Config = TracegradConfig
