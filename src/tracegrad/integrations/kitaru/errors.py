"""Named errors for the optional Kitaru integration.

This module does not import the Kitaru SDK.  Importing it must be safe in a
core-only install.
"""

from __future__ import annotations

KITARU_EXTRA = "tracegrad[kitaru]"
KITARU_PIN = "kitaru>=0.22,<0.23"

INSTALL_MESSAGE = (
    "Kitaru support is an optional extra, not part of core tracegrad.\n"
    f'  install it with:  uv tool install "{KITARU_EXTRA}"\n'
    f"  the extra pins {KITARU_PIN}\n"
    "Then run `kitaru login` against your server. "
    "tracegrad stores no Kitaru secrets.\n"
    "Core commands still work without it: "
    "tracegrad run --traces … / apply / trends."
)

NO_BACKEND_MESSAGE = (
    "tracegrad verify needs a backend; without one nothing was verified.\n"
    "  install the extra:  uv tool install \"tracegrad[kitaru]\"\n"
    "  then:  tracegrad verify --backend kitaru --run <run-id>\n"
    "A verify that exits 0 having done nothing would read as verified in CI.\n"
    "run, apply, and trends still work without a backend."
)


class KitaruError(ValueError):
    """A Kitaru integration failure with an actionable message."""


class KitaruNotInstalled(KitaruError):
    """The extra is not installed."""

    def __init__(self, message: str = INSTALL_MESSAGE) -> None:
        super().__init__(message)


class KitaruSourceError(KitaruError):
    """Fetching or mapping a Kitaru cohort failed."""


class KitaruVerifyError(KitaruError):
    """Replay verification could not start or complete."""
