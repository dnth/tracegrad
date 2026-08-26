"""Lazy availability check for the optional Kitaru extra."""

from __future__ import annotations

from importlib import import_module

from .errors import KitaruNotInstalled


def kitaru_available() -> bool:
    """Whether the Kitaru SDK can be imported in this environment."""

    try:
        import_module("kitaru")
    except ImportError:
        return False
    return True


def require_kitaru() -> None:
    """Raise an actionable install message when the extra is missing."""

    if not kitaru_available():
        raise KitaruNotInstalled()
