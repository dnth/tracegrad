"""Optional Kitaru integration.

Importing this package does not import the Kitaru SDK.  Mapping, scoring, and
graph helpers are usable in a core-only install; client/source/verify modules
call :func:`require_kitaru` before touching the SDK.
"""

from __future__ import annotations

from .errors import (
    INSTALL_MESSAGE,
    KITARU_EXTRA,
    KITARU_PIN,
    NO_BACKEND_MESSAGE,
    KitaruError,
    KitaruNotInstalled,
    KitaruSourceError,
    KitaruVerifyError,
)
from .mapping import (
    MAPPING_VERSION,
    MappedTrace,
    SourceDrop,
    map_session,
)
from .require import kitaru_available, require_kitaru

__all__ = [
    "INSTALL_MESSAGE",
    "KITARU_EXTRA",
    "KITARU_PIN",
    "MAPPING_VERSION",
    "NO_BACKEND_MESSAGE",
    "KitaruError",
    "KitaruNotInstalled",
    "KitaruSourceError",
    "KitaruVerifyError",
    "MappedTrace",
    "SourceDrop",
    "kitaru_available",
    "map_session",
    "require_kitaru",
]
