"""RFC 6901 JSON Pointer resolution for Kitaru node selectors.

Selectors are never guessed and never stringified as a fallback: a pointer that
does not land on a string is a resolution failure, and a missing selector is
the same.  This module does not import the Kitaru SDK.
"""

from __future__ import annotations

from typing import Any


def unescape_token(token: str) -> str:
    """Decode one JSON Pointer token (``~1`` → ``/``, ``~0`` → ``~``)."""

    return token.replace("~1", "/").replace("~0", "~")


def resolve_pointer(document: Any, pointer: str) -> Any:
    """Resolve an RFC 6901 JSON Pointer.

    Raises:
        ValueError: The pointer is malformed or does not exist.
    """

    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise ValueError(f"JSON Pointer must be empty or start with '/': {pointer!r}")

    current = document
    for raw_token in pointer[1:].split("/"):
        token = unescape_token(raw_token)
        if isinstance(current, list):
            if token == "-":
                raise ValueError("JSON Pointer '-' is not resolvable on a list")
            if not token.isdigit() or (len(token) > 1 and token.startswith("0")):
                raise ValueError(f"invalid array index {token!r}")
            index = int(token)
            if index >= len(current):
                raise ValueError(f"array index {index} out of range")
            current = current[index]
            continue
        if isinstance(current, dict):
            if token not in current:
                raise ValueError(f"key {token!r} not found")
            current = current[token]
            continue
        raise ValueError(f"cannot traverse {type(current).__name__} with {token!r}")
    return current


def resolve_text_selector(document: Any, selector: str | None) -> str | None:
    """Resolve a selector onto a string, or ``None`` on any failure.

    Non-string landings are failures: tool payloads and nested objects must
    not become ``Trace.input`` / ``Trace.output`` / a system prompt.
    """

    if selector is None:
        return None
    try:
        value = resolve_pointer(document, selector)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, str) else None
