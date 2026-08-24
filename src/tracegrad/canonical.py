"""Canonical serialization and hashing shared by the deterministic core.

Every content address, cache key, and config fingerprint in tracegrad is a
SHA-256 over the canonical JSON encoding defined here.  The encoding is stable
across processes and Python versions: sorted keys, no insignificant whitespace,
UTF-8, and no NaN or infinity.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

HASH_PREFIX = "sha256:"


def canonical_json(value: Any) -> str:
    """Encode ``value`` in tracegrad's canonical JSON form."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def sha256_hex(value: str | bytes) -> str:
    """Return the bare hex SHA-256 digest of ``value``."""

    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def content_hash(value: Any) -> str:
    """Return a ``sha256:``-prefixed digest of the canonical JSON of ``value``."""

    return HASH_PREFIX + sha256_hex(canonical_json(value))


def text_hash(text: str) -> str:
    """Return a ``sha256:``-prefixed digest of raw text."""

    return HASH_PREFIX + sha256_hex(text)
