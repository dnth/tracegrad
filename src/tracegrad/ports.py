"""The shape the deterministic core is allowed to know about a model backend.

The orchestrator has to pass a backend from A to B without becoming a model-aware
module itself.  Importing the concrete backend for that would smuggle the model
layer across the determinism boundary, so the *shape* lives here, on the
deterministic side, and the implementations live in ``llm``.

Nothing in this module can talk to anything.  That is the point.

``VerificationBackend`` is the same idea for Phase 2 (ADR 0010): the
orchestrator holds a backend without becoming backend-aware.  The Kitaru
implementation lives under ``integrations/kitaru/``.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable


@runtime_checkable
class Backend(Protocol):
    """One completion call, and the name the instrument records it under."""

    name: str

    def complete(
        self,
        system: str,
        user: str,
        *,
        cacheable_prefix: str | None = None,
        schema: Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Any: ...


@runtime_checkable
class VerificationBackend(Protocol):
    """Replay-verify a candidate without the orchestrator knowing the vendor."""

    name: str

    def preflight(self, request: Any) -> None: ...

    def submit(self, request: Any) -> Any: ...

    def collect(self, request: Any, submitted: Any) -> Any: ...
