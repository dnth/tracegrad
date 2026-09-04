"""Tool-policy spec the Kitaru backend is allowed to emit.

The values are the Kitaru wire strings.  The only policy the tracegrad path
constructs is recorded history with ``on_miss=fail``.  There is no parameter
that can select passthrough (ADR / issue #9 hard invariant).
"""

from __future__ import annotations

from typing import Final, Literal

HistoryScopeName = Literal["cohort_version"]
OnMissName = Literal["fail"]

HISTORY_SCOPE: Final[HistoryScopeName] = "cohort_version"
ON_MISS: Final[OnMissName] = "fail"
POLICY_TYPE: Final[str] = "history"

# Wire form of HistoryConfig(scope=COHORT_VERSION, on_miss=FAIL).
RECORDED_HISTORY_POLICY: Final[dict[str, str]] = {
    "type": POLICY_TYPE,
    "scope": HISTORY_SCOPE,
    "on_miss": ON_MISS,
}


def recorded_history_policy() -> dict[str, str]:
    """The only tool policy tracegrad will attach to a replay."""

    return dict(RECORDED_HISTORY_POLICY)


def asserts_no_passthrough(policy: dict[str, str]) -> None:
    """Fail closed if a policy other than recorded-history/fail is supplied."""

    if policy.get("type") != POLICY_TYPE:
        raise ValueError("tracegrad refusals: only history tool policy is allowed")
    if policy.get("scope") != HISTORY_SCOPE:
        raise ValueError("tracegrad refusals: history scope must be cohort_version")
    if policy.get("on_miss") != ON_MISS:
        raise ValueError("tracegrad refusals: history on_miss must be fail")
    if str(policy.get("on_miss", "")).lower() == "passthrough":
        raise ValueError("passthrough tool policy is not reachable through tracegrad")
