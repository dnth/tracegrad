"""Kitaru SDK gateway.  The only module that constructs ``KitaruAPIClient``.

Imported only after :func:`require_kitaru`.  Credentials and server URL come
from Kitaru's own config; tracegrad stores no secrets.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .errors import KitaruSourceError, KitaruVerifyError
from .require import require_kitaru

FETCH_JOBS = 8
_EXPERIMENT_RUN_POLL_INTERVAL = 2.0
_TERMINAL_EXPERIMENT_RUN_STATUSES = frozenset({"completed", "failed", "canceled"})


@dataclass(frozen=True)
class CohortResolution:
    """Immutable cohort version chosen once for a run (ADR 0004)."""

    cohort_id: str
    cohort_name: str
    cohort_version_id: str
    display_version: str | None
    version_number: int
    agent_id: str
    session_count: int


def configured_server_url() -> str:
    """Kitaru's own configured server URL; empty when unset."""

    try:
        require_kitaru()
        from kitaru.client.config import get_server_url
    except Exception:
        return ""
    return get_server_url() or ""


def _uuid(value: str | uuid.UUID) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


class KitaruGateway:
    """Thin async wrapper around ``KitaruAPIClient``."""

    def __init__(self, client: Any | None = None) -> None:
        require_kitaru()
        if client is not None:
            self._client = client
            self._owns = False
            return
        from kitaru.client import KitaruAPIClient

        try:
            self._client = KitaruAPIClient()
        except RuntimeError as exc:
            raise KitaruSourceError(
                "no Kitaru server URL is configured. Run `kitaru login` "
                "against your server; tracegrad stores no Kitaru secrets."
            ) from exc
        self._owns = True

    @property
    def client(self) -> Any:
        return self._client

    @property
    def base_url(self) -> str:
        url = getattr(self._client, "base_url", None) or getattr(
            self._client, "_base_url", None
        )
        return str(url) if url else ""

    async def close(self) -> None:
        if self._owns:
            await self._client.close()

    async def server_info(self) -> Any:
        return await self._client.info.get()

    async def resolve_cohort(
        self, name: str, version_ref: str | None = None
    ) -> CohortResolution:
        """Resolve a cohort name to one immutable version.  Once, for the run."""

        from kitaru.api_models.v1.cohort import CohortListParams
        from kitaru.api_models.v1.filter import FilterCondition, FilterOp

        page = await self._client.cohorts.list(
            CohortListParams(
                filter=FilterCondition(field="name", op=FilterOp.EQ, value=name)
            )
        )
        if not page.items:
            raise KitaruSourceError(f"kitaru cohort {name!r} was not found")
        cohort = page.items[0]
        versions = await self._list_versions(cohort.id)
        chosen = self._pick_version(versions, cohort.latest_version, version_ref)
        if chosen is None:
            raise KitaruSourceError(
                f"kitaru cohort {name!r} has no version matching {version_ref!r}"
                if version_ref
                else f"kitaru cohort {name!r} has no versions"
            )
        return CohortResolution(
            cohort_id=str(cohort.id),
            cohort_name=cohort.name,
            cohort_version_id=str(chosen.id),
            display_version=chosen.display_version,
            version_number=int(chosen.version),
            agent_id=str(cohort.agent_id),
            session_count=int(chosen.session_count),
        )

    def _pick_version(
        self, versions: list[Any], latest: int, version_ref: str | None
    ) -> Any | None:
        if not versions:
            return None
        if version_ref is None:
            return max(
                (item for item in versions if item.version == latest),
                default=max(versions, key=lambda item: item.version),
                key=lambda item: item.version,
            )
        for item in versions:
            if str(item.id) == version_ref:
                return item
            if item.display_version == version_ref:
                return item
            if str(item.version) == version_ref:
                return item
        return None

    async def _list_versions(self, cohort_id: uuid.UUID) -> list[Any]:
        from kitaru.api_models.v1.cohort_version import CohortVersionListParams

        versions: list[Any] = []
        params = CohortVersionListParams()
        while True:
            page = await self._client.cohorts.list_versions(cohort_id, params)
            versions.extend(page.items)
            cursor = getattr(page, "next_cursor", None)
            if not cursor:
                return versions
            params = CohortVersionListParams(cursor=cursor)

    async def list_sessions(self, cohort_version_id: str) -> list[Any]:
        from kitaru.api_models.v1.filter import FilterCondition, FilterOp
        from kitaru.api_models.v1.session import SessionListParams

        params = SessionListParams(
            filter=FilterCondition(
                field="cohort_version_id",
                op=FilterOp.EQ,
                value=cohort_version_id,
            ),
            size=100,
        )
        sessions: list[Any] = []
        async for session in self._client.sessions.iter(params):
            sessions.append(session)
        return sessions

    async def session_bundle(self, session_id: str) -> tuple[Any, tuple[Any, ...], tuple[Any, ...]]:
        session_uuid = _uuid(session_id)
        full = await self._client.sessions.get_with_nodes(session_uuid)
        evaluations = await self._evaluations_for(session_uuid)
        return full.session, tuple(full.nodes), tuple(evaluations)

    async def _evaluations_for(self, session_id: uuid.UUID) -> list[Any]:
        from kitaru.api_models.v1.evaluation import EvaluationListParams
        from kitaru.api_models.v1.filter import FilterCondition, FilterOp

        params = EvaluationListParams(
            filter=FilterCondition(
                field="session_id", op=FilterOp.EQ, value=str(session_id)
            )
        )
        return [item async for item in self._client.evaluations.iter(params)]

    async def fetch_records(
        self, sessions: Sequence[Any], *, jobs: int = FETCH_JOBS
    ) -> list[tuple[Any, tuple[Any, ...], tuple[Any, ...]]]:
        semaphore = asyncio.Semaphore(max(1, jobs))

        async def one(session: Any) -> tuple[Any, tuple[Any, ...], tuple[Any, ...]]:
            async with semaphore:
                return await self.session_bundle(str(session.id))

        if not sessions:
            return []
        # Fail-closed: one 404/timeout aborts the batch. Do not map a partial
        # cohort. Wrap so the CLI prints KitaruSourceError instead of a traceback.
        try:
            return list(await asyncio.gather(*(one(session) for session in sessions)))
        except KitaruSourceError:
            raise
        except Exception as exc:
            raise KitaruSourceError(
                "source fetch aborted: fetching a session payload failed "
                f"({type(exc).__name__}: {exc}). The batch is not mapped."
            ) from exc

    async def evaluator_id(self, name: str) -> str:
        from kitaru.api_models.v1.evaluator import EvaluatorListParams
        from kitaru.api_models.v1.filter import FilterCondition, FilterOp

        page = await self._client.evaluators.list(
            EvaluatorListParams(
                filter=FilterCondition(field="name", op=FilterOp.EQ, value=name)
            )
        )
        if not page.items:
            raise KitaruSourceError(f"kitaru evaluator {name!r} was not found")
        return str(page.items[0].id)

    async def get_agent_version(self, agent_version_id: str) -> Any:
        return await self._client.agent_versions.get(_uuid(agent_version_id))

    async def get_cohort_version(self, cohort_version_id: str) -> Any:
        return await self._client.cohort_versions.get(_uuid(cohort_version_id))

    async def list_live_workers(self) -> list[Any]:
        from kitaru.api_models.v1.worker import WorkerListParams

        # kitaru 0.22 WorkerListParams is FilterableListParams (extra='forbid').
        # include_stale is not a field; worker_covers_agent_version skips live=False.
        return [item async for item in self._client.workers.iter(WorkerListParams())]

    async def create_experiment(self, request: Any) -> Any:
        return await self._client.experiments.create(request)

    async def start_run(self, experiment_id: str, request: Any) -> Any:
        return await self._client.experiments.start_run(_uuid(experiment_id), request)

    async def get_experiment_run(self, run_id: str) -> Any:
        return await self._client.experiment_runs.get(_uuid(run_id))

    async def wait_for_experiment_run(self, run_id: str, timeout: float | None = None) -> Any:
        """Poll ``experiment_runs.get`` until completed, failed, or canceled.

        Same loop as kitaru CLI ``poll_run``. Do not wrap ``KitaruClient``:
        its ``close()`` shuts the shared API client, and wait is not a method
        on the 0.22 API client (close / context-manager only).
        """

        run_uuid = _uuid(run_id)
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if deadline is None:
                run = await self._client.experiment_runs.get(run_uuid)
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Timed out waiting for experiment run {run_id}")
                try:
                    run = await asyncio.wait_for(
                        self._client.experiment_runs.get(run_uuid),
                        timeout=remaining,
                    )
                except TimeoutError as exc:
                    raise TimeoutError(
                        f"Timed out waiting for experiment run {run_id}"
                    ) from exc

            status = getattr(run, "status", None)
            if str(getattr(status, "value", status) or "") in _TERMINAL_EXPERIMENT_RUN_STATUSES:
                return run

            if deadline is None:
                await asyncio.sleep(_EXPERIMENT_RUN_POLL_INTERVAL)
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for experiment run {run_id}")
            try:
                await asyncio.wait_for(
                    asyncio.sleep(min(_EXPERIMENT_RUN_POLL_INTERVAL, remaining)),
                    timeout=remaining,
                )
            except TimeoutError as exc:
                raise TimeoutError(
                    f"Timed out waiting for experiment run {run_id}"
                ) from exc

    async def list_replays(self, experiment_run_id: str) -> list[Any]:
        from kitaru.api_models.v1.filter import FilterCondition, FilterOp
        from kitaru.api_models.v1.replay import ReplayListParams

        params = ReplayListParams(
            filter=FilterCondition(
                field="experiment_run_id",
                op=FilterOp.EQ,
                value=experiment_run_id,
            )
        )
        return [item async for item in self._client.replays.iter(params)]

    async def evaluation_aggregates(self, experiment_run_id: str) -> list[Any]:
        """Headline stats from the UI-support namespace.

        ``/api/v1/ui/`` is a UI-support namespace, not an obvious third-party
        contract.  This is the most likely thing to move under us; the
        ``<0.23`` pin contains it.
        """

        response = await self._client.request(
            "GET",
            f"/api/v1/ui/experiment-runs/{experiment_run_id}/evaluation-aggregates",
        )
        return response.json()

    async def session_nodes(self, session_id: str) -> tuple[Any, ...]:
        full = await self._client.sessions.get_with_nodes(_uuid(session_id))
        return tuple(full.nodes)

    async def evaluations_for(self, session_id: str) -> list[Any]:
        return await self._evaluations_for(_uuid(session_id))


def worker_covers_agent_version(worker: Any, agent_version_id: str) -> bool:
    """Whether a live worker claims this agent version."""

    if getattr(worker, "live", False) is False:
        return False
    scope = getattr(worker, "scope", None)
    claims = getattr(scope, "claims", None) or ()
    target = str(agent_version_id)
    for claim in claims:
        kind = getattr(getattr(claim, "kind", None), "value", getattr(claim, "kind", None))
        if str(kind) != "agent":
            continue
        claimed = getattr(claim, "agent_version_id", None)
        if claimed is None or str(claimed) == target:
            return True
    return False


def run_async(coro: Any) -> Any:
    """Run one coroutine from the sync CLI."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise KitaruVerifyError("kitaru client cannot nest inside a running event loop")
