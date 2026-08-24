"""Durable local state for tracegrad runs.

The project state lives under ``.tracegrad/``:

* ``lock`` — process-wide exclusive run lock;
* ``distilled/`` — content-addressed distilled traces;
* ``ledgers/`` — append-only cross-run JSONL ledgers;
* ``reports/`` — persisted reports;
* ``runs/<run-id>/resume.json`` — per-run interruption checkpoints;
* ``snapshots/`` — pre-apply prompt snapshots;
* ``.gitignore`` — keeps generated state out of version control.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, TypeAlias

JsonObject: TypeAlias = dict[str, object]

STATE_DIRNAME = ".tracegrad"
GITIGNORE_CONTENT = "*\n!.gitignore\n"


class StateError(RuntimeError):
    """Base error for local state operations."""


class StateLockError(StateError):
    """Raised when a state lock cannot be acquired or released."""


@dataclass(frozen=True)
class StateLayout:
    """Canonical paths in one project's state directory."""

    root: Path
    lock: Path
    distilled: Path
    ledgers: Path
    reports: Path
    snapshots: Path
    runs: Path
    gitignore: Path

    @property
    def resume_directory(self) -> Path:
        return self.runs

    @property
    def data_directories(self) -> tuple[Path, ...]:
        return (self.distilled, self.ledgers, self.reports, self.snapshots)

    def resume_path(self, run_id: str) -> Path:
        if not run_id or Path(run_id).name != run_id:
            raise ValueError("run_id must be a non-empty path-safe name")
        return self.runs / run_id / "resume.json"

    @property
    def state_dir(self) -> Path:
        return self.root

    @property
    def lock_path(self) -> Path:
        return self.lock


def _layout(project_root: str | Path | StateLayout) -> StateLayout:
    if isinstance(project_root, StateLayout):
        return project_root
    root = Path(project_root)
    state_root = root if root.name == STATE_DIRNAME else root / STATE_DIRNAME
    return StateLayout(
        root=state_root,
        lock=state_root / "lock",
        distilled=state_root / "distilled",
        ledgers=state_root / "ledgers",
        reports=state_root / "reports",
        snapshots=state_root / "snapshots",
        runs=state_root / "runs",
        gitignore=state_root / ".gitignore",
    )


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def initialize(project_root: str | Path) -> StateLayout:
    """Create the state layout and its protective nested ``.gitignore``."""

    layout = _layout(project_root)
    layout.root.mkdir(parents=True, exist_ok=True)
    for directory in layout.data_directories + (layout.resume_directory,):
        directory.mkdir(parents=True, exist_ok=True)
    if not layout.gitignore.exists():
        atomic_write(layout.gitignore, GITIGNORE_CONTENT)
    return layout


class StateLock:
    """An exclusive lock file acquired with ``O_EXCL``."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._descriptor: int | None = None

    def acquire(self) -> "StateLock":
        if self._descriptor is not None:
            raise StateLockError(f"lock already held by this owner: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                self.path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
        except FileExistsError as exc:
            raise StateLockError(f"lock already held: {self.path}") from exc
        try:
            os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
            os.fsync(descriptor)
        except BaseException:
            os.close(descriptor)
            self.path.unlink(missing_ok=True)
            raise
        self._descriptor = descriptor
        _fsync_directory(self.path.parent)
        return self

    def release(self) -> None:
        descriptor = self._descriptor
        if descriptor is None:
            return
        self._descriptor = None
        try:
            os.close(descriptor)
            self.path.unlink(missing_ok=True)
            _fsync_directory(self.path.parent)
        except OSError as exc:
            raise StateLockError(f"could not release lock {self.path}: {exc}") from exc

    def __enter__(self) -> "StateLock":
        return self.acquire()

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.release()


def atomic_write(path: str | Path, content: bytes | bytearray | str) -> None:
    """Replace ``path`` with fully fsynced content, never a partial file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = content.encode("utf-8") if isinstance(content, str) else bytes(content)
    descriptor: int | None = None
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
        _fsync_directory(target.parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_json(path: str | Path, value: Mapping[str, object]) -> None:
    """Serialize one JSON object and atomically replace ``path``."""

    encoded = json.dumps(
        dict(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    atomic_write(path, encoded + "\n")


def append_jsonl(path: str | Path, record: Mapping[str, object]) -> None:
    """Durably append exactly one JSON object followed by one newline."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            dict(record),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    with target.open("ab") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(target.parent)


def load_jsonl(path: str | Path) -> list[JsonObject]:
    """Read an append-only JSONL ledger, rejecting malformed records."""

    target = Path(path)
    if not target.exists():
        return []
    records: list[JsonObject] = []
    try:
        for line_number, line in enumerate(
            target.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("record is not a JSON object")
            records.append(value)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise StateError(f"could not read JSONL ledger {target}: {exc}") from exc
    return records


def save_resume_state(
    project_root: str | Path | StateLayout,
    run_id: str,
    state: Mapping[str, object],
) -> Path:
    """Persist one run checkpoint atomically and return its path."""

    layout = initialize(project_root)
    target = layout.resume_path(run_id)
    atomic_write_json(target, state)
    return target


def load_resume_state(
    project_root: str | Path | StateLayout,
    run_id: str,
) -> JsonObject | None:
    """Read a run checkpoint, returning ``None`` when it has not started."""

    layout = _layout(project_root)
    target = layout.resume_path(run_id)
    if not target.exists():
        return None
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StateError(f"could not read resume state {target}: {exc}") from exc
    if not isinstance(value, dict):
        raise StateError(f"resume state is not a JSON object: {target}")
    return value
