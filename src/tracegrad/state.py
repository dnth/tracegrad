"""Durable local state for tracegrad runs.

The project state lives under ``.tracegrad/``:

* ``lock`` — process-wide exclusive run lock;
* ``distilled/`` — content-addressed distilled traces;
* ``ledgers/`` — append-only cross-run JSONL ledgers;
* ``reports/`` — persisted reports;
* ``runs/<run-id>/resume.json`` — per-run interruption checkpoints;
* ``snapshots/`` — pre-apply prompt snapshots;
* ``sources/`` — optional mapped-trace snapshots (JSONL the pipeline already reads);
* ``verification/`` — persisted replay-verification state;
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
    sources: Path
    verification: Path
    gitignore: Path

    @property
    def resume_directory(self) -> Path:
        return self.runs

    @property
    def data_directories(self) -> tuple[Path, ...]:
        return (
            self.distilled,
            self.ledgers,
            self.reports,
            self.snapshots,
            self.sources,
            self.verification,
        )

    def resume_path(self, run_id: str) -> Path:
        return self.runs / validate_run_id(run_id) / "resume.json"

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
        sources=state_root / "sources",
        verification=state_root / "verification",
        gitignore=state_root / ".gitignore",
    )


class PathContainmentError(StateError):
    """A path that would escape the directory it is supposed to stay inside."""


def validate_run_id(run_id: str) -> str:
    """Reject a run id that is not a plain directory name.

    Run ids reach the filesystem, and some of them arrive from a manifest or a
    persisted proposal rather than from the person running the command.
    """

    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a non-empty path-safe name")
    return run_id


def contained_path(base: str | Path, relative: str | Path) -> Path:
    """Resolve ``relative`` under ``base``, refusing anything that escapes it.

    The prompt template is named by a manifest, and a manifest is a file people
    share.  Without this, ``template_file: "../../../.bashrc"`` would make
    ``apply`` — the one command that writes — write there.
    """

    root = Path(base).resolve()
    target = (root / Path(relative)).resolve()
    if target != root and root not in target.parents:
        raise PathContainmentError(f"{relative} resolves outside {root}")
    return target


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


def _pid_in(path: Path) -> int | None:
    """Read the owning pid out of a lock file, if it records one."""

    try:
        content = path.read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError):
        return None
    _, _, value = content.strip().partition("pid=")
    return int(value) if value.isdigit() else None


def _process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # The process exists and belongs to someone else.
        return True
    except OSError:
        return True
    return True


class StateLock:
    """An exclusive lock file acquired with ``O_EXCL``.

    The lock records its owning pid, so a lock left behind by a killed process
    can be reclaimed.  Without that, one ``SIGKILL`` would wedge the project
    until someone deleted a file they have no reason to know about — and the
    resume-after-a-kill path is exactly the path that needs the lock back.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._descriptor: int | None = None
        self.broke_stale_lock: bool = False

    def acquire(self) -> "StateLock":
        if self._descriptor is not None:
            raise StateLockError(f"lock already held by this owner: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = self._open()
        except FileExistsError as exc:
            owner = _pid_in(self.path)
            if owner is not None and not _process_is_alive(owner):
                self.path.unlink(missing_ok=True)
                self.broke_stale_lock = True
                try:
                    descriptor = self._open()
                except FileExistsError as race:
                    raise StateLockError(f"lock already held: {self.path}") from race
                # Two processes can see the same dead pid and both try to
                # reclaim, so confirm the file on disk is still the one this
                # process created before trusting the lock.
                if not self._owns_file(descriptor):
                    os.close(descriptor)
                    raise StateLockError(
                        f"lost the race to reclaim a stale lock: {self.path}"
                    )
            else:
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

    def _open(self) -> int:
        return os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)

    def _owns_file(self, descriptor: int) -> bool:
        """Whether the path still names the file this descriptor refers to."""

        try:
            held = os.fstat(descriptor)
            on_disk = os.stat(self.path)
        except OSError:
            return False
        return (held.st_dev, held.st_ino) == (on_disk.st_dev, on_disk.st_ino)

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
