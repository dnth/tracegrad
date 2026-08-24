from pathlib import Path

import pytest

from tracegrad.state import (
    StateLock,
    StateLockError,
    append_jsonl,
    atomic_write,
    initialize,
    load_jsonl,
    load_resume_state,
    save_resume_state,
)


def test_lock_is_exclusive(tmp_path: Path) -> None:
    lock_path = tmp_path / "run.lock"
    first = StateLock(lock_path)
    first.acquire()
    try:
        with pytest.raises(StateLockError, match="already held"):
            StateLock(lock_path).acquire()
    finally:
        first.release()

    assert not lock_path.exists()


def test_atomic_write_replaces_without_temp_artifacts(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "state.json"

    atomic_write(target, b"old")
    atomic_write(target, b"new")

    assert target.read_bytes() == b"new"
    assert list(target.parent.iterdir()) == [target]


def test_jsonl_append_is_ordered_and_rereadable(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    append_jsonl(ledger, {"run": 1, "ok": True})
    append_jsonl(ledger, {"run": 2, "ok": False})

    assert load_jsonl(ledger) == [
        {"run": 1, "ok": True},
        {"run": 2, "ok": False},
    ]


def test_resume_state_persists_and_reloads(tmp_path: Path) -> None:
    initialize(tmp_path)
    expected = {"phase": "attribute", "completed": ["trace-1"], "attempt": 2}

    save_resume_state(tmp_path, "run-1", expected)

    assert load_resume_state(tmp_path, "run-1") == expected


def test_initialize_creates_layout_and_gitignore(tmp_path: Path) -> None:
    layout = initialize(tmp_path)

    assert layout.root == tmp_path / ".tracegrad"
    assert layout.gitignore.read_text(encoding="utf-8") == "*\n!.gitignore\n"
    for directory in layout.data_directories:
        assert directory.is_dir()
    assert layout.resume_directory.is_dir()
