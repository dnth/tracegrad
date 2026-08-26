"""Core-only regression: Kitaru is optional (issue #8 / #9)."""

from __future__ import annotations

import ast
import io
import sys
from pathlib import Path

import pytest

from tracegrad import cli
from tracegrad.integrations.kitaru.errors import (
    INSTALL_MESSAGE,
    KITARU_PIN,
    NO_BACKEND_MESSAGE,
    KitaruNotInstalled,
)
from tracegrad.integrations.kitaru.require import kitaru_available, require_kitaru

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "tracegrad"


def test_import_tracegrad_does_not_import_kitaru() -> None:
    sys.modules.pop("kitaru", None)
    import tracegrad as package

    assert package.__version__
    assert "kitaru" not in sys.modules


def test_only_the_integration_package_imports_the_kitaru_sdk() -> None:
    offenders: list[str] = []
    for source in PACKAGE.rglob("*.py"):
        rel = source.relative_to(PACKAGE)
        if rel.parts[:2] == ("integrations", "kitaru"):
            continue
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            if any(name == "kitaru" or name.startswith("kitaru.") for name in names):
                offenders.append(str(rel))
    assert offenders == [], offenders


def test_kitaru_pin_matches_the_issue() -> None:
    assert KITARU_PIN == "kitaru>=0.22,<0.23"
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert 'kitaru = ["kitaru>=0.22,<0.23"]' in text
    assert "kitaru>=" not in text.split("[project]")[1].split("[project.optional-dependencies]")[0]


def test_require_kitaru_is_actionable_when_missing() -> None:
    if kitaru_available():
        return
    try:
        require_kitaru()
    except KitaruNotInstalled as exc:
        assert "tracegrad[kitaru]" in str(exc)
        assert "uv tool install" in str(exc)
        assert INSTALL_MESSAGE.splitlines()[0] in str(exc)
    else:
        raise AssertionError("expected KitaruNotInstalled")


def test_source_kitaru_without_the_extra_is_actionable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    if kitaru_available():
        return
    stream = io.StringIO()
    code = cli.main(
        [
            "run",
            "--source",
            "kitaru",
            "--kitaru-cohort",
            "support-production",
            "--kitaru-evaluation",
            "quality",
            "--manifest",
            str(tmp_path / "missing.json"),
            "--project-root",
            str(tmp_path),
        ],
        out=stream,
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "tracegrad[kitaru]" in err
    assert "ImportError" not in err


def test_verify_without_a_backend_exits_nonzero(tmp_path: Path) -> None:
    stream = io.StringIO()
    code = cli.main(["verify", "--project-root", str(tmp_path)], out=stream)
    assert code == 1
    output = stream.getvalue()
    assert "needs a backend" in output
    assert NO_BACKEND_MESSAGE.splitlines()[0] in output


def test_verify_does_not_break_run_apply_or_trends(tmp_path: Path) -> None:
    """A missing backend is a verify failure, not a lock on the rest of the CLI."""

    init_code, _ = _run("init", "--project-root", str(tmp_path))
    trends_code, trends_out = _run("trends", "--project-root", str(tmp_path))
    apply_code, apply_out = _run("apply", "--all", "--project-root", str(tmp_path))
    assert init_code == 0
    assert trends_code == 0
    assert "at least two runs" in trends_out
    assert apply_code == 1
    assert "no proposal" in apply_out


def _run(*argv: str) -> tuple[int, str]:
    stream = io.StringIO()
    code = cli.main(list(argv), out=stream)
    return code, stream.getvalue()


def test_traces_and_source_are_mutually_exclusive(tmp_path: Path) -> None:
    stream = io.StringIO()
    code = cli.main(
        [
            "run",
            "--traces",
            str(tmp_path / "batch.jsonl"),
            "--source",
            "kitaru",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--project-root",
            str(tmp_path),
        ],
        out=stream,
    )
    assert code == 1
