"""CLI behaviour: argument handling, output, and the read/write split."""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

import pytest

from tracegrad import cli
from tracegrad.apply import build_proposal, save_proposal
from tracegrad.edits import resolve_edits
from tracegrad.gates import GateOutcome
from tracegrad.inventory import build_inventory
from tracegrad.schema import Cluster, Edit, Report
from tracegrad.state import atomic_write_json, initialize

EXAMPLE = Path(__file__).resolve().parents[1] / "example"
PROMPT = "Rules:\n- Be concise.\n- Cite the doc.\n"


def _run(*argv: str) -> tuple[int, str]:
    stream = io.StringIO()
    code = cli.main(list(argv), out=stream)
    return code, stream.getvalue()


def _example(tmp_path: Path) -> Path:
    shutil.copytree(EXAMPLE, tmp_path / "example")
    return tmp_path


def test_init_creates_the_state_directory(tmp_path: Path) -> None:
    code, output = _run("init", "--project-root", str(tmp_path))

    assert code == 0
    assert (tmp_path / ".tracegrad" / ".gitignore").is_file()
    assert "initialized" in output


def test_global_options_are_accepted_before_or_after_the_subcommand(tmp_path: Path) -> None:
    before = _run("--project-root", str(tmp_path), "init")
    after = _run("init", "--project-root", str(tmp_path))

    assert before[0] == after[0] == 0
    assert str(tmp_path) in before[1]
    assert str(tmp_path) in after[1]


def test_estimate_previews_cost_without_a_model(tmp_path: Path) -> None:
    project = _example(tmp_path)

    code, output = _run(
        "run",
        "--traces",
        str(project / "example" / "batch.jsonl"),
        "--manifest",
        str(project / "example" / "manifest.json"),
        "--project-root",
        str(project),
        "--base-directory",
        str(project / "example"),
        "--estimate",
    )

    assert code == 0
    assert "12 traces in the batch" in output
    assert "attribution call" in output


def test_run_reports_themes_drops_and_proposals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _example(tmp_path)
    from test_e2e import _attribution_response, _synthesis_response  # noqa: PLC0415

    from tracegrad.llm import FakeBackend  # noqa: PLC0415
    from tracegrad.pipeline import run_pipeline as real_run_pipeline  # noqa: PLC0415

    def fake_run_pipeline(*args: object, **kwargs: object):
        kwargs["attribution_backend"] = FakeBackend(handler=_attribution_response)
        kwargs["synthesis_backend"] = FakeBackend(handler=_synthesis_response)
        return real_run_pipeline(*args, **kwargs)

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    code, output = _run(
        "run",
        "--traces",
        str(project / "example" / "batch.jsonl"),
        "--manifest",
        str(project / "example" / "manifest.json"),
        "--project-root",
        str(project),
        "--base-directory",
        str(project / "example"),
    )

    assert code == 0
    assert "run run-0001" in output
    assert "1 x rationale-below-quality-floor" in output
    assert "missing-citation: 7/12" in output
    assert "1 edit(s) proposed" in output
    assert "evidence:" in output


def test_run_ids_increment(tmp_path: Path) -> None:
    layout = initialize(tmp_path)
    (layout.runs / "run-0001").mkdir(parents=True)
    (layout.runs / "run-0007").mkdir(parents=True)

    assert cli._next_run_id(tmp_path) == "run-0008"


def test_trends_needs_two_runs(tmp_path: Path) -> None:
    layout = initialize(tmp_path)
    atomic_write_json(
        layout.reports / "run-0001.json",
        Report(applied_prompt_hash="sha256:a", clusters=[]).model_dump(mode="json"),
    )

    code, output = _run("trends", "--project-root", str(tmp_path))

    assert code == 0
    assert "at least two runs" in output


def test_trends_prints_counts_intervals_and_the_advisory_note(tmp_path: Path) -> None:
    layout = initialize(tmp_path)
    for index, numerator in ((1, 40), (2, 10)):
        atomic_write_json(
            layout.reports / f"run-{index:04d}.json",
            Report(
                applied_prompt_hash="sha256:a",
                clusters=[Cluster(theme="tone", numerator=numerator, denominator=200)],
            ).model_dump(mode="json"),
        )

    code, output = _run("trends", "--project-root", str(tmp_path))

    assert code == 0
    assert "tone: 40/200" in output
    assert "95% CI" in output
    assert "improved" in output
    assert "never reverts on its own" in output


def test_status_reports_state_and_prompt_size(tmp_path: Path) -> None:
    project = _example(tmp_path)

    code, output = _run(
        "status",
        "--project-root",
        str(project),
        "--base-directory",
        str(project / "example"),
        "--manifest",
        str(project / "example" / "manifest.json"),
    )

    assert code == 0
    assert "runs recorded: 0" in output
    assert "instructions" in output
    assert "tokens (proxy count)" in output


def test_cli_commands_load_rc_from_project_root_not_base_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracegrad.llm import FakeBackend

    project_root = tmp_path / "project"
    base_directory = tmp_path / "base"
    project_root.mkdir()
    base_directory.mkdir()
    (project_root / ".tracegradrc").write_text(
        "minCoverage = 0.0\nminEffect = 0.1\nconvergenceRuns = 1\n",
        encoding="utf-8",
    )
    (base_directory / ".tracegradrc").write_text(
        "minCoverage = 1.0\nminEffect = 0.9\nconvergenceRuns = 99\n",
        encoding="utf-8",
    )

    prompt = base_directory / "prompt.md"
    prompt.write_text("- Be concise.\n", encoding="utf-8")
    manifest = base_directory / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "template_file": "prompt.md",
                "engine": "none",
                "judge_fingerprint": "test-judge",
            }
        ),
        encoding="utf-8",
    )
    traces = base_directory / "traces.jsonl"
    traces.write_text(
        json.dumps(
            {
                "trace_id": "trace-1",
                "input": "Question?",
                "output": "Answer.",
                "judge": {
                    "score": 0.0,
                    "rationale": "A rationale long enough to pass ingestion.",
                },
                "prompt_hash": "sha256:test",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    layout = initialize(project_root)
    for index, numerator in ((1, 80), (2, 20)):
        atomic_write_json(
            layout.reports / f"run-{index:04d}.json",
            Report(
                applied_prompt_hash="sha256:test",
                clusters=[Cluster(theme="drift", numerator=numerator, denominator=100)],
            ).model_dump(mode="json"),
        )
    (layout.ledgers / "runs.jsonl").write_text('{"proposed": 0}\n', encoding="utf-8")

    monkeypatch.setattr(cli, "resolve_attribution_backend", lambda _config: FakeBackend())

    attribute_code, attribute_output = _run(
        "attribute",
        "--traces",
        str(traces),
        "--manifest",
        str(manifest),
        "--project-root",
        str(project_root),
        "--base-directory",
        str(base_directory),
    )
    assert attribute_code == 0
    assert attribute_output == "attributed 0/1 traces\n"

    trends_code, trends_output = _run(
        "trends",
        "--project-root",
        str(project_root),
        "--base-directory",
        str(base_directory),
    )
    assert trends_code == 0
    assert "improved" in trends_output
    assert "no-signal" not in trends_output

    status_code, status_output = _run(
        "status",
        "--project-root",
        str(project_root),
        "--base-directory",
        str(base_directory),
    )
    assert status_code == 0
    assert "runs recorded: 1" in status_output
    assert "converged: 1 consecutive runs proposed nothing" in status_output



def _saved_proposal(project: Path) -> None:
    template = project / "prompt.md"
    template.write_text(PROMPT, encoding="utf-8")
    inventory = build_inventory(PROMPT)
    resolution = resolve_edits(
        inventory,
        [
            Edit(
                instruction_id=inventory.instructions[-1].instruction_id,
                operation="REWRITE",
                text="Always cite the doc.",
                covers_theme="missing-citation",
                watch_metric="missing-citation",
            )
        ],
    )
    proposal = build_proposal(
        run_id="run-0001",
        template_file="prompt.md",
        prompt=PROMPT,
        outcome=GateOutcome(kept=resolution.resolved, rejected=()),
    )
    save_proposal(project, proposal)


def test_apply_all_writes_the_template(tmp_path: Path) -> None:
    _saved_proposal(tmp_path)

    code, output = _run(
        "apply", "--all", "--project-root", str(tmp_path), "--base-directory", str(tmp_path)
    )

    assert code == 0
    assert "applied 1 edit(s)" in output
    assert "Always cite the doc." in (tmp_path / "prompt.md").read_text(encoding="utf-8")


def test_apply_accepts_selected_indices(tmp_path: Path) -> None:
    _saved_proposal(tmp_path)

    code, _ = _run(
        "apply",
        "--accept",
        "0",
        "--project-root",
        str(tmp_path),
        "--base-directory",
        str(tmp_path),
    )

    assert code == 0
    assert "Always cite the doc." in (tmp_path / "prompt.md").read_text(encoding="utf-8")


def test_apply_refuses_a_stale_proposal(tmp_path: Path) -> None:
    _saved_proposal(tmp_path)
    (tmp_path / "prompt.md").write_text(PROMPT + "- Edited by hand.\n", encoding="utf-8")

    code, output = _run(
        "apply", "--all", "--project-root", str(tmp_path), "--base-directory", str(tmp_path)
    )

    assert code == 1
    assert "stale" in output


def test_apply_records_rejections_for_the_memory_gate(tmp_path: Path) -> None:
    _saved_proposal(tmp_path)

    _run(
        "apply",
        "--accept",
        "",
        "--project-root",
        str(tmp_path),
        "--base-directory",
        str(tmp_path),
    )

    ledger = tmp_path / ".tracegrad" / "ledgers" / "rejections.jsonl"
    assert ledger.is_file()
    assert json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])["event"] == "rejected"


def test_apply_reverts_from_the_snapshot(tmp_path: Path) -> None:
    _saved_proposal(tmp_path)
    _run("apply", "--all", "--project-root", str(tmp_path), "--base-directory", str(tmp_path))

    code, output = _run(
        "apply", "--revert", "--project-root", str(tmp_path), "--base-directory", str(tmp_path)
    )

    assert code == 0
    assert "reverted" in output
    assert (tmp_path / "prompt.md").read_text(encoding="utf-8") == PROMPT


def test_apply_without_a_proposal_fails_cleanly(tmp_path: Path) -> None:
    code, output = _run("apply", "--all", "--project-root", str(tmp_path))

    assert code == 1
    assert "no proposal" in output


def test_errors_are_reported_without_a_traceback(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, _ = _run(
        "run",
        "--traces",
        str(tmp_path / "missing.jsonl"),
        "--manifest",
        str(tmp_path / "missing.json"),
        "--project-root",
        str(tmp_path),
    )

    assert code == 1
    assert "tracegrad:" in capsys.readouterr().err
