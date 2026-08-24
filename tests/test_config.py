from pathlib import Path

import pytest

from tracegrad.config import ConfigError, load_config, load_rc


def test_defaults_apply_when_rc_is_absent(tmp_path: Path) -> None:
    config = load_config(tmp_path)

    assert config.neverDelete == []
    assert config.minEffect == 0.05
    assert config.minCoverage == 0.8
    assert config.convergenceRuns == 2
    assert config.harness_presets["attribution"].provider == "openai"
    assert config.harness["synthesis"].provider == "claude"


def test_each_rc_field_parses(tmp_path: Path) -> None:
    (tmp_path / ".tracegradrc").write_text(
        """
neverDelete = ["keep/**", "README.md"]
minEffect = 0.12
minCoverage = 0.9
convergenceRuns = 4

[harness_presets.local]
provider = "command"
command = ["python", "harness.py"]
timeoutSeconds = 30

[harness_presets.remote]
provider = "openai"
model = "gpt-test"
""".lstrip(),
        encoding="utf-8",
    )

    config = load_rc(tmp_path)

    assert config.neverDelete == ["keep/**", "README.md"]
    assert config.minEffect == 0.12
    assert config.minCoverage == 0.9
    assert config.convergenceRuns == 4
    assert config.harness_presets["local"].command == ["python", "harness.py"]
    assert config.harness_presets["local"].timeoutSeconds == 30
    assert config.harness_presets["remote"].model == "gpt-test"


def test_malformed_toml_is_rejected_with_filename(tmp_path: Path) -> None:
    rc = tmp_path / ".tracegradrc"
    rc.write_text("minEffect = [", encoding="utf-8")

    with pytest.raises(ConfigError, match=r"\.tracegradrc"):
        load_config(rc)


def test_invalid_values_are_rejected_with_field_name(tmp_path: Path) -> None:
    rc = tmp_path / ".tracegradrc"
    rc.write_text("minCoverage = 1.5\nconvergenceRuns = 0\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="minCoverage|convergenceRuns"):
        load_config(rc)
