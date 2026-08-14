"""Repository-level contracts for truthful whole-project coverage measurement."""

from __future__ import annotations

from pathlib import Path

from coverage import Coverage

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_coverage_configuration_measures_library_and_cli_branches() -> None:
    configuration = Coverage(config_file=str(REPOSITORY_ROOT / "pyproject.toml")).config

    assert configuration.source == ["compbias", "scripts"]
    assert configuration.branch is True
    assert configuration.parallel is True
    assert configuration.patch == ["subprocess"]
    assert configuration.fail_under == 80
