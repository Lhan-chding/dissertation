"""Security contracts for CPU dependencies used by the release surface."""

from __future__ import annotations

from pathlib import Path

import tomllib
from packaging.requirements import Requirement

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_pillow_constraint_and_snapshot_require_all_2026_security_fixes() -> None:
    project = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = tuple(Requirement(value) for value in project["project"]["dependencies"])
    pillow = next(
        requirement for requirement in requirements if requirement.name.lower() == "pillow"
    )

    assert "12.1.1" not in pillow.specifier
    assert "12.2.0" not in pillow.specifier
    assert "12.3.0" in pillow.specifier

    snapshot = (REPOSITORY_ROOT / "requirements-lock.txt").read_text(encoding="utf-8")
    assert "pillow==12.3.0" in snapshot.lower().splitlines()
    assert "pillow==11.3.0" not in snapshot.lower().splitlines()


def test_pytest_constraint_and_snapshot_exclude_cve_2025_71176() -> None:
    project = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = tuple(
        Requirement(value) for value in project["project"]["optional-dependencies"]["dev"]
    )
    pytest_requirement = next(
        requirement for requirement in requirements if requirement.name.lower() == "pytest"
    )

    assert "8.4.2" not in pytest_requirement.specifier
    assert "9.0.3" in pytest_requirement.specifier

    snapshot = (REPOSITORY_ROOT / "requirements-lock.txt").read_text(encoding="utf-8")
    assert "pytest==9.0.3" in snapshot.lower().splitlines()
    assert "pytest==8.4.2" not in snapshot.lower().splitlines()
