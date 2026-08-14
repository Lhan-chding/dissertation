from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from compbias.recoverability.evidence import (
    SERVER_PACKAGE_PATHS,
    verify_server_package_lock,
)
from compbias.recoverability.preflight import (
    load_runtime_spec,
    run_metadata_preflight,
)

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "configs" / "recoverability" / "server_runtime_v1.yaml"
SERVER_LOCK = ROOT / "configs" / "recoverability" / "server_package_lock_v1.yaml"


def _versions() -> dict[str, str]:
    return {
        "accelerate": "1.14.0",
        "numpy": "2.5.2",
        "peft": "0.19.1",
        "qwen-vl-utils": "0.0.14",
        "scipy": "1.18.0",
        "torch": "2.8.0+cu128",
        "transformers": "5.14.1",
    }


def _inventory() -> dict[str, str]:
    result: dict[str, str] = {}
    for line in (ROOT / "requirements-gpu.lock.txt").read_text(encoding="utf-8").splitlines():
        name, version = line.split("==", 1)
        result[name] = version
    return result


def test_runtime_spec_is_exact_offline_and_never_authorizes_training() -> None:
    spec = load_runtime_spec(RUNTIME)

    assert dict(spec.exact_packages) == _versions()
    assert spec.requirements_lock_path == "requirements-gpu.lock.txt"
    assert spec.requirements_lock_sha256 == (
        "d928379a590e5071d9b5042fe99d480f57ab187f0cb3a74e13af219a6048aeb3"
    )
    assert spec.offline_required is True
    assert spec.downloads_allowed is False
    assert spec.model_loading_allowed is False
    assert spec.training_authorized is False


def test_metadata_preflight_passes_without_importing_or_loading_the_model() -> None:
    lookups: list[str] = []

    def lookup(package: str) -> str:
        lookups.append(package)
        return _versions()[package]

    report = run_metadata_preflight(
        load_runtime_spec(RUNTIME),
        repository_root=ROOT,
        version_lookup=lookup,
        inventory_lookup=_inventory,
        pip_check=lambda: (0, "No broken requirements found."),
        environ={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )

    assert report.ready is True
    assert report.large_gpu_started is False
    assert report.model_loaded is False
    assert report.training_authorized is False
    assert report.pip_check_passed is True
    assert tuple(lookups) == tuple(_versions())


@pytest.mark.parametrize(
    ("versions", "message"),
    [
        ({**_versions(), "transformers": "5.14.2"}, "version mismatch"),
        ({key: value for key, value in _versions().items() if key != "torch"}, "missing package"),
    ],
)
def test_metadata_preflight_fails_closed_on_missing_or_mismatched_packages(
    versions: dict[str, str], message: str
) -> None:
    def lookup(package: str) -> str:
        if package not in versions:
            raise LookupError(package)
        return versions[package]

    with pytest.raises(RuntimeError, match=message):
        run_metadata_preflight(
            load_runtime_spec(RUNTIME),
            repository_root=ROOT,
            version_lookup=lookup,
            inventory_lookup=_inventory,
            pip_check=lambda: (0, "No broken requirements found."),
            environ={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        )


def test_preflight_rejects_online_mode_or_failed_dependency_check() -> None:
    spec = load_runtime_spec(RUNTIME)
    lookup = _versions().__getitem__
    with pytest.raises(RuntimeError, match="offline"):
        run_metadata_preflight(
            spec,
            repository_root=ROOT,
            version_lookup=lookup,
            inventory_lookup=_inventory,
            pip_check=lambda: (0, "No broken requirements found."),
            environ={},
        )
    with pytest.raises(RuntimeError, match="pip check"):
        run_metadata_preflight(
            spec,
            repository_root=ROOT,
            version_lookup=lookup,
            inventory_lookup=_inventory,
            pip_check=lambda: (1, "broken"),
            environ={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        )


def test_server_package_lock_binds_runtime_protocol_runners_and_parsers() -> None:
    result = verify_server_package_lock(SERVER_LOCK, repository_root=ROOT)

    assert result.verified is True
    assert {item.relative_path for item in result.files} == SERVER_PACKAGE_PATHS


def test_server_package_lock_rejects_caller_selected_subset(tmp_path: Path) -> None:
    subset = tmp_path / "server_package_lock_v1.yaml"
    subset.write_text(
        "schema_version: 1\nfiles:\n"
        "  - path: README.md\n"
        "    sha256: "
        + __import__("hashlib").sha256((ROOT / "README.md").read_bytes()).hexdigest()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical"):
        verify_server_package_lock(subset, repository_root=ROOT)


def test_preflight_cli_has_no_execute_or_model_loading_surface() -> None:
    script = ROOT / "experiments" / "recoverability_v1" / "00_preflight.py"
    help_result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    source = script.read_text(encoding="utf-8")

    assert "--runtime" in help_result.stdout
    assert "--server-package-lock" in help_result.stdout
    assert "--output" in help_result.stdout
    assert "--execute" not in help_result.stdout
    assert "import torch" not in source
    assert "from_pretrained" not in source
