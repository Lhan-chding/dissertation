"""CPU acceptance reuse authenticates existing evidence without deleting it."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from sr_f1.contract import file_hash
from sr_f2.protocol import build_config

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/sr_f2/cpu_preflight.py"
SPEC = importlib.util.spec_from_file_location("sr_f2_cpu_preflight_test", SCRIPT)
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)
SOURCE_HASH = "a" * 64


def snapshot(directory):
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


@pytest.fixture
def accepted(tmp_path):
    root = tmp_path / "run"
    attempt = root / "deployment/cpu_attempts/original-attempt"
    attempt.mkdir(parents=True)
    checks = []
    for index, suite in enumerate(PREFLIGHT.SUITES):
        log = attempt / f"cpu-checks-{index}.log"
        log.write_text(f"{suite}: 1 passed\n")
        checks.append(
            dict(
                command=[sys.executable, "-m", "pytest", "-p", "no:cacheprovider", suite, "-q"],
                exit_code=0,
                log=str(log),
                log_sha256=file_hash(log),
            )
        )
    prompts = root / "NATIVE_DIAGNOSTIC_PROMPTS.json"
    prompts.write_text(json.dumps([dict(qid="q0", view="V0", request=dict(text="question"))]))
    receipt = dict(
        status="PASS",
        source_manifest_sha256=SOURCE_HASH,
        checks=checks,
        native_diagnostic_prompts_sha256=file_hash(prompts),
    )
    (root / "CPU_ACCEPTANCE.json").write_text(json.dumps(receipt))
    return root, receipt


def test_missing_acceptance_is_not_reused_or_created(tmp_path):
    assert PREFLIGHT.reuse_cpu_acceptance(tmp_path, SOURCE_HASH) is False
    assert snapshot(tmp_path) == {}


def test_complete_matching_acceptance_reuses_three_suite_evidence_unchanged(accepted):
    root, receipt = accepted
    before = snapshot(root)
    assert len(receipt["checks"]) == len(PREFLIGHT.SUITES) == 3
    assert PREFLIGHT.reuse_cpu_acceptance(root, SOURCE_HASH) is True
    assert snapshot(root) == before


@pytest.mark.parametrize(
    "defect",
    ["source", "log", "exit", "escaped_log", "escaped_symlink", "native_prompt", "command"],
)
def test_changed_acceptance_is_rejected_and_every_file_preserved(accepted, tmp_path, defect):
    root, receipt = accepted
    supplied_source = SOURCE_HASH
    if defect == "source":
        supplied_source = "b" * 64
    elif defect == "log":
        Path(receipt["checks"][0]["log"]).write_text("changed log\n")
    elif defect == "exit":
        receipt["checks"][0]["exit_code"] = 1
    elif defect in ("escaped_log", "escaped_symlink"):
        escaped = tmp_path / "outside.log"
        escaped.write_text("preserved outside log\n")
        if defect == "escaped_symlink":
            link = root / "deployment/cpu_attempts/linked.log"
            link.symlink_to(escaped)
            receipt["checks"][0]["log"] = str(link)
        else:
            receipt["checks"][0]["log"] = str(escaped)
        receipt["checks"][0]["log_sha256"] = file_hash(escaped)
    elif defect == "native_prompt":
        (root / "NATIVE_DIAGNOSTIC_PROMPTS.json").write_text("[]")
    else:
        receipt["checks"][0]["command"] = [sys.executable, "-m", "pytest", "-q"]
    (root / "CPU_ACCEPTANCE.json").write_text(json.dumps(receipt))
    before = snapshot(tmp_path)
    with pytest.raises(PermissionError):
        PREFLIGHT.reuse_cpu_acceptance(root, supplied_source)
    assert snapshot(tmp_path) == before
    assert (root / "CPU_ACCEPTANCE.json").exists()


def test_registered_scheduling_uses_four_two_gpu_pools_without_seed_waves():
    resources = build_config()["resources"]
    assert resources["scheduling"] == "persistent_pools_shared_dependency_queue"
    assert resources["seed_first_waves"] is False
    assert resources["max_workers_across_active_project_runs"] == 8
    assert resources["gpus_per_packed_job"] == 2
    assert resources["persistent_pool_count"] == 4
    assert resources["workers_per_gpu"] == 1
