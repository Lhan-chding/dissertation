"""Regressions against template promotion and stale/wrong-phase evidence."""

import copy
import json
from pathlib import Path

import pytest

from sr_f6.common import digest, file_hash
from sr_f6.contract import AMENDMENTS, validate_config
from sr_f6.gates import PREFLIGHT_REQUIRED, launch_check, require_census_ready

CONFIG = Path(__file__).resolve().parents[2] / "configs/sr_f6.json"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def preflight(root):
    (root / "CONFIG.json").write_bytes(CONFIG.read_bytes())
    repo = CONFIG.parent.parent
    source_files = list((repo / "src").rglob("*.py")) + list((repo / "tests/sr_f6").rglob("*.py"))
    write(
        root / "source.json",
        {
            "files": [
                {
                    "path": str(p.relative_to(repo)),
                    "bytes": p.stat().st_size,
                    "sha256": file_hash(p),
                }
                for p in source_files
            ]
        },
    )
    write(root / "data_member.json", {"fixture": "data"})
    write(
        root / "data.json",
        {
            "files": [
                {
                    "path": "data_member.json",
                    "bytes": (root / "data_member.json").stat().st_size,
                    "sha256": file_hash(root / "data_member.json"),
                }
            ]
        },
    )
    identity = {
        "config_sha256": file_hash(root / "CONFIG.json"),
        "implementation_digest": digest(AMENDMENTS),
        "source_manifest": {"path": "source.json", "sha256": file_hash(root / "source.json")},
        "data_manifest": {"path": "data.json", "sha256": file_hash(root / "data.json")},
    }
    write(root / "IDENTITY.json", identity)
    write(
        root / "evidence.json",
        {"status": "PASS", "context": identity, "checks": dict.fromkeys(PREFLIGHT_REQUIRED, True)},
    )
    receipt = {
        "status": "READY_FOR_CENSUS",
        "phase": "DEV_CAL",
        "execution_origin": "SERVER_ACTUAL",
        "optimizer_updates": 0,
        "context": identity,
        "evidence": [
            {
                "path": "evidence.json",
                "sha256": file_hash(root / "evidence.json"),
                "checks": sorted(PREFLIGHT_REQUIRED),
            }
        ],
    }
    write(root / "PREFLIGHT/CENSUS_READY.json", receipt)
    return receipt


def test_cpu_template_cannot_unlock_formal(tmp_path):
    (tmp_path / "CONFIG.json").write_bytes(CONFIG.read_bytes())
    for name in (
        "SOURCE_BINDING",
        "DATA_AND_MEASUREMENT",
        "PARENT_CENSUS",
        "RUNTIME_PILOT",
        "FORMAL_FREEZE",
    ):
        write(tmp_path / "GATES" / (name + ".json"), {"status": "PASS"})
    assert launch_check(tmp_path)["allowed"] is False


@pytest.mark.parametrize(
    "mutation", ["phase", "origin", "config", "evidence", "escape", "missing_check", "symlink"]
)
def test_invalid_preflight_is_rejected(tmp_path, mutation):
    receipt = preflight(tmp_path)
    if mutation == "phase":
        receipt["phase"] = "TEST_COMPOSITION"
    elif mutation == "origin":
        receipt["execution_origin"] = "CPU_ONLY"
    elif mutation == "config":
        config = json.loads(CONFIG.read_text())
        config["training"]["aux_coefficient"] = 0.25
        write(tmp_path / "CONFIG.json", config)
    elif mutation == "evidence":
        write(tmp_path / "evidence.json", {"status": "PASS"})
    elif mutation == "escape":
        receipt["evidence"][0]["path"] = "../evidence.json"
    elif mutation == "missing_check":
        receipt["evidence"][0]["checks"].pop()
    elif mutation == "symlink":
        (tmp_path / "linked.json").symlink_to(tmp_path / "evidence.json")
        receipt["evidence"][0]["path"] = "linked.json"
    write(tmp_path / "PREFLIGHT/CENSUS_READY.json", receipt)
    with pytest.raises((PermissionError, ValueError)):
        require_census_ready(tmp_path)


def test_bound_preflight_validates_but_does_not_unlock_formal(tmp_path):
    preflight(tmp_path)
    require_census_ready(tmp_path)
    assert not launch_check(tmp_path)["allowed"]


def test_recipe_drift_fails_before_outputs():
    original = json.loads(CONFIG.read_text())
    validate_config(original)
    for field, value in (("updates", 73), ("max_new_tokens", 1024), ("repeats", [0, 1])):
        changed = copy.deepcopy(original)
        changed["training"][field] = value
        with pytest.raises(PermissionError):
            validate_config(changed)


def test_manifest_member_changes_cannot_reuse_gate(tmp_path):
    preflight(tmp_path)
    write(tmp_path / "data_member.json", {"fixture": "changed after freeze"})
    with pytest.raises(PermissionError):
        require_census_ready(tmp_path)


def test_unchanged_members_reuse_verified_hashes(tmp_path, monkeypatch):
    import sr_f6.gates as gates

    preflight(tmp_path)
    require_census_ready(tmp_path)
    calls = []
    original = gates.file_hash

    def observe(path):
        calls.append(Path(path))
        return original(path)

    monkeypatch.setattr(gates, "file_hash", observe)
    require_census_ready(tmp_path)
    assert tmp_path / "data_member.json" not in calls
    assert not any(path.suffix == ".py" for path in calls)
    member = tmp_path / "data_member.json"
    member.write_text(member.read_text().replace("data", "else"))
    with pytest.raises(PermissionError, match="member changed"):
        require_census_ready(tmp_path)
    assert member in calls
