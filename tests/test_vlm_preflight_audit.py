"""End-to-end checks for the non-executing large-GPU readiness audit."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "preflight_vlm.py"
SFT_CONFIG = REPOSITORY_ROOT / "configs" / "vlm" / "qwen25vl3b_sft.yaml"
RL_CONFIG = REPOSITORY_ROOT / "configs" / "vlm" / "qwen25vl3b_joint_grpo.yaml"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("preflight_vlm_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_audit_records_the_gpu_boundary_without_starting_or_acknowledging(tmp_path) -> None:
    output = tmp_path / "preflight.json"

    completed = _run(
        "--sft-config",
        str(SFT_CONFIG),
        "--rl-config",
        str(RL_CONFIG),
        "--output",
        str(output),
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["audit_completed"] is True
    assert payload["large_gpu_started"] is False
    assert payload["large_gpu_acknowledged"] is False
    assert payload["ready_for_large_gpu_execution"] is False
    assert "large_gpu_acknowledgement_not_granted" in payload["blockers"]
    assert payload["pins"]["model_revision"]
    assert payload["pins"]["verl_revision"]
    assert payload["compatibility"]["prohibited_install"] == "pip install -e .[vllm]"
    assert payload["compatibility"]["dockerfile_sha256"]
    assert payload["compatibility"]["upstream_dockerfile_reproducible_as_is"] is False
    assert payload["compatibility"]["hardened_descendant_vendor_pending"] is True
    assert "hardened_descendant_container_pending" in payload["blockers"]
    assert "container_sbom_pending" in payload["blockers"]
    assert "container_vulnerability_policy_audit_pending" in payload["blockers"]
    assert payload["next_authorized_action"].startswith("vendor a hardened descendant")
    assert payload["reward_contract"] == {
        "outcome_only": True,
        "perception_reward_weight": 0.0,
        "process_reward_weight": 0.0,
    }


def test_audit_rejects_a_config_whose_pin_drifts_from_the_frozen_contract(
    tmp_path,
) -> None:
    raw = yaml.safe_load(SFT_CONFIG.read_text(encoding="utf-8"))
    raw["model"]["revision"] = "different-model-revision"
    drifted = tmp_path / "drifted.yaml"
    drifted.write_text(yaml.safe_dump(raw), encoding="utf-8")

    completed = _run(
        "--sft-config",
        str(drifted),
        "--rl-config",
        str(RL_CONFIG),
        "--output",
        str(tmp_path / "unused.json"),
    )

    assert completed.returncode != 0
    assert "model revision" in completed.stderr.lower()
    assert "pinned" in completed.stderr.lower()


def test_metadata_preflight_rejects_supply_chain_status_self_attestation(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(SFT_CONFIG.read_text(encoding="utf-8"))
    raw["compatibility_audit"]["status"] = "verified_on_target_hardware"
    raw["compatibility_audit"]["upstream_dockerfile_reproducible_as_is"] = True
    raw["compatibility_audit"]["hardened_descendant_vendor_pending"] = False
    self_attested = tmp_path / "self-attested.yaml"
    self_attested.write_text(yaml.safe_dump(raw), encoding="utf-8")

    completed = _run(
        "--sft-config",
        str(self_attested),
        "--rl-config",
        str(RL_CONFIG),
        "--output",
        str(tmp_path / "unused.json"),
    )

    assert completed.returncode != 0
    assert "hardened" in completed.stderr.lower() or "reproducib" in completed.stderr.lower()


def test_metadata_preflight_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    duplicated = tmp_path / "duplicate.yaml"
    duplicated.write_text(
        SFT_CONFIG.read_text(encoding="utf-8") + "\nmodel: {}\n",
        encoding="utf-8",
    )

    completed = _run(
        "--sft-config",
        str(duplicated),
        "--rl-config",
        str(RL_CONFIG),
        "--output",
        str(tmp_path / "unused.json"),
    )

    assert completed.returncode != 0
    assert "duplicate" in completed.stderr.lower()


def test_metadata_preflight_rejects_yaml_larger_than_one_mib(tmp_path: Path) -> None:
    oversized = tmp_path / "oversized.yaml"
    oversized.write_bytes(b"#" + (b"x" * (1024 * 1024)) + b"\n")

    completed = _run(
        "--sft-config",
        str(oversized),
        "--rl-config",
        str(RL_CONFIG),
        "--output",
        str(tmp_path / "unused.json"),
    )

    assert completed.returncode != 0
    assert "1 mib" in completed.stderr.lower()
    assert "traceback" not in completed.stderr.lower()


def test_metadata_preflight_rejects_yaml_deeper_than_64_levels(tmp_path: Path) -> None:
    too_deep = tmp_path / "too-deep.yaml"
    too_deep.write_text("unknown: " + ("[" * 65) + "0" + ("]" * 65), encoding="utf-8")

    completed = _run(
        "--sft-config",
        str(too_deep),
        "--rl-config",
        str(RL_CONFIG),
        "--output",
        str(tmp_path / "unused.json"),
    )

    assert completed.returncode != 0
    assert "nesting depth" in completed.stderr.lower()
    assert "64" in completed.stderr
    assert "traceback" not in completed.stderr.lower()


def test_metadata_preflight_rejects_yaml_with_more_than_100000_nodes(tmp_path: Path) -> None:
    too_many_nodes = tmp_path / "too-many-nodes.yaml"
    too_many_nodes.write_text(
        "unknown: [" + ("null," * 100_001) + "]",
        encoding="utf-8",
    )

    completed = _run(
        "--sft-config",
        str(too_many_nodes),
        "--rl-config",
        str(RL_CONFIG),
        "--output",
        str(tmp_path / "unused.json"),
    )

    assert completed.returncode != 0
    assert "node count" in completed.stderr.lower()
    assert "100000" in completed.stderr
    assert "traceback" not in completed.stderr.lower()


def test_metadata_preflight_normalizes_yaml_recursion_errors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    config = tmp_path / "config.yaml"
    config.write_text("key: value\n", encoding="utf-8")

    def recurse_forever(*_args, **_kwargs):
        raise RecursionError("synthetic parser recursion")

    monkeypatch.setattr(yaml, "load", recurse_forever)

    with pytest.raises(ValueError, match=r"cannot parse test configuration.*nesting"):
        module._load_yaml(config, "test configuration")


def test_metadata_preflight_rejects_yaml_alias_cycles_without_traceback(tmp_path: Path) -> None:
    cyclic = tmp_path / "cyclic.yaml"
    cyclic.write_text(
        "model: &model\n"
        "  name: Qwen/Qwen2.5-VL-3B-Instruct\n"
        "  revision: 66285546d2b821cf421d4f5eb2576359d3770cd3\n"
        "  extra: *model\n",
        encoding="utf-8",
    )

    completed = _run(
        "--sft-config",
        str(cyclic),
        "--rl-config",
        str(RL_CONFIG),
        "--output",
        str(tmp_path / "unused.json"),
    )

    assert completed.returncode != 0
    assert "alias" in completed.stderr.lower()
    assert "traceback" not in completed.stderr.lower()


def test_metadata_preflight_rejects_unknown_top_level_field(tmp_path: Path) -> None:
    raw = yaml.safe_load(SFT_CONFIG.read_text(encoding="utf-8"))
    raw["unexpected_permission"] = True
    unknown = tmp_path / "unknown-top-level.yaml"
    unknown.write_text(yaml.safe_dump(raw), encoding="utf-8")

    completed = _run(
        "--sft-config",
        str(unknown),
        "--rl-config",
        str(RL_CONFIG),
        "--output",
        str(tmp_path / "unused.json"),
    )

    assert completed.returncode != 0
    assert "unknown field" in completed.stderr.lower()
    assert "unexpected_permission" in completed.stderr


def test_metadata_preflight_rejects_unknown_nested_field(tmp_path: Path) -> None:
    raw = yaml.safe_load(RL_CONFIG.read_text(encoding="utf-8"))
    raw["reward"]["allow_process_override"] = True
    unknown = tmp_path / "unknown-nested.yaml"
    unknown.write_text(yaml.safe_dump(raw), encoding="utf-8")

    completed = _run(
        "--sft-config",
        str(SFT_CONFIG),
        "--rl-config",
        str(unknown),
        "--output",
        str(tmp_path / "unused.json"),
    )

    assert completed.returncode != 0
    assert "unknown field" in completed.stderr.lower()
    assert "rl reward" in completed.stderr.lower()
    assert "allow_process_override" in completed.stderr


@pytest.mark.parametrize("invalid_rate", [float("nan"), float("inf"), 0.97, 1.01, "0.98"])
def test_metadata_preflight_rejects_invalid_parser_validity_gate(
    tmp_path: Path,
    invalid_rate: object,
) -> None:
    raw = yaml.safe_load(SFT_CONFIG.read_text(encoding="utf-8"))
    raw["gate"]["minimum_parse_rate"] = invalid_rate
    invalid = tmp_path / "invalid-parse-rate.yaml"
    invalid.write_text(yaml.safe_dump(raw), encoding="utf-8")

    completed = _run(
        "--sft-config",
        str(invalid),
        "--rl-config",
        str(RL_CONFIG),
        "--output",
        str(tmp_path / "unused.json"),
    )

    assert completed.returncode != 0
    assert "minimum parse rate" in completed.stderr.lower()
    assert "traceback" not in completed.stderr.lower()


def test_metadata_preflight_rejects_boolean_numeric_type_confusion(tmp_path: Path) -> None:
    raw = yaml.safe_load(RL_CONFIG.read_text(encoding="utf-8"))
    raw["reward"]["perception_reward_weight"] = False
    invalid = tmp_path / "boolean-weight.yaml"
    invalid.write_text(yaml.safe_dump(raw), encoding="utf-8")

    completed = _run(
        "--sft-config",
        str(SFT_CONFIG),
        "--rl-config",
        str(invalid),
        "--output",
        str(tmp_path / "unused.json"),
    )

    assert completed.returncode != 0
    assert "perception_reward_weight" in completed.stderr
    assert "type" in completed.stderr.lower()


def test_metadata_preflight_atomic_json_rejects_non_finite_numbers(tmp_path: Path) -> None:
    module = _load_module()
    target = tmp_path / "preflight.json"

    with pytest.raises(ValueError, match=r"JSON.*finite"):
        module._atomic_json_write(target, {"unsafe": float("nan")})

    assert not target.exists()


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("training", "algorithm", "ppo"),
        ("gate", "require_image_hidden_intervention", False),
    ],
)
def test_metadata_preflight_rejects_unpinned_rl_nested_values(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
) -> None:
    raw = yaml.safe_load(RL_CONFIG.read_text(encoding="utf-8"))
    raw[section][field] = value
    invalid = tmp_path / f"invalid-{field}.yaml"
    invalid.write_text(yaml.safe_dump(raw), encoding="utf-8")

    completed = _run(
        "--sft-config",
        str(SFT_CONFIG),
        "--rl-config",
        str(invalid),
        "--output",
        str(tmp_path / "unused.json"),
    )

    assert completed.returncode != 0
    assert field.replace("_", " ") in completed.stderr.lower()
    assert "pinned" in completed.stderr.lower()


def test_metadata_preflight_rejects_unpinned_reported_package_value(tmp_path: Path) -> None:
    raw = yaml.safe_load(SFT_CONFIG.read_text(encoding="utf-8"))
    raw["compatibility_audit"]["dockerfile_packages"]["torch"] = "untrusted"
    invalid = tmp_path / "invalid-docker-package.yaml"
    invalid.write_text(yaml.safe_dump(raw), encoding="utf-8")

    completed = _run(
        "--sft-config",
        str(invalid),
        "--rl-config",
        str(RL_CONFIG),
        "--output",
        str(tmp_path / "unused.json"),
    )

    assert completed.returncode != 0
    assert "dockerfile packages" in completed.stderr.lower()
    assert "pinned" in completed.stderr.lower()


def test_metadata_preflight_rejects_source_or_existing_output(tmp_path: Path) -> None:
    source_target = REPOSITORY_ROOT / "src/compbias/forbidden-preflight.json"
    existing = tmp_path / "existing.json"
    existing.write_text("preserve\n", encoding="utf-8")

    source_result = _run(
        "--sft-config",
        str(SFT_CONFIG),
        "--rl-config",
        str(RL_CONFIG),
        "--output",
        str(source_target),
    )
    existing_result = _run(
        "--sft-config",
        str(SFT_CONFIG),
        "--rl-config",
        str(RL_CONFIG),
        "--output",
        str(existing),
    )

    assert source_result.returncode != 0
    assert "artifacts" in source_result.stderr.lower()
    assert not source_target.exists()
    assert existing_result.returncode != 0
    assert "already exists" in existing_result.stderr.lower()
    assert existing.read_text(encoding="utf-8") == "preserve\n"


def test_metadata_preflight_output_is_limited_to_private_evidence_or_system_tmp() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match=r"private_vlm_evidence"):
        module._validated_preflight_output(REPOSITORY_ROOT / "artifacts/reports/not-private.json")


def test_metadata_preflight_atomic_write_never_clobbers_a_racing_target(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    target = tmp_path / "preflight.json"
    real_link = os.link

    def create_racing_target(source, destination, *, follow_symlinks=False):
        Path(destination).write_text("racing-owner\n", encoding="utf-8")
        return real_link(source, destination, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(module.os, "link", create_racing_target)

    with pytest.raises(FileExistsError):
        module._atomic_json_write(target, {"safe": True})

    assert target.read_text(encoding="utf-8") == "racing-owner\n"
