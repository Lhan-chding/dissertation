"""A second reviewed continuation retains the original sampling and source chain."""

# ruff: noqa: F811 -- pytest resolves the imported fixture by parameter name.

import copy
from pathlib import Path

import pytest
from test_r4_continuation import _decision, _rehash, stopped_parent  # noqa: F401

from src.core import canonical_hash, file_hash, write_json


def _diagnostic(mean=0.101, sequence=2.4):
    return {
        "mean_token_kl": mean,
        "sequence_log_ratio_p99_abs": sequence,
        "should_stop": mean > 0.1 or sequence > 2.0,
        "status": "STOP_DIAGNOSE",
        "alarms": {
            "mean_token_kl": mean > 0.1,
            "sequence_log_ratio_p99_abs": sequence > 2.0,
        },
        "thresholds": {
            "mean_token_kl": 0.1,
            "sequence_log_ratio_p99_abs": 2.0,
            "comparison": "strict_greater_than",
        },
    }


@pytest.mark.parametrize("mean,sequence", [(0.101, 1.0), (0.101, 2.4), (0.1, 2.4)])
def test_cumulative_warning_is_source_policy_specific(mean, sequence):
    from src.r4_continuation import (
        REVIEWED_CUMULATIVE_POLICY,
        REVIEWED_POLICY,
        allows_reviewed_warning,
        policy_for_name,
    )

    diagnostic = _diagnostic(mean, sequence)
    before = copy.deepcopy(diagnostic)
    assert allows_reviewed_warning(diagnostic, REVIEWED_CUMULATIVE_POLICY)
    assert allows_reviewed_warning(diagnostic, REVIEWED_POLICY) is (mean <= 0.1)
    policy = policy_for_name(REVIEWED_CUMULATIVE_POLICY["reviewed_warning_policy"])
    policy["steps"] = 1
    assert REVIEWED_CUMULATIVE_POLICY["steps"] == 64
    assert diagnostic == before


@pytest.mark.parametrize(
    "fault",
    [
        "nan",
        "inf",
        "large_integer",
        "negative",
        "bool",
        "flag",
        "numeric_flag",
        "threshold",
        "status",
        "stop",
        "no_alarm",
        "policy",
    ],
)
def test_cumulative_warning_rejects_unreviewed_or_inconsistent_diagnostics(fault):
    from src.r4_continuation import REVIEWED_CUMULATIVE_POLICY, allows_reviewed_warning

    diagnostic, policy = _diagnostic(), copy.deepcopy(REVIEWED_CUMULATIVE_POLICY)
    if fault in ("nan", "inf"):
        diagnostic["mean_token_kl"] = float(fault)
    elif fault == "large_integer":
        diagnostic["mean_token_kl"] = 10**500
    elif fault == "negative":
        diagnostic["sequence_log_ratio_p99_abs"] = -1.0
    elif fault == "bool":
        diagnostic["mean_token_kl"] = True
    elif fault == "flag":
        diagnostic["alarms"]["mean_token_kl"] = False
    elif fault == "numeric_flag":
        diagnostic["alarms"]["mean_token_kl"] = 1
    elif fault == "threshold":
        diagnostic["thresholds"]["mean_token_kl"] = 0.2
    elif fault == "status":
        diagnostic["status"] = "WITHIN_ENGINEERING_LIMITS"
    elif fault == "stop":
        diagnostic["should_stop"] = False
    elif fault == "no_alarm":
        diagnostic = _diagnostic(0.1, 2.0)
    else:
        policy["hard_stops"] = []
    assert not allows_reviewed_warning(diagnostic, policy)


@pytest.mark.parametrize("value", [None, "unknown", {}, True, []])
def test_policy_lookup_rejects_unknown_or_non_string_names(value):
    from src.r4_continuation import allows_reviewed_warning, policy_for_name

    with pytest.raises(ValueError, match="Unsupported"):
        policy_for_name(value)
    assert not allows_reviewed_warning(_diagnostic(), value)


def _v2_decision(binding, source):
    from src.r4_continuation import REVIEWED_CUMULATIVE_POLICY

    return {
        **_decision(binding, source),
        "schema_version": 2,
        "policy": copy.deepcopy(REVIEWED_CUMULATIVE_POLICY),
    }


def test_only_v2_can_authorize_a_finite_mean_stop(stopped_parent):
    from src.r4_continuation import validate_continuation_decision

    _, binding, source = stopped_parent
    binding["stop"]["diagnostic"] = _diagnostic()
    _rehash(binding)
    decision = _v2_decision(binding, source)
    assert validate_continuation_decision(decision, binding, source) == decision
    with pytest.raises(ValueError):
        validate_continuation_decision(_decision(binding, source), binding, source)
    decision["schema_version"] = 1
    with pytest.raises(ValueError):
        validate_continuation_decision(decision, binding, source)


def _second_parent(stopped_parent, tmp_path):
    from src.r4_continuation import prepare_continuation

    original, old_binding, source = stopped_parent
    parent = tmp_path / "first_continuation"
    context = prepare_continuation(
        original, parent, _decision(old_binding, source), old_binding, source, allow_training=True
    )
    identity = {
        **old_binding["identity"],
        "source_hash": canonical_hash(source),
        "continuation_hash": context["continuation_hash"],
    }
    runtime = {"identity": identity, "source": source, "continuation": context["contract"]}
    write_json(parent / "identity.json", identity)
    write_json(parent / "runtime_lock.json", runtime)
    write_json(parent / "alarm_stop.json", {"control_diagnostic": _diagnostic()})
    write_json(parent / "checkpoint_manifest.json", {"checkpoints": []})
    write_json(parent / "manifest.json", {"files": []})
    (parent / "checkpoint.pt").write_bytes(b"step51 full Adam and RNG")
    binding = {
        **old_binding,
        "root": str(parent),
        "identity": identity,
        "logical_sampling_identity": old_binding["identity"],
        "source": source,
        "continuation": context["contract"],
        "files": {
            str(p.relative_to(parent)): file_hash(p) for p in parent.rglob("*") if p.is_file()
        },
        "manifest_sha256": file_hash(parent / "manifest.json"),
        "runtime_lock_sha256": file_hash(parent / "runtime_lock.json"),
        "alarm_stop_sha256": file_hash(parent / "alarm_stop.json"),
        "stop": {
            "arm": "X_BASE",
            "step": 51,
            "checkpoint": {
                "checkpoint_path": str(parent / "checkpoint.pt"),
                "checkpoint_sha256": file_hash(parent / "checkpoint.pt"),
            },
            "diagnostic": _diagnostic(),
        },
    }
    _rehash(binding)
    next_source = {"source_commit": "c" * 40, "source_files": {"src/new.py": "c" * 64}}
    return parent, binding, next_source, old_binding["identity"]


def test_second_continuation_keeps_original_logical_identity_and_all_parent_bytes(
    stopped_parent, tmp_path
):
    from src.r4_continuation import prepare_continuation, verify_continuation

    parent, binding, source, logical = _second_parent(stopped_parent, tmp_path)
    before = {str(p.relative_to(parent)): p.read_bytes() for p in parent.rglob("*") if p.is_file()}
    out = tmp_path / "second_continuation"
    result = prepare_continuation(
        parent, out, _v2_decision(binding, source), binding, source, allow_training=True
    )
    assert result["sampling_identity"] == logical
    assert result["runtime_binding"]["logical_sampling_identity"] == logical
    assert result["runtime_binding"]["schema_version"] == 2
    assert result["sampling_identity"] != binding["identity"]
    assert verify_continuation(out, binding, source) == result
    assert {
        str(p.relative_to(parent)): p.read_bytes() for p in parent.rglob("*") if p.is_file()
    } == before


@pytest.mark.parametrize("fault", ["missing", "changed", "contract", "experiment", "file"])
def test_second_continuation_rejects_unbound_parent_logical_identity(
    stopped_parent, tmp_path, fault
):
    from src.r4_continuation import prepare_continuation

    parent, binding, source, logical = _second_parent(stopped_parent, tmp_path)
    if fault == "missing":
        del binding["logical_sampling_identity"]
    elif fault == "changed":
        binding["logical_sampling_identity"] = binding["identity"]
    elif fault == "contract":
        binding["continuation"]["logical_sampling_identity_sha256"] = "0" * 64
    elif fault == "experiment":
        binding["logical_sampling_identity"] = {**logical, "data_hash": "changed"}
    else:
        write_json(parent / "logical_sampling_identity.json", binding["identity"])
        binding["files"]["logical_sampling_identity.json"] = file_hash(
            parent / "logical_sampling_identity.json"
        )
    _rehash(binding)
    with pytest.raises(ValueError):
        prepare_continuation(
            parent,
            tmp_path / "invalid",
            _v2_decision(binding, source),
            binding,
            source,
            allow_training=True,
        )
    assert not (tmp_path / "invalid").exists()


def _chain(root, depth=2):
    current = root
    for index in range(depth + 1):
        runtime = {"identity": {"source_hash": str(index)}}
        if index < depth:
            runtime["continuation"] = {"parent_dir": "inherited_parent"}
        write_json(current / "runtime_lock.json", runtime)
        if index == depth:
            write_json(current / "evaluation/INITIAL/step_00/N/identity.json", {"owner": index})
        current /= "inherited_parent"


def test_resolver_locates_original_ledger_and_explicit_inherited_artifact(tmp_path):
    from src.r4_continuation import resolve_evidence

    root = tmp_path / "new"
    _chain(root)
    relative = "evaluation/INITIAL/step_00/N"
    result = resolve_evidence(root, relative)
    owner = root / "inherited_parent/inherited_parent"
    assert result == {
        "root": owner,
        "runtime": {"identity": {"source_hash": "2"}},
        "path": owner / relative,
        "relative": Path(relative),
    }
    explicit = resolve_evidence(root, "inherited_parent/inherited_parent/" + relative)
    assert explicit == result


@pytest.mark.parametrize(
    "fault",
    [
        "undeclared",
        "symlink",
        "runtime_symlink",
        "parent_symlink",
        "wrong_parent",
        "escape",
        "absolute",
        "deep",
        "missing",
    ],
)
def test_resolver_refuses_untrusted_or_unbounded_inheritance(tmp_path, fault):
    from src.r4_continuation import resolve_evidence

    root = tmp_path / "new"
    _chain(root, 33 if fault == "deep" else 2)
    relative = "evaluation/INITIAL/step_00/N"
    if fault == "undeclared":
        write_json(root / "runtime_lock.json", {"identity": {"source_hash": "0"}})
    elif fault == "symlink":
        (root / "evaluation").symlink_to(
            root / "inherited_parent/inherited_parent/evaluation", target_is_directory=True
        )
    elif fault == "runtime_symlink":
        (root / "runtime_lock.json").unlink()
        (root / "runtime_lock.json").symlink_to(root / "inherited_parent/runtime_lock.json")
    elif fault == "parent_symlink":
        (root / "inherited_parent").rename(tmp_path / "saved-parent")
        (root / "inherited_parent").symlink_to(root, target_is_directory=True)
    elif fault == "wrong_parent":
        write_json(
            root / "runtime_lock.json", {"identity": {}, "continuation": {"parent_dir": "../other"}}
        )
    elif fault == "escape":
        relative = "../new/runtime_lock.json"
    elif fault == "absolute":
        relative = str(root / "runtime_lock.json")
    elif fault == "missing":
        relative = "not-present.json"
    with pytest.raises((ValueError, FileNotFoundError)):
        resolve_evidence(root, relative)


def test_original_contract_hash_is_unchanged(stopped_parent, tmp_path):
    from src.r4_continuation import _inventory, prepare_continuation

    parent, binding, source = stopped_parent
    decision = _decision(binding, source)
    stable = {
        "schema_version": 1,
        "parent_audit_hash": binding["audit_hash"],
        "decision_sha256": canonical_hash(decision),
        "parent_snapshot_sha256": canonical_hash(_inventory(parent)),
        "logical_sampling_identity_sha256": canonical_hash(binding["identity"]),
        "reviewed_warning_policy": "sequence_p99_only_two_arms_to_step64",
    }
    result = prepare_continuation(
        parent, tmp_path / "child", decision, binding, source, allow_training=True
    )
    assert result["runtime_binding"] == {
        **stable,
        "continuation_hash": canonical_hash(stable),
        "parent_dir": "inherited_parent",
        "parent_binding": binding,
    }
