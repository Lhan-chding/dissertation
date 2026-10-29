"""Synthetic fail-closed admission controls; no GPU or production gate execution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sr_f6 import historical_closure, phases, pilot_contract
from sr_f6 import pilot_v3_admission as admission
from sr_f6.common import file_hash


@pytest.fixture
def root(tmp_path, monkeypatch):
    root = tmp_path.resolve() / phases.ROOT.name
    (root / "PHASES/PILOT_V3").mkdir(parents=True)
    monkeypatch.setattr(phases, "PROJECT", tmp_path.resolve())
    return root


def test_missing_real_upstream_cannot_authorize_or_seal(root):
    result = admission.seal_pilot_v3_admission(
        root, phase_identity_sha256="0" * 64, pilot_spec_sha256="1" * 64
    )
    assert result["status"] in ("FAIL", "UNKNOWN")
    assert result["training_authorized"] is result["generation_authorized"] is False
    assert not (root / admission.PATH).exists()
    assert (root / result["assessment_ref"]["path"]).is_file()


@pytest.mark.parametrize(
    "path",
    ["PHASES/PILOT_V1/ADMISSION.json", "ADMISSION/RECEIPT.json", "PHASES/FORMAL_V1/ADMISSION.json"],
)
def test_old_and_followup_licenses_cannot_be_used_for_v3(root, path):
    with pytest.raises(PermissionError, match="fixed PILOT_V3"):
        admission.require_pilot_v3_admission(
            root,
            phase_identity_sha256="0" * 64,
            pilot_spec_sha256="1" * 64,
            admission_ref=dict(path=path, sha256="2" * 64),
        )


def test_handwritten_v1_schema_at_v3_path_rejected(root):
    target = root / admission.PATH
    target.write_text(json.dumps(dict(schema="srf6-pilot-admission-v1", status="PASS")))
    with pytest.raises(PermissionError, match="phase/spec/scope"):
        admission.require_pilot_v3_admission(
            root,
            phase_identity_sha256="0" * 64,
            pilot_spec_sha256="1" * 64,
            admission_ref=dict(path=admission.PATH, sha256=file_hash(target)),
        )


def test_synthetic_registration_cannot_be_promoted(root, monkeypatch):
    context = dict(identity_sha256="0" * 64, phase="PILOT_V3")
    monkeypatch.setattr(admission, "_phase", lambda *args: context)
    ref = dict(path="overlay.json", sha256="1" * 64)
    confirmation = dict(path=historical_closure.BINDING_PATH, sha256="2" * 64)
    spec = dict(
        phase="PILOT_V3",
        input_overlay=ref,
        confirmation_binding_ref=confirmation,
        budgets=dict(
            primary_paths=9,
            updates_per_path=12,
            primary_logical_updates=108,
            responses_per_update=128,
            primary_responses=13824,
            recovery_new_responses=1024,
            dev_eval_responses=2304,
        ),
    )
    registered = dict(
        status="VERIFIED_REGISTRATION_NOT_AUTHORIZED",
        training_authorized=False,
        generation_authorized=False,
        spec=spec,
        overlay=dict(manifest_ref=ref),
        confirmation_binding=dict(binding_ref=confirmation, binding=dict(scope="SYNTHETIC_TEST")),
    )
    monkeypatch.setattr(pilot_contract, "verify_registration", lambda *args, **kwargs: registered)
    result = admission.assess_pilot_v3_admission(
        root, phase_identity_sha256="0" * 64, pilot_spec_sha256="1" * 64
    )
    assert result["status"] == "FAIL"
    assert result["training_authorized"] is False
    assert "Synthetic" in result["error"]
    assert result["kernel_correspondence"] is None


def test_scope_has_no_formal_or_test_authority(root):
    assessment = dict(
        status="PASS",
        request=dict(pilot_spec_sha256="1" * 64),
        phase_context=dict(identity_sha256="0" * 64),
        overlay=dict(manifest_ref=dict(path="overlay", sha256="2" * 64)),
        pilot_spec=dict(confirmation_binding_ref=dict(path="confirmation", sha256="3" * 64)),
        kernel_correspondence=dict(status="SYNTHETIC_TEST"),
    )
    with pytest.raises(PermissionError, match="complete actual"):
        admission._receipt(root, assessment)
    assert admission.SCOPE == "PILOT_V3_REGISTERED_ONLY"
    assert admission.PATH != "PHASES/PILOT_V1/ADMISSION.json"


def test_bound_alias_target_state_changes_invalidate_warm_proof(root):
    target = root.parent / "synthetic-bound-model"
    target.write_bytes(b"no model")
    alias = root / "alias"
    alias.symlink_to(target)
    before = historical_closure._state(alias)
    other = root.parent / "synthetic-other"
    other.write_bytes(b"no model")
    alias.unlink()
    alias.symlink_to(other)
    assert historical_closure._state(alias) != before
    outside = Path("/etc/hosts")
    alias.unlink()
    alias.symlink_to(outside)
    with pytest.raises(PermissionError):
        historical_closure._state(alias)
