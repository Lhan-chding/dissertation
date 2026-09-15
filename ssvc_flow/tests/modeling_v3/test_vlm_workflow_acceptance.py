"""Reference uncertainty cannot retrospectively change a frozen acceptance mask."""

import json

import numpy as np
import pytest

from src.modeling_v3 import workflow
from src.modeling_v3.io import sha256_file


def test_frozen_mask_keeps_unresolved_reference_cases_in_denominator(tmp_path, monkeypatch):
    arrays = tmp_path / "arrays.npz"
    np.savez(
        arrays,
        predictions=np.zeros((3, 4)),
        reference=np.array([[0.1] * 4, [np.nan] * 4, [0.3] * 4]),
        accepted=np.array([True, True, False]),
    )
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"arrays": {"path": str(arrays), "sha256": sha256_file(arrays)}}))
    seen = []

    def verified(config, spec, prediction, accepted, path):
        seen.append(accepted.copy())
        return {"scope": "VERIFIER_DISPATCH_FIXTURE"}

    monkeypatch.setattr(workflow, "_verify_acceptance_receipt", verified)
    workflow.run_artifact_command("evaluate", {}, [str(spec)], tmp_path / "out")
    result = json.loads((tmp_path / "out/EVALUATION.json").read_text())
    risk = result["risk_coverage"]
    assert seen[0].tolist() == [True, True, False]
    assert risk["accepted_cases"] == 2
    assert risk["accepted_unresolved_cases"] == 1
    assert risk["coverage"] == pytest.approx(2 / 3)
    assert risk["accepted_risk"] is None
    assert risk["accepted_resolved_risk"] == pytest.approx(0.01)
    assert risk["all_case_model_loss"] is None


def test_unresolved_mask_still_requires_real_acceptance_provenance(tmp_path):
    arrays = tmp_path / "arrays.npz"
    np.savez(
        arrays,
        predictions=np.zeros((1, 4)),
        reference=np.full((1, 4), np.nan),
        accepted=np.ones(1, bool),
    )
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"arrays": {"path": str(arrays), "sha256": sha256_file(arrays)}}))
    with pytest.raises(ValueError, match="acceptance receipt"):
        workflow.run_artifact_command("evaluate", {}, [str(spec)], tmp_path / "out")
