from __future__ import annotations

import csv
import json
from pathlib import Path

from compbias.audit.representation_invariance import audit_representation_invariance
from compbias.identification.claim_guardrails import assess_claim
from scripts.audit_frozen_regime import main as frozen_main
from scripts.audit_v1_claims import main as claims_main
from scripts.build_partial_id_certificate import main as partial_main


def test_claim_guardrails_reject_anatomical_claims_beyond_evidence() -> None:
    frozen = assess_claim(
        "Outcome RL improves visual acquisition.",
        acquisition_frozen=True,
        black_box=False,
    )
    black_box = assess_claim(
        "The true internal perception module compensates its error.",
        acquisition_frozen=False,
        black_box=True,
    )
    allowed = assess_claim(
        "We report an operational compensation certificate under interface family M.",
        acquisition_frozen=False,
        black_box=True,
    )

    assert not frozen.allowed
    assert "acquisition" in frozen.reason
    assert not black_box.allowed
    assert "operational" in black_box.replacement
    assert allowed.allowed


def test_representation_invariance_requires_hash_and_fixed_image_identity() -> None:
    clean = audit_representation_invariance(
        before_weight_sha256="a" * 64,
        after_weight_sha256="a" * 64,
        before_hidden={"sample-a": [1.0, 2.0], "sample-b": [3.0, 4.0]},
        after_hidden={"sample-a": [1.0, 2.0], "sample-b": [3.0, 4.0]},
        probe_before=0.81,
        probe_after=0.81,
        hidden_tolerance=1e-10,
        probe_tolerance=1e-10,
    )
    drift = audit_representation_invariance(
        before_weight_sha256="a" * 64,
        after_weight_sha256="a" * 64,
        before_hidden={"sample-a": [1.0, 2.0]},
        after_hidden={"sample-a": [1.0, 2.01]},
        probe_before=0.81,
        probe_after=0.82,
        hidden_tolerance=1e-4,
        probe_tolerance=1e-4,
    )

    assert clean.passed
    assert clean.maximum_hidden_drift == 0.0
    assert not drift.passed
    assert set(drift.failed_gates) == {"hidden_representation_drift", "probe_drift"}


def test_v1_claim_audit_writes_keep_demote_retract_and_rerun(tmp_path: Path) -> None:
    old = tmp_path / "old.md"
    old.write_text(
        "\n".join(
            (
                "7000 numerical property checks verify the equation.",
                "The selection law must use do-compensability.",
                "A single 16x16 PIL CNN proves the VLM mechanism.",
                "We still need a real VLM rerun.",
            )
        ),
        encoding="utf-8",
    )
    output = tmp_path / "audit.csv"

    assert claims_main(["--old-plan", str(old), "--out", str(output)]) == 0
    rows = list(csv.DictReader(output.read_text(encoding="utf-8").splitlines()))

    assert {row["decision"] for row in rows} == {"keep", "demote", "retract", "rerun"}
    selection = next(row for row in rows if row["claim_id"] == "V1-SELECTION-DO")
    assert selection["decision"] == "retract"
    assert selection["replacement_claim"].startswith("Natural conditional success")


def _validity(*, parser_reliability: float = 0.99) -> dict[str, object]:
    return {
        "oracle_loss": 0.0,
        "replay_js": 0.01,
        "replay_accuracy_gap": 0.01,
        "image_exclusion_gap": 0.0,
        "parser_reliability": parser_reliability,
        "natural_state_count_by_error": {"offset": 220},
        "natural_input_count_by_error": {"offset": 55},
    }


def test_partial_id_cli_excludes_invalid_favorable_interface(tmp_path: Path) -> None:
    source = tmp_path / "interfaces.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "bootstrap_draws": 2000,
                "confidence": 0.95,
                "seed": 17,
                "interfaces": [
                    {
                        "interface_id": "evidence_prefix",
                        "cluster_gammas": [-0.25, -0.22, -0.21, -0.20],
                        "validity": _validity(),
                    },
                    {
                        "interface_id": "post_projector",
                        "cluster_gammas": [-0.99, -0.98, -0.97, -0.96],
                        "validity": _validity(parser_reliability=0.20),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "certificate.json"

    assert partial_main(["--input", str(source), "--output", str(output)]) == 0
    result = json.loads(output.read_text(encoding="utf-8"))

    assert result["valid_interfaces"] == ["evidence_prefix"]
    assert result["invalid_interfaces"] == ["post_projector"]
    assert result["conclusion"] == "robust_compensation"
    assert result["claim_scope"] == "operational interfaces only; no unique anatomical boundary"


def test_frozen_regime_cli_reports_acquisition_invariance(tmp_path: Path) -> None:
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text(
        json.dumps(
            {
                "vision_weight_sha256": "a" * 64,
                "fixed_image_hidden": {"x": [0.1, 0.2]},
                "frozen_representation_probe": 0.75,
            }
        ),
        encoding="utf-8",
    )
    after.write_text(
        json.dumps(
            {
                "vision_weight_sha256": "a" * 64,
                "fixed_image_hidden": {"x": [0.1, 0.2]},
                "frozen_representation_probe": 0.75,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "audit.json"

    assert (
        frozen_main(
            [
                "--before",
                str(before),
                "--after",
                str(after),
                "--regime",
                "lm_only",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["acquisition_frozen"] is True
    assert "acquisition_improvement" in report["forbidden_claims"]
