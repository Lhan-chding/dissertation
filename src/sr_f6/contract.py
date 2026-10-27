"""Frozen science and explicit, prospective implementation clarifications."""

from __future__ import annotations

from pathlib import Path

from .common import digest, file_hash, read

PACKAGE_CONFIG_SHA256 = "59052e828fecec1a5cf15255cb0ef11f269406a087ee373688bd977012e319d9"
ROOT = Path("/projects/_ssd/varunssd/louis-ssvc/sr_f6_semantic_factorial_20261027")
PROJECT = ROOT.parent
DIAGNOSTIC_STEPS = (1, 4, 8, 12, 36, 72)
AMENDMENTS = {
    "version": "SRF6-PREMODEL-IMPLEMENTATION-20261027-v1",
    "made_before_any_srf6_model_output": True,
    "unchanged_science": (
        "six arms, three parents, three repeats, seeds, 72 updates, lambda .5, S:B 1:1"
    ),
    "lossless_projection": (
        "delete only raw bindings/aggregate members; never stringify Decimal literals"
    ),
    "composition": (
        "ordinary P0 proposal and actual four-view acceptance; unused P1 cannot select test roots"
    ),
    "field_observability": (
        "extract readable fields independently; V and finished still gate all auxiliary credit"
    ),
    "gradient_diagnostic_steps": list(DIAGNOSTIC_STEPS),
    "training_order_seed": (
        "include parent; within each parent/repeat all six arms share root order and draw streams"
    ),
    "formal_arm_order": (
        "seeded random order within each parent/repeat block, committed before outputs"
    ),
    "retention": {
        "confirmatory_candidate": "U11",
        "comparator": "corresponding frozen parent, shared panel and draw seeds",
        "panels": {"LEGACY": 0.01, "CHARTQA": 0.02},
        "method": (
            "one-sided paired fixed-parent t tests at margin, Holm across two "
            "panels; simultaneous 97.5% one-sided lower bounds"
        ),
        "other_arms": (
            "report all six arms versus parents descriptively; no unadjusted joint retention claim"
        ),
        "zero_variance": "NA, no affirmative noninferiority claim",
    },
    "job_prefix": "LH-SRF6-",
    "resource_layout": (
        "prefer two independent single-GPU workers per job; count all shared QoS reservations"
    ),
    "original_package": "preserved verbatim in docs/sr_f6/package",
}


def validate_config(config):
    """Reject recipe drift, including lower doses and smaller matrices."""
    original_path = (
        Path(__file__).resolve().parents[2] / "docs/sr_f6/package/configs/experiment.json"
    )
    if file_hash(original_path) != PACKAGE_CONFIG_SHA256:
        raise PermissionError(
            "Preserved original configuration no longer matches the source archive"
        )
    original = read(original_path)
    if config != original:
        raise PermissionError(
            "Configuration differs from the registered package; explicit revision required"
        )
    if Path(config["deployment"]["root"]) != ROOT:
        raise PermissionError("Unexpected run root")
    return {"config_digest": digest(config), "implementation_digest": digest(AMENDMENTS)}
