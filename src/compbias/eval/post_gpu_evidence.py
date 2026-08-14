"""Validate draft post-GPU bindings without authenticating completed jobs.

The public validator intentionally fails closed after structural validation until
an authenticated gate extension and trust root are implemented.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_GPU_UUID = re.compile(r"GPU-[A-Za-z0-9][A-Za-z0-9_.:-]{5,127}\Z")
_STAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\Z")
_PUBLIC_REVIEWER = re.compile(r"reviewer-[a-z0-9][a-z0-9-]{0,30}\Z")
_VISUAL_STYLES = (
    "baseline",
    "font_weight_bold",
    "size_compact",
    "rotation_tilted",
    "contrast_low",
    "background_grid",
    "occlusion_local",
    "blur_mild",
    "distractor_marks",
    "layout_shifted",
)
_TASK_FAMILIES = (
    "digit_offset",
    "count_transform",
    "gauge_calibration",
    "bar_chart_aggregate",
    "relation_rule",
)
_SPLITS = ("train", "calibration", "val", "iid_test", "ood_test")
_VISUAL_APPLICABILITY = {
    style: (
        ("digit_offset", "gauge_calibration", "bar_chart_aggregate")
        if style == "font_weight_bold"
        else _TASK_FAMILIES
    )
    for style in _VISUAL_STYLES
}
_STYLE_COUNTS = {
    "baseline": 200,
    "font_weight_bold": 120,
    "size_compact": 200,
    "rotation_tilted": 200,
    "contrast_low": 200,
    "background_grid": 200,
    "occlusion_local": 200,
    "blur_mild": 200,
    "distractor_marks": 200,
    "layout_shifted": 100,
}
_APPLICABLE_SAMPLE_COUNTS = {
    "baseline": dict.fromkeys(_TASK_FAMILIES, 40),
    "font_weight_bold": {
        "digit_offset": 40,
        "gauge_calibration": 40,
        "bar_chart_aggregate": 40,
    },
    "size_compact": dict.fromkeys(_TASK_FAMILIES, 40),
    "rotation_tilted": dict.fromkeys(_TASK_FAMILIES, 40),
    "contrast_low": dict.fromkeys(_TASK_FAMILIES, 40),
    "background_grid": dict.fromkeys(_TASK_FAMILIES, 40),
    "occlusion_local": dict.fromkeys(_TASK_FAMILIES, 40),
    "blur_mild": dict.fromkeys(_TASK_FAMILIES, 40),
    "distractor_marks": dict.fromkeys(_TASK_FAMILIES, 40),
    "layout_shifted": dict.fromkeys(_TASK_FAMILIES, 20),
}
_PHASE_D_FIELDS = frozenset(
    {
        "audit_report_schema_version",
        "sample_count",
        "split_audit",
        "split_audit_error",
        "split_clean",
        "solver_passes",
        "solver_pass_rate",
        "roundtrip_passes",
        "roundtrip_total",
        "roundtrip_pass_rate",
        "error_solver_passes",
        "error_solver_pass_rate",
        "rendered_image_count",
        "missing_images",
        "extra_images",
        "image_set_matches",
        "rendered_image_count_matches",
        "contact_sheet_sha256_matches",
        "contact_sheet_hash_mismatches",
        "manifest_sample_count_matches",
        "manifest_sample_ids_match",
        "manifest_content_sha256_matches",
        "manifest_config_sha256_matches",
        "manifest_dataset_file_sha256_matches",
        "manifest_image_sha256_matches",
        "manifest_self_sha256_matches",
        "preregistered_ood_factors_match_config",
        "noncanonical_rows",
        "image_path_mismatches",
        "privacy_issues",
        "image_question_answer_collisions",
        "style_counterbalance_violations",
        "evidence_manifest_sha256",
        "evidence_image_set_sha256",
        "visual_review_present",
        "human_reviewer_signoff",
        "human_review_binding_matches",
        "human_review",
        "visual_factor_realization_audit",
        "ood_image_shift",
        "style_semantic_joint_independence",
        "deterministic_replay",
        "answer_balance",
        "dataset",
        "automatic_audit_clean",
        "phase_d_ready",
    }
)


class PostGPUAuthenticationPending(RuntimeError):
    """The local draft is well formed but lacks an authenticated trust root."""


def _closed(value: object, *, name: str, fields: Sequence[str]) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping")
    result = dict(value)
    expected = set(fields)
    if set(result) != expected:
        raise ValueError(
            f"{name} must match the closed schema; "
            f"missing={sorted(expected - set(result))}, unknown={sorted(set(result) - expected)}"
        )
    return result


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _safe_command(value: object) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item or len(item) > 4096 for item in value)
    ):
        raise ValueError("execution audit command must be a non-empty bounded argument list")
    return value


def _existing_hash_bound_file(
    value: object,
    digest: object,
    *,
    name: str,
    sha256_file,
) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{name} path must be a non-empty string")
    path = Path(value).expanduser()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be an existing non-symlink file")
    expected = _digest(digest, name=f"{name} SHA-256")
    if sha256_file(path) != expected:
        raise ValueError(f"{name} SHA-256 mismatch")
    return path


def _validate_visual_factor_audit(value: object) -> None:
    visual = _closed(
        value,
        name="Phase-D visual-factor audit",
        fields=(
            "complete",
            "catalog",
            "observed_styles",
            "sample_counts",
            "applicability",
            "applicable_sample_counts",
            "applicability_violations",
            "applicable_coverage",
            "nonapplicable_baseline_contract",
        ),
    )
    expected_applicability = {
        style: list(families) for style, families in _VISUAL_APPLICABILITY.items()
    }
    if visual != {
        "complete": True,
        "catalog": list(_VISUAL_STYLES),
        "observed_styles": list(_VISUAL_STYLES),
        "sample_counts": _STYLE_COUNTS,
        "applicability": expected_applicability,
        "applicable_sample_counts": _APPLICABLE_SAMPLE_COUNTS,
        "applicability_violations": [],
        "applicable_coverage": True,
        "nonapplicable_baseline_contract": True,
    }:
        raise ValueError("Phase-D visual-factor audit is incomplete or drifted")


def _validate_joint_independence(value: object) -> None:
    joint = _closed(
        value,
        name="Phase-D style/semantic audit",
        fields=("complete", "criterion", "groups", "violations"),
    )
    groups = joint["groups"]
    expected_group_names = {
        f"{family}/{split}" for family in _TASK_FAMILIES for split in _SPLITS if split != "ood_test"
    }
    if (
        joint["complete"] is not True
        or joint["criterion"] != "fully_crossed_style_by_semantic_state"
        or joint["violations"] != []
        or not isinstance(groups, Mapping)
        or set(groups) != expected_group_names
    ):
        raise ValueError("Phase-D style/semantic audit is incomplete")
    for group_name, raw_group in groups.items():
        group = _closed(
            raw_group,
            name=f"Phase-D style/semantic group {group_name}",
            fields=(
                "semantic_state_count",
                "expected_styles",
                "fully_crossed_state_count",
                "sample_count",
                "style_counts",
            ),
        )
        family = str(group_name).split("/", maxsplit=1)[0]
        styles = [style for style in _VISUAL_STYLES[:-1] if family in _VISUAL_APPLICABILITY[style]]
        if group != {
            "semantic_state_count": 10,
            "expected_styles": styles,
            "fully_crossed_state_count": 10,
            "sample_count": 10 * len(styles),
            "style_counts": dict.fromkeys(styles, 10),
        }:
            raise ValueError(f"Phase-D style/semantic group {group_name} is not fully crossed")


def _validate_answer_balance(value: object) -> None:
    balance = _closed(
        value,
        name="Phase-D answer-balance audit",
        fields=(
            "complete",
            "groups",
            "iid_ood_exact_match",
            "numeric_exact_balance",
            "relation_multiclass_coverage",
            "violations",
        ),
    )
    groups = balance["groups"]
    from compbias.envs.cva_world.generator import GeneratorConfig, generate_dataset
    from compbias.io.manifests import canonical_json

    expected_groups: dict[str, object] = {}
    expected_samples = generate_dataset(
        GeneratorConfig(
            seed=20260814,
            samples_per_family_per_split=10,
            visual_styles=_VISUAL_STYLES,
            train_error_mechanism="offset_plus_2",
            ood_error_mechanism="offset_minus_2",
            preregistered_ood_factors=("visual_style", "error_mechanism"),
            realizations_per_semantic=2,
            fully_cross_iid_visual_styles=True,
        )
    )
    answers_by_group: dict[str, list[object]] = {}
    for sample in expected_samples:
        name = f"{sample.task_family.value}/{sample.split_keys.semantic_split.value}"
        answers_by_group.setdefault(name, []).append(sample.canonical_answer)
    for name, answers in answers_by_group.items():
        decoded: dict[str, object] = {}
        counts: dict[str, int] = {}
        for answer in answers:
            encoded = canonical_json(answer)
            decoded.setdefault(encoded, answer)
            counts[encoded] = counts.get(encoded, 0) + 1
        expected_groups[name] = {
            "sample_count": len(answers),
            "support": [decoded[key] for key in sorted(counts)],
            "frequencies": [
                {"answer": decoded[key], "count": counts[key]} for key in sorted(counts)
            ],
        }
    if (
        balance["complete"] is not True
        or balance["iid_ood_exact_match"] is not True
        or balance["numeric_exact_balance"] is not True
        or balance["relation_multiclass_coverage"] is not True
        or balance["violations"] != []
        or not isinstance(groups, Mapping)
        or canonical_json(dict(groups)) != canonical_json(expected_groups)
    ):
        raise ValueError("Phase-D answer-balance audit differs from deterministic CVA-v2")


def validate_ready_phase_d_audit(
    phase_d: Mapping[str, object],
    *,
    dataset_manifest_sha256: str,
    dataset_manifest_self_sha256: str,
    dataset_content_sha256: str,
    dataset_image_set_sha256: str,
    sample_ids: Sequence[str],
) -> None:
    """Validate the closed schema-2 automatic audit and bound human signoff."""

    root = _closed(phase_d, name="Phase-D audit", fields=tuple(_PHASE_D_FIELDS))
    exact = {
        "audit_report_schema_version": 2,
        "sample_count": 1820,
        "split_clean": True,
        "solver_passes": 1820,
        "solver_pass_rate": 1.0,
        "roundtrip_passes": 4020,
        "roundtrip_total": 4020,
        "roundtrip_pass_rate": 1.0,
        "error_solver_passes": 4020,
        "error_solver_pass_rate": 1.0,
        "rendered_image_count": 1820,
        "image_set_matches": True,
        "rendered_image_count_matches": True,
        "contact_sheet_sha256_matches": True,
        "manifest_sample_count_matches": True,
        "manifest_sample_ids_match": True,
        "manifest_content_sha256_matches": True,
        "manifest_config_sha256_matches": True,
        "manifest_dataset_file_sha256_matches": True,
        "manifest_image_sha256_matches": True,
        "manifest_self_sha256_matches": True,
        "preregistered_ood_factors_match_config": True,
        "visual_review_present": True,
        "human_reviewer_signoff": True,
        "human_review_binding_matches": True,
        "automatic_audit_clean": True,
        "phase_d_ready": True,
    }
    for field, expected in exact.items():
        if root[field] != expected:
            raise ValueError(f"Phase-D gate {field} is not accepted")
    for field in (
        "missing_images",
        "extra_images",
        "contact_sheet_hash_mismatches",
        "noncanonical_rows",
        "image_path_mismatches",
        "privacy_issues",
        "image_question_answer_collisions",
        "style_counterbalance_violations",
    ):
        if root[field] != []:
            raise ValueError(f"Phase-D gate {field} must be empty")
    dataset = _closed(
        root["dataset"],
        name="Phase-D dataset",
        fields=(
            "manifest_path",
            "manifest_file_sha256",
            "manifest_self_sha256",
            "content_sha256",
            "image_set_sha256",
        ),
    )
    if (
        dataset["manifest_file_sha256"] != dataset_manifest_sha256
        or dataset["manifest_self_sha256"] != dataset_manifest_self_sha256
        or dataset["content_sha256"] != dataset_content_sha256
        or dataset["image_set_sha256"] != dataset_image_set_sha256
        or root["evidence_manifest_sha256"] != dataset_manifest_self_sha256
        or root["evidence_image_set_sha256"] != dataset_image_set_sha256
    ):
        raise ValueError("Phase-D dataset/image evidence differs from frozen CVA-v2")
    human = _closed(
        root["human_review"],
        name="Phase-D human review",
        fields=(
            "signoff",
            "reviewer",
            "reviewer_type",
            "review_date",
            "review_result",
            "reviewed_image_count",
            "reviewed_sample_ids",
            "contact_sheets_reviewed",
            "binding_matches",
            "manifest_self_sha256",
            "integrity_scope",
        ),
    )
    reviewed = human["reviewed_sample_ids"]
    if (
        human["signoff"] is not True
        or human["reviewer_type"] != "human"
        or human["review_result"] != "pass"
        or human["binding_matches"] is not True
        or human["manifest_self_sha256"] != dataset_manifest_self_sha256
        or not isinstance(human["reviewer"], str)
        or _PUBLIC_REVIEWER.fullmatch(str(human["reviewer"])) is None
        or not isinstance(human["review_date"], str)
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}", human["review_date"]) is None
        or not isinstance(reviewed, list)
        or len(reviewed) < 200
        or len(set(reviewed)) != len(reviewed)
        or set(reviewed) - set(sample_ids)
        or human["reviewed_image_count"] != len(reviewed)
        or human["contact_sheets_reviewed"] != 73
        or human["integrity_scope"] != "self-reported review record; no external signature verified"
    ):
        raise ValueError("Phase-D requires a closed dataset-bound human review")
    replay = _closed(
        root["deterministic_replay"],
        name="Phase-D deterministic replay",
        fields=(
            "complete",
            "generator_matches",
            "renderer_matches",
            "contact_sheets_match",
            "generator_mismatches",
            "renderer_mismatches",
            "contact_sheet_mismatches",
        ),
    )
    if replay != {
        "complete": True,
        "generator_matches": True,
        "renderer_matches": True,
        "contact_sheets_match": True,
        "generator_mismatches": [],
        "renderer_mismatches": [],
        "contact_sheet_mismatches": [],
    }:
        raise ValueError("Phase-D deterministic replay is incomplete")
    split = _closed(
        root["split_audit"],
        name="Phase-D split audit",
        fields=(
            "scene_template_leaks",
            "answer_leaks",
            "visual_style_leaks",
            "error_mechanism_leaks",
            "ood_pair_mismatches",
            "ood_pair_count",
            "preregistered_ood_factors",
            "ood_changed_factors",
        ),
    )
    if (
        split
        != {
            "scene_template_leaks": [],
            "answer_leaks": [],
            "visual_style_leaks": [],
            "error_mechanism_leaks": [],
            "ood_pair_mismatches": [],
            "ood_pair_count": 100,
            "preregistered_ood_factors": ["visual_style", "error_mechanism"],
            "ood_changed_factors": ["visual_style", "error_mechanism"],
        }
        or root["split_audit_error"] is not None
    ):
        raise ValueError("Phase-D split audit is incomplete or contradictory")
    _validate_visual_factor_audit(root["visual_factor_realization_audit"])
    image_shift = _closed(
        root["ood_image_shift"],
        name="Phase-D OOD image-shift audit",
        fields=("complete", "checked_pair_count", "violations"),
    )
    if image_shift != {"complete": True, "checked_pair_count": 100, "violations": []}:
        raise ValueError("Phase-D ood_image_shift audit is incomplete")
    _validate_joint_independence(root["style_semantic_joint_independence"])
    _validate_answer_balance(root["answer_balance"])


def validate_post_gpu_execution_audit(
    audit: Mapping[str, object],
    *,
    artifact_type: str,
    stage: str,
    checkpoint_sha256: str,
    dataset_manifest_sha256: str,
    dataset_manifest_self_sha256: str,
    dataset_content_sha256: str,
    phase_d_audit_sha256: str,
    prediction_or_rollout_manifest_sha256: str,
    producer_config_sha256: str,
    producer_records_path: Path,
    producer_records_sha256: str,
    producer_record_count: int,
    seeds: Sequence[int],
    model_revision: str,
    verl_revision: str,
    sha256_file,
) -> None:
    """Validate draft bindings, then fail until authenticated clearance exists."""

    root = _closed(
        audit,
        name="post-GPU execution audit",
        fields=(
            "schema_version",
            "artifact_type",
            "stage",
            "status",
            "gpu_execution_completed",
            "started_at",
            "ended_at",
            "command",
            "gpu_uuids",
            "seeds",
            "model_revision",
            "verl_revision",
            "checkpoint_sha256",
            "dataset",
            "phase_d",
            "preflight_plan",
            "runtime_clearance",
            "producer",
            "state_injection_audit",
        ),
    )
    if root["schema_version"] != 3 or root["artifact_type"] != artifact_type:
        raise ValueError("post-GPU execution audit schema or artifact_type is invalid")
    if root["stage"] != stage or root["status"] != "completed":
        raise ValueError("post-GPU execution audit stage/status is invalid")
    if root["gpu_execution_completed"] is not True:
        raise ValueError("post-GPU execution audit does not record completed GPU execution")
    started = root["started_at"]
    ended = root["ended_at"]
    if (
        not isinstance(started, str)
        or not isinstance(ended, str)
        or _STAMP.fullmatch(started) is None
        or _STAMP.fullmatch(ended) is None
        or started >= ended
    ):
        raise ValueError("post-GPU execution audit timestamps are invalid or unordered")
    _safe_command(root["command"])
    gpu_uuids = root["gpu_uuids"]
    if (
        not isinstance(gpu_uuids, list)
        or not gpu_uuids
        or len(set(gpu_uuids)) != len(gpu_uuids)
        or any(not isinstance(item, str) or _GPU_UUID.fullmatch(item) is None for item in gpu_uuids)
    ):
        raise ValueError("post-GPU execution audit requires machine-shaped unique GPU UUIDs")
    if root["seeds"] != list(seeds):
        raise ValueError("post-GPU execution audit seeds differ from the registered protocol")
    for field, expected in (
        ("model_revision", model_revision),
        ("verl_revision", verl_revision),
        ("checkpoint_sha256", checkpoint_sha256),
    ):
        if root[field] != expected:
            raise ValueError(f"post-GPU execution audit {field} does not match")

    dataset = _closed(
        root["dataset"],
        name="post-GPU dataset binding",
        fields=("manifest_file_sha256", "manifest_self_sha256", "content_sha256"),
    )
    if dataset != {
        "manifest_file_sha256": dataset_manifest_sha256,
        "manifest_self_sha256": dataset_manifest_self_sha256,
        "content_sha256": dataset_content_sha256,
    }:
        raise ValueError("post-GPU execution audit dataset binding differs from frozen CVA-v2")
    phase_d = _closed(
        root["phase_d"],
        name="post-GPU Phase-D binding",
        fields=("audit_sha256", "schema_version", "phase_d_ready", "human_signoff"),
    )
    if phase_d != {
        "audit_sha256": phase_d_audit_sha256,
        "schema_version": 2,
        "phase_d_ready": True,
        "human_signoff": True,
    }:
        raise ValueError("post-GPU execution audit is not bound to ready reviewed Phase-D")
    plan = _closed(
        root["preflight_plan"],
        name="post-GPU preflight binding",
        fields=("path", "sha256", "artifact_type", "execution_permitted", "large_gpu_started"),
    )
    _existing_hash_bound_file(
        plan["path"], plan["sha256"], name="preflight plan", sha256_file=sha256_file
    )
    if (
        plan["artifact_type"] != "execution_plan"
        or plan["execution_permitted"] is not False
        or plan["large_gpu_started"] is not False
    ):
        raise ValueError("preflight plan must be the hardened non-executing boundary artifact")
    clearance = _closed(
        root["runtime_clearance"],
        name="post-GPU runtime clearance",
        fields=(
            "passed",
            "network_disabled",
            "local_files_only",
            "trust_remote_code",
            "use_safetensors",
            "container_image_digest",
            "wheelhouse_manifest_sha256",
            "sbom_sha256",
            "vulnerability_audit_sha256",
        ),
    )
    if (
        clearance["passed"] is not True
        or clearance["network_disabled"] is not True
        or clearance["local_files_only"] is not True
        or clearance["trust_remote_code"] is not False
        or clearance["use_safetensors"] is not True
        or not isinstance(clearance["container_image_digest"], str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", clearance["container_image_digest"]) is None
    ):
        raise ValueError("post-GPU runtime clearance is not hardened")
    for name in ("wheelhouse_manifest_sha256", "sbom_sha256", "vulnerability_audit_sha256"):
        _digest(clearance[name], name=f"runtime clearance {name}")
    producer = _closed(
        root["producer"],
        name="post-GPU producer binding",
        fields=(
            "config_path",
            "config_sha256",
            "records_path",
            "records_sha256",
            "record_count",
            "manifest_sha256",
        ),
    )
    _existing_hash_bound_file(
        producer["config_path"],
        producer["config_sha256"],
        name="producer config",
        sha256_file=sha256_file,
    )
    if producer != {
        "config_path": producer["config_path"],
        "config_sha256": producer_config_sha256,
        "records_path": str(producer_records_path),
        "records_sha256": producer_records_sha256,
        "record_count": producer_record_count,
        "manifest_sha256": prediction_or_rollout_manifest_sha256,
    }:
        raise ValueError("post-GPU execution audit producer binding differs from source artifacts")
    state = _closed(
        root["state_injection_audit"],
        name="post-GPU state-injection audit",
        fields=(
            "passed",
            "image_hidden",
            "isolation_mode",
            "adapter_sha256",
            "reviewed_adapter_sha256",
        ),
    )
    if (
        state["passed"] is not True
        or state["image_hidden"] is not True
        or state["isolation_mode"] != "separate_text_only_worker"
        or _digest(state["adapter_sha256"], name="state adapter SHA-256")
        != state["reviewed_adapter_sha256"]
    ):
        raise ValueError("post-GPU state-injection audit is not reviewed and isolated")
    raise PostGPUAuthenticationPending(
        "authenticated post-GPU gate extension is not implemented; self-attested runtime "
        "clearance cannot establish accepted execution evidence"
    )


__all__ = [
    "PostGPUAuthenticationPending",
    "validate_post_gpu_execution_audit",
    "validate_ready_phase_d_audit",
]
