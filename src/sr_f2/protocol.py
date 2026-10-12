"""Immutable SR-F2 registration, independent of all SR-F1.x receipts."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from sr_f1.contract import file_hash
from sr_f12.evaluation import verify_source_manifest
from sr_f12.protocol import build_config as parent_config
from sr_f12.protocol import immutable_json, object_hash

AMENDMENT_ID = "SR-F2-20261013"
ARMS = ("EARLY", "J6", "DEC_PRIME", "ADAPT")
SEEDS = (71001, 71002, 71003)
TEACHER_QOS = "soujanya-poria-startfund-2026-03"
PARENT_ROOT = "/projects/_ssd/varunssd/louis-ssvc/sr_f12_20261010"
DOCUMENT_DIR = Path(__file__).resolve().parents[2] / "docs/sr_f2"


def build_config(*, c4_rule="noninferiority", protocol_variants=("v0", "v3")):
    if c4_rule not in ("noninferiority", "literal_zero_rule"):
        raise ValueError("Unknown C4 clarification")
    protocol_variants = tuple(protocol_variants)
    if protocol_variants not in (("v0", "v3"), ("v0", "v1", "v2", "v3")):
        raise ValueError("Unregistered FREE/LIST panel")
    plan = parent_config(microbatch_sequences=1)
    plan.update(version=AMENDMENT_ID, scope="credit_timing_and_sufficient_certificate")
    plan["resources"].update(
        max_teacher_qos_gpus=8,
        max_teacher_qos_submitted_jobs_per_user=5,
        max_workers_across_active_project_runs=8,
        workers_per_gpu=1,
        gpus_per_packed_job=4,
        seed_first_waves=False,
        scheduling="two_persistent_pools_shared_dependency_queue",
        immediate_per_model_endpoint=True,
        teacher_gpu_limit_scope="all_users_actual_pending_and_unknown",
    )
    plan["training"].update(
        arms=list(ARMS),
        optional_arms=[],
        checkpoint_steps=list(range(0, 97, 8)),
        lr=5e-5,
        gradient_microbatch_sequences=1,
    )
    plan["reward"].update(
        scorer="v2",
        early_gate_last_step=12,
        gate_equal_h_tolerance=1e-9,
        dec_prime_full_batch_normalisation=False,
        dec_prime_group_max_absolute_bound="2*sqrt(7)",
        j6_contract="four_Alpha_and_two_gold_selected_Beta_allow_superset_to_eight",
    )
    plan["monitor"] = dict(
        pool="MONITOR",
        questions=128,
        questions_per_family=32,
        K=4,
        steps=list(range(0, 97, 8)),
        independent_rng=True,
        r1_gap_strictly_above=0.15,
        r2_e_drop_at_least=0.10,
        r2_a_gain_at_least=0.10,
        adapt_action="latch_A_to_J_per_family_R1_only",
    )
    plan["evaluation"].update(
        protocols=["evidence_answer", "answer_only"],
        train_fit_steps=[],
        endpoint_monitor_samples=0,
        monitor_only_at_baseline=False,
        endpoint_pools=["TEST_ID", "TEST_COMPOSITION", "CHARTQA"],
        endpoint_records_per_model=15300,
        test_summary_after_main_paths=12,
        publication_table_samples=None,
        complete_registered_samples=None,
    )
    plan["diagnostic_views"] = dict(
        historical_paths=12,
        historical_steps=list(range(0, 97, 8)),
        historical_records=79872,
        models=13,
        protocol_variants=list(protocol_variants),
        protocol_K=2,
        protocol_questions=256 * len(protocol_variants),
        free_max_new_tokens=256,
        list_max_new_tokens=128,
        free_list_prefill=False,
        answer_extraction="last_balanced_JSON_object",
        interval_roots=16,
        interval_charts=2,
        interval_K=4,
        interval_views=["V0", "V1", "V2", "V3", "V4"],
        interval_records=8320,
        never_enter_rl_loss=True,
    )
    plan["statistics"].update(
        primary_contrasts=[[arm, "J"] for arm in ARMS],
        primary_metrics=["TEST_ID_J", "TOPK_A", "TEST_ID_J", "TEST_ID_J"],
        primary_bonferroni_CI=0.9875,
        seed_rule="all_three_paired_same_sign",
        c4_rule=c4_rule,
        c4_noninferiority_margin_pp=-2,
        c4_seed_rule="all_three_strictly_gt_minus_2pp"
        if c4_rule == "noninferiority"
        else "all_three_same_sign",
        c4_secondary_untriggered_A_rule="CI_excludes_zero_and_three_seed_same_sign",
        c4_secondary_plus_2pp_is_gate=False,
        optional_secondary_contrasts=[],
        paired_t_intervals="report_sensitivity_not_replace_registered_dual_rule",
    )
    plan["technical_check"] = dict(
        receipt="TECHNICAL_CHECK_SR_F2.json",
        pilot_arms=list(ARMS),
        pilot_pool="ENGINE",
        pilot_updates=8,
        maximum_lr_adjustments=0,
        non_scientific_seed_required=True,
        require_finite=True,
        require_parameter_change=True,
        require_trigger_logs=True,
        require_committed_state_hashes=True,
        require_training_schedule_identity=True,
    )
    plan["history_policy"] = dict(
        preserve_all_prior_results=True,
        preserve_all_new_results=True,
        preserve_technical_failures=True,
        old_controls_reused_without_retraining=True,
        old_official_scores_unchanged=True,
        parent=PARENT_ROOT,
        missing_checkpoint_fallback="three_A_paths_with_monitor_only_if_missing_verified",
    )
    return plan


def validate_config(plan):
    if isinstance(plan, (str, Path)):
        plan = json.loads(Path(plan).read_text())
    expected = build_config(
        c4_rule=plan["statistics"]["c4_rule"],
        protocol_variants=plan["diagnostic_views"]["protocol_variants"],
    )
    if plan != expected:
        raise PermissionError("Unregistered SR-F2 configuration mutation")
    return copy.deepcopy(plan)


def scientific_matrix():
    return [
        dict(
            model_id=f"SRF2_{arm}_s{seed}",
            arm=arm,
            seed=seed,
            wave=i + 1,
            updates=96,
            gpu_count=1,
            qos=TEACHER_QOS,
        )
        for i, seed in enumerate(SEEDS)
        for arm in ARMS
    ]


def technical_matrix():
    return [
        dict(
            model_id=f"SRF2_PILOT_{arm}",
            arm=arm,
            seed=2026101301,
            wave=0,
            updates=8,
            gpu_count=1,
            qos=TEACHER_QOS,
        )
        for arm in ARMS
    ]


def register(root, plan, *, source_manifest, code_root, prior_audit, decisions):
    """Freeze only after actual CPU checks and explicit scope clarifications."""
    root, code_root = Path(root).resolve(strict=True), Path(code_root).resolve(strict=True)
    plan = validate_config(plan)
    if decisions.get("status") != "USER_CLARIFIED":
        raise PermissionError("C4 and diagnostic cardinality require recorded clarification")
    if decisions.get("c4_rule") != plan["statistics"]["c4_rule"] or (
        decisions.get("protocol_variants") != plan["diagnostic_views"]["protocol_variants"]
    ):
        raise PermissionError("Configuration differs from clarification")
    source_manifest = Path(source_manifest).resolve(strict=True)
    source_hash = verify_source_manifest(source_manifest, code_root)
    source = json.loads(source_manifest.read_text())
    commit = source.get("git_commit", "")
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise PermissionError("Committed source identity required")
    prior_audit = Path(prior_audit).resolve(strict=True)
    prior = json.loads(prior_audit.read_text())
    if prior.get("status") != "PASS":
        raise PermissionError("Prior data audit is not PASS")
    prior_sources = prior_audit.parent / "SOURCE_MANIFEST.json"
    if file_hash(prior_sources) != prior["source_manifest_sha256"]:
        raise PermissionError("Prior source inventory hash differs")
    required = (
        "CPU_ACCEPTANCE.json",
        "INPUT_IDENTITY.json",
        "DIAGNOSTIC_MANIFEST.json",
        "NATIVE_DIAGNOSTIC_PROMPTS.json",
        "prior_audit/blind/SUMMARY.json",
        "prior_audit/blind/SOURCE_MANIFEST.json",
        "prior_audit/rescore/SUMMARY.json",
        "prior_audit/rescore/SOURCE_MANIFEST.json",
        "prior_audit/chartqa122/COHORT_VERIFICATION.json",
        "prior_audit/chartqa122/COHORT_ROWS.jsonl",
        "prior_audit/chartqa122/SOURCE_MANIFEST.json",
    )
    for phase in ("blind", "rescore"):
        result = json.loads((root / "prior_audit" / phase / "SUMMARY.json").read_text())
        if result.get("status") != "PASS":
            raise PermissionError("Required prior CPU evidence phase failed: " + phase)
    if json.loads((root / required[0]).read_text()).get("status") != "PASS":
        raise PermissionError("CPU acceptance must pass before GPU")
    cohort = json.loads((root / "prior_audit/chartqa122/COHORT_VERIFICATION.json").read_text())
    if (
        cohort.get("status") != "PASS"
        or cohort.get("count") != 122
        or cohort.get("raw_evidence_status") != "RAW_AND_GOLD_REVERIFIED"
        or cohort.get("raw_output_count") != 1586
        or len(cohort.get("raw_model_ids", [])) != 13
        or cohort["rows"]["sha256"] != file_hash(root / "prior_audit/chartqa122/COHORT_ROWS.jsonl")
        or cohort["source_manifest_sha256"]
        != file_hash(root / "prior_audit/chartqa122/SOURCE_MANIFEST.json")
    ):
        raise PermissionError("Requested post-hoc ChartQA cohort and raw evidence not verified")
    receipt = dict(
        version=AMENDMENT_ID,
        status="FROZEN_BEFORE_GPU",
        root=str(root),
        source_commit=commit,
        source_manifest=str(source_manifest),
        code_root=str(code_root),
        source_manifest_sha256=source_hash,
        config_sha256=object_hash(plan),
        original_plan_sha256=file_hash(code_root / "docs/sr_f2/CLAUDE_SR_F2_PLAN_zh.md"),
        prior_audit=str(prior_audit),
        prior_audit_sha256=file_hash(prior_audit),
        prior_sources=str(prior_sources),
        prior_sources_sha256=file_hash(prior_sources),
        decisions=decisions,
        main_matrix=scientific_matrix(),
        bound_files={name: file_hash(root / name) for name in required},
        old_test_seen_for_design=True,
        new_test_seen=False,
    )
    if not (root / "REGISTRATION.json").exists():
        for name in ("runs", "technical", "raw", "sealed", "released"):
            if (root / name).exists() and any((root / name).rglob("*")):
                raise PermissionError("Registration must precede new GPU outputs")
    immutable_json(root / "config/SR_F2.json", plan)
    return immutable_json(root / "REGISTRATION.json", receipt)


def verify_registration(root, *, require_technical=False):
    root = Path(root).resolve(strict=True)
    if (root / "STOP").exists():
        raise PermissionError("SR-F2 STOP exists")
    receipt = json.loads((root / "REGISTRATION.json").read_text())
    plan = validate_config(root / "config/SR_F2.json")
    if (
        receipt.get("status") != "FROZEN_BEFORE_GPU"
        or receipt.get("root") != str(root)
        or receipt.get("config_sha256") != object_hash(plan)
        or receipt.get("main_matrix") != scientific_matrix()
    ):
        raise PermissionError("SR-F2 registration identity differs")
    actual = verify_source_manifest(receipt["source_manifest"], receipt["code_root"])
    if actual != receipt["source_manifest_sha256"]:
        raise PermissionError("SR-F2 registered source differs")
    if file_hash(receipt["prior_audit"]) != receipt["prior_audit_sha256"]:
        raise PermissionError("Prior audit changed")
    if file_hash(receipt["prior_sources"]) != receipt["prior_sources_sha256"]:
        raise PermissionError("Prior source inventory changed")
    for relative, expected in receipt["bound_files"].items():
        if file_hash(root / relative) != expected:
            raise PermissionError("Registered input/check evidence changed: " + relative)
    if require_technical:
        technical = json.loads((root / "TECHNICAL_CHECK_SR_F2.json").read_text())
        if (
            technical.get("status") != "PASS"
            or technical.get("config_sha256") != object_hash(plan)
            or technical.get("registration_sha256") != object_hash(receipt)
        ):
            raise PermissionError("Four-arm technical acceptance is absent or differs")
        history = json.loads((root / "HISTORY_COMPLETE.json").read_text())
        if (
            history.get("status") != "COMPLETE"
            or history.get("paths") != 12
            or history.get("registration_sha256") != object_hash(receipt)
        ):
            raise PermissionError("Historical trigger dry run must finish before science")
    return receipt


def verify_input_files(root):
    """Authenticate actual copied inputs once at each GPU worker's entry."""
    root = Path(root).resolve(strict=True)
    identity = json.loads((root / "INPUT_IDENTITY.json").read_text())
    for item in identity["files"]:
        relative = Path(item["path"])
        path = (root / relative).resolve(strict=True)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not path.is_relative_to(root)
            or file_hash(path) != item["sha256"]
            or path.stat().st_size != item["bytes"]
        ):
            raise PermissionError("Registered input bytes changed: " + str(relative))
    from sr_f1.data import load_schedule

    for seed in SEEDS:
        if object_hash(load_schedule(root, seed)) != identity["schedule_hashes"][str(seed)]:
            raise PermissionError("Frozen paired training stream changed")
    return identity


def finalize_readiness(root):
    """CPU-only barriers; complete worker evidence must predate each next stage."""
    from sr_f12.protocol import scientific_matrix as old_matrix

    from .orchestration import verify_task

    root = Path(root).resolve(strict=True)
    registration = verify_registration(root)
    registration_hash = object_hash(registration)
    history_tasks = [dict(kind="history", model_id=r["model_id"]) for r in old_matrix()]
    history_path = root / "HISTORY_COMPLETE.json"
    if history_path.exists():
        saved = json.loads(history_path.read_text())
        if (
            saved.get("status") != "COMPLETE"
            or saved.get("paths") != 12
            or saved.get("records") != 79872
            or saved.get("registration_sha256") != registration_hash
            or len(saved.get("tasks", [])) != 12
        ):
            raise PermissionError("Existing history readiness differs")
        for ref, task in zip(saved["tasks"], history_tasks, strict=True):
            complete = json.loads(
                (root / "tasks/history" / task["model_id"] / "COMPLETE.json").read_text()
            )
            if {key: ref.get(key) for key in ("kind", "model_id")} != task or object_hash(
                complete
            ) != ref["receipt_sha256"]:
                raise PermissionError("Previously accepted history completion changed")
    elif all(
        (root / "tasks/history" / t["model_id"] / "COMPLETE.json").exists() for t in history_tasks
    ):
        refs = [verify_task(root, task, registration_hash) for task in history_tasks]
        immutable_json(
            root / "HISTORY_COMPLETE.json",
            dict(
                status="COMPLETE",
                paths=12,
                records=79872,
                tasks=refs,
                registration_sha256=registration_hash,
            ),
        )
    pilot_tasks = [dict(kind="pilot", model_id=r["model_id"]) for r in technical_matrix()]
    technical_path = root / "TECHNICAL_CHECK_SR_F2.json"
    if technical_path.exists():
        from .runner import validate_technical_receipt

        technical = json.loads(technical_path.read_text())
        if technical.get("registration_sha256") != registration_hash:
            raise PermissionError("Existing technical readiness differs")
        validate_technical_receipt(technical, validate_config(root / "config/SR_F2.json"))
    elif all(
        (root / "tasks/pilot" / t["model_id"] / "COMPLETE.json").exists() for t in pilot_tasks
    ):
        from .runner import technical_check

        for task in pilot_tasks:
            verify_task(root, task, registration_hash)
        # All eight-step COMPLETE receipts exist: this path only aggregates and
        # validates completed pilots; no subprocess or GPU work is eligible.
        technical_check(validate_config(root / "config/SR_F2.json"), root)
