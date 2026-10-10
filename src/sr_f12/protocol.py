"""Independent SR-F1.2 registration; old plans and outputs are never rewritten."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from sr_f1.amendment import build_amended_config
from sr_f1.evaluation import TEST_POOLS, VIEWS, _seed

AMENDMENT_ID = "SR-F1.2-20261010"
BASELINE = "SRF1_2_COMMON_ZERO"
ARMS = ("A", "J", "GATE", "DEC")
SEEDS = (71001, 71002, 71003)
PRIMARY_CONTRASTS = (("J", "A"), ("GATE", "J"), ("GATE", "DEC"), ("DEC", "J"))
TEACHER_QOS = "soujanya-poria-startfund-2026-03"
DOCUMENT_DIR = Path(__file__).resolve().parents[2] / "docs/sr_f1/amendments/SR_F1_2_20261010"


def object_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def immutable_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    try:
        with path.open("x") as stream:
            stream.write(body)
    except FileExistsError:
        if json.loads(path.read_text()) != value:
            raise PermissionError(f"Immutable artifact differs: {path}") from None
    return value


def build_config(
    *, learning_rate=5e-5, microbatch_sequences=4, gradient_checkpointing=False, optional_part=False
):
    """Candidate defaults become final only after TECHNICAL_CHECK and freeze."""
    if isinstance(learning_rate, bool) or learning_rate not in (5e-5, 2e-5, 1e-4):
        raise ValueError("Unregistered learning rate")
    if type(microbatch_sequences) is not int or microbatch_sequences not in (1, 2, 4):
        raise ValueError("Microbatch must follow the registered 4, 2, 1 fallback")
    if type(gradient_checkpointing) is not bool or type(optional_part) is not bool:
        raise ValueError("Boolean technical choices required")
    if gradient_checkpointing and microbatch_sequences != 1:
        raise ValueError("Checkpointing is permitted only after one-sequence OOM")
    plan = copy.deepcopy(build_amended_config())
    plan["version"] = AMENDMENT_ID
    plan["resources"].update(
        gpu_qos=TEACHER_QOS,
        gpu_per_training_path=1,
        max_teacher_qos_gpus=5,
        count_pending_and_unknown=True,
        count_other_qos=False,
        touch_unrelated_jobs=False,
        seed_first_waves=True,
    )
    plan["model"]["lora"].update(
        layers="all_language_decoder_layers",
        targets="all_language_decoder_linear_layers",
        excluded=["visual", "merger", "projector", "lm_head", "embedding"],
        actual_module_names_receipt_required=True,
    )
    t = plan["training"]
    t.update(
        arms=list(ARMS),
        optional_arms=["PART"] if optional_part else [],
        lr=learning_rate,
        gradient_microbatch_sequences=microbatch_sequences,
        gradient_checkpointing=gradient_checkpointing,
        training_forward="full_sequence_right_padded_teacher_forcing",
        old_logprob="stop_gradient_current_training_logprob",
        tis_cap=2.0,
        backward_passes_per_microbatch=1,
        sampler_mismatch_mean_abs_max=0.01,
        sampler_mismatch_truncation_investigate_above=0.01,
        generation_batch_size=8,
        generation_seed="registered_first_seed_per_group",
        sampler_logprobs_field="sampler_logprobs",
        train_logprobs_storage="statistics_only_except_pilot_step1",
        pure_policy_gradient_norm=False,
    )
    plan["format_bridge"]["enabled"] = False
    plan["format_bridge"]["max_episodes"] = 0
    plan["evaluation"].update(
        answer_only_samples_per_prompt=2,
        answer_only_max_new_tokens=768,
        shift_protocols=["evidence_answer"],
        monitor_only_at_baseline=True,
        train_fit_steps=[32, 64, 96],
        baseline_train_fit_samples=128,
        baseline_measurement_samples=6656,
        endpoint_monitor_samples=0,
        baseline_measurement_test_access=False,
        test_summary_after_main_paths=12,
        within_question_batching=True,
        cross_question_batching=False,
        publication_table_samples=235892,
        complete_registered_samples=238964,
    )
    plan["diagnostic_views"].update(baseline_K=8, endpoint_K=4)
    plan["statistics"].update(
        primary_contrasts=[list(x) for x in PRIMARY_CONTRASTS],
        primary_bonferroni_CI=0.9875,
        optional_secondary_contrasts=[["PART", "J"], ["GATE", "PART"]],
        preregister_semantic_predictions_before_science=True,
    )
    plan["protocol_amendment"].update(
        id=AMENDMENT_ID,
        applies_to=["evidence_answer", "answer_only"],
        common_start=BASELINE,
        engine_learning_rate=learning_rate,
        learning_rate_option="registered_pilot_rule",
        allow_original_bridge=False,
    )
    plan["technical_check"] = dict(
        replaces="ENGINE",
        receipt="TECHNICAL_CHECK_SR_F1_2.json",
        pilot_arm="GATE",
        pilot_pool="ENGINE",
        pilot_updates=8,
        non_scientific_seed_required=True,
        pilot_initial_lr=5e-5,
        maximum_lr_adjustments=1,
        instability_lr=2e-5,
        insufficient_movement_lr=1e-4,
        stability_format_steps=[5, 6, 7, 8],
        stability_format_failure_max=0.20,
        final_token_kl_max=0.05,
        movement_mean_abs_logprob_min=0.01,
        zero_lora_comparisons=32,
        zero_lora_max_logprob_difference=0.0,
        batch_single_comparisons=16,
        batch_single_mean_abs_difference_max=0.002,
        batch_single_max_abs_difference_max=0.05,
        recovery_after_update=4,
        require_committed_state_hashes=True,
        require_resampled_bitwise_identity=False,
        optional_fast_kernels_only_above_seconds=900,
        optional_fast_kernels_selected=False,
    )
    plan["history_policy"] = dict(
        user_direct_override="delete old experiment results; do not preserve old outputs",
        supersedes_document_retention_clause=True,
        old_results_used_as_new_baseline=False,
        old_models_or_optimizer_state_used=False,
        new_run_results_retained=True,
    )
    return plan


def validate_config(plan):
    if isinstance(plan, (str, Path)):
        plan = json.loads(Path(plan).read_text())
    t = plan["training"]
    expected = build_config(
        learning_rate=t["lr"],
        microbatch_sequences=t["gradient_microbatch_sequences"],
        gradient_checkpointing=t["gradient_checkpointing"],
        optional_part=t.get("optional_arms") == ["PART"],
    )
    if plan != expected:
        raise PermissionError("Unregistered SR-F1.2 configuration mutation")
    return copy.deepcopy(plan)


def scientific_matrix(*, optional_part=False):
    main = [
        dict(
            model_id=f"SRF1_2_{arm}_s{seed}",
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
    if optional_part:
        main += [
            dict(
                model_id=f"SRF1_2_PART_s{s}",
                arm="PART",
                seed=s,
                wave=3,
                updates=96,
                gpu_count=1,
                qos=TEACHER_QOS,
                optional=True,
                scheduling="fifth_available_card_serial_paths",
            )
            for s in SEEDS
        ]
    return main


def generation_config(protocol):
    if protocol not in ("evidence_answer", "answer_only", "plain_answer"):
        raise ValueError("Unregistered protocol")
    return dict(
        do_sample=protocol != "plain_answer",
        temperature=1,
        top_p=1,
        top_k=0,
        max_new_tokens=768 if protocol != "plain_answer" else 128,
        num_return_sequences=1,
    )


def iter_model_slots(
    tasks,
    train_fit_qids,
    model_id,
    *,
    stage="baseline",
    step=None,
    test_released=False,
    optional_part=False,
):
    """Disjoint panels; baseline=6656, train_fit=128, endpoint excludes MONITOR.

    Endpoint TRAIN_FIT is a separate stage even at step96, preventing duplicate slots.
    The common model's endpoint diagnostic is already measured at baseline K8.
    """
    models = {x["model_id"] for x in scientific_matrix(optional_part=optional_part)} | {BASELINE}
    if model_id not in models or stage not in ("baseline", "train_fit", "final"):
        raise ValueError("Unregistered model/stage")
    step = (0 if model_id == BASELINE else 96) if step is None else step
    if (model_id == BASELINE and step != 0) or (model_id != BASELINE and step not in (32, 64, 96)):
        raise ValueError("Unregistered checkpoint")
    if stage == "baseline" and model_id != BASELINE:
        raise ValueError("Baseline is zero LoRA only")
    if stage == "final" and not test_released:
        raise PermissionError("TEST remains sealed until endpoint evaluation authorization")
    if stage == "final" and model_id != BASELINE and step != 96:
        raise ValueError("No intermediate TEST")
    rows = list(tasks.values()) if isinstance(tasks, dict) else list(tasks)
    task_map = {r["qid"]: r for r in rows}
    if len(task_map) != len(rows):
        raise ValueError("Duplicate qid")
    panels = []
    if stage == "baseline" or (stage == "final" and model_id != BASELINE):
        monitor = sorted((t for t in rows if t["pool"] == "MONITOR"), key=lambda t: t["qid"])
        v0 = [t for t in monitor if t["variant"] == "v0"]
        if len(monitor) != 512 or len(v0) != 64:
            raise ValueError("Incomplete MONITOR / diagnostic panel")
        if stage == "baseline":
            panels.append(("MONITOR", monitor, "evidence_answer", 8, None))
        for view in VIEWS:
            panels.append(
                (
                    "MONITOR_DIAGNOSTIC_ONLY",
                    v0,
                    "evidence_answer",
                    8 if stage == "baseline" else 4,
                    view,
                )
            )
    if stage == "train_fit":
        if len(train_fit_qids) != 32 or len(set(train_fit_qids)) != 32:
            raise ValueError("TRAIN_FIT requires 32 frozen qids")
        fit = [task_map[q] for q in train_fit_qids]
        if any(t["pool"] != "TRAIN" for t in fit):
            raise ValueError("Invalid TRAIN_FIT pool")
        panels.append(("TRAIN_FIT", fit, "evidence_answer", 4, None))
    if stage == "final":
        for pool in TEST_POOLS:
            panel = sorted((t for t in rows if t["pool"] == pool), key=lambda t: t["qid"])
            if len(panel) != (2048 if pool == "TEST_ID" else 128):
                raise ValueError(f"Incomplete {pool}")
            panels.append((pool, panel, "evidence_answer", 4, None))
            if pool == "TEST_ID":
                panels.append((pool, panel, "answer_only", 2, None))
    for pool, panel, protocol, k, view in panels:
        for index, task in enumerate(panel):
            for draw in range(k):
                identity = [model_id, step, pool, protocol, view or "original", task["qid"], draw]
                yield dict(
                    slot_id="|".join(map(str, identity)),
                    model_id=model_id,
                    step=step,
                    pool=pool,
                    protocol=protocol,
                    view=view,
                    qid=task["qid"],
                    root_id=task["root_id"],
                    family=task["family"],
                    chart=task["chart"],
                    variant=task["variant"],
                    draw=draw,
                    samples_per_prompt=k,
                    monitor_block=index // 16 if pool == "MONITOR" else None,
                    sampling_seed=_seed(
                        "SRF1_EVALUATION", step, pool, protocol, view, task["qid"], draw
                    ),
                    group_seed=_seed("SRF1_EVALUATION", step, pool, protocol, view, task["qid"], 0),
                    generation=generation_config(protocol),
                )


def pilot_learning_rate_decision(
    *,
    finite,
    format_failures_last_four,
    token_count_last_four,
    step8_kl,
    movement,
    adjusted=False,
    current_lr=5e-5,
):
    """Only stability/movement, never reward, A or J, selects the uniform learning rate."""
    import math

    if (
        type(finite) is not bool
        or token_count_last_four != 512
        or not 0 <= format_failures_last_four <= 512
    ):
        raise ValueError("Require four complete 128-sequence pilot updates")
    if current_lr not in (5e-5, 2e-5, 1e-4):
        raise ValueError("Unregistered learning rate")
    unstable = (
        not finite
        or not math.isfinite(step8_kl)
        or format_failures_last_four / 512 > 0.20
        or step8_kl > 0.05
    )
    if unstable:
        return dict(
            status="STOP_UNSTABLE" if adjusted else "RETRY_ONCE",
            learning_rate=current_lr if adjusted else 2e-5,
            reason="stability",
            adjusted=not adjusted,
        )
    if not math.isfinite(movement) or movement < 0:
        raise ValueError("Missing finite movement measurement")
    if not adjusted and movement < 0.01:
        return dict(
            status="RETRY_ONCE", learning_rate=1e-4, reason="insufficient_movement", adjusted=True
        )
    return dict(status="SELECTED", learning_rate=current_lr, reason="stable", adjusted=adjusted)


def register_amendment(root, plan, *, source_commit):
    """Register in a fresh F1.2 root; pilot choices remain explicitly provisional."""
    plan = validate_config(plan)
    root = Path(root).resolve(strict=True)
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("Require the committed source SHA")
    receipt = dict(
        amendment_id=AMENDMENT_ID,
        config_sha256=object_hash(plan),
        source_commit=source_commit,
        status="REGISTERED_PROVISIONAL_TECHNICAL_CHOICES",
        user_solution_sha256=hashlib.sha256(
            (DOCUMENT_DIR / "USER_SOLUTION_zh.md").read_bytes()
        ).hexdigest(),
        history_policy=plan["history_policy"],
        old_results_reused=False,
        technical_choices_require_validation=True,
    )
    if (root / "AMENDMENT.json").exists():
        return immutable_json(root / "AMENDMENT.json", receipt)
    for name in ("EXECUTION_FREEZE.json", "SCIENCE_FREEZE.json", "COMMON_START.json"):
        if (root / name).exists():
            raise PermissionError("New amendment cannot inherit an existing experiment")
    for name in ("training", "runs", "raw", "sealed", "released"):
        if (root / name).exists() and any((root / name).rglob("*")):
            raise PermissionError("Amendment registration must precede model outputs")
    immutable_json(root / "config/SR_F1_2.json", plan)
    return immutable_json(root / "AMENDMENT.json", receipt)


def freeze_scientific(
    root, plan, technical_check, predictions, *, source_commit, actual_lora_modules
):
    """Seal final engineering decisions and predictions before any scientific path."""
    from .runner import validate_technical_receipt

    plan = validate_config(plan)
    validate_technical_receipt(technical_check, plan)
    if technical_check.get("config_sha256") != object_hash(plan):
        raise PermissionError("Technical check does not certify this final configuration")
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("Require source commit identity")
    if not actual_lora_modules or actual_lora_modules != sorted(set(actual_lora_modules)):
        raise ValueError("Require the actual unique sorted language LoRA module list")
    if predictions.get("status") != "PREREGISTERED_BEFORE_SCIENCE":
        raise PermissionError("Semantic predictions must be registered first")
    root = Path(root).resolve(strict=True)
    for name in ("training", "runs"):
        if (root / name).exists() and any((root / name).rglob("*")):
            raise PermissionError("Predictions/configuration must precede scientific outputs")
    receipt = dict(
        amendment_id=AMENDMENT_ID,
        status="FROZEN_BEFORE_SCIENCE",
        config_sha256=object_hash(plan),
        source_commit=source_commit,
        technical_check_sha256=object_hash(technical_check),
        predictions_sha256=object_hash(predictions),
        actual_lora_modules=actual_lora_modules,
        lora_module_list_sha256=object_hash(actual_lora_modules),
        main_matrix=scientific_matrix(),
        optional_matrix=scientific_matrix(optional_part=True)[12:]
        if plan["training"]["optional_arms"]
        else [],
        test_summary_requires_all_main_terminal=True,
        history_policy=plan["history_policy"],
    )
    immutable_json(root / "config/SR_F1_2_FROZEN.json", plan)
    immutable_json(root / "TECHNICAL_CHECK_SR_F1_2.json", technical_check)
    immutable_json(root / "SEMANTIC_PREDICTIONS.json", predictions)
    return immutable_json(root / "SCIENCE_FREEZE.json", receipt)
