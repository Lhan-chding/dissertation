import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from mm_core.execution import (
    BudgetLedger,
    atomic_json,
    bind_format_report,
    checked_path,
    create_common_start,
    derive_seed,
    exclusive_json,
    object_hash,
    read_jsonl,
    register_stage_plan,
    sha256_file,
    verify_format_report,
    verify_gate,
    verify_stage_plan,
)
from mm_core.vl_runtime import hash_json


@pytest.fixture
def run(tmp_path):
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests/RESOURCE_OVERRIDE.json").write_text(
        json.dumps(
            {
                "max_concurrent_gpus": 5,
                "max_allocated_gpu_hours": None,
            }
        )
    )
    return tmp_path


@pytest.mark.parametrize("stage", ["MM-DEV", "MM-LOCK", "SER-J23", "MM-CAL", "MM-ONLINE"])
def test_forbidden_stage_rejected_before_file_access(run, stage):
    with pytest.raises(PermissionError, match="not authorized"):
        verify_gate(run, stage)


def test_unfrozen_gpu_request_rejected(run):
    (run / "manifests/PRE_INFERENCE_FREEZE.json").write_text('{"status":"DRAFT"}')
    with pytest.raises(PermissionError, match="freeze"):
        verify_gate(run, "FORMAT_BASE_TEST")


def test_budget_consumes_unknown_and_failed_attempts(run):
    ledger = BudgetLedger(run)
    ledger.reserve("completion_attempts", 4095, {"status": "unknown"})
    ledger.reserve("completion_attempts", 1, {"status": "failed"})
    with pytest.raises(RuntimeError, match="BUDGET_EXHAUSTED"):
        ledger.reserve("completion_attempts", 1, "retry")
    assert ledger.totals()["completion_attempts"] == 4096


@pytest.mark.parametrize("value", [True, -1, 0, float("nan"), float("inf"), 0.5])
def test_invalid_count_cannot_release_budget(run, value):
    with pytest.raises(ValueError):
        BudgetLedger(run).reserve("completion_attempts", value, "invalid")


def test_parallel_budget_reservations_are_serialized(run):
    def reserve(index):
        BudgetLedger(run).reserve("physical_optimizer_updates", 1, index)

    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(reserve, range(64)))
    assert BudgetLedger(run).totals()["physical_optimizer_updates"] == 64
    assert len(read_jsonl(run / "accounting/COST_LEDGER.jsonl")) == 64
    with pytest.raises(RuntimeError):
        reserve(65)


def test_gpu_hours_accounted_without_any_total_limit(run):
    ledger = BudgetLedger(run)
    ledger.reserve("allocated_gpu_hours", 5 * 1.5, {"gpus": 5, "hours": 1.5})
    ledger.reserve("allocated_gpu_hours", 1000, "additional_accounted_time")
    assert ledger.totals()["allocated_gpu_hours"] == 1007.5
    assert ledger.caps["allocated_gpu_hours"] is None
    assert all(row["status"] == "ALLOCATION_TIME_ESTIMATE" for row in read_jsonl(ledger.path))


def test_seed_changes_by_question_stage_and_sample():
    row = {"root_family_id": "r", "image_id": "i", "question_id": "q"}
    a = derive_seed("FORMAT_BASE_TEST", "model", row, 0)
    assert a == derive_seed("FORMAT_BASE_TEST", "model", row, 0)
    assert a != derive_seed("FORMAT_BASE_TEST", "model", row, 1)
    assert a != derive_seed("FORMAT_CHECK", "model", row, 0)
    assert a != derive_seed("FORMAT_BASE_TEST", "model", {**row, "question_id": "q2"}, 0)


def test_unsafe_evidence_path_rejected(run):
    for path in ("../secret", "/absolute"):
        with pytest.raises(ValueError):
            checked_path(run, path)


def test_run_claim_never_overwrites(run):
    path = run / "raw/claim.json"
    exclusive_json(path, {"status": "running"})
    with pytest.raises(FileExistsError):
        exclusive_json(path, {"status": "retry"})
    assert json.loads(path.read_text())["status"] == "running"


def _save_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def frozen_run(run):
    questions = []
    for split in ("FORMAT_TUNE", "FORMAT_CHECK"):
        for family in range(4):
            for chart in ("grouped_bar", "line"):
                for op in ("sum", "difference", "range"):
                    for visual in ("low", "high"):
                        for numeric in ("low", "high"):
                            image = f"{split}-{family}-{chart}-{visual}-{numeric}"
                            qid = f"{image}-{op}"
                            values = [34, 12, 23] if op == "range" else [34, 12]
                            questions.append(
                                dict(
                                    question_id=qid,
                                    image_id=image,
                                    root_family_id=f"{split}-{family}",
                                    split=split,
                                    chart_type=chart,
                                    operation=op,
                                    V=visual,
                                    D=numeric,
                                    true_values=values,
                                    delta=1,
                                    image_sha256=object_hash(image),
                                    prompt_sha256=object_hash(qid),
                                    readset_id=image,
                                    ordered_item_ids=[f"role{i}" for i in range(len(values))],
                                    quantity_units="count",
                                )
                            )
    _save_jsonl(run / "data/questions.jsonl", questions)
    freeze = dict(
        status="FROZEN",
        model_id="Qwen/Qwen3.5-9B",
        chat_template_kwargs={"enable_thinking": False},
        chat_template_kwargs_hash=hash_json({"enable_thinking": False}),
        dev_authorized=False,
        readability_review_status="PASS",
        old_work_stop_status="already_stopped",
        account_and_resource_permission_verified=True,
        model_weights_hash="base-model",
        processor_hash="processor",
        environment_lock_hash="env",
        chat_template_hash="chat",
        model_revision="revision",
        project_commit="commit",
        file_hashes={"data/questions.jsonl": sha256_file(run / "data/questions.jsonl")},
        source_hashes={},
    )
    atomic_json(run / "manifests/PRE_INFERENCE_FREEZE.json", freeze)
    return run


@pytest.mark.parametrize("model", [None, "Qwen/Qwen2.5-VL-3B-Instruct"])
def test_previous_model_freeze_cannot_dispatch(frozen_run, model):
    path = frozen_run / "manifests/PRE_INFERENCE_FREEZE.json"
    payload = json.loads(path.read_text())
    payload["model_id"] = model
    atomic_json(path, payload)
    with pytest.raises(PermissionError, match="requires Qwen"):
        verify_gate(frozen_run, "FORMAT_BASE_TEST")


def test_thinking_mode_cannot_change_after_model_override(frozen_run):
    path = frozen_run / "manifests/PRE_INFERENCE_FREEZE.json"
    payload = json.loads(path.read_text())
    payload["chat_template_kwargs"] = {"enable_thinking": True}
    atomic_json(path, payload)
    with pytest.raises(PermissionError, match="thinking mode"):
        verify_gate(frozen_run, "FORMAT_BASE_TEST")


def _finish_panel(root, stage, *, shards=2, skip_shard=None, invalid=False):
    plan = register_stage_plan(root, stage, shards)
    questions = {q["question_id"]: q for q in read_jsonl(root / "data/questions.jsonl")}
    for shard in range(shards):
        if shard == skip_shard:
            continue
        rows = []
        for slot in plan["expected_slots"]:
            if slot["shard"] != shard:
                continue
            q = questions[slot["question_id"]]
            truth = q["true_values"]
            answer = (
                sum(truth)
                if q["operation"] == "sum"
                else (
                    truth[0] - truth[1]
                    if q["operation"] == "difference"
                    else max(truth) - min(truth)
                )
            )
            rows.append(
                dict(
                    request_id=slot["request_id"],
                    seed=slot["seed"],
                    question_id=q["question_id"],
                    sample_index=slot["sample_index"],
                    stage=stage,
                    model_hash=plan["model_hash"],
                    processor_hash="processor",
                    image_sha256=q["image_sha256"],
                    prompt_hash=q["prompt_sha256"],
                    status="completed",
                    truncated=False,
                    raw_text="{}" if invalid else json.dumps(dict(readings=truth, answer=answer)),
                )
            )
        directory = root / f"raw/{stage}"
        _save_jsonl(directory / f"outputs_{shard}.jsonl", rows)
        atomic_json(
            directory / f"SHARD_{shard}_STARTED.json",
            dict(stage=stage, shard=shard, shards=shards, freeze_hash=plan["pre_freeze_hash"]),
        )
        atomic_json(
            directory / f"SHARD_{shard}_COMPLETE.json",
            dict(
                stage=stage,
                shard=shard,
                shards=shards,
                status="completed",
                completed=len(rows),
                output_hash=sha256_file(directory / f"outputs_{shard}.jsonl"),
            ),
        )
    return plan


def _pass_base(root):
    _finish_panel(root, "FORMAT_BASE_TEST")
    assert bind_format_report(root, "FORMAT_BASE_TEST")["gate_passed"]
    return create_common_start(root)


def test_stage_plan_rejects_mixed_partition_counts(frozen_run):
    plan = register_stage_plan(frozen_run, "FORMAT_BASE_TEST", 2)
    assert len(plan["expected_slots"]) == 384
    assert verify_stage_plan(frozen_run, "FORMAT_BASE_TEST", 1, 2) == plan
    with pytest.raises(PermissionError, match="stage plan"):
        register_stage_plan(frozen_run, "FORMAT_BASE_TEST", 3)
    with pytest.raises(PermissionError, match="Shard count"):
        verify_stage_plan(frozen_run, "FORMAT_BASE_TEST", 1, 3)


def test_complete_sharded_format_gate_and_same_model_measurement(frozen_run):
    common = _pass_base(frozen_run)
    assert common["model_hash"] == "base-model"
    _finish_panel(frozen_run, "FORMAT_CHECK", shards=5)
    report = bind_format_report(frozen_run, "FORMAT_CHECK")
    assert report["gate_passed"] and report["coverage"]["complete"]
    assert report["coverage"]["recorded_slots"] == 384
    assert verify_format_report(frozen_run, "FORMAT_CHECK") == report
    assert verify_gate(frozen_run, "MEASUREMENT_AUDIT")["model_weights_hash"] == "base-model"


def test_missing_shard_cannot_establish_common_start(frozen_run):
    _finish_panel(frozen_run, "FORMAT_BASE_TEST", skip_shard=1)
    report = bind_format_report(frozen_run, "FORMAT_BASE_TEST")
    assert report["status"] == "INCOMPLETE"
    assert not report["gate_passed"]
    assert report["coverage"]["missing_shards"] == [1]
    with pytest.raises(PermissionError, match="complete base"):
        create_common_start(frozen_run)


def _rewrite_outputs_and_receipt(root, stage, shard, mutate):
    directory = root / f"raw/{stage}"
    path = directory / f"outputs_{shard}.jsonl"
    rows = read_jsonl(path)
    mutate(rows)
    _save_jsonl(path, rows)
    receipt_path = directory / f"SHARD_{shard}_COMPLETE.json"
    receipt = json.loads(receipt_path.read_text())
    receipt.update(output_hash=sha256_file(path), completed=len(rows))
    atomic_json(receipt_path, receipt)


def test_duplicate_sample_cannot_be_deduplicated_into_a_gate(frozen_run):
    _finish_panel(frozen_run, "FORMAT_BASE_TEST")
    _rewrite_outputs_and_receipt(
        frozen_run, "FORMAT_BASE_TEST", 0, lambda rows: rows.append(rows[0])
    )
    with pytest.raises(PermissionError, match="duplicate"):
        bind_format_report(frozen_run, "FORMAT_BASE_TEST")


def test_foreign_model_output_fails_provenance(frozen_run):
    _finish_panel(frozen_run, "FORMAT_BASE_TEST")
    _rewrite_outputs_and_receipt(
        frozen_run, "FORMAT_BASE_TEST", 0, lambda rows: rows[0].update(model_hash="other-model")
    )
    with pytest.raises(PermissionError, match="provenance"):
        bind_format_report(frozen_run, "FORMAT_BASE_TEST")


def test_changed_raw_output_cannot_reuse_previous_pass(frozen_run):
    _finish_panel(frozen_run, "FORMAT_BASE_TEST")
    bind_format_report(frozen_run, "FORMAT_BASE_TEST")
    _rewrite_outputs_and_receipt(
        frozen_run, "FORMAT_BASE_TEST", 0, lambda rows: rows[0].update(raw_text="{}")
    )
    with pytest.raises(PermissionError, match="evidence or provenance"):
        verify_format_report(frozen_run, "FORMAT_BASE_TEST")


def test_changed_common_model_cannot_reuse_independent_check(frozen_run):
    _pass_base(frozen_run)
    _finish_panel(frozen_run, "FORMAT_CHECK")
    bind_format_report(frozen_run, "FORMAT_CHECK")
    path = frozen_run / "manifests/COMMON_START.json"
    common = json.loads(path.read_text())
    atomic_json(path, {**common, "model_hash": "changed-model"})
    with pytest.raises(PermissionError, match="Common start provenance"):
        verify_gate(frozen_run, "MEASUREMENT_AUDIT")


def test_failed_independent_check_stops_training_and_measurement(frozen_run):
    _pass_base(frozen_run)
    _finish_panel(frozen_run, "FORMAT_CHECK", invalid=True)
    assert not bind_format_report(frozen_run, "FORMAT_CHECK")["gate_passed"]
    with pytest.raises(PermissionError, match="did not pass"):
        verify_gate(frozen_run, "MEASUREMENT_AUDIT")
    for stage in ("BRIDGE", "FORMAT_TUNE_POST_BRIDGE"):
        with pytest.raises(PermissionError, match="already dispatched"):
            verify_gate(frozen_run, stage)
    with pytest.raises(PermissionError, match="already dispatched"):
        create_common_start(frozen_run)


def test_stale_visual_review_cannot_freeze_a_new_preflight(frozen_run):
    from mm_core.execution import _validate_preflight

    atomic_json(
        frozen_run / "manifests/PROCESSOR_PREFLIGHT.json", dict(status="CPU_PASS_PENDING_VISUAL")
    )
    with pytest.raises(PermissionError, match="not bound"):
        _validate_preflight(frozen_run, {}, dict(status="PASS", preflight_hash="previous"), {})


def test_changed_rng_cannot_reuse_registered_request_identity(frozen_run):
    _finish_panel(frozen_run, "FORMAT_BASE_TEST")
    _rewrite_outputs_and_receipt(
        frozen_run, "FORMAT_BASE_TEST", 0, lambda rows: rows[0].update(seed=rows[0]["seed"] + 1)
    )
    with pytest.raises(PermissionError, match="provenance"):
        bind_format_report(frozen_run, "FORMAT_BASE_TEST")


def test_failed_physical_attempt_is_not_a_format_only_bridge_trigger(frozen_run):
    _finish_panel(frozen_run, "FORMAT_BASE_TEST")
    _rewrite_outputs_and_receipt(
        frozen_run,
        "FORMAT_BASE_TEST",
        0,
        lambda rows: rows[0].update(status="failed", raw_text=None),
    )
    receipt_path = frozen_run / "raw/FORMAT_BASE_TEST/SHARD_0_COMPLETE.json"
    receipt = json.loads(receipt_path.read_text())
    atomic_json(receipt_path, {**receipt, "completed": receipt["completed"] - 1})
    report = bind_format_report(frozen_run, "FORMAT_BASE_TEST")
    assert report["coverage"]["failed_attempts"] == 1
    assert not report["coverage"]["complete"] and not report["gate_passed"]
    with pytest.raises(PermissionError, match="complete base"):
        create_common_start(frozen_run)


def test_fixed_bridge_common_start_verifies_adapter_bytes(frozen_run):
    from mm_core.vl_runtime import hash_json

    _finish_panel(frozen_run, "FORMAT_BASE_TEST", invalid=True)
    assert not bind_format_report(frozen_run, "FORMAT_BASE_TEST")["gate_passed"]
    prefreeze_hash = sha256_file(frozen_run / "manifests/PRE_INFERENCE_FREEZE.json")
    trigger = dict(
        triggered=True,
        root_cause="FORMAT_COMPLIANCE",
        pre_freeze_hash=prefreeze_hash,
        base_report_hash=sha256_file(frozen_run / "tables/FORMAT_BASE_TEST_REPORT.json"),
    )
    atomic_json(frozen_run / "manifests/BRIDGE_TRIGGER.json", trigger)
    adapter = frozen_run / "engineering/bridge_adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"test-only-adapter")
    hashes = {"adapter_model.safetensors": sha256_file(adapter / "adapter_model.safetensors")}
    receipt = dict(
        status="COMPLETED",
        physical_updates=16,
        sequence_exposures=128,
        pre_freeze_hash=prefreeze_hash,
        trigger_hash=sha256_file(frozen_run / "manifests/BRIDGE_TRIGGER.json"),
        adapter_path=str(adapter),
        adapter_hash=hash_json(hashes),
        file_hashes=hashes,
    )
    atomic_json(frozen_run / "engineering/BRIDGE_RECEIPT.json", receipt)
    common = create_common_start(frozen_run)
    assert common["adapter_file_hashes"] == hashes
    assert common["model_hash"] == object_hash(
        dict(base_model_weights_hash="base-model", adapter_file_hashes=hashes)
    )
    verify_gate(frozen_run, "FORMAT_TUNE_POST_BRIDGE")
    with pytest.raises(PermissionError, match="Post-bridge descriptive"):
        verify_gate(frozen_run, "FORMAT_CHECK")
    _finish_panel(frozen_run, "FORMAT_TUNE_POST_BRIDGE", invalid=True)
    post = bind_format_report(frozen_run, "FORMAT_TUNE_POST_BRIDGE")
    assert post["coverage"]["complete"] and not post["gate_passed"]
    verify_gate(frozen_run, "FORMAT_CHECK")
    (adapter / "adapter_model.safetensors").write_bytes(b"different-adapter")
    with pytest.raises(PermissionError, match="adapter bytes changed"):
        verify_gate(frozen_run, "FORMAT_CHECK")
