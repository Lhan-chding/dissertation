import json
from pathlib import Path

import pytest

from mm_core.execution import atomic_json, sha256_file
from mm_core.release import (
    format_not_ready,
    release_integrity,
    release_report,
    verify_scoring_receipt,
)
from mm_core.scoring import build_format_report
from mm_core.vl_runtime import CHAT_TEMPLATE_KWARGS, MODEL_ID, hash_json


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def scored_run(tmp_path):
    root = tmp_path / "audit"
    root.mkdir()
    questions = root / "data/questions.jsonl"
    output = root / "raw/MEASUREMENT_AUDIT/outputs_0.jsonl"
    scored = root / "scoring/MEASUREMENT_AUDIT/SCORED_OUTPUTS.jsonl"
    jsonl(questions, [{"question_id": "q"}])
    jsonl(output, [{"request_id": "r", "raw_text": "old response"}])
    jsonl(scored, [{"request_id": "r", "A_full": True}])
    source = Path(__file__).resolve().parents[2] / "src/mm_core"
    receipt = dict(
        status="SCORED",
        scored_record_count=1,
        question_file_sha256=sha256_file(questions),
        raw_file_sha256=[dict(name=str(output.relative_to(root)), sha256=sha256_file(output))],
        field_score_file_sha256=[],
        scored_outputs_sha256=sha256_file(scored),
        implementation_sha256=sha256_file(source / "scoring.py"),
        contracts_sha256=sha256_file(source / "contracts.py"),
    )
    atomic_json(root / "tables/MEASUREMENT_AUDIT/SCORING_RECEIPT.json", receipt)
    freeze = dict(
        status="FROZEN",
        dev_authorized=False,
        model_id=MODEL_ID,
        chat_template_kwargs=CHAT_TEMPLATE_KWARGS,
        chat_template_kwargs_hash=hash_json(CHAT_TEMPLATE_KWARGS),
        readability_review_status="PASS",
        old_work_stop_status="already_stopped",
        account_and_resource_permission_verified=True,
        model_weights_hash="model",
        processor_hash="processor",
        environment_lock_hash="env",
        chat_template_hash="chat",
        model_revision="revision",
        project_commit="commit",
        file_hashes={"data/questions.jsonl": sha256_file(questions)},
        source_hashes={},
    )
    atomic_json(root / "manifests/PRE_INFERENCE_FREEZE.json", freeze)
    atomic_json(
        root / "manifests/RESOURCE_OVERRIDE.json",
        dict(max_concurrent_gpus=5, max_allocated_gpu_hours=None),
    )
    return root, freeze, receipt


def test_format_failure_mapping_matches_real_scorer_output():
    question = dict(
        question_id="q",
        root_family_id="r",
        split="FORMAT_CHECK",
        readset_id="set",
        image_hash="image",
        chart_type="line",
        operation="sum",
        V="low",
        D="low",
        true_values=[12, 34],
        delta=1,
        ordered_item_ids=["x", "y"],
        quantity_units="count",
    )
    outputs = [
        dict(
            request_id=f"request-{i}",
            question_id="q",
            sample_index=i,
            stage="FORMAT_CHECK",
            model_hash="m",
            status="completed",
            raw_text="{}",
            truncated=False,
        )
        for i in range(4)
    ]
    report = build_format_report([question], outputs, stage="FORMAT_CHECK", strict_panel=False)
    assert report["status"] == "NOT_PASSED"
    assert report["cohorts"][0]["status"] == "FORMAT_NOT_READY"
    assert format_not_ready(report)
    report["coverage"] = {"complete": False}
    assert not format_not_ready(report)


def test_scoring_receipt_verifies_current_inputs_and_output(scored_run):
    root, freeze, receipt = scored_run
    assert verify_scoring_receipt(root, receipt) == receipt
    integrity, complete = release_integrity(root, freeze, None, None, receipt, None)
    assert integrity["status"] == "PASS" and not complete
    path = root / "raw/MEASUREMENT_AUDIT/outputs_0.jsonl"
    jsonl(path, [{"request_id": "r", "raw_text": "changed response; same request ID"}])
    with pytest.raises(PermissionError, match="Stale scoring input"):
        verify_scoring_receipt(root, receipt)
    integrity, _ = release_integrity(root, freeze, None, None, receipt, None)
    assert integrity["status"] == "FAIL"
    assert integrity["issues"][0]["artifact"] == "scoring_receipt"


@pytest.mark.parametrize("target", ["question", "scored", "new_shard"])
def test_scoring_receipt_rejects_changed_dataset_outputs_or_shard_set(scored_run, target):
    root, _, receipt = scored_run
    path = {
        "question": "data/questions.jsonl",
        "scored": "scoring/MEASUREMENT_AUDIT/SCORED_OUTPUTS.jsonl",
        "new_shard": "raw/MEASUREMENT_AUDIT/outputs_1.jsonl",
    }[target]
    jsonl(root / path, [{"changed": True}])
    with pytest.raises(PermissionError):
        verify_scoring_receipt(root, receipt)


def test_release_cannot_label_stale_evidence_ready(scored_run, tmp_path):
    root, _, _ = scored_run
    jsonl(
        root / "raw/MEASUREMENT_AUDIT/outputs_0.jsonl", [{"request_id": "r", "raw_text": "changed"}]
    )
    templates = tmp_path / "templates"
    atomic_json(templates / "DEV_FREEZE_PROPOSAL.template.json", {})
    summary = release_report(root, templates)
    assert summary["status"] == "BLOCKED_EVIDENCE_INTEGRITY"
    assert summary["evidence_integrity"]["status"] == "FAIL"
    assert summary["dev_authorized"] is False


def test_release_validates_current_freeze_model_identity(scored_run):
    root, freeze, _ = scored_run
    atomic_json(root / "manifests/PRE_INFERENCE_FREEZE.json", {**freeze, "model_id": "old-model"})
    integrity, _ = release_integrity(root, freeze, None, None, None, None)
    assert integrity["status"] == "FAIL"
    assert integrity["issues"][0]["artifact"] == "pre_inference_freeze"


@pytest.mark.parametrize("artifact", ["common", "format"])
def test_status_strings_do_not_replace_common_and_format_provenance(scored_run, artifact):
    (
        root,
        freeze,
        _,
    ) = scored_run
    common, format_report = None, None
    if artifact == "common":
        common = dict(status="VERIFIED", model_hash="unverified-model")
        atomic_json(root / "manifests/COMMON_START.json", common)
    else:
        format_report = dict(status="PASS", gate_passed=True)
        atomic_json(root / "tables/FORMAT_CHECK_REPORT.json", format_report)
    integrity, _ = release_integrity(root, freeze, common, format_report, None, None)
    assert integrity["status"] == "FAIL"
    assert (
        integrity["issues"][0]["artifact"]
        == {"common": "common_start", "format": "independent_format"}[artifact]
    )


def test_complete_measurement_requires_all_shard_receipts_and_frozen_row_identity(
    tmp_path, monkeypatch
):
    from mm_core.execution import object_hash, read_jsonl
    from mm_core.release import verify_measurement_complete

    questions = [
        dict(question_id=f"q{i}", image_sha256=f"image{i}", prompt_sha256=f"prompt{i}")
        for i in range(384)
    ]
    jsonl(tmp_path / "data/questions.jsonl", questions)
    slots = [
        dict(
            request_id=f"r{i}-{k}", question_id=f"q{i}", sample_index=k, seed=i * 4 + k, shard=i % 2
        )
        for i in range(384)
        for k in range(4)
    ]
    plan = dict(shards=2, model_hash="model", pre_freeze_hash="freeze", expected_slots=slots)
    monkeypatch.setattr("mm_core.release.verify_stage_plan", lambda *args: plan)
    directory = tmp_path / "raw/MEASUREMENT_AUDIT"
    for shard in range(2):
        rows = []
        for slot in slots:
            if slot["shard"] != shard:
                continue
            q = questions[int(slot["question_id"][1:])]
            rows.append(
                dict(
                    request_id=slot["request_id"],
                    stage="MEASUREMENT_AUDIT",
                    question_id=slot["question_id"],
                    sample_index=slot["sample_index"],
                    seed=slot["seed"],
                    model_hash="model",
                    processor_hash="processor",
                    status="completed",
                    image_sha256=q["image_sha256"],
                    prompt_hash=q["prompt_sha256"],
                )
            )
        output = directory / f"outputs_{shard}.jsonl"
        jsonl(output, rows)
        atomic_json(
            directory / f"SHARD_{shard}_STARTED.json",
            dict(
                stage="MEASUREMENT_AUDIT",
                shard=shard,
                shards=2,
                freeze_hash="freeze",
                stage_plan_hash=object_hash(plan),
            ),
        )
        atomic_json(
            directory / f"SHARD_{shard}_COMPLETE.json",
            dict(
                stage="MEASUREMENT_AUDIT",
                shard=shard,
                shards=2,
                status="completed",
                completed=len(rows),
                output_hash=sha256_file(output),
            ),
        )
    freeze = dict(processor_hash="processor")
    assert verify_measurement_complete(tmp_path, plan, freeze)
    receipt_path = directory / "SHARD_1_COMPLETE.json"
    receipt = json.loads(receipt_path.read_text())
    receipt_path.unlink()
    assert not verify_measurement_complete(tmp_path, plan, freeze)
    atomic_json(receipt_path, receipt)
    output = directory / "outputs_1.jsonl"
    rows = read_jsonl(output)
    rows[0]["model_hash"] = "different-model"
    jsonl(output, rows)
    atomic_json(receipt_path, {**receipt, "output_hash": sha256_file(output)})
    with pytest.raises(PermissionError, match="frozen request"):
        verify_measurement_complete(tmp_path, plan, freeze)
