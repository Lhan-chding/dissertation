"""Frozen exports must be real files and preserve honest phase boundaries."""

import csv
import json

import pytest


def fixture_metrics():
    prompt = {
        "prompt_id": "p0",
        "base_scene_id": "s0",
        "constraint_family": "trend",
        "interface": "SYMBOLIC_FRESH",
        "rollout_count": 16,
        "pooled": {"pX": 0.0, "qX": None},
    }
    group = {
        "status": "MEASURED",
        "prompt_count": 1,
        "rollout_count": 16,
        "pooled": {"pX": 0.0, "qX": None},
    }
    return {
        "overall": group,
        "by_prompt": {"p0": prompt},
        "by_group": {"trend/SYMBOLIC_FRESH": group},
        "greedy": {
            "overall": dict(group, rollout_count=1),
            "by_prompt": {"p0": dict(prompt, rollout_count=1)},
            "by_group": {"trend/SYMBOLIC_FRESH": dict(group, rollout_count=1)},
        },
    }


def export_fixture(
    out, kind="REAL_CUDA_MODEL", status="PASS", *, complete=False, model="qwen35_9b", track="N"
):
    from src.core import canonical_hash, phase_artifacts, write_json
    from src.frozen_report import write_frozen_report

    specs = {
        "qwen35_9b": ("Qwen/Qwen3.5-9B", 32),
        "qwen25vl_3b": ("Qwen/Qwen2.5-VL-3B-Instruct", 36),
        "qwen25vl_7b": ("Qwen/Qwen2.5-VL-7B-Instruct", 28),
    }
    model_id, layers = specs[model]
    spec = {"id": model_id, "expected_layers": layers, "revision": "a" * 40}
    count = (288 if track == "N" else 176) if complete else 1
    scene_count = 144 if track == "N" else count
    metrics = fixture_metrics()
    metrics.update(track=track, expected_rollouts_per_prompt=16)
    for mode, block in (("sample", metrics), ("greedy", metrics["greedy"])):
        prompt = block["by_prompt"]["p0"]
        block["by_prompt"] = {
            f"p{i}": dict(prompt, prompt_id=f"p{i}", base_scene_id=f"s{i // 2}")
            for i in range(count)
        }
        block["overall"].update(
            prompt_count=count,
            rollout_count=count * (16 if mode == "sample" else 1),
            independent_scene_count=len({p["base_scene_id"] for p in block["by_prompt"].values()}),
        )
    identity = {
        "phase": "P3",
        "model_key": model,
        "model_id": model_id,
        "model_revision": spec["revision"],
        "model_hash": canonical_hash(spec),
        "track": track,
        "execution_kind": kind,
        "data_hash": "d" * 64,
        "config_hash": "c" * 64,
    }
    audit = {
        "phase": "P3",
        "execution_kind": kind,
        "model_key": model,
        "model_id": model_id,
        "model_revision": spec["revision"],
        "track": track,
        "passed": True,
        "raw_sample_count": count * 17,
        "sampled_rollout_count": count * 16,
        "greedy_rollout_count": count,
        "optimizer_updates": 0,
        "backward_calls": 0,
        "frozen_parameter_hash_before": "f" * 64,
        "frozen_parameter_hash_after": "f" * 64,
    }
    runtime = {
        "identity": identity,
        "model_spec": spec,
        "model_audit": {"model_id": model_id, "model_revision": spec["revision"]},
        "frozen_parameter_hash": "f" * 64,
        "optimizer_updates": 0,
        "P1_evidence": {
            "model_audit": {
                "model_id": model_id,
                "model_revision": spec["revision"],
                "execution_kind": kind,
            }
        },
    }
    runtime["max_new_tokens"] = 64 if track == "N" else 48
    runtime["generation_protocol"] = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "num_beams": 1,
        "do_sample": True,
        "max_new_tokens": runtime["max_new_tokens"],
    }
    data = {
        "track": track,
        "data_hash": identity["data_hash"],
        "scene_count": scene_count,
        "prompt_count": count,
    }
    paths = write_frozen_report(out, [], metrics, audit)
    for name, value in (
        ("identity.json", identity),
        ("model_audit.json", audit),
        ("runtime_lock.json", runtime),
        ("data_manifest.json", data),
    ):
        write_json(out / name, value)
        paths.append(out / name)
    phase_artifacts(out, "P3", status, audit, paths)
    return paths


def test_parquet_is_real_and_preserves_undefined_metrics(tmp_path):
    import pyarrow.parquet as pq

    paths = export_fixture(tmp_path)
    assert tmp_path / "metrics_by_scene.parquet" in paths
    parquet = tmp_path / "metrics_by_scene.parquet"
    assert parquet.read_bytes()[:4] == b"PAR1"
    rows = pq.read_table(parquet).to_pylist()
    assert len(rows) == 2
    assert {row["decode_mode"] for row in rows} == {"sample", "greedy"}
    assert all(row["pooled.qX"] is None for row in rows)
    with (tmp_path / "metrics_by_group.csv").open() as stream:
        csv_rows = list(csv.DictReader(stream))
    assert all(row["pooled.qX"] == "" for row in csv_rows)
    assert json.loads((tmp_path / "metrics.json").read_text())["overall"]["pooled"]["qX"] is None


@pytest.mark.parametrize("kind", ["REAL_CUDA_MODEL", "CPU_FAKE_ADAPTER_FIXTURE"])
def test_report_marks_subtask_scope_and_unrun_training(tmp_path, kind):
    export_fixture(tmp_path, kind)
    text = (tmp_path / "results_report_zh.md").read_text()
    assert kind in text
    assert "单个模型/轨道子任务" in text
    assert "不能据此宣告整个 P3 通过" in text
    status = json.loads((tmp_path / "table_status.json").read_text())
    assert status["training_steps.parquet"]["status"] == "NOT_RUN"
    assert status["flow_forks.parquet"]["status"] == "NOT_RUN"
    assert not (tmp_path / "training_steps.parquet").exists()


def test_aggregate_imports_real_completed_subtask_without_whole_p3_pass(tmp_path):
    from src.report import build_report

    export_fixture(tmp_path / "runs/P3/qwen35_9b/N", complete=True)
    build_report(tmp_path / "runs", tmp_path / "report")
    statuses = json.loads((tmp_path / "report/table_status.json").read_text())
    assert statuses["metrics_by_scene.parquet"]["status"] == "AVAILABLE_SUBTASKS"
    assert (tmp_path / "report/P3/qwen35_9b/N/metrics_by_scene.parquet").exists()
    phases = json.loads((tmp_path / "report/phase_status.json").read_text())
    assert phases["P3"]["status"] == "PARTIAL"
    text = (tmp_path / "report/results_report_zh.md").read_text()
    assert "P3 冻结评估子任务已记录" in text
    assert "先运行 P1" not in text


@pytest.mark.parametrize(
    "status,kind",
    [
        ("FAIL", "REAL_CUDA_MODEL"),
        ("PARTIAL", "REAL_CUDA_MODEL"),
        ("PASS", "CPU_FAKE_ADAPTER_FIXTURE"),
    ],
)
def test_aggregate_does_not_promote_failed_partial_or_fixture(tmp_path, status, kind):
    from src.report import build_report

    export_fixture(tmp_path / "runs/P3/qwen35_9b/N", kind, status)
    build_report(tmp_path / "runs", tmp_path / "report")
    statuses = json.loads((tmp_path / "report/table_status.json").read_text())
    assert statuses["metrics_by_scene.parquet"]["status"] == "NOT_RUN"
    assert not (tmp_path / "report/P3/qwen35_9b/N/metrics_by_scene.parquet").exists()


def test_tampered_completed_evidence_is_not_imported(tmp_path):
    from src.report import build_report

    source = tmp_path / "runs/P3/qwen35_9b/N"
    export_fixture(source)
    (source / "metrics.json").write_text("{}")
    build_report(tmp_path / "runs", tmp_path / "report")
    statuses = json.loads((tmp_path / "report/table_status.json").read_text())
    assert statuses["metrics_by_scene.parquet"]["status"] == "NOT_RUN"
    phases = json.loads((tmp_path / "report/phase_status.json").read_text())
    assert phases["P3"]["details"]["subtasks"][0]["import_status"] == "INVALID_EVIDENCE"


def test_exports_real_metrics_schema_including_empty_fixed_groups(tmp_path):
    import pyarrow.parquet as pq

    from src.frozen_metrics import build_frozen_metrics
    from src.frozen_report import write_frozen_report

    rows = [
        {
            "prompt_id": "p0",
            "base_scene_id": "s0",
            "constraint_family": "trend",
            "interface": "SYMBOLIC_FRESH",
            "category": "I",
            "decode_mode": "sample" if i < 16 else "greedy",
            "rollout_index": i if i < 16 else 0,
            "completion_length": 3,
            "stop_reason": "length",
            "syntax_valid": False,
            "copy_observation": False,
            "constraint_satisfaction": None,
        }
        for i in range(17)
    ]
    write_frozen_report(
        tmp_path,
        rows,
        build_frozen_metrics(rows),
        {"execution_kind": "REAL_CUDA_MODEL", "model_key": "qwen35_9b", "track": "N"},
    )
    records = pq.read_table(tmp_path / "metrics_by_scene.parquet").to_pylist()
    assert len(records) == 2
    assert all(record["pooled.qX"] is None for record in records)
    with (tmp_path / "metrics_by_group.csv").open() as stream:
        groups = list(csv.DictReader(stream))
    assert len(groups) == 12
    assert sum(row["status"] == "NA" for row in groups) == 10


def test_full_p3_needs_all_six_distinct_model_track_tasks(tmp_path):
    from src.report import build_report

    for model in ("qwen25vl_3b", "qwen35_9b", "qwen25vl_7b"):
        for track in ("L", "N"):
            export_fixture(
                tmp_path / "runs/P3" / model / track, complete=True, model=model, track=track
            )
    phases = build_report(tmp_path / "runs", tmp_path / "report")
    assert phases["P3"]["status"] == "PASS"
    assert phases["P3"]["details"]["missing_required_subtasks"] == []


def test_manifest_cannot_reference_artifacts_outside_subtask(tmp_path):
    from src.core import file_hash, write_json
    from src.report import build_report

    source = tmp_path / "runs/P3/qwen35_9b/N"
    export_fixture(source)
    outside = source.parent / "outside.json"
    outside.write_text("{}")
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["files"].append({"path": "../outside.json", "sha256": file_hash(outside)})
    write_json(source / "manifest.json", manifest)
    phases = build_report(tmp_path / "runs", tmp_path / "report")
    assert phases["P3"]["details"]["subtasks"][0]["import_status"] == "INVALID_EVIDENCE"
    assert not (tmp_path / "report/P3/qwen35_9b/N/metrics.json").exists()


def test_status_only_edit_cannot_promote_cpu_fixture(tmp_path):
    from src.core import write_json
    from src.report import build_report

    source = tmp_path / "runs/P3/qwen35_9b/N"
    export_fixture(source, "CPU_FAKE_ADAPTER_FIXTURE", complete=True)
    status = json.loads((source / "status.json").read_text())
    status["details"]["execution_kind"] = "REAL_CUDA_MODEL"
    write_json(source / "status.json", status)
    phases = build_report(tmp_path / "runs", tmp_path / "report")
    assert phases["P3"]["details"]["subtasks"][0]["import_status"] == "INVALID_EVIDENCE"


def test_copied_bank_with_relabeled_status_cannot_complete_matrix(tmp_path):
    import shutil

    from src.core import write_json
    from src.report import build_report

    original = tmp_path / "source"
    export_fixture(original, complete=True)
    for model in ("qwen25vl_3b", "qwen35_9b", "qwen25vl_7b"):
        for track in ("L", "N"):
            source = tmp_path / "runs/P3" / model / track
            shutil.copytree(original, source)
            status = json.loads((source / "status.json").read_text())
            status["details"].update(model_key=model, track=track)
            write_json(source / "status.json", status)
    phases = build_report(tmp_path / "runs", tmp_path / "report")
    assert phases["P3"]["status"] == "PARTIAL"
    assert (
        sum(row["import_status"] == "IMPORTED" for row in phases["P3"]["details"]["subtasks"]) == 1
    )


@pytest.mark.parametrize(
    "filename,field,value",
    [
        ("identity.json", "execution_kind", "CPU_FAKE_ADAPTER_FIXTURE"),
        ("metrics.json", "track", "L"),
        ("data_manifest.json", "prompt_count", 1),
        ("model_audit.json", "model_revision", "main"),
    ],
)
def test_cross_file_identity_cannot_be_repaired_by_rehashing_one_file(
    tmp_path, filename, field, value
):
    from src.core import file_hash, write_json
    from src.report import build_report

    source = tmp_path / "runs/P3/qwen35_9b/N"
    export_fixture(source, complete=True)
    data = json.loads((source / filename).read_text())
    data[field] = value
    write_json(source / filename, data)
    manifest = json.loads((source / "manifest.json").read_text())
    for entry in manifest["files"]:
        if entry["path"] == filename:
            entry["sha256"] = file_hash(source / filename)
    write_json(source / "manifest.json", manifest)
    phases = build_report(tmp_path / "runs", tmp_path / "report")
    assert phases["P3"]["details"]["subtasks"][0]["import_status"] == "INVALID_EVIDENCE"


def test_cross_model_changed_dataset_keeps_complete_matrix_partial(tmp_path):
    from src.core import file_hash, write_json
    from src.report import build_report

    for model in ("qwen25vl_3b", "qwen35_9b", "qwen25vl_7b"):
        for track in ("L", "N"):
            export_fixture(
                tmp_path / "runs/P3" / model / track, complete=True, model=model, track=track
            )
    source = tmp_path / "runs/P3/qwen25vl_7b/N"
    for name in ("identity.json", "data_manifest.json", "runtime_lock.json"):
        data = json.loads((source / name).read_text())
        target = data["identity"] if name == "runtime_lock.json" else data
        target["data_hash"] = "e" * 64
        write_json(source / name, data)
    manifest = json.loads((source / "manifest.json").read_text())
    for entry in manifest["files"]:
        entry["sha256"] = file_hash(source / entry["path"])
    write_json(source / "manifest.json", manifest)
    phases = build_report(tmp_path / "runs", tmp_path / "report")
    assert phases["P3"]["status"] == "PARTIAL"
    assert len(phases["P3"]["details"]["incomparable_subtasks"]) == 1
    assert all(row["import_status"] == "IMPORTED" for row in phases["P3"]["details"]["subtasks"])


@pytest.mark.parametrize(
    "missing", ["model_audit.json", "identity.json", "runtime_lock.json", "data_manifest.json"]
)
def test_required_identity_file_must_be_hashed_in_manifest(tmp_path, missing):
    from src.core import write_json
    from src.report import build_report

    source = tmp_path / "runs/P3/qwen35_9b/N"
    export_fixture(source, complete=True)
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["files"] = [row for row in manifest["files"] if row["path"] != missing]
    write_json(source / "manifest.json", manifest)
    phases = build_report(tmp_path / "runs", tmp_path / "report")
    assert phases["P3"]["details"]["subtasks"][0]["import_status"] == "INVALID_EVIDENCE"


def test_small_n_panel_cannot_be_imported_as_formal_real_result(tmp_path):
    from src.report import build_report

    export_fixture(tmp_path / "runs/P3/qwen35_9b/N", complete=False)
    phases = build_report(tmp_path / "runs", tmp_path / "report")
    assert phases["P3"]["details"]["subtasks"][0]["import_status"] == "INVALID_EVIDENCE"
