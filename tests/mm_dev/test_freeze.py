"""Independent F2 freeze boundary tests using CPU-only identity doubles."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mm_core.execution import object_hash, sha256_file
from mm_dev import data, freeze
from mm_dev.common import verify_execution

REPO = Path(__file__).resolve().parents[2]
PLAN_PATH = REPO / "docs/mm_dev_f2/design/config/MM_DEV_F2.json"
COMMIT = "1" * 40


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


@pytest.fixture
def freeze_case(tmp_path, monkeypatch):
    """Valid receipt structure; model/data validation are separately tested boundaries."""
    plan = json.loads(PLAN_PATH.read_text())
    root = tmp_path / "new"
    root.mkdir()
    history_root = tmp_path / "historical"
    historical_identity = {
        **plan["model"],
        "plan_id": "MM-CORE-HISTORICAL",
        "authorized": False,
        "model_path": "/frozen/snapshot",
        "model_files": [{"name": "config.json", "bytes": 2, "sha256": "c" * 64}],
        "max_allocated_gpu_hours": 8,
    }
    write(history_root / "manifests/PRE_INFERENCE_FREEZE.json", historical_identity)
    historical_hash = sha256_file(history_root / "manifests/PRE_INFERENCE_FREEZE.json")
    renderer = {
        "library": "Pillow",
        "version": "12.3.0",
        "font": {
            "name": "Arial.ttf",
            "sha256": "525979822591a3447cfc49d943d6f7683508e25543407871c0ed8fed05fd2bd9",
        },
    }
    history = {
        "status": "VERIFIED_FROZEN_HISTORICAL_INPUTS",
        "renderer": renderer,
        "evidence_sha256": {"manifests/PRE_INFERENCE_FREEZE.json": historical_hash},
        "ordered_pairs": {"low": [], "high": []},
        "range_sets": [],
        "source_hashes": [],
        "root_families": [],
        "visible_source_hashes": [],
        "image_hashes": [],
        "original_pixel_hashes": [],
        "processed_image_hashes": [],
        "processed_pixel_hashes": [],
    }
    write(root / "data/history_exclusions.json", history)
    for name in (
        "questions",
        "sidecars",
        "images",
        "sources",
        "model_inputs",
        "processor_routing",
        "processed_images",
    ):
        (root / f"data/{name}.jsonl").write_text("{}\n")
    (root / "data/images").mkdir()
    (root / "data/images/original.png").write_bytes(b"original-image-bytes")
    (root / "data/processed_images").mkdir()
    (root / "data/processed_images/processed.png").write_bytes(b"processed-image-bytes")
    qa = {}
    for mode in ("original", "processed"):
        sheets, samples = [], []
        for pool, count in data.POOLS.items():
            name = f"data/qa/{mode}_{pool.lower()}_first_last.png"
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_bytes((mode + pool).encode())
            sheets.append({"path": name, "sha256": sha256_file(root / name)})
            for index in (0, count - 1):
                for chart in ("grouped_bar", "line"):
                    for visual in ("low", "high"):
                        for difficulty in ("low", "high"):
                            image_id = (
                                f"mmdev-f2-{pool.lower()}-r{index:04d}"
                                f"-d{difficulty[0]}-v{visual[0]}-{chart}"
                            )
                            folder = "processed_images" if mode == "processed" else "images"
                            samples.append(
                                {
                                    "image_id": image_id,
                                    "image_path": f"data/{folder}/{image_id}.png",
                                    "pool": pool,
                                    "root_index": index,
                                    "chart_type": chart,
                                    "V": visual,
                                    "D": difficulty,
                                }
                            )
        qa[mode] = {
            "selection_rule": "first_and_last_registered_root_per_pool_all_chart_V_D_cells",
            "selected_image_count": 80,
            "samples": samples,
            "contact_sheets": sheets,
            "thumbnails_do_not_replace_full_resolution_review": True,
        }
    expected_env = json.loads(
        (PLAN_PATH.parent.parent / "evidence/E06_PROCESSOR_ENVIRONMENT_LOCK.json").read_text()
    )
    processor_identity = {
        **{
            key: plan["model"][key]
            for key in (
                "model_id",
                "processor_hash",
                "tokenizer_hash",
                "chat_template_hash",
                "chat_template_kwargs",
                "chat_template_kwargs_hash",
                "linear_kernel_identity_hash",
            )
        },
        "model_type": "qwen3_5",
        "architecture_verified": True,
        "full_attention_layers": [3, 7, 11, 15, 19, 23, 27, 31],
        "linear_kernel_identity": expected_env["linear_kernel_identity"],
        "processor_geometry": {"patch_size": 16, "merge_size": 2, "temporal_patch_size": 2},
    }
    report = {
        "plan_id": plan["plan_id"],
        "status": "CPU_PASS_PENDING_VISUAL",
        "processor_status": "ACTUAL_CPU_PROCESSOR_VERIFIED",
        "visual_review_status": "CPU_PASS_PENDING_VISUAL",
        "root_families": 292,
        "numeric_sources": 584,
        "images": 2336,
        "questions": 7008,
        "root_counts": data.POOLS,
        "panel_question_counts": {p: n * 24 for p, n in data.POOLS.items()},
        "processed_image_count": 2336,
        "processed_question_count": 7008,
        "renderer": renderer,
        "processor_identity": processor_identity,
        "processor_environment": {
            **{
                key: expected_env[key]
                for key in (
                    "processor_class",
                    "processor_preprocess_source_sha256",
                    "torch_cuda_build",
                    "cuda_context_initialized",
                    "model_weights_loaded",
                    "new_model_calls",
                )
            },
            "packages": {
                key: expected_env["packages"][key]
                for key in (
                    "torch",
                    "torchvision",
                    "transformers",
                    "Pillow",
                    "numpy",
                    "tokenizers",
                )
            },
        },
        "source_audit": {
            "status": "PASS",
            "historical_exclusion": "PASS",
            "numeric_sources": 584,
            "root_families": 292,
            "numeric_content_duplicates": 0,
        },
        "image_audit": {
            "status": "PASS",
            "processed_image_audit": "PASS",
            "historical_image_exclusion": "PASS",
            "findings": [],
            "images": 2336,
        },
        "qa": qa,
        "model_weights_loaded": False,
        "new_model_calls": 0,
    }
    report_path = root / "manifests/DATA_MATERIALIZATION.json"
    visual_path = root / "manifests/VISUAL_REVIEW.json"
    write(report_path, report)
    permission = {
        "plan_id": plan["plan_id"],
        "authorized": True,
        "run_root": str(root.resolve()),
        "max_gpus_concurrent": 5,
        "gpus_per_worker": 1,
        "gres": "gpu:pro6000:1",
        "owner": "test",
        "account": "rose",
        "qos": "override-limits-but-killable",
    }
    write(root / "manifests/ALLOCATION_PERMISSION.json", permission)

    def bind_visual():
        current = json.loads(report_path.read_text())
        write(
            visual_path,
            {
                "status": "PASS",
                "data_materialization_sha256": sha256_file(report_path),
                "model_outcomes_available": False,
                "reviewer": "independent unit-test reviewer",
                "reviewed_at_utc": "2026-10-09T00:00:00Z",
                "reviewed_contact_sheets": {
                    x["path"]: x["sha256"]
                    for value in current["qa"].values()
                    for x in value["contact_sheets"]
                },
            },
        )

    bind_visual()
    model_report = {
        "plan_id": plan["plan_id"],
        "status": "PASS",
        "historical_freeze_sha256": historical_hash,
        "model_identity": historical_identity,
        "actual_model_files": historical_identity["model_files"],
        "runtime_processor_identity": processor_identity,
        "packages": expected_env["packages"],
        "python": expected_env["python"],
        "model_weights_loaded": False,
        "model_calls": 0,
        "cuda_context_initialized": False,
    }
    monkeypatch.setattr(freeze, "inspect_model", lambda *args: copy.deepcopy(model_report))
    monkeypatch.setattr(
        freeze,
        "validate_materialization",
        lambda *args: {"status": "PASS", "questions": 7008, "images": 2336},
    )
    # Both import styles retain the independently validated history boundary.
    monkeypatch.setattr(freeze, "load_history", lambda *args: copy.deepcopy(history), raising=False)
    monkeypatch.setattr(data, "load_history", lambda *args: copy.deepcopy(history))
    return SimpleNamespace(
        root=root,
        history_root=history_root,
        historical_hash=historical_hash,
        report=report,
        report_path=report_path,
        visual_path=visual_path,
        bind_visual=bind_visual,
        model=model_report,
        history=history,
        permission=permission,
        plan=plan,
    )


def create(case):
    return freeze.freeze_run(PLAN_PATH, case.root, case.history_root, code_commit=COMMIT)


def test_valid_freeze_binds_data_qa_and_preserves_history(freeze_case):
    case = freeze_case
    result = create(case)
    assert result["plan_id"] == case.plan["plan_id"]
    assert result["authorized_stage"] == "MM-DEV" and result["authorized"] is True
    assert result["resource_policy"] == case.plan["resource"]
    assert result["resource_policy"]["gpu_hours_policy"] == "ACCOUNTING_ONLY"
    assert result["resource_policy"]["max_allocated_gpu_hours"] is None
    assert result["resource_policy"]["max_wallclock_hours_for_study"] is None
    assert result["inherited_old_resource_caps"] is False
    assert (
        sha256_file(case.history_root / "manifests/PRE_INFERENCE_FREEZE.json")
        == case.historical_hash
    )
    all_files = {
        str(p.relative_to(case.root)) for p in (case.root / "data").rglob("*") if p.is_file()
    }
    assert all_files <= result["input_hashes"].keys()
    assert len([p for p in result["input_hashes"] if p.startswith("data/qa/")]) == 10
    assert "src/mm_dev/common.py" in result["source_hashes"]
    assert verify_execution(PLAN_PATH, case.root, full_hashes=True)["status"] == "FROZEN"
    with pytest.raises(FileExistsError):
        create(case)


@pytest.mark.parametrize(
    "key,value",
    [
        ("status", "PENDING"),
        ("data_materialization_sha256", "forged"),
        ("model_outcomes_available", True),
        ("model_outcomes_available", None),
        ("reviewed_contact_sheets", {}),
    ],
)
def test_forged_visual_receipt_rejected(freeze_case, key, value):
    receipt = json.loads(freeze_case.visual_path.read_text())
    receipt[key] = value
    write(freeze_case.visual_path, receipt)
    with pytest.raises(PermissionError):
        create(freeze_case)
    assert not (freeze_case.root / "manifests/F2_FREEZE.json").exists()


def test_reviewed_sheet_tamper_rejected(freeze_case):
    path = freeze_case.report["qa"]["processed"]["contact_sheets"][0]["path"]
    (freeze_case.root / path).write_bytes(b"changed after review")
    with pytest.raises(PermissionError, match="sheet changed"):
        create(freeze_case)


@pytest.mark.parametrize(
    "key,value",
    [
        ("processor_status", "NOT_EXECUTED"),
        ("processed_image_count", 2335),
        ("processed_question_count", 7007),
    ],
)
def test_missing_processed_coverage_rejected(freeze_case, key, value):
    report = copy.deepcopy(freeze_case.report)
    report[key] = value
    write(freeze_case.report_path, report)
    freeze_case.bind_visual()
    with pytest.raises(PermissionError):
        create(freeze_case)


@pytest.mark.parametrize("mode", ["processed", "original", "all"])
def test_missing_qa_mode_cannot_be_authorized_by_rebinding_visual(freeze_case, mode):
    report = copy.deepcopy(freeze_case.report)
    if mode == "all":
        report["qa"] = {}
    else:
        del report["qa"][mode]
    write(freeze_case.report_path, report)
    freeze_case.bind_visual()
    with pytest.raises(PermissionError):
        create(freeze_case)


@pytest.mark.parametrize("field", ["contact_sheets", "samples", "selected_image_count"])
def test_incomplete_fixed_qa_selection_rejected(freeze_case, field):
    report = copy.deepcopy(freeze_case.report)
    if field == "selected_image_count":
        report["qa"]["processed"][field] = 79
    else:
        report["qa"]["processed"][field] = report["qa"]["processed"][field][:-1]
    write(freeze_case.report_path, report)
    freeze_case.bind_visual()
    with pytest.raises(PermissionError):
        create(freeze_case)


@pytest.mark.parametrize(
    "field,value",
    [
        ("root_families", 291),
        ("numeric_sources", 583),
        ("images", 2335),
        ("questions", 7007),
        ("root_counts", {"ENGINE_F2": 4}),
        ("panel_question_counts", {"DEV_EVAL": 1}),
    ],
)
def test_report_counts_must_match_f2_contract(freeze_case, field, value):
    report = copy.deepcopy(freeze_case.report)
    report[field] = value
    write(freeze_case.report_path, report)
    freeze_case.bind_visual()
    with pytest.raises(PermissionError):
        create(freeze_case)


@pytest.mark.parametrize(
    "key",
    [
        "model_id",
        "processor_hash",
        "tokenizer_hash",
        "chat_template_hash",
        "chat_template_kwargs_hash",
        "linear_kernel_identity_hash",
    ],
)
def test_materialized_processor_identity_must_match_freezing_model(freeze_case, key):
    report = copy.deepcopy(freeze_case.report)
    report["processor_identity"][key] = "wrong identity"
    write(freeze_case.report_path, report)
    freeze_case.bind_visual()
    with pytest.raises(PermissionError):
        create(freeze_case)


@pytest.mark.parametrize("field", ["source_audit", "image_audit"])
def test_failed_data_audit_cannot_freeze(freeze_case, field):
    report = copy.deepcopy(freeze_case.report)
    report[field]["status"] = "FAIL"
    write(freeze_case.report_path, report)
    freeze_case.bind_visual()
    with pytest.raises(PermissionError):
        create(freeze_case)


def test_history_exclusion_receipt_cannot_be_replaced(freeze_case):
    altered = {**freeze_case.history, "source_hashes": ["substituted-history"]}
    write(freeze_case.root / "data/history_exclusions.json", altered)
    with pytest.raises(PermissionError):
        create(freeze_case)


def test_renderer_identity_cannot_drift(freeze_case):
    report = copy.deepcopy(freeze_case.report)
    report["renderer"]["font"]["sha256"] = "wrong-font"
    write(freeze_case.report_path, report)
    freeze_case.bind_visual()
    with pytest.raises(PermissionError):
        create(freeze_case)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_gpus_concurrent", 6),
        ("gpus_per_worker", 2),
        ("gres", "gpu:a100:1"),
        ("authorized", False),
        ("run_root", "/another/run"),
    ],
)
def test_allocation_permission_contract(freeze_case, field, value):
    permission = {**freeze_case.permission, field: value}
    write(freeze_case.root / "manifests/ALLOCATION_PERMISSION.json", permission)
    with pytest.raises(PermissionError):
        create(freeze_case)


def test_permission_receipt_is_in_frozen_input_identity(freeze_case):
    result = create(freeze_case)
    key = "manifests/ALLOCATION_PERMISSION.json"
    assert result["input_hashes"][key] == sha256_file(freeze_case.root / key)


def test_frozen_qa_tamper_fails_full_validator(freeze_case):
    create(freeze_case)
    name = freeze_case.report["qa"]["original"]["contact_sheets"][0]["path"]
    (freeze_case.root / name).write_bytes(b"post-freeze tamper")
    with pytest.raises(PermissionError, match="Frozen input identity"):
        verify_execution(PLAN_PATH, freeze_case.root, full_hashes=True)


def test_invalid_commit_and_input_symlink_rejected(freeze_case):
    with pytest.raises(ValueError, match="committed code"):
        freeze.freeze_run(PLAN_PATH, freeze_case.root, freeze_case.history_root, code_commit="HEAD")
    (freeze_case.root / "data/link.json").symlink_to(
        freeze_case.root / "data/history_exclusions.json"
    )
    with pytest.raises(PermissionError, match="symlinks"):
        create(freeze_case)


@pytest.fixture
def inspect_case(tmp_path, monkeypatch):
    import torch

    plan = copy.deepcopy(json.loads(PLAN_PATH.read_text()))
    model_path = tmp_path / "snapshot"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}")
    files = [{"name": "config.json", "bytes": 2, "sha256": sha256_file(model_path / "config.json")}]
    plan["model"]["model_weights_hash"] = object_hash({f["name"]: f["sha256"] for f in files})
    identity = {**plan["model"], "model_path": str(model_path), "model_files": files}
    historical_root = tmp_path / "history"
    write(historical_root / "manifests/PRE_INFERENCE_FREEZE.json", identity)
    env = json.loads(
        (PLAN_PATH.parent.parent / "evidence/E06_PROCESSOR_ENVIRONMENT_LOCK.json").read_text()
    )
    calls = []

    def processor_only(path):
        calls.append(path)
        return SimpleNamespace(identity=copy.deepcopy(identity))

    monkeypatch.setattr(freeze, "load_plan", lambda _: plan)
    monkeypatch.setattr(freeze.importlib.metadata, "version", lambda name: env["packages"][name])
    monkeypatch.setattr(freeze.platform, "python_version", lambda: env["python"])
    monkeypatch.setattr(freeze.QwenRuntime, "processor_only", processor_only)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    return SimpleNamespace(
        plan=plan, history_root=historical_root, identity=identity, env=env, calls=calls
    )


def test_cpu_model_inspection_without_weight_loading(inspect_case):
    result = freeze.inspect_model(PLAN_PATH, inspect_case.history_root)
    assert result["status"] == "PASS"
    assert result["model_weights_loaded"] is False and result["model_calls"] == 0
    assert result["cuda_context_initialized"] is False
    assert len(inspect_case.calls) == 1


def test_historical_model_conflict_stops_before_processor(inspect_case):
    changed = {**inspect_case.identity, "processor_hash": "other-processor"}
    write(inspect_case.history_root / "manifests/PRE_INFERENCE_FREEZE.json", changed)
    with pytest.raises(PermissionError, match="Historical model identity conflict"):
        freeze.inspect_model(PLAN_PATH, inspect_case.history_root)
    assert not inspect_case.calls


def test_environment_package_conflict_stops_before_processor(inspect_case, monkeypatch):
    monkeypatch.setattr(freeze.importlib.metadata, "version", lambda _: "unexpected-new-version")
    with pytest.raises(PermissionError, match="Pinned package changed"):
        freeze.inspect_model(PLAN_PATH, inspect_case.history_root)
    assert not inspect_case.calls


def test_python_conflict_stops_before_processor(inspect_case, monkeypatch):
    monkeypatch.setattr(freeze.platform, "python_version", lambda: "3.99.0")
    with pytest.raises(PermissionError, match="Pinned Python version changed"):
        freeze.inspect_model(PLAN_PATH, inspect_case.history_root)
    assert not inspect_case.calls


@pytest.fixture
def freeze_cli():
    spec = importlib.util.spec_from_file_location(
        "mm_dev_freeze_cli_tests", REPO / "scripts/mm_dev/validate_freeze.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mode", ["--create", "--inspect-model"])
def test_cli_rejects_missing_identity_arguments(freeze_cli, monkeypatch, tmp_path, mode):
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_freeze.py", "--plan", str(PLAN_PATH), "--run-root", str(tmp_path), mode],
    )
    with pytest.raises(SystemExit) as exc:
        freeze_cli.main()
    assert exc.value.code == 2


def test_cli_default_validation_requires_all_input_hashes(freeze_cli, monkeypatch, tmp_path):
    calls = []

    def check(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "FROZEN", "plan_id": "test"}

    monkeypatch.setattr(freeze_cli, "verify_execution", check)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_freeze.py",
            "--plan",
            str(PLAN_PATH),
            "--run-root",
            str(tmp_path),
            "--require-engine",
        ],
    )
    freeze_cli.main()
    assert calls[0][1] == {"require_engine": True, "full_hashes": True}
