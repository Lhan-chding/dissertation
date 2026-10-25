"""One CPU preparation pass; neither loads model weights nor starts training."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

from sr_f4.common import now, read, rows, write, write_rows
from srf5_ref.schedule import build_schedule


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def freeze_retention(root):
    """Preserve the established task/source stratification before new outputs."""
    from sr_f4.c_pilot_data import (
        CHARTQA_SEED,
        RETENTION_SEED,
        select_chartqa,
        select_retention,
    )

    source = Path("/projects/_ssd/varunssd/louis-ssvc/sr_f4_consolidated_20261022")
    old = Path("/projects/_ssd/varunssd/louis-ssvc/sr_f3_ce_protocol_fix_20261016/selected_v2")
    legacy = select_retention(rows(source / "DATA/TASKS.jsonl"))
    chartqa = select_chartqa(rows(old / "external/chartqa/TASKS_AUDIT_ONLY.jsonl"))
    if len(legacy) != 128 or len(chartqa) != 250:
        raise ValueError("Retention subset count differs")
    write_rows(root / "RETENTION/LEGACY_TASKS.jsonl", legacy)
    write_rows(root / "RETENTION/CHARTQA_TASKS.jsonl", chartqa)
    for name, selected, manifest in (
        ("LEGACY", legacy, source / "DATA/IMAGE_MANIFEST.jsonl"),
        ("CHARTQA", chartqa, source / "DATA/CHARTQA_IMAGES.jsonl"),
    ):
        wanted = {r["qid"] for r in selected}
        routes = [r for r in rows(manifest) if r["qid"] in wanted]
        if len(routes) != len(wanted) or {r["qid"] for r in routes} != wanted:
            raise ValueError("Retention image routes missing or duplicated")
        write_rows(root / f"RETENTION/{name}_SOURCE_ROUTES.jsonl", routes)
    receipt = dict(
        status="FROZEN_BEFORE_NEW_MODEL_OUTPUTS",
        source_root=str(source),
        legacy_source=str(source / "DATA/TASKS.jsonl"),
        chartqa_source=str(old / "external/chartqa/TASKS_AUDIT_ONLY.jsonl"),
        legacy_seed=RETENTION_SEED,
        chartqa_seed=CHARTQA_SEED,
        legacy_questions=128,
        chartqa_questions=250,
        legacy_selection="two roots per family/variant; both chart styles retained",
        chartqa_selection="inherited sorted-ID seeded sample from fixed source",
        legacy_task_digest=digest(legacy),
        chartqa_task_digest=digest(chartqa),
        all_48_formal_models_share_same_subsets=True,
        model_outputs_used=False,
        image_transfer_status="source routes frozen; stage locally before evaluation",
    )
    write(root / "RETENTION/FROZEN.json", receipt)
    return receipt


def prepare(root, config_file, revision):
    from .data import audit_historical_worlds, build

    root = Path(root)
    config = read(config_file)
    if (root / "CONFIG.json").exists():
        if read(root / "CONFIG.json") != config:
            raise ValueError("Existing configuration differs; preserve original root")
        if (root / "CPU_READY.json").exists():
            return read(root / "CPU_READY.json")
        raise ValueError("Partial preparation requires diagnosis, not blind rebuild")
    root.mkdir(parents=True, exist_ok=True)
    write(root / "CONFIG.json", config)
    identities = {}
    model_path = Path(config["model"]["model_path"])
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(model_path / "config.json")
    for parent, value in config["model"]["parents"].items():
        adapter = Path(value)
        metadata = read(adapter / "adapter_config.json")
        tensors = adapter / "adapter_model.safetensors"
        identities[parent] = dict(
            path=value,
            config=metadata,
            tensor_bytes=tensors.stat().st_size,
            tensor_mtime_ns=tensors.stat().st_mtime_ns,
        )
    environment = dict(
        at=now(),
        python=platform.python_version(),
        platform=platform.platform(),
        code_revision=revision,
        snapshot=config["model"]["snapshot"],
        parent_adapters=identities,
        dependencies={},
    )
    for package in ("torch", "transformers", "peft", "Pillow", "numpy", "scipy"):
        environment["dependencies"][package] = importlib.metadata.version(package)
    write(root / "ENVIRONMENT_LOCK.json", environment)
    freeze_retention(root)
    # This is the only full construction/oracle/split audit in preparation.
    audit = build(root, config)
    historical = audit_historical_worlds(
        root,
        [
            "/projects/_ssd/varunssd/louis-ssvc/sr_f4_c_local_pilot_20261024/DATA/TASKS.jsonl",
            "tests/sr_f5/collision_J1_SFAIL.json",
        ],
    )
    tests = [
        "tests/sr_f5",
        "tests/sr_f4/test_data.py",
        "tests/sr_f3_ce/test_contract.py",
        "tests/sr_f3_ce/test_protocol_revision.py",
    ]
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--import-mode=importlib", "-q", *tests],
        capture_output=True,
        text=True,
    )
    (root / "CPU_TESTS.log").write_text(result.stdout + result.stderr)
    receipt = dict(
        at=now(),
        status="PASS" if result.returncode == 0 else "FAIL",
        returncode=result.returncode,
        test_paths=tests,
        model_calls=0,
        code_revision=revision,
        scope="CPU only; no GPU or scientific claim",
    )
    write(root / "CPU_TESTS.json", receipt)
    if result.returncode:
        raise RuntimeError("CPU checks failed; see CPU_TESTS.log")
    calibration = dict(
        version=1,
        purpose="G2_DEV_CAL_NO_UPDATES",
        config_digest=digest(config),
        code_revision=revision,
        parents=config["parents"],
        split="dev",
        questions_per_parent=192,
        draws=8,
        expected_responses=4608,
        question_ids=[r["qid"] for r in rows(root / "DATA/dev/MODEL_INPUTS.jsonl")],
        generation={
            k: config["training"][k] for k in ("temperature", "top_p", "top_k", "max_new_tokens")
        },
        review=dict(
            reviewer_type="program",
            unique_per_parent_family=20,
            parser="sr_f5.measurement",
            no_human_review_claim=True,
        ),
        test_outputs_generated=False,
        optimization_allowed=False,
        master_seed=config["master_seed"],
    )
    write(root / "G2_SPEC_FROZEN.json", calibration)
    train = list(rows(root / "DATA/train/TASKS_PRIVATE.jsonl"))
    pilot = list(rows(root / "DATA/pilot/TASKS_PRIVATE.jsonl"))
    matrix = []
    for parent in config["parents"]:
        for repeat in config["repeats"]:
            schedule = build_schedule(train, config["master_seed"], parent, repeat)
            write(root / f"SCHEDULES/{parent}-r{repeat}.json", schedule)
            for arm in config["arms"]:
                matrix.append(
                    dict(
                        parent=parent,
                        repeat=repeat,
                        arm=arm,
                        updates=72,
                        answers=9216,
                        plan_digest=digest(schedule),
                        status="NOT_AUTHORIZED_UNTIL_G0_G4_PASS",
                    )
                )
        write(
            root / f"SCHEDULES/pilot-{parent}.json",
            build_schedule(pilot, config["master_seed"], parent, 0, 8),
        )
    write(root / "RUN_MATRIX.json", dict(status="PROSPECTIVE_NOT_LAUNCHED", paths=matrix))
    for stage, remaining in {
        "G0": ["actual loaded trainable parameter and tokenizer identity"],
        "G1": ["actual processor geometry", "actual tokenizer span fixtures", "recovery parity"],
        "G2": ["4608 actual parent model responses and measurements"],
        "G3": ["15 pilot paths and actual numerical/recovery gates"],
        "G4": ["previous gates and final source/data/evaluation freeze"],
    }.items():
        write(
            root / f"GATES/{stage}.json",
            dict(status="NOT_RUN", remaining=remaining, cpu_receipt="CPU_TESTS.json"),
        )
    ready = dict(
        at=now(),
        status="CPU_READY_FOR_G2_PREFLIGHT",
        code_revision=revision,
        data_audit=audit,
        historical_audit=historical,
        model_calls=0,
        formal_training_started=False,
        no_full_base_model_hash=True,
    )
    write(root / "CPU_READY.json", ready)
    return ready


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/sr_f5/experiment.json"))
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.config, args.revision), ensure_ascii=False))
