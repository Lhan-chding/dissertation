"""Freeze the package and local cluster bindings before scientific sampling."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import sys
from pathlib import Path

from .common import now, read, rows, write, write_rows


def registered_tasks():
    tasks = []
    for source in ("B0", "B2_DEV", "C_TRAIN_PROBE"):
        for block in range(3):
            tasks.append(
                dict(
                    task_id=f"{source}-b{block}",
                    kind="probe",
                    source=source,
                    block=block,
                    dependencies=[],
                    resume_supported=True,
                )
            )
    train = []
    for block in range(3):
        for arm in ("J_UNIFORM_NEW_REPEAT", "LZ"):
            key = f"A-b{block}-{arm}"
            train.append(key)
            tasks.append(
                dict(
                    task_id=key,
                    kind="train",
                    track="A",
                    block=block,
                    arm=arm,
                    dependencies=[f"B0-b{block}"],
                    resume_supported=True,
                )
            )
    probes = [t["task_id"] for t in tasks if t["kind"] == "probe"]
    tasks.append(
        dict(task_id="B3-replay", kind="replay", dependencies=probes, resume_supported=True)
    )
    evaluations = []
    for block in range(3):
        for arm in ("PARENT", "J_UNIFORM_NEW_REPEAT", "LZ"):
            for shard in range(2):
                key = f"E-b{block}-{arm}-s{shard}"
                evaluations.append(key)
                tasks.append(
                    dict(
                        task_id=key,
                        kind="eval",
                        phase="regression",
                        block=block,
                        arm=arm,
                        shard=shard,
                        shards=2,
                        dependencies=[] if arm == "PARENT" else [f"A-b{block}-{arm}"],
                        resume_supported=True,
                    )
                )
                confirm_key = f"EC-b{block}-{arm}-s{shard}"
                evaluations.append(confirm_key)
                tasks.append(
                    dict(
                        task_id=confirm_key,
                        kind="eval",
                        phase="confirm",
                        evaluation_task_id=key,
                        block=block,
                        arm=arm,
                        shard=shard,
                        shards=2,
                        dependencies=[*train, "B3-replay"],
                        resume_supported=True,
                    )
                )
    tasks.append(
        dict(task_id="REPORT", kind="report", dependencies=evaluations, resume_supported=True)
    )
    tasks.append(
        dict(task_id="PACKAGE", kind="package", dependencies=["REPORT"], resume_supported=True)
    )
    return tasks


def prepare(root, old_root, code):
    from .data import build_dataset

    root, old_root, code = map(Path, (root, old_root, code))
    root.mkdir(parents=True, exist_ok=True)
    if (root / "FREEZE_A_F1.json").exists():
        return read(root / "CONFIG.json")
    prior = read(old_root / "prior/MODEL_ENVIRONMENT_IDENTITY.json")
    audit = read(old_root / "parents/PARENT_AUDIT.json")
    if audit["status"] != "PASS":
        raise ValueError("Previously registered parents are unavailable")
    parents = {
        str(i): str(old_root / p["adapter_path"])
        for i, p in enumerate(sorted(audit["parents"], key=lambda r: r["parent_id"]))
    }
    if len(parents) != 3 or any(
        not (Path(p) / "adapter_model.safetensors").is_file() for p in parents.values()
    ):
        raise FileNotFoundError("All three inherited parent adapters are required")
    config = dict(
        experiment_id="SRF4-CONSOLIDATED-20261022",
        old_root=str(old_root),
        model_path=prior["model_path"],
        parents=parents,
        python=sys.executable,
        qos="soujanya-poria-startfund-2026-03",
        account="rose",
        partition="cluster02",
        gpu_limit=8,
        max_submit_jobs=5,
        job_name_prefix="LH-SRF4",
        code_root=str(code),
        c_gate_prerequisites=dict(sampler_tests_passed=True, hard_fixture_false_positive=0),
    )
    write(root / "CONFIG.json", config)
    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "peft", "numpy", "Pillow")
    }
    write(root / "ENVIRONMENT.json", dict(python=sys.version, versions=versions, at=now()))
    write(
        root / "INPUT_MANIFEST.json",
        dict(
            package="SRF4_CODEX_EXECUTION_PACKAGE_20261022.zip",
            specification=str(code / "docs/sr_f4/spec"),
            old_root=str(old_root),
            parents=parents,
            model_snapshot=prior["model_path"],
            existing_registered_model_id=prior.get("model_weights_hash"),
            verification="registered identities reused; no full model/package/output hash sweeps",
        ),
    )
    write(
        root / "FREEZE_F0.json",
        dict(
            at=now(),
            status="FROZEN",
            design="SRF4-CONSOLIDATED-20261022-v1.0.0",
            A=dict(paths=6, updates_per_path=64, answers_per_update=128),
            C=dict(paths=24, conditional=True),
            response_budgets=dict(
                B0=3840,
                B2_DEV=4800,
                C_TRAIN_PROBE=1152,
                A_TRAIN=49152,
                final_evaluations=106760,
                B_CONFIRM_PARENT_included=1344,
                CHARTQA_b0_training_arms_included=5000,
                total_with_C_blocked=165704,
            ),
            confirm_outputs_seen=False,
            betas=[0.25],
            gpu_hours_stop=None,
        ),
    )
    if not (root / "DATA/DATA_READY.json").exists():
        build_dataset(old_root / "plan", root / "DATA", old_image_root=old_root / "data")
    # Reuse external benchmark images by hardlink; never copy gold into a prompt.
    external_root = old_root / "external/chartqa"
    external_freeze = read(external_root / "FREEZE.json")
    image_root = Path(external_freeze.get("image_root", old_root))
    external_inputs = list(rows(external_root / "MODEL_INPUTS.jsonl"))
    routes = []
    for item in external_inputs:
        relative = "external/chartqa/" + Path(item["image_file"]).name
        destination = root / "DATA" / relative
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = image_root / item["image_file"]
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
        routes.append(dict(item, image_file=relative, pool="CHARTQA"))
    write_rows(root / "DATA/CHARTQA_IMAGES.jsonl", routes)
    write(root / "TASKS.json", registered_tasks())
    write(
        root / "C_GATE_REPORT.json",
        dict(
            decision="BLOCK_C",
            status="MEASUREMENT_NOT_READY",
            reasons=[
                "No real human blind audit has been supplied; probes/replay are still pending"
            ],
            human_audit="MISSING",
            AI_labels_are_human=False,
        ),
    )
    write(root / "FREEZE_C_F1.json", dict(status="BLOCK_C", human_audit="MISSING", C_updates=0))
    write(
        root / "C_RUN_REGISTRY.json",
        [
            dict(
                block=b,
                repeat=r,
                arm=a,
                status="SKIPPED_BY_PREREG_GATE",
                reason="MISSING_REAL_HUMAN_BLIND_AUDIT",
            )
            for b in range(3)
            for r in (1, 2)
            for a in ("J", "EVID", "SELECT", "SELECT_SHUF")
        ],
    )
    checks = read(code / "docs/sr_f4/VALIDATION.json")
    if checks.get("status") != "PASS":
        raise ValueError("Targeted production adapter checks did not pass")
    write(
        root / "FREEZE_A_F1.json",
        dict(
            at=now(),
            status="FROZEN_BEFORE_A_UPDATE",
            config=config,
            data=read(root / "DATA/DATA_READY.json"),
            targeted_validation=checks,
            native_contract=read(code / "docs/sr_f4/configs/native_contract.resolved.json"),
            sampler="srf4_ref.sampling.plan_step; weighted feasible tail; uniform world then chart",
            rollout_rng="independent per-draw torch.Generator; G32 uses four G8 calls",
            overrides=[
                "User requested no redundant hashes/smoke runs; no extra GPU ENGINE sampling",
                "QoS grants 8 PRO6000, MaxSubmitPU5; four two-GPU jobs with independent refill",
            ],
            no_confirm_model_calls=True,
        ),
    )
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.old_root, args.code), indent=2))
