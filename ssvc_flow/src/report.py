"""Assemble evidence without manufacturing missing model/optimizer rows."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from .core import OFFICIAL_FROZEN_MODELS, canonical_hash, file_hash, source_commit, write_json

TABLES = [
    "metrics_by_scene.parquet",
    "metrics_by_group.csv",
    "training_steps.parquet",
    "paired_effects.csv",
    "flow_forks.parquet",
]
QUESTIONS = [
    "原来的 3B 问题是什么？",
    "9B 的初始支持改变了什么？",
    "同一辅助奖励在 9B 上造成什么变化？",
    "变化来自哪里？",
    "SSVC 当前有无必要且有无可测空间？",
    "下一阶段应只补哪一个关键缺口？",
]


FROZEN_IDENTITY_FILES = (
    "model_audit.json",
    "identity.json",
    "runtime_lock.json",
    "data_manifest.json",
)


def _validate_frozen_identity(directory, details):
    audit, identity, runtime, data = [
        json.loads((directory / name).read_text()) for name in FROZEN_IDENTITY_FILES
    ]
    metrics = json.loads((directory / "metrics.json").read_text())
    tables = json.loads((directory / "table_status.json").read_text())
    for name in ("metrics_by_scene.parquet", "metrics_by_group.csv"):
        if tables[name].get("execution_kind") != "REAL_CUDA_MODEL":
            raise ValueError("hashed table export carries non-real execution identity")
    for key in ("phase", "execution_kind", "model_key", "model_id", "model_revision", "track"):
        if details.get(key) != audit.get(key) or audit.get(key) != identity.get(key):
            raise ValueError(f"frozen scientific identity mismatch: {key}")
    if (
        audit.get("phase") != "P3"
        or audit.get("execution_kind") != "REAL_CUDA_MODEL"
        or audit.get("passed") is not True
    ):
        raise ValueError("hashed audit is not passed real P3 evidence")
    if runtime.get("identity") != identity:
        raise ValueError("runtime and stored identity mismatch")
    model_key, track = audit["model_key"], audit["track"]
    if model_key not in OFFICIAL_FROZEN_MODELS or track not in ("N", "L"):
        raise ValueError("unknown frozen model/track identity")
    model_id, layers = OFFICIAL_FROZEN_MODELS[model_key]
    revision = audit["model_revision"]
    if audit["model_id"] != model_id or not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError("official model or immutable revision mismatch")
    spec = runtime["model_spec"]
    if (
        spec.get("id") != model_id
        or spec.get("revision") != revision
        or spec.get("expected_layers") != layers
        or canonical_hash(spec) != identity["model_hash"]
    ):
        raise ValueError("runtime model specification/hash mismatch")
    for label, model_audit in (
        ("runtime", runtime["model_audit"]),
        ("P1", runtime["P1_evidence"]["model_audit"]),
    ):
        if model_audit.get("model_id") != model_id or model_audit.get("model_revision") != revision:
            raise ValueError(f"{label} model identity differs from frozen audit")
        if model_audit.get("execution_kind", "REAL_CUDA_MODEL") != "REAL_CUDA_MODEL":
            raise ValueError(f"{label} carries non-real execution identity")
    if (
        runtime.get("optimizer_updates") != 0
        or audit.get("optimizer_updates") != 0
        or audit.get("backward_calls") != 0
    ):
        raise ValueError("frozen evidence contains training operations")
    frozen_hash = runtime.get("frozen_parameter_hash")
    if (
        not frozen_hash
        or audit.get("frozen_parameter_hash_before") != frozen_hash
        or audit.get("frozen_parameter_hash_after") != frozen_hash
    ):
        raise ValueError("frozen parameter hashes are inconsistent")
    if (
        data.get("track") != track
        or metrics.get("track") != track
        or data.get("data_hash") != identity.get("data_hash")
    ):
        raise ValueError("data/metrics scientific identity mismatch")
    count, scene_count = data["prompt_count"], data["scene_count"]
    if type(count) is not int or type(scene_count) is not int or count < 1 or scene_count < 1:
        raise ValueError("invalid data manifest counts")
    if (track == "N" and (count != 288 or scene_count != 144)) or (
        track == "L" and count != scene_count
    ):
        raise ValueError("incomplete formal frozen panel")
    if metrics.get("expected_rollouts_per_prompt") != 16:
        raise ValueError("frozen panel requires 16 samples per prompt")
    expected_scenes = None
    prompt_ids = None
    for block, repetitions in ((metrics, 16), (metrics["greedy"], 1)):
        prompts = block["by_prompt"]
        scenes = {row["base_scene_id"] for row in prompts.values()}
        if len(prompts) != count or any(
            row.get("rollout_count") != repetitions for row in prompts.values()
        ):
            raise ValueError("per-prompt frozen counts mismatch")
        if prompt_ids is not None and (set(prompts) != prompt_ids or scenes != expected_scenes):
            raise ValueError("greedy and sampled panel identities mismatch")
        prompt_ids, expected_scenes = set(prompts), scenes
        overall = block["overall"]
        if (
            overall.get("prompt_count") != count
            or overall.get("rollout_count") != count * repetitions
            or overall.get("independent_scene_count") != len(scenes)
        ):
            raise ValueError("aggregate frozen panel counts mismatch")
        if track == "N" and len(scenes) != scene_count:
            raise ValueError("N panel independent scene count mismatch")
    if (
        audit.get("sampled_rollout_count") != count * 16
        or audit.get("greedy_rollout_count") != count
        or audit.get("raw_sample_count") != count * 17
    ):
        raise ValueError("audit and metric generation counts mismatch")
    length = 64 if track == "N" else 48
    if runtime.get("max_new_tokens") != length:
        raise ValueError("frozen token cap differs from protocol")
    protocol = runtime["generation_protocol"]
    required_protocol = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "num_beams": 1,
        "do_sample": True,
        "max_new_tokens": length,
    }
    if protocol != required_protocol:
        raise ValueError("frozen generation protocol differs from untransformed softmax")
    return {
        "data_hash": identity["data_hash"],
        "max_new_tokens": length,
        "generation_protocol": protocol,
        "expected_rollouts_per_prompt": 16,
    }


def _import_frozen(root, out):
    """Only import complete, hash-checked real subtask exports."""
    from .frozen_report import EXPORT_FILES

    observations, imported = [], []
    candidates = sorted((root / "P3").rglob("status.json"))
    nested = [p for p in candidates if p.parent != root / "P3"]
    if nested:
        candidates = nested
    for path in candidates:
        document = json.loads(path.read_text())
        details = document.get("details", {})
        relative = path.parent.relative_to(root)
        item = {"path": str(relative), "status": document.get("status"), **details}
        item["import_status"] = "NOT_IMPORTED"
        observations.append(item)
        if (
            document.get("status") != "PASS"
            or details.get("execution_kind") != "REAL_CUDA_MODEL"
            or details.get("passed") is not True
        ):
            continue
        try:
            manifest = json.loads((path.parent / "manifest.json").read_text())
            recorded = set()
            for entry in manifest["files"]:
                artifact = (path.parent / entry["path"]).resolve()
                if not artifact.is_relative_to(path.parent.resolve()):
                    raise ValueError("manifest artifact escapes its subtask directory")
                if file_hash(artifact) != entry["sha256"]:
                    raise ValueError(f"manifest hash mismatch: {entry['path']}")
                recorded.add(artifact)
            for name in (*EXPORT_FILES, *FROZEN_IDENTITY_FILES):
                if (path.parent / name).resolve() not in recorded:
                    raise ValueError(f"export absent from manifest: {name}")
            comparison_identity = _validate_frozen_identity(path.parent, details)
            import pyarrow.parquet as pq

            pq.read_table(path.parent / "metrics_by_scene.parquet")
            json.loads((path.parent / "metrics.json").read_text())
            destination = out / relative
            destination.mkdir(parents=True, exist_ok=True)
            if destination.resolve() == path.parent.resolve():
                raise ValueError("report output must differ from its evidence source")
            for name in EXPORT_FILES:
                shutil.copyfile(path.parent / name, destination / name)
            write_json(
                destination / "import_provenance.json",
                {
                    "source_directory": str(relative),
                    "source_manifest_sha256": file_hash(path.parent / "manifest.json"),
                    "scope": "summary exports only; raw rows/checkpoints remain in source run",
                    "files": [
                        {"path": name, "sha256": file_hash(destination / name)}
                        for name in EXPORT_FILES
                    ],
                },
            )
            item["import_status"] = "IMPORTED"
            item["comparison_identity"] = comparison_identity
            imported.append(item)
        except (ValueError, KeyError, TypeError, OSError, ImportError) as error:
            item["import_status"] = "INVALID_EVIDENCE"
            item["reason"] = str(error)
    required = {
        (model, track)
        for model in ("qwen25vl_3b", "qwen35_9b", "qwen25vl_7b")
        for track in ("L", "N")
    }
    completed = {(item.get("model_key"), item.get("track")) for item in imported}
    incomparable = []
    for track in ("L", "N"):
        signatures = {
            canonical_hash(item["comparison_identity"])
            for item in imported
            if item.get("track") == track
        }
        if len(signatures) > 1:
            incomparable.append(f"{track}: data or decoding protocol differs across subtasks")
    phase = {
        "status": "PASS" if required <= completed and not incomparable else "PARTIAL",
        "details": {
            "subtasks": observations,
            "incomparable_subtasks": incomparable,
            "missing_required_subtasks": [
                f"{model}/{track}" for model, track in sorted(required - completed)
            ],
            "scope": "three required models, each with L and N frozen observations",
        },
    }
    return phase if observations else None, imported


def build_report(run_root, out):
    root, out = Path(run_root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    phases = {
        path.parent.name: json.loads(path.read_text())
        for path in sorted(root.glob("*/status.json"))
    }
    all_phases = {f"P{i}": phases.get(f"P{i}", {"status": "NOT_RUN"}) for i in range(10)}
    frozen_phase, imported_frozen = _import_frozen(root, out)
    if frozen_phase is not None:
        all_phases["P3"] = frozen_phase
    write_json(out / "phase_status.json", all_phases)
    write_json(
        out / "run_manifest.json",
        {
            "source_commit": source_commit(),
            "files": [
                {"path": str(path.relative_to(root)), "sha256": file_hash(path)}
                for path in sorted(root.rglob("manifest.json"))
            ],
        },
    )
    table_status = {
        name: {"status": "NOT_RUN", "reason": "No real-model stage observations imported"}
        for name in TABLES
    }
    if imported_frozen:
        for name in ("metrics_by_scene.parquet", "metrics_by_group.csv"):
            table_status[name] = {
                "status": "AVAILABLE_SUBTASKS",
                "files": [str(Path(item["path"]) / name) for item in imported_frozen],
                "reason": "separate model/track exports; not pooled across models",
            }
    table_status["frozen_support_comparison.csv"] = {
        "status": "NOT_RUN",
        "reason": "No verified frozen subtask metrics imported",
    }
    if imported_frozen:
        from .frozen_comparison import write_support_comparison

        try:
            comparison_files = write_support_comparison(
                out,
                [
                    {
                        **item,
                        "metrics": json.loads((out / item["path"] / "metrics.json").read_text()),
                    }
                    for item in imported_frozen
                ],
            )
            table_status["frozen_support_comparison.csv"] = {
                "status": "AUXILIARY_DESCRIPTIVE",
                "files": [str(path.relative_to(out)) for path in comparison_files],
                "reason": "fixed model-specific strata; overall retained; not causal effects",
            }
        except (ValueError, KeyError, TypeError, OSError) as error:
            table_status["frozen_support_comparison.csv"] = {
                "status": "BLOCKED",
                "reason": str(error),
            }
    write_json(out / "table_status.json", table_status)
    model_audit = root / "P1/model_audit.json"
    write_json(
        out / "model_audit.json",
        json.loads(model_audit.read_text())
        if model_audit.exists()
        else {"status": "PENDING_GPU", "measurements": None},
    )
    data_manifest = root / "P0/data_manifest.json"
    write_json(
        out / "data_manifest.json",
        json.loads(data_manifest.read_text())
        if data_manifest.exists()
        else {"status": "SEE_GENERATED_DATA_MANIFEST"},
    )
    failures = [p.read_text() for p in sorted(root.glob("*/failures.jsonl"))]
    (out / "failures.jsonl").write_text("".join(failures), encoding="utf-8")
    measurement_state = (
        "P1 兼容性测量已记录；P3 科学概率测量尚未运行。CPU 数学结果不能作为 Qwen 实验结果。"
        if model_audit.exists()
        else "真实模型结果尚待 NTU GPU 测试；CPU 数学结果不能作为 Qwen 实验结果。"
    )
    if imported_frozen:
        measurement_state = (
            "P3 冻结评估子任务已记录；完整阶段状态见 phase_status.json。"
            "冻结观测不证明训练收益、奖励安全性或模型规模因果关系。"
        )
    lines = ["# SSVC 本地实现交接", "", measurement_state, ""]
    lines.extend(
        f"- {phase}: `{row['status']}`; {json.dumps(row.get('details', {}), ensure_ascii=False)}"
        for phase, row in all_phases.items()
    )
    if table_status["frozen_support_comparison.csv"]["status"] == "AUXILIARY_DESCRIPTIVE":
        lines.extend(
            [
                "",
                "辅助支持区间比较见 frozen_support_comparison_zh.md；"
                "全部 overall 保留，不作为规模因果或训练收益结论。",
            ]
        )
    for question in QUESTIONS:
        lines.extend(["", f"**{question}**", ""])
        if question == QUESTIONS[-1]:
            if imported_frozen:
                lines.append(
                    "核对并补齐 P3 所需模型/轨道子任务及先前验收缺口，再按实测预算评审后续短训练。"
                )
            elif model_audit.exists() and all_phases["P1"].get("status") == "PASS":
                lines.append(
                    "核对 P0/P2 遗留验收和 P1 实测预算，进入 P3 冻结评估；尚无训练效果结论。"
                )
            else:
                lines.append(
                    "先运行 P1 兼容性与显存 smoke，返回完整状态和失败日志；未通过前不进入正式训练。"
                )
        else:
            lines.append(
                "已导入的冻结计数见各 P3 子任务 results_report_zh.md；"
                "未执行训练干预，尚无因果结论。"
                if imported_frozen
                else "本次没有执行对应的真实模型测量，因此尚无新计数或因果结论。"
                "历史数据仅见 P0 审计。"
            )
    lines.extend(
        [
            "",
            "未运行表格的状态在 table_status.json；不生成虚假的数值行或伪装为 Parquet 的文本文件。",
            "",
        ]
    )
    (out / "results_report_zh.md").write_text("\n".join(lines), encoding="utf-8")
    return all_phases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)
    build_report(args.run_root, args.out)


if __name__ == "__main__":
    main()
