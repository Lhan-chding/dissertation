"""Assemble evidence without manufacturing missing model/optimizer rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import file_hash, source_commit, write_json

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


def build_report(run_root, out):
    root, out = Path(run_root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    phases = {
        path.parent.name: json.loads(path.read_text())
        for path in sorted(root.glob("*/status.json"))
    }
    all_phases = {f"P{i}": phases.get(f"P{i}", {"status": "NOT_RUN"}) for i in range(10)}
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
    lines = ["# SSVC 本地实现交接", "", measurement_state, ""]
    lines.extend(
        f"- {phase}: `{row['status']}`; {json.dumps(row.get('details', {}), ensure_ascii=False)}"
        for phase, row in all_phases.items()
    )
    for question in QUESTIONS:
        lines.extend(["", f"**{question}**", ""])
        if question == QUESTIONS[-1]:
            lines.append(
                "先运行 P1 兼容性与显存 smoke，返回完整状态和失败日志；未通过前不进入正式训练。"
            )
        else:
            lines.append(
                "本次没有执行对应的真实模型测量，因此尚无新计数或因果结论。历史数据仅见 P0 审计。"
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
