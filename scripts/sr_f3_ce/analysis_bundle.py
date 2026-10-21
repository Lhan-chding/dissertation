# ruff: noqa: RUF001
"""Explicit-inventory, streaming results archive; no inference or tensor access."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import stat
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath

CHUNK = 1 << 20
SCHEMA = "sr_f3_ce_local_analysis_bundle_v1"
EXPECTED = dict(
    training_groups=12288,
    training_records=98304,
    support=2304,
    monitor=26112,
    engine=2880,
    endpoint=233340,
)
FORBIDDEN = {".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".parquet", ".pyc", ".pyo"}
REPORTS = ("PRIMARY_ANALYSIS.json", "EXTERNAL_ANALYSIS.json", "PREFIX_DIAGNOSTICS.json")
GENERATED = "_analysis_bundle/"
RELEASE_FILES = {
    *REPORTS,
    "ENDPOINT_RAW.jsonl.gz",
    "ENDPOINT_SCORES.jsonl",
    "RELEASE.json",
    "RELEASE_SCORED.json",
    "RAW_RESCORING.json",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def loads(value):
    return json.loads(
        value,
        object_pairs_hook=pairs,
        parse_constant=lambda value: require(False, "Nonfinite JSON"),
    )


def encoded(value):
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2) + "\n"
    ).encode()


def safe(name):
    require(
        isinstance(name, str) and name and "\\" not in name and "\x00" not in name, "Unsafe path"
    )
    p = PurePosixPath(name)
    require(
        not p.is_absolute()
        and not re.match(r"^[A-Za-z]:", name)
        and all(part not in ("", ".", "..") for part in name.split("/")),
        "Unsafe path",
    )
    require(
        p.suffix.lower() not in FORBIDDEN
        and ("checkpoints" not in p.parts or p.suffix.lower() == ".json")
        and "models" not in p.parts
        and "__pycache__" not in p.parts,
        "Forbidden payload",
    )
    return name


def digest(stream):
    h, size = hashlib.sha256(), 0
    for data in iter(lambda: stream.read(CHUNK), b""):
        h.update(data)
        size += len(data)
    return dict(sha256=h.hexdigest(), bytes=size)


def pin(record):
    require(
        isinstance(record, dict)
        and re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", ""))
        and type(record.get("bytes")) is int
        and record["bytes"] >= 0,
        "Invalid hash pin",
    )
    return {k: record[k] for k in ("sha256", "bytes")}


def inventory(manifest):
    files = manifest.get("files")
    require(isinstance(files, dict) and files, "Explicit files required")
    for name, record in files.items():
        safe(name)
        require(not name.startswith(GENERATED), "Reserved generated path")
        pin(record)
        require(isinstance(record.get("category"), str) and record["category"], "Category required")
    for name in REPORTS:
        require("release/" + name in files, "Missing complete analysis: " + name)
    require("release/ENDPOINT_RAW.jsonl.gz" in files, "Missing endpoint raw")
    require("release/RAW_RESCORING.json" in files, "Missing original rescore receipt")
    require("release/MANIFEST.json" in files, "Missing original release manifest")
    return files


def local_file(root, name):
    safe(name)
    path = root
    for component in name.split("/"):
        path = path / component
        require(not path.is_symlink(), "Symlink input forbidden")
    require(path.is_file() and path.resolve().is_relative_to(root), "Missing input")
    return path


def raw_fields(row):
    require(isinstance(row, dict), "Raw record must be object")
    tokens = row.get("tokens")
    require(
        isinstance(tokens, list) and tokens and all(type(t) is int and t >= 0 for t in tokens),
        "Original generated token IDs missing",
    )
    require(
        all(isinstance(row.get(k), str) for k in ("full_text", "raw_text"))
        and isinstance(row.get("model_input", {}).get("prompt"), str),
        "Original full text/prompt missing",
    )
    require(
        row.get("finish_reason") in ("balanced", "final_tag", "eos", "length")
        and type(row.get("max_new_tokens")) is int
        and 0 < len(tokens) <= row["max_new_tokens"],
        "Original termination/budget missing",
    )


def category_of(name, value):
    if name == "release/ENDPOINT_RAW.jsonl.gz":
        return "endpoint_raw"
    return {
        "training_raw": "training_group",
        "monitor_raw": "monitor_group",
        "engine_raw": "engine_group",
    }.get(value, value)


def inspect_payload(package, name, category, counts, indexes):
    if category in ("training_group", "monitor_group", "engine_group"):
        value = loads(package.read(name))
        rows = value.get("records")
        require(isinstance(rows, list) and rows, "Raw group missing records")
        if category == "training_group":
            require(len(rows) == 8, "Each training group must contain eight answers")
            counts["training_groups"] += 1
            counts["training_records"] += len(rows)
        else:
            counts[category.removesuffix("_group")] += len(rows)
        for row in rows:
            raw_fields(row)
        indexes[category].append(dict(path=name, records=len(rows)))
    elif category == "support":
        value = loads(package.read(name))
        tokens = value.get("tokens")
        require(
            isinstance(value.get("target"), dict)
            and isinstance(value["target"].get("text"), str)
            and isinstance(value["target"].get("object"), dict)
            and isinstance(value["target"].get("spans"), dict)
            and isinstance(tokens, list)
            and tokens
            and all(type(t) is int and t >= 0 for t in tokens),
            "Support target/tokens missing",
        )
        for key in ("answer_mask", "work_mask"):
            mask = value.get(key)
            require(
                isinstance(mask, list)
                and len(mask) == len(tokens)
                and all(x in (0, 1) for x in mask),
                "Support mask missing",
            )
        require(isinstance(value.get("mask_audit"), dict), "Support mask audit missing")
        counts["support"] += 1
        indexes[category].append(dict(path=name, records=1))
    elif category == "endpoint_raw":
        require(name == "release/ENDPOINT_RAW.jsonl.gz", "Unexpected endpoint aggregate")
        with package.open(name) as compressed, gzip.GzipFile(fileobj=compressed) as raw:
            for line in io.TextIOWrapper(raw, encoding="utf-8"):
                require(line.strip(), "Blank endpoint raw row")
                raw_fields(loads(line))
                counts["endpoint"] += 1
    elif name.endswith(".gz"):
        with package.open(name) as compressed, gzip.GzipFile(fileobj=compressed) as raw:
            for _ in iter(lambda: raw.read(CHUNK), b""):
                pass


def verify_receipt_links(package, files):
    counts = dict(task_artifact_links=0, panel_group_links=0)
    for name in files:
        if name.startswith("tasks/") and name.endswith("/COMPLETE.json"):
            value = loads(package.read(name))
            artifacts = value.get("artifacts", [])
            require(isinstance(artifacts, list), "Task artifacts must be a list")
            for artifact in artifacts:
                path = artifact.get("path")
                if path in files:
                    require(
                        pin(artifact) == pin(files[path]), "Task artifact binding differs: " + path
                    )
                    counts["task_artifact_links"] += 1
        if name.startswith(("monitor/", "engine/")) and name.endswith("/COVERAGE.json"):
            value = loads(package.read(name))
            groups = value.get("groups")
            if not isinstance(groups, list):
                continue
            seen = set()
            for group in groups:
                gid = group.get("group_id")
                require(
                    isinstance(gid, str) and gid and gid not in seen,
                    "Duplicate/missing panel group",
                )
                seen.add(gid)
                path = str(PurePosixPath(name).parent / "groups" / (gid + ".json"))
                safe(path)
                require(
                    path in files and files[path]["sha256"] == group.get("sha256"),
                    "Panel COVERAGE group binding differs: " + path,
                )
                counts["panel_group_links"] += 1
    return counts


def scheduler_report(snapshot):
    if snapshot is None:
        return "\n## 调度成本\n本包未提供调度快照，不推断资源成本或总墙钟。\n"
    require(
        snapshot.get("returncode") == 0 and isinstance(snapshot.get("stdout"), str),
        "Scheduler snapshot failed",
    )
    rows, gpu_hours = [], 0.0
    zone = timezone(timedelta(hours=8))

    def display(value):
        if value in ("Unknown", "None", "", "N/A"):
            return value
        return (
            datetime.fromisoformat(value).replace(tzinfo=timezone.utc).astimezone(zone).isoformat()
        )

    for line in snapshot["stdout"].splitlines():
        if not line.strip():
            continue
        fields = line.rstrip("|").split("|")
        require(len(fields) == 10, "Scheduler columns differ")
        job, state, exit_code, _submit, start, end, elapsed, tres, node, qos = fields
        require(elapsed.isdigit(), "Scheduler elapsed missing")
        allocation = dict(part.split("=", 1) for part in tres.split(",") if "=" in part)
        # Total gres/gpu takes precedence over its type-specific components; never sum both.
        gpu = int(
            allocation.get(
                "gres/gpu", sum(int(v) for k, v in allocation.items() if k.startswith("gres/gpu:"))
            )
        )
        hours = gpu * int(elapsed) / 3600
        gpu_hours += hours
        rows.append(
            [
                job,
                state,
                exit_code,
                allocation.get("cpu", "0"),
                gpu,
                int(elapsed),
                round(hours, 6),
                display(start),
                display(end),
                node,
                qos,
            ]
        )
    return (
        "\n## 调度成本与技术失败（Asia/Singapore）\n快照UTC时间："
        + str(snapshot.get("at_utc"))
        + "。\n"
        + table(
            [
                "作业",
                "状态",
                "退出码",
                "CPU分配",
                "GPU分配",
                "运行秒",
                "分配GPU小时",
                "启动SGT",
                "结束SGT",
                "节点",
                "QoS",
            ],
            rows,
        )
        + "所列作业截至快照的分配GPU小时合计："
        + str(round(gpu_hours, 6))
        + "。按实际AllocTRES GPU数乘ElapsedRaw计算，不等于GPU kernel时间。"
        + "无GPU分配的作业按CPU作业记录。"
        + "OOM、取消和未启动记录完整列示；PENDING或运行中的恢复作业不是完成交付，"
        + "总墙钟尚不能按其预计时间结算。\n"
    )


def verify(path):
    with zipfile.ZipFile(path) as package:
        names = package.namelist()
        require(len(names) == len(set(names)), "Duplicate ZIP member")
        for info in package.infolist():
            safe(info.filename)
            require(
                not info.is_dir() and not stat.S_ISLNK(info.external_attr >> 16),
                "Nonregular ZIP member",
            )
        manifest_bytes = package.read(GENERATED + "SOURCE_MANIFEST.json")
        manifest = loads(manifest_bytes)
        files = inventory(manifest)
        archive_manifest = loads(package.read(GENERATED + "ARCHIVE_MANIFEST.json"))
        require(archive_manifest.get("schema") == SCHEMA, "Archive schema differs")
        payloads = archive_manifest["files"]
        require(
            set(names) == set(payloads) | {GENERATED + "ARCHIVE_MANIFEST.json"},
            "ZIP inventory differs",
        )
        require(
            set(n for n in names if not n.startswith(GENERATED)) == set(files),
            "Source inventory differs",
        )
        for name, record in payloads.items():
            with package.open(name) as stream:
                require(digest(stream) == pin(record), "ZIP hash/size differs: " + name)
        # Full ZipExtFile reads above check CRC at EOF, without a second testzip scan.
        counts = dict.fromkeys(EXPECTED, 0)
        indexes = {
            key: [] for key in ("training_group", "support", "monitor_group", "engine_group")
        }
        for name, record in files.items():
            require(pin(payloads[name]) == pin(record), "Source binding differs")
            inspect_payload(package, name, category_of(name, record["category"]), counts, indexes)
        require(counts == EXPECTED, "Scientific coverage differs: " + str(counts))
        release_manifest = loads(package.read("release/MANIFEST.json"))
        release_entries = release_manifest["files"]
        require(
            isinstance(release_entries, list)
            and len({r["path"] for r in release_entries}) == len(release_entries),
            "Duplicate release entry",
        )
        require(
            {entry["path"] for entry in release_entries} == RELEASE_FILES,
            "Incomplete original release manifest",
        )
        for entry in release_entries:
            name = "release/" + safe(entry["path"])
            require(
                name in files and pin(entry) == pin(files[name]), "Release manifest binding differs"
            )
        rescore = loads(package.read("release/RAW_RESCORING.json"))
        require(
            rescore.get("status") == "PASS"
            and rescore.get("records") == EXPECTED["endpoint"]
            and rescore.get("mismatches") == []
            and rescore.get("gzip_read_to_eof") is True
            and rescore.get("raw_sha256") == files["release/ENDPOINT_RAW.jsonl.gz"]["sha256"],
            "Original scoring receipt does not bind endpoint raw",
        )
        receipt_links = verify_receipt_links(package, files)
        actual_indexes = loads(package.read(GENERATED + "RAW_INDEX.json"))
        require(actual_indexes == indexes, "Raw index differs")
        report = package.read(GENERATED + "REPORT_ZH.md").decode()
        expected_report = report_text(
            {name: loads(package.read("release/" + name)) for name in REPORTS},
            loads(package.read("delivery_context/SCHEDULER_COST_SNAPSHOT.json"))
            if "delivery_context/SCHEDULER_COST_SNAPSHOT.json" in files
            else None,
        )
        require(report == expected_report, "Fact report differs from complete analysis JSON")
    with Path(path).open("rb") as stream:
        archive_pin = digest(stream)
    return dict(
        schema=SCHEMA,
        status="VERIFIED_LOCAL_ANALYSIS_DATA_BUNDLE",
        archive=archive_pin,
        counts=counts,
        receipt_links=receipt_links,
        source_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        checks=dict.fromkeys(
            (
                "zip_crc",
                "all_member_sha256",
                "gzip_eof",
                "raw_fields",
                "fixed_coverage",
                "source_inventory",
                "complete_analysis_report",
                "original_rescore_receipt_binding",
            ),
            "PASS",
        ),
        server_delivery_gates="NOT_RUN_OR_CLAIMED",
        fresh_scoring="NOT_RUN",
        checkpoint_verification="NOT_RUN",
        inference_or_training_rerun=False,
    )


def cell(value):
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    return text.replace("|", "\\|").replace("\n", "<br>")


def table(headers, rows):
    return (
        "\n"
        + "| "
        + " | ".join(headers)
        + " |\n| "
        + " | ".join(["---"] * len(headers))
        + " |\n"
        + "".join("| " + " | ".join(cell(x) for x in row) + " |\n" for row in rows)
        + "\n"
    )


def leaves(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from leaves(item, prefix + "." + key if prefix else key)
    else:
        yield prefix, value


def report_text(analyses, scheduler=None):
    parts = [
        "# SR-F3-CE 紧凑分析资料包\n",
        "本包保留所选原始文件的原路径与原字节；完整不利结果与无效回答不筛除。"
        "本次仅核验传输、原文结构与固定覆盖，不新增实验或重新评分，"
        "不代表旧服务器交付门禁通过，也不包含模型或检查点。\n",
        "训练 12,288 组 / 98,304 回答；支持 2,304；MONITOR 26,112；ENGINE 2,880；端点 233,340。\n",
        "以下逐项呈现原分析 JSON 的完整结构和值；保留各注册臂、种子、误差分解及不利数据，"
        "不生成新的科学结论。\n",
    ]
    primary = analyses["PRIMARY_ANALYSIS.json"]
    parts.append("\n## 三项主对比（百分点）\n")
    rows = []
    for key in ("C1", "C2", "C3"):
        item = primary["primary"][key]
        blocks = item["per_block_pp"]
        require(len(blocks) == 3, "Primary comparison must retain three blocks")
        ci = item["CI_simultaneous_pp"]
        reading = (
            "正向区间"
            if ci[0] > 0
            else "负向区间，保留"
            if ci[1] < 0
            else "跨零，未确认；不代表等效"
        )
        rows.append(
            [
                key,
                item["arm1"] + "−" + item["arm0"] + "/" + item["scope"],
                item["effect_pp"],
                *blocks,
                item["CI95_pp"],
                ci,
                item["paired_t_sensitivity"]["CI95_pp"],
                reading,
            ]
        )
    parts.append(
        table(
            [
                "对比",
                "臂/范围",
                "差值pp",
                "b0",
                "b1",
                "b2",
                "95%区间",
                "同时区间",
                "配对t敏感性95%",
                "解读",
            ],
            rows,
        )
    )
    parts.append("三块等权；回答数不等于独立训练重复数。区间范围与多重比较规则保留在原分析附录。\n")
    parts.append("\n## 外部面板全部指标\n")
    for pool, models in analyses["EXTERNAL_ANALYSIS.json"].items():
        rows = [
            [model, metric, value]
            for model, summary in models.items()
            for metric, value in leaves(summary)
        ]
        parts.extend(["\n### " + pool + "\n", table(["模型或对比", "原指标路径", "原值"], rows)])
    parts.append(
        "ChartQA 为已使用的回归基准；ChartQAPro 为冻结子集。官方、历史、v2与类型审计指标分别保留，"
        "UNRESOLVED 不默认归类为数值题，类型指标不替代官方分数。\n"
    )
    parts.append("\n## PREFIX 完整轨迹前缀诊断\n")
    prefix = analyses["PREFIX_DIAGNOSTICS.json"]
    rows = [
        [
            panel,
            item["records"],
            item["mean_generated_tokens"],
            budget,
            values["value_correct"],
            values["finish"],
            values["presentation"],
        ]
        for panel, item in prefix["panels"].items()
        for budget, values in item["budgets"].items()
    ]
    parts.append(
        table(
            [
                "模型/协议",
                "回答数",
                "平均tokens",
                "观察前缀",
                "A原比例",
                "完成原比例",
                "接口原比例",
            ],
            rows,
        )
    )
    parts.append("这些是同一完整轨迹的观察前缀，不是独立短预算实验，也不是反事实短预算能力。\n")
    parts.append(scheduler_report(scheduler))
    for name in REPORTS:
        parts.extend(
            [
                "\n## 完整原分析附录：" + name + "\n",
                "```json\n",
                encoded(analyses[name]).decode(),
                "```\n",
            ]
        )
    return "".join(parts)


def build(root, manifest_path, output):
    root = Path(root).resolve(strict=True)
    output = Path(output).absolute()
    receipt = Path(str(output) + ".verification.json")
    checksum = Path(str(output) + ".sha256")
    require(not output.is_relative_to(root), "Output must be outside source root")
    require(all(not p.exists() for p in (output, receipt, checksum)), "Output already exists")
    manifest_bytes = Path(manifest_path).read_bytes()
    manifest = loads(manifest_bytes)
    files = inventory(manifest)
    indexes = {key: [] for key in ("training_group", "support", "monitor_group", "engine_group")}
    analyses = {}
    scheduler = None
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".analysis_", suffix=".zip", dir=output.parent)
    os.close(descriptor)
    started = time.time()
    try:
        print(
            json.dumps(dict(stage="ARCHIVE_BEGIN", at=time.time(), files=len(files))),
            file=sys.stderr,
            flush=True,
        )
        payloads = {}
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True
        ) as package:
            for name, record in files.items():
                path = local_file(root, name)
                h, size = hashlib.sha256(), 0
                info = zipfile.ZipInfo(name)
                info.compress_type = (
                    zipfile.ZIP_STORED
                    if name.endswith((".gz", ".xz", ".zip"))
                    else zipfile.ZIP_DEFLATED
                )
                info._compresslevel = 1
                with path.open("rb") as source, package.open(info, "w", force_zip64=True) as target:
                    for data in iter(lambda: source.read(CHUNK), b""):
                        target.write(data)
                        h.update(data)
                        size += len(data)
                actual = dict(sha256=h.hexdigest(), bytes=size)
                require(actual == pin(record), "Input hash/size differs: " + name)
                payloads[name] = actual
                category = category_of(name, record["category"])
                if category in indexes:
                    value = loads(path.read_bytes())
                    indexes[category].append(
                        dict(
                            path=name, records=1 if category == "support" else len(value["records"])
                        )
                    )
                if name == "delivery_context/SCHEDULER_COST_SNAPSHOT.json":
                    scheduler = loads(path.read_bytes())
                if name in {"release/" + x for x in REPORTS}:
                    analyses[Path(name).name] = loads(path.read_bytes())
            generated = {
                "SOURCE_MANIFEST.json": manifest_bytes,
                "RAW_INDEX.json": encoded(indexes),
                "REPORT_ZH.md": report_text(analyses, scheduler).encode(),
                "START_HERE.md": (
                    "# 阅读入口\n\n先读 REPORT_ZH.md，再查 RAW_INDEX.json。原始文件位于其原路径。"
                    "SOURCE_MANIFEST.json 逐一列出选择范围；未自动收集目录、未读取权重。"
                    "此包是独立本地分析资料，不声称旧服务器验收通过。\n"
                ).encode(),
            }
            for name, data in generated.items():
                name = GENERATED + name
                package.writestr(name, data)
                payloads[name] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
            package.writestr(
                GENERATED + "ARCHIVE_MANIFEST.json", encoded(dict(schema=SCHEMA, files=payloads))
            )
        print(
            json.dumps(
                dict(stage="VERIFY_BEGIN", at=time.time(), bytes=Path(temporary).stat().st_size)
            ),
            file=sys.stderr,
            flush=True,
        )
        result = verify(temporary)
        result.update(elapsed_seconds=time.time() - started, verified_at=time.time())
        os.link(temporary, output)
        with receipt.open("x") as stream:
            stream.write(encoded(result).decode())
        with checksum.open("x") as stream:
            stream.write(result["archive"]["sha256"] + "  " + output.name + "\n")
        print(
            json.dumps(dict(stage="VERIFIED", at=time.time(), sha256=result["archive"]["sha256"])),
            file=sys.stderr,
            flush=True,
        )
        return result
    finally:
        Path(temporary).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        result = verify(args.output)
    else:
        parser.error(
            "--root and --manifest required"
        ) if not args.root or not args.manifest else None
        result = build(args.root, args.manifest, args.output)
    print(encoded(result).decode(), end="")


if __name__ == "__main__":
    main()
