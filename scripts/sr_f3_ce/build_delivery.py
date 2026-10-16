#!/usr/bin/env python3
"""Build a verified, rescorable delivery after SR-F3-CE's actual final release.

This auxiliary program never submits jobs, generates answers, modifies frozen
sources, or releases sealed results. Run with the experiment's Python environment.
Large raw artifacts are streamed directly into the ZIP, without staging copies.
"""
# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import importlib.util
import io
import itertools
import json
import math
import os
import re
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

EXPERIMENT = "SR-F3-CE-20261016"
RELEASE_FILES = (
    "RELEASE.json",
    "RELEASE_SCORED.json",
    "RAW_RESCORING.json",
    "ENDPOINT_RAW.jsonl.gz",
    "ENDPOINT_SCORES.jsonl",
    "PRIMARY_ANALYSIS.json",
    "EXTERNAL_ANALYSIS.json",
    "PREFIX_DIAGNOSTICS.json",
)
WEIGHTS = {".pt", ".pth", ".bin", ".safetensors", ".ckpt"}
FONTS = {".ttf", ".otf", ".ttc", ".woff", ".woff2"}
AGGREGATES = (
    "TRAINING_GROUPS.jsonl.gz",
    "SUPPORT_MASKS.jsonl.gz",
    "MONITOR_RAW.jsonl.gz",
    "ENGINE_RAW.jsonl.gz",
)
DELIVERY_MEMBERS = {
    "START_HERE.md",
    "REPORT_ZH.md",
    "DELIVERY_PROVENANCE.json",
    "DELIVERY_COVERAGE.json",
    "HELPER_SOURCE_MANIFEST.json",
    "SERVER_ONLY.json",
    "COSTS.json",
    "SCHEDULER_COST_RAW.json",
    "OPERATIONS_SNAPSHOT.json",
    "ENDPOINT_RAW.jsonl.gz",
    "ENDPOINT_SCORES.jsonl",
    "TASK_SIDECARS.jsonl.gz",
    "IMAGE_SOURCE_MAP.jsonl.gz",
    "UNRESOLVED_TYPES.jsonl.gz",
    "RAW_SOURCE_MAP.jsonl.gz",
    "TRAINING_SIGNAL_INDEX.json",
    *AGGREGATES,
}


def fail(message):
    raise PermissionError(message)


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            fail("Duplicate JSON key: " + key)
        result[key] = value
    return result


def loads(raw):
    return json.loads(
        raw, object_pairs_hook=_pairs, parse_constant=lambda value: fail("Nonfinite JSON: " + value)
    )


def read(path):
    return loads(Path(path).read_text(encoding="utf-8"))


def encoded(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def sha_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def safe_name(name):
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or "\x00" in name
        or re.match(r"^[A-Za-z]:", name)
    ):
        fail("Unsafe archive path")
    path = PurePosixPath(name)
    if path.is_absolute() or any(p in ("", ".", "..") for p in name.split("/")):
        fail("Unsafe archive path: " + name)
    return name


def bounded(root, relative):
    root = Path(root).resolve(strict=True)
    path = (root / safe_name(relative)).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        fail("Source escapes its registered root: " + relative)
    return path


def check_file(path, record):
    if path.stat().st_size != record["bytes"] or sha_file(path) != record["sha256"]:
        fail("Original artifact changed: " + str(path))


def load_frozen_api(root, *, extension_manifest=None):
    """Authenticate every deployed source before importing that exact source tree."""
    root = Path(root).resolve(strict=True)
    sys.dont_write_bytecode = True
    # This guards existing caches in the frozen tree; ordinary package imports remain.
    # Do not delete caches or claim to authenticate arbitrary third-party/native imports.
    for path in sorted((root / "code").rglob("*")):
        if path.is_file() and path.suffix.lower() in {".pyc", ".pyo"}:
            fail("Frozen source tree contains unbound Python bytecode: " + str(path))
    identity = read(root / "RUN_IDENTITY.json")
    if identity.get("experiment_id") != EXPERIMENT:
        fail("Wrong experiment")
    manifest_path = root / "deployment/SOURCE_MANIFEST.json"
    manifest = read(manifest_path)
    if not manifest.get("files"):
        fail("Frozen source inventory missing")
    names = set()
    for item in manifest["files"]:
        if item["path"] in names:
            fail("Duplicate frozen source path")
        names.add(item["path"])
        check_file(bounded(root / "code", item["path"]), item)
    code = (root / "code/src").resolve(strict=True)
    for name, module in list(sys.modules.items()):
        if name.startswith(("sr_f3_ce", "sr_f12", "sr_f1", "mm_core", "mm_dev")):
            file = getattr(module, "__file__", None)
            if file and not Path(file).resolve().is_relative_to(code):
                fail("Use a fresh process; an unfrozen scientific module is already imported")
    sys.path.insert(0, str(code))
    modules = {
        name: importlib.import_module("sr_f3_ce." + name)
        for name in (
            "protocol",
            "orchestration",
            "endpoint_runner",
            "evaluation",
            "runner",
            "checkpoint",
            "analysis",
            "contract",
            "external",
        )
    }
    if any(not Path(m.__file__).resolve().is_relative_to(code) for m in modules.values()):
        fail("Scientific imports did not come from root/code/src")
    api = SimpleNamespace(**modules)
    config = read(root / "plan/config/SR_F3_CE.json")
    revision = config.get("protocol_revision", "v1")
    if revision != "v1":
        api.protocol_source_guard = api.protocol.verify_run_source_revision(
            root, source_manifest=manifest
        )
    api.observability = load_observability_extension(root, api, extension_manifest)
    return api


def load_observability_extension(root, api, manifest_path=None):
    """Authenticate the unique freeze-bound extension before executing its loader."""
    api.protocol.verify_run(root, science=True)
    identity, freeze = read(root / "RUN_IDENTITY.json"), read(root / "EXECUTION_FREEZE.json")
    bound = freeze.get("engineering_files", {})
    matches = [
        name
        for name in bound
        if re.fullmatch(r"native/observability/[0-9a-f]{40}/SOURCE_MANIFEST\.json", name)
    ]
    if len(matches) != 1:
        fail("Scientific freeze must bind exactly one canonical observability extension")
    selected = bounded(root, matches[0])
    if manifest_path is not None and Path(manifest_path).resolve(strict=True) != selected:
        fail("Requested observability extension differs from the unique scientific freeze")
    directory = selected.parent
    for filename in ("SOURCE_MANIFEST.json", "CPU_VALIDATION.json", "AMENDMENT.json"):
        path = bounded(root, str((directory / filename).relative_to(root)))
        if bound.get(str(path.relative_to(root))) != sha_file(path):
            fail("Observability manifest/CPU validation/amendment is not directly freeze-bound")
    manifest = read(selected)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("experiment_id") != EXPERIMENT
        or manifest.get("status") != "FROZEN_OBSERVABILITY_EXTENSION"
        or manifest.get("extension_commit") != directory.name
        or manifest.get("base_code_commit") != identity["code_commit"]
        or manifest.get("run_identity_sha256") != sha_file(root / "RUN_IDENTITY.json")
        or manifest.get("base_source_manifest_sha256")
        != sha_file(root / "deployment/SOURCE_MANIFEST.json")
    ):
        fail("Observability extension identity differs from the frozen scientific source")
    sources = {}
    for row in manifest.get("sources", []):
        name = safe_name(row["path"])
        if name in sources:
            fail("Duplicate observability source identity")
        path = bounded(directory, name)
        check_file(path, row)
        sources[name] = (path, row)
    if not sources.keys() >= {"instrumented_execution.py", "observe_runtime.py"}:
        fail("Observability extension lacks its authenticated launcher/observer source")
    cpu = manifest.get("cpu_validation", {})
    if cpu.get("path") != "CPU_VALIDATION.json":
        fail("Observability CPU validation path differs")
    check_file(directory / "CPU_VALIDATION.json", cpu)
    validation = read(directory / "CPU_VALIDATION.json")
    if (
        validation.get("status") != "PASS"
        or validation.get("scope") != "CPU_TOY_TRANSPARENCY_AND_ORCHESTRATION"
        or type(validation.get("tests_passed")) is not int
        or validation["tests_passed"] <= 0
        or validation.get("gpu_validation_performed") is not False
        or type(validation.get("new_model_generations")) is not int
        or validation["new_model_generations"] != 0
        or type(validation.get("new_optimizer_steps")) is not int
        or validation["new_optimizer_steps"] != 0
        or validation.get("source_sha256")
        != {name: row["sha256"] for name, (_, row) in sources.items()}
    ):
        fail("Observability validation must preserve its actual CPU-only evidence boundary")
    path, expected = sources["instrumented_execution.py"]
    source = path.read_bytes()
    if len(source) != expected["bytes"] or hashlib.sha256(source).hexdigest() != expected["sha256"]:
        fail("Observability launcher changed before import")
    name = "srf3_delivery_observability_" + digest([str(path), expected["sha256"]])
    module = importlib.util.module_from_spec(importlib.util.spec_from_file_location(name, path))
    sys.modules[name] = module
    # Execute this validated buffer. A loader may reread the path or use an unbound .pyc,
    # including an unchecked-hash cache even when sys.dont_write_bytecode is true.
    exec(compile(source, str(path), "exec"), module.__dict__)
    module.verify_extension(root, selected, science=True)
    controller = module.install(root, selected)
    if (
        api.orchestration.Controller is not controller
        or controller.__name__ != "InstrumentedController"
    ):
        fail("Verified observability Controller was not installed for real terminal validation")
    return SimpleNamespace(
        module=module,
        manifest_path=selected,
        controller=controller,
        identity=dict(
            manifest_path=matches[0],
            manifest_sha256=sha_file(selected),
            extension_commit=manifest["extension_commit"],
            base_code_commit=manifest["base_code_commit"],
            **(
                {"protocol_source_guard": manifest["protocol_source_guard"]}
                if "protocol_source_guard" in manifest
                else {}
            ),
            observer_source_sha256=sources["observe_runtime.py"][1]["sha256"],
            amendment_sha256=sha_file(directory / "AMENDMENT.json"),
            cpu_validation_sha256=sha_file(directory / "CPU_VALIDATION.json"),
        ),
    )


def require_final_release(root, api):
    """No TEST content is opened before Controller's full terminal/receipt checks."""
    api.protocol.verify_run(root, science=True)
    if getattr(api, "observability", None) is not None:
        extension = api.observability
        extension.module.verify_extension(root, extension.manifest_path, science=True)
        if api.orchestration.Controller is not extension.controller:
            fail("Observability Controller was replaced before task terminal verification")
    completed = api.endpoint_runner._completed(root)
    required = set(api.orchestration.TASKS)
    missing = required - completed
    if missing:
        fail(
            "Delivery is sealed/incomplete; verified Controller tasks missing: "
            + ", ".join(sorted(missing))
        )
    # Controller.completed authenticates accounting, allocation/script, process,
    # freeze and every bound output byte. Keep these exact receipts in delivery.
    receipts = {
        task: read(Path(root) / "tasks" / task / "COMPLETE.json") for task in sorted(required)
    }
    if (
        len([t for t in required if t.startswith("train--")]) != 12
        or len([t for t in required if t.startswith("eval--")]) != 15
        or "release" not in required
    ):
        fail("Frozen task matrix differs from twelve training paths and fifteen endpoints")
    return receipts


class ArrayHash:
    """The exact canonical JSON list hash, without retaining all raw rows."""

    def __init__(self):
        self.value, self.count = hashlib.sha256(b"["), 0

    def add(self, item):
        if self.count:
            self.value.update(b",")
        self.value.update(encoded(item))
        self.count += 1

    def hexdigest(self):
        value = self.value.copy()
        value.update(b"]")
        return value.hexdigest()


def json_lines(stream):
    for line in stream:
        if not line.strip():
            fail("Blank line in complete JSONL evidence")
        yield loads(line)


def raw_fields(row):
    tokens = row.get("tokens")
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(type(t) is not int or t < 0 for t in tokens)
        or not isinstance(row.get("raw_text"), str)
        or not isinstance(row.get("model_input", {}).get("prompt"), str)
        or row.get("finish_reason") not in ("balanced", "final_tag", "eos", "length")
        or type(row.get("max_new_tokens")) is not int
        or not 0 < len(tokens) <= row["max_new_tokens"]
    ):
        fail("Original prompt/text/tokens/termination/budget missing")


def verify_release_stream(root, groups, tasks, scorer):
    """Verify delivered gzip against original groups and rescore every row again."""
    root, destination = Path(root), Path(root) / "release"
    manifest = read(destination / "MANIFEST.json")
    inventory = {item["path"]: item for item in manifest["files"]}
    if len(inventory) != len(manifest["files"]) or set(inventory) != set(RELEASE_FILES):
        fail("Release manifest does not contain the complete fixed delivery inventory")
    for name, item in inventory.items():
        check_file(bounded(destination, name), item)
    release, scored, rescoring = [
        read(destination / name)
        for name in (
            "RELEASE.json",
            "RELEASE_SCORED.json",
            "RAW_RESCORING.json",
        )
    ]
    if (
        release.get("status") != "RELEASED"
        or release.get("experiment_id") != EXPERIMENT
        or any(scored.get(k) != v for k, v in release.items())
        or scored.get("raw_rescoring") != "PASS"
        or scored.get("projection_kind") != "verified_raw_rescore_projection_v1"
        or scored.get("rescoring_receipt_sha256") != sha_file(destination / "RAW_RESCORING.json")
        or rescoring.get("status") != "PASS"
        or rescoring.get("mismatches") != []
        or rescoring.get("gzip_read_to_eof") is not True
        or rescoring.get("raw_sha256") != inventory["ENDPOINT_RAW.jsonl.gz"]["sha256"]
        or rescoring.get("records") != release.get("records")
    ):
        fail("Release/full-original-rescoring receipt chain differs")
    projection, seen, score_hashes, raw_hashes, counts = ArrayHash(), set(), {}, {}, Counter()
    with (
        gzip.open(destination / "ENDPOINT_RAW.jsonl.gz", "rt", encoding="utf-8") as raw_stream,
        (destination / "ENDPOINT_SCORES.jsonl").open(encoding="utf-8") as score_stream,
    ):
        raw_rows, compact_rows = json_lines(raw_stream), json_lines(score_stream)
        for group, original, entry in groups:
            gid, recomputed = group["group_id"], []
            if gid in raw_hashes:
                fail("Duplicate original endpoint group")
            raw_hashes[gid] = digest(original)
            for source, slot in zip(original, group["slots"], strict=True):
                delivered, compact = next(raw_rows, None), next(compact_rows, None)
                if delivered != source or delivered is None:
                    fail("Delivered raw differs from original endpoint bytes/content")
                raw_fields(delivered)
                sid = delivered["slot_id"]
                if sid in seen or any(delivered.get(k) != v for k, v in slot.items()):
                    fail("Duplicate or changed endpoint slot")
                seen.add(sid)
                score = scorer(delivered, tasks[delivered["qid"]])
                expected = dict(
                    slot,
                    group_id=gid,
                    generation_status=delivered["generation_status"],
                    technical_validation_errors=delivered.get("technical_validation_errors", []),
                    score=score,
                )
                if compact != expected:
                    fail("Full original rescoring differs from delivered endpoint score")
                projection.add(compact)
                recomputed.append(dict(slot_id=sid, score=score))
                counts[delivered["pool"]] += 1
            score_hashes[gid] = digest(recomputed)
            stored = read(destination / "scores" / (safe_name(gid) + ".json"))
            if stored != dict(
                raw_sha256=entry["sha256"], scores=recomputed, scorer_version=EXPERIMENT
            ):
                fail("Original score-group chain differs")
        if next(raw_rows, None) is not None or next(compact_rows, None) is not None:
            fail("Unregistered extra endpoint rows")
    if (
        len(seen) != release["records"]
        or raw_hashes != release["raw_groups"]
        or projection.hexdigest() != scored["analysis_rows_sha256"]
        or digest(score_hashes) != rescoring["group_scores_sha256"]
    ):
        fail("Endpoint coverage/hash/full-rescore chain differs")
    return dict(
        status="VERIFIED_COMPLETE_RELEASE",
        records=len(seen),
        pool_counts=dict(counts),
        raw_sha256=rescoring["raw_sha256"],
        gzip_read_to_eof=True,
        analysis_rows_sha256=projection.hexdigest(),
        full_original_rescoring="PASS",
        independent_scorer_claimed=False,
        method="delivered gzip versus original groups, then frozen production rescoring",
    )


def _validate_analysis(analysis):
    if (
        analysis.get("status") != "NUMERICAL_ANALYSIS_COMPLETE"
        or analysis.get("experiment_id") != EXPERIMENT
        or analysis.get("all_registered_blocks_retained") is not True
        or analysis.get("adverse_results_retained") is not True
        or set(analysis.get("primary", {})) != {"C1", "C2", "C3"}
    ):
        fail("Complete frozen analysis schema required")
    specs = (
        ("EXEC", "PREFIX", "REPAIR"),
        ("EXEC", "GOLD", "REPAIR"),
        ("EXEC", "GOLD", "TEST_COMPOSITION"),
    )
    for key, spec in zip(("C1", "C2", "C3"), specs, strict=True):
        item = analysis["primary"][key]
        if (
            tuple(item.get(k) for k in ("arm1", "arm0", "scope")) != spec
            or item.get("metric") != "value_correct"
            or len(item["per_block_pp"]) != 3
            or item.get("bootstrap_replicates") != 10000
            or item.get("bootstrap_seed") != 2026101651
            or not math.isclose(item["confidence_simultaneous"], 1 - 0.05 / 3)
            or item["paired_t_sensitivity"].get("df") != 2
        ):
            fail("Primary contrast/three-block/uncertainty contract differs")
    arms = {"PARENT", "RL", "GOLD", "PREFIX", "EXEC"}
    metrics = {
        "value_correct",
        "certificate",
        "certificate_truth",
        "certificate_sufficient",
        "presentation",
        "finish",
        "J",
        "legacy_full",
        "value_parseable",
    }
    scopes = {"REPAIR", "LEGACY", "TEST_COMPOSITION", "TEST_LANGUAGE", "TEST_RENDER", "TEST_ARITY"}
    if set(analysis.get("decomposition", {})) != scopes or set(
        analysis.get("retention", {})
    ) != arms - {"PARENT"}:
        fail("Repair/transfer/retention analysis is incomplete")
    for values in analysis["decomposition"].values():
        if set(values) != arms or any(set(row) != metrics for row in values.values()):
            fail("A registered arm or decomposition metric was omitted")
        for row in values.values():
            if any(len(item["per_block"]) != 3 for item in row.values()):
                fail("All three blocks are required for decomposition")
    for values in analysis["retention"].values():
        if set(values) != {"value_correct", "J"} or any(
            set(item["by_family"]) != {"CROSS", "THRESHOLD", "TOPK", "INTERVAL"}
            for item in values.values()
        ):
            fail("Retention must retain every family and both registered metrics")
    expected_secondary = {
        scope + ":" + comparison
        for scope in ("REPAIR", "TEST_COMPOSITION")
        for comparison in ("GOLD-RL", "PREFIX-GOLD", "EXEC-RL")
    }
    expected_protocols = {
        protocol + ":" + comparison
        for protocol in ("structured", "answer_only", "natural_work")
        for comparison in ("EXEC-PREFIX", "EXEC-GOLD")
    }
    if (
        set(analysis.get("secondary", {})) != expected_secondary
        or set(analysis.get("canonical_protocols", {})) != expected_protocols
    ):
        fail("Secondary/protocol contrasts cannot be selectively omitted")
    if set(analysis.get("canonical_protocol_rates", {})) != {
        "structured",
        "answer_only",
        "natural_work",
    }:
        fail("Independent output protocol rates are incomplete")
    for protocol, values in analysis["canonical_protocol_rates"].items():
        required_metrics = (
            metrics
            if protocol == "structured"
            else {"value_correct", "presentation", "finish", "value_parseable"}
        )
        if set(values) != arms or any(set(row) != required_metrics for row in values.values()):
            fail("Every arm and applicable protocol metric is required")
    return analysis


def _number(value, scale=1):
    return "UNKNOWN" if value is None else f"{float(value) * scale:+.3f}"


def _interval(values, scale=1):
    return "UNKNOWN" if values is None else "[" + ", ".join(_number(v, scale) for v in values) + "]"


def _table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    return (
        "\n".join(
            [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |",
                *["| " + " | ".join(cell(v) for v in row) + " |" for row in rows],
            ]
        )
        + "\n"
    )


def render_report(
    analysis,
    external,
    prefix,
    coverage,
    costs,
    prior,
    external_status,
    training_signals=None,
    work_trajectory=None,
):
    """Render actual values, including every adverse block; never manufacture findings."""
    _validate_analysis(analysis)
    text = [
        "# SR-F3-CE 中文事实报告\n",
        "本报告来自已验证释放的固定 12 条训练路径和 15 个端点。"
        "完整原始回答、失败回答与技术事件均保留。"
        "科学结论与技术完成状态分开；ChartQAPro 若 UNAVAILABLE，外部任务仍未完成。\n",
        "## 计划与实际\n",
        _table(
            ["项目", "计划", "已验证实际"],
            [
                [
                    "训练路径 / 更新",
                    "12 / 768",
                    f"{coverage['training_paths']} / {coverage['logical_updates']}",
                ],
                ["训练回答", "98,304", coverage["training_records"]],
                ["支持 teacher forcing", "2,304", coverage["support_examples"]],
                ["MONITOR 回答", "26,112", coverage["monitor_records"]],
                ["ENGINE 协议诊断", "2,880", coverage["engine_diagnostic_records"]],
                ["ENGINE 新 rollout / 重放 / 物理更新", "512 / 256 / 6", "512 / 256 / 6"],
                ["端点原文", "229,500 + 15 × 可用 ChartQAPro 图像根", coverage["records"]],
            ],
        ),
        "## 三个主要对比：全部三个实际块在同表\n",
    ]
    if coverage.get("protocol_revision", "v1") != "v1":
        text.insert(
            2,
            (
                "当前科学记录使用已冻结的 "
                + coverage["protocol_revision"]
                + " 协议及 indent=2 教师序列化。表中 ENGINE、恢复缓存、MONITOR 和训练/端点"
                "均来自本次选择后的新运行；旧 v1 原文及提示消融独立保留，不计入这些分母。"
                "选择只依三父块的 value_parseable 技术门槛，不依正确率或科学臂表现。"
                "2,304 条消融和 3,904 条历史 v1 原文的保留链见 "
                "[PROTOCOL_REVISION_RETENTION.json](PROTOCOL_REVISION_RETENTION.json)。\n"
            ),
        )
    rows = []
    for name in ("C1", "C2", "C3"):
        item = analysis["primary"][name]
        ci = item["CI_simultaneous_pp"]
        interpretation = (
            "正向增量获支持"
            if ci[0] > 0
            else "负向结果，保留"
            if ci[1] < 0
            else "本设置未确认；不能称等效"
        )
        rows.append(
            [
                name,
                f"{item['arm1']}−{item['arm0']} / {item['scope']}",
                _number(item["effect_pp"]),
                *map(_number, item["per_block_pp"]),
                _interval(item["CI95_pp"]),
                _interval(ci),
                _interval(item["paired_t_sensitivity"]["CI95_pp"]),
                "/".join(
                    str(
                        sum(
                            x > 0 if sign == "+" else x < 0 if sign == "-" else x == 0
                            for x in item["per_block_pp"]
                        )
                    )
                    for sign in ("+", "-", "0")
                ),
                interpretation,
            ]
        )
    text += [
        _table(
            [
                "对比",
                "范围",
                "差值 pp",
                "b0",
                "b1",
                "b2",
                "95%",
                "98.3333%",
                "df=2 t 敏感性95%",
                "正/负/零块数",
                "事实解释",
            ],
            rows,
        ),
        "区间采用 family 内 root 簇重采样 10,000 次、seed 2026101651、三个块等权。"
        "不确定性条件于这三个实际父模型及未来训练实现；回答数量不是训练重复次数。"
        "2 pp 是描述尺度，不是功效保证或事后停止标准。\n",
        "## 修复、真正迁移、保持及证据/接口分解\n",
    ]
    for scope, arms in analysis["decomposition"].items():
        rows = []
        for arm, metrics in arms.items():
            for metric, item in metrics.items():
                rows.append(
                    [
                        arm,
                        metric,
                        _number(item["rate"], 100),
                        *[_number(x, 100) for x in item["per_block"]],
                    ]
                )
        text += [f"### {scope}\n", _table(["臂", "指标", "%", "b0 %", "b1 %", "b2 %"], rows)]
    text += [
        "REPAIR 属于已进入训练/支持设计的修复学习；TEST_COMPOSITION 才检验注册的新 AST 组合。"
        "LANGUAGE、RENDER、ARITY 各自单列，不能合并成通用泛化证明。\n",
        "## 相对父起点的保持与每族退化\n",
    ]
    rows = []
    for arm, metrics in analysis["retention"].items():
        for metric, item in metrics.items():
            for family, value in item["by_family"].items():
                rows.append(
                    [
                        arm,
                        metric,
                        family,
                        _number(value["effect_pp"]),
                        *map(_number, value["per_block_pp"]),
                        _interval(value["CI95_pp"]),
                        "最坏族" if family == item["worst_family"] else "",
                        "低于 −2 pp" if value["effect_pp"] < -2 else "未低于描述边界",
                    ]
                )
    text += [
        _table(["臂", "指标", "family", "差值 pp", "b0", "b1", "b2", "95%", "最坏", "边界"], rows),
        "不能用某族收益抵消另一族下降，未校准的平均值不是安全保证。\n",
        "## 预声明次级对比与无辅助 RL\n",
    ]
    rows = [
        [
            name,
            _number(v["effect_pp"]),
            *map(_number, v["per_block_pp"]),
            _interval(v["CI95_pp"]),
            _interval(v["CI_simultaneous_pp"]),
        ]
        for name, v in analysis["secondary"].items()
    ]
    text += [
        _table(["对比", "差值 pp", "b0", "b1", "b2", "95%", "存档区间"], rows),
        "## 独立输出协议与完成成本\n",
    ]
    rows = [
        [name, _number(v["effect_pp"]), *map(_number, v["per_block_pp"]), _interval(v["CI95_pp"])]
        for name, v in analysis["canonical_protocols"].items()
    ]
    text += [_table(["协议对比", "差值 pp", "b0", "b1", "b2", "95%"], rows)]
    for protocol, arms in analysis["canonical_protocol_rates"].items():
        rows = [
            [arm, metric, _number(v["rate"], 100), *[_number(x, 100) for x in v["per_block"]]]
            for arm, metrics in arms.items()
            for metric, v in metrics.items()
        ]
        text += [f"### {protocol}\n", _table(["臂", "指标", "%", "b0", "b1", "b2"], rows)]
    rows = [
        [
            key,
            v["records"],
            f"{v['mean_generated_tokens']:.3f}",
            budget,
            _number(scores["value_correct"], 100),
            _number(scores["finish"], 100),
            _number(scores["presentation"], 100),
        ]
        for key, v in prefix["panels"].items()
        for budget, scores in v["budgets"].items()
    ]
    text += [
        _table(["模型/协议", "回答数", "平均 tokens", "观察前缀", "A %", "完成 %", "接口 %"], rows),
        "128/256/512/1024 是同一完整轨迹的观察前缀，不是独立短预算实验；"
        "不能据此声称反事实短预算能力。\n",
        "## 外部回归、类型双审与未解析题\n",
        "ChartQA 是反复用于开发的回归基准，不是全新确认集。ChartQAPro 只代表冻结子集，"
        "官方分数、历史/v2 分数和类型审计分别呈现；UNRESOLVED 不默认归为一般数值。\n",
    ]
    for pool, models in external.items():
        text.append(f"### {pool}\n")
        if not models:
            text.append(
                "外部未完成："
                + json.dumps(external_status.get(pool, {}), ensure_ascii=False)
                + "\n"
            )
        for model, summary in models.items():
            if model == "paired_image_cluster_contrasts":
                rows = [
                    [
                        name,
                        _number(v["point"], 100),
                        *[_number(x, 100) for x in v["per_block"]],
                        _interval(v.get("CI95"), 100),
                    ]
                    for name, v in summary.items()
                ]
                text.append(_table(["图像簇对比", "差值 pp", "b0", "b1", "b2", "95%"], rows))
            else:
                text += [
                    f"**{model}**\n",
                    "```json\n" + json.dumps(summary, ensure_ascii=False, indent=2) + "\n```\n",
                ]
    if training_signals is not None:
        text += [
            "## 训练信号与监督分配\n",
            "[训练信号索引](TRAINING_SIGNAL_INDEX.json) 保留全部物理更新回执入口，"
            "包括每组真实 J、均值/总体标准差/优势/常数组，逐 support 的 answer/work CE、"
            "字段 token 数，以及 policy/KL/CE 的加权和未加权值。"
            "1/16/32/64 的分项梯度范数/夹角、clip 前后范数、参数差和概率门禁值另外摘录。"
            "每项均指向包内原始回执及 SHA-256；重复逻辑步的物理尝试完整保留，"
            "不能据此增加训练重复次数。\n",
            _table(
                ["路径", "已覆盖逻辑步", "保留物理更新回执", "诊断步"],
                [
                    [
                        run,
                        len(values["logical_steps"]),
                        len(values["attempts"]),
                        "/".join(map(str, values["diagnostic_steps"])),
                    ]
                    for run, values in training_signals["runs"].items()
                ],
            ),
            "PREFIX 与 EXEC 完整支持序列相同，GOLD 支持较短；实际 token 和梯度成本不相同。"
            "接受充分证据作为所有臂共有基础，EXEC 收益不能称为该基础的单独因果证明。\n",
        ]
    if work_trajectory is not None:
        text += [
            "## work 字段随训练的描述统计\n",
            "[全部逐步与 MONITOR 统计](WORK_OCCURRENCE_TRAJECTORY.json) "
            "保留 768 个训练步及 51 个固定 MONITOR 面板。"
            "分子是第一个严格 JSON 对象中存在小写顶层 work 键的回答；同时保留非空字符串计数。"
            "全体回答率包含无效输出，可解析回答率仅使用 value_parseable=1 子集；"
            "分母为零时 UNKNOWN。"
            "这些统计不参与奖励、提示选择或训练路由。下表来自同一固定 MONITOR 池。\n",
            _table(
                ["模型", "块", "步", "全部回答", "work/全部 %", "可解析回答", "work/可解析 %"],
                [
                    [
                        item["model_id"],
                        item["block"],
                        item["step"],
                        item["summary"]["records"],
                        _number(item["summary"]["work_fraction_all_records"], 100),
                        item["summary"]["parseable_answers"],
                        _number(item["summary"]["work_fraction_parseable_answers"], 100),
                    ]
                    for item in work_trajectory["monitor"]
                ],
            ),
        ]
    text += [
        "完整未解析类型和双审原件见 "
        "[UNRESOLVED_TYPES.jsonl.gz](UNRESOLVED_TYPES.jsonl.gz) 与 evidence/external。\n",
        "## 实际成本、重放与技术失败\n",
        "前瞻观测按每个物理 worker attempt 保留 BEGIN/END、安装身份和 FINAL 回执。"
        "实际 hook 计数只归当前最内层 runtime scope，model_hook_layers 不重复加入模型阶段计数。"
        "父 scope 墙钟不与子 scope 累加，视觉时间与模型阶段时间为重叠轴，不能相加；"
        "这些是主机回调时间，不是 GPU kernel 时间。缺 END、FINAL 或覆盖不完整的尝试标 UNKNOWN；"
        "本次 ENGINE 在观测扩展安装前完成，未回补观测，其缺失阶段成本仍未知。\n",
        "```json\n" + json.dumps(costs, ensure_ascii=False, indent=2) + "\n```\n",
        "UNKNOWN 表示账本未记录或无法无歧义汇总；没有据配置、申请时长或重复行推算 GPU 小时。"
        "失败尝试和重算仍保留，不能以成功样本成本替代实际成本。\n",
        "## 历史材料缺件\n",
        "```json\n" + json.dumps(prior, ensure_ascii=False, indent=2) + "\n```\n",
        "非关键旧诊断的 MISSING 如实保留；没有生成替代旧样本，不改变完整新实验的分母。\n",
        "## 能与不能推出的结论\n",
        "主要区间跨零只能写“本设置未确认”，不能写等效。仅修复提高而新组合未提高，"
        "只能支持特定组合学习。任何负结果、异向块、退化族和失败回答都不得删除。"
        "EXEC 与 PREFIX 比较同一支持文本下的监督分配，没有完全分离信息内容与优化尺度；"
        "不证明内部过程忠实、通用能力提升、容量下界或固定策略最优。"
        "三个块同向也不是任意重训种子的保证。\n",
        "原文重评分采用冻结生产实现的第二次全量读取；没有声称新独立评分器。"
        "权重及完整 checkpoint 留在服务器，逐文件身份见 [SERVER_ONLY.json](SERVER_ONLY.json)；"
        "此包可直接重评分，但不是包含模型权重的重训包。\n",
    ]
    return "\n".join(text)


def collect_training(root, api, dataset, store):
    groups, supports = [], []
    for run in dataset.runs:
        directory = root / "runs" / run["run_id"]
        state, marker = api.checkpoint.read_checkpoint(directory / "checkpoints", step=64)
        if (
            state["committed_logical_step"] != 64
            or state["run_identity"]["stage"] != "SCIENCE"
            or state["run_identity"]["run_id"] != run["run_id"]
        ):
            fail("Training delivery is not a registered scientific step-64 endpoint")
        schedule = dataset.schedules[run["block"]]
        api.runner.verify_committed_records(directory, state, schedule, dataset.tasks, store)
        if (
            read(root / "tasks" / ("train--" + run["run_id"]) / "EVIDENCE.json")["evidence"][
                "checkpoint_sha256"
            ]
            != marker["sha256"]
        ):
            fail("Final checkpoint differs from terminal training evidence")
        expected = set()
        for item in schedule:
            for slot, qid in enumerate(item["rl_qids"]):
                path = directory / "rollouts" / f"{item['step']:02d}-{slot:02d}.json"
                expected.add(path.name)
                value = read(path)
                if len(value["records"]) != 8:
                    fail("Every training group must retain eight answers")
                for row in value["records"]:
                    raw_fields(row)
                    if row["model_input"] != store.model_input(qid):
                        fail("Training prompt differs from frozen input")
                    task = dataset.tasks[qid]
                    actual = api.contract.score(
                        row["full_text"],
                        task["world"],
                        task["query"],
                        finish_reason=row["finish_reason"],
                        protocol="structured",
                    )
                    if row["score"] != actual or row["reward"] != actual["J"]:
                        fail("Original training answer reparse differs")
                groups.append(
                    (path, dict(run_id=run["run_id"], task_sidecar=dataset.tasks[qid]), 8)
                )
            if run["arm"] != "RL":
                for qid in item["support_qids"]:
                    path = directory / "support" / f"{item['step']:02d}-{qid}.json"
                    audit = read(path)
                    native = read(root / "native/support_masks" / f"{qid}-{run['arm']}.json")
                    if any(
                        audit[k] != native[k]
                        for k in ("tokens", "answer_mask", "work_mask", "mask_audit")
                    ):
                        fail("Support tokens/masks differ from native audit")
                    if audit["target"] != native["original_target"]:
                        fail("Support teacher differs from frozen native target")
                    supports.append(
                        (
                            path,
                            dict(
                                run_id=run["run_id"],
                                task_sidecar=dataset.tasks[qid],
                                model_input=store.model_input(qid),
                                native_mask_path=f"evidence/native/support_masks/{qid}-{run['arm']}.json",
                            ),
                            1,
                        )
                    )
        if {
            p.name for p in (directory / "rollouts").glob("[0-9][0-9]-[0-9][0-9].json")
        } != expected:
            fail("Training raw group inventory differs from fixed 64-step stream")
        del state
    if len(groups) != 12 * 64 * 16 or len(supports) != 2304:
        fail("Training/support full coverage differs")
    return groups, supports


def work_occurrence_trajectory(root, api, dataset, training, monitor):
    """Recompute descriptive work rates from all already retained current-run raw."""
    if not slot_revision_kwargs(root):
        return None
    models = {model["model_id"]: model for model in api.evaluation.endpoint_models()}
    result = {}
    for kind, entries, expected_panels, expected_rows in (
        ("training", training, 12 * 64, 128),
        ("monitor", monitor, 3 + 12 * 4, 512),
    ):
        panels = defaultdict(list)
        paths = defaultdict(list)
        for path, metadata, _ in entries:
            original = read(path)
            if kind == "training":
                model_id, step = metadata["run_id"], int(path.name[:2])
            else:
                model_id = path.parents[2].name
                step = int(path.parents[1].name.removeprefix("step-"))
            if model_id not in models:
                fail("Work trajectory contains an unregistered model")
            key = (model_id, step)
            for row in original["records"]:
                if kind == "monitor":
                    task = dataset.tasks[row["qid"]]
                    score = api.contract.score(
                        row["raw_text"],
                        task["world"],
                        task["query"],
                        finish_reason=row["finish_reason"],
                        protocol="structured",
                    )
                    row = dict(row, score=score)
                panels[key].append(row)
            paths[key].append(path)
        if len(panels) != expected_panels:
            fail("Work trajectory lacks complete fixed current-run panel coverage")
        values = []
        for (model_id, step), rows in sorted(panels.items()):
            if len(rows) != expected_rows:
                fail("Work trajectory denominator differs from fixed raw coverage")
            model = models[model_id]
            allowed = (
                range(1, 65)
                if kind == "training"
                else ((0,) if model["arm"] == "PARENT" else (16, 32, 48, 64))
            )
            if step not in allowed or (kind == "training" and model["arm"] == "PARENT"):
                fail("Work trajectory has an unregistered observation step")
            summary = api.runner.work_occurrence(rows)
            if kind == "monitor":
                summary = dict(
                    summary,
                    scope="fixed checkpoint MONITOR raw; invalid text retained in denominator",
                )
            sources = [
                dict(path="evidence/" + str(path.relative_to(root)), sha256=sha_file(path))
                for path in paths[(model_id, step)]
            ]
            if kind == "training":
                receipt_path = (
                    root / "runs" / model_id / "work_occurrence" / f"step-{step:02d}.json"
                )
                receipt = read(receipt_path)
                if (
                    receipt.get("step") != step
                    or receipt.get("summary") != summary
                    or (receipt.get("raw_group_sha256") != [item["sha256"] for item in sources])
                ):
                    fail("Stored training work occurrence differs from its exact original rollouts")
                sources.append(
                    dict(
                        path="evidence/" + str(receipt_path.relative_to(root)),
                        sha256=sha_file(receipt_path),
                    )
                )
            values.append(
                dict(
                    model_id=model_id,
                    arm=model["arm"],
                    block=model["block"],
                    step=step,
                    summary=summary,
                    sources=sources,
                )
            )
        result[kind] = values
    return dict(
        protocol_revision=slot_revision_kwargs(root)["revision"],
        scope="descriptive existing raw only; no reward or selection changes",
        new_model_calls=0,
        training=result["training"],
        monitor=result["monitor"],
    )


def training_signal_index(root, runs):
    result = {}
    fields = (
        "policy",
        "kl",
        "aux_ce",
        "policy_weighted",
        "kl_weighted",
        "aux_ce_weighted",
        "gradient_norm_before_clip",
        "gradient_norm_after_clip",
        "parameter_max_absolute_change",
        "parameter_l2_change",
        "optimizer_steps",
        "optimizer_updates",
        "constant_group_fraction",
        "gradient_diagnostics",
        "microbatch_size",
        "sampler_train_mean_absolute_difference",
        "sampler_train_p99_absolute_difference",
        "sampler_train_max_absolute_difference",
        "tis_truncated_fraction",
        "tis_cap",
    )
    for run in runs:
        attempts, logical_steps, diagnostic_steps = [], set(), set()
        for path in sorted((root / "runs" / run["run_id"] / "update_attempts").glob("*.json")):
            original = read(path)
            step, metrics = original["step"], original["metrics"]
            if (
                type(step) is not int
                or not 1 <= step <= 64
                or metrics.get("step") != step
                or metrics.get("arm") != run["arm"]
                or metrics.get("finite") is not True
                or metrics.get("engineering_zero_aux") is not False
                or metrics.get("scientific_recipe_eligible") is not True
                or metrics.get("optimizer_updates") != 1
                or any(key not in metrics for key in fields)
                or len(metrics.get("reward_groups", [])) != 16
                or len(metrics.get("support_losses", [])) != (0 if run["arm"] == "RL" else 4)
            ):
                fail("Retained training signal receipt is incomplete or not a scientific update")
            logical_steps.add(step)
            item = dict(
                step=step,
                path="evidence/" + str(path.relative_to(root)),
                sha256=sha_file(path),
                bytes=path.stat().st_size,
                constant_group_fraction=metrics["constant_group_fraction"],
            )
            if step in {1, 16, 32, 64}:
                if not metrics["gradient_diagnostics"]:
                    fail("Fixed component gradient diagnostic missing")
                diagnostic_steps.add(step)
                item["fixed_diagnostic"] = {key: metrics[key] for key in fields}
            attempts.append(item)
        if logical_steps != set(range(1, 65)) or diagnostic_steps != {1, 16, 32, 64}:
            fail("Training signal coverage does not include all steps and fixed diagnostics")
        result[run["run_id"]] = dict(
            logical_steps=sorted(logical_steps),
            diagnostic_steps=sorted(diagnostic_steps),
            attempts=attempts,
        )
    return dict(
        runs=result,
        scope="all retained applied physical attempts; repeated logical steps are not merged",
        all_raw_receipts_included=True,
    )


def collect_panel(root, api, store, tasks, slots, directory, policy_hash):
    coverage = read(directory / "COVERAGE.json")
    expected = api.evaluation.group_slots(slots)
    if (
        coverage["records"] != len(slots)
        or coverage["slots_sha256"] != digest(slots)
        or coverage["policy_hash"] != policy_hash
        or coverage["records_sha256"] != digest(coverage["groups"])
        or [g["group_id"] for g in expected] != [g["group_id"] for g in coverage["groups"]]
    ):
        fail("Panel policy/coverage manifest differs")
    results = []
    for group, entry in zip(expected, coverage["groups"], strict=True):
        path = directory / "groups" / (safe_name(group["group_id"]) + ".json")
        if sha_file(path) != entry["sha256"]:
            fail("Original monitor/ENGINE group changed")
        value = read(path)
        if (
            value["group_id"] != group["group_id"]
            or digest(value["records"]) != value["records_sha256"]
        ):
            fail("Monitor/ENGINE original content hash differs")
        if len(value["records"]) != len(group["slots"]):
            fail("Incomplete monitor/ENGINE group")
        for index, (row, slot) in enumerate(zip(value["records"], group["slots"], strict=True)):
            request = store.model_input(slot["qid"], protocol=slot["protocol"])
            if slot["view"] is not None:
                request = api.evaluation.diagnostic_input(tasks[slot["qid"]], request, slot["view"])
            if row["model_input"] != request:
                fail("Monitor/ENGINE actual prompt differs")
            raw_fields(row)
            api.evaluation.validate_record(
                row,
                slot,
                input_hash=digest(request),
                model_identity=row["model_identity"],
                adapter_identity=row["adapter_identity"],
                source_sha256=sha_file(root / "RUN_IDENTITY.json"),
                index=index,
                count=len(value["records"]),
            )
        qid = group["slots"][0]["qid"]
        results.append((path, dict(task_sidecar=tasks[qid]), len(value["records"])))
    return results


def slot_revision_kwargs(root):
    """Reconstruct the selected generation identities, retaining legacy call shapes."""
    path = Path(root) / "plan/config/SR_F3_CE.json"
    revision = read(path).get("protocol_revision", "v1") if path.is_file() else "v1"
    if revision not in {"v1", "v2", "c2"}:
        fail("Unknown protocol revision while reconstructing delivery slots")
    return {} if revision == "v1" else {"revision": revision}


def collect_panels(root, api, dataset, store):
    monitor, engine = [], []
    revision = slot_revision_kwargs(root)
    parents = {p["parent_id"]: p for p in read(root / "parents/PARENT_AUDIT.json")["parents"]}
    for model in api.evaluation.endpoint_models():
        for step in (0,) if model["arm"] == "PARENT" else (16, 32, 48, 64):
            mid = model["model_id"]
            directory = root / "monitor" / mid / f"step-{step:02d}"
            policy = (
                parents[mid]["policy_hash"]
                if step == 0
                else read(root / "runs" / mid / "checkpoints" / f"commit-{step:02d}.json")[
                    "field_hashes"
                ]["parameters"]
            )
            complete = read(directory / "COMPLETE.json")
            coverage = read(directory / "COVERAGE.json")
            if any(complete.get(k) != v for k, v in coverage.items()):
                fail("Monitor completion differs from original coverage")
            monitor += collect_panel(
                root,
                api,
                store,
                dataset.tasks,
                api.evaluation.monitor_slots(dataset.tasks, mid, step, **revision),
                directory,
                policy,
            )
    slots = api.evaluation.engine_slots(dataset.tasks, **revision)
    for block, parent in enumerate(api.evaluation.PARENTS):
        engine += collect_panel(
            root,
            api,
            store,
            dataset.tasks,
            [s for s in slots if s["model_id"] == parent],
            root / "engine" / f"b{block}",
            parents[parent]["policy_hash"],
        )
    if sum(r[2] for r in monitor) != 26112 or sum(r[2] for r in engine) != 2880:
        fail("Complete MONITOR/ENGINE diagnostic coverage required")
    return monitor, engine


def protocol_revision_retention(root, api):
    """Verify separate original-v1 and ablation copies without adding science rows."""
    revision = slot_revision_kwargs(root).get("revision", "v1")
    if revision == "v1":
        return None
    guard = api.protocol.verify_run_source_revision(root)
    if guard.get("protocol_revision") != revision:
        fail("Retention source guard differs from selected protocol")
    selection_path = root / "PROTOCOL_SELECTION.json"
    selection = read(selection_path)
    if sha_file(selection_path) != guard["protocol_selection_sha256"] or (
        selection.get("outputs") != 2304 or len(selection.get("raw_evidence", [])) != 9
    ):
        fail("Selected protocol lacks exactly nine complete ablation cells")
    archive_name = "SR_F3_CE_PROTOCOL_STOP_EVIDENCE_FULL_UNDER400MB_20261016.tar.xz"
    history = root / "protocol_revision/history_v1"
    archive, receipt_path = history / archive_name, history / (archive_name + ".verification.json")
    frozen = dict(
        bytes=219281700, sha256="d67af2a3a00f44713c3bcb62d526f90f12e392f5fa0fe2d1872f54a3e1184d60"
    )
    check_file(archive, frozen)
    receipt = read(receipt_path)
    if (
        receipt.get("status") != "VERIFIED_FULL_LOSSLESS_REPACK_UNDER_400_DECIMAL_MB"
        or any(receipt.get("archive", {}).get(k) != v for k, v in frozen.items())
        or receipt.get("actual_unique_new_output_rows") != 3904
        or receipt.get("file_contents_changed") != 0
        or receipt.get("files_removed") != 0
        or receipt.get("scientific_experiment_complete") is not False
    ):
        fail("Historical v1 archive receipt differs from original full preservation")
    copied = root / "protocol_revision/ablation"
    if sha_file(copied / "PROTOCOL_SELECTION.json") != sha_file(selection_path):
        fail("Retained ablation selection differs from the adopted selection")
    records = [selection["ablation_manifest"]]
    seen, count = set(), 0
    for cell in selection["raw_evidence"]:
        if len(cell["raw_groups"]) != 64:
            fail("Ablation copy must retain 64 groups for each variant/parent cell")
        records.extend(
            [
                cell["complete"],
                cell["terminal"],
                *cell["raw_groups"],
                *cell.get("raw_artifacts", []),
            ]
        )
        if "allocation" in cell:
            records.append(cell["allocation"])
        completion = read(bounded(copied, cell["complete"]["path"]))
        allocation = bounded(
            copied, str(Path(cell["complete"]["path"]).with_name("ALLOCATION.json"))
        )
        if sha_file(allocation) != completion.get("allocation_sha256"):
            fail("Retained ablation allocation differs from completed cell")
        for raw in cell["raw_groups"]:
            value = read(bounded(copied, raw["path"]))
            rows = value.get("records", [])
            if len(rows) != 4 or digest(rows) != value.get("records_sha256"):
                fail("Retained ablation raw group coverage/hash differs")
            for row in rows:
                if not row.get("slot_id") or row["slot_id"] in seen:
                    fail("Retained ablation has missing/duplicate raw identities")
                seen.add(row["slot_id"])
                count += 1
    if count != 2304 or len({r["path"] for r in records}) != len(records):
        fail("Retained ablation record or source inventory differs")
    for record in records:
        check_file(bounded(copied, record["path"]), record)
    return dict(
        protocol_revision=revision,
        selection_sha256=sha_file(selection_path),
        selection_path="evidence/PROTOCOL_SELECTION.json",
        historical_v1=dict(
            unique_records=3904,
            archive_path="evidence/" + str(archive.relative_to(root)),
            archive_sha256=frozen["sha256"],
            archive_bytes=frozen["bytes"],
            receipt_path="evidence/" + str(receipt_path.relative_to(root)),
            receipt_sha256=sha_file(receipt_path),
            scientific_experiment_complete=False,
            scope="original stopped v1 record, unchanged historical decision label",
        ),
        prompt_ablation=dict(
            unique_records=count,
            cells=9,
            groups=576,
            original_root=guard["ablation_root"],
            selected_revision=selection["selected_revision"],
            counts=selection["counts"],
            denominators=selection["denominators"],
            descriptive_statistics=selection["descriptive_statistics"],
            selection_uses_only="value_parseable",
            artifact_root="evidence/protocol_revision/ablation",
            source_files=records,
        ),
        included_in_current_science_or_engine_denominators=False,
        extra_optimizer_steps=0,
    )


def historical_status(root):
    report = read(root / "prior/RAW_EXPORT.json")
    if (
        report.get("operation_completed") is not True
        or report.get("critical") is not False
        or report.get("blocks_new_science") is not False
        or report.get("status") not in ("COMPLETE", "MISSING")
    ):
        fail("Historical export lacks an explicit completed/noncritical status")
    manifest = read(bounded(root, report["file_manifest"]))
    for item in manifest["files"]:
        check_file(bounded(root, item["export_path"]), item)
    check_file(bounded(root, report["record_index"]["path"]), report["record_index"])
    return dict(
        status=report["status"],
        noncritical=True,
        missing=[item for item in report["items"] if item["status"] != "COMPLETE"],
        indexed_raw_records=report["indexed_raw_records"],
        copied_source_files=report.get("copied_source_files", "UNKNOWN"),
        original_report="evidence/prior/RAW_EXPORT.json",
    )


def _allocated_gpus(tres):
    entries = {}
    for token in tres.split(","):
        if token.startswith("gres/gpu"):
            key, separator, raw = token.partition("=")
            if not separator or not raw.isdigit() or key in entries:
                fail("Ambiguous actual GPU allocation TRES")
            entries[key] = int(raw)
    typed = sum(value for key, value in entries.items() if key.startswith("gres/gpu:"))
    if "gres/gpu" in entries and typed and entries["gres/gpu"] != typed:
        fail("Actual generic/typed GPU TRES disagree")
    return entries.get("gres/gpu", typed)


def scheduler_costs(root, *, execute=subprocess.run):
    """One read-only sacct query, authenticated against this run's actual job IDs."""
    registered = {}
    for path in sorted((root / "orchestration/jobs").glob("*/INTENT.json")):
        state = read(path.with_name("STATE.json"))
        intent = read(path)
        if state.get("intent_sha256") != sha_file(path) or state.get("key") != intent.get("key"):
            fail("Scheduler cost intent identity differs")
        job = state.get("job_id")
        if not isinstance(job, str) or not job.isdigit() or job in registered:
            fail("Unique actual registered scheduler job IDs required")
        terminal_path = path.with_name("TERMINAL.json")
        registered[job] = dict(
            intent=intent, terminal=read(terminal_path), terminal_sha256=sha_file(terminal_path)
        )
    columns = (
        "JobIDRaw",
        "State",
        "ExitCode",
        "User",
        "Account",
        "QOS",
        "JobName",
        "ElapsedRaw",
        "Submit",
        "Start",
        "End",
        "AllocTRES",
    )
    command = [
        "sacct",
        "-X",
        "-n",
        "-P",
        "-j",
        ",".join(sorted(registered)),
        "--format=JobIDRaw,State,ExitCode,User%100,Account%100,QOS%100,JobName%100,"
        "ElapsedRaw,Submit,Start,End,AllocTRES%500",
    ]
    raw = dict(
        command=command,
        requested_job_ids=sorted(registered),
        timezone="UTC",
        queried_at=datetime.now(timezone.utc).isoformat(),
    )
    if not registered:
        return dict(status="UNKNOWN", reason="No verified allocation IDs"), raw
    try:
        response = execute(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
            env=dict(os.environ, TZ="UTC", SLURM_TIME_FORMAT="standard"),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raw.update(error_type=type(exc).__name__, error=str(exc))
        return dict(status="UNKNOWN", reason="sacct unavailable: " + str(exc)), raw
    raw.update(returncode=response.returncode, stdout=response.stdout, stderr=response.stderr)
    if response.returncode:
        return dict(status="UNKNOWN", reason="sacct returned a nonzero exit"), raw
    fetched = {}
    for line in response.stdout.splitlines():
        fields = line.split("|")
        if len(fields) == len(columns) + 1 and fields[-1] == "":
            fields.pop()
        if len(fields) != len(columns):
            fail("sacct cost column inventory differs")
        item = dict(zip(columns, fields, strict=True))
        job = item["JobIDRaw"]
        if job not in registered or job in fetched:
            fail("Unregistered or duplicated scheduler accounting job")
        fetched[job] = item
    jobs, starts, ends = [], [], []
    for job, saved in registered.items():
        if job not in fetched:
            jobs.append(dict(job_id=job, status="UNKNOWN", reason="No sacct row retained"))
            continue
        item, terminal, intent = fetched[job], saved["terminal"], saved["intent"]
        observed = dict(
            job_id=job,
            state=item["State"].split()[0].rstrip("+"),
            exit_code=item["ExitCode"],
            user=item["User"],
            account=item["Account"],
            qos=item["QOS"],
            name=item["JobName"],
        )
        if any(terminal.get(k) != v for k, v in observed.items()) or (
            observed["qos"] != intent["qos"] or observed["name"] != intent["job_name"]
        ):
            fail("Live scheduler cost identity differs from INTENT/TERMINAL")
        try:
            elapsed = int(item["ElapsedRaw"])
            if elapsed < 0:
                raise ValueError("Negative elapsed time")
            submitted = datetime.fromisoformat(item["Submit"]).replace(tzinfo=timezone.utc)
            ended = datetime.fromisoformat(item["End"]).replace(tzinfo=timezone.utc)
            unstarted = item["Start"] in ("Unknown", "None", "N/A", "")
            if unstarted:
                if elapsed:
                    raise ValueError("Nonzero elapsed with unknown start")
                started, queue_kind = ended, "queued_until_terminal_without_start"
            else:
                started = datetime.fromisoformat(item["Start"]).replace(tzinfo=timezone.utc)
                queue_kind = "submit_to_actual_start"
            queue = (started - submitted).total_seconds()
            if queue < 0 or ended < started:
                raise ValueError("Invalid actual scheduler timeline")
            allocation = item["AllocTRES"].strip()
            if not allocation and not unstarted:
                raise ValueError("Actual allocation TRES unavailable for started job")
            gpus = _allocated_gpus(allocation)
            if intent.get("gpu_count", 0) > 0 and not gpus and not unstarted:
                raise ValueError("Actual GPU allocation missing for registered GPU job")
        except (ValueError, TypeError) as exc:
            jobs.append(dict(observed, status="UNKNOWN", reason=str(exc), raw=item))
            continue
        starts.append(submitted)
        ends.append(ended)
        jobs.append(
            dict(
                observed,
                status="VERIFIED",
                task_ids=intent["task_ids"],
                elapsed_seconds=elapsed,
                submit_utc=submitted.isoformat(),
                start_utc=None if unstarted else started.isoformat(),
                end_utc=ended.isoformat(),
                queue_seconds=queue,
                queue_kind=queue_kind,
                actual_allocated_gpus=gpus,
                gpu_hours=elapsed * gpus / 3600,
                allocation_tres=item["AllocTRES"],
                terminal_receipt_sha256=saved["terminal_sha256"],
            )
        )
    known = [row for row in jobs if row["status"] == "VERIFIED"]
    complete = len(known) == len(registered)
    totals = dict(
        gpu_hours=sum(row["gpu_hours"] for row in known),
        queue_seconds=sum(row["queue_seconds"] for row in known),
        allocation_elapsed_seconds_sum=sum(row["elapsed_seconds"] for row in known),
        total_wall_seconds=(max(ends) - min(starts)).total_seconds() if starts else 0,
    )
    success = [row for row in known if row["state"] == "COMPLETED"]
    failed = [row for row in known if row["state"] != "COMPLETED"]
    return dict(
        status="VERIFIED" if complete else "UNKNOWN",
        jobs=jobs,
        totals=totals if complete else {k: "UNKNOWN" for k in totals},
        verified_subtotals=totals,
        successful_attempt_gpu_hours=sum(row["gpu_hours"] for row in success),
        failed_attempt_gpu_hours=sum(row["gpu_hours"] for row in failed),
        scope="registered scientific/orchestration allocations, including failed predecessors",
        excludes="active auxiliary dispatcher and delivery process; no final lease accounting yet",
        queue_aggregation="sum of per-allocation waiting; concurrent waits can overlap",
        wall_definition="earliest actual Submit to latest actual End; not summed job elapsed",
    ), raw


def summarize_observation_events(path, *, execution_binding=None, installation_id=None):
    """Summarize actual innermost-scope hooks; never sum parent scope wall times."""
    pending, seen, categories, issues = {}, set(), {}, []
    identity_keys = (
        "installation_id",
        "runtime_id",
        "scope_id",
        "parent_scope_id",
        "kind",
        "purpose",
        "protocol",
        "pid",
    )
    stat_keys = {
        "entered",
        "returned",
        "raised",
        "unresolved",
        "checkpoint_early_stop",
        "input_batch_items",
        "input_sequence_slots",
        "input_element_slots",
        "shape_unknown_calls",
        "host_inclusive_ns",
        "host_exclusive_ns",
        "image_items",
        "image_shape_unknown_calls",
    }
    parsed, ended = 0, 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = loads(line)
            except json.JSONDecodeError:
                if line.endswith("\n") or stream.read(1):
                    fail("Malformed observation event before file end")
                issues.append("TRUNCATED_FINAL_EVENT")
                break
            if (
                row.get("schema_version") != 1
                or row.get("event") not in {"SCOPE_BEGIN", "SCOPE_END"}
                or any(key not in row for key in identity_keys)
                or any(
                    not isinstance(row[key], str) or not row[key]
                    for key in ("installation_id", "runtime_id", "scope_id")
                )
                or (
                    row["parent_scope_id"] is not None
                    and (not isinstance(row["parent_scope_id"], str) or not row["parent_scope_id"])
                )
                or type(row.get("time_unix_ns")) is not int
                or row["time_unix_ns"] < 0
                or type(row["pid"]) is not int
                or row["pid"] <= 0
            ):
                fail("Observation event schema or physical identity differs")
            if execution_binding is not None and row.get("execution_binding") != execution_binding:
                fail("Observation event differs from its physical execution binding")
            if installation_id is not None and row["installation_id"] != installation_id:
                fail("Observation event differs from its installed observer identity")
            parsed += 1
            key = tuple(row[k] for k in ("installation_id", "runtime_id", "scope_id"))
            if row["event"] == "SCOPE_BEGIN":
                if key in seen:
                    fail("Duplicate observation scope BEGIN")
                if row["parent_scope_id"] is not None and not any(
                    parent["installation_id"] == row["installation_id"]
                    and parent["scope_id"] == row["parent_scope_id"]
                    for parent in pending.values()
                ):
                    fail("Observation nested scope lacks its active parent BEGIN")
                seen.add(key)
                pending[key] = row
                continue
            begin = pending.pop(key, None)
            if begin is None or any(begin[k] != row[k] for k in identity_keys):
                fail("Observation END lacks its exact physical BEGIN identity")
            elapsed = row.get("host_scope_elapsed_ns")
            if (
                row.get("outcome")
                not in {"RETURNED", "RAISED", "INTERRUPTED", "CHECKPOINT_EARLY_STOP"}
                or type(elapsed) is not int
                or elapsed < 0
                or row["time_unix_ns"] < begin["time_unix_ns"]
                or row.get("coverage_status") not in {"COMPLETE", "UNKNOWN"}
                or not isinstance(row.get("coverage_issues"), list)
            ):
                fail("Observation END outcome/coverage/timing schema differs")
            ended += 1
            category_key = encoded([row["kind"], row["purpose"], row["protocol"]]).decode()
            category = categories.setdefault(
                category_key,
                dict(
                    kind=row["kind"],
                    purpose=row["purpose"],
                    protocol=row["protocol"],
                    ended_scopes=0,
                    outcomes=Counter(),
                    scope_host_elapsed_ns=dict(minimum=elapsed, maximum=elapsed),
                    model_stages={},
                    visual_stages={},
                    unknown_coverage_scopes=0,
                ),
            )
            category["ended_scopes"] += 1
            category["outcomes"][row["outcome"]] += 1
            for limit, function in (("minimum", min), ("maximum", max)):
                category["scope_host_elapsed_ns"][limit] = function(
                    category["scope_host_elapsed_ns"][limit], elapsed
                )
            if row["coverage_status"] == "UNKNOWN":
                category["unknown_coverage_scopes"] += 1
                issues.append("SCOPE_COVERAGE_UNKNOWN:" + row["scope_id"])
            for axis in ("model_stages", "visual_stages"):
                stages = row.get(axis)
                if not isinstance(stages, dict):
                    fail("Observation model/visual stage counters missing")
                for stage, stats in stages.items():
                    if (
                        not isinstance(stats, dict)
                        or stats.keys() != stat_keys
                        or any(type(value) is not int or value < 0 for value in stats.values())
                    ):
                        fail("Observation stage counters must be known nonnegative integers")
                    if stats["host_exclusive_ns"] > stats["host_inclusive_ns"]:
                        fail("Exclusive host hook timing exceeds its inclusive interval")
                    if stats.get("entered", 0) != sum(
                        stats.get(name, 0)
                        for name in ("returned", "raised", "unresolved", "checkpoint_early_stop")
                    ):
                        fail("Observation entered/returned/failure counter conservation differs")
                    if stats.get("unresolved", 0):
                        issues.append(
                            "UNRESOLVED_HOOK:" + row["scope_id"] + ":" + axis + ":" + stage
                        )
                    aggregate = category[axis].setdefault(stage, Counter())
                    aggregate.update(stats)
            model_calls = sum(stats["entered"] for stats in row["model_stages"].values())
            visual_calls = sum(stats["entered"] for stats in row["visual_stages"].values())
            if (
                row["coverage_status"] == "COMPLETE"
                and row["outcome"] == "RETURNED"
                and row["kind"] in {"generation", "teacher_forcing"}
                and (not model_calls or (row.get("visual_expected") is True and not visual_calls))
            ):
                fail("Complete observed scope cannot replace missing model/visual hooks with zero")
    if pending:
        issues.append("SCOPES_WITHOUT_END")
    return dict(
        status="UNKNOWN" if issues else "COMPLETE_RECORDED_SCOPES",
        parsed_events=parsed,
        begun_scopes=len(seen),
        ended_scopes=ended,
        incomplete_scopes=[{k: value[k] for k in identity_keys} for value in pending.values()],
        coverage_issues=issues,
        by_kind_purpose_protocol=list(categories.values()),
        count_attribution=(
            "hooks belong only to their innermost runtime scope; model_hook_layers not added"
        ),
        timing_scope=(
            "host callback timings, not GPU kernel time; parent scope wall times are not summed; "
            "model/visual axes overlap and must not be added"
        ),
        shape_scope=(
            "input/image slot counters cover known shapes only; unknown-shape counts retained"
        ),
    )


def observability_costs(root, identity, required_tasks=()):
    directory = root / "accounting/observability"
    attempts, complete_tasks = [], set()
    for attempt in sorted(directory.glob("*/*")):
        if not attempt.is_dir():
            continue
        if identity is None:
            fail("Observed costs require the explicitly verified frozen extension")
        job_key, task_id = attempt.parent.name, attempt.name
        job = root / "orchestration/jobs" / job_key
        intent, state = read(job / "INTENT.json"), read(job / "STATE.json")
        binding = dict(
            experiment_id=EXPERIMENT,
            run_identity_sha256=sha_file(root / "RUN_IDENTITY.json"),
            execution_freeze_sha256=sha_file(root / "EXECUTION_FREEZE.json"),
            extension_manifest_sha256=identity["manifest_sha256"],
            amendment_sha256=identity["amendment_sha256"],
            job_key=job_key,
            job_id=state["job_id"],
            task_id=task_id,
            claim_id=intent["claim_id"],
        )
        if (
            state.get("intent_sha256") != sha_file(job / "INTENT.json")
            or intent.get("key") != job_key
            or task_id not in intent["task_ids"]
            or intent.get("extension_manifest_sha256") != identity["manifest_sha256"]
            or intent.get("amendment_sha256") != identity["amendment_sha256"]
            or task_id == "p0"
            or task_id.startswith("engine--")
            or (required_tasks and task_id not in required_tasks)
        ):
            fail("Observation attempt is not a registered prospective execution")
        installed, final_path, events_path = (
            attempt / name for name in ("INSTALLATION.json", "FINAL.json", "events.jsonl")
        )
        raw_sources = {
            path: dict(
                path="evidence/" + str(path.relative_to(root)),
                sha256=sha_file(path),
                bytes=path.stat().st_size,
            )
            for path in (installed, final_path, events_path)
            if path.is_file()
        }
        observer = None
        if installed.is_file():
            installation = read(installed)
            observer = installation.get("observer", {})
            if (
                installation.get("schema_version") != 1
                or installation.get("status") != "INSTALLED"
                or installation.get("execution_binding") != binding
                or observer.get("observer_source_sha256") != identity["observer_source_sha256"]
                or observer.get("old_engine_observed") is not False
                or not observer.get("installation_id")
            ):
                fail("Observer installation identity differs from authenticated source/job")
        summary = (
            summarize_observation_events(
                events_path,
                execution_binding=binding,
                installation_id=observer["installation_id"] if observer else None,
            )
            if events_path.is_file()
            else dict(status="UNKNOWN", coverage_issues=["MISSING_EVENTS"])
        )
        result = dict(
            job_key=job_key,
            task_id=task_id,
            job_id=state["job_id"],
            scope="physical worker attempt",
            observation_summary=summary,
            events_path="evidence/" + str(events_path.relative_to(root)),
            status="UNKNOWN",
            missing_or_incomplete_final=True,
            raw_sources=list(raw_sources.values()),
        )
        if final_path.is_file():
            final = read(final_path)
            event_record, snapshot = final.get("events", {}), final.get("observer_snapshot", {})
            if (
                observer is None
                or summary["status"] != "COMPLETE_RECORDED_SCOPES"
                or final.get("schema_version") != 1
                or final.get("status") != "COMPLETE"
                or final.get("execution_binding") != binding
                or final.get("installation_sha256") != sha_file(installed)
                or event_record.get("path") != str(events_path.relative_to(root))
                or event_record.get("rows") != summary["parsed_events"]
                or event_record.get("scope_begin") != summary["begun_scopes"]
                or event_record.get("scope_end") != summary["ended_scopes"]
                or snapshot.get("installation_id") != observer["installation_id"]
                or snapshot.get("schema_version") != 1
                or snapshot.get("sealed") is not True
                or snapshot.get("active_scopes") != []
                or snapshot.get("write_errors") != []
                or snapshot.get("events_written", {}).get("SCOPE_BEGIN", 0)
                != summary["begun_scopes"]
                or snapshot.get("events_written", {}).get("SCOPE_END", 0) != summary["ended_scopes"]
                or final.get("task_receipt_sha256")
                != sha_file(root / "tasks" / task_id / "COMPLETE.json")
            ):
                fail("Final observation receipt does not certify its actual event stream/task")
            check_file(events_path, event_record)
            overhead = snapshot.get("observer_overhead_ns")
            if not isinstance(overhead, dict) or any(
                type(value) is not int or value < 0 for value in overhead.values()
            ):
                fail("Final observer overhead counters are invalid")
            result.update(
                status="COMPLETE_RECORDED_ATTEMPT",
                missing_or_incomplete_final=False,
                observer_overhead_ns=overhead,
                final_receipt_write_cost=snapshot.get("final_receipt_write_cost", "UNKNOWN"),
                final_path="evidence/" + str(final_path.relative_to(root)),
                final_sha256=sha_file(final_path),
            )
            complete_tasks.add(task_id)
        current_paths = {path for path in (installed, final_path, events_path) if path.is_file()}
        if set(raw_sources) != current_paths:
            fail("Observation attempt file set changed during delivery validation")
        for path, record in raw_sources.items():
            check_file(path, record)
        attempts.append(result)
    if set(required_tasks) - complete_tasks:
        fail("Successful future tasks lack their verified observation completion receipts")
    return dict(
        status=(
            "COMPLETE_RECORDED_ATTEMPTS"
            if attempts and all(row["status"] == "COMPLETE_RECORDED_ATTEMPT" for row in attempts)
            else "UNKNOWN"
        ),
        attempts=attempts,
        completed_task_ids=sorted(complete_tasks),
        old_engine="UNKNOWN: no retrospective hooks, timing reconstruction or engine replay",
        current_run_engine="UNKNOWN: engineering precedes this observation extension",
        scope="prospective worker hooks only; unfinished attempts remain UNKNOWN",
        units="nanoseconds of host observation, never GPU kernel time",
        nonadditivity=(
            "parent scope times, model/visual axes and model_hook_layers must not be added"
        ),
    )


def cost_summary(root, *, observation_identity=None, observation_task_ids=()):
    counts, per_task, forward_purposes = Counter(), defaultdict(Counter), defaultdict(Counter)
    for path in sorted((root / "accounting").glob("*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for row in json_lines(stream):
                if type(row["count"]) is not int or row["count"] < 0:
                    fail("Invalid physical cost record")
                counts[row["kind"]] += row["count"]
                per_task[row["task_id"]][row["kind"]] += row["count"]
                if row["kind"] == "extra_forward_sequences":
                    metadata = row["metadata"]
                    purpose = metadata["purpose"]
                    tokens = metadata["completion_tokens"]
                    if type(tokens) is not int or tokens < 0:
                        fail("Invalid recorded extra-forward completion-token cost")
                    forward_purposes[purpose].update(
                        sequences=row["count"], completion_tokens=tokens
                    )
    update_costs = Counter()
    for directory in (root / "runs", root / "engine/resume"):
        for path in sorted(directory.glob("*/update_attempts/*.json")):
            for key, value in read(path)["metrics"]["total_cost"].items():
                update_costs[key] += value
    scheduler, scheduler_raw = scheduler_costs(root)
    probes = []
    for path in sorted((root / "engine/resume/probes").glob("*.json")):
        value = read(path)
        if path.name == "COMPLETE.json":
            continue  # Its summary repeats the per-attempt receipts below.
        if "total_cost" in value or "cost" in value:
            probes.append(
                dict(
                    path="evidence/" + str(path.relative_to(root)),
                    sha256=sha_file(path),
                    status="FAILED" if path.name.endswith(".FAILURE.json") else "MEASURED",
                    cost=value.get("total_cost", value.get("cost")) or "UNKNOWN",
                    physical_updates=0,
                )
            )
    return dict(
        accounted_physical_counts=dict(counts),
        per_task={key: dict(value) for key, value in sorted(per_task.items())},
        recorded_extra_forward_by_purpose={
            key: dict(value) for key, value in sorted(forward_purposes.items())
        },
        extra_forward_scope=(
            "Recorded calls/completion tokens only; overlaps training/probe receipt costs; "
            "never add these two representations together"
        ),
        engine_probe_physical_attempts=probes,
        engine_probe_summary_not_added_twice="evidence/engine/resume/probes/COMPLETE.json",
        successful_update_receipt_costs_including_their_oom_retries=dict(update_costs),
        additional_failure_costs=(
            "UNKNOWN: retained raw incidents may overlap retry receipts; not double-summed"
        ),
        gpu_hours=scheduler.get("totals", {}).get("gpu_hours", "UNKNOWN"),
        queue_seconds=scheduler.get("totals", {}).get("queue_seconds", "UNKNOWN"),
        total_wall_seconds=scheduler.get("totals", {}).get("total_wall_seconds", "UNKNOWN"),
        scheduler=scheduler,
        independently_launched_cpu_preparation=dict(
            included_in_registered_attempt_totals=False,
            accounting_receipt=(
                "evidence/CPU_PREPARATION_ACCOUNTING.json"
                if (root / "CPU_PREPARATION_ACCOUNTING.json").is_file()
                else "UNKNOWN: separate preparation accounting not retained"
            ),
        ),
        active_dispatcher_and_delivery_cpu_cost="UNKNOWN: active process has no final accounting",
        prospective_observations=observability_costs(
            root, observation_identity, observation_task_ids
        ),
        image_encoding_time="UNKNOWN",
        image_encoder_call_count=(
            "UNKNOWN for the whole experiment; prospective per-attempt hook counts "
            "are reported separately"
        ),
        prefill_time="UNKNOWN",
        decode_time="UNKNOWN",
        generation_seconds_limitation=(
            "Per-answer value is cumulative within a batch and includes "
            "postprocessing/fsync; not summed or called decode time"
        ),
        technical_incident_files=len(list((root / "technical_incidents").glob("*.json"))),
        scheduler_failed_attempts=sum(
            read(p).get("status") == "FAILED"
            for p in (root / "orchestration/jobs").glob("*/STATE.json")
        ),
    ), scheduler_raw


class HashWriter:
    def __init__(self, stream):
        self.stream, self.hash, self.size = stream, hashlib.sha256(), 0

    def write(self, data):
        self.hash.update(data)
        self.size += len(data)
        return self.stream.write(data)

    def flush(self):
        return self.stream.flush()


class DeliveryZip:
    def __init__(self, path):
        self.archive = zipfile.ZipFile(path, "x", allowZip64=True)
        self.files = {}
        self.seen = set()

    @contextmanager
    def member(self, name, *, stored=False, metadata=None):
        safe_name(name)
        if name in self.seen:
            fail("Duplicate archive path: " + name)
        self.seen.add(name)
        info = zipfile.ZipInfo(name, date_time=(2026, 10, 16, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        with self.archive.open(info, "w", force_zip64=True) as raw:
            stream = HashWriter(raw)
            yield stream
        self.files[name] = dict(
            path=name, sha256=stream.hash.hexdigest(), bytes=stream.size, **(metadata or {})
        )

    def add_bytes(self, name, data):
        with self.member(name) as stream:
            stream.write(data)

    def add_json(self, name, value):
        self.add_bytes(name, encoded(value) + b"\n")

    def add_path(self, name, path, expected=None):
        with (
            self.member(
                name, stored=Path(name).suffix in {".gz", ".png", ".jpg", ".jpeg", ".webp", ".whl"}
            ) as stream,
            path.open("rb") as source,
        ):
            for chunk in iter(lambda: source.read(8 << 20), b""):
                stream.write(chunk)
        if expected and (
            self.files[name]["sha256"] != expected["sha256"]
            or self.files[name]["bytes"] != expected["bytes"]
        ):
            fail("Source changed during streaming archive creation: " + str(path))

    def add_jsonl_gzip(self, name, rows):
        count = 0
        with (
            self.member(name, stored=True) as stream,
            gzip.GzipFile(fileobj=stream, mode="wb", filename="", mtime=0) as compressed,
        ):
            for row in rows:
                compressed.write(encoded(row) + b"\n")
                count += 1
        self.files[name]["jsonl_rows"] = count
        return count

    def finish(self):
        manifest = encoded(dict(schema_version=1, files=list(self.files.values()))) + b"\n"
        self.add_bytes("MANIFEST.json", manifest)
        checksum = "".join(f"{item['sha256']}  {name}\n" for name, item in self.files.items())
        self.add_bytes("SHA256SUMS", checksum.encode())
        self.archive.close()


def wrapped_originals(root, entries):
    """One original file per JSONL row, with exact UTF-8 bytes and task sidecar.

    Original text is a JSON string, so newline/indentation bytes survive and the
    source's SHA-256 can be checked after unpacking. This stores each raw once.
    """
    for path, metadata, records in entries:
        raw = path.read_bytes()
        expected = metadata.get("_source_identity")
        if (
            not expected
            or len(raw) != expected["bytes"]
            or hashlib.sha256(raw).hexdigest() != expected["sha256"]
        ):
            fail("Aggregate original changed after its authenticated source validation")
        original = raw.decode("utf-8")
        value = loads(original)
        if "records" in value and len(value["records"]) != records:
            fail("Aggregate source record count changed")
        yield dict(
            source_path=str(path.relative_to(root)),
            source_sha256=hashlib.sha256(raw).hexdigest(),
            source_bytes=len(raw),
            records=records,
            original_utf8=original,
            **{key: value for key, value in metadata.items() if key != "_source_identity"},
        )


def freeze_collections(root, collections, receipts):
    identities = {}
    for receipt in receipts.values():
        for row in receipt["artifacts"]:
            if row["path"] in identities and identities[row["path"]] != row:
                fail("Conflicting immutable raw source receipts")
            identities[row["path"]] = row
    frozen = {}
    for name, entries in collections.items():
        frozen[name] = []
        for path, metadata, count in entries:
            relative = str(path.relative_to(root))
            if relative not in identities:
                fail("Raw source is not authenticated by a completed task receipt: " + relative)
            identity = identities[relative]
            check_file(path, identity)
            frozen[name].append((path, dict(metadata, _source_identity=identity), count))
    return frozen


def _validate_delivered_original(row, name, tasks):
    safe_name(row["source_path"])
    raw = row["original_utf8"].encode()
    if len(raw) != row["source_bytes"] or hashlib.sha256(raw).hexdigest() != row["source_sha256"]:
        fail("Delivered original UTF-8 bytes differ")
    original = loads(raw)
    sidecar = row.get("task_sidecar")
    if not isinstance(sidecar, dict) or not sidecar.get("qid"):
        fail("Delivered original task sidecar missing")
    if tasks is not None and tasks.get(sidecar["qid"]) != sidecar:
        fail("Delivered aggregate task sidecar differs from task catalogue")
    if name == "SUPPORT_MASKS.jsonl.gz":
        tokens = original.get("tokens")
        if (
            row["records"] != 1
            or not isinstance(tokens, list)
            or not tokens
            or not original.get("target")
            or original.get("qid") != sidecar["qid"]
            or len(original.get("answer_mask", [])) != len(tokens)
            or len(original.get("work_mask", [])) != len(tokens)
            or not original.get("mask_audit")
            or not isinstance(row.get("model_input", {}).get("prompt"), str)
        ):
            fail("Delivered support lacks actual teacher, tokens, masks or prompt")
    else:
        records = original.get("records")
        if (
            not isinstance(records, list)
            or not records
            or len(records) != row["records"]
            or digest(records) != original.get("records_sha256")
        ):
            fail("Delivered aggregate original row count/hash differs")
        for record in records:
            raw_fields(record)
            if record.get("qid") != sidecar["qid"]:
                fail("Delivered raw record differs from task sidecar")


def verify_zip(path):
    """CRC -> safe unique names -> payload/manifests -> gzip EOF -> entry links."""
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad:
            fail("ZIP CRC failure: " + bad)
        names = archive.namelist()
        if len(names) != len(set(names)):
            fail("Duplicate archive members")
        for name in names:
            safe_name(name)
        if any((item.external_attr >> 16) & 0o170000 == 0o120000 for item in archive.infolist()):
            fail("ZIP symlinks are forbidden")
        manifest = loads(archive.read("MANIFEST.json"))
        records = {row["path"]: row for row in manifest["files"]}
        if len(records) != len(manifest["files"]) or set(records) | {
            "MANIFEST.json",
            "SHA256SUMS",
        } != set(names):
            fail("ZIP manifest/member inventory differs")
        complete_delivery = "DELIVERY_COVERAGE.json" in records
        if complete_delivery and not records.keys() >= DELIVERY_MEMBERS:
            fail("Complete delivery is missing required original/sidecar members")
        sums = {}
        for line in archive.read("SHA256SUMS").decode().splitlines():
            sha, name = line.split("  ", 1)
            if name in sums:
                fail("Duplicate checksum entry")
            sums[name] = sha
        if set(sums) != set(records) | {"MANIFEST.json"}:
            fail("Checksum inventory differs")
        for name in sorted(sums):
            value, size = hashlib.sha256(), 0
            with archive.open(name) as stream:
                for chunk in iter(lambda: stream.read(8 << 20), b""):
                    value.update(chunk)
                    size += len(chunk)
            if value.hexdigest() != sums[name] or (
                name in records
                and (size != records[name]["bytes"] or value.hexdigest() != records[name]["sha256"])
            ):
                fail("ZIP payload or manifest hash differs: " + name)
        tasks = None
        if complete_delivery:
            tasks = {}
            with (
                archive.open("TASK_SIDECARS.jsonl.gz") as member,
                gzip.GzipFile(fileobj=member) as binary,
                io.TextIOWrapper(binary, encoding="utf-8") as stream,
            ):
                for row in json_lines(stream):
                    qid = row.get("qid")
                    if not qid or qid in tasks:
                        fail("Missing or duplicate delivered task identity")
                    tasks[qid] = row
            if not tasks:
                fail("Delivered task sidecar catalogue is empty")
        aggregate_counts = {}
        for name, record in records.items():
            if not name.endswith(".jsonl.gz"):
                continue
            count, observations = 0, 0
            with (
                archive.open(name) as member,
                gzip.GzipFile(fileobj=member) as binary,
                io.TextIOWrapper(binary, encoding="utf-8") as stream,
            ):
                for row in json_lines(stream):
                    count += 1
                    if name in AGGREGATES:
                        _validate_delivered_original(row, name, tasks)
                        observations += row["records"]
                    elif name == "ENDPOINT_RAW.jsonl.gz":
                        raw_fields(row)
                        if tasks is not None and row.get("qid") not in tasks:
                            fail("Endpoint raw task sidecar missing")
            if "jsonl_rows" in record and count != record["jsonl_rows"]:
                fail("Compressed JSONL row coverage differs")
            aggregate_counts[name] = dict(rows=count, observations=observations)
        if "DELIVERY_COVERAGE.json" in records:
            coverage = loads(archive.read("DELIVERY_COVERAGE.json"))
            expected = dict(
                verified_tasks=33,
                training_paths=12,
                evaluated_models=15,
                logical_updates=768,
                training_records=98304,
                support_examples=2304,
                monitor_records=26112,
                engine_diagnostic_records=2880,
                engine_new_rollouts=512,
                engine_replayed_rollouts=256,
                engine_physical_updates=6,
            )
            if any(coverage.get(k) != v for k, v in expected.items()):
                fail("Delivered fixed matrix coverage differs")
            if coverage.get("protocol_revision", "v1") != "v1":
                if (
                    coverage.get("protocol_revision") not in {"v2", "c2"}
                    or coverage.get("protocol_ablation_records") != 2304
                    or coverage.get("historical_v1_unique_records") != 3904
                    or "PROTOCOL_REVISION_RETENTION.json" not in records
                    or "WORK_OCCURRENCE_TRAJECTORY.json" not in records
                ):
                    fail("Revised delivery omits separate historical/ablation coverage")
                retained = loads(archive.read("PROTOCOL_REVISION_RETENTION.json"))
                if (
                    retained.get("included_in_current_science_or_engine_denominators") is not False
                    or retained.get("prompt_ablation", {}).get("unique_records") != 2304
                    or retained.get("historical_v1", {}).get("unique_records") != 3904
                ):
                    fail("Historical or ablation data was substituted for current science")
            actual = {
                "TRAINING_GROUPS.jsonl.gz": 98304,
                "SUPPORT_MASKS.jsonl.gz": 2320,
                "MONITOR_RAW.jsonl.gz": 26112,
                "ENGINE_RAW.jsonl.gz": 3392,
            }
            if any(
                aggregate_counts.get(name, {}).get("observations") != n
                for name, n in actual.items()
            ):
                fail("Delivered aggregate original coverage differs")
            if aggregate_counts.get("ENDPOINT_RAW.jsonl.gz", {}).get("rows") != coverage["records"]:
                fail("Delivered endpoint gzip coverage differs")
            n = coverage["chartqapro"]["image_count"]
            if type(n) is not int or not 0 <= n <= 256 or coverage["records"] != 229500 + 15 * n:
                fail("Delivered endpoint count differs from frozen external subset")
        for name in ("START_HERE.md", "REPORT_ZH.md"):
            text = archive.read(name).decode()
            for link in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
                target = link.split("#", 1)[0]
                if not target or safe_name(target) not in records:
                    fail("Unresolved delivered entry link: " + link)
        return dict(
            entries=len(names),
            payload_bytes=sum(r["bytes"] for r in records.values()),
            aggregate_counts=aggregate_counts,
            checks={
                key: "PASS"
                for key in (
                    "crc",
                    "safe_unique_names",
                    "payload_hashes",
                    "manifest_hashes",
                    "entry_links",
                    "gzip_read_to_eof",
                )
            },
        )


def verify_bound_archive(path, *, json_values, jsonl_values, sources, originals):
    """Bind an archive to independently revalidated source bytes, including on resume.

    Structural ZIP checks cannot establish scientific provenance. Call this only
    after the complete frozen-source gates and full production rescoring. Every
    raw aggregate row, sidecar, mapping and included original is compared exactly.
    """
    with zipfile.ZipFile(path) as archive:
        manifest = {row["path"]: row for row in loads(archive.read("MANIFEST.json"))["files"]}
        for name, expected in json_values.items():
            if loads(archive.read(name)) != expected:
                fail("Delivery JSON differs from verified source evidence: " + name)
        for name, expected_rows in jsonl_values.items():
            with (
                archive.open(name) as member,
                gzip.GzipFile(fileobj=member) as binary,
                io.TextIOWrapper(binary, encoding="utf-8") as stream,
            ):
                sentinel = object()
                for actual, expected in itertools.zip_longest(
                    json_lines(stream), expected_rows(), fillvalue=sentinel
                ):
                    if actual != expected:
                        fail("Delivery raw/sidecar differs from verified source evidence: " + name)
        snapshot = loads(archive.read("OPERATIONS_SNAPSHOT.json"))
        snapshot_sources = {}
        if snapshot.get("scope") != "immutable files present before delivery publication":
            fail("Unknown operations snapshot scope")
        for row in snapshot["included"]:
            name = "evidence/" + safe_name(row["path"])
            if not name.startswith("evidence/operations/") or name in snapshot_sources:
                fail("Invalid operations snapshot inventory")
            if name not in originals or originals[name][1] != row:
                fail("Snapshotted operation evidence changed or disappeared")
            snapshot_sources[name] = originals[name]
        bound_sources = {
            **{
                name: value
                for name, value in originals.items()
                if not name.startswith("evidence/operations/")
            },
            **snapshot_sources,
            **sources,
        }
        expected_names = DELIVERY_MEMBERS | bound_sources.keys()
        if set(manifest) != expected_names:
            fail("Delivery included-original inventory differs from verified source inventory")
        for name, (source, record) in bound_sources.items():
            check_file(source, record)
            if (
                manifest[name]["sha256"] != record["sha256"]
                or manifest[name]["bytes"] != record["bytes"]
            ):
                fail("Delivered original differs from source identity: " + name)


def verify_cost_snapshot(root, snapshot, raw, current):
    """Recompute archived accounting without replacing its historical query time."""
    for attempt in snapshot.get("prospective_observations", {}).get("attempts", []):
        for record in attempt["raw_sources"]:
            if not record["path"].startswith("evidence/accounting/observability/"):
                fail("Observation cost source is outside its explicit ledger")
            check_file(bounded(root, record["path"].removeprefix("evidence/")), record)
    dynamic = {"scheduler", "gpu_hours", "queue_seconds", "total_wall_seconds"}
    if {k: v for k, v in snapshot.items() if k not in dynamic} != {
        k: v for k, v in current.items() if k not in dynamic
    }:
        fail("Archived physical cost counts differ from original ledgers")

    def saved_query(command, **_kwargs):
        if command != raw.get("command"):
            fail("Archived scheduler cost query differs from registered job IDs")
        if "error" in raw:
            raise OSError(raw["error"])
        return SimpleNamespace(
            returncode=raw["returncode"], stdout=raw["stdout"], stderr=raw["stderr"]
        )

    scheduler, _ = scheduler_costs(root, execute=saved_query)
    live_jobs = {
        row["job_id"]: row
        for row in current["scheduler"].get("jobs", [])
        if row["status"] == "VERIFIED"
    }
    for row in scheduler.get("jobs", []):
        if (
            row["status"] == "VERIFIED"
            and row["job_id"] in live_jobs
            and row != live_jobs[row["job_id"]]
        ):
            fail("Archived scheduler accounting differs from current actual job evidence")
    if snapshot["scheduler"] != scheduler or any(
        snapshot[key] != scheduler.get("totals", {}).get(key, "UNKNOWN")
        for key in dynamic - {"scheduler"}
    ):
        fail("Archived scheduler totals differ from retained actual accounting output")


def helper_sources(manifest_path=None):
    script = Path(__file__).resolve()
    files = {}
    if manifest_path:
        manifest_path = Path(manifest_path).resolve(strict=True)
        original = read(manifest_path)
        if (
            original.get("experiment_id") != EXPERIMENT
            or original.get("schema_version") != 1
            or not isinstance(original.get("files"), dict)
        ):
            fail("Auxiliary deployment manifest schema differs")
        for name, expected in original["files"].items():
            path = bounded(manifest_path.parent, name)
            if sha_file(path) != expected:
                fail("Auxiliary source differs from operations manifest")
            files[name] = path
        if "build_delivery.py" not in files or files["build_delivery.py"] != script:
            fail("Operations manifest does not bind this delivery builder")
        original_sha = sha_file(manifest_path)
    else:
        files["build_delivery.py"] = script
        checkout = script.parents[2]
        for name in ("docs/sr_f3_ce/DELIVERY.md", "tests/sr_f3_ce/test_delivery.py"):
            if (checkout / name).is_file():
                files[name] = checkout / name
        original = dict(
            schema_version=1,
            experiment_id=EXPERIMENT,
            operation_commit="UNKNOWN",
            scientific_code_commit="from EXECUTION_FREEZE",
            files={name: sha_file(path) for name, path in files.items()},
        )
        original_sha = digest(original)
    identities = [
        dict(path=name, sha256=sha_file(path), bytes=path.stat().st_size)
        for name, path in files.items()
    ]
    if any(row["sha256"] != original["files"][row["path"]] for row in identities):
        fail("Auxiliary source changed during manifest validation")
    return files, dict(
        original_manifest=original,
        original_manifest_sha256=original_sha,
        files=identities,
    )


def original_inventory(root, receipts, aggregate_paths, endpoint_map):
    """All local evidence; explicitly classify model/PT/font and full source datasets."""
    known = {}
    for receipt in receipts.values():
        for item in receipt["artifacts"]:
            if item["path"] in known and known[item["path"]] != item:
                fail("Conflicting task artifact identities")
            known[item["path"]] = item
    included, omitted = {}, []
    directories = (
        "plan",
        "deployment",
        "parents",
        "prior",
        "native",
        "data",
        "external",
        "runtime_identity",
        "tasks",
        "accounting",
        "technical_incidents",
        "orchestration",
        "operations",
        "runs",
        "monitor",
        "engine",
        "release",
        "sealed/endpoint",
        "preflight_job",
        "native_job",
        "native_job_v2",
        "prior_job",
        "type_audit_job",
        "type_audit_inputs",
        "prior_inventory",
        "parent_audit_code",
        "native_audit_code",
        "native_audit_code_v2",
        "authorization",
        "protocol_revision",
    )
    paths = {p for p in root.glob("*.json") if p.is_file()}
    for directory in directories:
        paths.update(p for p in (root / directory).rglob("*") if p.is_file())
    for path in sorted(paths):
        relative = str(path.relative_to(root))
        bounded(root, relative)
        if (
            any(p.startswith(".") or p == "__pycache__" for p in Path(relative).parts)
            or path.suffix == ".pyc"
        ):
            continue
        if relative.startswith("operations/") and (
            path.name in {"QUEUE_SNAPSHOT.json", "PROGRESS.json", "STATE.json"}
            or (
                path.suffix.lower() in {".log", ".out", ".err"}
                and not path.with_name("TERMINAL.json").is_file()
            )
            or "pending" in path.name.lower()
        ):
            continue
        if (
            path in aggregate_paths
            or relative in endpoint_map
            or relative
            in {
                "release/ENDPOINT_RAW.jsonl.gz",
                "release/ENDPOINT_SCORES.jsonl",
            }
        ):
            continue
        item = known.get(relative) or dict(
            path=relative, bytes=path.stat().st_size, sha256=sha_file(path)
        )
        reason = (
            "SERVER_ONLY_MODEL_OR_FULL_PT"
            if path.suffix.lower() in WEIGHTS
            else "FONT_IDENTITY_ONLY_NO_DISTRIBUTION"
            if path.suffix.lower() in FONTS
            else "SERVER_ONLY_FULL_SOURCE_DATASET_NOT_SELECTED_TASKS"
            if path.suffix.lower() == ".parquet"
            else None
        )
        if path.suffix.lower() in {".zip", ".whl"}:
            with zipfile.ZipFile(path) as archive:
                if any(Path(name).suffix.lower() in FONTS for name in archive.namelist()):
                    fail("Nested archive contains fonts; cannot distribute: " + relative)
        # Success journals repeat exactly the delivered completed raw rows. Keep
        # all partial/different journals, including any unfavorable/failed attempt.
        if path.name.endswith(".pending.jsonl"):
            final = path.with_name(path.name.removesuffix(".pending.jsonl") + ".json")
            if final in aggregate_paths or str(final.relative_to(root)) in endpoint_map:
                original = read(final).get("records", [])
                with path.open(encoding="utf-8") as stream:
                    pending = list(json_lines(stream))
                if len(pending) == len(original) and all(
                    all(actual.get(k) == v for k, v in old.items())
                    for old, actual in zip(pending, original, strict=True)
                ):
                    reason = "REDUNDANT_SUCCESS_JOURNAL_ALL_CONTENT_INCLUDED_IN_RAW"
        if reason:
            omitted.append(dict(item, server_path=str(path.resolve()), reason=reason))
        else:
            included["evidence/" + relative] = (path, item)
    return included, omitted


def model_sources(root):
    environment = read(root / "prior/MODEL_ENVIRONMENT_IDENTITY.json")
    model = Path(environment["model_path"]).resolve(strict=True)
    boundary = model.parent.parent if model.parent.name == "snapshots" else model
    included, omitted = {}, []
    for item in environment["model_files"]:
        name = safe_name(item["name"])
        if "/" in name:
            fail("Base model source must be a frozen top-level filename")
        path = (model / name).resolve(strict=True)
        if not path.is_relative_to(boundary):
            fail("Base snapshot symlink escapes its model repository")
        check_file(path, item)
        if Path(name).suffix.lower() in WEIGHTS | FONTS:
            omitted.append(
                dict(item, server_path=str(path), reason="SERVER_ONLY_BASE_WEIGHT_OR_FONT")
            )
        else:
            included["model_source/base/" + name] = (path, item)
    return included, omitted


def source_files(root):
    files, omitted = {}, []
    for item in read(root / "deployment/SOURCE_MANIFEST.json")["files"]:
        path = bounded(root / "code", item["path"])
        if Path(item["path"]).suffix.lower() in WEIGHTS | FONTS:
            omitted.append(
                dict(item, server_path=str(path), reason="SOURCE_BINARY_OR_FONT_IDENTITY_ONLY")
            )
        else:
            files["scientific_source/" + item["path"]] = (path, item)
    return files, omitted


def _jsonl_rows(path):
    with path.open(encoding="utf-8") as stream:
        yield from json_lines(stream)


def bound_external_panel(root, pool):
    panels = read(root / "external/PANELS.json")
    item = panels[pool]
    freeze_path = bounded(root, safe_name(item["root"]) + "/FREEZE.json")
    if sha_file(freeze_path) != item["freeze_sha256"]:
        fail("External panel pointer/freeze identity differs")
    return freeze_path.parent, read(freeze_path)


def unresolved_types(root):
    for pool in ("CHARTQA", "CHARTQAPRO"):
        directory, freeze = bound_external_panel(root, pool)
        path = directory / "TYPE_REGISTRY.json"
        if not path.exists():
            if pool == "CHARTQAPRO" and freeze["status"] == "UNAVAILABLE":
                continue
            fail("External type registry missing")
        registry = read(path)
        if digest(registry) != freeze["type_registry_sha256"]:
            fail("Active type registry differs from its frozen panel")
        actual = [row for row in registry["rows"] if row["status"] == "UNRESOLVED"]
        if len(actual) != registry["unresolved"] or registry.get("outputs_unseen") is not True:
            fail("Type audit unresolved inventory differs")
        for row in actual:
            yield dict(pool=pool, source_registry=str(path.relative_to(root)), **row)


def verify_images_and_types(root, api, dataset, store, external_tasks, external_stores):
    mappings, seen = [], {}
    for qid, task in dict(dataset.tasks, **external_tasks).items():
        current = external_stores.get(task["pool"], store)
        request = current.model_input(qid)
        path = bounded(current.image_root, request["image_file"])
        if not path.is_relative_to(root):
            fail("Registered images must remain in the experiment copy for complete delivery")
        sha = request["image_sha256"]
        if path not in seen:
            if sha_file(path) != sha:
                fail("Original model-visible image changed")
            seen[path] = sha
        elif seen[path] != sha:
            fail("Conflicting image identity")
        mappings.append(
            dict(
                qid=qid,
                pool=task["pool"],
                sha256=sha,
                original_image_file=request["image_file"],
                delivered_path="evidence/" + str(path.relative_to(root)),
            )
        )
    for pool in external_stores:
        directory, freeze = bound_external_panel(root, pool)
        registry = read(directory / "TYPE_REGISTRY.json")
        if digest(registry) != freeze["type_registry_sha256"]:
            fail("Active external type registry hash differs")
        items = [external_tasks[row["qid"]] for row in registry["rows"]]
        reviews = [review for row in registry["rows"] for review in row["reviews"]]
        if api.external.build_type_registry(items, reviews, outputs_unseen=True) != registry:
            fail("Type registry does not match its retained independent review evidence")
    return mappings


def _publish_bytes(path, data):
    if path.exists():
        if path.read_bytes() != data:
            fail("Immutable published receipt differs: " + str(path))
        return
    temporary = path.with_name(path.name + f".pending-{time.time_ns()}-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()
    _sync_directory(path.parent)


def _sync_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _external_receipt(output, checked, provenance, coverage):
    if checked["checks"].get("raw_rescoring_chain") != "PASS":
        fail("A structural ZIP check cannot establish an actual raw rescoring chain")
    receipt = dict(
        schema_version=1,
        status="VERIFIED_DELIVERY",
        archive=dict(
            path=str(output),
            sha256=sha_file(output),
            bytes=output.stat().st_size,
            entries=checked["entries"],
        ),
        run_identity_sha256=provenance["run_identity_sha256"],
        scientific_source_manifest_sha256=provenance["scientific_source_manifest_sha256"],
        helper_manifest_sha256=provenance["helper_manifest_sha256"],
        coverage=coverage,
        checks=checked["checks"],
    )
    if "observability_extension" in provenance:
        receipt["observability_extension"] = provenance["observability_extension"]
    _publish_bytes(Path(str(output) + ".verification.json"), encoded(receipt) + b"\n")
    _publish_bytes(
        Path(str(output) + ".sha256"), f"{receipt['archive']['sha256']}  {output.name}\n".encode()
    )
    return receipt


def _build(root, output, *, helper_manifest=None, extension_manifest=None):
    api = (
        load_frozen_api(root)
        if extension_manifest is None
        else load_frozen_api(root, extension_manifest=extension_manifest)
    )
    receipts = require_final_release(root, api)
    helpers, helper_receipt = helper_sources(helper_manifest)
    provenance = dict(
        schema_version=1,
        experiment_id=EXPERIMENT,
        run_identity_sha256=sha_file(root / "RUN_IDENTITY.json"),
        scientific_source_manifest_sha256=sha_file(root / "deployment/SOURCE_MANIFEST.json"),
        scientific_code_commit=read(root / "RUN_IDENTITY.json")["code_commit"],
        execution_freeze_sha256=sha_file(root / "EXECUTION_FREEZE.json"),
        helper_manifest_sha256=helper_receipt["original_manifest_sha256"],
        helper_builder_sha256=sha_file(Path(__file__).resolve()),
        completed_task_receipts={
            task: sha_file(root / "tasks" / task / "COMPLETE.json") for task in receipts
        },
    )
    if getattr(api, "protocol_source_guard", None) is not None:
        provenance["protocol_source_guard"] = api.protocol_source_guard
    if getattr(api, "observability", None) is not None:
        provenance["observability_extension"] = api.observability.identity
    dataset, store = api.runner.dataset_and_store(root)
    external_tasks, external_stores, panels, external_scorer = api.endpoint_runner._external(root)
    image_map = verify_images_and_types(root, api, dataset, store, external_tasks, external_stores)
    tasks = dict(dataset.tasks, **external_tasks)
    slots = api.evaluation.endpoint_slots(
        dataset.tasks, dataset.runs, external_tasks, **slot_revision_kwargs(root)
    )
    coverage = verify_release_stream(
        root,
        api.endpoint_runner._raw_groups(root, slots),
        tasks,
        lambda row, task: api.evaluation.score_record(row, task, external_scorer=external_scorer),
    )
    training, supports = collect_training(root, api, dataset, store)
    monitor, engine = collect_panels(root, api, dataset, store)
    # The ENGINE cached batches are distinct from the 2,880 diagnostic outputs;
    # replay uses these same 512 originals and never creates another 256 answers.
    resume_dir = root / "engine/resume/continuous"
    engine_state, _ = api.checkpoint.read_checkpoint(resume_dir / "checkpoints", step=4)
    api.runner.verify_committed_records(
        resume_dir, engine_state, api.runner.engine_schedule(dataset.tasks), dataset.tasks, store
    )
    engine_cached = []
    for index, path in enumerate(
        sorted((resume_dir / "rollouts").glob("[0-9][0-9]-[0-9][0-9].json"))
    ):
        row = read(path)
        if len(row["records"]) != 8 or digest(row["records"]) != row["records_sha256"]:
            fail("ENGINE cached raw coverage/hash differs")
        for raw in row["records"]:
            raw_fields(raw)
        qid = engine_state["used_task_qids"][index]
        engine_cached.append(
            (path, dict(task_sidecar=dataset.tasks[qid], role="ENGINE_CACHED_UPDATE"), 8)
        )
    if len(engine_cached) != 64:
        fail("ENGINE must retain exactly 512 unique update rollouts")
    for step in range(1, 5):
        for qid in engine_state["used_support_qids"][(step - 1) * 4 : step * 4]:
            path = resume_dir / "support" / f"{step:02d}-{qid}.json"
            supports.append(
                (
                    path,
                    dict(
                        run_id="ENGINE",
                        task_sidecar=dataset.tasks[qid],
                        model_input=store.model_input(qid),
                        role="ENGINE_SUPPORT",
                    ),
                    1,
                )
            )
    coverage.update(
        verified_tasks=len(receipts),
        training_paths=12,
        evaluated_models=15,
        logical_updates=768,
        training_records=98304,
        support_examples=2304,
        monitor_records=26112,
        engine_diagnostic_records=2880,
        engine_new_rollouts=512,
        engine_replayed_rollouts=256,
        engine_physical_updates=6,
        chartqapro=panels["CHARTQAPRO"],
        all_unique_raw_included=True,
        protocol_revision=slot_revision_kwargs(root).get("revision", "v1"),
        raw_scope="current selected run only; historical v1 and prompt ablation excluded",
    )
    analysis = _validate_analysis(read(root / "release/PRIMARY_ANALYSIS.json"))
    external, prefix = (
        read(root / "release/EXTERNAL_ANALYSIS.json"),
        read(root / "release/PREFIX_DIAGNOSTICS.json"),
    )
    prior = historical_status(root)
    revision_retention = protocol_revision_retention(root, api)
    if revision_retention is not None:
        coverage.update(protocol_ablation_records=2304, historical_v1_unique_records=3904)
    signals = training_signal_index(root, dataset.runs)
    work_trajectory = work_occurrence_trajectory(root, api, dataset, training, monitor)
    extension = getattr(api, "observability", None)
    costs, scheduler_cost_raw = cost_summary(
        root,
        observation_identity=extension.identity if extension else None,
        observation_task_ids=tuple(
            task for task in receipts if task != "p0" and not task.startswith("engine--")
        )
        if extension
        else (),
    )
    report = render_report(
        analysis,
        external,
        prefix,
        coverage,
        costs,
        prior,
        read(root / "release/RELEASE.json")["external_status"],
        signals,
        work_trajectory,
    )
    collections = dict(
        zip(AGGREGATES, (training, supports, monitor, engine + engine_cached), strict=True)
    )
    collections = freeze_collections(root, collections, receipts)
    aggregate_paths = {path for entries in collections.values() for path, _, _ in entries}
    endpoint_map = {}
    for model in api.evaluation.endpoint_models():
        for item in read(root / "sealed/endpoint" / model["model_id"] / "RAW_MANIFEST.json")[
            "raw_groups"
        ]:
            endpoint_map[item["path"]] = dict(item, delivered_in="ENDPOINT_RAW.jsonl.gz")
    originals, omissions = original_inventory(root, receipts, aggregate_paths, endpoint_map)
    models, model_omissions = model_sources(root)
    code, code_omissions = source_files(root)
    server_only = dict(
        weights_and_full_pt_not_included=True,
        fonts_not_distributed=True,
        no_unique_raw_omitted=True,
        files=omissions + model_omissions + code_omissions,
        note=(
            "Redundant successful journals contain only already-delivered raw fields; "
            "all partial/different journals retained."
        ),
    )
    manifest = {r["path"]: r for r in read(root / "release/MANIFEST.json")["files"]}
    helper_records = {row["path"]: row for row in helper_receipt["files"]}
    sources = {
        **models,
        **code,
        **{
            name: (root / "release" / name, manifest[name])
            for name in ("ENDPOINT_RAW.jsonl.gz", "ENDPOINT_SCORES.jsonl")
        },
        **{"helper/source/" + name: (path, helper_records[name]) for name, path in helpers.items()},
    }
    mappings = list(endpoint_map.values())
    for name, entries in collections.items():
        mappings.extend(
            dict(
                path=str(path.relative_to(root)),
                sha256=metadata["_source_identity"]["sha256"],
                delivered_in=name,
                jsonl_row=i,
            )
            for i, (path, metadata, _) in enumerate(entries, 1)
        )
    json_values = {
        "DELIVERY_PROVENANCE.json": provenance,
        "DELIVERY_COVERAGE.json": coverage,
        "HELPER_SOURCE_MANIFEST.json": helper_receipt,
        "SERVER_ONLY.json": server_only,
        "TRAINING_SIGNAL_INDEX.json": signals,
    }
    if revision_retention is not None:
        json_values["PROTOCOL_REVISION_RETENTION.json"] = revision_retention
        json_values["WORK_OCCURRENCE_TRAJECTORY.json"] = work_trajectory
    jsonl_values = {
        **{
            name: lambda entries=entries: wrapped_originals(root, entries)
            for name, entries in collections.items()
        },
        "TASK_SIDECARS.jsonl.gz": lambda: (tasks[qid] for qid in sorted(tasks)),
        "IMAGE_SOURCE_MAP.jsonl.gz": lambda: iter(image_map),
        "UNRESOLVED_TYPES.jsonl.gz": lambda: unresolved_types(root),
        "RAW_SOURCE_MAP.jsonl.gz": lambda: iter(mappings),
    }

    def verify_complete(path):
        checked = verify_zip(path)
        verify_bound_archive(
            path,
            json_values=json_values,
            jsonl_values=jsonl_values,
            sources=sources,
            originals=originals,
        )
        with zipfile.ZipFile(path) as archive:
            archived_costs = loads(archive.read("COSTS.json"))
            verify_cost_snapshot(
                root, archived_costs, loads(archive.read("SCHEDULER_COST_RAW.json")), costs
            )
            expected_report = render_report(
                analysis,
                external,
                prefix,
                coverage,
                archived_costs,
                prior,
                read(root / "release/RELEASE.json")["external_status"],
                signals,
                work_trajectory,
            )
            if archive.read("REPORT_ZH.md") != expected_report.encode():
                fail("Chinese report differs from the actual released analysis and cost snapshot")
        checked["checks"]["raw_rescoring_chain"] = "PASS"
        return checked

    if output.exists():
        return _external_receipt(output, verify_complete(output), provenance, coverage)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".pending-{time.time_ns()}-{os.getpid()}")
    writer = DeliveryZip(temporary)
    try:
        writer.add_json("DELIVERY_PROVENANCE.json", provenance)
        writer.add_json("DELIVERY_COVERAGE.json", coverage)
        writer.add_json("HELPER_SOURCE_MANIFEST.json", helper_receipt)
        writer.add_json("TRAINING_SIGNAL_INDEX.json", signals)
        writer.add_json("SERVER_ONLY.json", server_only)
        if revision_retention is not None:
            writer.add_json("PROTOCOL_REVISION_RETENTION.json", revision_retention)
            writer.add_json("WORK_OCCURRENCE_TRAJECTORY.json", work_trajectory)
        writer.add_json("COSTS.json", costs)
        writer.add_json("SCHEDULER_COST_RAW.json", scheduler_cost_raw)
        writer.add_json(
            "OPERATIONS_SNAPSHOT.json",
            dict(
                scope="immutable files present before delivery publication",
                after_builder_completion_receipts="NOT_YET_CREATED_BY_ACTIVE_DISPATCHER",
                excluded_mutable_files=[
                    "QUEUE_SNAPSHOT.json",
                    "PROGRESS.json",
                    "STATE.json",
                    "*.log/*.out/*.err without adjacent TERMINAL.json",
                    "locks",
                ],
                included=[
                    item
                    for name, (_, item) in originals.items()
                    if name.startswith("evidence/operations/")
                ],
            ),
        )
        writer.add_bytes("REPORT_ZH.md", report.encode())
        writer.add_bytes(
            "START_HERE.md",
            (
                "# SR-F3-CE 可重评分交付\n\n"
                "先读 [中文事实报告](REPORT_ZH.md)，再核对 [实际覆盖](DELIVERY_COVERAGE.json)、"
                "[来源](DELIVERY_PROVENANCE.json)、[成本](COSTS.json)、[服务器保留项](SERVER_ONLY.json)。\n\n"
                "原文：[全部端点](ENDPOINT_RAW.jsonl.gz)、[端点评分](ENDPOINT_SCORES.jsonl)、"
                "[训练分组](TRAINING_GROUPS.jsonl.gz)、[支持字段](SUPPORT_MASKS.jsonl.gz)、"
                "[MONITOR](MONITOR_RAW.jsonl.gz)、[ENGINE](ENGINE_RAW.jsonl.gz)。\n\n"
                "四个聚合 gzip 的每行含 original_utf8（精确保留原文件文本）、source_sha256、"
                "records 和 task_sidecar；json.loads(row['original_utf8']) 恢复原对象，"
                "对该字符串 UTF-8 字节计算 SHA-256 可复核原件身份。端点 gzip 原样保留。"
                "全部题目 sidecar 位于 [TASK_SIDECARS.jsonl.gz](TASK_SIDECARS.jsonl.gz)。\n\n"
                "科学源码在 scientific_source，独立交付辅助源码在 helper/source，"
                "两套 manifest 都保留。"
                "既有文档中的服务器绝对路径仅是来源位置；[原始路径映射](RAW_SOURCE_MAP.jsonl.gz)"
                "和 [SHA256 清单](SHA256SUMS) 定位实际交付内容。权重和完整 PT 不入包，字体不分发。"
                "本包可重评分，不能称包含权重的重训包；"
                "外部 UNAVAILABLE 与历史 MISSING 保持原标签。\n"
            ).encode(),
        )
        for name in ("ENDPOINT_RAW.jsonl.gz", "ENDPOINT_SCORES.jsonl"):
            writer.add_path(name, root / "release" / name, manifest[name])
        for name, entries in collections.items():
            writer.add_jsonl_gzip(name, wrapped_originals(root, entries))
        writer.add_jsonl_gzip("TASK_SIDECARS.jsonl.gz", (tasks[qid] for qid in sorted(tasks)))
        writer.add_jsonl_gzip("IMAGE_SOURCE_MAP.jsonl.gz", iter(image_map))
        writer.add_jsonl_gzip("UNRESOLVED_TYPES.jsonl.gz", unresolved_types(root))
        writer.add_jsonl_gzip("RAW_SOURCE_MAP.jsonl.gz", iter(mappings))
        for name, (path, expected) in sorted({**originals, **models, **code}.items()):
            writer.add_path(name, path, expected)
        for name, path in helpers.items():
            writer.add_path("helper/source/" + name, path, helper_records[name])
        writer.finish()
        checked = verify_complete(temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, output)  # exclusive atomic publication; never overwrite
        temporary.unlink()
        _sync_directory(output.parent)
        return _external_receipt(output, checked, provenance, coverage)
    finally:
        writer.archive.close()


def build(root, output, *, helper_manifest=None, extension_manifest=None):
    root, output = Path(root).resolve(strict=True), Path(output).resolve()
    if output.suffix != ".zip" or (
        output.is_relative_to(root) and not output.is_relative_to(root / "delivery")
    ):
        fail("Use a new .zip output outside sources, or below runroot/delivery")
    try:
        return _build(
            root, output, helper_manifest=helper_manifest, extension_manifest=extension_manifest
        )
    except BaseException as exc:
        output.parent.mkdir(parents=True, exist_ok=True)
        _publish_bytes(
            Path(str(output) + f".failure-{time.time_ns()}.json"),
            encoded(
                dict(
                    status="DELIVERY_FAILED",
                    error_type=type(exc).__name__,
                    error=str(exc),
                    retained_temporary=[
                        str(p)
                        for p in sorted(output.parent.glob(output.name + ".pending-*"))
                        if p.is_file()
                    ],
                    published=output.is_file(),
                    published_archive_sha256=sha_file(output) if output.is_file() else None,
                    verification_receipt_exists=Path(str(output) + ".verification.json").is_file(),
                )
            )
            + b"\n",
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--helper-manifest", type=Path)
    parser.add_argument(
        "--extension-manifest",
        type=Path,
        help="Must equal the unique extension manifest bound by the scientific freeze",
    )
    args = parser.parse_args(argv)
    result = build(
        args.root,
        args.output,
        helper_manifest=args.helper_manifest,
        extension_manifest=args.extension_manifest,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
