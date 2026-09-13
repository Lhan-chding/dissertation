#!/usr/bin/env python3
"""Build an exhaustive, lossless small-artifact Markdown record. No remote I/O."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import html
import io
import json
import re
import tempfile
from pathlib import Path

REF = "__complete_fact_record_reference__"
SCHEMA = "__field_schema__"
EXCLUDED_NAMES = {
    "samples.jsonl",
    "audit_rollouts.jsonl",
    "diagnostic_rollouts.jsonl",
    "fixed_reference_rollouts.json",
    "resume_probe.pt",
    "origin.pt",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
}
EXCLUDED_SUFFIXES = {
    ".pt",
    ".bin",
    ".safetensors",
    ".npz",
    ".npy",
    ".tar",
    ".gz",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".pdf",
}


class Number(str):
    """Keep the original JSON numeric literal, without binary-float rounding."""


def read_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            if key in {REF, SCHEMA}:
                raise ValueError(f"Reserved reference key in source: {key}")
            result[key] = value
        return result

    return json.loads(text, parse_float=Number, parse_int=Number, object_pairs_hook=unique)


def dump_json(value, indent=None, level=0, sort_keys=False):
    if isinstance(value, Number):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    if isinstance(value, dict):
        items = sorted(value.items()) if sort_keys else value.items()
        pairs = [
            json.dumps(k, ensure_ascii=False)
            + (": " if indent else ":")
            + dump_json(v, indent, level + 1, sort_keys)
            for k, v in items
        ]
        left, right = "{", "}"
    elif isinstance(value, list):
        pairs = [dump_json(v, indent, level + 1, sort_keys) for v in value]
        left, right = "[", "]"
    else:
        raise TypeError(type(value).__name__)
    if not pairs:
        return left + right
    if indent is None:
        return left + ",".join(pairs) + right
    pad = " " * indent
    return (
        left
        + "\n"
        + pad * (level + 1)
        + (",\n" + pad * (level + 1)).join(pairs)
        + "\n"
        + pad * level
        + right
    )


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return dump_json(value, sort_keys=True).encode("utf-8")


def fenced(text, language=""):
    fence = "`" * max(
        3, max((len(s) for s in text.split() if s and set(s) == {"`"}), default=0) + 1
    )
    while fence in text:
        fence += "`"
    return f"{fence}{language}\n{text.rstrip()}\n{fence}\n"


def cell(value):
    if isinstance(value, (dict, list)):
        value = dump_json(value)
    elif value is None:
        value = "null"
    escaped = (
        html.escape(str(value), quote=False)
        .replace("|", "&#124;")
        .replace("\r", "&#13;")
        .replace("\n", "<br>")
        .replace("\t", "&#9;")
    )
    return re.sub(r"^ +| +$", lambda match: "&#32;" * len(match.group()), escaped)


def table(headers, rows):
    return (
        "\n".join(
            [
                "| " + " | ".join(cell(v) for v in headers) + " |",
                "| " + " | ".join("---" for _ in headers) + " |",
                *("| " + " | ".join(cell(v) for v in row) + " |" for row in rows),
            ]
        )
        + "\n"
    )


def collect_subtrees(value, counts, samples, min_bytes=900):
    if not isinstance(value, (dict, list)):
        return
    content = canonical(value)
    if len(content) >= min_bytes:
        key = digest(content)
        counts[key] += 1
        samples.setdefault(key, value)
    for child in value.values() if isinstance(value, dict) else value:
        collect_subtrees(child, counts, samples, min_bytes)


def scalar_type(value):
    if isinstance(value, Number):
        return "number"
    if isinstance(value, bool):
        return "boolean"
    if value is None:
        return "null"
    if isinstance(value, str):
        return "string"
    raise ValueError("CSV aliases require scalar JSON values")


def restore_csv_alias(rows, descriptor):
    header, *values = rows
    group_key = descriptor["group_key"]
    field_types = descriptor["field_types"]
    if set(header) != {*field_types, group_key}:
        raise ValueError("CSV alias fields differ from the declared JSON schema")
    result = {group: [] for group in descriptor["groups"]}
    for values_row in values:
        row = dict(zip(header, values_row, strict=True))
        group = row.pop(group_key)
        restored = {}
        for key, value in row.items():
            kind = field_types[key]
            if kind == "number":
                parsed = read_json(value)
                if not isinstance(parsed, Number):
                    raise ValueError("CSV alias number is not a JSON numeric literal")
            elif kind == "boolean" and value in {"True", "False"}:
                parsed = value == "True"
            elif kind == "null" and value == "None":
                parsed = None
            elif kind == "string":
                parsed = value
            else:
                raise ValueError(f"CSV alias scalar does not match type {kind}")
            restored[key] = parsed
        result[group].append(restored)
    return result


def verify_rendered_document(output, records):
    """Independently parse the written Markdown and restore every structured value."""
    text = output.read_text(encoding="utf-8")
    anchors = list(re.finditer(r'<a id="([FJS]\d+)"></a>\n\n### [^\n]+\n', text))
    sections = {
        match.group(1): text[
            match.end() : anchors[index + 1].start() if index + 1 < len(anchors) else len(text)
        ]
        for index, match in enumerate(anchors)
    }

    def payload(identifier):
        match = re.search(
            r"^(`{3,})json\n(.*?)\n\1\s*$", sections[identifier], re.MULTILINE | re.DOTALL
        )
        if match is None:
            raise AssertionError(f"Missing structured payload in Markdown: {identifier}")
        return json.loads(match.group(2), parse_float=Number, parse_int=Number)

    definitions = {
        identifier: payload(identifier) for identifier in sections if identifier.startswith("J")
    }
    schemas = {
        identifier: payload(identifier) for identifier in sections if identifier.startswith("S")
    }

    def restore(value):
        if isinstance(value, dict):
            if set(value) == {REF}:
                return restore(definitions[value[REF]])
            if set(value) == {SCHEMA, "values"}:
                return {
                    key: restore(child)
                    for key, child in zip(schemas[value[SCHEMA]], value["values"], strict=True)
                }
            return {key: restore(child) for key, child in value.items()}
        if isinstance(value, list):
            return [restore(child) for child in value]
        return value

    def resolve_section(identifier):
        section_id = identifier
        visited = set()
        while True:
            if section_id in visited:
                raise AssertionError("Cyclic file-content alias in Markdown")
            visited.add(section_id)
            alias = re.search(r"^与 \[(F\d+)\]\(#\1\) 字节相同", sections[section_id], re.MULTILINE)
            if alias is None:
                break
            section_id = alias.group(1)
        return section_id

    def parsed_table(identifier):
        lines = [
            line
            for line in sections[resolve_section(identifier)].splitlines()
            if line.startswith("| ") and line.endswith(" |")
        ]
        rows = [
            [
                html.unescape(value.strip().replace("<br>", "\n"))
                for value in line[2:-2].split(" | ")
            ]
            for line in lines
        ]
        if len(rows) >= 2:
            del rows[1]
        return rows

    checked_json = checked_tables = checked_rows = checked_cells = checked_aliases = 0
    for index, record in enumerate(records):
        identifier = f"F{index + 1:04d}"
        section_id = resolve_section(identifier)
        if record["kind"] == "json":
            actual = restore(payload(section_id))
            if canonical(actual) != canonical(record["value"]):
                raise AssertionError(f"Written JSON differs from source: {record['path']}")
            checked_json += 1
        elif record["kind"] == "csv_equivalent_json":
            descriptor = payload(section_id)
            actual = restore_csv_alias(parsed_table(descriptor["csv_section"]), descriptor)
            if canonical(actual) != canonical(record["value"]):
                raise AssertionError(
                    f"Written CSV alias differs from source JSON: {record['path']}"
                )
            checked_aliases += 1
        elif record["kind"] in {"csv", "parquet"}:
            rows = parsed_table(section_id)
            expected = record["rows"]
            if len(rows) != len(expected):
                raise AssertionError(f"Written table row coverage differs: {record['path']}")
            for actual_row, expected_row in zip(rows, expected, strict=True):
                if len(actual_row) != len(expected_row):
                    raise AssertionError(f"Written table column coverage differs: {record['path']}")
                for actual, value in zip(actual_row, expected_row, strict=True):
                    if isinstance(value, (dict, list)):
                        parsed = json.loads(actual, parse_float=Number, parse_int=Number)
                        if canonical(restore(parsed)) != canonical(value):
                            raise AssertionError(
                                f"Written structured table cell differs: {record['path']}"
                            )
                    elif actual != ("null" if value is None else str(value)):
                        raise AssertionError(f"Written table value differs: {record['path']}")
                    checked_cells += 1
            checked_tables += 1
            checked_rows += max(0, len(rows) - 1)
    return {
        "status": "PASS",
        "json_files_restored": checked_json,
        "csv_alias_json_files_restored": checked_aliases,
        "tables_restored": checked_tables,
        "data_rows_restored": checked_rows,
        "cells_checked": checked_cells,
    }


def collect_files(source_root, work_dir, spec):
    records = []
    manifest_path = source_root / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    bindings = {item["relative_path"]: item for item in manifest.get("files", [])}
    stages = tuple(
        spec.get("stages", sorted({item["stage"] for item in manifest.get("files", [])}))
    )
    for path in sorted(source_root.rglob("*")) if source_root.exists() else []:
        if not path.is_file():
            continue
        relative = path.relative_to(source_root)
        stage = relative.parts[0] if relative.parts[0] in stages else "SOURCE_MANIFEST"
        raw = path.read_bytes()
        record = {
            "path": str(relative),
            "stage": stage,
            "bytes": len(raw),
            "sha256": digest(raw),
            "local_path": str(path.resolve()),
        }
        binding = bindings.get(str(relative))
        if binding:
            if record["bytes"] != binding["bytes"] or record["sha256"] != binding["sha256"]:
                raise ValueError(f"Transfer identity mismatch: {relative}")
            record["server_path"] = binding["server_path"]
            record["transfer_identity"] = "MATCH"
        if path.name in EXCLUDED_NAMES or path.suffix.lower() in EXCLUDED_SUFFIXES:
            record.update(
                kind="excluded",
                reason="原始逐输出缓存、tokenizer词表、图像、checkpoint、归档或张量；仅保留路径、大小和hash",
            )
        elif path.suffix.lower() == ".parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError:
                record.update(
                    kind="pending_parquet", reason="等待等值全部行列CSV或具备Parquet读取依赖"
                )
            else:
                data = pq.read_table(path)
                rows = [[row[name] for name in data.column_names] for row in data.to_pylist()]
                record.update(
                    kind="parquet", rows=[data.column_names, *rows], schema=str(data.schema)
                )
                if "audit_json" in data.column_names:
                    column = data.column_names.index("audit_json")
                    strings = [row[column] for row in rows]
                    for row in rows:
                        row[column] = read_json(row[column])
                    record["decoded_json_columns"] = ["audit_json"]
                    record["audit_json_source_string_sha256"] = [
                        digest(value.encode()) for value in strings
                    ]
        elif path.suffix.lower() == ".json":
            record.update(kind="json", value=read_json(raw.decode("utf-8")))
        elif path.suffix.lower() == ".csv":
            rows = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline="")))
            if rows and any(len(row) != len(rows[0]) for row in rows):
                raise ValueError(f"Ragged CSV: {path}")
            record.update(kind="csv", rows=rows)
        elif path.suffix.lower() in {".yaml", ".yml", ".toml", ".txt", ".log", ".sbatch"}:
            record.update(kind="text", text=raw.decode("utf-8"))
        elif path.suffix.lower() == ".jsonl":
            record.update(
                kind="json",
                value=[
                    read_json(line) for line in raw.decode("utf-8").splitlines() if line.strip()
                ],
            )
        elif path.suffix.lower() == ".md":
            record.update(
                kind="excluded",
                reason="叙述报告不作数值真源；以同阶段完整机器JSON/CSV和验收回执覆盖，避免复制解释建议",
            )
        else:
            record.update(kind="unknown", reason="未分类格式，最终发布前必须说明或纳入")
        records.append(record)
    missing_files = set(bindings) - {r["path"] for r in records}
    if missing_files:
        raise ValueError(f"Manifest files missing locally: {sorted(missing_files)}")
    by_path = {r["path"]: r for r in records}
    for equivalence in spec.get("json_csv_equivalences", []):
        structured = by_path.get(equivalence["json"])
        tabular = by_path.get(equivalence["csv"])
        group_key = equivalence["group_key"]
        if not structured or not tabular:
            continue
        header, *rows = tabular["rows"]
        expected = []
        field_types = {}
        for group, values in structured["value"].items():
            for value in values:
                if set(value) != set(header) - {group_key}:
                    raise ValueError(f"CSV/JSON fields differ: {structured['path']}")
                row = {**value, group_key: group}
                for key, scalar in value.items():
                    kind = scalar_type(scalar)
                    if key in field_types and field_types[key] != kind:
                        raise ValueError("CSV alias JSON field types differ between rows")
                    field_types[key] = kind
                expected.append([str(row[key]) for key in header])
        if expected != rows:
            raise ValueError(f"CSV/JSON equality failed: {structured['path']}")
        structured["kind"] = "csv_equivalent_json"
        structured["equivalent_csv"] = tabular["path"]
        structured["equivalence"] = "ALL_ROWS_FIELDS_NUMERIC_LEXEMES_MATCH"
        structured["equivalence_group_key"] = group_key
        descriptor = {
            "csv_section": f"F{records.index(tabular) + 1:04d}",
            "group_key": group_key,
            "field_types": field_types,
            "groups": list(structured["value"]),
        }
        if canonical(restore_csv_alias(tabular["rows"], descriptor)) != canonical(
            structured["value"]
        ):
            raise ValueError(f"CSV alias reconstruction failed: {structured['path']}")
        structured["reconstruction"] = descriptor
    supplement_paths = [work_dir / path for path in spec.get("receipts", [])]
    extra_list = work_dir / "additional_receipts.json"
    if extra_list.exists():
        supplement_paths += [Path(path) for path in json.loads(extra_list.read_text())]
    for path in supplement_paths:
        if not path.exists():
            raise ValueError(f"Expected supplementary fact source missing: {path}")
        raw = path.read_bytes()
        records.append(
            {
                "path": "SUPPLEMENT/" + str(path),
                "local_path": str(path.resolve()),
                "stage": "ACCEPTANCE_HISTORY",
                "bytes": len(raw),
                "sha256": digest(raw),
                "server_path": "本地既有回执：" + str(path),
                "kind": "json",
                "value": read_json(raw.decode("utf-8")),
            }
        )
    return records


def verify_completion_sources(work_dir, spec, receipt, records):
    requirements = spec.get("completion_sources")
    bindings = receipt.get("verified_sources")
    if not requirements or not isinstance(bindings, dict) or set(bindings) != set(requirements):
        raise ValueError("Finalization requires all configured acceptance source bindings")
    included = {record["local_path"]: record for record in records}
    for role, requirement in requirements.items():
        path = (work_dir / requirement["path"]).resolve()
        binding = bindings[role]
        raw = path.read_bytes()
        identity = {"path": str(path), "bytes": len(raw), "sha256": digest(raw)}
        if any(
            type(binding.get(key)) is not type(value) or binding.get(key) != value
            for key, value in identity.items()
        ):
            raise ValueError(f"Acceptance source binding differs: {role}")
        record = included.get(str(path))
        if (
            record is None
            or record["kind"] != "json"
            or record["sha256"] != identity["sha256"]
            or record["bytes"] != identity["bytes"]
        ):
            raise ValueError(f"Acceptance source is not fully included: {role}")
        checks = requirement.get("checks")
        if not checks:
            raise ValueError(f"Acceptance source requires configured result checks: {role}")
        value = json.loads(raw)
        for check in checks:
            observed = value
            for component in check["field"]:
                observed = observed[component]
            expected = check["equals"]
            if type(observed) is not type(expected) or observed != expected:
                raise ValueError(f"Acceptance source result differs: {role}: {check['field']}")


def build(work_dir, source_root, output, completion_receipt=None):
    spec_path = work_dir / "document_spec.json"
    spec = json.loads(spec_path.read_text()) if spec_path.exists() else {}
    records = collect_files(source_root, work_dir, spec)
    stages = tuple(
        spec.get(
            "stages",
            sorted(
                {
                    r["stage"]
                    for r in records
                    if r["stage"] not in {"SOURCE_MANIFEST", "ACCEPTANCE_HISTORY"}
                }
            ),
        )
    )
    present = {r["stage"] for r in records}
    missing = [stage for stage in stages if stage not in present]
    blockers = [r["path"] for r in records if r["kind"] in {"pending_parquet", "unknown"}]
    completed = completion_receipt is not None
    completion_text = None
    if completed:
        completion_text = completion_receipt.read_text()
        receipt = json.loads(completion_text)
        policy = spec.get(
            "completion_policy", {"status": "PASS", "all_stage_sources_verified": True}
        )
        if any(
            type(receipt.get(key)) is not type(value) or receipt.get(key) != value
            for key, value in policy.items()
        ):
            raise ValueError("Finalization requires the configured complete acceptance receipt")
        if missing or blockers:
            raise ValueError(f"Incomplete coverage: stages={missing}; files={blockers}")
        verify_completion_sources(work_dir, spec, receipt, records)
    counts, samples = collections.Counter(), {}
    schema_counts = collections.Counter()

    def count_schemas(value):
        if isinstance(value, dict):
            keys = tuple(sorted(value))
            if len(keys) >= 4 and len(json.dumps(keys, ensure_ascii=False)) >= 180:
                schema_counts[keys] += 1
            for child in value.values():
                count_schemas(child)
        elif isinstance(value, list):
            for child in value:
                count_schemas(child)

    for record in records:
        if record["kind"] == "json":
            collect_subtrees(record["value"], counts, samples)
            count_schemas(record["value"])
        elif record["kind"] in {"csv", "parquet"}:
            for row in record["rows"][1:]:
                for value in row:
                    if isinstance(value, (dict, list)):
                        collect_subtrees(value, counts, samples)
                        count_schemas(value)
    duplicate_keys = {key for key, count in counts.items() if count > 1}
    definitions, definition_ids = {}, {}
    schemas = {}

    def transform(value, skip=None):
        if isinstance(value, (dict, list)):
            content = canonical(value)
            key = digest(content) if len(content) >= 900 else None
            if key in duplicate_keys and key != skip:
                if key not in definition_ids:
                    definition_ids[key] = f"J{len(definition_ids) + 1:04d}"
                    definitions[definition_ids[key]] = None
                    definitions[definition_ids[key]] = transform(value, skip=key)
                return {REF: definition_ids[key]}
            if isinstance(value, dict):
                keys = tuple(sorted(value))
                if schema_counts[keys] > 1:
                    if keys not in schemas:
                        schemas[keys] = f"S{len(schemas) + 1:04d}"
                    return {
                        SCHEMA: schemas[keys],
                        "values": [transform(value[key]) for key in keys],
                    }
                return {key: transform(child) for key, child in value.items()}
            return [transform(child) for child in value]
        return value

    def restore(value):
        if isinstance(value, dict):
            if set(value) == {REF}:
                return restore(definitions[value[REF]])
            if set(value) == {SCHEMA, "values"}:
                keys = next(
                    keys for keys, identifier in schemas.items() if identifier == value[SCHEMA]
                )
                if len(keys) != len(value["values"]):
                    raise AssertionError("Shared field schema length differs from its values")
                return {
                    key: restore(child) for key, child in zip(keys, value["values"], strict=True)
                }
            return {key: restore(child) for key, child in value.items()}
        if isinstance(value, list):
            return [restore(child) for child in value]
        return value

    title = spec.get("title", "实验事实记录")
    parts = [
        f"# {title}\n",
        "状态：**最终验收及来源完整性已通过**。\n"
        if completed
        else "状态：**DRAFT/待来源到齐及完整验收**。本文件尚不是最终交付。\n",
    ]
    parts += [
        "本记录按阶段保留机器结果中的配置、环境、计数、群体、bank、candidate、曲线、置信区间、状态和证据身份；不添加新的效果解释或后续研究建议。`PASS`字段按其原始检查范围保留。\n",
        "## 数据表示与覆盖规则\n",
        "CSV按原字段及原十进制字符串逐行呈现；JSON数值保留原始数字字面量，"
        "不经二进制浮点重写。相同JSON子树仅定义一次，其他位置使用 `"
        + REF
        + "` 引用，定义完整保存在文末；可无损还原每一源JSON。相同文件内容只呈现一次，"
        "所有原路径仍列在清单。缺测、空值、警告及失败字段保留。"
        "配置中列出的CSV和JSON等价视图经逐行逐字段核对后仅展示一次；重建映射按文件记录。"
        "Parquet的audit_json字符串解码为完整JSON对象后引用共享定义，保留所有字段和数值，"
        "其原字符串hash另存覆盖回执。\n",
        "附录JSON的重复字段名使用共享表头表示：`"
        + SCHEMA
        + "` 指向文末S编号的字段列表，`values`依该列表顺序保留每个字段的完整值；"
        "这与表头一次、数据逐行的表示等价。J编号引用完整相同内容，S编号仅复用字段名，"
        "不能据S编号相同推断实验值相同。所有转换逐文件执行无损重构比对。\n",
        "原始逐输出completion/token序列、逐token概率、tokenizer词表、图像缓存、checkpoint、"
        "梯度张量及大型归档不嵌入；已提供的这类文件仅记录路径、大小和SHA-256。"
        "小型统计汇总、梯度范数及候选响应表属于结果，全部纳入。"
        "Parquet保留所有行列和schema，数值使用Python的可往返表示。叙述报告不作为机器数表替代。\n",
        *[paragraph + "\n" for paragraph in spec.get("preamble", [])],
        "## 阶段覆盖\n",
        table(
            ["阶段", "当前来源文件数", "覆盖状态"],
            [
                [
                    s,
                    sum(r["stage"] == s for r in records),
                    "已发现来源；验收见原回执" if s in present else "待接收",
                ]
                for s in stages
            ],
        ),
        "## 详细数表索引\n",
        table(
            ["阶段", "完整数表", "数据行", "字段数"],
            [
                [
                    r["stage"],
                    f"[F{index + 1:04d} · {r['path']}](#F{index + 1:04d})",
                    max(0, len(r["rows"]) - 1),
                    len(r["rows"][0]) if r["rows"] else 0,
                ]
                for index, r in enumerate(records)
                if r["kind"] in {"csv", "parquet"}
            ],
        ),
        (work_dir / spec["history_file"]).read_text() if spec.get("history_file") else "",
        "## 源文件清单\n",
        "本地来源根目录：`"
        + str(source_root)
        + "`。原服务器路径与传输身份按随附SOURCE_MANIFEST原字段保留。\n",
        table(
            ["阶段", "相对路径", "原服务器路径", "bytes", "SHA-256", "处理"],
            [
                [
                    r["stage"],
                    r["path"],
                    r.get("server_path", "本地传输清单"),
                    r["bytes"],
                    r["sha256"],
                    r["kind"],
                ]
                for r in records
            ],
        ),
    ]
    if missing or blockers:
        parts += [
            "待补齐项："
            + json.dumps(
                {"missing_stages": missing, "unrendered_files": blockers}, ensure_ascii=False
            )
            + "。\n"
        ]
    seen_content = {}
    coverage = []
    for stage in ("SOURCE_MANIFEST", *stages, "ACCEPTANCE_HISTORY"):
        subset = [r for r in records if r["stage"] == stage]
        if not subset:
            continue
        parts.append(f"## {stage}：完整机器记录\n")
        for record in subset:
            location = "F" + str(records.index(record) + 1).zfill(4)
            parts.append(f'<a id="{location}"></a>\n\n### {location} · {record["path"]}\n')
            identity = (record["kind"], record["sha256"])
            item = {
                key: value for key, value in record.items() if key not in {"value", "rows", "text"}
            }
            if identity in seen_content:
                target = seen_content[identity]
                parts.append(f"与 [{target}](#{target}) 字节相同；完整内容见该处。\n")
                item["same_content_as"] = seen_content[identity]
                coverage.append(item)
                continue
            seen_content[identity] = location
            if record["kind"] == "json":
                transformed = transform(record["value"])
                if canonical(restore(transformed)) != canonical(record["value"]):
                    raise AssertionError(f"JSON lossless reconstruction failed: {record['path']}")
                parts.append(fenced(dump_json(transformed), "json"))
                item["json_reconstruction"] = "EXACT_NUMERIC_LEXEMES_AND_FIELDS"
                item["canonical_sha256"] = digest(canonical(record["value"]))
            elif record["kind"] in {"csv", "parquet"}:
                rows = record["rows"]
                parts.append(
                    f"数据行：{max(0, len(rows) - 1)}；字段：{len(rows[0]) if rows else 0}。\n"
                )
                if record["kind"] == "parquet":
                    parts.append(fenced(record["schema"], "text"))
                    if record.get("decoded_json_columns"):
                        parts.append(
                            "`audit_json`列按其完整JSON结构呈现，重复结构引用文末共享定义。\n"
                        )
                if rows:
                    transformed_rows = []
                    for row in rows[1:]:
                        transformed_row = []
                        for value in row:
                            if isinstance(value, (dict, list)):
                                transformed = transform(value)
                                if canonical(restore(transformed)) != canonical(value):
                                    raise AssertionError(
                                        f"Table cell reconstruction failed: {record['path']}"
                                    )
                                transformed_row.append(transformed)
                            else:
                                transformed_row.append(value)
                        transformed_rows.append(transformed_row)
                    parts.append(table(rows[0], transformed_rows))
                item.update(data_rows=max(0, len(rows) - 1), columns=len(rows[0]) if rows else 0)
            elif record["kind"] == "csv_equivalent_json":
                parts.append(
                    "完整数值见 `"
                    + record["equivalent_csv"]
                    + "` 的全部行列，已逐行逐字段核对一致。重建规则：按 `"
                    + record["equivalence_group_key"]
                    + "` 依原行序分组，组内每行去掉分组字段；"
                    "保留原数值字面量与布尔值，未省略其他字段。字段类型与分组清单：\n"
                )
                parts.append(fenced(dump_json(record["reconstruction"]), "json"))
            elif record["kind"] == "text":
                parts.append(fenced(record["text"], "text"))
            else:
                parts.append("未嵌入内容，原因：" + record["reason"] + "。\n")
            coverage.append(item)
    parts.append("## 共享JSON子树：完整定义\n")
    parts.append("以下编号只用于消除重复，既不合并不同实验，也不删除源字段。\n")
    for identifier, value in definitions.items():
        parts.append(f'<a id="{identifier}"></a>\n\n### {identifier}\n')
        parts.append(fenced(dump_json(value), "json"))
    parts.append("## 共享字段schema：完整表头\n")
    for keys, identifier in schemas.items():
        parts.append(f'<a id="{identifier}"></a>\n\n### {identifier}\n')
        parts.append(fenced(json.dumps(keys, ensure_ascii=False), "json"))
    if completed:
        parts.append("## 最终验收与来源覆盖回执\n")
        parts.append(fenced(completion_text, "json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output.parent,
        prefix=output.name + ".",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write("\n".join(parts))
    try:
        rendered_verification = verify_rendered_document(temporary, records)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    validation = {
        "status": "COMPLETE" if completed else "DRAFT",
        "source_root": str(source_root),
        "output": str(output),
        "output_bytes": output.stat().st_size,
        "output_sha256": digest(output.read_bytes()),
        "source_count": len(records),
        "missing_stages": missing,
        "unrendered_files": blockers,
        "shared_json_subtrees": len(definitions),
        "shared_field_schemas": len(schemas),
        "file_coverage": coverage,
        "rendered_verification": rendered_verification,
        "note": "Rendering/source-coverage check only; does not replace experimental acceptance.",
    }
    (work_dir / "document_coverage.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {key: value for key, value in validation.items() if key != "file_coverage"},
            ensure_ascii=False,
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--completion-receipt", type=Path)
    args = parser.parse_args()
    work_dir = args.work_dir.resolve()
    source_root = args.source_root.resolve() if args.source_root else work_dir / "server_sources"
    build(work_dir, source_root, args.out.resolve(), args.completion_receipt)


if __name__ == "__main__":
    main()
