"""Read-only P0 export of released SR-F2 raw evidence and historical images.

This is a provenance exporter, not a scorer. Every exported raw group is bound
to the original released worker receipt. Missing old evidence is reported per
item and never repaired by generation or by changing a historical score.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

from .preflight import (
    _destination,
    _publish_temporary,
    bounded_file,
    copy_verified,
    file_hash,
    object_hash,
    read_json,
    write_once,
)

ENDPOINTS = tuple(
    f"SRF2_{arm}_s{seed}"
    for seed in (71001, 71002, 71003)
    for arm in ("EARLY", "J6", "DEC_PRIME", "ADAPT")
)
BASELINE = "SRF1_2_COMMON_ZERO"
DIAGNOSTIC_MODELS = (
    BASELINE,
    *(
        f"SRF1_2_{arm}_s{seed}"
        for seed in (71001, 71002, 71003)
        for arm in ("A", "J", "GATE", "DEC")
    ),
)
RAW_FIELDS = (
    "text",
    "raw_text",
    "tokens",
    "sampler_logprobs",
    "generation",
    "sampling_parameters",
    "finish_reason",
    "truncated",
    "prompt_tensor_hash",
    "input_hash",
    "image_hash",
    "completion_token_count",
)


class MissingEvidence(ValueError):
    """An old source is absent, malformed, or differs from its attested identity."""


def write_jsonl_once(root, relative, rows):
    path = _destination(root, relative)
    digest = hashlib.sha256()
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".index-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            for row in rows:
                data = (
                    json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
                ).encode()
                digest.update(data)
                stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            _publish_temporary(temporary, path, digest.hexdigest())
        finally:
            temporary.unlink(missing_ok=True)
    return {"path": str(relative), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def _missing(item, error, **fields):
    return dict(item=item, status="MISSING", reason=str(error), **fields)


class PriorExporter:
    def __init__(self, parent_root, output_root, authorization, registration):
        self.root = Path(parent_root).resolve(strict=True)
        self.output = Path(output_root).resolve(strict=True)
        if self.output.is_relative_to(self.root) or self.root.is_relative_to(self.output):
            raise PermissionError("The prior export must be independent of its source root")
        self.authorization, self.registration = authorization, registration
        self.files = {}
        self.records = []
        self.bound = dict(registration["bound_files"])

    def source(self, relative, checksum, *, json_value=True):
        """Read attested bytes, then copy exactly those bytes without replacing files."""
        try:
            path = bounded_file(self.root, relative)
        except FileNotFoundError as error:
            raise MissingEvidence(f"Source absent: {relative}") from error
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != checksum:
            raise MissingEvidence(
                f"Source hash mismatch: {relative}; expected={checksum}; actual={actual}"
            )
        exported = copy_verified(path, self.output, Path("prior/raw") / relative, checksum)
        self.files[relative] = dict(
            source_path=relative,
            source_absolute_path=str(self.root / relative),
            export_path=exported["path"],
            sha256=checksum,
            bytes=len(raw),
            source_line_count=len(raw.splitlines()),
        )
        if not json_value:
            return raw
        try:
            return json.loads(raw)
        except (ValueError, UnicodeError) as error:
            raise MissingEvidence(f"Invalid original JSON: {relative}") from error

    def registered_source(self, relative, *, json_value=True):
        if relative not in self.bound:
            raise MissingEvidence(
                f"Source is not bound to historical registration/input identity: {relative}"
            )
        return self.source(relative, self.bound[relative], json_value=json_value)

    def panel(self, kind, model_id):
        """Authenticate released task -> COMPLETE -> every selected group hash."""
        relative = f"tasks/{kind}/{model_id}/COMPLETE.json"
        try:
            task_bytes = bounded_file(self.root, relative).read_bytes()
            task = json.loads(task_bytes)
        except (FileNotFoundError, ValueError) as error:
            raise MissingEvidence(f"Task receipt absent or invalid: {relative}") from error
        refs = [
            r
            for r in self.authorization["tasks"]
            if r.get("kind") == kind and r.get("model_id") == model_id
        ]
        if (
            len(refs) != 1
            or refs[0]["receipt_sha256"] != object_hash(task)
            or task.get("status") != "COMPLETE"
            or task.get("kind") != kind
            or task.get("model_id") != model_id
            or task.get("registration_sha256") != object_hash(self.registration)
        ):
            raise MissingEvidence(f"Task differs from released authorization: {relative}")
        directory = (
            f"sealed/evaluation/{model_id}" if kind == "endpoint" else f"diagnostics/{model_id}"
        )
        artifacts = task.get("artifacts", [])
        if len(artifacts) != 1 or artifacts[0].get("path") != f"{directory}/COMPLETE.json":
            raise MissingEvidence(f"Task completion binding differs: {relative}")
        self.source(relative, hashlib.sha256(task_bytes).hexdigest())
        done = self.source(artifacts[0]["path"], artifacts[0]["sha256"])
        groups = done.get("groups", [])
        if (
            done.get("status") != "COMPLETE"
            or done.get("model_id") != model_id
            or done.get("records") != task.get("records")
            or not groups
            or len({r["group_id"] for r in groups}) != len(groups)
        ):
            raise MissingEvidence(f"Panel completion identity differs: {directory}")
        return directory, {r["group_id"]: r for r in groups}

    def group(self, directory, refs, key, count, *, gold_location):
        group_id = object_hash(key)
        relative = f"{directory}/groups/{group_id}.json"
        ref = refs.get(group_id)
        if ref is None:
            raise MissingEvidence(f"Released panel omits group: {relative}")
        group = self.source(relative, ref["raw_sha256"])
        rows = group.get("records", [])
        if (
            group.get("group_id") != group_id
            or len(rows) != count
            or group.get("records_sha256") != object_hash(rows)
            or ref.get("records_sha256") != object_hash(rows)
        ):
            raise MissingEvidence(f"Group record identity/count differs: {relative}")
        missing_fields = []
        for index, row in enumerate(rows):
            actual = tuple(
                row.get(k) for k in ("model_id", "step", "pool", "protocol", "view", "qid")
            )
            if (
                actual != tuple(key)
                or row.get("draw") != index
                or row.get("experiment_id") != "SR-F2-20261013"
            ):
                raise MissingEvidence(f"Raw slot identity differs: {relative} /records/{index}")
            required = list(RAW_FIELDS)
            if key[3] in ("free", "list"):
                required += ["generated_text", "assistant_prefill", "answer_extraction"]
            field_types = dict.fromkeys(required, str)
            field_types.update(
                tokens=list,
                sampler_logprobs=list,
                generation=dict,
                sampling_parameters=dict,
                truncated=bool,
                completion_token_count=int,
            )
            absent = [name for name in required if type(row.get(name)) is not field_types[name]]
            missing_fields.extend(f"/records/{index}/{name}" for name in absent)
            self.records.append(
                dict(
                    model_id=key[0],
                    pool=key[2],
                    protocol=key[3],
                    view=key[4],
                    qid=key[5],
                    root_id=row.get("root_id"),
                    draw=index,
                    source_path=relative,
                    source_sha256=ref["raw_sha256"],
                    source_json_pointer=f"/records/{index}",
                    source_line=1 if self.files[relative]["source_line_count"] == 1 else None,
                    export_path=self.files[relative]["export_path"],
                    gold_source=gold_location,
                    actual_prompt_text=row.get("text"),
                    generation=row.get("generation"),
                    sampling_parameters=row.get("sampling_parameters"),
                    finish_reason=row.get("finish_reason"),
                    truncated=row.get("truncated"),
                    assistant_prefill=row.get("assistant_prefill"),
                    answer_extraction=row.get("answer_extraction"),
                    completion_token_count=row.get("completion_token_count"),
                    information_granted=row.get("information_granted"),
                    input_hash=row.get("input_hash"),
                    image_hash=row.get("image_hash"),
                    missing_raw_fields=absent,
                )
            )
        if missing_fields:
            raise MissingEvidence(
                f"Original raw fields absent or invalid in {relative}: {missing_fields}"
            )
        return len(rows)

    def export_model(self, kind, model_id, specs):
        """Keep every required item in the denominator, including missing groups."""
        try:
            directory, refs = self.panel(kind, model_id)
        except MissingEvidence as error:
            return [
                _missing(
                    f"{model_id}/{spec['label']}",
                    error,
                    expected_records=len(spec["questions"]) * spec["k"],
                    exported_records=0,
                )
                for spec in specs
            ]
        results = []
        for spec in specs:
            missing, exported = [], 0
            for question in spec["questions"]:
                key = (
                    model_id,
                    0 if model_id == BASELINE else 96,
                    spec["pool"],
                    spec["protocol"],
                    spec["view"],
                    question["qid"],
                )
                try:
                    exported += self.group(
                        directory, refs, key, spec["k"], gold_location=question["gold_source"]
                    )
                except MissingEvidence as error:
                    missing.append(dict(qid=question["qid"], reason=str(error)))
            results.append(
                dict(
                    item=f"{model_id}/{spec['label']}",
                    status="MISSING" if missing else "COMPLETE",
                    expected_records=len(spec["questions"]) * spec["k"],
                    exported_records=exported,
                    missing_groups=missing,
                )
            )
        return results


def _jsonl_metadata(exporter, relative):
    raw = exporter.registered_source(relative, json_value=False)
    result = {}
    for line, content in enumerate(raw.splitlines(), 1):
        try:
            row = json.loads(content)
        except ValueError as error:
            raise MissingEvidence(f"Invalid metadata JSONL: {relative}:{line}") from error
        qid = row.get("qid")
        if not isinstance(qid, str) or qid in result:
            raise MissingEvidence(f"Duplicate/missing metadata qid: {relative}:{line}")
        result[qid] = dict(
            row,
            gold_source=dict(
                source_path=relative, source_line=line, source_sha256=exporter.bound[relative]
            ),
        )
    return result


def _image_exports(exporter, input_identity, chartqa_path, chartqa):
    from .external import audit_history_images

    history_path = exporter.output / "prior/HISTORICAL_IMAGES.json"
    receipt = audit_history_images(
        exporter.root / "INPUT_IDENTITY.json",
        exporter.root,
        history_path,
        chartqa_manifest=exporter.root / chartqa_path,
    )
    inventory = {r["path"]: r for r in input_identity["files"]}
    copies = []
    unique = {row["image_file"]: row["image_sha256"] for row in chartqa.values()}
    for relative, digest in sorted(unique.items()):
        if inventory.get(relative, {}).get("sha256") != digest:
            raise MissingEvidence(f"ChartQA image not attested by historical inputs: {relative}")
        path = bounded_file(exporter.root, relative)
        copied = copy_verified(
            path, exporter.output, Path("external/legacy_images") / relative, digest
        )
        copies.append(dict(source_path=relative, source_sha256=digest, **copied))
    images = dict(
        status="COMPLETE",
        model_calls=0,
        image_root="external/legacy_images",
        image_count=len(copies),
        chartqa_manifest_source=chartqa_path,
        historical_receipt_sha256=file_hash(history_path),
        files=copies,
    )
    write_once(exporter.output, "prior/LEGACY_IMAGE_COPY.json", images)
    return dict(
        item="historical_images_and_chartqa_copy",
        status="COMPLETE",
        checked_images=receipt["checked_images"],
        copied_chartqa_images=len(copies),
        historical_receipt="prior/HISTORICAL_IMAGES.json",
        copy_receipt="prior/LEGACY_IMAGE_COPY.json",
    )


def export_prior(parent_root, output_root):
    """Export only the old experiment; new TEST outputs are never enumerated/read."""
    root, output = Path(parent_root).resolve(strict=True), Path(output_root).resolve(strict=True)
    audit = read_json(bounded_file(output, "parents/PARENT_AUDIT.json"))
    authorization = read_json(bounded_file(root, "released/AUTHORIZATION.json"))
    registration = read_json(bounded_file(root, "REGISTRATION.json"))
    if (
        audit.get("status") != "PASS"
        or Path(audit["parent_root"]).resolve(strict=True) != root
        or file_hash(root / "released/AUTHORIZATION.json") != audit["authorization_sha256"]
        or object_hash(registration) != audit["registration_object_hash"]
        or authorization.get("registration_sha256") != object_hash(registration)
        or authorization.get("experiment_id") != "SR-F2-20261013"
    ):
        raise PermissionError("Prior export requires the same authenticated parent audit")
    exporter = PriorExporter(root, output, authorization, registration)
    items = []
    exporter.source("REGISTRATION.json", file_hash(root / "REGISTRATION.json"))
    exporter.source("released/AUTHORIZATION.json", audit["authorization_sha256"])
    inputs = exporter.registered_source("INPUT_IDENTITY.json")
    if inputs.get("status") != "PASS":
        raise PermissionError("Historical input identity is not PASS")
    for entry in inputs["files"]:
        if entry["path"] in exporter.bound and exporter.bound[entry["path"]] != entry["sha256"]:
            raise PermissionError("Registration and input inventory hashes conflict")
        exporter.bound[entry["path"]] = entry["sha256"]
    chartqa = None
    try:
        candidates = [
            name for name in exporter.bound if name.endswith("/chartqa/TEST_MANIFEST.jsonl")
        ]
        if len(candidates) != 1:
            raise MissingEvidence("Unique historical ChartQA manifest is absent")
        chartqa_path = candidates[0]
        chartqa = _jsonl_metadata(exporter, chartqa_path)
        if len(chartqa) != 2500:
            raise MissingEvidence("Historical ChartQA manifest does not contain 2500 questions")
        spec = dict(
            label="CHARTQA",
            pool="CHARTQA",
            protocol="plain_answer",
            view=None,
            k=1,
            questions=list(chartqa.values()),
        )
        for model_id in ENDPOINTS:
            items.extend(exporter.export_model("endpoint", model_id, [spec]))
    except MissingEvidence as error:
        items.extend(_missing(f"{model_id}/CHARTQA", error) for model_id in ENDPOINTS)
    try:
        manifest = exporter.registered_source("DIAGNOSTIC_MANIFEST.json")
        tasks = _jsonl_metadata(exporter, "manifests/TASKS_GOLD_AUDIT_ONLY.jsonl")
        for relative in (
            "NATIVE_DIAGNOSTIC_PROMPTS.json",
            "manifests/MODEL_INPUTS.jsonl",
            "manifests/DIAGNOSTIC_INPUTS_AUDIT_ONLY.jsonl",
            "manifests/ROOTS_AUDIT_ONLY.jsonl",
        ):
            try:
                exporter.registered_source(relative, json_value=not relative.endswith(".jsonl"))
                items.append(dict(item=relative, status="COMPLETE"))
            except MissingEvidence as error:
                items.append(_missing(relative, error))
        if (
            set(manifest["protocol_models"]) != set(DIAGNOSTIC_MODELS)
            or len(manifest["protocol_qids"]) != 512
            or len(set(manifest["protocol_qids"])) != 512
            or len(manifest["interval_qids"]) != 32
            or len(set(manifest["interval_qids"])) != 32
            or manifest["interval_views"] != ["V0", "V1", "V2", "V3", "V4"]
            or manifest["protocol_K"] != 2
            or manifest["interval_K"] != 4
        ):
            raise MissingEvidence("Historical diagnostic panel cardinalities differ")
        absent = set(manifest["protocol_qids"] + manifest["interval_qids"]) - tasks.keys()
        if absent:
            raise MissingEvidence(f"Diagnostic gold questions missing: {sorted(absent)}")
        interval = [tasks[qid] for qid in manifest["interval_qids"]]
        roots = sorted({row["root_id"] for row in interval})
        items.append(
            dict(
                item="V0-V4_requested_at_least_32_numeric_roots",
                status="COMPLETE" if len(roots) >= 32 else "MISSING",
                requested_minimum_roots=32,
                available_numeric_roots=len(roots),
                available_questions=len(interval),
                root_ids=roots,
                reason="All historical registered questions are exported. The old panel has "
                "16 numeric roots rendered as 32 questions; no new samples are generated.",
            )
        )
        specs = [
            dict(
                label=p.upper(),
                pool="TEST_ID",
                protocol=p,
                view=None,
                k=2,
                questions=[tasks[qid] for qid in manifest["protocol_qids"]],
            )
            for p in ("free", "list")
        ]
        specs.extend(
            dict(
                label=view,
                pool="TEST_COMPOSITION",
                protocol="evidence_answer",
                view=view,
                k=4,
                questions=interval,
            )
            for view in manifest["interval_views"]
        )
        for model_id in DIAGNOSTIC_MODELS:
            items.extend(exporter.export_model("diagnostic", model_id, specs))
    except MissingEvidence as error:
        items.append(_missing("historical_diagnostics", error))
    if chartqa is not None and len(chartqa) == 2500:
        try:
            items.append(_image_exports(exporter, inputs, chartqa_path, chartqa))
        except (MissingEvidence, FileNotFoundError, ValueError, PermissionError) as error:
            items.append(_missing("historical_images_and_chartqa_copy", error))
    else:
        items.append(_missing("historical_images_and_chartqa_copy", "ChartQA manifest unavailable"))
    files = sorted(exporter.files.values(), key=lambda row: row["source_path"])
    write_once(output, "prior/raw/FILE_MANIFEST.json", dict(source_root=str(root), files=files))
    records = write_jsonl_once(output, "prior/raw/RECORD_INDEX.jsonl", exporter.records)
    report = dict(
        status="COMPLETE" if all(item["status"] == "COMPLETE" for item in items) else "MISSING",
        operation_completed=True,
        stage="P0_A_PRIOR_RAW_EXPORT",
        critical=False,
        blocks_new_science=False,
        source_root=str(root),
        model_calls=0,
        new_test_outputs_accessed=False,
        historical_scores_modified=False,
        parent_audit_sha256=file_hash(output / "parents/PARENT_AUDIT.json"),
        items=items,
        counts=dict(Counter(item["status"] for item in items)),
        copied_source_files=len(files),
        indexed_raw_records=len(exporter.records),
        file_manifest="prior/raw/FILE_MANIFEST.json",
        record_index=records,
        historical_scoring_source="prior/source/src/sr_f2/scoring.py",
        historical_runtime_source="prior/source/src/sr_f2/runtime.py",
        scope="Released SR-F2 raw evidence only; development attribution; "
        "no rescoring or new sampling",
    )
    write_once(output, "prior/RAW_EXPORT.json", report)
    return report
