"""Explicit interface-v2 read adapters; they never change or authorize a run.

File inventories and visual observations remain immutable upstream evidence.
The derived objects below describe compatibility only: files still require the
original production verifier, and all scientific gates remain independent.
"""

from __future__ import annotations

import copy
import hashlib
import re
from pathlib import Path, PurePosixPath

from .common import digest, safe_evidence

CONSUMER_VERSION = "srf6-receipt-consumer-interface-v2"


class CompatibilityError(ValueError):
    """Recorded evidence cannot be translated without losing information."""


def require_consumer_version(version):
    """No implicit fallback or truthy flag may activate the new consumer."""
    if version != CONSUMER_VERSION:
        raise CompatibilityError("Explicit interface-v2 receipt consumer version required")


def _same_typed_value(left, right):
    """Do not equate JSON true with 1, including inside inventory metadata."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same_typed_value(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _same_typed_value(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def normalize_visual_item(item):
    """Read only the three documented aliases; preserve failures and observations."""
    if not isinstance(item, dict):
        raise CompatibilityError("Visual item must be an object")
    if "_migration" in item:
        raise CompatibilityError("Adapt the original visual item, not a derived migration")
    out = copy.deepcopy(item)
    mapping = {}
    for canonical, alias in (
        ("reviewed_source", "source_viewed"),
        ("reviewed_processed", "processed_viewed"),
        ("issues", "defects"),
    ):
        if canonical not in item and alias not in item:
            raise CompatibilityError("Missing visual field: " + canonical)
        if (
            canonical in item
            and alias in item
            and not _same_typed_value(item[canonical], item[alias])
        ):
            raise CompatibilityError("Conflicting visual aliases: " + canonical)
        field = canonical if canonical in item else alias
        value = item[field]
        if canonical == "issues":
            if not isinstance(value, list):
                raise CompatibilityError("Visual issues must be a list")
        elif type(value) is not bool:
            raise CompatibilityError("Visual view flags must be literal booleans")
        out[canonical] = copy.deepcopy(value)
        mapping[canonical] = field
    for name in ("qid", "observation", "status"):
        if not isinstance(item.get(name), str) or not item[name].strip():
            raise CompatibilityError("Missing visual field: " + name)
    for name in ("source_sha256", "processed_file_sha256"):
        if not _sha256(item.get(name)):
            raise CompatibilityError("Invalid visual digest: " + name)
    out["_migration"] = dict(
        schema="srf6-visual-alias-read-adapter-v1",
        consumer_version=CONSUMER_VERSION,
        source_object_sha256=digest(item),
        field_mapping=mapping,
        new_visual_review_performed=False,
        gate_signed=False,
    )
    return out


def coalesce_asset_inventory(rows):
    """Separate question references from unique logical files without discarding qids.

    Repeated paths must agree on every non-qid field, including target, content
    identity and metadata. Distinct paths with the same content stay distinct.
    This does not resolve a path, inspect a symlink or validate a file's bytes;
    callers must still pass unique_files to the unchanged verify_bound_files.
    """
    if not isinstance(rows, list) or not rows:
        raise CompatibilityError("Nonempty question image inventory required")
    unique, indexes, references, qids = [], {}, [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise CompatibilityError("Image inventory row must be an object")
        qid = row.get("qid")
        if not isinstance(qid, str) or not qid.strip() or qid in qids:
            raise CompatibilityError("Missing or duplicate image question ID")
        qids.add(qid)
        bound = {key: copy.deepcopy(value) for key, value in row.items() if key != "qid"}
        for key in ("path", "resolved_path"):
            value = bound.get(key)
            if not isinstance(value, str):
                raise CompatibilityError("Missing or invalid image path: " + key)
            path = PurePosixPath(value)
            if not path.is_absolute() or ".." in path.parts or str(path) != value:
                raise CompatibilityError("Noncanonical logical image path: " + key)
        if not _sha256(bound.get("sha256")):
            raise CompatibilityError("Invalid image content digest")
        if type(bound.get("bytes")) is not int or bound["bytes"] < 0:
            raise CompatibilityError("Invalid image file size")
        key = bound["path"]
        if key in indexes:
            index = indexes[key]
            if not _same_typed_value(unique[index], bound):
                raise CompatibilityError("Conflicting repeated image inventory: " + key)
        else:
            index = len(unique)
            indexes[key] = index
            unique.append(bound)
        references.append(dict(qid=qid, file_index=index, path=key))
    return dict(
        schema="srf6-file-inventory-with-question-references-v1",
        consumer_version=CONSUMER_VERSION,
        input_object_sha256=digest(rows),
        unique_files=unique,
        question_references=references,
        question_count=len(rows),
        unique_file_count=len(unique),
        verification_required=True,
        gate_signed=False,
    )


def assess_receipt_compatibility(root, *, consumer_version):
    """Read an original run or delivered receipt mirror, without touching its files.

    This is a CPU translation exercise of recorded source/visual receipts, not
    _source or _processor admission. Server assets, tokenizer and actual image
    pixels are deliberately not revalidated here. The report can be saved by a
    caller to a new version's own output directory; this function never writes.
    """
    from .review_package import _json

    require_consumer_version(consumer_version)
    root = Path(root).resolve(strict=True)
    refs = {}

    def read(relative):
        path = safe_evidence(root, relative)
        raw = path.read_bytes()
        refs[relative] = dict(path=relative, sha256=hashlib.sha256(raw).hexdigest())
        value = _json(raw)
        if not isinstance(value, dict):
            raise CompatibilityError("Receipt object required: " + relative)
        return value

    source = read("SOURCE_BINDING.json")
    visual = read("PREFLIGHT/VISUAL_REVIEW.json")
    processor = read("PREFLIGHT/PROCESSOR.json")
    panels = {}
    for panel, expected in (("LEGACY", 128), ("CHARTQA", 250)):
        adapted = coalesce_asset_inventory(source["retention_images"][panel])
        if adapted["question_count"] != expected:
            raise CompatibilityError("Recorded retention question coverage differs: " + panel)
        panels[panel] = adapted
    if visual.get("processor_receipt_sha256") != refs["PREFLIGHT/PROCESSOR.json"]["sha256"]:
        raise CompatibilityError("Visual review belongs to a different processor receipt")
    images = processor.get("images", [])
    reviews = [normalize_visual_item(item) for item in visual.get("items", [])]
    if (
        len(images) != 48
        or len(reviews) != 48
        or len({item["qid"] for item in images}) != 48
        or len({item["qid"] for item in reviews}) != 48
        or {item["qid"] for item in images} != {item["qid"] for item in reviews}
    ):
        raise CompatibilityError("Recorded processor and visual question coverage differs")
    by_qid = {item["qid"]: item for item in images}
    for item in reviews:
        for name in ("source_sha256", "processed_file_sha256"):
            if item[name] != by_qid[item["qid"]][name]:
                raise CompatibilityError("Recorded visual image identity differs: " + name)
        for name in ("source_file", "processed_file"):
            if name in item and item[name] != by_qid[item["qid"]][name]:
                raise CompatibilityError("Recorded visual image path differs: " + name)
    unresolved = [
        item["qid"]
        for item in reviews
        if item["status"] != "PASS"
        or item["reviewed_source"] is not True
        or item["reviewed_processed"] is not True
        or item["issues"] != []
    ]
    return dict(
        schema="srf6-receipt-compatibility-assessment-v1",
        consumer_version=consumer_version,
        status="RECORDED_RECEIPTS_ADAPTED_NOT_ADMISSION",
        input_root=str(root),
        inputs=list(refs.values()),
        retention_images=panels,
        visual=dict(
            original_status=visual.get("status"),
            items=reviews,
            item_count=len(reviews),
            aliased_item_count=sum(
                any(key != value for key, value in item["_migration"]["field_mapping"].items())
                for item in reviews
            ),
            unresolved_qids=unresolved,
            new_visual_review_performed=False,
        ),
        server_asset_validation="NOT_RUN",
        processor_pixel_and_tokenizer_validation="NOT_RUN",
        scientific_assessment="NOT_RUN",
        training_authorized=False,
        generation_authorized=False,
        gate_status="NOT_SIGNED",
    )
