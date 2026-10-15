"""Validate two blind annotation passes and publish external typed v1.

This file is an operator-side data preparation script, not scientific source.
Default mode performs read-only validation. --publish requires both original
reviewer evidence directories and a CPU Slurm job whose name ends in -LH.
It calls the already frozen
prepare_chartqa/freeze_chartqapro functions and never calls a model or scorer.

Each review JSON contains experiment_id, reviewer_id, blind_inputs_only: true,
used_gold: false, used_model_output: false, saw_other_review: false, and reviews.
Each review row contains qid, reviewer_id, input_identity, method, kind,
comparison, evidence, used_gold: false, and used_model_output: false.
Both reviewers independently cover every frozen external qid. UNRESOLVED rows
are real reviewed outcomes with explicit reasons, not missing annotations.

Example (only run after both review files have been independently finalized):
  PYTHONPATH="$RUN/code/src" "$PYTHON" /path/finalize_external_types.py \
    --root "$RUN" --review-a /path/A.json --review-b /path/B.json \
    --review-a-evidence-dir /path/A-evidence --review-b-evidence-dir /path/B-evidence \
    --auxiliary-code-commit FULL_COMMITTED_GIT_SHA \
    --publish
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lines(path):
    result = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                raise ValueError("Blank input row")
            result.append(json.loads(line))
    return result


def no_science(root):
    for name in ("EXECUTION_FREEZE.json", "runs", "sealed"):
        if (root / name).exists():
            raise PermissionError(f"Typed audit must finish before science: {name} exists")


def bounded(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise PermissionError("External artifact path is not a safe relative path")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root):
        raise PermissionError("External artifact escaped the experiment root")
    return path


def fsync_directory(directory):
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_bytes_once(path, content):
    """Publish complete durable bytes exclusively; interrupted writes are retryable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise PermissionError(f"Immutable typed-audit artifact differs: {path}")
        # A previous attempt may have linked complete bytes before its directory
        # fsync was interrupted. Repeating that fsync completes the same write.
        fsync_directory(path.parent)
        return
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            if stream.write(content) != len(content):
                raise OSError("Incomplete typed-audit temporary write")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # Same-directory hard linking installs complete bytes atomically
            # and cannot replace an existing immutable artifact.
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise PermissionError(f"Immutable typed-audit artifact differs: {path}") from None
    finally:
        Path(temporary).unlink(missing_ok=True)
    fsync_directory(path.parent)


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def evidence_inventory(directory, sha_file):
    """Inventory supplied rules, self-tests, candidates, and visual evidence."""
    if Path(directory).is_symlink():
        raise PermissionError("Reviewer evidence root cannot be a symlink")
    directory = Path(directory).resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("Reviewer evidence must be a directory")
    records = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise PermissionError("Reviewer evidence cannot contain symlinks")
        if path.is_dir():
            continue
        if not path.is_file():
            raise PermissionError("Reviewer evidence must contain only regular files")
        records.append(
            dict(
                path=str(path.relative_to(directory)),
                sha256=sha_file(path),
                bytes=path.stat().st_size,
            )
        )
    if not records:
        raise ValueError("A reviewer evidence directory cannot be empty")
    return records


def archive_evidence(source, destination, inventory, sha_file):
    source, destination = Path(source).resolve(strict=True), Path(destination)
    target_root = destination.resolve()
    if target_root.is_relative_to(source) or source.is_relative_to(target_root):
        raise PermissionError("Reviewer evidence source and archive cannot overlap")
    for item in inventory:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise PermissionError("Evidence manifest path escapes its source")
        path = source / relative
        if not path.resolve(strict=True).is_relative_to(source):
            raise PermissionError("Evidence manifest resolves outside its source")
        if (
            path.is_symlink()
            or sha_file(path) != item["sha256"]
            or path.stat().st_size != item["bytes"]
        ):
            raise PermissionError("Reviewer evidence changed after inventory")
        target = destination / relative
        if not target.resolve().is_relative_to(target_root):
            raise PermissionError("Evidence archive path escapes its destination")
        write_bytes_once(target, path.read_bytes())
        if sha_file(target) != item["sha256"] or target.stat().st_size != item["bytes"]:
            raise PermissionError("Archived reviewer evidence differs")


def annotation_set(path, items, expected_id, external, object_hash):
    # Retain exactly the bytes validated here. The original path can change
    # while evidence is inventoried; later archival must not reopen it.
    content = Path(path).read_bytes()
    payload = json.loads(content.decode("utf-8"))
    reviewer = payload.get("reviewer_id")
    if (
        payload.get("experiment_id") != expected_id
        or not isinstance(reviewer, str)
        or not reviewer.strip()
        or payload.get("blind_inputs_only") is not True
        or payload.get("used_gold") is not False
        or payload.get("used_model_output") is not False
        or payload.get("saw_other_review") is not False
    ):
        raise PermissionError("Reviewer identity or explicit blindness attestation is missing")
    rows = payload.get("reviews")
    if not isinstance(rows, list) or len(rows) != len(items):
        raise ValueError("Each independent pass must cover all frozen external questions")
    by_qid = {}
    kinds = {*external.TYPE_KINDS, "UNRESOLVED"}
    forbidden = {
        "gold",
        "answer",
        "Answer",
        "Year",
        "prediction",
        "model_output",
        "raw_text",
    }

    def contains_prohibited_field(value):
        if isinstance(value, dict):
            return bool(forbidden & set(value)) or any(
                contains_prohibited_field(child) for child in value.values()
            )
        if isinstance(value, list):
            return any(contains_prohibited_field(child) for child in value)
        return False

    for row in rows:
        qid = row.get("qid")
        if qid not in items or qid in by_qid or contains_prohibited_field(row):
            raise PermissionError("Unknown/duplicate qid or prohibited label/output field")
        if (
            row.get("reviewer_id") != reviewer
            or row.get("input_identity") != external.blind_item_identity(items[qid])
            or row.get("method") not in external.REVIEW_METHODS
            or row.get("used_gold") is not False
            or row.get("used_model_output") is not False
            or row.get("kind") not in kinds
        ):
            raise PermissionError(f"Blind annotation identity or method differs for {qid}")
        evidence = row.get("evidence")
        if not isinstance(evidence, (str, dict)) or not evidence:
            raise ValueError(f"An actual per-question evidence/rejection reason is required: {qid}")
        if isinstance(evidence, str) and not evidence.strip():
            raise ValueError("Whitespace is not annotation evidence")
        kind, rule = row["kind"], row.get("comparison")
        if kind == "UNRESOLVED":
            if rule is not None:
                raise ValueError("UNRESOLVED has no numeric normalization or comparison")
        elif kind in ("YEAR", "COUNT_INT"):
            if rule != "EXACT":
                raise ValueError(
                    "Calendar years and discrete integer counts require exact matching"
                )
        elif rule not in ("EXACT", "RELATIVE_5PCT"):
            raise ValueError("Unregistered external numeric comparison")
        by_qid[qid] = row
    scope_hash = object_hash(
        {qid: external.blind_item_identity(item) for qid, item in sorted(items.items())}
    )
    if payload.get("annotation_scope_sha256", scope_hash) != scope_hash:
        raise PermissionError("Annotation scope hash differs")
    return payload, by_qid, scope_hash, content


def input_panel(root, item, pool, external, object_hash, sha_file):
    from sr_f1.chartqa import PROMPT

    directory = bounded(root, item["root"])
    freeze = read(directory / "FREEZE.json")
    if sha_file(directory / "FREEZE.json") != item["freeze_sha256"]:
        raise PermissionError("v0 external freeze hash differs")
    if pool == "CHARTQAPRO" and freeze["status"] == "UNAVAILABLE":
        if not freeze.get("reason") or freeze.get("image_count") != 0:
            raise PermissionError("Invalid original ChartQAPro unavailability receipt")
        return directory, freeze, {}, None
    store = external.ExternalInputStore(directory)
    raw_inputs = lines(directory / "MODEL_INPUTS.jsonl")
    inputs = {row["qid"]: row for row in raw_inputs}
    if len(inputs) != len(raw_inputs):
        raise ValueError("Duplicate frozen external inputs")
    if pool == "CHARTQA":
        prefix = PROMPT + "\n"
        if len(inputs) != 2500 or any(not row["prompt"].startswith(prefix) for row in raw_inputs):
            raise PermissionError("ChartQA original prompt interface differs")
        # Recover only question/visible-image identity; do not open gold sidecars.
        items = {
            qid: dict(
                qid=qid,
                question=row["prompt"][len(prefix) :],
                image_sha256=row["image_sha256"],
            )
            for qid, row in inputs.items()
        }
    else:
        selection = read(directory / "SELECTION.json")
        if object_hash(selection) != freeze["selection_sha256"]:
            raise PermissionError("Original gold-blind ChartQAPro selection changed")
        items = {row["qid"]: row for row in selection["selected"]}
        if set(items) != set(inputs):
            raise PermissionError("Original selected questions and model inputs differ")
        if any(
            {"Answer", "Year", "answer", "gold", "prediction"} & set(row) for row in items.values()
        ):
            raise PermissionError("Gold/output leaked into the original blind selection")
    for qid in inputs:
        request = store.model_input(qid)
        image = (store.image_root / request["image_file"]).resolve(strict=True)
        if (
            not image.is_relative_to(store.image_root)
            or sha_file(image) != request["image_sha256"]
            or items[qid]["image_sha256"] != request["image_sha256"]
        ):
            raise PermissionError("Original visible image identity differs")
    registry = read(directory / "TYPE_REGISTRY.json")
    if {row["qid"] for row in registry["rows"]} != set(items):
        raise PermissionError("Original type registry coverage differs")
    for row in registry["rows"]:
        if (
            row["input_identity"] != external.blind_item_identity(items[row["qid"]])
            or row["status"] != "UNRESOLVED"
            or row["reviews"]
        ):
            raise PermissionError("This revision is only for the unreviewed all-UNRESOLVED v0")
    return directory, freeze, items, store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--review-a", required=True, type=Path)
    parser.add_argument("--review-b", required=True, type=Path)
    parser.add_argument("--review-a-evidence-dir", type=Path)
    parser.add_argument("--review-b-evidence-dir", type=Path)
    parser.add_argument("--auxiliary-code-commit")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    # Import the exact deployed implementation, never a newly edited checkout.
    sys.path.insert(0, str(root / "code/src"))
    from sr_f3_ce import external
    from sr_f3_ce.evaluation import EXPERIMENT_ID, object_hash, sha_file
    from sr_f3_ce.protocol import verify_run
    from sr_f12.evaluation import durable_json, shard_lease

    identity, _ = verify_run(root)
    if "external/PANELS.json" in identity.get("bound_files", {}):
        raise PermissionError(
            "The panel pointer is already bound by RUN_IDENTITY; do not replace it"
        )
    if Path(external.__file__).resolve() != (root / "code/src/sr_f3_ce/external.py").resolve():
        raise PermissionError("Typed finalization imported a different scientific source tree")
    no_science(root)
    revision = root / "external/revisions/typed_v1"
    if not revision.resolve().is_relative_to(root):
        raise PermissionError("Revision directory escapes experiment root")
    live_index = root / "external/PANELS.json"
    original_index = revision / "PANELS.v0.json"
    base_index = original_index if original_index.exists() else live_index
    v0_bytes = base_index.read_bytes()
    panels = json.loads(v0_bytes.decode("utf-8"))
    if panels.get("status") != "FROZEN" or panels.get("experiment_id") != EXPERIMENT_ID:
        raise PermissionError("Original external panel identity is not frozen")
    panel_data = {
        pool: input_panel(root, panels[pool], pool, external, object_hash, sha_file)
        for pool in ("CHARTQA", "CHARTQAPRO")
    }
    items = {}
    for _, _, panel_items, _ in panel_data.values():
        if set(items) & set(panel_items):
            raise ValueError("External qids overlap between benchmarks")
        items.update(panel_items)
    reviews_a, by_a, scope_hash, review_a_bytes = annotation_set(
        args.review_a, items, EXPERIMENT_ID, external, object_hash
    )
    reviews_b, by_b, other_scope, review_b_bytes = annotation_set(
        args.review_b, items, EXPERIMENT_ID, external, object_hash
    )
    if reviews_a["reviewer_id"] == reviews_b["reviewer_id"] or scope_hash != other_scope:
        raise PermissionError("Two distinct independent reviewer identities are required")
    review_snapshots = {"REVIEW_A.json": review_a_bytes, "REVIEW_B.json": review_b_bytes}
    review_hashes = {
        name: hashlib.sha256(content).hexdigest() for name, content in review_snapshots.items()
    }
    supplied_evidence = {"A": args.review_a_evidence_dir, "B": args.review_b_evidence_dir}
    if args.publish and any(path is None for path in supplied_evidence.values()):
        raise PermissionError("Publication requires both original reviewer evidence directories")
    if args.publish and re.fullmatch(r"[0-9a-f]{40}", args.auxiliary_code_commit or "") is None:
        raise PermissionError("Publication requires the full committed auxiliary-tool Git SHA")
    evidence_manifests = {
        label: evidence_inventory(directory, sha_file)
        for label, directory in supplied_evidence.items()
        if directory is not None
    }
    merged, merged_registries, reconciliation = {}, {}, {}
    for pool, (_, _, panel_items, _) in panel_data.items():
        merged[pool] = [row for qid in sorted(panel_items) for row in (by_a[qid], by_b[qid])]
        registry = external.build_type_registry(
            list(panel_items.values()), merged[pool], outputs_unseen=True
        )
        merged_registries[pool] = registry
        reconciliation[pool] = []
        for row in registry["rows"]:
            a, b = by_a[row["qid"]], by_b[row["qid"]]
            expected_resolved = a["kind"] != "UNRESOLVED" and (
                a["kind"],
                a["comparison"],
            ) == (
                b["kind"],
                b["comparison"],
            )
            if (row["status"] == "RESOLVED") != expected_resolved or len(row["reviews"]) != 2:
                raise PermissionError(
                    "Merged type result does not follow the pre-output two-review rule"
                )
            reason = (
                "independent_agreement"
                if expected_resolved
                else "both_reviewers_unresolved"
                if a["kind"] == b["kind"] == "UNRESOLVED"
                else "independent_disagreement_or_insufficient_evidence"
            )
            reconciliation[pool].append(
                dict(
                    qid=row["qid"],
                    status=row["status"],
                    kind=row["kind"],
                    comparison=row["comparison"],
                    reason=reason,
                )
            )
    summary = {
        pool: dict(
            questions=len(registry["rows"]),
            status_counts=dict(Counter(row["status"] for row in registry["rows"])),
            type_counts=dict(Counter(row["kind"] for row in registry["rows"])),
        )
        for pool, registry in merged_registries.items()
    }
    if not args.publish:
        print(
            json.dumps(
                dict(
                    status="VALIDATED_NOT_PUBLISHED",
                    annotation_scope_sha256=scope_hash,
                    panels=summary,
                    model_calls=0,
                ),
                sort_keys=True,
            )
        )
        return
    if not os.environ.get("SLURM_JOB_ID", "").isdigit() or not os.environ.get(
        "SLURM_JOB_NAME", ""
    ).endswith("-LH"):
        raise PermissionError("Publication must run as an authorized CPU Slurm job ending in -LH")
    with shard_lease(revision):
        no_science(root)
        if live_index.read_bytes() != v0_bytes and not (revision / "PANELS.v1.json").exists():
            raise PermissionError("Current panel pointer changed before versioned finalization")
        write_bytes_once(original_index, v0_bytes)
        for name, content in review_snapshots.items():
            write_bytes_once(revision / name, content)
            if sha_file(revision / name) != review_hashes[name]:
                raise PermissionError("Archived review differs from its validated byte snapshot")
        for label, inventory in evidence_manifests.items():
            destination = revision / "reviewer_evidence" / label
            archive_evidence(supplied_evidence[label], destination, inventory, sha_file)
            durable_json(
                revision / "reviewer_evidence" / (label + "_MANIFEST.json"),
                dict(
                    reviewer_id=reviews_a["reviewer_id"]
                    if label == "A"
                    else reviews_b["reviewer_id"],
                    files=inventory,
                    total_bytes=sum(item["bytes"] for item in inventory),
                    archive_scope=(
                        "all files in operator-supplied original reviewer evidence directory"
                    ),
                    completeness_of_operator_supplied_evidence_not_inferred=True,
                    scientific_model_calls=0,
                ),
            )
        durable_json(
            revision / "BLIND_SCOPE.json",
            dict(
                experiment_id=EXPERIMENT_ID,
                annotation_scope_sha256=scope_hash,
                inputs={
                    qid: external.blind_item_identity(item) for qid, item in sorted(items.items())
                },
                gold_accessed_for_type_assignment=False,
                model_outputs_accessed=False,
            ),
        )
        durable_json(revision / "RECONCILIATION.json", reconciliation)
        for pool, (directory, _, _, _) in panel_data.items():
            for name in ("FREEZE.json", "TYPE_REGISTRY.json", "SELECTION.json"):
                if (directory / name).exists():
                    write_bytes_once(revision / "v0" / pool / name, (directory / name).read_bytes())
        history = read(root / "prior/HISTORICAL_IMAGES.json")
        sources = root / "external/chartqapro_sources"
        new_dirs = {pool: root / ("external/" + pool.lower() + "_typed_v1") for pool in panel_data}
        if any(not directory.resolve().is_relative_to(root) for directory in new_dirs.values()):
            raise PermissionError("Typed output directory escapes experiment root")
        new_freezes = {
            "CHARTQA": external.prepare_chartqa(
                root / "prior/raw/external/chartqa/TEST_MANIFEST.jsonl",
                root / "external/legacy_images",
                new_dirs["CHARTQA"],
                history_receipt=history,
                type_reviews=merged["CHARTQA"],
            )
        }
        if panel_data["CHARTQAPRO"][1]["status"] == "UNAVAILABLE":
            new_freezes["CHARTQAPRO"] = panel_data["CHARTQAPRO"][1]
            durable_json(new_dirs["CHARTQAPRO"] / "FREEZE.json", new_freezes["CHARTQAPRO"])
        else:
            new_freezes["CHARTQAPRO"] = external.freeze_chartqapro(
                sources / "test-00000-of-00001.parquet",
                new_dirs["CHARTQAPRO"],
                history_receipt=history,
                scorer_source=sources / "evaluate_predictions.py",
                prompt_source=sources / "vlmeval_chartqapro_utils.py",
                anls_wheel=sources / "anls-0.0.2-py3-none-any.whl",
                type_reviews=merged["CHARTQAPRO"],
            )
        identities = {}
        for pool, (old_dir, old_freeze, panel_items, old_store) in panel_data.items():
            new_dir, new_freeze = new_dirs[pool], new_freezes[pool]
            ignored = {"type_registry_sha256", "typed_unresolved_count"}
            if {key: value for key, value in old_freeze.items() if key not in ignored} != {
                key: value for key, value in new_freeze.items() if key not in ignored
            }:
                raise PermissionError("A non-type external freeze setting changed")
            if not panel_items:
                identities[pool] = dict(status="UNAVAILABLE", reason=new_freeze["reason"])
                continue
            for name in (
                "MODEL_INPUTS.jsonl",
                "TASKS_AUDIT_ONLY.jsonl",
                "SELECTION.json",
            ):
                if (old_dir / name).exists() and sha_file(old_dir / name) != sha_file(
                    new_dir / name
                ):
                    raise PermissionError(
                        f"Frozen question/input/selection bytes changed: {pool}/{name}"
                    )
            actual_registry = read(new_dir / "TYPE_REGISTRY.json")
            if actual_registry != merged_registries[pool]:
                raise PermissionError("Published registry differs from the blind merge")
            new_store = external.ExternalInputStore(new_dir)
            for qid in panel_items:
                request = new_store.model_input(qid)
                if request != old_store.model_input(qid):
                    raise PermissionError("Exact model input changed during typed audit")
                image = (new_store.image_root / request["image_file"]).resolve(strict=True)
                if (
                    not image.is_relative_to(new_store.image_root)
                    or sha_file(image) != request["image_sha256"]
                ):
                    raise PermissionError("Published visible image changed")
            identities[pool] = dict(
                questions=len(panel_items),
                inputs_sha256=sha_file(new_dir / "MODEL_INPUTS.jsonl"),
                audit_tasks_sha256=sha_file(new_dir / "TASKS_AUDIT_ONLY.jsonl"),
                type_registry_sha256=object_hash(actual_registry),
                image_and_prompt_identity_unchanged=True,
                selection_unchanged=True,
            )
        audit = dict(
            status="TWO_INDEPENDENT_REVIEW_PASSES_COMPLETED",
            experiment_id=EXPERIMENT_ID,
            run_identity_sha256=sha_file(root / "RUN_IDENTITY.json"),
            code_commit=identity["code_commit"],
            scientific_source_changed=False,
            model_calls=0,
            generation_budget_changed=False,
            before_execution_freeze=True,
            before_any_new_test=True,
            original_panels_sha256=sha_file(original_index),
            annotation_scope_sha256=scope_hash,
            reviewer_ids=[reviews_a["reviewer_id"], reviews_b["reviewer_id"]],
            review_files=review_hashes,
            reviewer_evidence_manifests={
                label: sha_file(revision / "reviewer_evidence" / (label + "_MANIFEST.json"))
                for label in evidence_manifests
            },
            review_independence_basis=(
                "two separately supplied blind review attestations; retained original evidence"
            ),
            correctness_certified_by_row_count=False,
            no_minimum_resolution_rate=True,
            unresolved_and_disagreements_preserved=True,
            summary=summary,
            identities=identities,
            preparation_script_sha256=sha_file(Path(__file__)),
            auxiliary_code_commit=args.auxiliary_code_commit,
            auxiliary_commit_identity_supplied_by_operator=True,
        )
        durable_json(revision / "TYPE_AUDIT_RECEIPT.json", audit)
        updated = copy.deepcopy(panels)
        for pool, new_dir in new_dirs.items():
            updated[pool].update(
                root=str(new_dir.relative_to(root)),
                freeze_sha256=sha_file(new_dir / "FREEZE.json"),
            )
        updated.update(
            typed_audit_revision="typed_v1",
            typed_audit_receipt=str((revision / "TYPE_AUDIT_RECEIPT.json").relative_to(root)),
            typed_audit_receipt_sha256=sha_file(revision / "TYPE_AUDIT_RECEIPT.json"),
        )
        new_index = revision / "PANELS.v1.json"
        write_bytes_once(new_index, json_bytes(updated))
        no_science(root)
        current = live_index.read_bytes()
        if current not in (v0_bytes, new_index.read_bytes()):
            raise PermissionError("Live panel pointer changed concurrently; do not overwrite")
        if current != new_index.read_bytes():
            fd, temporary = tempfile.mkstemp(prefix="PANELS.typed_v1.", dir=live_index.parent)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(new_index.read_bytes())
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, live_index)
                directory_fd = os.open(live_index.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                if Path(temporary).exists():
                    Path(temporary).unlink()
        publication = dict(
            status="PUBLISHED_TYPED_V1_BEFORE_SCIENCE",
            previous_panels_sha256=sha_file(original_index),
            current_panels_sha256=sha_file(live_index),
            typed_audit_receipt_sha256=sha_file(revision / "TYPE_AUDIT_RECEIPT.json"),
        )
        durable_json(revision / "PUBLICATION.json", publication)
        print(json.dumps(dict(publication, panels=summary, model_calls=0), sort_keys=True))


if __name__ == "__main__":
    main()
