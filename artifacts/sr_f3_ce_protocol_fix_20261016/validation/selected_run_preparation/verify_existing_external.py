"""Read-only preparation check: no selected root, Slurm call, publication, or model."""

import collections
import hashlib
import importlib.util
import json
import sys
import tarfile
from pathlib import Path

sys.dont_write_bytecode = True
WT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
OLD = WT / "artifacts/sr_f3_ce_20261016/engine_gate_audit_20261016"
EXT = OLD / "external_input_snapshot"
REVIEW = OLD / "snapshot/external/revisions/typed_v1"
SOURCE = WT / "artifacts/sr_f3_ce_protocol_fix_20261016/deployment/SOURCE_MANIFEST.json"
COMMIT = "652e52d530520a276b6420ff0f4ee479dcde6db1"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def record(path):
    return dict(path=str(path.relative_to(WT)), bytes=path.stat().st_size, sha256=sha(path))


source = read(SOURCE)
assert source["code_commit"] == COMMIT
bundle = SOURCE.with_name("SOURCE_CODE.tar")
with tarfile.open(bundle) as archive:
    members = {
        m.name.removeprefix("code/").rstrip("/"): m for m in archive.getmembers() if m.isfile()
    }
    assert set(members) == {r["path"] for r in source["files"]}
    for item in source["files"]:
        m = members[item["path"]]
        assert (
            m.size == item["bytes"]
            and hashlib.sha256(archive.extractfile(m).read()).hexdigest() == item["sha256"]
        ), item["path"]
source_by_path = {r["path"]: r for r in source["files"]}
for relative, item in source_by_path.items():
    if relative.startswith("src/") or relative == "scripts/sr_f3_ce/finalize_external_types.py":
        p = WT / relative
        assert p.stat().st_size == item["bytes"] and sha(p) == item["sha256"], relative
sys.path.insert(0, str(WT / "src"))
from sr_f1.chartqa import PROMPT  # noqa: E402
from sr_f3_ce import external  # noqa: E402
from sr_f3_ce.evaluation import object_hash  # noqa: E402

finalizer = WT / "scripts/sr_f3_ce/finalize_external_types.py"
assert sha(finalizer) == "1a75df6bf5fa1c0ef22a23ee7fd766a8e22f5a1c6c1fe7d579c4ab6135106aaf"
spec = importlib.util.spec_from_file_location("selected_prep_original4146", finalizer)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

manifest_path = EXT / "EXTERNAL_INPUTS_MANIFEST.json"
manifest = read(manifest_path)
assert len(manifest["files"]) == 1785
for row in manifest["files"]:
    path = EXT / row["path"]
    assert (
        not path.is_symlink() and path.stat().st_size == row["bytes"] and sha(path) == row["sha256"]
    )

chartqa = [
    json.loads(s) for s in (EXT / "external/chartqa/MODEL_INPUTS.jsonl").read_text().splitlines()
]
prefix = PROMPT + "\n"
assert len(chartqa) == 2500 and all(r["prompt"].startswith(prefix) for r in chartqa)
items_qa = {
    r["qid"]: dict(
        qid=r["qid"], question=r["prompt"][len(prefix) :], image_sha256=r["image_sha256"]
    )
    for r in chartqa
}
selection = read(EXT / "external/chartqapro/SELECTION.json")
items_pro = {r["qid"]: r for r in selection["selected"]}
assert len(items_pro) == 256 and not set(items_qa) & set(items_pro)
assert object_hash(selection) == read(EXT / "external/chartqapro/FREEZE.json")["selection_sha256"]
items = {**items_qa, **items_pro}
a, by_a, scope, a_bytes = module.annotation_set(
    REVIEW / "REVIEW_A.json", items, external.EXPERIMENT_ID, external, object_hash
)
b, by_b, scope_b, b_bytes = module.annotation_set(
    REVIEW / "REVIEW_B.json", items, external.EXPERIMENT_ID, external, object_hash
)
assert scope == scope_b == read(REVIEW / "TYPE_AUDIT_RECEIPT.json")["annotation_scope_sha256"]
assert a["reviewer_id"] != b["reviewer_id"]
summary = {}
for pool, pool_items in [("CHARTQA", items_qa), ("CHARTQAPRO", items_pro)]:
    registry = external.build_type_registry(
        list(pool_items.values()),
        [r for q in sorted(pool_items) for r in (by_a[q], by_b[q])],
        outputs_unseen=True,
    )
    summary[pool] = dict(
        questions=len(registry["rows"]),
        status_counts=dict(collections.Counter(r["status"] for r in registry["rows"])),
        type_counts=dict(collections.Counter(r["kind"] for r in registry["rows"])),
    )
assert summary == read(REVIEW / "TYPE_AUDIT_RECEIPT.json")["summary"]

history_path = OLD / "preparation_snapshot/prior/HISTORICAL_IMAGES.json"
history = read(history_path)
history_pixels = {r["image_sha256"]: r["pixel_sha256"] for r in history["rows"]}
for row in chartqa:
    path = EXT / "external/legacy_images" / row["image_file"]
    assert sha(path) == row["image_sha256"]
    assert external._pixel_hash(path.read_bytes()) == history_pixels[row["image_sha256"]]
pro_inputs = [
    json.loads(s) for s in (EXT / "external/chartqapro/MODEL_INPUTS.jsonl").read_text().splitlines()
]
assert {r["qid"] for r in pro_inputs} == set(items_pro)
for row in pro_inputs:
    path = EXT / "external/chartqapro" / row["image_file"]
    pixel = external._pixel_hash(path.read_bytes())
    assert sha(path) == row["image_sha256"] == items_pro[row["qid"]]["image_sha256"]
    assert pixel == items_pro[row["qid"]]["image_pixel_sha256"]
    assert (
        row["image_sha256"] not in history["image_sha256"] and pixel not in history["pixel_sha256"]
    )

copy_rows = []
for row in manifest["files"]:
    destination = row["path"]
    category = "BYTE_IDENTICAL_ACTIVE_INPUT_REUSE"
    if destination == "external/chartqa/FREEZE.json":
        destination = "protocol_revision/external_v1_original/chartqa/FREEZE.json"
        category = "PROVENANCE_ONLY_REBUILD_ACTIVE_ABSOLUTE_IMAGE_ROOT"
    copy_rows.append(
        dict(
            source=str((EXT / row["path"]).relative_to(WT)),
            source_root_relative=row["path"],
            destination=destination,
            bytes=row["bytes"],
            sha256=row["sha256"],
            category=category,
        )
    )
for rel in [
    "PANELS.v0.json",
    "PANELS.v1.json",
    "TYPE_AUDIT_RECEIPT.json",
    "PUBLICATION.json",
    "BLIND_SCOPE.json",
    "RECONCILIATION.json",
    "REVIEW_A.json",
    "REVIEW_B.json",
    "reviewer_evidence/A_MANIFEST.json",
    "reviewer_evidence/B_MANIFEST.json",
]:
    p = REVIEW / rel
    copy_rows.append(
        dict(
            source=str(p.relative_to(WT)),
            source_root_relative="external/revisions/typed_v1/" + rel,
            destination="protocol_revision/type_review_v1/" + rel,
            bytes=p.stat().st_size,
            sha256=sha(p),
            category="ORIGINAL_REVIEW_PROVENANCE_NOT_NEW_HUMAN_REVIEW",
        )
    )
review_evidence = {}
for label in ("A", "B"):
    supplied = module.evidence_inventory(REVIEW / "reviewer_evidence" / label, sha)
    prior = read(REVIEW / "reviewer_evidence" / (label + "_MANIFEST.json"))
    assert supplied == prior["files"]
    review_evidence[label] = dict(
        files=len(supplied),
        bytes=sum(r["bytes"] for r in supplied),
        manifest_sha256=sha(REVIEW / "reviewer_evidence" / (label + "_MANIFEST.json")),
    )
    for row in supplied:
        rel = "reviewer_evidence/" + label + "/" + row["path"]
        copy_rows.append(
            dict(
                source=str((REVIEW / rel).relative_to(WT)),
                source_root_relative="external/revisions/typed_v1/" + rel,
                destination="protocol_revision/type_review_v1/" + rel,
                bytes=row["bytes"],
                sha256=row["sha256"],
                category="ORIGINAL_REVIEW_EVIDENCE_BYTES_PRESERVE_NEVER_EXECUTE",
            )
        )
for rel in ["FREEZE.json", "MODEL_INPUTS.jsonl", "TASKS_AUDIT_ONLY.jsonl", "TYPE_REGISTRY.json"]:
    p = EXT / "external/chartqa" / rel
    if rel != "FREEZE.json":
        copy_rows.append(
            dict(
                source=str(p.relative_to(WT)),
                source_root_relative="external/chartqa/" + rel,
                destination="protocol_revision/external_v1_original/chartqa/" + rel,
                bytes=p.stat().st_size,
                sha256=sha(p),
                category="UNCHANGED_INPUT_COMPARISON_PROVENANCE",
            )
        )
for rel in [
    "FREEZE.json",
    "MODEL_INPUTS.jsonl",
    "TASKS_AUDIT_ONLY.jsonl",
    "TYPE_REGISTRY.json",
    "SELECTION.json",
]:
    p = EXT / "external/chartqapro" / rel
    copy_rows.append(
        dict(
            source=str(p.relative_to(WT)),
            source_root_relative="external/chartqapro/" + rel,
            destination="protocol_revision/external_v1_original/chartqapro/" + rel,
            bytes=p.stat().st_size,
            sha256=sha(p),
            category="UNCHANGED_INPUT_COMPARISON_PROVENANCE",
        )
    )
source_identity = read(EXT / "external/chartqapro_sources/SOURCE_IDENTITY.json")
parquet = source_identity["assets"]["test-00000-of-00001.parquet"]
assert parquet["sha256"] == external.PARQUET_SHA256
copy_rows.append(
    dict(
        source=None,
        source_root_relative="external/chartqapro_sources/test-00000-of-00001.parquet",
        destination="external/chartqapro_sources/test-00000-of-00001.parquet",
        bytes=parquet["bytes"],
        sha256=parquet["sha256"],
        category="SERVER_EXISTING_PINNED_ASSET_VERIFY_BEFORE_COPY_NOT_IN_LOCAL_SNAPSHOT",
    )
)
assert len({r["destination"] for r in copy_rows}) == len(copy_rows)
save(
    "EXTERNAL_COPY_ALLOWLIST.json",
    dict(
        schema_version=1,
        status="PREPARATION_ONLY_NO_SELECTION_NO_COPY_PERFORMED",
        old_server_root="/projects/_ssd/varunssd/louis-ssvc/sr_f3_ce_20261016",
        source_manifest=record(manifest_path),
        records=copy_rows,
        active_live_panels="REBUILD_DO_NOT_COPY_OLD_TYPED_POINTER",
        preserve_pycache_only_in_historical_review_provenance=True,
        rows=len(copy_rows),
    ),
)
save(
    "EXTERNAL_READONLY_VALIDATION.json",
    dict(
        status="PASS_READONLY_EXISTING_EVIDENCE",
        scientific_commit=COMMIT,
        source_manifest=record(SOURCE),
        checked_scientific_source_files=len(source["files"]),
        source_archive_verified_exact=True,
        current_imported_src_and_finalizer_match_frozen_manifest=True,
        current_HEAD_documentation_and_added_dispatcher_not_substituted_into_scientific_archive=True,
        finalizer=record(finalizer),
        original_review_files={
            name: record(REVIEW / name) for name in ["REVIEW_A.json", "REVIEW_B.json"]
        },
        original_review_evidence=review_evidence,
        local_external_input_files_verified=1785,
        visible_questions=2756,
        unique_chartqa_image_files=len({r["image_file"] for r in chartqa}),
        unique_chartqapro_image_files=len({r["image_file"] for r in pro_inputs}),
        pixel_identity_verified=True,
        chartqapro_history_byte_and_pixel_exclusion_verified=True,
        annotation_scope_sha256=scope,
        annotation_set_function="unchanged_4146_finalizer.annotation_set",
        merged_summary=summary,
        review_original_bytes_unchanged=True,
        new_human_review=False,
        new_model_calls=0,
        new_run_created=False,
        protocol_selection_read=False,
        full_new_run_publication_tested=False,
        publication_limit=(
            "New-root verify_run, exact v0 rebuild, Slurm allocation "
            "and 4146 --publish remain future checks"
        ),
        chartqapro_parquet="SERVER_ONLY_REQUIRED_NOT_READ_OR_DOWNLOADED",
        history_receipt=record(history_path),
    ),
)
print(
    json.dumps(
        dict(
            status="PASS",
            source_files=len(source["files"]),
            local_external_files=1785,
            blind_items=len(items),
            copy_allowlist_records=len(copy_rows),
            scope=scope,
            summary=summary,
        ),
        sort_keys=True,
    )
)
