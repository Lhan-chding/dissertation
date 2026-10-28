"""Freeze the independent 9,216-answer confirmation, never training permission.

The original producer supplies unchanged model/configuration dependencies. The
new deployment produces new DEV_SCHEMA_CONFIRM data and evidence. The frozen
v3 screen is verified in its own interpreter; it is never relabeled as this
producer. Actual processing and a new independent 48-image review precede the
limited generation seal. Historical model hashes use their existing stat cache.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import socket
import subprocess
import sys
from pathlib import Path

from srf6_ref.core import FAMILIES, seed_for

from . import interface_screen_registration as old
from .common import digest, file_hash, now, read, safe_evidence, write_once
from .interface_v3_protocol import VERSION as PROMPT_VERSION
from .source_binding import PROJECT, inventory_file, stat_cache_key, verify_bound_files

ROOT = PROJECT / "sr_f6_confirm_v3_20261028"
ORIGIN_ROOT = PROJECT / "sr_f6_semantic_factorial_20261027"
SCREEN_ROOT = PROJECT / "sr_f6_interface_v3_20261028"
SCREEN_REVISION = "5ca3fe96c71941e30213b0d7390ae0de2b405a1c"
SCREEN_DEPLOYMENT = PROJECT / "sr_f6_code_5ca3fe9"
PHASE = "SCHEMA_CONFIRM_V3"
SPLIT = "DEV_SCHEMA_CONFIRM"
VERSION = "srf6-confirm-v3"
MASTER_SEED = 20261028
PARENTS = ("b0", "b1", "b2")
REGISTRATION_SCHEMA = "srf6-confirm-registration-v1"
PREFLIGHT_SCHEMA = "srf6-confirm-preflight-v1"
SEAL_SCHEMA = "srf6-confirm-preflight-seal-v1"
TOKENIZER_FIXTURES = old.TOKENIZER_FIXTURES
FROZEN_IMPLEMENTATIONS = (
    "src/sr_f6/interface_v3_protocol.py",
    "src/sr_f6/data.py",
    "src/srf6_ref/design.py",
    "src/srf6_ref/core.py",
    "src/srf6_ref/credit.py",
    "src/srf6_ref/census.py",
)


def _require(value, message):
    if not value:
        raise PermissionError(message)


def _path(path, *, exists=True):
    path = Path(path)
    _require(
        path.is_absolute() and path.is_relative_to(PROJECT) and ".." not in path.parts,
        "Confirmation path is outside the authorized project",
    )
    _require(PROJECT.resolve(strict=True) == PROJECT, "Project must be canonical")
    for current in (path, *path.parents):
        _require(not current.is_symlink(), "Symlinked confirmation paths are forbidden")
        if current == PROJECT:
            break
    _require(path.resolve(strict=exists) == path, "Confirmation path must be canonical")
    return path


def _root(root, *, exists=True):
    root = _path(root, exists=exists)
    _require(root == ROOT, "Only the independent confirmation root is allowed")
    return root


def _actual_deployment():
    return Path(__file__).resolve(strict=True).parents[2]


def _deployment(revision):
    _require(
        isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision),
        "Full immutable Git revision required",
    )
    deployment = _path(_actual_deployment())
    match = re.fullmatch(r"sr_f6_code_([0-9a-f]{7,40})", deployment.name)
    _require(
        deployment.parent == PROJECT and match and revision.startswith(match.group(1)),
        "Actual confirmation deployment differs from its revision",
    )
    _require(deployment != SCREEN_DEPLOYMENT, "Confirmation needs its own actual deployment")
    return deployment


def _ref(root, relative):
    return {"path": relative, "sha256": file_hash(safe_evidence(root, relative))}


def _bound(root, reference, *, lines=False):
    return old._read_reference(root, reference, lines=lines)


def _source_paths(deployment):
    return old._source_paths(deployment)


def check_deployment_imports(deployment):
    script = "\n".join(
        (
            "import sys",
            "from pathlib import Path",
            "root = Path(sys.argv[1]).resolve(strict=True)",
            "sys.path.insert(0, str(root / 'src'))",
            "import sr_f6.confirm_registration",
            "import sr_f6.confirm_data",
            "import sr_f6.confirm_runtime",
            "import sr_f6.confirm_dispatch",
            "import sr_f6.confirm_census",
            "import sr_f6.confirm_evidence",
            "import sr_f6.confirm_analysis",
            "from sr_f1.contract import PACKAGE, load_config, _reference",
            "assert PACKAGE == root / 'docs/sr_f1/package'",
            "assert Path(_reference.__file__).is_relative_to(PACKAGE)",
            "assert load_config()",
            "packages = {p.name for p in (root / 'src').iterdir() if p.is_dir()}",
            "for name, module in tuple(sys.modules.items()):",
            "    if name.split('.')[0] in packages:",
            "        path = getattr(module, '__file__', None)",
            "        if path: assert Path(path).resolve().is_relative_to(root / 'src'), name",
            "print('CONFIRM_PRODUCTION_IMPORTS_RESOLVED; model_calls=0; optimizer_updates=0')",
        )
    )
    process = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(deployment)],
        cwd=deployment,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return dict(
        status="PASS",
        stdout=process.stdout.strip(),
        stderr=process.stderr.strip(),
        model_calls=0,
        optimizer_updates=0,
    )


def _screen_state():
    """Cheap exact metadata inventory after a full frozen-source validation.

    A changed predecessor observation fails closed, rather than repeatedly
    rehashing its saved answers. This is a revalidation hint bound to the full
    content verification, not a substitute scientific identity.
    """
    base = _path(SCREEN_ROOT)
    registration = read(safe_evidence(base, "REGISTRATION.json"))
    paths = set()
    for name in ("REGISTRATION.json", "PREFLIGHT.json", "PREFLIGHT_SEAL.json"):
        paths.add(safe_evidence(base, name))
    for key in ("source_manifest", "config"):
        paths.add(safe_evidence(base, registration[key]["path"]))
    for refs in registration["stages"].values():
        paths.update(safe_evidence(base, ref["path"]) for ref in refs.values())
    manifest = _bound(base, registration["source_manifest"])
    preflight = read(safe_evidence(base, "PREFLIGHT.json"))
    for entries in (
        manifest["files"],
        registration["upstream_files"],
        registration["image_files"],
        preflight["verified_dependencies"],
    ):
        paths.update(_path(row["path"]) for row in entries)
    prior = registration["design_basis"]
    for key in ("registration", "assessment"):
        paths.add(safe_evidence(_path(prior["prior_root"]), prior[key]["path"]))
    for relative in ("STAGE1/SCREEN", "STAGE1/accounting", "STAGE1/runtime_identity"):
        directory = _path(base / relative)
        for path in directory.rglob("*"):
            _path(path)
            if path.is_file():
                paths.add(path)
    stage2 = _path(base / "STAGE2")
    for path in stage2.rglob("*"):
        _path(path)
    _require(
        not any(p.is_file() for p in stage2.rglob("*")),
        "Passed stage one cannot have stage-two executions",
    )
    return [
        dict(path=str(path), stat_cache_key=stat_cache_key(path.stat())) for path in sorted(paths)
    ]


def _screen_basis():
    base, deployment = _path(SCREEN_ROOT), _path(SCREEN_DEPLOYMENT)
    registration = read(safe_evidence(base, "REGISTRATION.json"))
    _require(
        registration.get("revision") == SCREEN_REVISION
        and registration.get("deployment") == str(deployment)
        and registration.get("root") == str(base)
        and registration.get("phase") == "INTERFACE_SCREEN_V3"
        and registration.get("training_allowed") is False,
        "Frozen screen producer differs",
    )
    before = _screen_state()
    script = "\n".join(
        (
            "import json, sys",
            "from pathlib import Path",
            "deployment = Path(sys.argv[1]).resolve(strict=True)",
            "sys.path.insert(0, str(deployment / 'src'))",
            "from sr_f6.interface_v3_screen import _stage1_assessment",
            "from sr_f6.interface_v3_registration import _actual_deployment",
            "assert _actual_deployment() == deployment",
            "report = _stage1_assessment(Path(sys.argv[2]))",
            "assert report['decision']['status'] == 'SCREEN_PASS'",
            "assert report['decision']['stage2_parents'] == []",
            "assert report['responses'] == 576",
            "assert report['optimizer_updates'] == 0 and report['training_allowed'] is False",
            "print(json.dumps({'status': 'SCREEN_PASS',",
            " 'records_sha256': report['records_sha256'], 'responses': report['responses'],",
            " 'saved_token_validation': report['saved_token_validation'],",
            " 'accounting': report['accounting'], 'decision': report['decision']}))",
        )
    )
    process = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(deployment), str(base)],
        cwd=deployment,
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    result = json.loads(process.stdout)
    _require(before == _screen_state(), "Predecessor changed during complete verification")
    _require(
        result.get("status") == "SCREEN_PASS"
        and result.get("responses") == 576
        and result.get("saved_token_validation", {}).get("status") == "PASS"
        and result["saved_token_validation"].get("checked_responses") == 576
        and result.get("decision", {}).get("stage2_parents") == [],
        "Complete original screen pass required",
    )
    return dict(
        producer_phase="INTERFACE_SCREEN_V3",
        root=str(base),
        deployment=str(deployment),
        revision=SCREEN_REVISION,
        status="SCREEN_PASS",
        verification=result,
        registration=_ref(base, "REGISTRATION.json"),
        assessment=_ref(base, "STAGE1/SCREEN/SCREEN_ASSESSMENT.json"),
        seal=_ref(base, "STAGE1/SCREEN/ASSESSMENT_SEAL.json"),
        records_sha256=result["records_sha256"],
        verified_state=before,
        method="isolated immutable screen producer; full raw/score/ledger reconstruction",
        model_calls=0,
        optimizer_updates=0,
        training_authorization_reused=False,
    )


def _verify_screen_basis(basis):
    _require(
        basis.get("producer_phase") == "INTERFACE_SCREEN_V3"
        and basis.get("root") == str(SCREEN_ROOT)
        and basis.get("deployment") == str(SCREEN_DEPLOYMENT)
        and basis.get("revision") == SCREEN_REVISION
        and basis.get("status") == "SCREEN_PASS"
        and basis.get("training_authorization_reused") is False,
        "Confirmation predecessor identity differs",
    )
    report = _bound(SCREEN_ROOT, basis["assessment"])
    _bound(SCREEN_ROOT, basis["registration"])
    _bound(SCREEN_ROOT, basis["seal"])
    _require(
        report.get("decision", {}).get("status") == "SCREEN_PASS"
        and report["decision"].get("stage2_parents") == []
        and report.get("responses") == 576
        and report.get("records_sha256") == basis["records_sha256"]
        and basis.get("verified_state") == _screen_state(),
        "Previously validated complete screen evidence changed",
    )


def historical_roots(binding):
    roots = [
        ORIGIN_ROOT,
        Path(binding["previous_run"]),
        PROJECT / "sr_f4_c_local_pilot_20261024",
        PROJECT / "sr_f4_consolidated_20261022",
    ]
    _require(len(set(roots)) == 4, "Four distinct historical input producers required")
    return [_path(path) for path in roots]


def _frozen_implementations(deployment, files):
    """Compare already-bound source hashes, not another model/data content scan."""
    screen = read(safe_evidence(SCREEN_ROOT, "REGISTRATION.json"))
    original = _bound(SCREEN_ROOT, screen["source_manifest"])
    prior = {row["path"]: row["sha256"] for row in original["files"]}
    current = {row["path"]: row["sha256"] for row in files}
    result = {}
    for relative in FROZEN_IMPLEMENTATIONS:
        expected = prior.get(str(SCREEN_DEPLOYMENT / relative))
        _require(
            expected is not None and current.get(str(deployment / relative)) == expected,
            "Confirmed v3 prompt or original mathematical implementation changed: " + relative,
        )
        result[relative] = expected
    return result


def prompt_transition():
    return dict(
        version=PROMPT_VERSION,
        constructor="sr_f6.interface_v3_protocol.public_prompt(categories, query)",
        public_inputs=["world.categories", "query"],
        structured_splits=[
            SPLIT,
            "DEV_PILOT",
            "TRAIN",
            "DEV_EVAL",
            "TEST_ID",
            "TEST_COMPOSITION",
            "TEST_CLASSOUT",
        ],
        unchanged_protocols=["NATURAL_INTERFACE", "LEGACY", "CHARTQA"],
        preserve=[
            "world",
            "image",
            "qid",
            "root_id",
            "factor",
            "seeds",
            "arms",
            "lambda",
            "budgets",
            "strict_scorer",
            "stop_rule",
        ],
        future_inputs="separate immutable overlay before any later phase output",
        old_data_modified=False,
        future_model_access_authorized=False,
        parent_and_trained_endpoint_protocol="same frozen v3 prompt for each structured panel",
        test_results_read=False,
        selection_by_outputs=False,
    )


def make_plan(public, private, parent_hashes):
    _require(
        isinstance(private, dict) and set(parent_hashes) == set(PARENTS),
        "Exact private tasks and three parents required",
    )
    ordered = sorted(public, key=lambda row: row["qid"])
    expected = {
        f"{SPLIT}_{f}_{i:04d}_{p}_v{v}"
        for f in FAMILIES
        for i in range(8)
        for p in ("P0", "P1")
        for v in range(4)
    }
    _require(
        len(ordered) == len(private) == 384
        and {r["qid"] for r in ordered} == set(private) == expected,
        "Exact independent 384-question matrix required",
    )
    questions, attempts = [], []
    for item in ordered:
        task = private[item["qid"]]
        _require(
            item.get("split") == SPLIT
            and item.get("prompt_version") == PROMPT_VERSION
            and all(
                item.get(k) == task.get(k)
                for k in ("qid", "root_id", "family", "factor", "variant", "split")
            )
            and set(item["model_input"]) == {"prompt", "image_file", "image_sha256"},
            "Confirmation input identity or public boundary differs",
        )
        questions.append(
            dict(
                qid=item["qid"],
                public_sha256=digest(item),
                private_sha256=digest(task),
                model_input_sha256=digest(item["model_input"]),
            )
        )
    for parent in PARENTS:
        _require(
            isinstance(parent_hashes[parent], str)
            and re.fullmatch(r"[0-9a-f]{64}", parent_hashes[parent]),
            "Actual parent content hashes required",
        )
        for item in ordered:
            for draw in range(8):
                attempts.append(
                    dict(
                        parent=parent,
                        **{k: item[k] for k in ("qid", "root_id", "family", "factor", "variant")},
                        draw=draw,
                        seed=seed_for(
                            MASTER_SEED,
                            VERSION,
                            "confirm",
                            parent,
                            item["root_id"],
                            item["variant"],
                            draw,
                        ),
                        model_input_sha256=digest(item["model_input"]),
                    )
                )
    groups = []
    for parent in PARENTS:
        for item in ordered:
            groups.append(
                dict(
                    parent=parent,
                    qid=item["qid"],
                    group_id=f"{parent}/{item['qid']}",
                    seeds=[
                        seed_for(
                            MASTER_SEED,
                            VERSION,
                            "confirm",
                            parent,
                            item["root_id"],
                            item["variant"],
                            draw,
                        )
                        for draw in range(8)
                    ],
                    shuffle_seed=seed_for(
                        MASTER_SEED,
                        "census_shuffle",
                        parent,
                        item["factor"],
                        f"{parent}/{item['qid']}",
                    ),
                )
            )
    workers = []
    for parent in PARENTS:
        for shard in (0, 1):
            selected = ordered[shard::2]
            workers.append(
                dict(
                    worker_id=f"{parent}-s{shard}-of-2",
                    parent=parent,
                    shard=shard,
                    shards=2,
                    qids=[r["qid"] for r in selected],
                    first_qid=selected[0]["qid"],
                    groups=192,
                    responses=1536,
                    alignment_sequences=8,
                )
            )
    return dict(
        schema="srf6-confirm-plan-v1",
        version=VERSION,
        phase=PHASE,
        source_split=SPLIT,
        master_seed=MASTER_SEED,
        prompt_version=PROMPT_VERSION,
        parents=list(PARENTS),
        factors=["P0", "P1"],
        root_indices=list(range(8)),
        root_count=48,
        engineered_anchor_roots=18,
        questions_per_parent=384,
        draws_per_question=8,
        expected_response_rows=9216,
        parent_content_hashes=dict(parent_hashes),
        parent_policy_hashes=dict(parent_hashes),
        seed_rule="seed_for(20261028,'srf6-confirm-v3','confirm',parent,root_id,variant,draw)",
        seed_excludes_factor=True,
        shuffle_seed_rule="seed_for(20261028,'census_shuffle',parent,factor,f'{parent}/{qid}')",
        credit_algorithms=["J", "COND", "GLOBAL", "SHUF"],
        questions=questions,
        attempts=attempts,
        groups=groups,
        workers=workers,
        first_g8_replay=dict(
            sequences_per_worker=8,
            workers=6,
            total_sequences=48,
            extra_generated_responses=0,
            optimizer_updates=0,
            mean_absolute_difference_max=0.01,
            max_absolute_difference_max=5.0,
            tis_truncated_fraction_max=0.01,
        ),
        gates=dict(
            observability_cells=18,
            per_parent_factor_denominator=1536,
            each_count_at_least={k: 1383 for k in ("V", "S_known", "B_known")},
            unknown_in_denominator=True,
            original_cond_shuf_support="UNCHANGED",
            original_train_roots=288,
            train_mean_coverage_gap_min=0.05,
            train_strict_gain_roots_min=96,
            new_independent_response_review=360,
            review_seed=2026102701,
            actual_numerical_admission="REQUIRED_SEPARATELY",
        ),
        followup_requirements=dict(
            independent_response_review=dict(
                count=360,
                seed=2026102701,
                new_original_answers_only=True,
                authorization="SEPARATE_INDEPENDENT_REVIEW_REQUIRED",
            ),
            numerical_replay_selection=dict(
                qid=ordered[0]["qid"],
                parents=list(PARENTS),
                draws=list(range(8)),
                sequences=24,
                new_generated_responses=0,
                optimizer_updates=0,
                authorization="SEPARATE_ACTUAL_NUMERICAL_ADMISSION_REQUIRED",
            ),
        ),
        prompt_transition=prompt_transition(),
        on_partial_or_technical_failure="PRESERVE_NO_AUTOMATIC_RESAMPLING",
        on_scientific_failure="STOP_PRESERVE_ALL_NEGATIVE_RESULTS",
        generation_allowed=False,
        training_allowed=False,
        evaluation_allowed=False,
        optimizer_updates=0,
    )


def register(root, revision):
    from . import confirm_data
    from .contract import validate_config
    from .runtime import verify_binding_inputs

    root = _root(root, exists=False)
    _require(not root.exists(), "Preserve an existing confirmation namespace")
    origin, deployment = _path(ORIGIN_ROOT), _deployment(revision)
    resources = old.validate_deployment_assets(deployment)
    imports = check_deployment_imports(deployment)
    config = read(safe_evidence(origin, "CONFIG.json"))
    validate_config(config)
    _require(
        file_hash(origin / "IDENTITY.json") == old.ORIGIN_IDENTITY_SHA256
        and file_hash(origin / "PHASES/PILOT_V1/IDENTITY.json") == old.PILOT_IDENTITY_SHA256,
        "Original producer identity differs",
    )
    old._validate_producer(origin, deployment)
    binding, _ = verify_binding_inputs(origin, config)
    for parent in PARENTS:
        verify_binding_inputs(origin, config, parent)
    basis = _screen_basis()
    roots = historical_roots(binding)
    root.mkdir()
    with (root / "CONFIG.json").open("xb") as stream:
        stream.write((origin / "CONFIG.json").read_bytes())
    write_once(
        root / "REGISTRATION_STARTED.json",
        dict(
            phase=PHASE,
            revision=revision,
            at=now(),
            original_root=str(origin),
            predecessor=basis,
            generation_allowed=False,
            training_allowed=False,
            model_calls=0,
        ),
    )
    build = confirm_data.build(root, historical_roots=roots)
    public, private = confirm_data.load_inputs(root)
    parent_hashes = {p: binding["parents"][p]["content_sha256"] for p in PARENTS}
    plan = make_plan(public, private, parent_hashes)
    write_once(root / "CONFIRM_SPEC_FROZEN.json", plan)
    files = [inventory_file(deployment / relative) for relative in _source_paths(deployment)]
    frozen_implementations = _frozen_implementations(deployment, files)
    write_once(root / "SOURCE_MANIFEST.json", dict(revision=revision, files=files))
    upstream_names = tuple(
        dict.fromkeys(
            (*old.SOURCE_REFERENCES, "DATA/CPU_AUDIT.json", "DATA/ROOT_CERTIFICATES_PRIVATE.jsonl")
        )
    )
    record = dict(
        schema=REGISTRATION_SCHEMA,
        phase=PHASE,
        version=VERSION,
        status="BOUND_AWAITING_ACTUAL_PROCESSOR_AND_NEW_VISUAL_REVIEW",
        root=str(root),
        origin_root=str(origin),
        source_root=str(origin),
        deployment=str(deployment),
        revision=revision,
        prompt_version=PROMPT_VERSION,
        deployment_resources=resources,
        deployment_import_check=imports,
        frozen_implementations=frozen_implementations,
        original_identity_sha256=old.ORIGIN_IDENTITY_SHA256,
        historical_pilot_identity_sha256=old.PILOT_IDENTITY_SHA256,
        predecessor_screen=basis,
        historical_v2_status="SCREEN_FAIL_STOP_UNCHANGED",
        old_census_raw_status="DELETED_BY_USER_NOT_REUSED",
        source_manifest=_ref(root, "SOURCE_MANIFEST.json"),
        source_manifest_sha256=file_hash(root / "SOURCE_MANIFEST.json"),
        config=_ref(root, "CONFIG.json"),
        config_canonical_sha256=digest(config),
        data_manifest=_ref(root, "DATA/MANIFEST.json"),
        data_sha256=file_hash(root / "DATA/MANIFEST.json"),
        spec=_ref(root, "CONFIRM_SPEC_FROZEN.json"),
        parent_content_hashes=parent_hashes,
        upstream_files=[inventory_file(origin / relative) for relative in upstream_names],
        historical_roots=[str(path) for path in roots],
        data_build_status=build.get("status"),
        prompt_transition=prompt_transition(),
        expected_responses=9216,
        generation_allowed=False,
        training_allowed=False,
        evaluation_allowed=False,
        optimizer_updates=0,
    )
    _verify_screen_basis(basis)
    write_once(root / "REGISTRATION.json", record)
    return record


def verify_registered(root, parent=None, require_preflight=False):
    from . import confirm_data
    from .contract import validate_config
    from .runtime import _verify_module_origins, verify_binding_inputs

    _require(parent is None or parent in PARENTS, "Unknown confirmation parent")
    _require(type(require_preflight) is bool, "Preflight requirement must be explicit")
    root = _root(root)
    record = read(safe_evidence(root, "REGISTRATION.json"))
    _require(
        record.get("schema") == REGISTRATION_SCHEMA
        and record.get("phase") == PHASE
        and record.get("version") == VERSION
        and record.get("root") == str(root)
        and record.get("origin_root") == record.get("source_root") == str(ORIGIN_ROOT)
        and record.get("prompt_version") == PROMPT_VERSION
        and record.get("generation_allowed") is False
        and record.get("training_allowed") is False
        and record.get("evaluation_allowed") is False
        and type(record.get("optimizer_updates")) is int
        and record["optimizer_updates"] == 0
        and record.get("expected_responses") == 9216
        and record.get("historical_v2_status") == "SCREEN_FAIL_STOP_UNCHANGED"
        and record.get("old_census_raw_status") == "DELETED_BY_USER_NOT_REUSED"
        and record.get("prompt_transition") == prompt_transition(),
        "Independent confirmation scope differs",
    )
    origin, deployment = _path(ORIGIN_ROOT), _deployment(record["revision"])
    _require(record["deployment"] == str(deployment), "Actual executing deployment differs")
    _require(
        record["deployment_resources"] == old.validate_deployment_assets(deployment)
        and record.get("deployment_import_check", {}).get("status") == "PASS",
        "Complete actual deployment assets and imports required",
    )
    manifest = _bound(root, record["source_manifest"])
    _require(
        manifest["revision"] == record["revision"]
        and record["source_manifest_sha256"] == record["source_manifest"]["sha256"]
        and len(manifest["files"]) == len({r["path"] for r in manifest["files"]})
        and {r["path"] for r in manifest["files"]}
        == {str(deployment / p) for p in _source_paths(deployment)},
        "Complete actual source inventory differs",
    )
    source_check = verify_bound_files(manifest["files"])
    _require(
        record.get("frozen_implementations")
        == _frozen_implementations(deployment, manifest["files"]),
        "Frozen v3 prompt and mathematics binding differs",
    )
    _verify_module_origins(
        deployment,
        {
            "files": [
                {"path": str(Path(r["path"]).relative_to(deployment))} for r in manifest["files"]
            ]
        },
        tuple(sys.modules.values()),
    )
    upstream_names = set(
        (*old.SOURCE_REFERENCES, "DATA/CPU_AUDIT.json", "DATA/ROOT_CERTIFICATES_PRIVATE.jsonl")
    )
    _require(
        len(record["upstream_files"]) == len(upstream_names)
        and {r["path"] for r in record["upstream_files"]}
        == {str(origin / p) for p in upstream_names},
        "Original provenance inventory differs",
    )
    verify_bound_files(record["upstream_files"])
    _require(
        record["original_identity_sha256"] == old.ORIGIN_IDENTITY_SHA256
        and file_hash(origin / "IDENTITY.json") == old.ORIGIN_IDENTITY_SHA256
        and record["historical_pilot_identity_sha256"] == old.PILOT_IDENTITY_SHA256
        and file_hash(origin / "PHASES/PILOT_V1/IDENTITY.json") == old.PILOT_IDENTITY_SHA256,
        "Original producer identity changed",
    )
    config = _bound(root, record["config"])
    validate_config(config)
    _require(
        (root / "CONFIG.json").read_bytes() == (origin / "CONFIG.json").read_bytes()
        and digest(config) == record["config_canonical_sha256"],
        "Original configuration bytes changed",
    )
    old._validate_producer(origin, deployment)
    _verify_screen_basis(record["predecessor_screen"])
    _bound(root, record["data_manifest"])
    _require(record["data_sha256"] == record["data_manifest"]["sha256"], "Data identity differs")
    data_check = confirm_data.verify_data(root)
    public, private = confirm_data.load_inputs(root)
    public = sorted(public, key=lambda row: row["qid"])
    binding, model_check = verify_binding_inputs(origin, config, parent)
    if parent is None:
        for selected in PARENTS:
            verify_binding_inputs(origin, config, selected)
    parent_hashes = {p: binding["parents"][p]["content_sha256"] for p in PARENTS}
    _require(
        record["parent_content_hashes"] == parent_hashes
        and record["historical_roots"] == [str(p) for p in historical_roots(binding)],
        "Frozen parent or history identity changed",
    )
    plan = _bound(root, record["spec"])
    _require(
        plan == make_plan(public, private, parent_hashes),
        "Frozen confirmation seeds, partitions, or conditions changed",
    )
    pin = file_hash(root / "REGISTRATION.json")
    preflight = None
    if require_preflight:
        seal = read(safe_evidence(root, "PREFLIGHT/SEAL.json"))
        _require(
            seal.get("schema") == SEAL_SCHEMA
            and seal.get("phase") == PHASE
            and seal.get("registration_sha256") == pin
            and seal.get("scope") == "REGISTERED_9216_CONFIRMATION_ONLY"
            and seal.get("generation_allowed") is True
            and seal.get("training_allowed") is False
            and seal.get("evaluation_allowed") is False,
            "New actual processor and visual seal required",
        )
        preflight = _bound(root, seal["processor"])
        _validate_processor(preflight, pin, public)
        _validate_native(
            root, preflight["native_regression"], pin, record["source_manifest_sha256"]
        )
        visual = _bound(root, seal["visual"])
        _validate_visual(root, visual, preflight, seal["processor"])
        verify_bound_files(preflight["verified_dependencies"])
        preflight = {**preflight, "visual": visual, "seal": seal}
    return dict(
        root=root,
        registration_root=root,
        input_root=root,
        source_root=origin,
        origin_root=origin,
        deployment=deployment,
        registration=record,
        registration_sha256=pin,
        source_manifest=manifest,
        config=config,
        source_binding=binding,
        public=public,
        private=private,
        tasks=private,
        plan=plan,
        spec=plan,
        spec_sha256=record["spec"]["sha256"],
        spec_object_sha256=digest(plan),
        preflight=preflight,
        verification=dict(source=source_check, data=data_check, model=model_check),
        provenance_inputs=[
            record[k] for k in ("config", "source_manifest", "data_manifest", "spec")
        ],
    )


def _validate_processor(receipt, pin, public):
    _require(
        receipt.get("schema") == PREFLIGHT_SCHEMA
        and receipt.get("phase") == PHASE
        and receipt.get("status") == "ACTUAL_PROCESSOR_COMPLETE_VISUAL_REVIEW_PENDING"
        and receipt.get("execution_origin") == "SERVER_ACTUAL"
        and receipt.get("registration_sha256") == pin
        and type(receipt.get("model_calls")) is int
        and receipt["model_calls"] == 0
        and type(receipt.get("optimizer_updates")) is int
        and receipt["optimizer_updates"] == 0
        and receipt.get("training_allowed") is False
        and receipt.get("generation_allowed") is False
        and receipt.get("new_visual_review_required") is True,
        "Complete actual confirmation processor evidence required",
    )
    inputs = receipt.get("inputs", [])
    _require(
        len(inputs) == len(public) == 384
        and len({r["qid"] for r in inputs}) == 384
        and {r["qid"] for r in inputs} == {r["qid"] for r in public},
        "All 384 actual input routes required",
    )
    by_qid = {r["qid"]: r for r in inputs}
    for row in public:
        actual = by_qid[row["qid"]]
        _require(
            actual.get("model_input_sha256") == digest(row["model_input"]),
            "Actual route differs from frozen public input",
        )
        old._validate_route(actual["routing"], row["model_input"], receipt["identity"])
    images = receipt.get("images", [])
    expected = {r["qid"] for r in public if r["factor"] == "P1" and r["variant"] == 0}
    _require(
        len(images) == len(expected) == 48 and {r["qid"] for r in images} == expected,
        "Exactly the 48 new registered anchors require actual pixel artifacts",
    )
    public_map = {r["qid"]: r for r in public}
    for image in images:
        item = public_map[image["qid"]]
        _require(
            image.get("source_sha256") == item["model_input"]["image_sha256"]
            and image.get("source_file") == item["model_input"]["image_file"]
            and image.get("routing") == by_qid[item["qid"]]["routing"]
            and isinstance(image.get("processed_rgb_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", image["processed_rgb_sha256"]),
            "New processor image identity differs",
        )
    fixtures = receipt.get("tokenizer_fixtures", [])
    _require(
        len(fixtures) == len(TOKENIZER_FIXTURES)
        and all(
            row.get("text") == text
            and row.get("exact_roundtrip") is True
            and isinstance(row.get("token_ids"), list)
            and row["token_ids"]
            and all(type(token) is int and token >= 0 for token in row["token_ids"])
            for row, text in zip(fixtures, TOKENIZER_FIXTURES, strict=True)
        ),
        "Actual tokenizer roundtrips required",
    )
    _require(bool(receipt.get("verified_dependencies")), "Processor dependencies required")
    _require(
        isinstance(receipt.get("native_regression"), dict),
        "Actual server native regression reference required",
    )


def _validate_native(root, reference, pin, source_pin):
    report = _bound(root, reference)
    _require(
        report.get("schema") == "srf6-confirm-native-regression-v1"
        and report.get("status") == "PASS"
        and report.get("returncode") == 0
        and report.get("phase") == PHASE
        and report.get("registration_sha256") == pin
        and report.get("source_manifest_sha256") == source_pin
        and report.get("execution_origin") == "SERVER_ACTUAL"
        and type(report.get("model_calls")) is int
        and report["model_calls"] == 0
        and type(report.get("optimizer_updates")) is int
        and report["optimizer_updates"] == 0,
        "Actual new-source CPU regression differs",
    )
    log = report["log"]
    _require(
        file_hash(safe_evidence(root, log["path"])) == log["sha256"],
        "Actual native regression log changed",
    )
    return report


def _native_regression(context):
    root = context["root"]
    path = root / "PREFLIGHT/NATIVE_REGRESSION.json"
    _require(not path.exists(), "Preserve an existing native regression result")
    command = [
        sys.executable,
        "-B",
        "-m",
        "pytest",
        "--import-mode=importlib",
        "-q",
        "tests/sr_f6/test_measurement.py",
        "tests/sr_f6/test_objective.py",
        "tests/sr_f6/test_credit.py",
        "tests/sr_f6/test_runtime.py",
        "tests/sr_f6/test_confirm_data.py",
        "tests/sr_f6/test_confirm_registration.py",
        "tests/sr_f6/test_confirm_runtime.py",
    ]
    process = subprocess.run(
        command, cwd=context["deployment"], capture_output=True, text=True, check=False, timeout=600
    )
    log = root / "PREFLIGHT/NATIVE_REGRESSION.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x") as stream:
        stream.write(process.stdout + process.stderr)
    report = dict(
        schema="srf6-confirm-native-regression-v1",
        phase=PHASE,
        status="PASS" if process.returncode == 0 else "FAIL",
        at=now(),
        returncode=process.returncode,
        command=command,
        registration_sha256=context["registration_sha256"],
        source_manifest_sha256=context["registration"]["source_manifest_sha256"],
        execution_origin="SERVER_ACTUAL",
        log=_ref(root, str(log.relative_to(root))),
        model_calls=0,
        optimizer_updates=0,
        scope="CPU code and tensor regression; actual GPU numerical gate remains separate",
    )
    write_once(path, report)
    reference = _ref(root, str(path.relative_to(root)))
    _validate_native(
        root,
        reference,
        context["registration_sha256"],
        context["registration"]["source_manifest_sha256"],
    )
    return reference


def _validate_visual(root, visual, processor, processor_ref):
    _require(
        visual.get("schema") == "srf6-confirm-visual-review-v1"
        and visual.get("phase") == PHASE
        and visual.get("status") == "PASS"
        and visual.get("reviewer_type") in {"HUMAN", "MODEL_VISUAL"}
        and isinstance(visual.get("reviewer_identity"), str)
        and visual["reviewer_identity"].strip()
        and visual.get("independent_of_processor_program") is True
        and visual.get("processor_receipt") == processor_ref
        and visual.get("registration_sha256") == processor["registration_sha256"]
        and visual.get("original_screen_visual_reused") is False,
        "New independent visual judgments are required; old images cannot authorize this data",
    )
    images = {row["qid"]: row for row in processor["images"]}
    reviews = visual.get("items", [])
    _require(
        len(reviews) == 48
        and len({r["qid"] for r in reviews}) == 48
        and {r["qid"] for r in reviews} == set(images),
        "All 48 new images need separate observations",
    )
    for item in reviews:
        image = images[item["qid"]]
        _require(
            item.get("status") == "PASS"
            and isinstance(item.get("observation"), str)
            and item["observation"].strip()
            and item.get("source_sha256") == image["source_sha256"]
            and item.get("processed_file_sha256") == image["processed_file_sha256"]
            and item.get("labels_readable") is True
            and item.get("bars_and_legend_readable") is True
            and item.get("no_clipping") is True,
            "Incomplete, failed, or stale visual judgment",
        )
        _require(
            file_hash(safe_evidence(root, image["processed_file"]))
            == image["processed_file_sha256"],
            "Reviewed processed image changed",
        )
    evidence = visual.get("review_evidence")
    _require(
        isinstance(evidence, dict) and set(evidence) == {"path", "sha256"},
        "Actual visual inspection evidence required",
    )
    _require(
        file_hash(safe_evidence(root, evidence["path"])) == evidence["sha256"],
        "Visual inspection evidence changed",
    )


def preflight(root):
    from mm_dev.data import reconstruct_rgb
    from sr_f5.runtime import NUMERICS_V2

    from .runtime import Runtime

    root = _root(root)
    _require(
        not (root / "PREFLIGHT/PROCESSOR.json").exists(),
        "Preserve the existing actual preflight result",
    )
    pin = file_hash(safe_evidence(root, "REGISTRATION.json"))
    receipt = dict(
        schema=PREFLIGHT_SCHEMA,
        phase=PHASE,
        status="FAIL",
        at=now(),
        host=socket.gethostname(),
        execution_origin="SERVER_ACTUAL",
        registration_sha256=pin,
        model_calls=0,
        optimizer_updates=0,
        training_allowed=False,
        generation_allowed=False,
        new_visual_review_required=True,
    )
    try:
        context = verify_registered(root)
        receipt["native_regression"] = _native_regression(context)
        runtime = Runtime.processor_only(
            context["config"]["model"]["model_path"], protocol_revision="v2"
        )
        runtime._srf5_numerics = dict(NUMERICS_V2)
        plan = read(safe_evidence(root, "DATA/PROCESSOR_REVIEW_PLAN.json"))
        expected = {
            r["qid"] for r in context["public"] if r["factor"] == "P1" and r["variant"] == 0
        }
        _require(
            plan.get("count") == 48
            and len(plan.get("items", [])) == 48
            and {r["qid"] for r in plan["items"]} == expected,
            "Exact new anchor review plan required",
        )
        destination = root / "PREFLIGHT/processor"
        _require(not destination.exists(), "Preserve existing processor pixels")
        destination.mkdir(parents=True)
        inputs, images = [], []
        for item in context["public"]:
            model = item["model_input"]
            prepared = runtime.prepare_input(model, root, protocol="structured_v4")
            route = copy.deepcopy(prepared["routing"])
            old._validate_route(route, model, runtime.identity)
            ids = prepared["inputs"]["input_ids"]
            _require(
                ids.ndim == 2 and ids.shape[0] == 1 and ids.shape[-1] > 0,
                "Actual prompt token tensor missing",
            )
            inputs.append(dict(qid=item["qid"], model_input_sha256=digest(model), routing=route))
            if item["qid"] in expected:
                rgb = reconstruct_rgb(
                    prepared["inputs"]["pixel_values"],
                    route["image_grid_thw"],
                    runtime.processor.image_processor,
                )
                path = destination / f"{item['qid']}.png"
                rgb.save(path)
                images.append(
                    dict(
                        qid=item["qid"],
                        family=item["family"],
                        source_file=model["image_file"],
                        source_sha256=model["image_sha256"],
                        processed_file=str(path.relative_to(root)),
                        processed_file_sha256=file_hash(path),
                        processed_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                        routing=route,
                        reviewer_type="PROGRAM",
                        visual_review="NOT_RUN",
                    )
                )
        tokenizer, fixtures = runtime.processor.tokenizer, []
        for text in TOKENIZER_FIXTURES:
            ids = tokenizer.encode(text, add_special_tokens=False)
            _require(
                ids
                and tokenizer.decode(
                    ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                == text,
                "Actual tokenizer roundtrip failed",
            )
            fixtures.append(dict(text=text, token_ids=ids, exact_roundtrip=True))
        dependencies = context["source_binding"]["model_files"]
        receipt.update(
            status="ACTUAL_PROCESSOR_COMPLETE_VISUAL_REVIEW_PENDING",
            identity=copy.deepcopy(runtime.identity),
            inputs=inputs,
            images=images,
            tokenizer_fixtures=fixtures,
            verified_dependencies=dependencies,
        )
        _validate_processor(receipt, pin, context["public"])
        verify_bound_files(dependencies)
        after = verify_registered(root)
        _require(after["registration_sha256"] == pin, "Registration changed during preflight")
    except Exception as exc:
        receipt.update(status="FAIL", error=dict(type=type(exc).__name__, message=str(exc)))
        write_once(root / "PREFLIGHT/PROCESSOR.json", receipt)
        raise
    write_once(root / "PREFLIGHT/PROCESSOR.json", receipt)
    return receipt


def seal_preflight(root, visual_ref):
    root = _root(root)
    _require(not (root / "PREFLIGHT/SEAL.json").exists(), "Preserve the existing preflight seal")
    context = verify_registered(root)
    processor_ref = _ref(root, "PREFLIGHT/PROCESSOR.json")
    processor = _bound(root, processor_ref)
    _validate_processor(processor, context["registration_sha256"], context["public"])
    _validate_native(
        root,
        processor["native_regression"],
        context["registration_sha256"],
        context["registration"]["source_manifest_sha256"],
    )
    visual = _bound(root, visual_ref)
    _validate_visual(root, visual, processor, processor_ref)
    verify_bound_files(processor["verified_dependencies"])
    seal = dict(
        schema=SEAL_SCHEMA,
        phase=PHASE,
        registration_sha256=context["registration_sha256"],
        processor=processor_ref,
        visual=visual_ref,
        scope="REGISTERED_9216_CONFIRMATION_ONLY",
        generation_allowed=True,
        training_allowed=False,
        evaluation_allowed=False,
    )
    write_once(root / "PREFLIGHT/SEAL.json", seal)
    verify_registered(root, require_preflight=True)
    return seal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    register_cmd = sub.add_parser("register")
    register_cmd.add_argument("--root", type=Path, required=True)
    register_cmd.add_argument("--revision", required=True)
    sub.add_parser("preflight").add_argument("--root", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--parent", choices=PARENTS)
    verify.add_argument("--require-preflight", action="store_true")
    seal = sub.add_parser("seal")
    seal.add_argument("--root", type=Path, required=True)
    seal.add_argument("--visual-path", required=True)
    seal.add_argument("--visual-sha256", required=True)
    args = parser.parse_args()
    if args.command == "register":
        result = register(args.root, args.revision)
    elif args.command == "preflight":
        result = preflight(args.root)
    elif args.command == "seal":
        result = seal_preflight(args.root, dict(path=args.visual_path, sha256=args.visual_sha256))
    else:
        context = verify_registered(args.root, args.parent, args.require_preflight)
        result = dict(
            status="VERIFIED_REGISTERED_CONFIRMATION",
            phase=PHASE,
            registration_sha256=context["registration_sha256"],
            training_allowed=False,
        )
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("status", "phase", "registration_sha256", "training_allowed")
                if k in result
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
