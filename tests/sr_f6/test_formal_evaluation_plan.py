"""SYNTHETIC_ONLY: real provenance/plan consumers, no real data or model calls.

Extend the existing CPU source-chain producer before it freezes S0/PILOT/FORMAL.
Only fixture file writers/load-location discovery are patched. Phase validation,
PILOT provenance, data/source hashes, parent tensor evidence, binding and all plan
consumers run normally; these fixtures can never certify actual model execution.
"""

from __future__ import annotations

import copy
import errno
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from sr_f1.chartqa import PROMPT
from sr_f3_ce.prompts import CONTRACT_V2
from sr_f6 import formal_evaluation_plan as plan
from sr_f6 import registration, review_package, source_binding
from sr_f6.common import file_hash
from sr_f6.data import public_prompt
from srf6_ref.core import FAMILIES, seed_for

# Explicit sibling loading also works with pytest --import-mode=importlib. No
# tests-directory sys.path or prior collection of the sibling module is needed.
_fixture_spec = importlib.util.spec_from_file_location(
    "_srf6_formal_registration_fixture", Path(__file__).with_name("test_formal_registration.py")
)
source_fixture = importlib.util.module_from_spec(_fixture_spec)
sys.modules[_fixture_spec.name] = source_fixture
_fixture_spec.loader.exec_module(source_fixture)
source_factory, prototype = source_fixture.factory, source_fixture.prototype
write, read = source_fixture.write, source_fixture.read


def old_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@pytest.fixture
def factory(source_factory, monkeypatch, prototype):
    repo = Path(__file__).parents[2]
    original_freeze = registration.freeze_pilot_registration
    fixture_write = source_fixture.write

    def create(
        *,
        mutate_data=None,
        mutate_retention=None,
        mutate_sources=None,
        mutate_pilot=None,
        anchored=True,
    ):
        additions = {}

        def produce(root):
            project = root.parent
            current = project / ("sr_f6_code_" + "c" * 7)
            old_code = project / "SYNTHETIC_F5_CODE"
            old_run = project / "SYNTHETIC_F5_RUN"
            image_root = project / "SYNTHETIC_F4" / "DATA"
            for relative in plan.SOURCE_DEPENDENCIES:
                target = current / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(repo / relative, target)
                if relative in (*plan.INTERFACE_FILES, *plan.INHERITED_FILES):
                    target = old_code / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(repo / relative, target)
            for panel, split in plan.SPLITS.items():
                if panel in ("DEV_EVAL", "NATURAL_INTERFACE"):
                    continue
                private, public = [], []
                for family in FAMILIES:
                    for index in range(plan.PANELS[panel][0] // 24):
                        rid = f"{split}_{family}_{index:04d}"
                        for view in range(4):
                            task = dict(
                                copy.deepcopy(prototype),
                                qid=f"{rid}_P0_v{view}",
                                root_id=rid,
                                split=split,
                                family=family,
                                factor="P0",
                                variant=view,
                            )
                            private.append(task)
                            public.append(
                                dict(
                                    **{
                                        k: task[k]
                                        for k in (
                                            "qid",
                                            "root_id",
                                            "split",
                                            "family",
                                            "factor",
                                            "variant",
                                        )
                                    },
                                    protocol="structured_v4",
                                    model_input=dict(
                                        prompt=public_prompt(task),
                                        image_file="DATA/images/synthetic.png",
                                        image_sha256=file_hash(root / "DATA/images/synthetic.png"),
                                    ),
                                )
                            )
                write(root, f"DATA/{split}/MODEL_INPUTS.jsonl", public, lines=True)
                write(root, f"DATA/{split}/TASKS_PRIVATE.jsonl", private, lines=True)
            ids = [
                json.loads(line)
                for line in (root / "DATA/TEST_ID/TASKS_PRIVATE.jsonl").read_text().splitlines()
            ]
            chosen = [t for f in FAMILIES for t in [t for t in ids if t["family"] == f][:16]]
            remainder = [t for t in ids if t not in chosen]
            remainder.sort(
                key=lambda t: (seed_for(2026102701, "natural_interface", t["qid"]), t["qid"])
            )
            natural, public = [], []
            for old in chosen + remainder[:32]:
                task = dict(
                    copy.deepcopy(old),
                    qid="NATURAL_" + old["qid"],
                    paired_id_qid=old["qid"],
                    split="NATURAL_INTERFACE",
                    protocol="natural_answer",
                )
                natural.append(task)
                public.append(
                    dict(
                        **{
                            k: task[k]
                            for k in ("qid", "root_id", "split", "family", "factor", "variant")
                        },
                        protocol="natural_answer",
                        model_input=dict(
                            prompt=public_prompt(task, natural=True),
                            image_file="DATA/images/synthetic.png",
                            image_sha256=file_hash(root / "DATA/images/synthetic.png"),
                        ),
                    )
                )
            write(root, "DATA/NATURAL_INTERFACE/TASKS_PRIVATE.jsonl", natural, lines=True)
            write(root, "DATA/NATURAL_INTERFACE/MODEL_INPUTS.jsonl", public, lines=True)
            if mutate_data:
                mutate_data(root)
            retention, image_inventory = {}, {}
            for panel, count in [("LEGACY", 128), ("CHARTQA", 250)]:
                tasks, routes, images = [], [], []
                for index in range(count):
                    qid = f"SYNTHETIC_{panel}_{index:04d}"
                    relative = f"images/{qid}.png"
                    image = image_root / relative
                    image.parent.mkdir(parents=True, exist_ok=True)
                    image.write_bytes(f"SYNTHETIC_NOT_PROCESSOR_EVIDENCE:{qid}".encode())
                    sha = file_hash(image)
                    if panel == "LEGACY":
                        family = ("CROSS", "INTERVAL", "THRESHOLD", "TOPK")[index // 32]
                        task = dict(
                            qid=qid,
                            root_id=f"SYNTHETIC_{family}_{index // 4}",
                            family=family,
                            variant=index % 4,
                            pool="TEST_ID",
                            chart="grouped_bar",
                            render_style="main",
                            answer=1,
                            world={"synthetic": True},
                            query={"synthetic": True},
                            prompt="Synthetic question.\n" + CONTRACT_V2,
                        )
                        route = dict(
                            qid=qid,
                            image_file=relative,
                            image_sha256=sha,
                            chart="grouped_bar",
                            render_style="main",
                            width=1024,
                            height=768,
                        )
                    else:
                        task = dict(
                            qid=qid,
                            root_id=qid,
                            pool="CHARTQA",
                            family="EXTERNAL",
                            answer="1",
                            question="Synthetic question?",
                            image_sha256=sha,
                        )
                        route = dict(
                            qid=qid,
                            pool="CHARTQA",
                            prompt=PROMPT + "\n" + task["question"],
                            image_file=relative,
                            image_sha256=sha,
                        )
                    tasks.append(task)
                    routes.append(route)
                    images.append(
                        dict(source_binding.inventory_file(image, boundary=project), qid=qid)
                    )
                write(old_run, f"RETENTION/{panel}_TASKS.jsonl", tasks, lines=True)
                write(old_run, f"RETENTION/{panel}_SOURCE_ROUTES.jsonl", routes, lines=True)
                retention[panel] = tasks
                image_inventory[panel] = images
            frozen = dict(
                status="FROZEN_BEFORE_NEW_MODEL_OUTPUTS",
                source_root=str(image_root.parent),
                legacy_source=str(image_root / "TASKS.jsonl"),
                chartqa_source=str(project / "SYNTHETIC_CHARTQA_ORIGINAL.jsonl"),
                legacy_seed=2026102309,
                chartqa_seed=2026102310,
                legacy_questions=128,
                chartqa_questions=250,
                legacy_task_digest=old_digest(retention["LEGACY"]),
                chartqa_task_digest=old_digest(retention["CHARTQA"]),
                all_48_formal_models_share_same_subsets=True,
                model_outputs_used=False,
            )
            write(old_run, "RETENTION/FROZEN.json", frozen)
            if mutate_retention:
                mutate_retention(old_run, image_inventory)
            additions.update(
                source_files=[
                    source_binding.inventory_file(old_code / name, boundary=project)
                    for name in plan.INHERITED_FILES
                ],
                previous_source=str(old_code),
                previous_run=str(old_run),
                retention={
                    name: source_binding.inventory_file(
                        old_run / "RETENTION" / name, boundary=project
                    )
                    for name in plan.RETENTION_FILES
                },
                retention_images=image_inventory,
                retention_image_root=str(image_root),
            )
            receipt = dict(
                schema="srf6-retention-interface-source-binding-v1",
                status="BOUND_CONTENT_ONLY_NOT_AUTHORIZED",
                execution_origin="SERVER_ACTUAL_SOURCE_READ",
                source_deployment=str(old_code),
                files=[
                    dict(
                        source_binding.inventory_file(old_code / name, boundary=project),
                        relative=name,
                    )
                    for name in plan.INTERFACE_FILES
                ],
                model_calls=0,
                optimizer_updates=0,
                training_authorized=False,
                evaluation_authorized=False,
                formal_test_access=False,
                fixture_scope="SYNTHETIC_ONLY_NO_ACTUAL_SERVER_OR_MODEL_EVIDENCE",
            )
            write(project, plan.OPERATIONS_RECEIPT, receipt)
            if mutate_sources:
                mutate_sources(project, current, old_code)
            monkeypatch.setattr(plan, "_actual_deployment", lambda: current)
            monkeypatch.setattr(
                plan,
                "_loaded_module_paths",
                lambda: {
                    name: current / ("src/" + name.replace(".", "/") + ".py")
                    for name in ("sr_f1.chartqa", "sr_f4.evaluation", "sr_f3_ce.prompts")
                },
            )

        def complete_manifest(root, relative, value, **kwargs):
            if relative == "DATA/MANIFEST.json":
                value = dict(
                    value,
                    files=[
                        dict(
                            path=p.relative_to(root).as_posix(),
                            sha256=file_hash(p),
                            bytes=p.stat().st_size,
                        )
                        for p in sorted((root / "DATA").rglob("*"))
                        if p.is_file()
                    ],
                )
            return fixture_write(root, relative, value, **kwargs)

        def complete_binding(root, **kwargs):
            value = read(root, "SOURCE_BINDING.json")
            write(root, "SOURCE_BINDING.json", dict(value, **additions))
            return original_freeze(root, **kwargs)

        monkeypatch.setattr(source_fixture, "write", complete_manifest)
        monkeypatch.setattr(registration, "freeze_pilot_registration", complete_binding)
        return source_factory(mutate_data=produce, mutate_pilot=mutate_pilot, anchored=anchored)

    return create


@pytest.fixture
def case(factory):
    case = factory()
    result = plan.bind_retention_protocol(
        case.root, phase_context=case.context, pilot_spec_ref=case.pilot_spec_ref
    )
    case.binding_ref = result["binding_ref"]
    return case


def args(case):
    return dict(
        phase_context=case.context,
        pilot_spec_ref=case.pilot_spec_ref,
        retention_protocol_binding_ref=case.binding_ref,
    )


def build(case):
    return plan.build_formal_evaluation_plan(case.root, **args(case))


def rows(bundle, panel):
    return [json.loads(line) for line in bundle["files"][f"panels/{panel}.jsonl"].splitlines()]


def test_complete_real_provenance_construction_has_all_ids_and_no_authority(case):
    bundle = build(case)
    spec = bundle["spec"]
    assert bundle["published"] is False
    assert spec["budgets"] == dict(
        dev36_checkpoints=54,
        dev36_responses=5184,
        final_checkpoints=57,
        final_responses_per_checkpoint=6010,
        final_responses=342570,
        total_responses=347754,
        question_groups=2138,
        stored_draw_seeds=6106,
    )
    assert len(bundle["files"]) == 11
    assert all(bundle[k] is False and spec[k] is False for k in plan.AUTHORIZATION)
    assert spec["execution_status"] == "NOT_RUN"
    assert not (case.root / plan.formal.DIRECTORY).exists()
    assert spec["parent_tensors"] == case.pilot_result["spec"]["parent_tensors"]
    all_rows = [r for p in plan.PANELS for r in rows(bundle, p)]
    assert len(all_rows) == 2138 and sum(len(r["seeds"]) for r in all_rows) == 6106
    for p, (questions, draws) in plan.PANELS.items():
        data = rows(bundle, p)
        assert len(data) == questions and len({r["qid"] for r in data}) == questions
        assert all(
            r["draws"] == draws
            and set(r["model_input"]) == {"prompt", "image_file", "image_sha256"}
            for r in data
        )
        assert spec["panels"][p]["protocol_profile"]["generation"] == plan.GENERATION
    assert spec["panels"]["CHARTQA"]["draws"] == 1
    assert spec["panels"]["CHARTQA"]["protocol_profile"]["generation"]["do_sample"] is True
    assert (
        "exact(frozen gold_answer)"
        in spec["panels"]["NATURAL_INTERFACE"]["protocol_profile"]["expected"]
    )
    assert set(spec["panels"]["CHARTQA"]["protocol_profile"]["scoring_calls"]) == {
        "historical_code",
        "v2",
    }
    assert spec["panels"]["LEGACY"]["protocol_profile"]["primary_field"] == "value_correct"
    assert "SYNTHETIC_F4/DATA" in spec["panels"]["CHARTQA"]["image_root_binding"]["path"]
    assert "FORMAL_SPEC_FROZEN.json" not in bundle["files"][plan.SPEC_FILE].decode()


def test_all_111_selectors_are_future_commits_or_original_parents(case):
    bundle = build(case)
    endpoints = [
        json.loads(line) for line in bundle["files"]["ENDPOINT_SELECTORS.jsonl"].splitlines()
    ]
    assert len(endpoints) == len({e["endpoint_id"] for e in endpoints}) == 111
    for stage, count, step in [("DEV36", 54, 36), ("FINAL", 54, 72)]:
        selected = [
            e
            for e in endpoints
            if e["stage"] == stage and e["kind"] == "FUTURE_FORMAL_COMMIT_SELECTOR"
        ]
        assert len(selected) == count
        for e in selected:
            assert e["checkpoint_step"] == step and e["lane"] == "PRIMARY"
            assert e["expected_commit_path"].endswith(f"/step-{step:03d}/COMMIT.json")
            assert not (case.root / e["expected_commit_path"]).exists()
            assert not any(k in e for k in ("commit_sha256", "policy_sha256", "receipt", "status"))
    assert len([e for e in endpoints if e["kind"] == "ORIGINAL_S0_PARENT"]) == 3
    assert all(
        e["access"] == "ALL_54_STEP72_PLUS_SEPARATE_TEST_GRANT"
        for e in endpoints
        if e["stage"] == "FINAL"
    )
    assert bundle["spec"]["access"]["training_grant_implies_test_access"] is False


def test_shared_streams_and_natural_exact_pairing(case):
    bundle = build(case)
    ids = {r["qid"]: r for r in rows(bundle, "ID")}
    for row in rows(bundle, "NATURAL_INTERFACE"):
        assert row["seeds"] == ids[row["paired_id_qid"]]["seeds"]
    for panel in plan.PANELS:
        if panel == "NATURAL_INTERFACE":
            continue
        domain = "formal_dev_eval" if panel == "DEV_EVAL" else "formal_final_eval"
        for row in rows(bundle, panel):
            assert row["seeds"] == [
                seed_for(2026102701, domain, panel, row["qid"], draw)
                for draw in range(row["draws"])
            ]
    assert bundle["spec"]["seed_recipe"]["excluded"] == [
        "parent",
        "repeat",
        "arm",
        "checkpoint",
        "lane",
        "task",
    ]


def test_idempotent_freeze_and_full_verification(case):
    result = plan.freeze_formal_evaluation_plan(case.root, **args(case))
    target = case.root / plan.DIRECTORY
    before = {p: (p.stat().st_ino, p.read_bytes()) for p in target.rglob("*") if p.is_file()}
    assert plan.freeze_formal_evaluation_plan(case.root, **args(case)) == result
    verified = plan.verify_formal_evaluation_plan(
        case.root, phase_context=case.context, expected_spec_sha256=result["spec_sha256"]
    )
    assert verified["status"] == "VERIFIED_PLAN_NOT_AUTHORIZED"
    assert before == {p: (p.stat().st_ino, p.read_bytes()) for p in before}
    assert all(verified[k] is False for k in plan.AUTHORIZATION)


@pytest.mark.parametrize(
    "kind",
    [
        "original_source",
        "formal_source",
        "receipt",
        "binding",
        "parent",
        "data_image",
        "retention_image",
        "private",
        "missing",
    ],
)
def test_changed_actual_source_or_input_cannot_be_rebound_as_a_plan(case, kind):
    if kind == "original_source":
        path = (
            Path(read(case.root, "SOURCE_BINDING.json")["previous_source"])
            / plan.INTERFACE_FILES[0]
        )
    elif kind == "formal_source":
        path = case.current / plan.INTERFACE_FILES[0]
    elif kind == "receipt":
        path = case.root.parent / plan.OPERATIONS_RECEIPT
    elif kind == "binding":
        path = case.root / plan.BINDING_PATH
    elif kind == "parent":
        path = case.root / case.refs["b0"]["path"]
    elif kind == "data_image":
        path = case.root / "DATA/images/synthetic.png"
    elif kind == "retention_image":
        path = Path(
            read(case.root, "SOURCE_BINDING.json")["retention_images"]["CHARTQA"][0]["path"]
        )
    else:
        path = case.root / "DATA/TEST_ID/TASKS_PRIVATE.jsonl"
    if kind == "missing":
        path.unlink()
    else:
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises((PermissionError, FileNotFoundError, ValueError)):
        build(case)
    assert not (case.root / plan.DIRECTORY).exists()


@pytest.mark.parametrize(
    "kind", ["duplicate", "missing", "wrong_natural_pair", "private_prompt", "wrong_image_hash"]
)
def test_self_consistent_internal_manifest_cannot_change_contract(factory, kind):
    def mutate(root):
        path = (
            root / "DATA/NATURAL_INTERFACE/TASKS_PRIVATE.jsonl"
            if kind == "wrong_natural_pair"
            else root / "DATA/TEST_ID/MODEL_INPUTS.jsonl"
        )
        data = [json.loads(line) for line in path.read_text().splitlines()]
        if kind == "duplicate":
            data[-1] = data[0]
        elif kind == "missing":
            data.pop()
        elif kind == "wrong_natural_pair":
            data[0]["paired_id_qid"] = data[1]["paired_id_qid"]
        elif kind == "private_prompt":
            data[0]["model_input"]["answer"] = "secret"
        else:
            data[0]["model_input"]["image_sha256"] = "f" * 64
        write(root, str(path.relative_to(root)), data, lines=True)

    case = factory(mutate_data=mutate)
    case.binding_ref = plan.bind_retention_protocol(
        case.root, phase_context=case.context, pilot_spec_ref=case.pilot_spec_ref
    )["binding_ref"]
    with pytest.raises(PermissionError):
        build(case)


@pytest.mark.parametrize(
    "kind", ["duplicate", "wrong_digest", "missing_field", "foreign_route", "wrong_prompt"]
)
def test_retention_bound_bytes_are_not_enough_without_original_schema(factory, kind):
    def mutate(old, images):
        name = (
            "CHARTQA_SOURCE_ROUTES.jsonl"
            if kind in ("foreign_route", "wrong_prompt")
            else "CHARTQA_TASKS.jsonl"
        )
        path = old / "RETENTION" / name
        data = [json.loads(line) for line in path.read_text().splitlines()]
        if kind == "duplicate":
            data[-1] = data[0]
        elif kind == "wrong_digest":
            data[0]["answer"] = "2"
        elif kind == "missing_field":
            del data[0]["question"]
        elif kind == "foreign_route":
            data[0]["image_file"] = "../outside.png"
        else:
            data[0]["prompt"] = "gold leaked"
        write(old, "RETENTION/" + name, data, lines=True)
        if kind in ("duplicate", "missing_field"):
            frozen = read(old, "RETENTION/FROZEN.json")
            frozen["chartqa_task_digest"] = old_digest(data)
            write(old, "RETENTION/FROZEN.json", frozen)

    case = factory(mutate_retention=mutate)
    case.binding_ref = plan.bind_retention_protocol(
        case.root, phase_context=case.context, pilot_spec_ref=case.pilot_spec_ref
    )["binding_ref"]
    with pytest.raises((PermissionError, KeyError)):
        build(case)


def test_no_actual_pilot_upstream_anchor_is_rejected(factory):
    case = factory(anchored=False)
    with pytest.raises(PermissionError, match="upstream producer evidence"):
        plan.bind_retention_protocol(
            case.root, phase_context=case.context, pilot_spec_ref=case.pilot_spec_ref
        )


def test_source_receipt_cannot_substitute_another_deployment_or_missing_module(factory):
    def mutate(project, current, old):
        receipt = read(project, plan.OPERATIONS_RECEIPT)
        receipt["source_deployment"] = str(current)
        write(project, plan.OPERATIONS_RECEIPT, receipt)

    case = factory(mutate_sources=mutate)
    with pytest.raises(PermissionError):
        plan.bind_retention_protocol(
            case.root, phase_context=case.context, pilot_spec_ref=case.pilot_spec_ref
        )


def test_phase_relabel_is_rejected(case):
    wrong = copy.deepcopy(case.context)
    wrong["phase"] = "PILOT_V1"
    with pytest.raises(PermissionError):
        plan.build_formal_evaluation_plan(case.root, **dict(args(case), phase_context=wrong))


def test_cached_retention_images_are_not_rehashed_when_unchanged(case, monkeypatch):
    def forbidden(_):
        raise AssertionError("No unchanged-image rehash")

    monkeypatch.setattr(source_binding, "file_sha256", forbidden)
    assert build(case)["spec"]["endpoint_count"] == 111


@pytest.mark.parametrize("kind", ["missing_commit", "corrupt_payload", "wrong_budget"])
def test_partial_or_modified_frozen_plan_is_preserved_and_rejected(case, monkeypatch, kind):
    result = plan.freeze_formal_evaluation_plan(case.root, **args(case))
    target = case.root / plan.DIRECTORY
    if kind == "missing_commit":
        (target / "BINDING.json").unlink()
        monkeypatch.setattr(
            plan,
            "build_formal_evaluation_plan",
            lambda *a, **k: pytest.fail("partial read must fail before build"),
        )
    elif kind == "corrupt_payload":
        (target / "panels/CHARTQA.jsonl").write_text("[]\n")
    else:
        spec = read(target, plan.SPEC_FILE)
        spec["budgets"]["final_responses"] = 1
        write(target, plan.SPEC_FILE, spec)
        result["spec_sha256"] = file_hash(target / plan.SPEC_FILE)
    before = {p: p.read_bytes() for p in target.rglob("*") if p.is_file()}
    with pytest.raises(PermissionError):
        plan.verify_formal_evaluation_plan(
            case.root, phase_context=case.context, expected_spec_sha256=result["spec_sha256"]
        )
    assert before == {p: p.read_bytes() for p in before}


def test_concurrent_publisher_lock_rejects_without_mutation(case):
    parent = case.root / Path(plan.DIRECTORY).parent
    parent.mkdir(parents=True, exist_ok=True)
    with (parent / ".formal-evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            plan.freeze_formal_evaluation_plan(case.root, **args(case))
    assert not (case.root / plan.DIRECTORY).exists()


@pytest.mark.parametrize("error", [errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP])
def test_ceph_unsupported_rename_exact_link_publication_and_retry(case, monkeypatch, error):
    def unsupported(*args):
        raise OSError(error, "SYNTHETIC unsupported FS")

    monkeypatch.setattr(review_package, "_rename_exclusive", unsupported)
    result = plan.freeze_formal_evaluation_plan(case.root, **args(case))
    assert plan.freeze_formal_evaluation_plan(case.root, **args(case)) == result
    stage = case.root / Path(plan.DIRECTORY).parent / ".FORMAL_EVAL_FROZEN.staging"
    assert stage.exists()
    assert os.path.samestat(
        (stage / "BINDING.json").stat(), (case.root / plan.DIRECTORY / "BINDING.json").stat()
    )


def test_collision_and_failed_stage_are_never_replaced(case, monkeypatch):
    original = review_package._publish

    def collide(stage, target):
        target.mkdir(mode=0o700)
        (target / "foreign").write_text("preserve")
        return original(stage, target)

    monkeypatch.setattr(review_package, "_publish", collide)
    with pytest.raises(FileExistsError):
        plan.freeze_formal_evaluation_plan(case.root, **args(case))
    target = case.root / plan.DIRECTORY
    assert (target / "foreign").read_text() == "preserve"
    stage = target.with_name(".FORMAL_EVAL_FROZEN.staging")
    assert (stage / "BINDING.json").exists()
    monkeypatch.setattr(review_package, "_publish", original)
    with pytest.raises(PermissionError):
        plan.freeze_formal_evaluation_plan(case.root, **args(case))


def test_interrupt_before_binding_preserves_partial_target_and_complete_stage(case, monkeypatch):
    def unsupported(*args):
        raise OSError(errno.EINVAL, "SYNTHETIC unsupported FS")

    original = os.link

    def interrupted(src, dst, *args, **kwargs):
        if Path(src).name == "BINDING.json":
            raise OSError("SYNTHETIC interruption before commit")
        return original(src, dst, *args, **kwargs)

    monkeypatch.setattr(review_package, "_rename_exclusive", unsupported)
    monkeypatch.setattr(os, "link", interrupted)
    with pytest.raises(OSError):
        plan.freeze_formal_evaluation_plan(case.root, **args(case))
    target = case.root / plan.DIRECTORY
    assert target.is_dir() and not (target / "BINDING.json").exists()
    assert (target.with_name(".FORMAL_EVAL_FROZEN.staging") / "BINDING.json").exists()
    before = {p: p.read_bytes() for p in target.rglob("*") if p.is_file()}
    monkeypatch.setattr(os, "link", original)
    with pytest.raises(PermissionError):
        plan.freeze_formal_evaluation_plan(case.root, **args(case))
    assert before == {p: p.read_bytes() for p in before}


def test_loaded_prompt_helper_must_come_from_actual_formal_deployment(case, monkeypatch):
    monkeypatch.setattr(
        plan,
        "_loaded_module_paths",
        lambda: {"sr_f4.evaluation": case.old / "src/sr_f4/evaluation.py"},
    )
    with pytest.raises(PermissionError, match="Loaded retention prompt helper"):
        build(case)


@pytest.mark.parametrize("relative", plan.INHERITED_FILES)
def test_valid_new_phase_manifest_cannot_change_original_native_scorer_or_stop(factory, relative):
    def mutate(project, current, old):
        path = current / relative
        path.write_bytes(path.read_bytes() + b"\n# synthetic semantic drift\n")

    case = factory(mutate_sources=mutate)
    with pytest.raises(PermissionError, match="native scorer/stop differs"):
        plan.bind_retention_protocol(
            case.root, phase_context=case.context, pilot_spec_ref=case.pilot_spec_ref
        )


def test_all_eleven_published_file_references_are_returned(case):
    result = plan.freeze_formal_evaluation_plan(case.root, **args(case))
    assert len(result["frozen_files"]) == 11
    assert result["binding_ref"] in result["frozen_files"]
    for ref in result["frozen_files"]:
        assert file_hash(case.root / ref["path"]) == ref["sha256"]
