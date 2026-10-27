"""CPU-only source/data fixtures; no model execution or SERVER_ACTUAL evidence."""

from __future__ import annotations

import copy
import fcntl
import json
import shutil
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from sr_f6 import formal_registration as formal
from sr_f6 import phases, registration
from sr_f6.common import digest, file_hash
from sr_f6.data import public_prompt
from sr_f6.planning import draw_seed, formal_paths, root_schedule
from sr_f6.rollouts import _bound, _registration_inputs
from sr_f6.transactions import RunTransactions
from srf6_ref.core import FAMILIES, seed_for
from srf6_ref.design import make_root
from srf6_ref.planning import ARMS


def write(root, relative, value, *, lines=False):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"".join(registration._payload(row) for row in value)
        if lines
        else registration._payload(value)
    )
    return dict(path=relative, sha256=file_hash(path))


def read(root, relative):
    return json.loads((root / relative).read_bytes())


@pytest.fixture(scope="module")
def prototype():
    return make_root("DEV_PILOT", "BINDING", 0, 2026102712)["tasks"]["P1"][0]


@pytest.fixture
def factory(tmp_path, monkeypatch, prototype):
    def create(*, mutate_data=None, mutate_pilot=None, anchored=True, register_formal=True):
        project = tmp_path.resolve() / "project"
        project.mkdir()
        monkeypatch.setattr(phases, "PROJECT", project)
        root = project / phases.ROOT.name
        root.mkdir()
        old = project / ("sr_f6_code_" + "a" * 7)
        pilot = project / ("sr_f6_code_" + "b" * 7)
        current = project / ("sr_f6_code_" + "c" * 7)
        (old / "src/sr_f6").mkdir(parents=True)
        (old / "src/sr_f6/runtime.py").write_text("# SYNTHETIC original source\n")
        shutil.copytree(old, pilot)
        (pilot / "src/sr_f6/registration.py").write_text("# SYNTHETIC pilot source\n")
        shutil.copytree(pilot, current)
        (current / "src/sr_f6/formal_registration.py").write_text("# SYNTHETIC formal source\n")
        monkeypatch.setattr(registration, "_actual_deployment", lambda: pilot)
        monkeypatch.setattr(formal, "_actual_deployment", lambda: current)
        original = Path(__file__).parents[2] / "configs/sr_f6.json"
        (root / "CONFIG.json").write_bytes(original.read_bytes())
        config = read(root, "CONFIG.json")
        write(root, "IMPLEMENTATION_REGISTRATION.json", phases.AMENDMENTS)
        image = "DATA/images/synthetic.png"
        (root / image).parent.mkdir(parents=True)
        (root / image).write_bytes(b"SYNTHETIC_NOT_PROCESSOR_EVIDENCE")
        members = [image]
        for split, count, factors in (
            ("TRAIN", 48, ("P0", "P1")),
            ("DEV_PILOT", 8, ("P0", "P1")),
            ("DEV_EVAL", 4, ("P0",)),
        ):
            private, public = [], []
            for family in FAMILIES:
                for index in range(count):
                    rid = f"{split}_{family}_{index:04d}"
                    for factor in factors:
                        for view in range(4):
                            task = {
                                **copy.deepcopy(prototype),
                                "qid": f"{rid}_{factor}_v{view}",
                                "root_id": rid,
                                "family": family,
                                "factor": factor,
                                "variant": view,
                                "split": split,
                            }
                            private.append(task)
                            public.append(
                                dict(
                                    **{
                                        k: task[k]
                                        for k in (
                                            "qid",
                                            "root_id",
                                            "family",
                                            "factor",
                                            "variant",
                                            "split",
                                        )
                                    },
                                    protocol="structured_v4",
                                    model_input=dict(
                                        prompt=public_prompt(task),
                                        image_file=image,
                                        image_sha256=file_hash(root / image),
                                    ),
                                )
                            )
            for name, rows in (("MODEL_INPUTS", public), ("TASKS_PRIVATE", private)):
                relative = f"DATA/{split}/{name}.jsonl"
                write(root, relative, rows, lines=True)
                members.append(relative)
        for parent in formal.PARENTS:
            for split, count, repeats in (("TRAIN", 48, range(3)), ("DEV_PILOT", 8, [0])):
                for repeat in repeats:
                    relative = f"DATA/SCHEDULES/{split}-{parent}-r{repeat}.json"
                    write(
                        root,
                        relative,
                        root_schedule(count, config["master_seed"], parent, repeat, split=split),
                    )
                    members.append(relative)
        write(root, "DATA/FORMAL_PATH_ORDER.json", formal_paths(config))
        members.append("DATA/FORMAL_PATH_ORDER.json")
        if mutate_data:
            mutate_data(root)
        data_ref = write(
            root,
            "DATA/MANIFEST.json",
            dict(
                config_sha256=digest(config),
                files=[
                    dict(path=p, sha256=file_hash(root / p), bytes=(root / p).stat().st_size)
                    for p in members
                ],
            ),
        )
        source = dict(
            files=[
                dict(
                    path="src/sr_f6/runtime.py",
                    sha256=file_hash(old / "src/sr_f6/runtime.py"),
                    bytes=(old / "src/sr_f6/runtime.py").stat().st_size,
                )
            ]
        )
        source_ref = write(root, "SOURCE_MANIFEST.json", source)
        origin = dict(
            config_sha256=file_hash(root / "CONFIG.json"),
            implementation_digest=digest(phases.AMENDMENTS),
            revision="a" * 40,
            source_manifest=source_ref,
            data_manifest=data_ref,
        )
        origin_ref = write(root, "IDENTITY.json", origin)
        binding = dict(
            config_file_sha256=origin["config_sha256"],
            parents={
                p: dict(path=config["model"]["parents"][p], content_sha256=str(i + 4) * 64)
                for i, p in enumerate(formal.PARENTS)
            },
        )
        write(root, "SOURCE_BINDING.json", binding)
        refs = {}
        for index, parent in enumerate(formal.PARENTS):
            run_id = f"SYNTHETIC-census-{parent}"
            refs[parent] = write(
                root,
                f"runtime_identity/{run_id}.json",
                dict(
                    run_id=run_id,
                    parent_id=parent,
                    reference_parent=parent,
                    step=0,
                    run_root=str(root),
                    source_sha256=source_ref["sha256"],
                    source_hash=source_ref["sha256"],
                    data_hash=data_ref["sha256"],
                    snapshot=config["model"]["snapshot"],
                    dtype="float32",
                    input_float_dtype="float32",
                    probability_dtype="float32",
                    reference_hash=str(index + 1) * 64,
                    policy_hash=str(index + 1) * 64,
                    parent_content_hash=binding["parents"][parent]["content_sha256"],
                    trainable_parameters=[
                        dict(name="synthetic.adapter", shape=[2, 1], dtype="torch.float32")
                    ],
                    fresh_optimizer=False,
                    bound_files_before_load={"SYNTHETIC": True},
                    bound_files_after_load={"SYNTHETIC": True},
                ),
            )
        original_evidence = write(root, "EVIDENCE/S0.json", dict(status="NOT_RUN", context=origin))
        origin_upstream = dict(
            producer_phase=phases.ORIGIN,
            identity_sha256=origin_ref["sha256"],
            evidence=[original_evidence],
        )

        def bind(phase, deployment, revision, upstream):
            manifest = phases.build_manifest(deployment, revision=revision)
            added = set(p["path"] for p in manifest["files"]) - {"src/sr_f6/runtime.py"}
            delta = phases.source_delta(
                source, manifest, reasons={p: "Synthetic prospective source" for p in added}
            )
            return phases.register_phase(
                root,
                phase=phase,
                deployment=deployment,
                revision=revision,
                origin_deployment=old,
                upstream=upstream,
                expected_delta=delta,
            )

        pilot_context = bind("PILOT_V1", pilot, "b" * 40, [origin_upstream])
        pilot_result = registration.freeze_pilot_registration(
            root, phase_context=pilot_context, parent_references=refs
        )
        pilot_spec_path = f"{registration.DIRECTORY}/{registration.SPEC_FILE}"
        if mutate_pilot:
            mutate_pilot(root, pilot_spec_path)
        pilot_spec_ref = dict(path=pilot_spec_path, sha256=file_hash(root / pilot_spec_path))
        pilot_evidence = write(
            root,
            "EVIDENCE/PILOT.json",
            dict(
                status="NOT_RUN",
                context=pilot_context["identity"],
                inputs=[pilot_spec_ref] if anchored else [],
            ),
        )
        upstream = [
            origin_upstream,
            dict(
                producer_phase="PILOT_V1",
                identity_sha256=pilot_context["identity_sha256"],
                evidence=[pilot_evidence],
            ),
        ]
        context = bind(formal.PHASE, current, "c" * 40, upstream) if register_formal else None
        return SimpleNamespace(
            root=root,
            current=current,
            old=old,
            pilot=pilot,
            config=config,
            context=context,
            pilot_context=pilot_context,
            pilot_spec_ref=pilot_spec_ref,
            refs=refs,
            origin_ref=origin_ref,
            data_ref=data_ref,
            pilot_result=pilot_result,
        )

    return create


@pytest.fixture
def case(factory):
    return factory()


def build(case):
    return formal.build_formal_registration(
        case.root, phase_context=case.context, pilot_spec_ref=case.pilot_spec_ref
    )


def freeze(case):
    return formal.freeze_formal_registration(
        case.root, phase_context=case.context, pilot_spec_ref=case.pilot_spec_ref
    )


def verify(case, result):
    return formal.verify_formal_registration(
        case.root, phase_context=case.context, expected_spec_sha256=result["spec_sha256"]
    )


def test_all_54_exact_order_original_sources_and_no_authorization(case):
    before = copy.deepcopy((case.context, case.pilot_spec_ref))
    bundle = build(case)
    assert (case.context, case.pilot_spec_ref) == before
    assert not bundle["published"] and not Path(bundle["directory"]).exists()
    assert len(bundle["files"]) == 109
    spec = bundle["spec"]
    assert [
        (p["parent"], p["repeat"], p["arm"], p["block_order"], p["run_id"]) for p in spec["paths"]
    ] == [
        (p["parent"], p["repeat"], p["arm"], p["block_order"], p["run_id"])
        for p in formal_paths(case.config)
    ]
    assert spec["budgets"]["primary_responses"] == 54 * 72 * 128 == 497664
    assert spec["budgets"]["primary_logical_updates"] == 3888
    assert spec["actual_source_manifest"] == case.context["identity"]["source_manifest"]
    assert (
        spec["pilot_producer"]["source_manifest"]
        == case.pilot_context["identity"]["source_manifest"]
    )
    assert spec["origin_identity"] == case.origin_ref
    assert (
        len(
            {
                spec["actual_source_manifest"]["sha256"],
                spec["pilot_producer"]["source_manifest"]["sha256"],
                case.context["identity"]["origin"]["source_manifest"]["sha256"],
            }
        )
        == 3
    )
    assert spec["status"] == "FROZEN_NOT_AUTHORIZED" and spec["gate_status"] == "NOT_SIGNED"
    assert "SERVER_ACTUAL" not in spec["scope"]
    assert not any(
        spec[k]
        for k in (
            "training_authorized",
            "generation_authorized",
            "evaluation_authorized",
            "formal_test_access",
        )
    )
    assert spec["outstanding_requirements"]["evaluation_plan_included"] is False
    assert "evaluation" not in spec and not list(case.root.rglob("COMMIT.json"))


def test_transactions_and_formal_rollout_consume_all_54_paired_exposure_and_streams(case):
    result = freeze(case)
    data = read(case.root, "DATA/MANIFEST.json")
    shared = {}
    for path in result["spec"]["paths"]:
        directory = Path(result["directory"]) / path["run_directory"]
        identity = read(directory, "RUN_IDENTITY.json")
        with RunTransactions(directory, identity) as tx:
            assert tx.published() == []
            value = _bound(tx, directory, identity["rollout_registration"])
            assert value["identity"] == {
                k: v for k, v in identity.items() if k != "rollout_registration"
            }
            public, private = _registration_inputs(tx, case.root, value, case.config, data)
            assert len(public) == len(private) == 2304
        parent, repeat, arm = path["parent"], path["repeat"], path["arm"]
        assert identity["reference_sha256"] == str(int(parent[-1]) + 1) * 64
        assert identity["reference_sha256"] != identity["parent_content_sha256"]
        assert identity["parent_reference_evidence"] == case.refs[parent]
        assert value["factor"] == ARMS[arm][0]
        schedule = value["schedule"]
        assert len(schedule) == 72 and all(len(step) == 4 for step in schedule)
        assert Counter(root for step in schedule for root in step) == Counter(
            {f"TRAIN_{f}_{i:04d}": 1 for f in FAMILIES for i in range(48)}
        )
        for offset in range(0, 72, 12):
            roots = [r for step in schedule[offset : offset + 12] for r in step]
            assert all(sum(r.startswith(f"TRAIN_{f}_") for r in roots) == 8 for f in FAMILIES)
        key = parent, repeat
        paired = (schedule, value["draw_seeds"])
        assert shared.setdefault(key, paired) == paired
        for step, group, draw in ((1, 0, 0), (36, 9, 7), (72, 15, 3)):
            assert value["draw_seeds"][step - 1][group][draw] == draw_seed(
                case.config["master_seed"], parent, repeat, step, group // 4, group % 4, draw
            )
            assert value["shuffle_seeds"][step - 1][group] == seed_for(
                case.config["master_seed"], "formal_shuffle", parent, repeat, arm, step, group
            )
        assert identity["diagnostic_row_indices"] == list(formal.DIAGNOSTIC_ROWS)
    assert len(shared) == 9
    assert len({digest(v[0]) for v in shared.values()}) == 9
    assert len({digest(v[1]) for v in shared.values()}) == 9
    diagnostics = result["spec"]["diagnostics"]
    assert diagnostics["steps"] == [1, 4, 8, 12, 36, 72]
    assert Counter(i // 8 for i in diagnostics["row_indices"]) == Counter(
        {g: 8 for g in (0, 1, 4, 5, 8, 9, 12, 13)}
    )
    assert result["spec"]["checkpoints"]["permanent_steps"] == [0, 12, 36, 72]
    assert result["spec"]["initialization"]["fresh_optimizer"] is True
    assert verify(case, result)["training_authorized"] is False


@pytest.mark.parametrize(
    "kind", ["caller_context", "wrong_deployment", "current_source", "pilot_source"]
)
def test_real_phase_reverification_rejects_source_relabel_or_mutation(case, monkeypatch, kind):
    if kind == "caller_context":
        case.context["training_allowed"] = True
    elif kind == "wrong_deployment":
        monkeypatch.setattr(formal, "_actual_deployment", lambda: case.pilot)
    else:
        target = case.current if kind == "current_source" else case.pilot
        (target / "src/sr_f6/runtime.py").write_text("changed source\n")
    with pytest.raises(PermissionError):
        build(case)


def test_missing_formal_phase_is_rejected(factory):
    case = factory(register_formal=False)
    with pytest.raises(PermissionError):
        build(case)


def test_pilot_plan_requires_actual_upstream_receipt_reference(factory):
    case = factory(anchored=False)
    with pytest.raises(PermissionError, match="upstream producer evidence"):
        build(case)


def test_unchanged_image_uses_existing_phase_cache_but_changed_content_blocks(case, monkeypatch):
    original = phases.file_hash
    image = case.root / "DATA/images/synthetic.png"
    reads = []

    def observed(path):
        if Path(path) == image:
            reads.append(str(path))
        return original(path)

    monkeypatch.setattr(phases, "file_hash", observed)
    build(case)
    assert reads == []  # Registration reused the phase binding's content/stat cache.
    content = image.read_bytes()
    image.write_bytes(bytes([content[0] ^ 1]) + content[1:])
    with pytest.raises(PermissionError):
        build(case)
    assert reads == [str(image)]


@pytest.mark.parametrize("kind", ["schedule", "prompt", "missing_qid"])
def test_self_consistent_data_identity_does_not_override_training_contract(factory, kind):
    def mutate(root):
        if kind == "schedule":
            path = "DATA/SCHEDULES/TRAIN-b0-r0.json"
            values = read(root, path)
            values[0][0] = values[1][0]
            write(root, path, values)
        else:
            path = "DATA/TRAIN/MODEL_INPUTS.jsonl"
            values = [json.loads(line) for line in (root / path).read_text().splitlines()]
            if kind == "prompt":
                values[0]["model_input"]["prompt"] += " hidden gold"
            else:
                values.pop()
            write(root, path, values, lines=True)

    case = factory(mutate_data=mutate)
    with pytest.raises(PermissionError):
        build(case)


@pytest.mark.parametrize("kind", ["endpoint", "file_hash", "source"])
def test_original_tensor_evidence_not_endpoint_inventory_or_relabel(factory, kind):
    def mutate(root, spec_path):
        spec = read(root, spec_path)
        ref = spec["parent_references"]["b0"]
        value = read(root, ref["path"])
        if kind == "endpoint":
            value["step"] = 12
        elif kind == "file_hash":
            value["policy_hash"] = value["reference_hash"] = value["parent_content_hash"]
        else:
            value["source_sha256"] = spec["actual_source_manifest"]["sha256"]
        updated = write(root, ref["path"], value)
        spec["parent_references"]["b0"] = updated
        spec["parent_tensors"]["b0"]["runtime_identity"] = updated
        write(root, spec_path, spec)

    case = factory(mutate_pilot=mutate)
    with pytest.raises(PermissionError):
        build(case)


def test_idempotence_preserves_root_identity_and_allows_actual_run_subdirectories(case):
    original = (case.root / "IDENTITY.json").read_bytes()
    result = freeze(case)
    paths = [p for p in Path(result["directory"]).rglob("*.json")]
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
    assert len(before) == 109 and freeze(case) == result
    run = Path(result["directory"]) / result["spec"]["paths"][0]["run_directory"]
    (run / "checkpoints").mkdir()
    (run / "rollouts").mkdir()
    assert verify(case, result)["status"] == "VERIFIED_REGISTRATION_NOT_AUTHORIZED"
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
    assert (case.root / "IDENTITY.json").read_bytes() == original


@pytest.mark.parametrize("kind", ["registration", "identity", "missing", "extra", "symlink"])
def test_corrupt_or_partial_bundle_cannot_be_overwritten(case, kind):
    result = freeze(case)
    run = Path(result["directory"]) / result["spec"]["paths"][0]["run_directory"]
    path = run / ("RUN_IDENTITY.json" if kind == "identity" else "ROLLOUT_REGISTRATION.json")
    if kind in {"registration", "identity"}:
        value = json.loads(path.read_bytes())
        value["forged"] = True
        write(run, path.name, value)
    elif kind == "missing":
        path.unlink()
    elif kind == "extra":
        (run.parent / "UNREGISTERED").mkdir()
    else:
        alias = case.root / "preserved-original.json"
        alias.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(alias)
    snapshot = {p: p.read_bytes() for p in Path(result["directory"]).rglob("*.json")}
    for action in (lambda: verify(case, result), lambda: freeze(case)):
        with pytest.raises((PermissionError, FileNotFoundError)):
            action()
    assert snapshot == {p: p.read_bytes() for p in snapshot}


def test_wrong_pin_and_exclusive_publisher_lock(case):
    result = freeze(case)
    with pytest.raises(PermissionError, match="pin differs"):
        verify(case, {**result, "spec_sha256": "9" * 64})
    with (Path(result["directory"]).parent / ".formal-registration.lock").open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            freeze(case)


@pytest.mark.parametrize("after_rename", [False, True])
def test_atomic_rename_failure_preserved_and_exact_retry(case, monkeypatch, after_rename):
    original = formal.os.rename

    def interrupted(source, destination):
        if after_rename:
            original(source, destination)
        raise RuntimeError("synthetic publication interruption")

    monkeypatch.setattr(formal.os, "rename", interrupted)
    with pytest.raises(RuntimeError, match="publication interruption"):
        freeze(case)
    destination = case.root / formal.DIRECTORY
    assert destination.exists() is after_rename
    orphans = list(destination.parent.glob(".formal-registration-*/FORMAL_SPEC_FROZEN.json"))
    assert len(orphans) == (0 if after_rename else 1)
    monkeypatch.setattr(formal.os, "rename", original)
    assert verify(case, freeze(case))["generation_authorized"] is False
    assert all(path.exists() for path in orphans)
