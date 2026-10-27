"""CPU registration fixtures; no actual GPU, model, measurement or gate PASS."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from sr_f6 import phases, registration
from sr_f6.common import digest, file_hash
from sr_f6.data import public_prompt
from sr_f6.planning import draw_seed, formal_paths, root_schedule
from sr_f6.rollouts import _bound, _registration_inputs
from sr_f6.transactions import RunTransactions
from srf6_ref.core import FAMILIES, seed_for
from srf6_ref.design import make_root


def write(path, value, *, lines=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        b"".join(registration._payload(row) for row in value)
        if lines
        else registration._payload(value)
    )
    path.write_bytes(payload)
    return {"path": path.name, "sha256": hashlib.sha256(payload).hexdigest()}


def entry(base, relative):
    path = base / relative
    return dict(path=relative, sha256=file_hash(path), bytes=path.stat().st_size)


@pytest.fixture(scope="module")
def prototype():
    return make_root("DEV_PILOT", "BINDING", 0, 2026102712)["tasks"]["P1"][0]


@pytest.fixture
def factory(tmp_path, monkeypatch, prototype):
    """Full phase provenance + tiny synthetic imagery, without source-check mocks."""

    def create(*, mutate_data=None):
        project = tmp_path / "project"
        project.mkdir()
        monkeypatch.setattr(phases, "PROJECT", project)
        root = project / phases.ROOT.name
        root.mkdir()
        old_revision, new_revision = "a" * 40, "b" * 40
        old = project / f"sr_f6_code_{old_revision[:7]}"
        new = project / f"sr_f6_code_{new_revision[:7]}"
        (old / "src/sr_f6").mkdir(parents=True)
        (old / "src/sr_f6/runtime.py").write_text("# CPU fixture historical source\n")
        shutil.copytree(old, new)
        (new / "src/sr_f6/registration.py").write_text("# CPU fixture new phase source\n")
        monkeypatch.setattr(registration, "_actual_deployment", lambda: new)
        original = Path(__file__).parents[2] / "configs/sr_f6.json"
        (root / "CONFIG.json").write_bytes(original.read_bytes())
        config = json.loads(original.read_bytes())
        write(root / "IMPLEMENTATION_REGISTRATION.json", phases.AMENDMENTS)
        image_relative = "DATA/images/cpu-fixture.png"
        image = root / image_relative
        image.parent.mkdir(parents=True)
        image.write_bytes(b"CPU_SYNTHETIC_FIXTURE_NOT_PROCESSOR_EVIDENCE")
        image_hash = file_hash(image)
        members = [image_relative]
        for split, count, factors in (
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
                            safe = {
                                key: task[key]
                                for key in (
                                    "qid",
                                    "root_id",
                                    "family",
                                    "factor",
                                    "variant",
                                    "split",
                                )
                            }
                            public.append(
                                dict(
                                    **safe,
                                    protocol="structured_v4",
                                    model_input=dict(
                                        prompt=public_prompt(task),
                                        image_file=image_relative,
                                        image_sha256=image_hash,
                                    ),
                                )
                            )
            for name, value in (("MODEL_INPUTS", public), ("TASKS_PRIVATE", private)):
                relative = f"DATA/{split}/{name}.jsonl"
                write(root / relative, value, lines=True)
                members.append(relative)
        for parent in registration.PARENTS:
            relative = f"DATA/SCHEDULES/DEV_PILOT-{parent}-r0.json"
            write(
                root / relative,
                root_schedule(8, config["master_seed"], parent, 0, split="DEV_PILOT"),
            )
            members.append(relative)
        order_path = "DATA/FORMAL_PATH_ORDER.json"
        write(root / order_path, formal_paths(config))
        members.append(order_path)
        if mutate_data:
            mutate_data(root)
        data = dict(config_sha256=digest(config), files=[entry(root, p) for p in members])
        write(root / "DATA/MANIFEST.json", data)
        source = dict(files=[entry(old, "src/sr_f6/runtime.py")])
        write(root / "SOURCE_MANIFEST.json", source)
        identity = dict(
            config_sha256=file_hash(root / "CONFIG.json"),
            implementation_digest=digest(phases.AMENDMENTS),
            revision=old_revision,
            source_manifest=dict(
                path="SOURCE_MANIFEST.json", sha256=file_hash(root / "SOURCE_MANIFEST.json")
            ),
            data_manifest=dict(
                path="DATA/MANIFEST.json", sha256=file_hash(root / "DATA/MANIFEST.json")
            ),
        )
        write(root / "IDENTITY.json", identity)
        binding = dict(
            config_file_sha256=identity["config_sha256"],
            parents={
                parent: dict(
                    path=config["model"]["parents"][parent], content_sha256=str(i + 4) * 64
                )
                for i, parent in enumerate(registration.PARENTS)
            },
        )
        write(root / "SOURCE_BINDING.json", binding)
        refs = {}
        for i, parent in enumerate(registration.PARENTS):
            run_id = f"cpu-fixture-census-{parent}"
            value = dict(
                run_id=run_id,
                parent_id=parent,
                reference_parent=parent,
                step=0,
                run_root=str(root),
                source_sha256=identity["source_manifest"]["sha256"],
                source_hash=identity["source_manifest"]["sha256"],
                data_hash=identity["data_manifest"]["sha256"],
                snapshot=config["model"]["snapshot"],
                dtype="float32",
                input_float_dtype="float32",
                probability_dtype="float32",
                reference_hash=str(i + 1) * 64,
                policy_hash=str(i + 1) * 64,
                parent_content_hash=binding["parents"][parent]["content_sha256"],
                trainable_parameters=[
                    dict(name="fixture_adapter.weight", shape=[1, 2], dtype="torch.float32")
                ],
                fresh_optimizer=False,
                bound_files_before_load={"fixture": True},
                bound_files_after_load={"fixture": True},
            )
            relative = f"runtime_identity/{run_id}.json"
            write(root / relative, value)
            refs[parent] = dict(path=relative, sha256=file_hash(root / relative))
        write(root / "EVIDENCE/CENSUS.json", dict(status="NOT_RUN", context=identity))
        upstream = [
            dict(
                producer_phase=phases.ORIGIN,
                identity_sha256=file_hash(root / "IDENTITY.json"),
                evidence=[
                    dict(
                        path="EVIDENCE/CENSUS.json", sha256=file_hash(root / "EVIDENCE/CENSUS.json")
                    )
                ],
            )
        ]
        current = phases.build_manifest(new, revision=new_revision)
        delta = phases.source_delta(
            source,
            current,
            reasons={"src/sr_f6/registration.py": "Prospective CPU fixture registration source"},
        )
        context = phases.register_phase(
            root,
            phase="PILOT_V1",
            deployment=new,
            revision=new_revision,
            origin_deployment=old,
            upstream=upstream,
            expected_delta=delta,
        )
        return SimpleNamespace(
            root=root,
            new=new,
            old=old,
            config=config,
            refs=refs,
            context=context,
            identity=identity,
            binding=binding,
        )

    return create


@pytest.fixture
def project(factory):
    return factory()


def build(project):
    return registration.build_pilot_registration(
        project.root, phase_context=project.context, parent_references=project.refs
    )


def freeze(project):
    return registration.freeze_pilot_registration(
        project.root, phase_context=project.context, parent_references=project.refs
    )


def verify(project, result):
    return registration.verify_pilot_registration(
        project.root, phase_context=project.context, expected_spec_sha256=result["spec_sha256"]
    )


def test_exact_nine_order_and_non_authorizing_boundaries(project):
    before = copy.deepcopy((project.context, project.refs))
    bundle = build(project)
    assert (project.context, project.refs) == before
    assert build(project) == bundle
    assert not (project.root / registration.DIRECTORY).exists()
    spec = bundle["spec"]
    assert [(p["parent"], p["arm"]) for p in spec["paths"]] == [
        ("b0", "U11"),
        ("b0", "C_SHUF"),
        ("b0", "U10"),
        ("b1", "U10"),
        ("b1", "C_SHUF"),
        ("b1", "U11"),
        ("b2", "U10"),
        ("b2", "U11"),
        ("b2", "C_SHUF"),
    ]
    assert len(bundle["files"]) == 19
    assert spec["budgets"] == dict(
        primary_paths=9,
        updates_per_path=12,
        primary_logical_updates=108,
        responses_per_update=128,
        primary_responses=13824,
        recovery_new_responses=1024,
        dev_eval_responses=2304,
    )
    assert spec["actual_source_manifest"] == project.context["identity"]["source_manifest"]
    assert spec["actual_source_manifest"]["sha256"] != project.identity["source_manifest"]["sha256"]
    assert spec["gate_status"] == "NOT_SIGNED"
    assert spec["training_authorized"] is spec["generation_authorized"] is False
    assert spec["execution_status"] == "NOT_RUN"
    assert not any(project.root.rglob("COMMIT.json"))


def test_real_rollout_parser_and_transaction_identity_no_reference_cycle(project):
    result = freeze(project)
    destination = Path(result["directory"])
    registrations = {}
    for path in result["spec"]["paths"]:
        run = destination / path["run_directory"]
        identity = json.loads((run / "RUN_IDENTITY.json").read_bytes())
        with RunTransactions(run, identity) as tx:
            value = _bound(tx, run, identity["rollout_registration"])
            assert value["identity"] == {
                k: v for k, v in identity.items() if k != "rollout_registration"
            }
            assert "rollout_registration" not in value["identity"]
            assert "spec_sha256" not in identity
            assert tx.published() == []
            data = json.loads((project.root / "DATA/MANIFEST.json").read_bytes())
            publics, tasks = _registration_inputs(tx, project.root, value, project.config, data)
            assert len(publics) == len(tasks) == 384
            assert all(v["split"] == "DEV_PILOT" for v in tasks.values())
            registrations[(path["parent"], path["arm"])] = value
    for parent in registration.PARENTS:
        values = [registrations[(parent, arm)] for arm in registration.ARMS]
        assert values[0]["schedule"] == values[1]["schedule"] == values[2]["schedule"]
        assert values[0]["draw_seeds"] == values[1]["draw_seeds"] == values[2]["draw_seeds"]
        schedule = values[0]["schedule"]
        assert len(schedule) == 12 and len({r for step in schedule for r in step}) == 48
        for value in values:
            assert value["factor"] == "P1"
            assert value["draw_seeds"][0][0][0] == draw_seed(
                project.config["master_seed"], parent, 0, 1, 0, 0, 0
            )
            assert value["shuffle_seeds"][0][0] == seed_for(
                project.config["master_seed"],
                "pilot_shuffle",
                parent,
                0,
                value["identity"]["arm"],
                1,
                0,
            )
    assert registrations[("b0", "U11")]["schedule"] != registrations[("b1", "U11")]["schedule"]


def test_diagnostics_unique_recovery_and_evaluation_full_budget(project):
    spec = build(project)["spec"]
    diagnostics = spec["diagnostics"]
    assert diagnostics["group_indices"] == [0, 1, 4, 5, 8, 9, 12, 13]
    assert len(diagnostics["row_indices"]) == len(set(diagnostics["row_indices"])) == 64
    assert diagnostics["pilot_steps"] == [1, 4, 8, 12]
    assert diagnostics["registered_all_steps"] == [1, 4, 8, 12, 36, 72]
    recovery = spec["recovery"]
    assert (recovery["parent"], recovery["arm"], recovery["repeat"]) == ("b0", "U11", 0)
    assert recovery["update_steps"] == list(range(5, 13))
    assert recovery["new_generated_responses"] == 1024
    assert recovery["reuse_primary_steps5_to12_raw"] is recovery["adds_scientific_repeat"] is False
    assert recovery["implementation_status"] == "NOT_IMPLEMENTED"
    assert recovery["comparison"]["rule"] == "EXACT_TREE_NO_POSTHOC_TOLERANCE"
    assert recovery["comparison"]["continuity_required"] is True
    assert recovery["actual_source_commit_binding"] == "REQUIRED_SEPARATELY_AFTER_STEP4_PUBLICATION"
    evaluation = spec["evaluation"]
    assert evaluation["checkpoint_count"] == len(evaluation["endpoints"]) == 12
    assert sum(p["responses"] for p in evaluation["endpoints"]) == 2304
    assert sum(p["kind"] == "FROZEN_PARENT" for p in evaluation["endpoints"]) == 3
    assert evaluation["recovery_endpoint_is_additional_checkpoint"] is False
    assert evaluation["formal_test_access"] is False
    for parent, rows in evaluation["seeds_by_parent"].items():
        assert len(rows) == len({r["qid"] for r in rows}) == 96
        for row in rows:
            assert row["seeds"] == [
                seed_for(project.config["master_seed"], "pilot_dev_eval", parent, row["qid"], draw)
                for draw in range(2)
            ]


def test_original_s0_schema_has_no_new_phase_fields_and_tensor_not_file_identity(project):
    result = build(project)
    for parent, reference in project.refs.items():
        evidence = json.loads((project.root / reference["path"]).read_bytes())
        assert not {"phase", "execution_mode", "phase_identity_sha256"} & evidence.keys()
        for path in result["spec"]["paths"]:
            if path["parent"] == parent:
                identity = result["files"][f"{path['run_directory']}/RUN_IDENTITY.json"]
                assert (
                    identity["reference_sha256"]
                    == evidence["reference_hash"]
                    == evidence["policy_hash"]
                )
                assert identity["reference_sha256"] != identity["parent_content_sha256"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("step", 1),
        ("step", False),
        ("fresh_optimizer", True),
        ("policy_hash", "9" * 64),
        ("reference_hash", "4" * 64),
        ("source_hash", "9" * 64),
        ("source_sha256", "9" * 64),
        ("parent_id", "b1"),
        ("reference_parent", "b1"),
        ("data_hash", "9" * 64),
        ("snapshot", "wrong"),
        ("run_root", "/wrong"),
        ("dtype", "bfloat16"),
        ("input_float_dtype", "float16"),
        ("probability_dtype", "bfloat16"),
        ("parent_content_hash", "1" * 64),
        ("trainable_parameters", []),
        ("bound_files_after_load", None),
        ("run_id", "wrong"),
    ],
)
def test_wrong_parent_evidence_rejected_even_with_recomputed_reference(project, field, value):
    reference = project.refs["b0"]
    path = project.root / reference["path"]
    evidence = json.loads(path.read_bytes())
    evidence[field] = value
    write(path, evidence)
    reference["sha256"] = file_hash(path)
    with pytest.raises(PermissionError):
        build(project)


def test_file_inventory_cannot_be_used_as_policy_tensor_even_if_both_fields_match(project):
    reference = project.refs["b0"]
    path = project.root / reference["path"]
    evidence = json.loads(path.read_bytes())
    evidence["reference_hash"] = evidence["policy_hash"] = evidence["parent_content_hash"]
    write(path, evidence)
    reference["sha256"] = file_hash(path)
    with pytest.raises(PermissionError, match="cannot substitute"):
        build(project)


@pytest.mark.parametrize("kind", ["context", "phase", "deployment", "source"])
def test_phase_context_reverified_not_trusted(project, monkeypatch, kind):
    if kind == "context":
        project.context["training_allowed"] = True
    elif kind == "phase":
        project.context["phase"] = "FORMAL_V1"
    elif kind == "deployment":
        monkeypatch.setattr(registration, "_actual_deployment", lambda: project.old)
    else:
        (project.new / "src/sr_f6/registration.py").write_text("# changed after freeze\n")
    with pytest.raises(PermissionError):
        build(project)


def change_data(root, kind):
    if kind in {"schedule", "order"}:
        relative = (
            "DATA/SCHEDULES/DEV_PILOT-b0-r0.json"
            if kind == "schedule"
            else "DATA/FORMAL_PATH_ORDER.json"
        )
        value = json.loads((root / relative).read_bytes())
        value.reverse()
        write(root / relative, value)
        return
    split = "DEV_PILOT" if kind == "pilot_prompt" else "DEV_EVAL"
    path = root / f"DATA/{split}/MODEL_INPUTS.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if kind in {"pilot_prompt", "eval_prompt"}:
        rows[0]["model_input"]["prompt"] += " private answer"
    elif kind == "eval_image":
        rows[0]["model_input"]["image_sha256"] = "9" * 64
    elif kind == "eval_missing":
        rows.pop()
    elif kind == "eval_duplicate":
        rows[0] = rows[1]
    elif kind == "eval_test_split":
        rows[0]["split"] = "TEST_IN"
    write(path, rows, lines=True)


@pytest.mark.parametrize(
    "kind",
    [
        "schedule",
        "order",
        "pilot_prompt",
        "eval_prompt",
        "eval_image",
        "eval_missing",
        "eval_duplicate",
        "eval_test_split",
    ],
)
def test_self_consistent_provenance_does_not_override_registered_content(factory, kind):
    project = factory(mutate_data=lambda root: change_data(root, kind))
    with pytest.raises(PermissionError):
        build(project)


def test_freeze_verify_idempotent_and_future_run_outputs_allowed(project):
    result = freeze(project)
    destination = Path(result["directory"])
    snapshots = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in destination.rglob("*.json")}
    assert len(snapshots) == 19
    assert freeze(project) == result
    assert verify(project, result)["status"] == "VERIFIED_REGISTRATION_NOT_AUTHORIZED"
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in snapshots} == snapshots
    # Actual execution remains unauthorized; this only proves directory compatibility.
    first = destination / result["spec"]["paths"][0]["run_directory"]
    (first / "checkpoints").mkdir()
    (first / "rollouts").mkdir()
    assert verify(project, result)["training_authorized"] is False


@pytest.mark.parametrize(
    "kind", ["spec", "identity", "registration", "missing", "extra_arm", "extra_repeat", "symlink"]
)
def test_published_tamper_or_partial_bundle_never_overwritten(project, kind):
    result = freeze(project)
    destination = Path(result["directory"])
    run = destination / result["spec"]["paths"][0]["run_directory"]
    if kind in {"spec", "identity", "registration"}:
        target = (
            destination / registration.SPEC_FILE
            if kind == "spec"
            else run / ("RUN_IDENTITY.json" if kind == "identity" else "ROLLOUT_REGISTRATION.json")
        )
        value = json.loads(target.read_bytes())
        value["forged"] = True
        write(target, value)
    elif kind == "missing":
        (run / "ROLLOUT_REGISTRATION.json").unlink()
    elif kind == "extra_arm":
        (run.parent / "U00").mkdir()
    elif kind == "extra_repeat":
        (run.parent.parent / "r1").mkdir()
    else:
        target = run / "ROLLOUT_REGISTRATION.json"
        other = project.root / "copied-registration.json"
        other.write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(other)
    with pytest.raises((PermissionError, FileNotFoundError)):
        verify(project, result)
    with pytest.raises((PermissionError, FileNotFoundError)):
        freeze(project)


def test_wrong_pin_and_second_publication_lock_rejected(project):
    result = freeze(project)
    with pytest.raises(PermissionError, match="pin differs"):
        verify(project, {**result, "spec_sha256": "9" * 64})
    lock = Path(result["directory"]).parent / ".pilot-registration.lock"
    with lock.open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            freeze(project)


def test_unpublished_staging_is_not_registration(project):
    staging = project.root / registration.DIRECTORY
    staging = staging.with_name(".pilot-registration-orphan")
    staging.mkdir(parents=True)
    write(staging / registration.SPEC_FILE, dict(status="forged"))
    result = freeze(project)
    assert (staging / registration.SPEC_FILE).exists()
    assert verify(project, result)["training_authorized"] is False


@pytest.mark.parametrize("after_rename", [False, True])
def test_atomic_publication_failure_and_exact_retry(project, monkeypatch, after_rename):
    original = registration.os.rename

    def interrupted(source, destination):
        if after_rename:
            original(source, destination)
        raise RuntimeError("CPU fixture interruption at publication boundary")

    monkeypatch.setattr(registration.os, "rename", interrupted)
    with pytest.raises(RuntimeError, match="fixture interruption"):
        freeze(project)
    destination = project.root / registration.DIRECTORY
    assert destination.exists() is after_rename
    if after_rename:
        published = {path: path.read_bytes() for path in destination.rglob("*.json")}
        assert len(published) == 19
    else:
        orphans = list(destination.parent.glob(".pilot-registration-*/PILOT_SPEC_FROZEN.json"))
        assert len(orphans) == 1
    monkeypatch.setattr(registration.os, "rename", original)
    result = freeze(project)
    assert verify(project, result)["generation_authorized"] is False
    if after_rename:
        assert published == {path: path.read_bytes() for path in published}
    else:
        assert orphans[0].exists()


@pytest.mark.parametrize("kind", ["missing", "duplicate", "stale_sha", "symlink"])
def test_parent_reference_boundaries(project, kind):
    if kind == "missing":
        del project.refs["b2"]
    elif kind == "duplicate":
        project.refs["b2"] = project.refs["b0"]
    elif kind == "stale_sha":
        project.refs["b0"]["sha256"] = "9" * 64
    else:
        path = project.root / project.refs["b0"]["path"]
        other = project.root / "parent-copy.json"
        other.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(other)
    with pytest.raises(PermissionError):
        build(project)
