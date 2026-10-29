"""SYNTHETIC plan tests, never real confirmation, numerical or scientific PASS.

The historical-confirmation dependency is isolated by a visibly synthetic test
stub in positive registration tests. Phase/source/DATA/parent and overlay
verification remain real. Actual confirmation rejection is tested separately.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sr_f6 import historical_closure, pilot_v3_inputs, pilot_v3_registration, registration
from sr_f6.common import file_hash
from sr_f6.planning import draw_seed, formal_paths
from sr_f6.rollouts import _bound
from sr_f6.transactions import RunTransactions
from srf6_ref.core import seed_for
from tests.sr_f6.test_pilot_v3_inputs import v3_factory, v3_project  # noqa: F401
from tests.sr_f6.test_registration import factory, prototype, write  # noqa: F401


@pytest.fixture
def registered_inputs(v3_project, monkeypatch):  # noqa: F811 - imported pytest fixture
    project = v3_project
    project.overlay = pilot_v3_inputs.freeze_overlay(project.root, phase_context=project.context)
    relative = "PHASES/PILOT_V3/CONFIRMATION_BINDING.json"
    value = {
        "status": "SYNTHETIC_UNIT_FIXTURE_NOT_CONFIRMATION",
        "evidence_scope": "CPU_TEST_ONLY_NO_SCIENTIFIC_PASS",
        "phase_identity_sha256": project.context["identity_sha256"],
        "training_authorized": False,
        "generation_authorized": False,
    }
    write(project.root / relative, value)
    project.confirmation_ref = {"path": relative, "sha256": file_hash(project.root / relative)}
    project.actual_confirmation_verifier = historical_closure.require_confirmation_binding

    def synthetic_dependency(root, *, phase_context, confirmation_binding_ref):
        assert root == project.root and phase_context == project.context
        bound = _bound(SimpleNamespace(), root, confirmation_binding_ref)
        if bound != value or confirmation_binding_ref != project.confirmation_ref:
            raise PermissionError("Synthetic dependency changed")
        return {
            "binding_ref": copy.deepcopy(confirmation_binding_ref),
            "binding": bound,
            "verification": {"evidence_scope": "SYNTHETIC_UNIT_FIXTURE_NOT_CONFIRMATION"},
        }

    monkeypatch.setattr(historical_closure, "require_confirmation_binding", synthetic_dependency)
    return project


def build(project):
    return pilot_v3_registration.build_pilot_v3_registration(
        project.root,
        phase_context=project.context,
        parent_references=project.refs,
        overlay_ref=project.overlay["manifest_ref"],
        confirmation_binding_ref=project.confirmation_ref,
    )


def freeze(project):
    return pilot_v3_registration.freeze_pilot_v3_registration(
        project.root,
        phase_context=project.context,
        parent_references=project.refs,
        overlay_ref=project.overlay["manifest_ref"],
        confirmation_binding_ref=project.confirmation_ref,
    )


def verify(project, result):
    return pilot_v3_registration.verify_pilot_v3_registration(
        project.root,
        phase_context=project.context,
        expected_spec_sha256=result["spec_sha256"],
    )


def test_exact_nine_complete_seed_streams_and_budgets(registered_inputs):
    project = registered_inputs
    before = copy.deepcopy(
        (project.context, project.refs, project.overlay, project.confirmation_ref)
    )
    result = build(project)
    assert before == (project.context, project.refs, project.overlay, project.confirmation_ref)
    assert build(project) == result
    assert not (project.root / pilot_v3_registration.DIRECTORY).exists()
    spec = result["spec"]
    assert spec["phase"] == "PILOT_V3"
    assert spec["schema"] == pilot_v3_registration.SCHEMA
    assert spec["data_manifest"] == project.identity["data_manifest"]
    assert spec["actual_source_manifest"] == project.context["identity"]["source_manifest"]
    assert spec["input_overlay"] == project.overlay["manifest_ref"]
    assert spec["confirmation_binding_ref"] == project.confirmation_ref
    assert spec["training_authorized"] is spec["generation_authorized"] is False
    assert spec["execution_status"] == "NOT_RUN"
    expected_order = [
        row
        for row in formal_paths(project.config)
        if row["repeat"] == 0 and row["arm"] in registration.ARMS
    ]
    assert [(row["parent"], row["arm"]) for row in spec["paths"]] == [
        (row["parent"], row["arm"]) for row in expected_order
    ]
    assert len(result["files"]) == 19
    assert spec["budgets"] == {
        "primary_paths": 9,
        "updates_per_path": 12,
        "primary_logical_updates": 108,
        "responses_per_update": 128,
        "primary_responses": 13824,
        "recovery_new_responses": 1024,
        "dev_eval_responses": 2304,
    }
    for path in spec["paths"]:
        assert "/PILOT_V3/" in path["run_id"]
        value = result["files"][f"{path['run_directory']}/ROLLOUT_REGISTRATION.json"]
        identity = value["identity"]
        assert identity["phase"] == "PILOT_V3"
        assert identity["input_overlay_sha256"] == spec["input_overlay"]["sha256"]
        assert identity["input_overlay"] == spec["input_overlay"]
        assert identity["prompt_version"] == "srf6-interface-v3"
        assert (
            value["inputs"]["public"] == project.overlay["inputs_by_split"]["DEV_PILOT"]["public"]
        )
        assert value["inputs"]["private"]["path"] == "DATA/DEV_PILOT/TASKS_PRIVATE.jsonl"
        assert (
            len(value["schedule"]) == len(value["draw_seeds"]) == len(value["shuffle_seeds"]) == 12
        )
        for step in range(1, 13):
            for group in range(16):
                assert value["draw_seeds"][step - 1][group] == [
                    draw_seed(
                        project.config["master_seed"],
                        path["parent"],
                        0,
                        step,
                        group // 4,
                        group % 4,
                        draw,
                    )
                    for draw in range(8)
                ]
                assert value["shuffle_seeds"][step - 1][group] == seed_for(
                    project.config["master_seed"],
                    "pilot_shuffle",
                    path["parent"],
                    0,
                    path["arm"],
                    step,
                    group,
                )
    evaluation = spec["evaluation"]
    assert evaluation["inputs"] == project.overlay["inputs_by_split"]["DEV_EVAL"]
    assert len(evaluation["endpoints"]) == 12
    assert sum(row["responses"] for row in evaluation["endpoints"]) == 2304
    assert sum(row["kind"] == "FROZEN_PARENT" for row in evaluation["endpoints"]) == 3
    for parent, rows in evaluation["seeds_by_parent"].items():
        assert len(rows) == 96
        for row in rows:
            assert row["seeds"] == [
                seed_for(project.config["master_seed"], "pilot_dev_eval", parent, row["qid"], draw)
                for draw in range(2)
            ]
    recovery = spec["recovery"]
    assert (recovery["parent"], recovery["arm"], recovery["repeat"]) == ("b0", "U11", 0)
    assert recovery["update_steps"] == list(range(5, 13))
    assert recovery["new_generated_responses"] == 1024
    assert recovery["physical_directory"].startswith("PHASE_RUNS/PILOT_V3/")
    assert recovery["binding_path"].startswith("PHASE_RUNS/PILOT_V3/")
    assert evaluation["recovery_endpoint_is_additional_checkpoint"] is False
    assert len(spec["diagnostics"]["row_indices"]) == 64


def test_scientific_fields_exactly_equal_old_fixed_generator(registered_inputs):
    project = registered_inputs
    _, config, _, data_ref, order_ref, order, parents, binding_ref = registration._inputs(
        project.root, project.context, project.refs
    )
    old = registration._assemble_registration(
        project.root,
        project.context,
        project.refs,
        config,
        data_ref,
        order_ref,
        order,
        parents,
        binding_ref,
        shared_inputs={
            name: data_ref(f"DATA/DEV_PILOT/{filename}.jsonl")
            for name, filename in (("public", "MODEL_INPUTS"), ("private", "TASKS_PRIVATE"))
        },
        evaluation_inputs={
            name: data_ref(f"DATA/DEV_EVAL/{filename}.jsonl")
            for name, filename in (("public", "MODEL_INPUTS"), ("private", "TASKS_PRIVATE"))
        },
    )
    new = build(project)
    for field in (
        "parent_tensors",
        "parent_references",
        "order_source",
        "budgets",
        "diagnostics",
        "data_manifest",
        "source_binding",
    ):
        assert new["spec"][field] == old["spec"][field]
    for path in new["spec"]["paths"]:
        key = f"{path['run_directory']}/ROLLOUT_REGISTRATION.json"
        for field in (
            "format",
            "split",
            "factor",
            "master_seed",
            "schedule",
            "shuffle_domain",
            "draw_seeds",
            "shuffle_seeds",
        ):
            assert new["files"][key][field] == old["files"][key][field]
    for field in (
        "comparison",
        "update_steps",
        "new_generated_responses",
        "continuity_rule",
        "reuse_primary_steps5_to12_raw",
    ):
        assert new["spec"]["recovery"][field] == old["spec"]["recovery"][field]
    for field in ("descriptive_summary", "seeds_by_parent", "responses", "draws_per_question"):
        assert new["spec"]["evaluation"][field] == old["spec"]["evaluation"][field]


def test_transaction_binding_is_acyclic_and_immutable(registered_inputs):
    project = registered_inputs
    result = freeze(project)
    for path in result["spec"]["paths"]:
        run = Path(result["directory"]) / path["run_directory"]
        identity = json.loads((run / "RUN_IDENTITY.json").read_bytes())
        assert "pilot_spec_sha256" not in identity and "spec_sha256" not in identity
        assert identity["source_sha256"] == result["spec"]["actual_source_manifest"]["sha256"]
        with RunTransactions(run, identity) as tx:
            rollout = _bound(tx, run, identity["rollout_registration"])
            assert rollout["identity"] == {
                k: v for k, v in identity.items() if k != "rollout_registration"
            }
            assert tx.published() == []
    verified = verify(project, result)
    assert verified["status"] == "VERIFIED_REGISTRATION_NOT_AUTHORIZED"
    assert verified["overlay"] == project.overlay
    assert (
        verified["confirmation_binding"]["verification"]["evidence_scope"]
        == "SYNTHETIC_UNIT_FIXTURE_NOT_CONFIRMATION"
    )
    assert verified["training_authorized"] is verified["generation_authorized"] is False


def test_idempotent_publish_preserves_every_byte(registered_inputs):
    project = registered_inputs
    result = freeze(project)
    destination = Path(result["directory"])
    original = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in destination.rglob("*.json")}
    assert len(original) == 19
    assert freeze(project) == result
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in original} == original


@pytest.mark.parametrize(
    "kind",
    [
        "spec_seed",
        "run_seed",
        "identity",
        "overlay",
        "confirmation",
        "parent",
        "source",
        "old_phase",
    ],
)
def test_tamper_or_relabel_rejected(registered_inputs, kind):
    project = registered_inputs
    result = freeze(project)
    spec = result["spec"]
    destination = Path(result["directory"])
    if kind == "spec_seed":
        spec["evaluation"]["seeds_by_parent"]["b0"][0]["seeds"][0] += 1
        path = destination / pilot_v3_registration.SPEC_FILE
        write(path, spec)
        result["spec_sha256"] = file_hash(path)
    elif kind in {"run_seed", "identity"}:
        path = (
            destination
            / spec["paths"][0]["run_directory"]
            / ("ROLLOUT_REGISTRATION.json" if kind == "run_seed" else "RUN_IDENTITY.json")
        )
        value = json.loads(path.read_bytes())
        if kind == "run_seed":
            value["draw_seeds"][0][0][0] += 1
        else:
            value["input_overlay_sha256"] = "9" * 64
        write(path, value)
    elif kind == "overlay":
        path = project.root / project.overlay["inputs_by_split"]["DEV_PILOT"]["public"]["path"]
        path.write_bytes(path.read_bytes() + b"{}\n")
    elif kind == "confirmation":
        write(project.root / project.confirmation_ref["path"], {"status": "PASS"})
    elif kind == "parent":
        reference = project.refs["b0"]
        path = project.root / reference["path"]
        value = json.loads(path.read_bytes())
        value["step"] = 12
        write(path, value)
    elif kind == "source":
        Path(pilot_v3_registration.__file__).write_text("# synthetic change after freeze\n")
    else:
        project.context["phase"] = "PILOT_V1"
    with pytest.raises(PermissionError):
        verify(project, result)


def test_partial_publish_preserved_and_never_completed(registered_inputs):
    directory = registered_inputs.root / pilot_v3_registration.DIRECTORY
    directory.mkdir(parents=True)
    partial = directory / pilot_v3_registration.SPEC_FILE
    partial.write_bytes(b"SYNTHETIC partial registration\n")
    with pytest.raises((PermissionError, FileNotFoundError)):
        freeze(registered_inputs)
    assert partial.read_bytes() == b"SYNTHETIC partial registration\n"


def test_failed_publication_keeps_attempt_without_published_plan(registered_inputs, monkeypatch):
    def fail(*args):
        raise OSError("SYNTHETIC interrupted publication")

    monkeypatch.setattr(pilot_v3_registration.os, "rename", fail)
    with pytest.raises(OSError, match="SYNTHETIC"):
        freeze(registered_inputs)
    assert not (registered_inputs.root / pilot_v3_registration.DIRECTORY).exists()
    attempts = list(
        (registered_inputs.root / "PHASE_RUNS/PILOT_V3").glob(".pilot-v3-registration-*")
    )
    assert len(attempts) == 1
    assert len(list(attempts[0].rglob("*.json"))) == 19


def test_actual_confirmation_boundary_rejects_synthetic_input(registered_inputs, monkeypatch):
    monkeypatch.setattr(
        historical_closure,
        "require_confirmation_binding",
        registered_inputs.actual_confirmation_verifier,
    )
    with pytest.raises((PermissionError, FileNotFoundError)):
        build(registered_inputs)
    assert not (registered_inputs.root / pilot_v3_registration.DIRECTORY).exists()
