"""Early generation is single-model complete; score release stays all-model complete."""

import json

import pytest
from test_endpoint import LiveMainSlurm, completed_matrix, write

from sr_f1.evaluation import sha_file
from sr_f12 import endpoint
from sr_f12.early_authorization import (
    FROZEN_FILES,
    POLICY,
    create_model_acceptance,
    verify_early_implementation,
    verify_generation_authorization,
)
from sr_f12.protocol import AMENDMENT_ID, TEACHER_QOS, object_hash, scientific_matrix


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = completed_matrix(tmp_path / "run")
    monkeypatch.setattr("getpass.getuser", lambda: "owner")
    monkeypatch.setattr("sr_f12.runner.validate_technical_receipt", lambda *args: None)
    codes = []
    for name, commit in (("training", "a" * 40), ("endpoint", "b" * 40)):
        code = tmp_path / name
        code.mkdir()
        (code / "implementation.py").write_text(name)
        manifest = tmp_path / (name + ".json")
        write(
            manifest,
            dict(
                git_commit=commit, files={"implementation.py": sha_file(code / "implementation.py")}
            ),
        )
        codes.append((code, manifest, commit))
    training, early = codes
    plan = json.loads((root / "config/SR_F1_2_FROZEN.json").read_text())
    write(root / "config/SR_F1_2.json", plan)
    write(
        root / "AMENDMENT.json",
        dict(amendment_id=AMENDMENT_ID, source_commit=training[2], config_sha256=object_hash(plan)),
    )
    freeze = json.loads((root / "SCIENCE_FREEZE.json").read_text())
    write(root / "SCIENCE_FREEZE.json", dict(freeze, source_commit=training[2]))
    config = dict(root=str(root), user="owner", version=AMENDMENT_ID, code_root=str(training[0]))
    write(root / "orchestration/CONFIG.json", config)
    write(root / "endpoint_orchestration/CONFIG.json", config)
    registration = dict(
        status="AUTHORIZED_EARLY_SEALED_GENERATION",
        policy=POLICY,
        version=AMENDMENT_ID,
        user="owner",
        root=str(root),
        code_root=str(early[0]),
        source_manifest=str(early[1]),
        source_commit=early[2],
        source_manifest_sha256=object_hash(json.loads(early[1].read_text())),
        training_source_commit=training[2],
        training_code_root=str(training[0]),
        training_source_manifest=str(training[1]),
        training_source_manifest_sha256=object_hash(json.loads(training[1].read_text())),
        endpoint_config_sha256=sha_file(root / "endpoint_orchestration/CONFIG.json"),
        frozen_files={name: sha_file(root / name) for name in FROZEN_FILES},
    )
    write(root / "EARLY_ENDPOINT_IMPLEMENTATION.json", registration)
    model = scientific_matrix()[0]["model_id"]
    directory = root / "orchestration/tasks" / model
    directory.mkdir(parents=True)
    script = directory / "run.sbatch"
    script.write_text("#!/bin/sh\nexit 0\n")
    state = dict(task_id=model, status="COMPLETE", job_id="123")
    intent = dict(
        task_id=model,
        gpu_count=1,
        job_name="srf12-" + model,
        source_sha256=registration["training_source_manifest_sha256"],
        script=str(script),
        script_sha256=sha_file(script),
    )
    terminal = dict(
        job_id="123",
        state="COMPLETED",
        exit_code="0:0",
        user="owner",
        account="rose",
        qos=TEACHER_QOS,
        name=intent["job_name"],
    )
    done = json.loads((root / "runs" / model / "COMPLETE.json").read_text())
    verified = dict(
        task_id=model,
        receipt_sha256=object_hash(done),
        checkpoint_sha256=done["checkpoint"]["sha256"],
    )
    for name, value in (
        ("STATE", state),
        ("INTENT", intent),
        ("TERMINAL", terminal),
        ("VERIFIED_COMPLETE", verified),
    ):
        write(directory / (name + ".json"), value)
    write(root / "orchestration/STATE.json", dict(tasks={model: state}))
    backend = LiveMainSlurm()
    backend.accounting["123"] = terminal
    return root, model, backend, early


def test_selected_complete_model_allows_sealed_acceptance_while_others_incomplete(setup):
    root, model, backend, _early = setup
    (root / "runs" / scientific_matrix()[-1]["model_id"] / "COMPLETE.json").unlink()
    result = create_model_acceptance(root, model, slurm=backend)
    assert result["model_completion"]["status"] == "SELECTED_MAIN_PATHS_VERIFIED"
    assert len(result["model_completion"]["paths"]) == 1
    assert verify_generation_authorization(root, model, slurm=backend) == result
    with pytest.raises(FileNotFoundError):
        endpoint.verify_main_completion(root)
    assert not (root / "released").exists()


@pytest.mark.parametrize("models", [[], ["unknown"], ["a", "a"], "not-a-list", [None]])
def test_selected_model_ids_fail_closed(setup, models):
    with pytest.raises(PermissionError, match="model selection"):
        endpoint.verify_main_completion(setup[0], model_ids=models)


@pytest.mark.parametrize(
    "tamper",
    ["active", "failed", "missing_terminal", "cached", "script", "checkpoint", "fit", "duplicate"],
)
def test_unfinished_forged_or_changed_model_cannot_generate(setup, tamper):
    root, model, backend, _early = setup
    directory = root / "orchestration/tasks" / model
    if tamper == "active":
        backend.jobs["123"] = {}
    elif tamper == "failed":
        backend.accounting["123"] = dict(backend.accounting["123"], state="FAILED")
    elif tamper == "missing_terminal":
        backend.accounting.clear()
    elif tamper == "cached":
        write(root / "orchestration/STATE.json", dict(tasks={}))
    elif tamper == "script":
        (directory / "run.sbatch").write_text("changed")
    elif tamper == "checkpoint":
        (root / "runs" / model / "checkpoints/step-96.pt").write_bytes(b"bad")
    elif tamper == "fit":
        write(
            root / "runs" / model / "train_fit/step-32/COMPLETE.json",
            dict(status="COMPLETE", count=1),
        )
    else:
        write(
            root / "orchestration/tasks" / scientific_matrix()[1]["model_id"] / "STATE.json",
            dict(job_id="123"),
        )
    with pytest.raises(PermissionError):
        create_model_acceptance(root, model, slurm=backend)
    assert not (root / "endpoint_orchestration/model_acceptance" / (model + ".json")).exists()


@pytest.mark.parametrize(
    "target",
    ["endpoint_source", "training_source", "frozen", "endpoint_config", "registration_owner"],
)
def test_implementation_registration_and_source_tamper_rejected(setup, target):
    root, _model, _backend, early = setup
    registration_path = root / "EARLY_ENDPOINT_IMPLEMENTATION.json"
    registration = json.loads(registration_path.read_text())
    if target == "endpoint_source":
        (early[0] / "implementation.py").write_text("changed")
    elif target == "training_source":
        (root.parent / "training/implementation.py").write_text("changed")
    elif target == "frozen":
        write(root / "SEMANTIC_PREDICTIONS.json", {})
    elif target == "endpoint_config":
        write(root / "endpoint_orchestration/CONFIG.json", {})
    else:
        write(registration_path, dict(registration, user="somebody"))
    with pytest.raises(PermissionError):
        verify_early_implementation(root, *early)


def test_acceptance_tamper_and_missing_acceptance_rejected(setup):
    root, model, backend, _early = setup
    with pytest.raises(FileNotFoundError):
        verify_generation_authorization(root, model, slurm=backend)
    accepted = create_model_acceptance(root, model, slurm=backend)
    write(
        root / "endpoint_orchestration/model_acceptance" / (model + ".json"),
        dict(accepted, intent_sha256="forged"),
    )
    with pytest.raises(PermissionError, match="acceptance differs"):
        verify_generation_authorization(root, model, slurm=backend)


def test_score_release_requires_original_all_twelve_gate_first(setup, monkeypatch):
    root, model, backend, _early = setup
    create_model_acceptance(root, model, slurm=backend)
    monkeypatch.setattr(
        "sr_f12.early_authorization.verify_generation_authorization",
        lambda *a: pytest.fail("early gate cannot replace all-model gate"),
    )
    with pytest.raises(FileNotFoundError):
        endpoint.release_scores(root, model)
    assert not (root / "released").exists()


def test_final_release_links_exact_early_model_state(setup, monkeypatch):
    root, model, backend, _early = setup
    accepted = create_model_acceptance(root, model, slurm=backend)
    full = endpoint.verify_main_completion(root)
    directory = root / "sealed/evaluation" / model
    write(directory / "GENERATION_AUTHORIZATION.json", accepted)
    complete = dict(
        status="COMPLETE",
        generation_gate="MODEL_TERMINAL_SEALED",
        generation_authorization_sha256=object_hash(accepted),
        panels=[],
    )
    write(directory / "COMPLETE.json", complete)
    monkeypatch.setattr(endpoint, "verify_endpoint_authorization", lambda root: full)
    monkeypatch.setattr(
        "sr_f12.early_authorization.verify_generation_authorization", lambda *a: accepted
    )
    monkeypatch.setattr(endpoint, "read_jsonl", lambda *a: [])
    monkeypatch.setattr(endpoint, "load_verified_manifest", lambda *a: [])
    result = endpoint.release_scores(root, model)
    assert result["status"] == "RELEASED_AFTER_ALL_MAIN_COMPLETE"
    full["paths"][0]["checkpoint_sha256"] = "different"
    with pytest.raises(PermissionError, match="differs from final matrix"):
        endpoint.release_scores(root, model)


def test_early_worker_rejects_gate_before_loading_test(setup, monkeypatch):
    root, model, _backend, early = setup
    monkeypatch.setattr(
        "sr_f12.early_authorization.verify_generation_authorization",
        lambda *a: (_ for _ in ()).throw(PermissionError("sealed")),
    )
    monkeypatch.setattr(endpoint, "read_jsonl", lambda *a: pytest.fail("TEST was read"))
    with pytest.raises(PermissionError, match="sealed"):
        endpoint.run_endpoint(
            root,
            model,
            source_manifest=early[1],
            code_root=early[0],
            model_path=root,
            early_sealed=True,
        )


def test_early_worker_only_generates_unscored_panels_and_distinct_receipt(setup, monkeypatch):
    root, model, backend, early = setup
    accepted = create_model_acceptance(root, model, slurm=backend)
    adapter = json.loads((root / "runs" / model / "adapters/step-96/IDENTITY.json").read_text())
    write(
        root / "MODEL_ENVIRONMENT_IDENTITY.json",
        dict(model_path=str(root), model_weights_hash="base"),
    )
    write(root / "manifests/TRAIN_FIT_QIDS.json", [])
    monkeypatch.setattr(
        "sr_f12.early_authorization.verify_generation_authorization", lambda *a: accepted
    )
    monkeypatch.setattr(
        endpoint,
        "verify_endpoint_authorization",
        lambda *a: pytest.fail("early generation does not pretend to authorize full matrix"),
    )
    monkeypatch.setattr("mm_dev.runtime.configure_audited_backend", lambda: {})

    class Runtime:
        def __init__(self, *args, **kwargs):
            self.identity = {}

        def verify_identity(self, *args, **kwargs):
            pass

        def current_adapter_identity(self):
            return dict(adapter_parameter_hash=adapter["policy_hash"])

    monkeypatch.setattr("sr_f12.runtime.SRF12Runtime", Runtime)
    monkeypatch.setattr(endpoint, "read_jsonl", lambda *a: [])
    monkeypatch.setattr(endpoint, "iter_model_slots", lambda *a, **kw: [])
    monkeypatch.setattr(endpoint, "chartqa_panel", lambda *a: ([], {}, {}))
    calls = []

    def generate(*args, **kwargs):
        assert kwargs["score_non_test"] is False
        path = args[6]
        count = 2500 if path.name == "chartqa" else 15104
        result = dict(status="COMPLETE", generated_records=count)
        write(path / "COMPLETE.json", result)
        calls.append(path.name)
        return result

    monkeypatch.setattr(endpoint, "evaluate_shard", generate)
    result = endpoint.run_endpoint(
        root,
        model,
        source_manifest=early[1],
        code_root=early[0],
        model_path=root,
        early_sealed=True,
    )
    assert result["generation_gate"] == "MODEL_TERMINAL_SEALED"
    assert result["generation_authorization_sha256"] == object_hash(accepted)
    assert "main_completion_sha256" not in result
    assert result["aggregate_scores_released"] is False
    assert calls == ["final", "chartqa"]
    assert not (root / "released").exists()
