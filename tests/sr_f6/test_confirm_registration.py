"""Independent confirmation boundaries; fixtures never certify real model work."""

import copy
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from sr_f6 import confirm_registration as registration
from sr_f6.common import digest, file_hash, write_once
from srf6_ref.core import FAMILIES, seed_for


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    monkeypatch.setattr(registration, "PROJECT", root)
    monkeypatch.setattr(registration.old, "PROJECT", root)
    monkeypatch.setattr(registration, "ROOT", root / "confirm")
    monkeypatch.setattr(registration, "ORIGIN_ROOT", root / "original")
    monkeypatch.setattr(registration, "SCREEN_ROOT", root / "screen")
    monkeypatch.setattr(registration, "SCREEN_DEPLOYMENT", root / "sr_f6_code_5ca3fe9")
    return root


@pytest.fixture
def inputs():
    public, private = [], {}
    for family in FAMILIES:
        for index in range(8):
            rid = f"DEV_SCHEMA_CONFIRM_{family}_{index:04d}"
            for factor in ("P0", "P1"):
                for variant in range(4):
                    qid = f"{rid}_{factor}_v{variant}"
                    identity = dict(
                        qid=qid,
                        root_id=rid,
                        family=family,
                        factor=factor,
                        variant=variant,
                        split=registration.SPLIT,
                    )
                    row = dict(
                        **identity,
                        protocol="structured_v4",
                        prompt_version=registration.PROMPT_VERSION,
                        model_input=dict(
                            prompt="unit-test-only-" + qid,
                            image_file="DATA/images/test.png",
                            image_sha256="a" * 64,
                        ),
                    )
                    public.append(row)
                    private[qid] = dict(**identity, unit_test=True)
    return sorted(public, key=lambda row: row["qid"]), private


@pytest.fixture
def hashes():
    return dict(zip(registration.PARENTS, ("a" * 64, "b" * 64, "c" * 64), strict=True))


def test_plan_freezes_exact_independent_g8_matrix_partitions_and_credit(inputs, hashes):
    public, private = inputs
    before = copy.deepcopy(inputs)
    plan = registration.make_plan(public, private, hashes)
    assert inputs == before
    assert len(plan["attempts"]) == 9216 and len(plan["groups"]) == 1152
    assert len(plan["workers"]) == 6
    assert plan["engineered_anchor_roots"] == 18
    assert plan["parent_policy_hashes"] == hashes
    assert plan["training_allowed"] is plan["generation_allowed"] is False
    assert plan["gates"]["observability_cells"] == 18
    assert set(plan["gates"]["each_count_at_least"].values()) == {1383}
    assert plan["gates"]["review_seed"] == 2026102701
    assert plan["first_g8_replay"]["total_sequences"] == 48
    later = plan["followup_requirements"]["numerical_replay_selection"]
    assert later["qid"] == public[0]["qid"] and later["sequences"] == 24
    assert later["new_generated_responses"] == later["optimizer_updates"] == 0
    attempts = {(r["parent"], r["qid"], r["draw"]): r for r in plan["attempts"]}
    assert len(attempts) == 9216
    for row in plan["attempts"]:
        assert row["seed"] == seed_for(
            20261028,
            "srf6-confirm-v3",
            "confirm",
            row["parent"],
            row["root_id"],
            row["variant"],
            row["draw"],
        )
        other = row["qid"].replace("_P0_", "_P1_")
        assert row["seed"] == attempts[row["parent"], other, row["draw"]]["seed"]
    for group in plan["groups"]:
        task = private[group["qid"]]
        assert group["shuffle_seed"] == seed_for(
            20261028, "census_shuffle", group["parent"], task["factor"], group["group_id"]
        )
        assert group["seeds"] == [
            attempts[group["parent"], group["qid"], d]["seed"] for d in range(8)
        ]
    for parent in registration.PARENTS:
        workers = [r for r in plan["workers"] if r["parent"] == parent]
        for worker in workers:
            assert worker["qids"] == [r["qid"] for r in public[worker["shard"] :: 2]]
            assert worker["first_qid"] == worker["qids"][0]
            assert worker["groups"] == 192 and worker["responses"] == 1536
            assert "task_id" not in worker
        assert set(workers[0]["qids"]).isdisjoint(workers[1]["qids"])


@pytest.mark.parametrize(
    "mutation", ["old_split", "missing", "duplicate", "private", "secret", "prompt"]
)
def test_plan_rejects_partial_or_masqueraded_data(inputs, hashes, mutation):
    public, private = inputs
    if mutation == "old_split":
        public[0]["split"] = private[public[0]["qid"]]["split"] = "DEV_CAL"
    elif mutation == "missing":
        public.pop()
    elif mutation == "duplicate":
        public[-1] = public[0]
    elif mutation == "private":
        private[public[0]["qid"]]["factor"] = "P9"
    elif mutation == "secret":
        public[0]["model_input"]["gold_answer"] = 2
    else:
        public[0]["prompt_version"] = "srf6-interface-v2"
    with pytest.raises(PermissionError):
        registration.make_plan(public, private, hashes)


def test_prompt_transition_frozen_for_future_panels_without_access():
    transition = registration.prompt_transition()
    assert transition["public_inputs"] == ["world.categories", "query"]
    assert set(transition["structured_splits"]) == {
        "DEV_SCHEMA_CONFIRM",
        "DEV_PILOT",
        "TRAIN",
        "DEV_EVAL",
        "TEST_ID",
        "TEST_COMPOSITION",
        "TEST_CLASSOUT",
    }
    assert transition["unchanged_protocols"] == ["NATURAL_INTERFACE", "LEGACY", "CHARTQA"]
    assert transition["future_model_access_authorized"] is False
    assert transition["test_results_read"] is False


def test_path_and_actual_deployment_boundaries(project, monkeypatch):
    original = project / "original"
    original.mkdir()
    with pytest.raises(PermissionError, match="independent confirmation"):
        registration._root(original)
    with pytest.raises(PermissionError, match="outside"):
        registration._path(project.parent)
    (project / "confirm").symlink_to(original, target_is_directory=True)
    with pytest.raises(PermissionError, match="Symlinked"):
        registration._root(project / "confirm")
    deployment = project / "sr_f6_code_abcdef1"
    deployment.mkdir()
    monkeypatch.setattr(registration, "_actual_deployment", lambda: deployment)
    assert registration._deployment("abcdef1" + "0" * 33) == deployment
    with pytest.raises(PermissionError):
        registration._deployment("0" * 40)
    monkeypatch.setattr(registration, "_actual_deployment", lambda: registration.SCREEN_DEPLOYMENT)
    registration.SCREEN_DEPLOYMENT.mkdir()
    with pytest.raises(PermissionError, match="own actual"):
        registration._deployment(registration.SCREEN_REVISION)


def test_unknown_parent_rejected_before_access():
    with pytest.raises(PermissionError, match="Unknown"):
        registration.verify_registered("/missing", parent="b3")


@pytest.mark.parametrize("changed", [None, *registration.FROZEN_IMPLEMENTATIONS])
def test_confirmation_cannot_change_the_successful_prompt_or_fixed_math(project, changed):
    screen = project / "screen"
    screen.mkdir()
    prior = [
        dict(path=str(registration.SCREEN_DEPLOYMENT / relative), sha256="a" * 64)
        for relative in registration.FROZEN_IMPLEMENTATIONS
    ]
    write_once(screen / "SOURCE_MANIFEST.json", {"files": prior})
    write_once(
        screen / "REGISTRATION.json",
        {"source_manifest": registration._ref(screen, "SOURCE_MANIFEST.json")},
    )
    deployment = project / "sr_f6_code_abcdef1"
    current = [
        dict(path=str(deployment / relative), sha256="b" * 64 if relative == changed else "a" * 64)
        for relative in registration.FROZEN_IMPLEMENTATIONS
    ]
    if changed is None:
        assert registration._frozen_implementations(deployment, current) == dict.fromkeys(
            registration.FROZEN_IMPLEMENTATIONS, "a" * 64
        )
    else:
        with pytest.raises(PermissionError, match="implementation changed"):
            registration._frozen_implementations(deployment, current)


@pytest.fixture
def processor(inputs):
    public, _ = inputs
    route = dict(
        native_input_tensor_hashes=dict.fromkeys(
            ("input_ids", "attention_mask", "mm_token_type_ids", "pixel_values", "image_grid_thw"),
            "1" * 64,
        ),
        input_tensor_hash="2" * 64,
        chat_text_sha256="3" * 64,
        source_image_sha256="a" * 64,
        processor_hash="b" * 64,
        chat_template_hash="c" * 64,
        source_size=[1280, 896],
        input_float_dtype="float32",
        probability_dtype="float32",
        stop_rule="first_balanced_json",
        assistant_prefill="{",
        sr_f6_protocol_version="structured_v4-srf6-1.0",
        image_grid_thw=[[1, 46, 66]],
        processed_size=[1056, 736],
        image_token_count=759,
    )
    receipt = dict(
        schema=registration.PREFLIGHT_SCHEMA,
        phase=registration.PHASE,
        status="ACTUAL_PROCESSOR_COMPLETE_VISUAL_REVIEW_PENDING",
        execution_origin="SERVER_ACTUAL",
        registration_sha256="f" * 64,
        model_calls=0,
        optimizer_updates=0,
        training_allowed=False,
        generation_allowed=False,
        new_visual_review_required=True,
        identity=dict(processor_hash="b" * 64, chat_template_hash="c" * 64),
        inputs=[
            dict(
                qid=r["qid"],
                model_input_sha256=digest(r["model_input"]),
                routing=copy.deepcopy(route),
            )
            for r in public
        ],
        images=[
            dict(
                qid=r["qid"],
                source_file=r["model_input"]["image_file"],
                source_sha256="a" * 64,
                processed_file=f"PREFLIGHT/processor/{r['qid']}.png",
                processed_file_sha256="d" * 64,
                processed_rgb_sha256="e" * 64,
                routing=copy.deepcopy(route),
            )
            for r in public
            if r["factor"] == "P1" and r["variant"] == 0
        ],
        tokenizer_fixtures=[
            dict(text=text, token_ids=[1, 2], exact_roundtrip=True)
            for text in registration.TOKENIZER_FIXTURES
        ],
        verified_dependencies=[dict(path="unit-test-only-not-real-acceptance")],
        native_regression=dict(path="PREFLIGHT/NATIVE_REGRESSION.json", sha256="b" * 64),
    )
    return receipt, public


def test_processor_completion_is_not_visual_or_sampling_permission(processor):
    receipt, public = processor
    registration._validate_processor(receipt, "f" * 64, public)
    assert receipt["generation_allowed"] is False
    assert receipt["new_visual_review_required"] is True


@pytest.mark.parametrize(
    "mutation",
    [
        "partial",
        "duplicate",
        "prompt",
        "old48",
        "pin",
        "route",
        "native",
        "fixture",
        "bool",
        "permission",
    ],
)
def test_actual_processor_rejects_incomplete_or_changed_evidence(processor, mutation):
    receipt, public = processor
    if mutation == "partial":
        receipt["inputs"].pop()
    elif mutation == "duplicate":
        receipt["inputs"][-1] = receipt["inputs"][0]
    elif mutation == "prompt":
        receipt["inputs"][0]["model_input_sha256"] = "0" * 64
    elif mutation == "old48":
        receipt["images"][0]["qid"] = "DEV_CAL_OLD_P1_v0"
    elif mutation == "pin":
        receipt["registration_sha256"] = "0" * 64
    elif mutation == "route":
        receipt["inputs"][0]["routing"]["stop_rule"] = "lenient"
    elif mutation == "native":
        receipt.pop("native_regression")
    elif mutation == "fixture":
        receipt["tokenizer_fixtures"][0]["token_ids"] = [True]
    elif mutation == "bool":
        receipt["model_calls"] = False
    else:
        receipt["generation_allowed"] = True
    with pytest.raises(PermissionError):
        registration._validate_processor(receipt, "f" * 64, public)


@pytest.fixture
def visual(project, processor):
    root = project / "confirm"
    root.mkdir()
    receipt, _ = processor
    for image in receipt["images"]:
        path = root / image["processed_file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"unit-test-image")
        image["processed_file_sha256"] = file_hash(path)
    write_once(root / "PREFLIGHT/PROCESSOR.json", receipt)
    processor_ref = registration._ref(root, "PREFLIGHT/PROCESSOR.json")
    write_once(root / "PREFLIGHT/visual_observations.json", {"unit_test": True})
    judgment = dict(
        schema="srf6-confirm-visual-review-v1",
        phase=registration.PHASE,
        status="PASS",
        reviewer_type="MODEL_VISUAL",
        reviewer_identity="explicit-unit-test-fixture",
        independent_of_processor_program=True,
        processor_receipt=processor_ref,
        registration_sha256=receipt["registration_sha256"],
        original_screen_visual_reused=False,
        review_evidence=registration._ref(root, "PREFLIGHT/visual_observations.json"),
        items=[
            dict(
                qid=r["qid"],
                status="PASS",
                observation="synthetic fixture only",
                source_sha256=r["source_sha256"],
                processed_file_sha256=r["processed_file_sha256"],
                labels_readable=True,
                bars_and_legend_readable=True,
                no_clipping=True,
            )
            for r in receipt["images"]
        ],
    )
    return root, judgment, receipt, processor_ref


def test_new_visual_review_requires_all_new_actual_images(visual):
    registration._validate_visual(*visual)


@pytest.mark.parametrize(
    "mutation",
    [
        "program",
        "reuse",
        "missing",
        "duplicate",
        "observation",
        "clipping",
        "image",
        "trace",
        "registration",
        "independent",
    ],
)
def test_visual_autopass_old_review_and_modified_pixels_rejected(visual, mutation):
    root, judgment, receipt, ref = visual
    if mutation == "program":
        judgment["reviewer_type"] = "PROGRAM"
    elif mutation == "reuse":
        judgment["original_screen_visual_reused"] = True
    elif mutation == "missing":
        judgment["items"].pop()
    elif mutation == "duplicate":
        judgment["items"][-1] = judgment["items"][0]
    elif mutation == "observation":
        judgment["items"][0]["observation"] = "  "
    elif mutation == "clipping":
        judgment["items"][0]["no_clipping"] = False
    elif mutation == "image":
        (root / receipt["images"][0]["processed_file"]).write_bytes(b"changed")
    elif mutation == "trace":
        judgment["review_evidence"]["sha256"] = "0" * 64
    elif mutation == "registration":
        judgment["registration_sha256"] = "0" * 64
    else:
        judgment["independent_of_processor_program"] = False
    with pytest.raises(PermissionError):
        registration._validate_visual(root, judgment, receipt, ref)


def test_failed_preflight_preserved_before_any_processor_load(project, monkeypatch):
    root = project / "confirm"
    root.mkdir()
    write_once(root / "REGISTRATION.json", {"unit_test": True})

    def fail(*args, **kwargs):
        raise PermissionError("changed actual source")

    monkeypatch.setattr(registration, "verify_registered", fail)
    with pytest.raises(PermissionError, match="changed actual source"):
        registration.preflight(root)
    receipt = json.loads((root / "PREFLIGHT/PROCESSOR.json").read_text())
    assert receipt["status"] == "FAIL" and receipt["model_calls"] == 0
    assert not (root / "PREFLIGHT/SEAL.json").exists()
    with pytest.raises(PermissionError, match="Preserve"):
        registration.preflight(root)


def test_existing_registration_never_overwritten(project):
    (project / "confirm").mkdir()
    with pytest.raises(PermissionError, match="Preserve"):
        registration.register(project / "confirm", "a" * 40)


def test_predecessor_runs_only_in_its_immutable_source(project, monkeypatch):
    screen, deployment = project / "screen", project / "sr_f6_code_5ca3fe9"
    screen.mkdir()
    package = deployment / "src/sr_f6"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "interface_v3_registration.py").write_text(
        "from pathlib import Path\ndef _actual_deployment():\n"
        "    return Path(__file__).resolve().parents[2]\n"
    )
    report = dict(
        decision=dict(status="SCREEN_PASS", stage2_parents=[]),
        responses=576,
        optimizer_updates=0,
        training_allowed=False,
        records_sha256="d" * 64,
        saved_token_validation=dict(status="PASS", checked_responses=576),
        accounting={},
    )
    (package / "interface_v3_screen.py").write_text(
        "import json\nfrom pathlib import Path\n"
        "def _stage1_assessment(root):\n"
        "    return json.loads((root / 'STAGE1/SCREEN/SCREEN_ASSESSMENT.json').read_text())\n"
    )
    write_once(
        screen / "REGISTRATION.json",
        dict(
            revision=registration.SCREEN_REVISION,
            deployment=str(deployment),
            root=str(screen),
            phase="INTERFACE_SCREEN_V3",
            training_allowed=False,
        ),
    )
    write_once(screen / "STAGE1/SCREEN/SCREEN_ASSESSMENT.json", report)
    write_once(screen / "STAGE1/SCREEN/ASSESSMENT_SEAL.json", {"unit_test": True})
    monkeypatch.setattr(registration, "_screen_state", lambda: [{"unit_test_state": 1}])
    result = registration._screen_basis()
    assert result["deployment"] == str(deployment)
    assert result["status"] == "SCREEN_PASS" and result["training_authorization_reused"] is False
    registration._verify_screen_basis(result)
    monkeypatch.setattr(registration, "_screen_state", lambda: [{"unit_test_state": 2}])
    with pytest.raises(PermissionError, match="evidence changed"):
        registration._verify_screen_basis(result)


@pytest.mark.parametrize("status", ["SCREEN_FAIL_STOP", "STAGE2_REQUIRED", "COMPLETE"])
def test_prior_fake_or_failed_screen_not_a_permission(project, monkeypatch, status):
    (project / "screen").mkdir()
    (project / "sr_f6_code_5ca3fe9").mkdir()
    basis = dict(
        producer_phase="INTERFACE_SCREEN_V3",
        root=str(project / "screen"),
        deployment=str(project / "sr_f6_code_5ca3fe9"),
        revision=registration.SCREEN_REVISION,
        status=status,
        training_authorization_reused=False,
    )
    with pytest.raises(PermissionError, match="predecessor identity"):
        registration._verify_screen_basis(basis)


def test_cpu_launcher_cannot_seal_visual_or_allow_unknown_work():
    path = Path(__file__).resolve().parents[2] / "scripts/sr_f6/confirm_cpu.sbatch"
    text = path.read_text()
    assert '"$SRF6_ACTION" == prepare' in text and '"$SRF6_ACTION" == preflight' in text
    assert "CUDA_VISIBLE_DEVICES=''" in text
    assert "#SBATCH --job-name=LH-SRF6-" in text
    assert "soujanya-poria-startfund-2026-03" in text
    assert "seal --root" not in text
    assert "#SBATCH --mem=12G" in text


@pytest.fixture
def cpu_launcher(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    operations = project / "sr_f6_operations_20261027/CONFIRM_V3_EXECUTION_20261028"
    operations.mkdir(parents=True)
    code = project / "sr_f6_code_abcdef1"
    code.mkdir()
    python = project / "envs/ssvc-py312/bin/python3.12"
    python.parent.mkdir(parents=True)
    python.write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\nprintf "%s\\n" "$*" >> "$TEST_CALLS"\n'
        'action="$4"; while [[ "$1" != --root ]]; do shift; done; root="$2"\n'
        '[[ -z "$CUDA_VISIBLE_DEVICES" ]]\n'
        'if [[ "$action" == register ]]; then mkdir "$root"; fi\n'
    )
    python.chmod(0o700)
    original = Path(__file__).resolve().parents[2] / "scripts/sr_f6/confirm_cpu.sbatch"
    script = tmp_path / "launch.sh"
    script.write_text(
        original.read_text().replace("/projects/_ssd/varunssd/louis-ssvc", str(project))
    )
    calls = tmp_path / "calls.txt"
    env = dict(
        os.environ,
        SRF6_CODE=str(code),
        SRF6_ROOT=str(project / "sr_f6_confirm_v3_20261028"),
        SRF6_ACTION="prepare",
        SRF6_REVISION="abcdef1" + "0" * 33,
        SLURM_JOB_ID="1234",
        SLURM_JOB_NAME="LH-SRF6-Confirm-CPU",
        TEST_CALLS=str(calls),
    )
    return script, env, calls


def test_cpu_prepare_registers_then_preflights_using_operation_cache(cpu_launcher):
    script, env, calls = cpu_launcher
    process = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    lines = calls.read_text().splitlines()
    assert len(lines) == 2 and " register --root " in lines[0] and " preflight --root " in lines[1]
    project = Path(env["SRF6_ROOT"]).parent
    assert (
        project / "sr_f6_operations_20261027/CONFIRM_V3_EXECUTION_20261028/cpu_cache_1234/tmp"
    ).is_dir()
    assert not (Path(env["SRF6_ROOT"]) / "cpu_cache_1234").exists()


@pytest.mark.parametrize("mutation", ["partial_root", "wrong_revision", "wrong_prefix", "seal"])
def test_cpu_prepare_rejects_unsafe_or_duplicate_execution(cpu_launcher, mutation):
    script, env, calls = cpu_launcher
    if mutation == "partial_root":
        Path(env["SRF6_ROOT"]).mkdir()
    elif mutation == "wrong_revision":
        env["SRF6_REVISION"] = "0" * 40
    elif mutation == "wrong_prefix":
        env["SLURM_JOB_NAME"] = "OtherProject"
    else:
        env["SRF6_ACTION"] = "seal"
    process = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    assert process.returncode != 0
    assert not calls.exists()


def test_actual_isolated_production_import_chain_requires_inherited_assets(tmp_path):
    repository = Path(__file__).resolve().parents[2]
    deployment = tmp_path / "isolated"
    shutil.copytree(
        repository / "src", deployment / "src", ignore=shutil.ignore_patterns("__pycache__")
    )
    for relative in registration.old.DEPLOYMENT_ASSETS[3:]:
        target = deployment / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / relative, target)
    result = registration.check_deployment_imports(deployment)
    assert result["status"] == "PASS" and result["model_calls"] == 0
    (deployment / "docs/sr_f1/package/PACKAGE_SHA256.json").unlink()
    with pytest.raises(subprocess.CalledProcessError):
        registration.check_deployment_imports(deployment)


@pytest.fixture
def registered(project, inputs, hashes, monkeypatch):
    """Real registration files, explicitly simulated external provenance only."""
    from sr_f6 import confirm_data, contract, runtime, source_binding

    origin, deployment = project / "original", project / "sr_f6_code_abcdef1"
    origin.mkdir()
    deployment.mkdir()
    (deployment / "actual.py").write_text("# synthetic test source only\n")
    previous = project / "previous"
    previous.mkdir()
    (project / "sr_f4_c_local_pilot_20261024").mkdir()
    (project / "sr_f4_consolidated_20261022").mkdir()
    binding = dict(
        previous_run=str(previous),
        parents={p: dict(content_sha256=hashes[p]) for p in registration.PARENTS},
    )
    for relative, data in (
        ("CONFIG.json", {"unit_test_original_config": True}),
        ("IDENTITY.json", {"unit_test_identity": "origin"}),
        ("PHASES/PILOT_V1/IDENTITY.json", {"unit_test_identity": "pilot"}),
        ("SOURCE_BINDING.json", binding),
        ("DATA/CPU_AUDIT.json", {"unit_test": True}),
        ("DATA/ROOT_CERTIFICATES_PRIVATE.jsonl", {"unit_test": True}),
    ):
        write_once(origin / relative, data)
    monkeypatch.setattr(
        registration.old, "ORIGIN_IDENTITY_SHA256", file_hash(origin / "IDENTITY.json")
    )
    monkeypatch.setattr(
        registration.old,
        "PILOT_IDENTITY_SHA256",
        file_hash(origin / "PHASES/PILOT_V1/IDENTITY.json"),
    )
    monkeypatch.setattr(
        registration.old,
        "SOURCE_REFERENCES",
        ("CONFIG.json", "IDENTITY.json", "SOURCE_BINDING.json"),
    )
    monkeypatch.setattr(registration, "_actual_deployment", lambda: deployment)
    monkeypatch.setattr(registration, "_source_paths", lambda _: ["actual.py"])
    monkeypatch.setattr(registration, "_frozen_implementations", lambda *args: {"unit_test": True})
    monkeypatch.setattr(registration.old, "validate_deployment_assets", lambda _: {})
    monkeypatch.setattr(registration, "check_deployment_imports", lambda _: {"status": "PASS"})
    monkeypatch.setattr(registration.old, "_validate_producer", lambda *args: None)
    monkeypatch.setattr(registration, "_screen_basis", lambda: {"synthetic_test_only": True})
    monkeypatch.setattr(registration, "_verify_screen_basis", lambda _: None)
    monkeypatch.setattr(contract, "validate_config", lambda _: None)
    monkeypatch.setattr(runtime, "_verify_module_origins", lambda *args: None)
    calls = []

    def binding_inputs(root, config, parent=None):
        calls.append(parent)
        assert root == origin
        return binding, {}

    monkeypatch.setattr(runtime, "verify_binding_inputs", binding_inputs)
    monkeypatch.setattr(
        registration,
        "inventory_file",
        lambda path: source_binding.inventory_file(path, boundary=project),
    )
    monkeypatch.setattr(
        registration,
        "verify_bound_files",
        lambda files: source_binding.verify_bound_files(files, boundary=project),
    )
    public, private = inputs

    def build(root, *, historical_roots):
        assert historical_roots == registration.historical_roots(binding)
        write_once(root / "DATA/MANIFEST.json", {"synthetic_test_only": True})
        return {"status": "TEST_ONLY_NOT_REAL_ACCEPTANCE"}

    monkeypatch.setattr(confirm_data, "build", build)
    monkeypatch.setattr(
        confirm_data, "load_inputs", lambda root: copy.deepcopy((list(reversed(public)), private))
    )
    monkeypatch.setattr(confirm_data, "verify_data", lambda root: {"synthetic_test_only": True})
    root = project / "confirm"
    record = registration.register(root, "abcdef1" + "0" * 33)
    return root, record, calls


def test_independent_registration_reconstructs_exact_plans_and_never_old_authorization(registered):
    root, record, calls = registered
    context = registration.verify_registered(root)
    assert context["root"] == context["input_root"] == context["registration_root"] == root
    assert context["source_root"] == context["origin_root"] == registration.ORIGIN_ROOT
    assert context["plan"] == context["spec"]
    assert context["public"] == sorted(context["public"], key=lambda row: row["qid"])
    for worker in context["plan"]["workers"]:
        assert worker["qids"] == [row["qid"] for row in context["public"][worker["shard"] :: 2]]
    assert context["spec_object_sha256"] == digest(context["spec"])
    assert context["spec_sha256"] == file_hash(root / "CONFIRM_SPEC_FROZEN.json")
    assert context["registration_sha256"] == file_hash(root / "REGISTRATION.json")
    assert context["spec_sha256"] != context["registration_sha256"]
    assert record["old_census_raw_status"] == "DELETED_BY_USER_NOT_REUSED"
    assert (root / "CONFIG.json").read_bytes() == (
        registration.ORIGIN_ROOT / "CONFIG.json"
    ).read_bytes()
    assert calls[-4:] == [None, "b0", "b1", "b2"]
    with pytest.raises(PermissionError, match="Evidence is missing"):
        registration.verify_registered(root, require_preflight=True)


@pytest.mark.parametrize("mutation", ["seed", "shuffle", "first", "gate", "overlay"])
def test_changed_frozen_confirm_spec_cannot_be_referenced_into_acceptance(registered, mutation):
    root, record, _ = registered
    path = root / "CONFIRM_SPEC_FROZEN.json"
    spec = json.loads(path.read_text())
    if mutation == "seed":
        spec["attempts"][0]["seed"] += 1
    elif mutation == "shuffle":
        spec["groups"][0]["shuffle_seed"] += 1
    elif mutation == "first":
        spec["workers"][0]["first_qid"] = spec["workers"][0]["qids"][1]
    elif mutation == "gate":
        spec["gates"]["each_count_at_least"]["V"] = 1382
    else:
        spec["prompt_transition"]["structured_splits"].remove("TEST_COMPOSITION")
    path.write_text(json.dumps(spec))
    record["spec"]["sha256"] = file_hash(path)
    (root / "REGISTRATION.json").write_text(json.dumps(record))
    with pytest.raises(PermissionError, match="seeds, partitions, or conditions"):
        registration.verify_registered(root)


def test_new_source_inventory_change_is_rejected(registered):
    root, record, _ = registered
    path = Path(record["deployment"]) / "actual.py"
    path.write_text("# changed unregistered source\n")
    with pytest.raises(PermissionError, match="Bound file"):
        registration.verify_registered(root)


def test_seal_requires_native_and_new_visual_evidence_without_generating_them(visual, monkeypatch):
    root, judgment, processor, processor_ref = visual
    context = dict(
        root=root,
        registration_sha256=processor["registration_sha256"],
        registration={"source_manifest_sha256": "a" * 64},
        public=[],
    )
    monkeypatch.setattr(registration, "verify_registered", lambda *args, **kwargs: context)
    monkeypatch.setattr(registration, "_validate_processor", lambda *args: None)
    monkeypatch.setattr(registration, "verify_bound_files", lambda _: {})
    called = []
    monkeypatch.setattr(registration, "_validate_native", lambda *args: called.append(args))
    write_once(root / "PREFLIGHT/VISUAL_REVIEW.json", judgment)
    ref = registration._ref(root, "PREFLIGHT/VISUAL_REVIEW.json")
    result = registration.seal_preflight(root, ref)
    assert called and result["processor"] == processor_ref and result["visual"] == ref
    assert result["generation_allowed"] is True
    assert result["training_allowed"] is result["evaluation_allowed"] is False
    assert result["scope"] == "REGISTERED_9216_CONFIRMATION_ONLY"
    with pytest.raises(PermissionError, match="Preserve"):
        registration.seal_preflight(root, ref)
