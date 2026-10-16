"""CPU fixtures only: fixed ablation coverage, source trust and one-shot selection."""

from __future__ import annotations

import copy
import getpass
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from mm_core.vl_runtime import hash_json
from sr_f3_ce import ablation as a
from sr_f3_ce import runner
from sr_f3_ce.evaluation import engine_slots
from sr_f3_ce.orchestration import TEACHER_QOS
from sr_f3_ce.prompts import prefill_for, prompt_for_revision
from sr_f3_ce.protocol import PARENTS, digest, file_hash, read_json
from sr_f3_ce.runtime import decoded_protocol

REPO = Path(__file__).resolve().parents[2]
PLAN = REPO / "artifacts/sr_f3_ce_20261016/plan"
REVISION = REPO / "artifacts/sr_f3_ce_protocol_fix_20261016"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    return path


@pytest.fixture(scope="module")
def tasks():
    rows = map(json.loads, (PLAN / "manifests/TASKS_AUDIT_ONLY.jsonl").read_text().splitlines())
    return {row["qid"]: row for row in rows if row["pool"] == "ENGINE"}


def fake_registration(root, tasks):
    """Real slots and prompt transformation; fake images/context hashes, no model."""
    slots = a.slots_for(tasks)
    requests, candidates = {}, {}
    for variant, revision in a.VARIANTS.items():
        requests[variant] = {
            qid: dict(
                prompt=prompt_for_revision(task["prompt"], revision),
                image_file=qid + ".png",
                image_sha256=digest(["fixture-image", qid]),
            )
            for qid, task in tasks.items()
        }
        candidates[variant] = dict(
            protocol_revision=revision,
            assistant_prefill=prefill_for("structured", revision),
            package_sha256=digest(["package", revision]),
            config_sha256=digest(["config", revision]),
            model_inputs_sha256=digest(["inputs", revision]),
            native_contexts={
                qid: dict(
                    prompt_tensor_hash=digest([qid, revision]),
                    assistant_prefill_token_ids=[1] if revision != "c2" else [1, 2],
                )
                for qid in tasks
            },
        )
    return tasks, slots, requests, candidates


@pytest.fixture
def case(tmp_path, monkeypatch, tasks):
    root, previous = tmp_path / "ablation", tmp_path / "stopped"
    root.mkdir()
    previous.mkdir()
    for name in a._bound_names():
        write(root / name, {"fixture": name})
    source_files = []
    for name in a.REQUIRED_SOURCE_FILES:
        path = root / "code" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, path)
        source_files.append(dict(path=name, sha256=file_hash(path), bytes=path.stat().st_size))
    write(root / "deployment/SOURCE_MANIFEST.json", dict(files=source_files, code_commit="a" * 40))
    request = root / "authorization/REQUEST.md"
    request.parent.mkdir()
    shutil.copyfile(
        REVISION / "authorization/CLAUDE_SR_F3_CE_PROTOCOL_FIX_FOR_CODEX_zh.md", request
    )
    identity = dict(
        run_stage="PROTOCOL_ABLATION",
        code_commit="a" * 40,
        bound_files={
            "authorization/REQUEST.md": file_hash(request),
            "deployment/SOURCE_MANIFEST.json": file_hash(root / "deployment/SOURCE_MANIFEST.json"),
        },
    )
    write(root / "RUN_IDENTITY.json", identity)
    prior = dict(
        model_weights_hash=digest("base"),
        processor_hash=digest("processor"),
        tokenizer_hash=digest("tokenizer"),
        chat_template_hash=digest("chat"),
    )
    write(root / "prior/MODEL_ENVIRONMENT_IDENTITY.json", prior)
    parents = [
        dict(
            parent_id=parent,
            adapter_path=f"parents/b{block}",
            policy_hash=digest(["policy", block]),
        )
        for block, parent in enumerate(PARENTS)
    ]
    for parent in parents:
        adapter_path = root / parent["adapter_path"]
        adapter_path.mkdir()
        write(adapter_path / "adapter_config.json", dict(fixture="config"))
        (adapter_path / "README.md").write_text("Fixture parent model card")
        (adapter_path / "adapter_model.safetensors").write_bytes(b"fixture-only-no-weights")
        write(
            adapter_path / "IDENTITY.json",
            dict(
                policy_hash=parent["policy_hash"],
                files={
                    name: file_hash(adapter_path / name)
                    for name in ("README.md", "adapter_config.json", "adapter_model.safetensors")
                },
            ),
        )
        parent["copied_files"] = [
            a._record(root, adapter_path / name)
            for name in (
                "README.md",
                "adapter_config.json",
                "adapter_model.safetensors",
                "IDENTITY.json",
            )
        ]
    write(root / "parents/PARENT_AUDIT.json", dict(status="PASS", parents=parents))
    for name in (
        "STOP",
        "RUN_IDENTITY.json",
        "deployment/SOURCE_MANIFEST.json",
        "STOP_DECISION.json",
    ):
        write(previous / name, dict(previous=name))
    monkeypatch.setattr(
        a, "verify_run", lambda path: (read_json(Path(path) / "RUN_IDENTITY.json"), {})
    )
    registration = fake_registration(root, tasks)
    monkeypatch.setattr(a, "_registration", lambda path: copy.deepcopy(registration))
    monkeypatch.setattr(
        runner, "dataset_and_store", lambda path: (SimpleNamespace(tasks=tasks), None)
    )
    manifest = a.freeze_ablation(
        root,
        previous_root=previous,
        instruction="authorization/REQUEST.md",
        stop_decision="STOP_DECISION.json",
    )
    return SimpleNamespace(
        root=root,
        previous=previous,
        tasks=tasks,
        manifest=manifest,
        prior=prior,
        parents=parents,
        registration=registration,
    )


class Slurm:
    def __init__(self):
        self.fields, self.accounting = {}, {}
        self.show_calls, self.terminal_calls = [], []

    def add(self, task, job):
        user = getpass.getuser()
        self.fields[job] = dict(
            JobId=job,
            JobName=task["job_name"],
            Account="rose",
            Partition="cluster02",
            QOS=TEACHER_QOS,
            JobState="RUNNING",
            Requeue="0",
            UserId=user + "(1000)",
            ReqTRES="cpu=4,gres/gpu=1,gres/gpu:pro6000=1",
            AllocTRES="cpu=4,gres/gpu=1,gres/gpu:pro6000=1",
        )
        self.accounting[job] = dict(
            job_id=job,
            state="COMPLETED",
            exit_code="0:0",
            user=user,
            account="rose",
            qos=TEACHER_QOS,
            name=task["job_name"],
        )

    def show(self, job):
        self.show_calls.append(job)
        return self.fields[job]

    def terminal(self, job):
        self.terminal_calls.append(job)
        return self.accounting[job]


class Runtime:
    """CPU stub deliberately exposes no optimizer, step or teacher-forward method."""

    def __init__(self, case, block, variant, *, fail_after=None):
        self.case, self.block, self.variant = case, block, variant
        self.revision = a.VARIANTS[variant]
        self.identity = dict(protocol_revision=self.revision)
        self.calls, self.fail_after = [], fail_after
        self.by_request = {
            digest(row): qid for qid, row in case.manifest["requests"][variant].items()
        }

    def stable_model_identity(self):
        return dict(
            parent_id=PARENTS[self.block],
            protocol_revision=self.revision,
            base_model_weights_hash=self.case.prior["model_weights_hash"],
            **{
                key: self.case.prior[key]
                for key in ("processor_hash", "tokenizer_hash", "chat_template_hash")
            },
        )

    def current_adapter_identity(self):
        parent = self.case.parents[self.block]
        return dict(
            base_model_weights_hash=self.case.prior["model_weights_hash"],
            adapter_parameter_hash=parent["policy_hash"],
            adapter_path=str(self.case.root / parent["adapter_path"]),
        )

    def generate_group(self, request, image_root, seeds, *, protocol, generation, on_completion):
        assert set(request) == {"prompt", "image_file", "image_sha256"}
        assert image_root == self.case.root / "data"
        assert protocol == "structured" and generation["num_return_sequences"] == 4
        self.calls.append(
            dict(request=copy.deepcopy(request), seeds=list(seeds), generation=generation)
        )
        qid = self.by_request[digest(request)]
        task = self.case.tasks[qid]
        for index, seed in enumerate(seeds):
            if index == self.fail_after:
                raise RuntimeError("CPU fixture interruption")
            # P1 exactly 192/256 for b0/b2. P2/P3 have deliberately wrong answers
            # but remain parseable, so answer quality cannot influence selection.
            valid = not (self.variant == "P1" and self.block in (0, 2) and index == 0)
            raw = (
                json.dumps(dict(evidence=[], answer=999999), indent=2)
                if valid
                else '{"series":"wrong"}'
            )
            prefill = prefill_for("structured", self.revision)
            assert raw.startswith(prefill)
            text = raw[len(prefill) :]
            context = self.case.manifest["candidates"][self.variant]["native_contexts"][qid]
            parameters = dict(generation, num_return_sequences=1)
            row = dict(
                model_input=copy.deepcopy(request),
                prompt=request["prompt"],
                protocol=protocol,
                protocol_revision=self.revision,
                seed=seed,
                group_seed=seeds[0],
                group_row_index=index,
                group_size=4,
                tokens=[10],
                raw_tokens=[10],
                backend_sequence_tokens=[10],
                backend_padding_count=0,
                sampler_logprobs=[-0.25],
                completion_token_count=1,
                prompt_tensor_hash=context["prompt_tensor_hash"],
                input_routing=dict(
                    assistant_prefill_token_ids=context["assistant_prefill_token_ids"]
                ),
                model_identity=self.stable_model_identity(),
                adapter_identity=self.current_adapter_identity(),
                generation_status="COMPLETE",
                technical_validation_errors=[],
                finish_reason="balanced",
                truncated=False,
                boundary_token_count=1,
                balanced_token_count=1,
                max_new_tokens=1024,
                token_prefix_character_ends=[len(raw)],
                sampling_parameters=parameters,
                sampling_hash=hash_json(parameters),
                **decoded_protocol(text, protocol, revision=self.revision),
            )
            assert task["pool"] == "ENGINE"
            on_completion(row)


def complete_all(case, monkeypatch):
    slurm, runtimes = Slurm(), []
    for i, (registered, task) in enumerate(a.task_matrix().items(), 100):
        job = str(i)
        slurm.add(task, job)
        monkeypatch.setenv("SLURM_JOB_ID", job)
        runtime = Runtime(case, task["block"], task["variant"])
        runtimes.append(runtime)

        def load(
            root,
            parent_id,
            task_id,
            *,
            protocol_revision,
            training,
            task=task,
            registered=registered,
            runtime=runtime,
        ):
            assert root == case.root and parent_id == task["parent_id"] and task_id == registered
            assert protocol_revision == task["protocol_revision"] and training is False
            return runtime

        monkeypatch.setattr(runner, "load_parent_runtime", load)
        receipt = a.execute_parent_variant(case.root, task["block"], task["variant"], slurm=slurm)
        assert receipt["records"] == 256
        assert len(receipt["raw_groups"]) == 64 and len(receipt["raw_artifacts"]) == 128
    assert sum(len(runtime.calls) for runtime in runtimes) == 576
    return slurm


def test_fixed_slots_preserve_effective_original_seeds_and_prefixes(tasks):
    original = {
        s["slot_id"]: s
        for s in engine_slots(tasks)
        if s["protocol"] == "structured" and s["view"] is None
    }
    slots = a.slots_for(tasks)
    assert len(slots) == len({s["slot_id"] for s in slots}) == 2304
    assert len(a.groups_for(slots)) == 576
    for slot in slots:
        old = original[slot["original_slot_id"]]
        assert (slot["sampling_seed"], slot["group_seed"], slot["generation"]) == (
            old["sampling_seed"],
            old["group_seed"],
            old["generation"],
        )
        assert slot["assistant_prefill"] == prefill_for("structured", slot["protocol_revision"])
    assert all(task["job_name"].endswith("-LH") for task in a.task_matrix().values())


@pytest.mark.parametrize(
    "p1,p2,p3,status,revision",
    [
        ((230, 256, 230), (231, 231, 231), (256, 256, 256), "PROTOCOL_SELECTED", "v2"),
        ((230, 0, 230), (231, 230, 256), (231, 231, 231), "PROTOCOL_SELECTED", "c2"),
        ((230, 256, 230), (230, 256, 256), (256, 230, 256), "STOP_PROTOCOL_FAILURE", None),
        ((231, 0, 230), (256, 256, 256), (256, 256, 256), "STACK_REPRODUCTION_FAILURE", None),
        ((230, 256, 231), (256, 256, 256), (256, 256, 256), "STACK_REPRODUCTION_FAILURE", None),
    ],
)
def test_exact_threshold_and_priority(p1, p2, p3, status, revision):
    counts = {
        a.task_id(block, variant): rows[block]
        for variant, rows in zip(a.VARIANTS, (p1, p2, p3), strict=True)
        for block in range(3)
    }
    assert a.select_counts(counts) == dict(status=status, selected_revision=revision)


@pytest.mark.parametrize("bad", [True, 257, -1, 230.0])
def test_no_unregistered_or_partial_count_decision(bad):
    counts = dict.fromkeys(a.task_matrix(), 230)
    counts["ablation--P2--b0"] = bad
    with pytest.raises(ValueError, match="Exactly nine"):
        a.select_counts(counts)
    counts.pop("ablation--P2--b0")
    with pytest.raises(ValueError, match="Exactly nine"):
        a.select_counts(counts)


def test_registration_authenticates_real_candidate_packages_and_exact_requests(
    tmp_path, monkeypatch, tasks
):
    from sr_f3_ce import native_audit

    (tmp_path / "plan").symlink_to(PLAN, target_is_directory=True)
    for revision in ("v2", "c2"):
        target = tmp_path / "candidates" / revision
        target.mkdir(parents=True)
        (target / "plan").symlink_to(
            REVISION / "candidates" / revision / "plan", target_is_directory=True
        )

    class Store:
        def model_input(self, qid, *, protocol):
            assert protocol == "structured"
            return dict(
                prompt=tasks[qid]["prompt"],
                image_file=qid + ".png",
                image_sha256=digest(["fixture-image", qid]),
            )

    def verified_context(root, revision):
        assert root == tmp_path
        return dict(
            status="PASS",
            protocol_revision=revision,
            unique_contexts=64,
            parent_panel_contexts=192,
            model_calls=0,
            weights_loaded=False,
            teacher_forward_calls=0,
            assistant_prefill=prefill_for("structured", revision),
            contexts={
                qid: dict(
                    model_input_sha256=digest(
                        dict(
                            Store().model_input(qid, protocol="structured"),
                            prompt=prompt_for_revision(tasks[qid]["prompt"], revision),
                        )
                    ),
                    prompt_tensor_hash=digest([qid, revision]),
                    routing=dict(assistant_prefill_token_ids=[1]),
                )
                for qid in tasks
            },
        )

    monkeypatch.setattr(
        runner, "dataset_and_store", lambda root: (SimpleNamespace(tasks=tasks), Store())
    )
    monkeypatch.setattr(native_audit, "verify_protocol_contexts", verified_context)
    found_tasks, slots, requests, candidates = a._registration(tmp_path)
    assert found_tasks == tasks and len(slots) == 2304
    assert requests["P1"] == requests["P3"] and requests["P1"] != requests["P2"]
    for variant, revision in a.VARIANTS.items():
        plan = (
            tmp_path / "plan" if revision == "v1" else tmp_path / "candidates" / revision / "plan"
        )
        assert candidates[variant]["package_sha256"] == file_hash(plan / "PACKAGE_SHA256.json")
    original = verified_context

    def wrong_context(root, revision):
        value = original(root, revision)
        value["contexts"][next(iter(tasks))]["model_input_sha256"] = digest("wrong prompt")
        return value

    monkeypatch.setattr(native_audit, "verify_protocol_contexts", wrong_context)
    with pytest.raises(PermissionError, match="CPU accepted context"):
        a._registration(tmp_path)


def test_manifest_is_prospective_and_old_stop_is_immutable(case):
    assert a.verify_manifest(case.root) == case.manifest
    assert a.runtime_revision_for_task(case.root, "ablation--P3--b0") == "c2"
    with pytest.raises(PermissionError, match="registered"):
        a.runtime_revision_for_task(case.root, "engine--b0")
    before = file_hash(case.previous / "STOP")
    (case.root / "ablation/raw").mkdir(parents=True)
    with pytest.raises(PermissionError, match="precede"):
        a.freeze_ablation(
            case.root,
            previous_root=case.previous,
            instruction="authorization/REQUEST.md",
            stop_decision="STOP_DECISION.json",
        )
    assert file_hash(case.previous / "STOP") == before


@pytest.mark.parametrize("target", ["previous", "candidate", "request", "stage"])
def test_manifest_refuses_changed_trust_roots(case, target):
    if target == "previous":
        (case.previous / "STOP").write_text("changed")
    elif target == "candidate":
        (case.root / "candidates/c2/plan/PROTOCOL_REVISION.json").write_text("changed")
    elif target == "request":
        (case.root / "authorization/REQUEST.md").write_text("different instruction")
    else:
        identity = read_json(case.root / "RUN_IDENTITY.json")
        write(
            case.root / "RUN_IDENTITY.json", dict(identity, run_stage="SELECTED_PROTOCOL_REVISION")
        )
    with pytest.raises(PermissionError):
        a.verify_manifest(case.root)


@pytest.mark.parametrize(
    "attack", ["missing_entry", "extra_python", "bytecode", "changed_loaded_module"]
)
def test_source_inventory_cannot_omit_or_replace_executed_source(case, attack):
    source_path = case.root / "deployment/SOURCE_MANIFEST.json"
    source = read_json(source_path)
    if attack == "missing_entry":
        source["files"] = source["files"][1:]
        write(source_path, source)
    elif attack == "extra_python":
        write(case.root / "code/src/unmanifested.py", {"not": "bound"})
    elif attack == "bytecode":
        (case.root / "code/unchecked.pyc").write_bytes(b"unchecked bytecode")
    else:
        (case.root / "code/src/sr_f3_ce/ablation.py").write_text("wrong implementation")
    with pytest.raises(PermissionError):
        a.verify_manifest(case.root)


@pytest.mark.parametrize(
    "field,value",
    [
        ("JobName", "SRF3CE-ABLATION-P1-b0"),
        ("QOS", "override"),
        ("Account", "other"),
        ("JobState", "PENDING"),
        ("ReqTRES", "gres/gpu=2,gres/gpu:pro6000=2"),
        ("Requeue", "1"),
        ("AllocTRES", "gres/gpu=0"),
        ("UserId", "another(1000)"),
    ],
)
def test_real_allocation_gate_before_any_model_load(case, monkeypatch, field, value):
    task, slurm = a.task_matrix()["ablation--P1--b0"], Slurm()
    slurm.add(task, "100")
    slurm.fields["100"][field] = value
    monkeypatch.setenv("SLURM_JOB_ID", "100")
    calls = []
    monkeypatch.setattr(runner, "load_parent_runtime", lambda *x, **k: calls.append(1))
    with pytest.raises(PermissionError, match="allocation"):
        a.execute_parent_variant(case.root, 0, "P1", slurm=slurm)
    assert not calls
    assert len(list((case.root / "technical_incidents").glob("*.json"))) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("code_commit", "b" * 40),
        ("code_commit", "unresolved-ref"),
        ("bound_files", {"authorization/REQUEST.md": "0" * 64}),
    ],
)
def test_source_commit_and_manifest_binding_must_agree(case, field, value):
    identity = read_json(case.root / "RUN_IDENTITY.json")
    identity[field] = value
    write(case.root / "RUN_IDENTITY.json", identity)
    with pytest.raises(PermissionError, match="source commit"):
        a.verify_manifest(case.root)


def test_group_keeps_full_raw_and_cached_group_never_resamples(case):
    group = a.groups_for(case.manifest["slots"])[0]
    runtime = Runtime(case, 0, "P1")
    directory = case.root / "ablation/raw/ablation--P1--b0"
    raw = a._group(case.root, case.manifest, group, runtime, case.tasks, directory)
    assert a._group(case.root, case.manifest, group, runtime, case.tasks, directory) == raw
    assert len(runtime.calls) == 1
    saved = read_json(case.root / raw["path"])
    assert len(saved["records"]) == 4
    assert saved["records"][0]["raw_text"] == '{"series":"wrong"}'
    assert saved["records"][0]["raw_tokens"] == [10]
    assert saved["records"][0]["task_sidecar_sha256"] == digest(
        case.tasks[group["slots"][0]["qid"]]
    )


def test_partial_stream_never_resamples_and_failure_is_not_parseability(case):
    group = a.groups_for(case.manifest["slots"])[0]
    broken, clean = Runtime(case, 0, "P1", fail_after=2), Runtime(case, 0, "P1")
    directory = case.root / "ablation/raw/ablation--P1--b0"
    with pytest.raises(RuntimeError, match="interruption"):
        a._group(case.root, case.manifest, group, broken, case.tasks, directory)
    with pytest.raises(RuntimeError, match="no automatic resampling"):
        a._group(case.root, case.manifest, group, clean, case.tasks, directory)
    assert not clean.calls
    pending = directory / "groups" / (group["group_id"] + ".pending.jsonl")
    assert len(pending.read_text().splitlines()) == 2
    with pytest.raises((PermissionError, FileNotFoundError)):
        a.select_protocol(case.root, slurm=Slurm())
    assert not (case.root / "PROTOCOL_SELECTION.json").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("runtime_protocol_revision", "v1"),
        ("runtime_assistant_prefill", "{"),
        ("prompt", "wrong"),
        ("group_seed", 3),
        ("source_sha256", "wrong"),
        ("task_sidecar_sha256", "wrong"),
        ("prompt_tensor_hash", "wrong"),
        ("backend_sequence_tokens", [11]),
        ("sampling_hash", "wrong"),
        ("boundary_token_count", None),
    ],
)
def test_native_identity_cannot_be_masked_by_slot_metadata(case, field, value):
    group = next(
        g for g in a.groups_for(case.manifest["slots"]) if g["slots"][0]["ablation_variant"] == "P3"
    )
    runtime = Runtime(case, 0, "P3")
    directory = case.root / "ablation/raw/ablation--P3--b0"
    raw = a._group(case.root, case.manifest, group, runtime, case.tasks, directory)
    row = read_json(case.root / raw["path"])["records"][0]
    row[field] = value
    slot = group["slots"][0]
    with pytest.raises((PermissionError, ValueError)):
        a._validate(
            row,
            slot,
            case.manifest["requests"]["P3"][slot["qid"]],
            case.tasks[slot["qid"]],
            runtime.stable_model_identity(),
            runtime.current_adapter_identity(),
            file_hash(case.root / "ABLATION_MANIFEST.json"),
            0,
            case.manifest["candidates"]["P3"]["native_contexts"][slot["qid"]],
        )


def test_complete_2304_only_parseability_selects_and_readonly_reverification(case, monkeypatch):
    slurm = complete_all(case, monkeypatch)
    selected = a.select_protocol(case.root, slurm=slurm)
    assert selected["status"] == "PROTOCOL_SELECTED" and selected["selected_revision"] == "v2"
    assert selected["outputs"] == 2304 and set(selected["denominators"].values()) == {256}
    assert selected["counts"]["ablation--P1--b0"] == 192
    assert selected["counts"]["ablation--P1--b2"] == 192
    assert selected["counts"]["ablation--P2--b0"] == 256
    assert selected["descriptive_statistics"]["ablation--P2--b0"]["metrics"]["value_correct"] == 0
    assert len(slurm.terminal_calls) == 9
    assert a.verify_selection(case.root) == selected
    assert a.select_protocol(case.root, slurm=object()) == selected
    assert len(slurm.terminal_calls) == 9
    assert selected["scientific_execution_authorized"] is False
    assert len(selected["raw_evidence"]) == 9
    with pytest.raises(PermissionError, match="receipt-only"):
        a.execute_parent_variant(case.root, 2, "P3", slurm=slurm)


@pytest.mark.parametrize("state", ["FAILED", "TIMEOUT", "UNKNOWN", "COMPLETED_NONZERO"])
def test_receipt_without_successful_terminal_cannot_select(case, monkeypatch, state):
    slurm = complete_all(case, monkeypatch)
    slurm.accounting["100"]["state"] = state if state != "COMPLETED_NONZERO" else "COMPLETED"
    if state == "COMPLETED_NONZERO":
        slurm.accounting["100"]["exit_code"] = "1:0"
    with pytest.raises(PermissionError, match="terminal accounting"):
        a.select_protocol(case.root, slurm=slurm)
    assert not (case.root / "PROTOCOL_SELECTION.json").exists()


def test_self_consistent_wrong_parent_is_rejected_independently_of_completion(case):
    runtime = Runtime(case, 0, "P1")
    wrong = dict(runtime.current_adapter_identity(), adapter_parameter_hash=digest("different"))
    with pytest.raises(PermissionError, match="parent provenance"):
        a._parent_identity(
            case.root, a.task_matrix()["ablation--P1--b0"], runtime.stable_model_identity(), wrong
        )


def test_changed_parent_original_bytes_are_not_trusted_by_declared_policy_hash(case):
    runtime = Runtime(case, 0, "P1")
    (case.root / "parents/b0/adapter_model.safetensors").write_bytes(b"different")
    with pytest.raises(PermissionError, match="artifact bytes changed"):
        a._parent_identity(
            case.root,
            a.task_matrix()["ablation--P1--b0"],
            runtime.stable_model_identity(),
            runtime.current_adapter_identity(),
        )


def test_duplicate_or_missing_raw_originals_block_selection(case, monkeypatch):
    slurm = complete_all(case, monkeypatch)
    group_dir = case.root / "ablation/raw/ablation--P1--b0/groups"
    original = next(group_dir.glob("*.pending.jsonl"))
    original.unlink()
    with pytest.raises(PermissionError, match="raw artifact"):
        a.select_protocol(case.root, slurm=slurm)
    assert not (case.root / "PROTOCOL_SELECTION.json").exists()


def test_selection_tamper_never_chooses_new_threshold_or_revision(case, monkeypatch):
    slurm = complete_all(case, monkeypatch)
    a.select_protocol(case.root, slurm=slurm)
    selected_path = case.root / "PROTOCOL_SELECTION.json"
    value = read_json(selected_path)
    value["selected_revision"] = "c2"
    write(selected_path, value)
    with pytest.raises(PermissionError, match="complete immutable evidence"):
        a.verify_selection(case.root)
