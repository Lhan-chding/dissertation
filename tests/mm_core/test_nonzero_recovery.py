"""CPU regression checks for the new extension; no claim of real CUDA execution."""

import copy
import hashlib
import json

import pytest

from mm_core.execution import atomic_json, sha256_file
from mm_core.nonzero_recovery import (
    ADDITIONAL_CAPS,
    NONZERO_RECIPE,
    PRIOR_COSTS,
    SCOPE,
    SEED,
    NonzeroLedger,
    nonzero_sft_step,
    require_nonzero_allocation,
    verify_nonzero_gate,
)
from mm_core.training import (
    capture_rng,
    deterministic_order,
    load_checkpoint,
    save_checkpoint,
    state_hash,
    trainable_state,
)
from mm_core.vl_runtime import MODEL_ID, seed_all


@pytest.fixture
def frozen_root(tmp_path):
    root = tmp_path / "run"
    project = tmp_path / "project"
    originals = (
        "__init__",
        "allocations",
        "contracts",
        "execution",
        "generator",
        "inventory",
        "release",
        "runner",
        "scoring",
        "training",
        "vl_runtime",
    )
    source_hashes = {}
    old_hashes = {}
    for name in originals:
        path = project / "src/mm_core" / (name + ".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# immutable fixture " + name)
        old_hashes[path.name] = sha256_file(path)
        source_hashes[str(path.relative_to(project))] = sha256_file(path)
    for relative in ("src/mm_core/nonzero_recovery.py", "scripts/mm_core/run_nonzero_recovery.py"):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# separately frozen CPU fixture")
        source_hashes[relative] = sha256_file(path)
    original = dict(
        status="FROZEN",
        model_id=MODEL_ID,
        dev_authorized=False,
        source_hashes=old_hashes,
        model_weights_hash="a" * 64,
    )
    atomic_json(root / "manifests/PRE_INFERENCE_FREEZE.json", original)
    atomic_json(
        root / "manifests/COMMON_START.json",
        dict(status="VERIFIED", untouched_base=True, adapter_path=None, model_hash="a" * 64),
    )
    (root / "data/images").mkdir(parents=True)
    rows = []
    file_hashes = {}
    for index in range(12):
        image_path = f"data/images/{index}.png"
        (root / image_path).write_bytes(f"CPU-image-fixture-{index}".encode())
        prompt = f"Fixed fixture prompt {index}"
        row = dict(
            question_id=f"q-{index:02}",
            split="ENGINE_TEST",
            image_path=image_path,
            image_sha256=sha256_file(root / image_path),
            prompt=prompt,
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            true_values=[11, 22],
            operation="sum",
        )
        rows.append(row)
        file_hashes[image_path] = row["image_sha256"]
    (root / "data/questions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    file_hashes["data/questions.jsonl"] = sha256_file(root / "data/questions.jsonl")
    original["file_hashes"] = {"data/questions.jsonl": file_hashes["data/questions.jsonl"]}
    atomic_json(root / "manifests/PRE_INFERENCE_FREEZE.json", original)
    freeze = dict(
        run_root=str(root.resolve()),
        original_run_root=str((tmp_path / "original_audit").resolve()),
        status="FROZEN",
        scope=SCOPE,
        authorized=True,
        mmdev_authorized=False,
        seed=SEED,
        recipe=copy.deepcopy(NONZERO_RECIPE),
        source_hashes=source_hashes,
        file_hashes=file_hashes,
        prior_costs=PRIOR_COSTS,
        resources=dict(max_concurrent_gpus=5, allocated_gpu_hours="accounting_only"),
        pre_inference_freeze_sha256=sha256_file(root / "manifests/PRE_INFERENCE_FREEZE.json"),
        common_start_sha256=sha256_file(root / "manifests/COMMON_START.json"),
        stream_question_ids=[r["question_id"] for r in deterministic_order(rows, SEED, 8)],
        no_automatic_retry=True,
        common_start_replacement=False,
    )
    atomic_json(root / "manifests/NONZERO_FREEZE.json", freeze)
    return root, project, freeze


def test_gate_accepts_only_frozen_extension_and_outcome_independent_stream(frozen_root):
    root, project, freeze = frozen_root
    result = verify_nonzero_gate(root, source_root=project)
    assert [row["question_id"] for row in result[-1]] == freeze["stream_question_ids"]
    changed = copy.deepcopy(freeze)
    changed["stream_question_ids"] = list(reversed(changed["stream_question_ids"]))
    atomic_json(root / "manifests/NONZERO_FREEZE.json", changed)
    with pytest.raises(PermissionError, match="outcome-independent"):
        verify_nonzero_gate(root, source_root=project)


@pytest.mark.parametrize(
    "key,value",
    [
        ("authorized", False),
        ("mmdev_authorized", True),
        ("no_automatic_retry", False),
        ("common_start_replacement", True),
        ("seed", SEED + 1),
    ],
)
def test_gate_rejects_scope_changes(frozen_root, key, value):
    root, project, freeze = frozen_root
    freeze[key] = value
    atomic_json(root / "manifests/NONZERO_FREEZE.json", freeze)
    with pytest.raises(PermissionError):
        verify_nonzero_gate(root, source_root=project)


def test_gate_rejects_any_old_module_mutation(frozen_root):
    root, project, _ = frozen_root
    (project / "src/mm_core/training.py").write_text("# altered")
    with pytest.raises(PermissionError, match="implementation changed"):
        verify_nonzero_gate(root, source_root=project)


def test_allocation_binding_no_restart_and_no_alias(frozen_root):
    root, _, _ = frozen_root
    allocation = dict(
        status="BOUND",
        scope=SCOPE,
        run_root=str(root.resolve()),
        freeze_sha256=sha256_file(root / "manifests/NONZERO_FREEZE.json"),
        slurm_job_id="12345",
        allocated_gpus=1,
        gpu_model_contains="PRO 6000",
        job_name="mmcore-nonzero-sft-qwen35",
    )
    atomic_json(root / "manifests/NONZERO_ALLOCATION.json", allocation)
    import os

    live = (
        f"JobId=12345 JobName=mmcore-nonzero-sft-qwen35 JobState=RUNNING "
        f"UserId=fixture({os.getuid()}) Requeue=0 Restarts=0 NumNodes=1 "
        "AllocTRES=node=1,gres/gpu=1,gres/gpu:pro6000=1 TresPerNode=gres/gpu:pro6000:1"
    )

    def query(job):
        return live

    assert (
        require_nonzero_allocation(root, environ={"SLURM_JOB_ID": "12345"}, scheduler_query=query)[
            "allocation"
        ]
        == allocation
    )
    for env in (
        {},
        {"SLURM_JOB_ID": "54321"},
        {"SLURM_JOB_ID": "12345", "SLURM_RESTART_COUNT": "1"},
    ):
        with pytest.raises(PermissionError):
            require_nonzero_allocation(root, environ=env, scheduler_query=query)
    with pytest.raises(PermissionError, match="GPU count"):
        require_nonzero_allocation(
            root,
            environ={"SLURM_JOB_ID": "12345"},
            scheduler_query=lambda job: live.replace("gres/gpu=1", "gres/gpu=2"),
        )


def test_cost_reservations_are_nonrefundable_and_finite(tmp_path):
    ledger = NonzeroLedger(tmp_path)
    for kind, count in ADDITIONAL_CAPS.items():
        ledger.reserve(kind, count, {"fixture": True})
        with pytest.raises(RuntimeError, match="BUDGET_EXHAUSTED"):
            ledger.reserve(kind, 1, {"repeat": True})
    ledger.reserve("allocated_gpu_hours", 100, {"accounting_only": True})
    assert ledger.totals()["allocated_gpu_hours"] == 100
    for bad in (True, 0, -1, float("nan"), 0.5):
        with pytest.raises(ValueError):
            ledger.reserve("completion_attempts", bad, {})


class TinyRuntime:
    def __init__(self, torch, zero=False):
        self.torch = torch
        self.model = torch.nn.Linear(2, 3)
        self.costs = []
        self.zero = zero

    def reserve(self, kind, count, **meta):
        self.costs.append((kind, count, meta))

    def prepare(self, row, root):
        return dict(
            inputs=self.torch.tensor([[float(row["fixture_input"]), 0.7]]),
            routing={"fixture": "synthetic_CPU_no_CUDA_claim"},
        )

    def encode_completion(self, text):
        return [0, 1]

    def sequence_forward(self, prepared, tokens, *, purpose, grad):
        self.reserve("extra_forward_sequences", 1, purpose=purpose)
        logits = self.model(prepared["inputs"])[0].log_softmax(-1)
        return dict(logprobs=logits[tokens] * (0 if self.zero else 1), vision_forward_calls=1)


def test_positive_updates_and_full_checkpoint_recovery_are_exact(tmp_path):
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    import random

    rows = [
        dict(question_id=f"q{i}", fixture_input=i + 1, true_values=[11, 22], operation="sum")
        for i in range(8)
    ]

    def setup():
        seed_all(SEED)
        runtime = TinyRuntime(torch)
        optimizer = torch.optim.AdamW(runtime.model.parameters(), lr=1e-5, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
        return runtime, optimizer, scheduler

    def advance(runtime, optimizer, scheduler, start, stop):
        metrics = []
        for step in range(start, stop):
            metrics.append(
                nonzero_sft_step(
                    runtime,
                    optimizer,
                    scheduler,
                    rows[step * 2 : step * 2 + 2],
                    tmp_path,
                    branch="CPU",
                    logical_step=step + 1,
                )
            )
            # Exercise all non-CUDA RNG restoration paths; not a synthetic GPU reward.
            random.random()
            np.random.rand()
            torch.rand(3)
        return metrics

    runtime, optimizer, scheduler = setup()
    reference = trainable_state(runtime.model)
    continuous = advance(runtime, optimizer, scheduler, 0, 4)
    final = save_checkpoint(
        tmp_path / "continuous.pt", runtime, optimizer, scheduler, 4, reference, stream_hash="fixed"
    )
    runtime, optimizer, scheduler = setup()
    first = advance(runtime, optimizer, scheduler, 0, 2)
    saved = save_checkpoint(
        tmp_path / "checkpoint.pt", runtime, optimizer, scheduler, 2, reference, stream_hash="fixed"
    )
    runtime, optimizer, scheduler = setup()
    load_checkpoint(tmp_path / "checkpoint.pt", runtime, optimizer, scheduler, "fixed")
    assert state_hash(capture_rng()) == state_hash(saved["rng"])
    second = advance(runtime, optimizer, scheduler, 2, 4)
    resumed = save_checkpoint(
        tmp_path / "resumed.pt", runtime, optimizer, scheduler, 4, reference, stream_hash="fixed"
    )
    assert first + second == continuous
    assert state_hash(resumed) == state_hash(final)
    assert all(
        row["gradient_norm_before_clip"] > 0 and row["parameter_delta_max_abs"] > 0
        for row in continuous
    )


def test_zero_gradient_never_counts_as_nonzero_acceptance(tmp_path):
    torch = pytest.importorskip("torch")
    runtime = TinyRuntime(torch, zero=True)
    optimizer = torch.optim.AdamW(runtime.model.parameters(), lr=1e-5, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    before = state_hash(trainable_state(runtime.model))
    rows = [
        dict(question_id=f"q{i}", fixture_input=i + 1, true_values=[11, 22], operation="sum")
        for i in range(2)
    ]
    with pytest.raises(RuntimeError, match="positive gradient"):
        nonzero_sft_step(
            runtime, optimizer, scheduler, rows, tmp_path, branch="CPU", logical_step=1
        )
    assert state_hash(trainable_state(runtime.model)) == before
    assert runtime.costs[0][0:2] == ("physical_optimizer_updates", 1)


def test_same_wrong_checkpoint_never_passes_binding(tmp_path):
    torch = pytest.importorskip("torch")
    from mm_core.nonzero_recovery import checkpoint_evidence, validate_checkpoint_binding

    with pytest.raises(ValueError, match="step/stream/field"):
        validate_checkpoint_binding({}, {}, 2, "fixed")
    seed_all(SEED)
    runtime = TinyRuntime(torch)
    optimizer = torch.optim.AdamW(runtime.model.parameters(), lr=1e-5, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    reference = dict(parameters=trainable_state(runtime.model), base_hash="frozen")
    rows = [
        dict(question_id=f"q{i}", fixture_input=i + 1, true_values=[11, 22], operation="sum")
        for i in range(2)
    ]
    for step in (1, 2):
        nonzero_sft_step(
            runtime, optimizer, scheduler, rows, tmp_path, branch="CPU", logical_step=step
        )
    state = save_checkpoint(
        tmp_path / "state.pt", runtime, optimizer, scheduler, 2, reference, stream_hash="fixed"
    )
    identity = dict(
        trainable_names=list(state["parameters"]),
        reference_hash=state_hash(reference),
        frozen_base_hash="frozen",
    )
    # A synthetic one-device RNG marker exercises receipt validation, not GPU execution.
    state["rng"]["cuda"] = [torch.get_rng_state()]
    validate_checkpoint_binding(state, identity, 2, "fixed")
    with pytest.raises(ValueError, match="step/stream/field"):
        validate_checkpoint_binding(state, identity, 4, "fixed")
    changed = copy.deepcopy(state)
    changed["reference"]["base_hash"] = "same-wrong-reference-on-both-branches"
    with pytest.raises(ValueError, match="reference"):
        validate_checkpoint_binding(changed, identity, 2, "fixed")
    changed = copy.deepcopy(state)
    changed["parameters"] = {}
    with pytest.raises(ValueError, match="parameters"):
        validate_checkpoint_binding(changed, identity, 2, "fixed")
    evidence = checkpoint_evidence(tmp_path / "state.pt", state)
    assert set(evidence["field_hashes"]) == set(state)


def test_copied_data_must_match_original_frozen_pool(frozen_root):
    root, project, freeze = frozen_root
    original = __import__("mm_core.execution", fromlist=["read_json"]).read_json(
        root / "manifests/PRE_INFERENCE_FREEZE.json"
    )
    original["file_hashes"]["data/questions.jsonl"] = "b" * 64
    atomic_json(root / "manifests/PRE_INFERENCE_FREEZE.json", original)
    freeze["pre_inference_freeze_sha256"] = sha256_file(
        root / "manifests/PRE_INFERENCE_FREEZE.json"
    )
    atomic_json(root / "manifests/NONZERO_FREEZE.json", freeze)
    with pytest.raises(PermissionError, match="original ENGINE_TEST"):
        verify_nonzero_gate(root, source_root=project)
