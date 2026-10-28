"""Bound G8 generation and actual unchanged native alignment, without GPU calls."""

import copy
import sys
from types import SimpleNamespace

import pytest
import torch

from sr_f6.confirm_runtime import ALIGNMENT_PURPOSE, ConfirmRuntime


@pytest.fixture
def runtime(tmp_path):
    runtime = ConfirmRuntime.__new__(ConfirmRuntime)
    runtime.torch = torch
    runtime.model = torch.nn.Linear(2, 2)
    runtime.model.register_buffer("sentinel", torch.zeros(1))
    for p in runtime.model.parameters():
        p.requires_grad_(False)
    runtime.identity = dict(
        step=0,
        phase="SCHEMA_CONFIRM_V3",
        prompt_version="srf6-interface-v3",
        confirmation_version="srf6-confirm-v3",
        parent_id="b0",
        run_id="task",
        task_id="task",
        parent_content_hash="p",
        source_sha256="s",
        data_hash="d",
        registration_sha256="r",
    )
    runtime._bound_identity = copy.deepcopy(runtime.identity)
    runtime._policy_hash = "a" * 64
    runtime._parameter_versions = {
        n: (id(p), p._version) for n, p in runtime.model.named_parameters()
    }
    runtime._buffer_versions = {n: (id(p), p._version) for n, p in runtime.model.named_buffers()}
    runtime._input_root = tmp_path
    runtime._permit = None
    runtime._attempted_qids = set()
    runtime._public = {
        q: dict(
            qid=q,
            split="DEV_SCHEMA_CONFIRM",
            model_input=dict(prompt=q, image_file=None, image_sha256=None),
        )
        for q in ("q", "q2", "other")
    }
    runtime._assigned_qids = {"q", "q2"}
    runtime._first_qid = "q"
    runtime._first_raw = []
    runtime._alignment_attempted = runtime._alignment_done = False
    runtime._seeds = {q: list(range(100, 108)) for q in runtime._public}
    runtime.account = lambda *args: None
    return runtime


def generation_permit(runtime):
    return dict(
        kind="generation",
        item=runtime._public["q"],
        seeds=runtime._seeds["q"],
        attempted=False,
        token_rows=0,
    )


def test_unrestricted_generation_and_forward_are_closed(runtime, tmp_path):
    for call in (
        lambda: runtime._generate_native_group({}, tmp_path, [100]),
        lambda: runtime.prepare_input({}, tmp_path),
        lambda: runtime.generate_group({}, tmp_path, [100]),
        lambda: runtime._native_forward(
            {}, completion_start=1, lengths=[1], purpose=ALIGNMENT_PURPOSE, grad=False
        ),
        lambda: runtime.batch_sequence_forward([], purpose=ALIGNMENT_PURPOSE, grad=False),
    ):
        with pytest.raises((PermissionError, ValueError)):
            call()


@pytest.mark.parametrize("mutation", ["value", "gradient", "replacement", "identity", "buffer"])
def test_parameter_and_buffer_guards_without_rehash(runtime, mutation):
    assert runtime.policy_tensor_identity() == "a" * 64
    if mutation == "value":
        runtime.model.weight.add_(1)
    elif mutation == "gradient":
        runtime.model.weight.requires_grad_(True)
    elif mutation == "replacement":
        runtime.model.weight = torch.nn.Parameter(runtime.model.weight.clone(), requires_grad=False)
    elif mutation == "buffer":
        runtime.model.sentinel.add_(1)
    else:
        runtime.identity["step"] = 1
    with pytest.raises(PermissionError, match="changed"):
        runtime.policy_tensor_identity()


@pytest.mark.parametrize(
    "field,value",
    [
        ("phase", "CENSUS_S0"),
        ("confirmation_version", "srf6-interface-v3"),
        ("prompt_version", "srf6-interface-v2"),
    ],
)
def test_fake_scientific_identity_is_rejected(runtime, field, value):
    runtime.identity[field] = value
    runtime._bound_identity = copy.deepcopy(runtime.identity)
    with pytest.raises(PermissionError, match="changed"):
        runtime.policy_tensor_identity()


def test_exact_g8_accounting(runtime):
    runtime._permit = generation_permit(runtime)
    with pytest.raises(PermissionError):
        runtime.reserve("completion_attempts", 2, group_seed=100, protocol="structured")
    runtime.reserve("completion_attempts", 8, group_seed=100, protocol="structured")
    with pytest.raises(PermissionError):
        runtime.reserve("completion_attempts", 8, group_seed=100, protocol="structured")
    for draw in range(8):
        runtime.reserve("generated_tokens", 1536, group_seed=100, row_index=draw)
    with pytest.raises(PermissionError):
        runtime.reserve("generated_tokens", 1, group_seed=100, row_index=8)
    with pytest.raises(PermissionError):
        runtime.reserve(
            "extra_forward_sequences", 1, purpose=ALIGNMENT_PURPOSE, completion_tokens=1
        )


def test_failed_sampling_consumes_qid_and_permit(runtime):
    def fail(*args, **kwargs):
        runtime.reserve("completion_attempts", 8, group_seed=100, protocol="structured")
        raise RuntimeError("original failure")

    runtime._generate_native_group = fail
    with pytest.raises(RuntimeError, match="original failure"):
        runtime.generate_registered("q", on_completion=lambda row: None)
    assert runtime._permit is None
    with pytest.raises(PermissionError, match="repeated"):
        runtime.generate_registered("q", on_completion=lambda row: None)


def test_worker_and_first_group_access(runtime):
    with pytest.raises(PermissionError, match="another registered worker"):
        runtime.generate_registered("other", on_completion=lambda row: None)
    with pytest.raises(PermissionError, match="alignment must finish"):
        runtime.generate_registered("q2", on_completion=lambda row: None)
    runtime._public["q"]["split"] = "DEV_CAL"
    with pytest.raises(PermissionError, match="split"):
        runtime.generate_registered("q", on_completion=lambda row: None)


@pytest.fixture
def native(runtime, monkeypatch):
    from sr_f6 import runtime as original

    class Config(SimpleNamespace):
        def to_dict(self):
            return vars(self)

    def generate(input_ids, generation_config, **kwargs):
        assert input_ids.shape[0] == 8
        assert generation_config.max_new_tokens == 1536
        assert generation_config.temperature == 1.0
        logits = torch.tensor([[0.0, 1.0]] * 8)
        return SimpleNamespace(
            sequences=torch.cat((input_ids, torch.ones((8, 1), dtype=torch.long)), 1),
            logits=[logits],
            scores=[logits],
        )

    def forward(input_ids, logits_to_keep, **kwargs):
        assert input_ids.shape == (1, 2)
        assert not torch.is_grad_enabled()
        torch.rand(1)  # Alignment must restore the caller RNG even through real forwards.
        return SimpleNamespace(logits=torch.tensor([[[0.0, 1.0]] * logits_to_keep]))

    runtime.model.generate = generate
    runtime.model.forward = forward
    runtime.model.get_rope_index = lambda input_ids, **kwargs: (
        torch.zeros((3, 1, input_ids.shape[-1]), dtype=torch.long),
        None,
    )
    runtime.generation_config = Config()
    runtime.processor = SimpleNamespace(
        tokenizer=SimpleNamespace(pad_token_id=0, decode=lambda *args, **kwargs: '"answer":1}')
    )
    runtime.protocol_revision = "v2"
    runtime.eos_ids = []
    runtime.image_calls = 0
    runtime.adapter_path = "registered-parent"
    prepared = dict(
        inputs=dict(
            input_ids=torch.tensor([[0]]),
            attention_mask=torch.ones((1, 1), dtype=torch.long),
            mm_token_type_ids=torch.zeros((1, 1), dtype=torch.long),
        ),
        routing={"input_tensor_hash": "h"},
    )
    runtime._srf5_prepared = prepared
    runtime.prepare_input = lambda *args, **kwargs: prepared
    runtime._validate_prepared_revision = lambda *args: None
    monkeypatch.setattr(
        original,
        "BatchedProtocolStop",
        lambda *args, **kwargs: SimpleNamespace(counts=[1] * 8, cuts=[12] * 8),
    )
    monkeypatch.setattr(original, "first_boundary_token", lambda *args, **kwargs: (1, 12))
    monkeypatch.setattr(original, "decode_generated", lambda *args: '"answer":1}')
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(StoppingCriteriaList=list))
    raw = []
    runtime.generate_registered("q", on_completion=raw.append)
    return runtime, raw


def test_actual_native_sampler_and_single_sequence_alignment(native):
    runtime, raw = native
    original = copy.deepcopy(raw)
    assert [r["seed"] for r in raw] == list(range(100, 108))
    assert all(r["phase"] == "SCHEMA_CONFIRM_V3" and r["group_size"] == 8 for r in raw)
    before = torch.get_rng_state().clone()
    events = []
    runtime.account = lambda kind, count, metadata: events.append((kind, count, metadata))
    output = runtime.replay_alignment(raw)
    assert len(output) == 8
    assert all(
        torch.allclose(value, torch.tensor(row["sampler_logprobs"]))
        for row, value in zip(raw, output, strict=True)
    )
    assert all(value.device.type == "cpu" and value.dtype == torch.float32 for value in output)
    assert (
        events
        == [("extra_forward_sequences", 1, dict(purpose=ALIGNMENT_PURPOSE, completion_tokens=1))]
        * 8
    )
    assert torch.equal(before, torch.get_rng_state())
    assert raw == original and runtime._alignment_done and runtime._permit is None
    with pytest.raises(PermissionError, match="once"):
        runtime.replay_alignment(raw)


@pytest.mark.parametrize(
    "mutation", ["tokens", "sampler_logprobs", "phase", "seed", "order", "short"]
)
def test_alignment_requires_original_first_eight_and_consumes_failed_permit(native, mutation):
    runtime, raw = native
    altered = copy.deepcopy(raw)
    if mutation == "order":
        altered.reverse()
    elif mutation == "short":
        altered.pop()
    else:
        altered[0][mutation] = {
            "tokens": [0],
            "sampler_logprobs": [0.0],
            "phase": "CENSUS_S0",
            "seed": 999,
        }[mutation]
    with pytest.raises(PermissionError, match="original first"):
        runtime.replay_alignment(altered)
    with pytest.raises(PermissionError, match="once"):
        runtime.replay_alignment(raw)
    assert not runtime._alignment_done


def test_alignment_forward_failure_restores_rng_and_prevents_retry(native):
    runtime, raw = native

    def fail(*args, **kwargs):
        torch.rand(1)
        raise RuntimeError("native fault")

    runtime.model.forward = fail
    before = torch.get_rng_state().clone()
    with pytest.raises((RuntimeError, TypeError)):
        runtime.replay_alignment(raw)
    assert torch.equal(before, torch.get_rng_state())
    assert runtime._permit is None and not runtime._alignment_done
    with pytest.raises(PermissionError, match="once"):
        runtime.replay_alignment(raw)


@pytest.fixture
def load_environment(tmp_path, monkeypatch):
    import json

    from mm_dev import runtime as backend
    from sr_f6 import confirm_runtime as module

    root, origin, adapter = (tmp_path / name for name in ("confirmation", "old-source", "adapter"))
    for directory in (root, origin, adapter):
        directory.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    config = dict(
        master_seed=20261027,
        training=dict(temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=1536),
        numerics=dict(
            model_dtype="float32", input_float_dtype="float32", probability_dtype="float32"
        ),
        model=dict(
            model_path="original-model", parents={"b0": str(adapter)}, snapshot="original-snapshot"
        ),
    )
    item = dict(qid="q", split="DEV_SCHEMA_CONFIRM", model_input={"prompt": "registered input"})
    context = dict(
        root=root,
        registration_root=root,
        input_root=root,
        source_root=origin,
        registration_sha256="a" * 64,
        config=config,
        public=[item],
        registration=dict(
            source_manifest_sha256="b" * 64,
            data_sha256="c" * 64,
            parent_content_hashes={"b0": "d" * 64},
        ),
        plan=dict(
            phase="SCHEMA_CONFIRM_V3",
            version="srf6-confirm-v3",
            source_split="DEV_SCHEMA_CONFIRM",
            attempts=[dict(parent="b0", qid="q", draw=i, seed=100 + i) for i in range(8)],
            workers=[dict(worker_id="b0-s0-of-2", parent="b0", shard=0, qids=["q"], first_qid="q")],
        ),
        preflight=dict(inputs=[dict(qid="q", routing={"input_tensor_hash": "route"})]),
    )
    checks, constructions = [], []

    def verify(base, *, parent, require_preflight):
        checks.append((base, parent, require_preflight))
        return copy.deepcopy(context)

    registration = SimpleNamespace(verify_registered=verify)
    monkeypatch.setitem(sys.modules, "sr_f6.confirm_registration", registration)

    class FakeConfirmRuntime(ConfirmRuntime):
        def __init__(self, model, *, adapter_path, output_root, protocol_revision, account):
            constructions.append((model, adapter_path, output_root, protocol_revision))
            self.model = torch.nn.Linear(2, 2).double()
            self.model.register_buffer("buffer", torch.zeros(1, dtype=torch.float64))
            self.identity = {"dtype": "float64"}
            self.account = account
            self.torch = torch

    monkeypatch.setattr(module, "ConfirmRuntime", FakeConfirmRuntime)
    monkeypatch.setattr(module, "configure_parent_training", lambda *a, **kw: None)
    monkeypatch.setattr(module, "seed_all", lambda *a: None)
    monkeypatch.setattr(backend, "configure_audited_backend", lambda: None)
    monkeypatch.setattr(backend, "actual_cuda_identity", lambda: {"fixture": "CUDA_API_MOCK_ONLY"})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    return SimpleNamespace(
        module=module,
        context=context,
        root=root,
        origin=origin,
        adapter=adapter,
        checks=checks,
        constructions=constructions,
        registration=registration,
        json=json,
    )


def test_production_loader_binds_new_inputs_and_original_model_with_actual_receipts(
    load_environment,
):
    e = load_environment
    task = "LH-SRF6-Confirm-100-b0-0"
    runtime = e.module.load_confirm(e.context, "b0", task)
    assert e.checks == [(e.root, "b0", True)] * 2
    assert e.constructions == [("original-model", str(e.adapter), e.root, "v2")]
    assert runtime._input_root == e.root and runtime.identity["source_root"] == str(e.origin)
    assert runtime._assigned_qids == {"q"} and runtime._first_qid == "q"
    assert all(
        not p.requires_grad and p.grad is None and p.dtype == torch.float32
        for p in runtime.model.parameters()
    )
    receipt = e.json.loads((e.root / f"CONFIRM/runtime_identity/{task}.json").read_text())
    assert receipt["source_sha256"] == "b" * 64 and receipt["data_hash"] == "c" * 64
    assert receipt["parent_content_hash"] == "d" * 64
    assert (
        receipt["phase"] == "SCHEMA_CONFIRM_V3"
        and receipt["execution_mode"] == "READONLY_SCHEMA_CONFIRM"
    )
    assert receipt["worker_id"] == "b0-s0-of-2" and receipt["shard"] == 0
    assert receipt["optimizer"] is receipt["scheduler"] is None
    assert receipt["fresh_optimizer"] is False and receipt["trainable_parameters"] == []
    runtime._permit = generation_permit(runtime)
    runtime.reserve("completion_attempts", 8, group_seed=100, protocol="structured")
    event = e.json.loads((e.root / f"CONFIRM/accounting/{task}.jsonl").read_text())
    assert event["task_id"] == task and event["parent"] == "b0"
    assert event["metadata"] == dict(
        group_seed=100,
        protocol="structured",
        qid="q",
        phase="SCHEMA_CONFIRM_V3",
        confirmation_version="srf6-confirm-v3",
        registration_sha256="a" * 64,
    )
    assert not list(e.origin.rglob("*.json"))
    with pytest.raises(FileExistsError, match="Preserve"):
        e.module.load_confirm(e.context, "b0", task)
    assert len(e.constructions) == 1


def test_production_load_rejects_changed_registration_before_identity_is_written(load_environment):
    e = load_environment
    original = e.registration.verify_registered

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        if len(e.checks) == 2:
            result["registration_sha256"] = "0" * 64
        return result

    e.registration.verify_registered = changed
    with pytest.raises(PermissionError, match="changed while loading"):
        e.module.load_confirm(e.context, "b0", "LH-SRF6-Confirm-100-b0-0")
    assert not list((e.root / "CONFIRM/runtime_identity").glob("*.json"))


@pytest.mark.parametrize(
    "task",
    ["LH-SRF6-Confirm-100-b1-0", "LH-SRF6-Confirm-100-b0-2", "LH-SRF6-V3-S1-100-b0-0", "../escape"],
)
def test_production_load_rejects_wrong_job_parent_shard(load_environment, task):
    with pytest.raises(ValueError, match="identifier"):
        load_environment.module.load_confirm(load_environment.context, "b0", task)
    assert not load_environment.constructions


def test_preflight_route_and_new_input_root_are_enforced(runtime, tmp_path, monkeypatch):
    from sr_f6.confirm_runtime import FP32Runtime

    route = dict(
        input_tensor_hash="input",
        processor_hash="processor",
        chat_template_hash="template",
        chat_text_sha256="text",
    )
    runtime._preflight_inputs = {"q": copy.deepcopy(route)}
    runtime._permit = generation_permit(runtime)
    monkeypatch.setattr(
        FP32Runtime, "prepare_input", lambda *a, **kw: dict(inputs={}, routing=copy.deepcopy(route))
    )
    item = runtime._public["q"]["model_input"]
    assert runtime.prepare_input(item, tmp_path)["routing"]["input_tensor_hash"] == "input"
    other = tmp_path / "old-source"
    other.mkdir()
    with pytest.raises(PermissionError, match="registered confirmation unit"):
        runtime.prepare_input(item, other)
    with pytest.raises(PermissionError, match="registered confirmation unit"):
        runtime.prepare_input(dict(item, prompt="other"), tmp_path)
    runtime._preflight_inputs["q"]["input_tensor_hash"] = "changed"
    with pytest.raises(PermissionError, match="processor preflight"):
        runtime.prepare_input(item, tmp_path)


def test_completed_alignment_that_fails_numeric_gate_cannot_allow_next_question(native):
    runtime, raw = native
    forwards = []

    def changed(input_ids, logits_to_keep, **kwargs):
        forwards.append(input_ids.detach().clone())
        return SimpleNamespace(logits=torch.tensor([[[1.0, 0.0]] * logits_to_keep]))

    runtime.model.forward = changed
    captured = []
    with pytest.raises(RuntimeError):
        runtime.replay_alignment(raw, on_replay=lambda draw, value: captured.append((draw, value)))
    assert [draw for draw, _ in captured] == list(range(8))
    assert all(value.device.type == "cpu" and value.dtype == torch.float32 for _, value in captured)
    assert len(forwards) == 8
    assert runtime._alignment_attempted and not runtime._alignment_done
    assert runtime._permit is None
    with pytest.raises(PermissionError, match="alignment must finish"):
        runtime.generate_registered("q2", on_completion=lambda row: None)
    with pytest.raises(PermissionError, match="once"):
        runtime.replay_alignment(raw)


def test_alignment_callback_preserves_partial_vectors_and_rng_on_native_failure(native):
    runtime, raw = native
    count, captured = [], []

    def fail_third(input_ids, logits_to_keep, **kwargs):
        count.append(1)
        torch.rand(1)
        if len(count) == 3:
            raise RuntimeError("third original forward failed")
        return SimpleNamespace(logits=torch.tensor([[[0.0, 1.0]] * logits_to_keep]))

    runtime.model.forward = fail_third
    before = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="third original"):
        runtime.replay_alignment(raw, on_replay=lambda draw, value: captured.append((draw, value)))
    assert [draw for draw, _ in captured] == [0, 1]
    assert torch.equal(before, torch.get_rng_state())
    assert not runtime._alignment_done and runtime._alignment_attempted


def test_alignment_callback_cannot_mutate_probability_gate_vectors(native):
    runtime, raw = native
    result = runtime.replay_alignment(raw, on_replay=lambda draw, value: value.fill_(100))
    assert runtime._alignment_done
    assert all(
        torch.allclose(value, torch.tensor(row["sampler_logprobs"]))
        for row, value in zip(raw, result, strict=True)
    )
