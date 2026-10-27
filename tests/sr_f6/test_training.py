"""Tiny CPU behavior fixtures; these are never real-model/GPU gate evidence."""

import copy
import hashlib
import json

import pytest
import torch
from PIL import Image

from mm_core.training import capture_rng, state_hash, trainable_state
from mm_core.vl_runtime import hash_json
from sr_f3_ce.training import fresh_optimizer
from sr_f6.common import file_hash
from sr_f6.objective import sequence_loss
from sr_f6.training import build_credit_group, trajectory_hash, update
from sr_f12.training import ProbabilityMismatch


class TinyCPUFixtureRuntime:
    def __init__(self, root):
        self.model = torch.nn.Linear(1, 1, bias=False)
        self.model.weight.data.zero_()
        self.reference = trainable_state(self.model)
        self.identity = {
            "run_id": "CPU_FIXTURE_ONLY",
            "parent_id": "b0",
            "step": 0,
            "run_root": str(root.resolve()),
            "source_sha256": "1" * 64,
            "reference_hash": state_hash(self.reference),
            "reference_parent": "b0",
            "diagnostic_row_indices": list(range(64)),
        }
        self.calls = []
        self.accounts = []
        self.fail_row = None
        self.nonfinite_row = None
        self.change_prompt = False
        self.mutate_parameter = False

    def prepare_input(self, model_input, root, protocol):
        inputs = {
            "input_ids": torch.tensor([[10, 11, 12]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "mm_token_type_ids": torch.zeros((1, 3), dtype=torch.long),
            "image_grid_thw": torch.tensor([[1, 1, 1]]),
            "pixel_values": torch.zeros((1, 3), dtype=torch.float32),
        }
        digest = state_hash(inputs)
        if self.change_prompt:
            inputs["input_ids"][0, 0] = 99
        return {"inputs": inputs, "routing": {"input_tensor_hash": digest}}

    def batch_reference_forward(self, records, reference):
        return [
            (-2 + reference["weight"].flatten()[0] * record["slope"]).expand(len(record["tokens"]))
            for record in records
        ]

    def batch_sequence_forward(self, records, purpose, grad):
        results = []
        for record in records:
            self.calls.append(record["row_id"])
            if self.fail_row == record["row_id"]:
                raise RuntimeError("CPU fixture injected forward failure")
            if self.mutate_parameter:
                with torch.no_grad():
                    self.model.weight.add_(1e-7)
            value = (-2 + self.model.weight.flatten()[0] * record["slope"]).expand(
                len(record["tokens"])
            )
            if self.nonfinite_row == record["row_id"]:
                value = value * float("nan")
            results.append(value)
        return results

    def reserve(self, kind, count, **metadata):
        self.accounts.append((kind, count, metadata))


def batch(tmp_path, arm="U10", *, length=2):
    runtime = TinyCPUFixtureRuntime(tmp_path)
    image = tmp_path / "fixture.png"
    Image.new("RGB", (2, 2), (0, 0, 0)).save(image)
    factor = "P0" if arm in ("U00", "U01") else "P1"
    records = []
    for group in range(16):
        prompt = f"CPU fixture question {group}"
        model_input = {
            "prompt": prompt,
            "image_file": "fixture.png",
            "image_sha256": file_hash(image),
        }
        prepared = runtime.prepare_input(model_input, tmp_path, "structured_v4")
        for draw in range(8):
            n = length if length == 1536 else length + draw % 2
            sampling = {
                "max_new_tokens": 1536,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 0,
                "do_sample": True,
            }
            row = {
                "row_id": f"{group}/{draw}",
                "group_id": f"group-{group}",
                "qid": f"q{group}",
                "parent": "b0",
                "factor": factor,
                "root_id": f"root-{group // 4}",
                "family": "BINDING",
                "variant": group % 4,
                "draw": draw,
                "run_id": runtime.identity["run_id"],
                "step": 1,
                "run_root": str(tmp_path.resolve()),
                "protocol": "structured_v4",
                "policy_hash": state_hash(runtime.reference),
                "source_sha256": "1" * 64,
                "adapter_identity": {
                    "run_id": runtime.identity["run_id"],
                    "parent": "b0",
                    "step": 0,
                },
                "tokens": list(range(n)),
                "raw_tokens": list(range(n)),
                "completion_token_count": n,
                "prompt_token_count": 3,
                "sampler_logprobs": [-2.0] * n,
                "sampler_logprob_source": "actual_generate_raw_logits_selected_token",
                "model_input": model_input,
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "prompt_tensor_hash": prepared["routing"]["input_tensor_hash"],
                "native_prompt_inputs": {
                    key: value.tolist()
                    for key, value in prepared["inputs"].items()
                    if key != "pixel_values"
                },
                "max_new_tokens": 1536,
                "sampling_parameters": sampling,
                "sampling_hash": hash_json(sampling),
                "generation_status": "COMPLETE",
                "technical_validation_errors": [],
                "slope": int(group == 0 and draw < 4),
                "score": {
                    "J": int(group == 0 and draw < 4),
                    "V": 1,
                    "finished": 1,
                    "S": int(draw == 7),
                    "B": int(draw == 7),
                },
            }
            row["token_path_sha256"] = trajectory_hash(row)
            records.append(row)
    credits = [
        build_credit_group(records[start : start + 8], arm, start + 17)
        for start in range(0, 128, 8)
    ]
    optimizer, scheduler = fresh_optimizer(runtime.model)
    return runtime, optimizer, scheduler, records, credits


def execute(items, **kwargs):
    runtime, optimizer, scheduler, records, credits = items
    return update(
        runtime,
        optimizer,
        scheduler,
        records,
        [8] * 16,
        runtime.reference,
        credits,
        diagnostic_row_indices=kwargs.pop("diagnostic_row_indices", tuple(range(64))),
        **kwargs,
    )


def assert_no_step(items, before, epoch):
    runtime, optimizer, scheduler, _, _ = items
    torch.testing.assert_close(runtime.model.weight, before, rtol=0, atol=0)
    assert not optimizer.state
    assert scheduler.last_epoch == epoch
    assert runtime.model.weight.grad is None


def test_equal_128_sequence_weight_all_zero_groups_retained_and_single_step(tmp_path):
    items = batch(tmp_path)
    runtime, optimizer, scheduler, records, credits = items
    immutable_rows, immutable_credits = copy.deepcopy(records), copy.deepcopy(credits)
    rng_before = state_hash(capture_rng())
    report = execute(items)
    assert report["gradient_norm_before_clip"] == pytest.approx(4 / 128)
    assert report["gradient_norm_after_clip"] == pytest.approx(4 / 128)
    assert report["sequence_count"] == len(runtime.calls) == 128
    assert report["optimizer_updates"] == 1
    assert int(optimizer.state[runtime.model.weight]["step"]) == 1
    assert scheduler.last_epoch == 1
    assert runtime.identity["step"] == 0  # Caller has not committed this update.
    assert report["status"] == "UPDATED_UNCOMMITTED"
    assert report["parameter_update_norm"] > 0
    assert len(report["probability_vectors"]) == 128
    assert report["rng_before_sha256"] == report["rng_after_sha256"] == rng_before
    assert state_hash(capture_rng()) == rng_before
    assert records == immutable_rows and credits == immutable_credits


@pytest.mark.parametrize("arm", ("U00", "U01", "U10", "U11", "C_GLOBAL", "C_SHUF"))
def test_all_six_arms_use_actual_bound_credit_and_kl_once(tmp_path, arm):
    items = batch(tmp_path, arm)
    report = execute(items)
    assert report["actual_credit_groups"] == items[4]
    assert report["kl"] == 0.0
    assert report["loss"] == pytest.approx(report["native"] + report["aux"] + report["kl"])
    channels = report["channel_gradients"]
    assert channels["status"] == "MEASURED"
    assert channels["qid_count"] == 8
    assert set(channels["norms"]) == {"native", "aux", "kl", "total"}
    assert channels["norms"]["kl"] == 0.0


def test_diagnostics_reuse_graph_do_not_pollute_formal_gradient(tmp_path):
    items = batch(tmp_path, "C_GLOBAL")
    runtime, _, _, records, credits = items
    manual = torch.zeros_like(runtime.model.weight)
    for row in records:
        coefficients = credits[int(row["group_id"].split("-")[-1])]["coefficients_by_row_id"][
            row["row_id"]
        ]
        logp = (-2 + runtime.model.weight.flatten()[0] * row["slope"]).expand(len(row["tokens"]))
        result = sequence_loss(
            logp,
            torch.tensor(row["sampler_logprobs"]),
            logp.detach(),
            coefficients["native"],
            coefficients["aux"],
        )
        manual += torch.autograd.grad(result["loss"] / 128, runtime.model.weight)[0]
    report = execute(items)
    assert report["gradient_norm_before_clip"] == pytest.approx(float(manual.norm()))
    assert (
        sum(count for kind, count, _ in runtime.accounts if kind == "diagnostic_gradient_calls")
        == 64 * 3
    )


def test_full_1536_token_trajectories_are_accepted(tmp_path):
    items = batch(tmp_path, length=1536)
    report = execute(items)
    assert report["token_count"] == 128 * 1536
    assert all(len(trace["tokens"]) == 1536 for trace in report["probability_vectors"])


def test_clip_is_once_after_accumulating_all_sequences(tmp_path):
    items = batch(tmp_path)
    for row in items[3]:
        row["slope"] *= 1000
    report = execute(items)
    assert report["gradient_norm_before_clip"] == pytest.approx(1000 * 4 / 128)
    assert report["gradient_norm_after_clip"] == pytest.approx(1.0)
    assert report["optimizer_updates"] == 1


@pytest.mark.parametrize("kind", ("mean", "tail", "tis"))
def test_real_probability_gates_block_before_adam_and_preserve_traces(tmp_path, kind):
    items = batch(tmp_path, length=64)
    runtime, _, scheduler, records, _ = items
    if kind == "mean":
        for row in records:
            row["sampler_logprobs"] = [-2.02] * len(row["tokens"])
    elif kind == "tail":
        records[0]["sampler_logprobs"][9] = -8.0
    else:
        # 1.04% at +0.7 nat breaches TIS while mean remains below 0.01.
        for row in records[:2]:
            for index in range(43):
                row["sampler_logprobs"][index] = -2.7
    for row in records:
        row["token_path_sha256"] = trajectory_hash(row)
    before, epoch = runtime.model.weight.detach().clone(), scheduler.last_epoch
    with pytest.raises(ProbabilityMismatch) as failure:
        execute(items)
    assert_no_step(items, before, epoch)
    audit = failure.value.diagnostics
    if kind in ("tail", "tis"):
        assert audit["sampler_train_mean_absolute_difference"] < 0.01
    if kind == "tail":
        assert audit["tis_truncated_fraction"] < 0.01
        assert audit["sampler_train_max_absolute_difference"] == 6.0
    if kind == "tis":
        assert audit["tis_truncated_fraction"] > 0.01
        assert audit["sampler_train_max_absolute_difference"] < 5.0
    saved = json.loads((tmp_path / "NUMERICAL_TOKEN_AUDIT.jsonl").read_text().splitlines()[-1])
    assert saved["status"] == "UNCOMMITTED_FAILURE_NO_OPTIMIZER_STEP"
    assert len(saved["probability_vectors"]) == 128
    assert len(saved["original_trajectories"]) == 128


@pytest.mark.parametrize(
    "field,value",
    (
        ("source_sha256", "2" * 64),
        ("policy_hash", "2" * 64),
        ("run_id", "wrong-run"),
        ("step", 2),
        ("token_path_sha256", "2" * 64),
    ),
)
def test_stale_or_tampered_identity_rejected_before_forward(tmp_path, field, value):
    items = batch(tmp_path)
    runtime, _, scheduler, records, _ = items
    records[0][field] = value
    before, epoch = runtime.model.weight.detach().clone(), scheduler.last_epoch
    with pytest.raises((ValueError, PermissionError)):
        execute(items)
    assert not runtime.calls
    assert_no_step(items, before, epoch)


def test_actual_prepared_prompt_tokens_must_match_saved_native_tokens(tmp_path):
    items = batch(tmp_path)
    runtime, _, scheduler, _, _ = items
    runtime.change_prompt = True
    before, epoch = runtime.model.weight.detach().clone(), scheduler.last_epoch
    with pytest.raises(PermissionError, match="Native prompt"):
        execute(items)
    assert_no_step(items, before, epoch)
    assert not runtime.calls


def test_reference_must_be_bound_parent_content(tmp_path):
    items = batch(tmp_path)
    runtime, _, scheduler, _, _ = items
    runtime.reference["weight"].add_(1)
    before, epoch = runtime.model.weight.detach().clone(), scheduler.last_epoch
    with pytest.raises(PermissionError, match="reference content"):
        execute(items)
    assert_no_step(items, before, epoch)


def test_forward_failure_and_nan_retain_trace_and_restore_rng(tmp_path):
    items = batch(tmp_path)
    runtime, _, scheduler, _, _ = items
    runtime.nonfinite_row = "0/2"
    before, epoch = runtime.model.weight.detach().clone(), scheduler.last_epoch
    rng = state_hash(capture_rng())
    with pytest.raises(FloatingPointError):
        execute(items)
    assert_no_step(items, before, epoch)
    assert state_hash(capture_rng()) == rng
    saved = json.loads((tmp_path / "NUMERICAL_TOKEN_AUDIT.jsonl").read_text().splitlines()[-1])
    assert saved["probability_vectors"][-1]["train_logprobs"] == ["nan"] * 2


def test_backend_exception_preserves_completed_and_original_trajectories(tmp_path):
    items = batch(tmp_path)
    runtime, _, scheduler, _, _ = items
    runtime.fail_row = "0/3"
    before, epoch = runtime.model.weight.detach().clone(), scheduler.last_epoch
    with pytest.raises(RuntimeError, match="injected forward"):
        execute(items)
    assert_no_step(items, before, epoch)
    saved = json.loads((tmp_path / "NUMERICAL_TOKEN_AUDIT.jsonl").read_text().splitlines()[-1])
    assert len(saved["probability_vectors"]) == 3
    assert len(saved["original_trajectories"]) == 128
    assert saved["requires_checkpoint_restore"] is False


def test_accidental_forward_parameter_mutation_is_flagged_for_restore(tmp_path):
    items = batch(tmp_path)
    runtime, optimizer, scheduler, _, _ = items
    runtime.mutate_parameter = True
    with pytest.raises(PermissionError, match="forward pass changed"):
        execute(items)
    assert not optimizer.state
    assert scheduler.last_epoch == 0
    saved = json.loads((tmp_path / "NUMERICAL_TOKEN_AUDIT.jsonl").read_text().splitlines()[-1])
    assert saved["status"] == "UNCOMMITTED_FAILURE_RESTORE_CHECKPOINT"
    assert saved["parameter_state_changed"] is True
    assert saved["optimizer_started"] is False


def test_registered_diagnostic_cannot_silently_be_omitted(tmp_path):
    items = batch(tmp_path)
    with pytest.raises(PermissionError, match="64 unique"):
        execute(items, diagnostic_row_indices=())
    assert not items[0].calls


def test_incomplete_group_or_root_matrix_rejected(tmp_path):
    items = batch(tmp_path)
    runtime, _, _, records, _ = items
    for row in records:
        row["root_id"] = "only-one-root"
    with pytest.raises(ValueError, match="four roots"):
        execute(items)
    assert not runtime.calls


def test_update_after_adam_exception_requires_checkpoint_restore(tmp_path, monkeypatch):
    items = batch(tmp_path)
    _, optimizer, _, _, _ = items
    original_step = optimizer.step

    def interrupted(*args, **kwargs):
        original_step(*args, **kwargs)
        raise RuntimeError("CPU fixture interruption after actual Adam step")

    monkeypatch.setattr(optimizer, "step", interrupted)
    with pytest.raises(RuntimeError, match="after actual Adam"):
        execute(items)
    saved = json.loads((tmp_path / "NUMERICAL_TOKEN_AUDIT.jsonl").read_text().splitlines()[-1])
    assert saved["status"] == "UPDATE_ATTEMPT_FAILED_RESTORE_CHECKPOINT"
    assert int(optimizer.state[items[0].model.weight]["step"]) == 1
    assert items[0].identity["step"] == 0
