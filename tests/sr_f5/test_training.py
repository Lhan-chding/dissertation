import copy
import json

import pytest
import torch

from sr_f3_ce.training import fresh_optimizer
from sr_f5.training import build_credit_group, update
from sr_f12.training import ProbabilityMismatch, sequence_objective
from srf5_ref.objective import sequence_loss


class FakeRuntime:
    def __init__(self):
        self.model = torch.nn.Linear(1, 1, bias=False)
        self.model.weight.data.zero_()
        self.identity = {"run_id": "fixture", "step": 0}
        self.calls = []
        self.accounts = []

    def prepare_input(self, model_input, root, protocol):
        return {"fixture": model_input}

    def batch_reference_forward(self, records, reference):
        return [torch.full((len(r["tokens"]),), -2.0) for r in records]

    def batch_sequence_forward(self, records, purpose, grad):
        self.calls.extend(r["row_id"] for r in records)
        return [
            (-2 + self.model.weight.flatten()[0] * r["slope"]).expand(len(r["tokens"]))
            for r in records
        ]

    def reserve(self, kind, count, **metadata):
        self.accounts.append((kind, count, metadata))


def batch(tmp_path, arm="J", *, length=2):
    records = []
    for group in range(16):
        for slot in range(8):
            n = length + (slot % 2)
            score = {
                "J": int(group != 0 or slot < 4),
                "J_M3": int(group != 0 or slot < 4),
                "V": 1,
                "finish": 1,
                "S": "PASS" if slot < 4 else "FAIL",
                "P": "PASS",
                "value_correct": int(group != 0 or slot < 4),
            }
            records.append(
                {
                    "row_id": f"{group}/{slot}",
                    "qid": f"q{group}",
                    "group_id": f"g{group}",
                    "tokens": list(range(n)),
                    "sampler_logprobs": [-2.0] * n,
                    "score": score,
                    "selected_mask": [True] + [False] * (n - 1),
                    "mask_ok": True,
                    "model_input": {"prompt": f"q{group}"},
                    "run_root": str(tmp_path),
                    "protocol": "structured_v3",
                    "adapter_identity": {"step": 0, "run_id": "fixture", "parent": None},
                    "slope": int(group == 0 and slot < 4),
                }
            )
    groups = [build_credit_group(records[i : i + 8], arm, i + 17) for i in range(0, 128, 8)]
    return records, groups


def test_native_forward_and_gradient_equivalence():
    rng = torch.Generator().manual_seed(2026102511)
    for length in (1, 7, 1024):
        current = (-torch.rand(length, generator=rng) * 5).requires_grad_()
        sampler = current.detach() + torch.randn(length, generator=rng) * 0.002
        reference = current.detach() + torch.randn(length, generator=rng) * 0.1
        for advantage in (0.0, -2.64575131, 0.37796447):
            native = sequence_objective(current, sampler, reference, advantage)
            actual = sequence_loss(current, sampler, reference, advantage)
            assert torch.allclose(actual["loss"], native["loss"], atol=1e-7, rtol=1e-6)
            a = torch.autograd.grad(native["loss"], current)[0]
            b = torch.autograd.grad(actual["loss"], current)[0]
            assert torch.allclose(a, b, atol=1e-7, rtol=1e-6)


def test_fixed_128_denominator_one_optimizer_step(tmp_path):
    runtime = FakeRuntime()
    optimizer, scheduler = fresh_optimizer(runtime.model)
    records, groups = batch(tmp_path)
    before = scheduler.last_epoch
    result = update(runtime, optimizer, scheduler, records, [8] * 16, {}, groups)
    # Four +1-advantage rows have a unit derivative, all others zero. Their
    # varying token lengths do not alter their equal 1/128 sequence weight.
    assert result["gradient_norm_before_clip"] == pytest.approx(4 / 128)
    assert result["gradient_norm_after_clip"] == pytest.approx(4 / 128)
    assert result["sequence_count"] == len(runtime.calls) == 128
    assert result["optimizer_updates"] == 1
    assert scheduler.last_epoch == before + 1
    assert int(optimizer.state[runtime.model.weight]["step"]) == 1
    assert len(result["probability_vectors"]) == 128
    assert result["parameter_update_norm"] > 0


@pytest.mark.parametrize("arm", ["J", "HARD", "DENSE", "LOCAL", "SHUF"])
def test_all_arm_receipts_recomputed_and_kept(tmp_path, arm):
    runtime = FakeRuntime()
    records, groups = batch(tmp_path, arm)
    optimizer, scheduler = fresh_optimizer(runtime.model)
    report = update(runtime, optimizer, scheduler, records, [8] * 16, {}, groups)
    assert report["actual_credit_groups"] == groups
    assert report["kl"] == 0
    if arm == "HARD":
        assert groups[1]["native"] != [0] * 8
    if arm in ("DENSE", "LOCAL"):
        assert groups[0]["aux"] == [0.125] * 4 + [-0.125] * 4


def test_tail_gate_prevents_step_even_when_mean_and_tis_pass(tmp_path):
    runtime = FakeRuntime()
    optimizer, scheduler = fresh_optimizer(runtime.model)
    records, groups = batch(tmp_path, length=64)
    records[0]["sampler_logprobs"][9] = -8.0
    before, epoch = runtime.model.weight.detach().clone(), scheduler.last_epoch
    with pytest.raises(ProbabilityMismatch, match="5 nat") as caught:
        update(runtime, optimizer, scheduler, records, [8] * 16, {}, groups)
    diag = caught.value.diagnostics
    assert diag["sampler_train_mean_absolute_difference"] < 0.01
    assert diag["tis_truncated_fraction"] < 0.01
    assert diag["sampler_train_max_absolute_difference"] == 6
    top = diag["sampler_train_top10"][0]
    assert (top["row_id"], top["token_position"], top["token_id"]) == ("0/0", 9, 9)
    assert (top["sampler_logprob"], top["train_logprob"], top["reference_logprob"]) == (-8, -2, -2)
    assert torch.equal(runtime.model.weight, before)
    assert scheduler.last_epoch == epoch
    assert not optimizer.state
    assert runtime.model.weight.grad is None
    saved = json.loads((tmp_path / "NUMERICAL_TOKEN_AUDIT.jsonl").read_text())
    assert saved["status"] == "UNCOMMITTED_FAILURE"
    assert len(saved["probability_vectors"]) == 128


def test_mean_gate_prevents_step(tmp_path):
    runtime = FakeRuntime()
    optimizer, scheduler = fresh_optimizer(runtime.model)
    records, groups = batch(tmp_path)
    for r in records:
        r["sampler_logprobs"] = [-2.1] * len(r["tokens"])
    with pytest.raises(ProbabilityMismatch, match="Mean"):
        update(runtime, optimizer, scheduler, records, [8] * 16, {}, groups)
    assert not optimizer.state


def test_credit_tampering_fails_before_forward(tmp_path):
    runtime = FakeRuntime()
    optimizer, scheduler = fresh_optimizer(runtime.model)
    records, groups = batch(tmp_path, "LOCAL")
    groups[0]["aux"][0] = 0
    with pytest.raises(ValueError, match="recomputed"):
        update(runtime, optimizer, scheduler, records, [8] * 16, {}, groups)
    assert not runtime.calls


def test_mask_and_token_identity_validation(tmp_path):
    records, _ = batch(tmp_path, "LOCAL")
    with pytest.raises(ValueError, match="mask_ok"):
        changed = copy.deepcopy(records[:8])
        changed[0]["mask_ok"] = False
        build_credit_group(changed, "LOCAL", 17)
    records[0]["tokens"][0] = -1
    groups = [build_credit_group(records[i : i + 8], "LOCAL", i + 17) for i in range(0, 128, 8)]
    runtime = FakeRuntime()
    optimizer, scheduler = fresh_optimizer(runtime.model)
    with pytest.raises(ValueError, match="unsigned"):
        update(runtime, optimizer, scheduler, records, [8] * 16, {}, groups)


def test_channel_gradients_do_not_contaminate_update(tmp_path):
    records, groups = batch(tmp_path, "LOCAL")
    plain, diagnosed = FakeRuntime(), FakeRuntime()
    o1, s1 = fresh_optimizer(plain.model)
    o2, s2 = fresh_optimizer(diagnosed.model)
    result1 = update(plain, o1, s1, records, [8] * 16, {}, groups)
    result2 = update(
        diagnosed, o2, s2, records, [8] * 16, {}, groups, diagnostic_row_indices=tuple(range(8))
    )
    assert torch.equal(plain.model.weight, diagnosed.model.weight)
    assert result1["gradient_norm_before_clip"] == result2["gradient_norm_before_clip"]
    assert result2["channel_gradients"]["status"] == "MEASURED"
    assert len(diagnosed.accounts) == 8
    assert result2["channel_gradients"]["native_auxiliary_cosine"] == pytest.approx(1)


def test_channel_gradients_cannot_run_on_unregistered_step(tmp_path):
    runtime = FakeRuntime()
    runtime.identity["step"] = 1
    optimizer, scheduler = fresh_optimizer(runtime.model)
    records, groups = batch(tmp_path)
    for record in records:
        record["adapter_identity"]["step"] = 1
    with pytest.raises(ValueError, match="registered steps"):
        update(
            runtime,
            optimizer,
            scheduler,
            records,
            [8] * 16,
            {},
            groups,
            diagnostic_row_indices=(0,),
        )
    assert not runtime.calls


def test_stale_rollout_rejected_without_hash_scan(tmp_path):
    runtime = FakeRuntime()
    optimizer, scheduler = fresh_optimizer(runtime.model)
    records, groups = batch(tmp_path)
    records[0]["adapter_identity"]["step"] = 8
    with pytest.raises(ValueError, match="adapter identity"):
        update(runtime, optimizer, scheduler, records, [8] * 16, {}, groups)
    assert not runtime.calls
