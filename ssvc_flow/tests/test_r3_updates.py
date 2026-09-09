"""Real tiny-torch fork orchestration; these tests are not CUDA evidence."""

import importlib
import json

import pytest
import torch
from test_fork_gradients import TinyAdapter, make_groups

from src.optimizer_fork import capture_state, load_checkpoint, state_hash


def setup(categories=("X", "S", "W", "I"), warm=False):
    adapter = TinyAdapter()
    optimizer = torch.optim.AdamW(
        [p for p in adapter.model.parameters() if p.requires_grad],
        lr=1e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    )
    if warm:
        sum(
            p.square().sum() + p.sum() for p in adapter.model.parameters() if p.requires_grad
        ).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    groups = make_groups(adapter, categories)
    origin = capture_state(adapter.model, optimizer, {"sampler": {"cursor": 17}})
    return adapter, optimizer, groups, origin


def test_api_available():
    module = importlib.import_module("src.r3_updates")
    assert callable(module.validate_reuse_bank)
    assert callable(module.run_bank_forks)


@pytest.mark.parametrize("warm", [False, True])
def test_validation_compares_actual_adam_and_restores_origin(warm):
    from src.r3_updates import validate_reuse_bank

    adapter, optimizer, groups, origin = setup(warm=warm)
    before = state_hash(origin)
    result = validate_reuse_bank(adapter, optimizer, origin, groups)
    assert result["adoptable"]
    assert result["validation_optimizer_updates"] == 4
    assert set(result["checks"]) == {"0.0", "1.0"}
    assert all(v["gradient_comparison"]["informative"] for v in result["checks"].values())
    assert state_hash(capture_state(adapter.model, optimizer, origin["metadata"])) == before
    assert state_hash(origin) == before


@pytest.mark.parametrize("reuse", [False, True])
def test_five_independent_candidates_and_lambda_zero_replay(tmp_path, reuse):
    from src.r3_updates import run_bank_forks, validate_reuse_bank

    adapter, optimizer, groups, origin = setup()
    score = validate_reuse_bank(adapter, optimizer, origin, groups)["score_bank"] if reuse else None
    result = run_bank_forks(
        adapter,
        optimizer,
        origin,
        groups,
        tmp_path / "bank",
        {"fixture": True, "bank_index": 6},
        reuse_authorized=reuse,
        score_bank=score,
    )
    assert result["status"] == "PASS"
    assert [c["lambda"] for c in result["candidates"]] == [0, 0.01, 0.25, 1, 2]
    assert result["candidate_optimizer_updates"] == 5
    assert result["replay_optimizer_updates"] == 1
    assert result["lambda_zero_replay"]["passed"]
    assert result["reuse_adopted"] == reuse
    assert set(result["answer_diagnostics"]) == {"A_BASE", "A_VALID"}
    for candidate in result["candidates"]:
        assert candidate["candidate_id"].startswith("bank_06_lambda_")
        checkpoint = load_checkpoint(candidate["checkpoint_path"], candidate["checkpoint_identity"])
        assert state_hash(checkpoint["optimizer"]) == candidate["optimizer_state_hash"]
        assert candidate["sgd_reference"]["probability_response"] == "NOT_MEASURED"
    assert state_hash(capture_state(adapter.model, optimizer, origin["metadata"])) == state_hash(
        origin
    )
    manifest = json.loads((tmp_path / "bank/manifest.json").read_text())
    assert any(r["path"].endswith(".pt") for r in manifest["files"])


def test_mature_adam_zero_reward_still_steps_and_all_candidates_agree(tmp_path):
    from src.r3_updates import run_bank_forks, validate_reuse_bank

    adapter, optimizer, groups, origin = setup(("I", "I", "I", "I"), warm=True)
    check = validate_reuse_bank(adapter, optimizer, origin, groups)
    assert not check["adoptable"]
    assert all(c["gradient_comparison"]["zero_consistent"] for c in check["checks"].values())
    result = run_bank_forks(adapter, optimizer, origin, groups, tmp_path / "bank", {})
    hashes = {c["parameter_hash"] for c in result["candidates"]}
    assert len(hashes) == 1
    assert all(c["update"]["actual_step_norm"] > 0 for c in result["candidates"])
    assert all(c["sgd_reference"]["absolute_delta_l2"] == 0 for c in result["candidates"])
    saved = load_checkpoint(
        result["candidates"][0]["checkpoint_path"], result["candidates"][0]["checkpoint_identity"]
    )
    assert all(float(s["step"]) == 2 for s in saved["optimizer"]["state"].values())


def test_old_probability_drift_forces_direct_fallback(tmp_path):
    from src.r3_updates import run_bank_forks, validate_reuse_bank

    adapter, optimizer, groups, origin = setup()
    groups[0][0]["old_logprobs"][0] += 0.1
    check = validate_reuse_bank(adapter, optimizer, origin, groups)
    assert not check["adoptable"]
    result = run_bank_forks(
        adapter,
        optimizer,
        origin,
        groups,
        tmp_path / "bank",
        {},
        reuse_authorized=True,
        score_bank=check["score_bank"],
    )
    assert not result["reuse_adopted"]
    assert all(c["gradient"]["backward_calls"] == 8 for c in result["candidates"])


def test_all_valid_bank_measures_invariant_and_missing_categories(tmp_path):
    from src.r3_updates import run_bank_forks

    adapter, optimizer, groups, origin = setup(("X", "S", "W", "X"))
    result = run_bank_forks(adapter, optimizer, origin, groups, tmp_path / "bank", {})
    assert result["all_valid_invariant"]["applicable"]
    assert result["all_valid_invariant"]["passed"]
    assert result["group_composition"]["validity_mixed_fraction"] == 0
    assert "I" in result["group_composition"]["prompts"][0]["missing_categories"]


def test_existing_bank_cannot_be_overwritten(tmp_path):
    from src.r3_updates import run_bank_forks

    adapter, optimizer, groups, origin = setup()
    out = tmp_path / "bank"
    out.mkdir()
    (out / "sentinel.json").write_text("{}")
    with pytest.raises(FileExistsError):
        run_bank_forks(adapter, optimizer, origin, groups, out, {})
    assert (out / "sentinel.json").read_text() == "{}"


def test_failure_after_mutation_restores_all_state(tmp_path, monkeypatch):
    import src.r3_updates as runtime

    adapter, optimizer, groups, origin = setup(warm=True)
    real = runtime.apply_gradient_update

    def fail(*args, **kwargs):
        real(*args, **kwargs)
        raise RuntimeError("injected after Adam")

    monkeypatch.setattr(runtime, "apply_gradient_update", fail)
    with pytest.raises(RuntimeError, match="injected"):
        runtime.run_bank_forks(adapter, optimizer, origin, groups, tmp_path / "bank", {})
    assert state_hash(capture_state(adapter.model, optimizer, origin["metadata"])) == state_hash(
        origin
    )


@pytest.mark.parametrize(
    "setting,value", [("lr", 0.02), ("weight_decay", 0.1), ("betas", (0.8, 0.999))]
)
def test_bad_origin_hyperparameters_are_rejected_before_restore(setting, value):
    from src.r3_updates import validate_reuse_bank

    adapter, optimizer, groups, origin = setup(warm=True)
    before = state_hash(capture_state(adapter.model, optimizer, origin["metadata"]))
    origin["optimizer"]["param_groups"][0][setting] = value
    with pytest.raises(ValueError, match="optimizer"):
        validate_reuse_bank(adapter, optimizer, origin, groups)
    assert state_hash(capture_state(adapter.model, optimizer, origin["metadata"])) == before
