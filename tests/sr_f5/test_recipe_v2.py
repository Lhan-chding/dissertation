"""CPU registration guards: no renderer, model, scheduler, or remote access."""

import copy
import json
from pathlib import Path

import pytest

from sr_f5 import recipe_v2 as recipe

REVISION = "a" * 40


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    old, new = tmp_path / "v1", tmp_path / "v2"
    for key, value in (("PROJECT", tmp_path), ("V1_ROOT", old), ("V2_ROOT", new)):
        monkeypatch.setattr(recipe, key, value)
    config = json.loads((Path(__file__).parents[2] / "configs/sr_f5/experiment.json").read_text())
    put(old / "CONFIG.json", config)
    put(
        old / "G3/FROZEN.json",
        dict(
            status="FROZEN_BEFORE_PILOT_OUTPUTS",
            code_revision=recipe.V1_REVISION,
            config_digest=recipe._frozen_digest(config),
        ),
    )
    decision = dict(
        decision="REGISTER_RECIPE_V2",
        recipe_version=2,
        source_revision=recipe.V1_REVISION,
        engineering_mismatch_found=False,
        preserve_v1=True,
        uniform_all_arms=True,
        new_config_digest=recipe.digest(recipe.derive_config(config)),
        reason="Reproducible precision-sensitive discrepancy; all paths require fresh validation",
    )
    put(old / "CONTINUATION/RECIPE_V2_DECISION.json", decision)
    for name in recipe.EVIDENCE:
        item = dict(
            status="DIAGNOSTIC_COMPLETE",
            science_revision=recipe.V1_REVISION,
            run_id="G3-b1-SHUF",
            checkpoint_step=6,
            optimizer_updates=0,
            gate_decision="NOT_AUTHORIZED_BY_DIAGNOSTIC",
        )
        if "PROBABILITY_CACHE_" in name:
            item.update(
                kind="FORCED_TOKEN_REPLAY_ONLY",
                new_scientific_samples=0,
                token_prefix_length=163,
                trials=[
                    dict(
                        name=trial,
                        status="REPLAY_COMPLETE",
                        rows=[{}] * count,
                        raw_logits_score_checks=[dict(raw_logits_scores_equal=True)] * 163,
                    )
                    for trial, count in (("G8", 8), ("single_row3", 1))
                ],
            )
        else:
            item.update(new_samples=0, rows=[{}] * 8)
        put(old / name, item)
    return old, new, config


def test_exact_config_delta_preserves_scientific_contract(setup):
    _old, new, original = setup
    before = copy.deepcopy(original)
    result = recipe.derive_config(original)
    assert original == before
    assert result["numerics"] == recipe.NUMERICS
    for split in ("dev", "pilot"):
        assert result["data"]["seeds"][split] == recipe.seed_for(
            original["master_seed"], "recipe_v2", split
        )
        result["data"]["seeds"][split] = original["data"]["seeds"][split]
    assert result["deployment"]["root"] == str(new)
    for key in ("numerics",):
        result.pop(key)
    for key in ("status", "experiment_id"):
        result[key] = original[key]
    result["deployment"]["root"] = original["deployment"]["root"]
    assert result == original
    with pytest.raises(PermissionError, match="single recipe"):
        recipe.derive_config(recipe.derive_config(original))


def test_registers_before_prepare_and_reuses_only_completed(setup, monkeypatch):
    old, new, original = setup
    calls = []

    def prepare(root, config_path, revision):
        assert root == new
        registration = json.loads((old / "CONTINUATION/RECIPE_V2.json").read_text())
        assert registration["status"] == "REGISTERED_BEFORE_CPU_PREPARATION"
        assert registration["identity"]["new_source_revision"] == revision
        assert registration["identity"]["new_config_digest"] == recipe.digest(
            json.loads(config_path.read_text())
        )
        calls.append(True)
        put(root / "CONFIG.json", recipe.derive_config(original))
        put(root / "CPU_TESTS.json", dict(status="PASS", code_revision=revision))
        result = dict(status="CPU_READY_FOR_G2_PREFLIGHT", code_revision=revision)
        put(root / "CPU_READY.json", result)
        return result

    monkeypatch.setattr(recipe, "prepare", prepare)
    first = recipe.register_and_prepare(REVISION)
    assert recipe.register_and_prepare(REVISION) == first
    assert len(calls) == 1
    assert json.loads((old / "CONFIG.json").read_text()) == original
    with pytest.raises(PermissionError, match="registration differs"):
        recipe.register_and_prepare("b" * 40)


def test_partial_preparation_is_not_blindly_retried(setup, monkeypatch):
    old, new, original = setup

    def crash(root, config_path, revision):
        put(new / "CONFIG.json", recipe.derive_config(original))
        raise RuntimeError("CPU preparation incident")

    monkeypatch.setattr(recipe, "prepare", crash)
    with pytest.raises(RuntimeError, match="incident"):
        recipe.register_and_prepare(REVISION)
    assert (old / "CONTINUATION/RECIPE_V2.json").exists()
    with pytest.raises(RuntimeError, match="incomplete or unknown"):
        recipe.register_and_prepare(REVISION)


@pytest.mark.parametrize("mutation", ["decision", "diagnosis", "frozen", "scores"])
def test_missing_decision_or_inconsistent_evidence_blocks_registration(setup, mutation):
    old, _new, _config = setup
    if mutation == "decision":
        path = old / "CONTINUATION/RECIPE_V2_DECISION.json"
        value = json.loads(path.read_text())
        value["decision"] = "DIAGNOSTIC_COMPLETE"
    elif mutation == "diagnosis":
        path = old / recipe.EVIDENCE[0]
        value = json.loads(path.read_text())
        value["status"] = "RUNNING"
    elif mutation == "frozen":
        path = old / "G3/FROZEN.json"
        value = json.loads(path.read_text())
        value["config_digest"] = "changed"
    else:
        path = old / recipe.EVIDENCE[2]
        value = json.loads(path.read_text())
        value["trials"][0]["raw_logits_score_checks"][0]["raw_logits_scores_equal"] = False
    put(path, value)
    with pytest.raises(PermissionError):
        recipe.register_and_prepare(REVISION)
    assert not (old / "CONTINUATION/RECIPE_V2.json").exists()


def test_existing_unregistered_output_or_symlink_is_protected(setup):
    old, new, _config = setup
    put(new / "G3/existing.json", {"keep": True})
    with pytest.raises(PermissionError, match="already contains"):
        recipe.register_and_prepare(REVISION)
    assert json.loads((new / "G3/existing.json").read_text()) == {"keep": True}
    path = old / recipe.EVIDENCE[0]
    saved = path.with_name("saved.json")
    path.rename(saved)
    path.symlink_to(saved)
    with pytest.raises(PermissionError, match="canonical file"):
        recipe.register_and_prepare(REVISION)
