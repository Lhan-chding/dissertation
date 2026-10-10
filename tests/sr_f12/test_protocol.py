import copy

import pytest

from sr_f1.amendment import build_amended_config
from sr_f12.protocol import (
    ARMS,
    BASELINE,
    build_config,
    iter_model_slots,
    pilot_learning_rate_decision,
    scientific_matrix,
    validate_config,
)


def tasks():
    rows = []
    for pool, n in (
        ("MONITOR", 512),
        ("TRAIN", 32),
        ("TEST_ID", 2048),
        ("TEST_COMPOSITION", 128),
        ("TEST_LANGUAGE", 128),
        ("TEST_RENDER", 128),
    ):
        for i in range(n):
            rows.append(
                dict(
                    qid=f"{pool}-{i:04}",
                    pool=pool,
                    root_id=f"{pool}-{i // 16}",
                    family=("CROSS", "THRESHOLD", "TOPK", "INTERVAL")[i % 4],
                    chart="line",
                    variant="v0" if pool != "MONITOR" or i < 64 else "v1",
                )
            )
    return rows


def test_config_independent_frozen_contract():
    old = build_amended_config()
    before = copy.deepcopy(old)
    config = build_config()
    assert validate_config(config) == config
    assert old == before
    assert config["training"]["arms"] == list(ARMS)
    assert config["training"]["tis_cap"] == 2
    assert config["statistics"]["primary_bonferroni_CI"] == 0.9875
    for key in ("reward", "data", "external_evaluation"):
        assert config[key] == old[key]
    for key in (
        "updates",
        "seeds",
        "prompts_per_update",
        "samples_per_prompt",
        "kl_beta",
        "ratio_clip",
        "max_new_tokens",
    ):
        assert config["training"][key] == old["training"][key]
    config["training"]["updates"] = 32
    with pytest.raises(PermissionError):
        validate_config(config)


def test_matrix_single_gpu_seed_first():
    matrix = scientific_matrix()
    assert len(matrix) == 12
    assert [r["arm"] for r in matrix[:4]] == list(ARMS)
    assert {r["seed"] for r in matrix[:4]} == {71001}
    assert {r["gpu_count"] for r in matrix} == {1}
    assert len(scientific_matrix(optional_part=True)) == 15


def test_exact_evaluation_layout_and_seal():
    rows = tasks()
    fit = [r["qid"] for r in rows if r["pool"] == "TRAIN"]
    baseline = list(iter_model_slots(rows, fit, BASELINE))
    assert len(baseline) == 6656
    assert not any(r["pool"].startswith("TEST") for r in baseline)
    model = scientific_matrix()[0]["model_id"]
    with pytest.raises(PermissionError):
        list(iter_model_slots(rows, fit, model, stage="final"))
    common = list(iter_model_slots(rows, fit, BASELINE, stage="final", test_released=True))
    final = list(iter_model_slots(rows, fit, model, stage="final", test_released=True))
    assert len(common) == 13824
    assert len(final) == 15104
    assert not any(r["pool"] == "MONITOR" for r in final)
    assert len([r for r in final if r["protocol"] == "answer_only"]) == 4096
    assert not any(r["protocol"] == "answer_only" and r["pool"] != "TEST_ID" for r in final)
    fit_slots = [
        r
        for step in (32, 64, 96)
        for r in iter_model_slots(rows, fit, model, stage="train_fit", step=step)
    ]
    assert len(fit_slots) == 384
    assert not {r["slot_id"] for r in fit_slots} & {r["slot_id"] for r in final}
    # Full schedule includes common fit once, each science fit three times, ChartQA.
    assert 6656 + 13824 + 12 * 15104 + 128 + 12 * 384 + 13 * 2500 == 238964


@pytest.mark.parametrize(
    "kwargs,lr,status",
    [
        ({}, 5e-5, "SELECTED"),
        ({"movement": 0.009}, 1e-4, "RETRY_ONCE"),
        ({"format_failures_last_four": 103}, 2e-5, "RETRY_ONCE"),
        ({"step8_kl": 0.050001}, 2e-5, "RETRY_ONCE"),
        ({"finite": False, "movement": 0.001}, 2e-5, "RETRY_ONCE"),
        ({"finite": False, "adjusted": True, "current_lr": 2e-5}, 2e-5, "STOP_UNSTABLE"),
        ({"adjusted": True, "current_lr": 1e-4, "movement": 0.001}, 1e-4, "SELECTED"),
    ],
)
def test_lr_rule(kwargs, lr, status):
    args = dict(
        finite=True,
        format_failures_last_four=102,
        token_count_last_four=512,
        step8_kl=0.05,
        movement=0.01,
    )
    args.update(kwargs)
    result = pilot_learning_rate_decision(**args)
    assert (result["learning_rate"], result["status"]) == (lr, status)


def test_registration_and_freeze_are_immutable(tmp_path, monkeypatch):
    from sr_f12 import runner
    from sr_f12.protocol import freeze_scientific, object_hash, register_amendment

    config = build_config()
    register_amendment(tmp_path, config, source_commit="a" * 40)
    register_amendment(tmp_path, config, source_commit="a" * 40)
    with pytest.raises(PermissionError):
        register_amendment(tmp_path, config, source_commit="b" * 40)
    monkeypatch.setattr(runner, "validate_technical_receipt", lambda receipt, plan: True)
    tech = dict(status="PASS", config_sha256=object_hash(config))
    predictions = dict(status="PREREGISTERED_BEFORE_SCIENCE")
    freeze = freeze_scientific(
        tmp_path,
        config,
        tech,
        predictions,
        source_commit="a" * 40,
        actual_lora_modules=["model.language_model.layers.0.mlp.down_proj"],
    )
    assert freeze["status"] == "FROZEN_BEFORE_SCIENCE"
    assert (tmp_path / "SCIENCE_FREEZE.json").exists()
    with pytest.raises(PermissionError):
        freeze_scientific(
            tmp_path,
            config,
            {**tech, "config_sha256": "wrong"},
            predictions,
            source_commit="a" * 40,
            actual_lora_modules=["x"],
        )


def test_registration_cannot_adopt_old_outputs(tmp_path):
    from sr_f12.protocol import register_amendment

    (tmp_path / "raw").mkdir()
    (tmp_path / "raw/old.json").write_text("{}")
    with pytest.raises(PermissionError):
        register_amendment(tmp_path, build_config(), source_commit="a" * 40)
