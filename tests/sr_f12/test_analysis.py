import copy

import pytest

from sr_f12.analysis import FAMILIES, paired_root_ci, preregister_predictions, semantic_state_table
from sr_f12.protocol import BASELINE


def baseline():
    rows = []
    views = (None, "keys_hint", "gold_values", "text_values", "neutral_hint", "blank_image")
    for family in FAMILIES:
        for question in range(128):
            for view in views if question < 16 else (None,):
                for draw in range(8):
                    e = int(draw >= 4)
                    p = float(e)
                    a = int(draw % 2 == 0)
                    rows.append(
                        dict(
                            model_id=BASELINE,
                            step=0,
                            protocol="evidence_answer",
                            family=family,
                            pool="MONITOR" if view is None else "MONITOR_DIAGNOSTIC_ONLY",
                            view=view,
                            qid=f"{family}-{question}",
                            root_id=f"{family}-{question // 16}",
                            variant="v0" if question < 16 else "v1",
                            draw=draw,
                            independent_score=dict(A=a, E=e, p_read=p, P=e, J=e * a),
                        )
                    )
    return rows


def test_exact_state_and_preregistered_ties():
    state = semantic_state_table(baseline())
    assert state["monitor_samples"] == 4096
    assert state["diagnostic_samples"] == 2560
    f = state["families"][0]
    assert f["D"]["probability"] == 0.5
    assert f["s_A"] == f["s_J"] == 1
    assert f["s_G"] == 0
    pred = preregister_predictions(state, {f: 384 for f in FAMILIES})
    assert pred["P1"]["strata"] == dict(
        CROSS="low", INTERVAL="low", THRESHOLD="middle", TOPK="high"
    )
    assert set(pred["P3"]["strata"].values()) == {"computation_bottleneck"}
    assert pred["mechanism"]["expected_pure_semantic_branch_rate"] == 0


def test_gating_readings_evidence_and_missing_support():
    rows = baseline()
    for row in rows:
        row["independent_score"].update(A=0, J=0)
        if row["family"] == "CROSS":
            row["independent_score"].update(E=1, p_read=row["draw"] / 8, P=0)
    state = semantic_state_table(rows)
    cross = next(f for f in state["families"] if f["family"] == "CROSS")
    assert cross["s_G"] == 1
    assert cross["D"]["denominator"] == 0
    assert cross["D"]["probability"] is None
    pred = preregister_predictions(state, {f: 384 for f in FAMILIES})
    assert pred["P3"]["strata"]["CROSS"] == "UNKNOWN_NO_P_SUPPORT"


@pytest.mark.parametrize("change", ["duplicate", "missing", "old", "test", "bad_score"])
def test_reject_unusable_baseline(change):
    rows = baseline()
    if change == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    if change == "missing":
        rows.pop()
    if change == "old":
        rows[0]["model_id"] = "SRF1_COMMON_START"
    if change == "test":
        rows[0]["pool"] = "TEST_ID"
    if change == "bad_score":
        rows[0]["independent_score"]["J"] = 1
    with pytest.raises((ValueError, PermissionError)):
        semantic_state_table(rows)


def test_bootstrap_four_comparison_family():
    result = paired_root_ci([[0.4] * 4] * 3, [[0.2] * 4] * 3, FAMILIES)
    assert result["CI98_75_pp"] == pytest.approx([20, 20])
    assert result["effect_pp"] == pytest.approx(20)
    assert len(result["primary_contrasts"]) == 4


def endpoint_rows():
    from sr_f12.protocol import scientific_matrix

    models = [BASELINE] + [r["model_id"] for r in scientific_matrix()]
    rows = []
    for model in models:
        for family in FAMILIES:
            for root in range(32):
                for q in range(16):
                    for draw in range(4):
                        # J-A difference only for alphabetically last/high tertile TOPK.
                        success = int(family == "TOPK" and ("_J_" in model or "_GATE_" in model))
                        rows.append(
                            dict(
                                model_id=model,
                                step=0 if model == BASELINE else 96,
                                pool="TEST_ID",
                                protocol="evidence_answer",
                                view=None,
                                root_id=f"{family}-{root}",
                                qid=f"{family}-{root}-{q}",
                                draw=draw,
                                family=family,
                                independent_score=dict(A=success, J=success, E=1, P=1, p_read=1),
                            )
                        )
    return rows


def test_predictions_root_analysis_requires_release_and_retains_undefined():
    from sr_f12.analysis import evaluate_predictions, primary_contrasts

    state = semantic_state_table(baseline())
    pred = preregister_predictions(state, {f: 384 for f in FAMILIES})
    rows = endpoint_rows()
    with pytest.raises(PermissionError):
        evaluate_predictions(rows, pred)
    results = evaluate_predictions(rows, pred, released=True)
    assert results[0]["effect_pp"] == 100
    assert results[0]["CI95_pp"] == [100, 100]
    assert results[1]["effect_pp"] == 0
    assert results[2]["status"] == "UNDEFINED_EMPTY_PREDICTED_STRATUM"
    contrasts = primary_contrasts(rows, released=True)
    assert contrasts[0]["effect_pp"] == 25
    assert len(contrasts) == 4


def test_mechanism_is_secondary_and_requires_all_groups():
    from sr_f12.analysis import mechanism_check
    from sr_f12.protocol import SEEDS

    pred = preregister_predictions(semantic_state_table(baseline()), {f: 384 for f in FAMILIES})
    rows = [
        dict(arm="GATE", seed=s, step=i, prompt_index=q, branch="target_J_refined")
        for s in SEEDS
        for i in range(1, 9)
        for q in range(16)
    ]
    result = mechanism_check(rows, pred)
    assert result["within_registered_ten_pp"]
    assert not result["training_gate"]
    with pytest.raises(ValueError):
        mechanism_check(rows[:-1], pred)
