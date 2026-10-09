import numpy as np
import pytest

from sr_f1.analysis import FAMILIES, conditional_rate, monitor_credit_audit, paired_root_ci


def test_whole_root_stratified_interval_retains_seed_directions():
    y = np.zeros((3, 8))
    x = np.array([[0.1] * 8, [-0.05] * 8, [0.25] * 8])
    result = paired_root_ci(x, y, list(FAMILIES) * 2)
    assert result["seed_effects_pp"] == pytest.approx([10, -5, 25])
    assert result["effect_pp"] == pytest.approx(10)
    assert result["CI95_pp"] == pytest.approx([10, 10])
    assert result["CI99_pp"] == pytest.approx([10, 10])
    assert result["scope"].startswith("root measurement")


def test_missing_technical_results_and_new_bootstrap_seed_are_rejected():
    x = np.zeros((3, 4))
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="Technical missing"):
        paired_root_ci(x, np.zeros((3, 4)), FAMILIES)
    with pytest.raises(ValueError, match="frozen"):
        paired_root_ci(np.zeros((3, 4)), np.zeros((3, 4)), FAMILIES, seed=1)
    assert conditional_rate(0, 0)["probability"] is None


def test_monitor_dec_uses_full_batch_with_constant_groups_retained():
    rows = []
    for q in range(16):
        for draw in range(8):
            j = int(q == 0 and draw < 4)
            e = int(q < 2 and draw < 6)
            rows.append(
                dict(
                    model_id="model",
                    pool="MONITOR",
                    monitor_block=0,
                    qid=f"{q:02d}",
                    draw=draw,
                    independent_score=dict(A=j, J=j, E=e, p_read=float(e)),
                )
            )
    result = monitor_credit_audit(rows)[0]
    dec = np.array(result["arms"]["DEC"]["final_advantage"])
    assert dec.shape == (16, 8)
    assert np.std(dec) == pytest.approx(1, abs=1e-7)
    assert np.allclose(dec[2:], 0, atol=1e-15)
    assert result["arms"]["GATE"]["branches"][2:] == ["no_semantic_contrast"] * 14
    with pytest.raises(ValueError, match="complete fixed"):
        monitor_credit_audit(rows[:-1])
