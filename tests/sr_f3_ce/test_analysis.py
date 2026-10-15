import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sr_f3_ce.analysis import contrast, paired_contrast, root_values


def test_bootstrap_matches_uploaded_reference_exactly():
    source = (
        Path(__file__).resolve().parents[2]
        / "artifacts/sr_f3_ce_20261016/plan/reference/statistics_reference.py"
    )
    spec = importlib.util.spec_from_file_location("srf3_reference_statistics", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(10)
    matrix = rng.uniform(-0.5, 0.5, (32, 3))
    family = [f for f in ("CROSS", "INTERVAL", "THRESHOLD", "TOPK") for _ in range(8)]
    expected = module.contrast(matrix, family)
    actual = contrast(matrix, family)
    for key in ("point", "per_block", "CI95", "CI_simultaneous"):
        assert actual[key] == expected[key]
    assert actual["paired_t_sensitivity"]["df"] == 2
    assert actual["bootstrap_replicates"] == 10000
    assert not actual["arbitrary_seed_generalization"]


def test_equal_family_weights_and_heterogeneous_blocks():
    matrix = [[0.3, -0.1, 0.1]] * 2 + [[0, 0, 0]] * 8
    value = contrast(matrix, ["A"] * 2 + ["B"] * 8)
    assert value["point"] == pytest.approx(0.05)
    assert value["per_block"] == pytest.approx([0.15, -0.05, 0.05])
    assert value["signs"] == dict(positive=2, zero=0, negative=1)
    with pytest.raises(ValueError):
        contrast([[1, 2]], ["A"])
    with pytest.raises(ValueError):
        contrast([[1, 1, float("nan")], [0, 0, 0]], ["A", "A"])


def fixture_rows():
    tasks, rows = {}, []
    for family in ("CROSS", "INTERVAL", "THRESHOLD", "TOPK"):
        for root in range(2):
            for chart in ("bar", "line"):
                qid = f"{family}_{root}_{chart}"
                task = dict(
                    qid=qid,
                    root_id=f"{family}_{root}",
                    pool="TEST_ID",
                    family=family,
                    mode="repair",
                    variant=2,
                )
                tasks[qid] = task
                for arm in ("EXEC", "GOLD"):
                    for block in range(3):
                        for draw in range(4):
                            rows.append(
                                dict(
                                    task,
                                    model_id=f"SRF3CE_{arm}_b{block}",
                                    arm=arm,
                                    block=block,
                                    step=64,
                                    protocol="structured",
                                    view=None,
                                    draw=draw,
                                    generation_status="COMPLETE",
                                    score={"value_correct": int(arm == "EXEC" and draw == 0)},
                                )
                            )
    return tasks, rows


def test_root_average_keeps_failed_draws_and_requires_registered_support():
    tasks, rows = fixture_rows()
    result = paired_contrast(rows, tasks, "EXEC", "GOLD", "REPAIR")
    assert result["effect_pp"] == 25
    assert result["per_block_pp"] == [25, 25, 25]
    with pytest.raises(RuntimeError, match="Complete frozen"):
        root_values(rows[1:], tasks, "EXEC", "REPAIR")
    with pytest.raises(ValueError, match="Duplicate"):
        root_values([*rows, rows[0]], tasks, "EXEC", "REPAIR")
    rows[0]["score"] = {}
    with pytest.raises(ValueError, match="Missing/nonbinary"):
        root_values(rows, tasks, "EXEC", "REPAIR")
