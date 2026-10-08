"""Synthetic-only SER-J23 measurement contracts; no E_CONFIRM2 data is accessed."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from ssvc_flow.src.exposure_position.semantics import (
    aggregate_events,
    deletion_projection,
    parse,
    score_output,
    validate_events,
)
from ssvc_flow.src.exposure_position.statistics import (
    ARMS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    FOCUS_CELLS,
    METRICS,
    PHASE_ID,
    IncompleteDataError,
    PairedRootBootstrap,
    _analyze_complete,
    _complete_confirmation,
    _model_keys,
    analyze_confirmation,
    cross_macro,
    summarize_tasks,
    trend_macro,
)


def synthetic_case(*, center=3, j=3, operation="sum4"):
    truth = [10, 20, 30, 40]
    observed = list(truth)
    observed[j] += 5
    relationships, rhs = [], []
    for a, b in sorted(tuple(sorted((center, k))) for k in range(4) if k != center):
        row = [0] * 4
        row[a] = row[b] = 1
        relationships.append(row)
        rhs.append(truth[a] + truth[b])
    public = dict(
        task_id="synthetic-task",
        root_id="synthetic-root",
        family="cross_series",
        observed=observed,
        H_original=relationships,
        b_original=rhs,
        operation=operation,
        legal_domain=[0, 99],
    )
    audit = dict(
        task_id=public["task_id"],
        root_id=public["root_id"],
        true_world=truth,
        center=center,
        corrupted_index=j,
        root_cohort="CORE32",
    )
    return public, audit


def row_for(raw="[10,20,30,40]", *, j=3, draw=0, **overrides):
    public, audit = synthetic_case(j=j)
    return dict(
        sample_id=f"sample-{draw}",
        phase_id=PHASE_ID,
        checkpoint_id="synthetic-model",
        parent="S96",
        block=0,
        logical_arm_id=ARMS[0],
        step=256,
        panel="E_CONFIRM2",
        draw_index=draw,
        **score_output(public, audit, raw),
        **overrides,
    )


@pytest.mark.parametrize(
    "raw",
    [
        "[true,20,30,40]",
        "[10.0,20,30,40]",
        "[1e1,20,30,40]",
        "[10,20,30]",
        "[10,20,30,40] trailing",
        "prefix [10,20,30,40]",
        "null",
        "[NaN,20,30,40]",
    ],
)
def test_exact_whole_string_json_rejects_bool_float_and_malformed(raw):
    assert parse(raw) is None
    row = row_for(raw)
    assert row["I"] and not row["parse_ok"]
    assert not any(row["q_events"])


def test_native_operations_partition_and_invariants():
    for raw, event in (
        ("[10,20,30,40]", "X"),
        ("[11,19,30,40]", "S"),
        ("[10,20,30,41]", "W"),
        ("[-1,20,30,40]", "I"),
        ("oops", "I"),
    ):
        row = row_for(raw)
        assert row["event"] == event
        assert sum(row[x] for x in "XSWI") == 1
        assert int(row["X"]) == int(row["F_val"]) - int(row["R"])
        assert row["projection_success"] == row["F_val"]
    for operation, raw in (("difference_pairs", "[11,20,31,40]"), ("range4", "[10,21,30,40]")):
        public, audit = synthetic_case(operation=operation)
        assert score_output(public, audit, raw)["S"]
    public, audit = synthetic_case(operation="not-an-operation")
    for raw in ("malformed", "[10,20,30,40]"):
        with pytest.raises(ValueError, match="operation"):
            score_output(public, audit, raw)


def test_out_of_domain_F_val_and_multi_edit_are_retained():
    row = row_for("[-100,20,30,40]")
    assert row["I"] and row["out_of_domain"] and row["parse_ok"]
    assert row["F_val"] and row["R"] and row["F_legal"] is None
    assert "F" not in row and "value_present" not in row
    assert row["out_of_domain_bits"] == [True, False, False, False]
    assert row["multi_edit_parse"] and not row["multi_edit_legal"]
    assert row["q_events"] == row["r_events"] == [True, False, False, True]


def test_native_projection_is_public_and_only_uses_existing_values():
    public, audit = synthetic_case()

    class PublicOnly(dict):
        def __getitem__(self, key):
            assert key not in ("true_world", "corrupted_index", "center")
            return super().__getitem__(key)

    before = copy.deepcopy(public)
    projected = deletion_projection(PublicOnly(public), [-100, 200, 30, 40])
    assert projected["projected_vector"] == audit["true_world"]
    assert projected["projection_verifier_calls"] == 4
    assert not deletion_projection(public, [10, 20, 30, 41])["projection_success"]
    assert public == before
    audit["true_world"] = tuple(audit["true_world"])
    assert score_output(public, audit, "[10,20,30,40]")["X"]
    assert isinstance(audit["true_world"], tuple)


def test_nonunique_task_is_a_contract_failure_even_for_correct_looking_answer():
    public, audit = synthetic_case()
    public["family"] = "trend"
    public["H_original"] = [[1, 1, 0, 0]]
    public["b_original"] = [30]
    audit["center"] = None
    with pytest.raises(ValueError, match="Unique"):
        score_output(public, audit, "[10,20,30,40]")


def test_all_answer_q_r_c_denominators_and_zero_denominator_null():
    rows = [row_for(raw) for raw in ("[10,20,30,40]", "[-100,20,30,40]", "oops", "[10,20,30,45]")]
    result = aggregate_events(rows)
    assert result["denominator_all_answers"] == 4
    assert result["X"] == 0.25 and result["F_val"] == 0.5 and result["R"] == 0.25
    assert result["q"] == [0.25, 0, 0, 0.5]
    assert result["r"] == [0.25, 0, 0, 0.25]
    assert result["c"] == [1 / 3, 0, 0, 1 / 3]
    assert result["F_legal"] == 0.25
    assert result["copy_observed"] == 0.25
    assert result["out_of_domain_by_position"] == [0.25, 0, 0, 0]
    perfect = aggregate_events([row_for()])
    assert perfect["c"] == [None] * 4
    assert perfect["c_undefined_reason"] == "no_non_X_answers"
    json.dumps(perfect, allow_nan=False)


def test_truncation_is_recorded_without_replacing_a_valid_response():
    public, audit = synthetic_case()
    row = score_output(public, audit, "[10,20,30,40]", stop_reason="max_new_tokens")
    assert row["X"] and row["truncated"]


@pytest.mark.parametrize(
    "field, value",
    [
        ("X", "false"),
        ("q_events", [False] * 4),
        ("r_events", [True] * 4),
        ("F_val", False),
        ("F_legal", None),
        ("projection_success", False),
    ],
)
def test_corrupted_scoring_record_fails_closed(field, value):
    row = row_for()
    row[field] = value
    with pytest.raises(ValueError):
        validate_events(row)


def test_task_averaging_keeps_all_draws_and_rejects_identity_gaps():
    rows = [row_for(draw=i) for i in range(8)]
    before = copy.deepcopy(rows)
    summary = summarize_tasks(rows)
    assert len(summary) == 1 and summary[0]["X"] == 1
    assert rows == before
    with pytest.raises(IncompleteDataError, match="complete"):
        summarize_tasks(rows[:-1])
    with pytest.raises(ValueError, match="sample_id"):
        summarize_tasks([*rows, rows[0]])
    duplicate_draw = {**rows[0], "sample_id": "other"}
    with pytest.raises(ValueError, match="checkpoint/task/draw"):
        summarize_tasks([*rows, duplicate_draw])
    with pytest.raises(IncompleteDataError, match="index"):
        summarize_tasks([{**rows[0], "draw_index": True}, *rows[1:]])
    with pytest.raises(ValueError, match="metadata"):
        summarize_tasks([{**rows[0], "root_cohort": "EXTRA96"}, *rows[1:]])


def synthetic_registry():
    """Identity/metadata fixtures only: no tasks or model outputs are generated."""
    rows = []
    for cohort, count in (("CORE32", 32), ("EXTRA96", 96)):
        for root in range(count):
            for center, j in (
                ((c, j) for c in range(4) for j in range(4)) if cohort == "CORE32" else FOCUS_CELLS
            ):
                rows.append(
                    dict(
                        task_id=f"synthetic-{cohort}-{root:03d}-{center}-{j}",
                        root_id=f"synthetic-{cohort}-{root:03d}",
                        root_cohort=cohort,
                        family="cross_series",
                        center=center,
                        corrupted_index=j,
                    )
                )
    for family, count in (("trend", 16), ("duplicate_encoding", 8)):
        for j in range(4):
            for root in range(count):
                identity = f"synthetic-{family}-{j}-{root:03d}"
                rows.append(
                    dict(
                        task_id=identity,
                        root_id=identity,
                        root_cohort=family,
                        family=family,
                        center=None,
                        corrupted_index=j,
                    )
                )
    assert len(rows) == 992
    return rows


def synthetic_task_means():
    """Complete synthetic task-mean matrix with intentionally known effects."""
    result = []
    for parent, block, arm in _model_keys():
        for task in synthetic_registry():
            x = float(
                task["family"] == "cross_series"
                and (task["center"], task["corrupted_index"]) in FOCUS_CELLS
            )
            row = dict.fromkeys(METRICS, 0.0)
            row.update(
                task,
                checkpoint_id=f"{parent}-{block}-{arm}",
                parent=parent,
                block=block,
                logical_arm_id=arm,
                step=0 if arm == "PARENT" else 256,
                panel="E_CONFIRM2",
                phase_id=PHASE_ID,
                n_draws=8,
                draw_indices=list(range(8)),
                X=x,
                F_val=x,
                W=1 - x,
                parse_ok=1.0,
                domain_ok=1.0,
                F_legal=x,
                copy_observed=1 - x,
            )
            row[f"q{task['corrupted_index']}"] = x
            result.append(row)
    return result


@pytest.fixture(scope="module")
def full_synthetic_matrix():
    return synthetic_task_means()


def test_complete_26_model_matrix_and_four_focus_cell_coverage(full_synthetic_matrix):
    groups, cells, strata = _complete_confirmation(full_synthetic_matrix, synthetic_registry())
    assert len(groups) == 26 and len(cells) == 24
    assert [len(roots) for roots in strata.values()] == [32, 96, 16, 16, 16, 16, 8, 8, 8, 8]
    assert all(len(cells["cross_series", c, j]) == 128 for c, j in FOCUS_CELLS)
    assert len(cells["cross_series", 0, 0]) == 32
    assert sum(key[1] is None for key in groups) == 2


@pytest.mark.parametrize(
    "mutation, error",
    [
        ("missing", IncompleteDataError),
        ("duplicate", ValueError),
        ("wrong_draw", IncompleteDataError),
        ("old_panel", ValueError),
        ("old_phase", ValueError),
        ("mixed_checkpoint", ValueError),
        ("mixed_source_hash", ValueError),
        ("wrong_cohort", ValueError),
    ],
)
def test_complete_matrix_rejects_missing_duplicated_mixed_or_old_answers(
    full_synthetic_matrix, mutation, error
):
    rows = list(full_synthetic_matrix)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[0])
    else:
        field, value = {
            "wrong_draw": ("draw_indices", list(range(7))),
            "old_panel": ("panel", "E_CONFIRM"),
            "old_phase": ("phase_id", "SER_J2_20261007"),
            "mixed_checkpoint": ("checkpoint_id", "different-checkpoint"),
            "mixed_source_hash": ("checkpoint_sha256", "a" * 64),
            "wrong_cohort": ("root_cohort", "EXTRA96"),
        }[mutation]
        rows[0] = {**rows[0], field: value}
    with pytest.raises(error):
        _complete_confirmation(rows)


def test_registry_binding_rejects_same_count_replacement(full_synthetic_matrix):
    registry = synthetic_registry()
    registry[0] = {**registry[0], "task_id": "replacement-task"}
    with pytest.raises(ValueError, match="registry"):
        _complete_confirmation(full_synthetic_matrix, registry)


def test_exact_pcg64_draw_order_stratification_and_shared_indices():
    strata = {
        "CORE32": tuple(f"c{i}" for i in range(32)),
        "EXTRA96": tuple(f"e{i}" for i in range(96)),
    }
    plan = PairedRootBootstrap.build(strata)
    assert plan.replicates == 10000 and plan.seed == 2026100809
    assert not plan.counts.flags.writeable
    expected = np.empty_like(plan.counts)
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    for r in range(10000):
        expected[r, :32] = np.bincount(rng.choice(32, size=32, replace=True), minlength=32)
        expected[r, 32:] = np.bincount(rng.choice(96, size=96, replace=True), minlength=96)
    np.testing.assert_array_equal(plan.counts, expected)
    values = np.r_[np.ones(32), np.zeros(96)]
    np.testing.assert_array_equal(plan.mean(values, plan.roots), np.full(10000, 0.25))
    roots = np.arange(128) / 127
    samples = plan.mean(np.stack([roots, 2 * roots, -roots]), plan.roots)
    np.testing.assert_allclose(samples[:, 1], 2 * samples[:, 0], atol=1e-15)
    np.testing.assert_allclose(samples[:, 2], -samples[:, 0], atol=1e-15)
    with pytest.raises(ValueError, match="complete frozen"):
        plan.mean(roots[:127], plan.roots[:127])


def test_stratified_non_cross_positions_cannot_disappear(full_synthetic_matrix):
    _, _, strata = _complete_confirmation(full_synthetic_matrix)
    plan = PairedRootBootstrap.build(strata, replicates=128)
    root_index = {root: i for i, root in enumerate(plan.roots)}
    for name, roots in strata.items():
        counts = plan.counts[:, [root_index[root] for root in roots]]
        assert np.all(counts.sum(axis=1) == len(roots)), name
    assert plan.sha256 == PairedRootBootstrap.build(strata, replicates=128).sha256


def test_formal_entry_point_cannot_change_frozen_replicates_or_seed():
    for kwargs in ({"bootstrap_replicates": 32}, {"seed": 0}, {"expected_draws": 4}):
        with pytest.raises(ValueError, match="Frozen"):
            analyze_confirmation([], **kwargs)
    assert BOOTSTRAP_REPLICATES == 10000


def test_fixed_macro_weights_do_not_follow_992_task_counts():
    cells = {(c, j): float((c, j) in FOCUS_CELLS) for c in range(4) for j in range(4)}
    assert cross_macro(cells) == 0.25
    assert cross_macro(cells) != 512 / 896
    assert trend_macro({j: float(j == 0) for j in range(4)}) == 0.25
    with pytest.raises(ValueError):
        cross_macro({key: value for key, value in cells.items() if key[1] != 0})


def test_full_analysis_macro_and_exact_endpoint_family_rules(full_synthetic_matrix):
    # All resampling paths are exercised using fewer replicates in this internal
    # synthetic test only. The public confirmation API prohibits this override.
    groups, cells, strata = _complete_confirmation(full_synthetic_matrix)
    for key, tasks in groups.items():
        _parent, _block, arm = key
        for task in tasks.values():
            if task["family"] != "cross_series":
                continue
            c, j = task["center"], task["corrupted_index"]
            if (c, j) == (3, 3):
                # Parent=1, A3=.75, B3=.5; all path outcomes fixed here.
                task = dict(task)
                task["X"] = {
                    ARMS[0]: 1.0,
                    ARMS[1]: 0.75,
                    ARMS[2]: 0.75,
                    ARMS[3]: 0.5,
                    "PARENT": 1.0,
                }[arm]
                task["F_val"] = task["X"]
                task["W"] = 1 - task["X"]
                task["F_legal"] = task["X"]
                task["q3"] = task["X"]
                task["q1"] = 0.25 if arm == ARMS[1] else 0.0
                task["q2"] = 0.5 if arm == ARMS[3] else 0.0
                task["r1"], task["r2"] = task["q1"], task["q2"]
                task["copy_observed"] = 1 - task["X"] - task["q1"] - task["q2"]
                tasks[task["task_id"]] = task
    result = _analyze_complete(groups, cells, strata, replicates=32, seed=BOOTSTRAP_SEED)
    assert result["primary"]["estimate"] == -0.25
    assert result["primary"]["harm_supported"]
    assert result["parent_contrasts"][f"{ARMS[3]}-PARENT"]["estimate"] == -0.5
    assert result["H2"]["estimate"] == -0.25
    assert result["secondary_family"]["endpoints"]["Psi_E"]["estimate"] == 0.75
    assert set(result["secondary_family"]["endpoints"]) == {"Psi_E", "G_3_2", "G_3_3"}
    for detail in result["secondary_family"]["endpoints"].values():
        assert "ci98_333333_bonferroni_three" in detail
        assert len(detail["per_parent_order"]) == 2
        assert all(len(row) == 3 for row in detail["per_parent_order"])
    for detail in result["noninferiority_family"]["endpoints"].values():
        assert detail["noninferiority_established"] and detail["margin_probability"] == 0.03
        assert detail["noninferiority_lower_threshold"] == -0.03
    for row in result["macro"]["metrics"]:
        if row["logical_arm_id"] == "PARENT":
            assert (
                row["X"]
                == {
                    "cross_macro": 0.25,
                    "trend_macro": 0.0,
                    "duplicate": 0.0,
                    "cross_trend_50_50": 0.125,
                }[row["macro"]]
            )
    assert result["scope"]["parent_generation_streams"] == 2
    assert result["bootstrap"]["all_models_metrics_and_endpoints_share_indices"]
    assert not result["bootstrap"]["resample_training_blocks"]
    assert not result["bootstrap"]["resample_draws"]
    json.dumps(result, allow_nan=False)


def test_full_formal_synthetic_stream_shared_intervals_and_adverse_results():
    """Exercise the real 206,336-row/10,000-replicate API with artificial events."""
    registry = synthetic_registry()
    templates = {}
    for j in range(4):
        public, audit = synthetic_case(j=j)
        for success in (False, True):
            raw = json.dumps(audit["true_world"] if success else public["observed"])
            templates[j, success] = score_output(public, audit, raw)

    def records():
        for parent, block, arm in _model_keys():
            checkpoint = f"synthetic-{parent}-{block}-{arm}"
            for task in registry:
                family, center, j = task["family"], task["center"], task["corrupted_index"]
                for draw in range(8):
                    success = family == "cross_series" and (center, j) in FOCUS_CELLS
                    if family == "cross_series" and arm == ARMS[3]:
                        root_number = int(task["root_id"].rsplit("-", 1)[1])
                        if (center, j) == (3, 3):
                            threshold = (4 if parent == "S96" else 6) - block
                            success = root_number % 2 == 0 and draw < threshold
                        elif (center, j) == (3, 1):
                            success = root_number % 3 == 0
                        elif (center, j) == (2, 2):
                            success = False
                    yield {
                        **templates[j, success],
                        **task,
                        "sample_id": f"{checkpoint}/{task['task_id']}/{draw}",
                        "phase_id": PHASE_ID,
                        "checkpoint_id": checkpoint,
                        "parent": parent,
                        "block": block,
                        "logical_arm_id": arm,
                        "step": 0 if arm == "PARENT" else 256,
                        "panel": "E_CONFIRM2",
                        "draw_index": draw,
                    }

    result = analyze_confirmation(records(), expected_tasks=registry)
    assert result["frozen_task_registry_bound"]
    assert sum(row["n_draws"] for row in result["task_metrics"]) == 206336
    assert result["primary"]["estimate"] == -0.75
    assert result["primary"]["per_parent"] == [-0.8125, -0.6875]
    assert result["primary"]["all_six_range"] == [-0.875, -0.625]
    assert result["primary"]["ci95_root_conditional"][0] < -0.75
    assert result["primary"]["ci95_root_conditional"][1] > -0.75
    assert not result["secondary_family"]["endpoints"]["G_3_2"]["direction_supported"]
    n3 = result["noninferiority_family"]["endpoints"]["N3_c3j3"]
    assert not n3["noninferiority_established"]
    assert n3["interpretation"] == "noninferiority_not_established"

    # An independent direct-count calculation uses the exact shared root plan.
    strata = result["bootstrap"]["strata"]
    plan = PairedRootBootstrap.build(strata)
    roots = tuple(strata["CORE32"] + strata["EXTRA96"])
    root_numbers = np.asarray([int(root.rsplit("-", 1)[1]) for root in roots])
    primary_values = (root_numbers % 2 == 0).astype(float) * 0.5 - 1
    expected_h3 = np.quantile(plan.mean(primary_values, roots), [0.025, 0.975], method="linear")
    np.testing.assert_allclose(result["primary"]["ci95_root_conditional"], expected_h3)
    secondary_values = (root_numbers % 3 == 0).astype(float) - 1
    expected_g32 = np.quantile(
        plan.mean(secondary_values, roots), [0.05 / 6, 1 - 0.05 / 6], method="linear"
    )
    np.testing.assert_allclose(
        result["secondary_family"]["endpoints"]["G_3_2"]["ci98_333333_bonferroni_three"],
        expected_g32,
    )
    assert result["bootstrap"]["shared_root_plan_sha256"] == plan.sha256
    json.dumps(result, allow_nan=False)
