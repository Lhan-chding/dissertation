from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mm_dev.analysis import (
    BOOTSTRAP_SEED,
    STATES,
    _hierarchical,
    _shape_diagnostics,
    bootstrap_weights,
    build_artifacts,
    interval,
    load_cost_accounting,
    number,
    opportunity_analysis,
    panel_cube,
    pretrain_features,
    response_analysis,
    saved_token_accounting,
    score_panel,
    training_support,
    utility_values,
    write_artifacts,
)
from mm_dev.contract import CELLS, PLAN_ID, STRATA, run_matrix

ROOT = Path(__file__).resolve().parents[2]


def plan_fixture(roots=4):
    plan = json.loads((ROOT / "docs/mm_dev_f2/design/config/MM_DEV_F2.json").read_text())
    plan["data"]["root_counts"]["DEV_EVAL"] = roots
    plan["statistics"]["crossfit_root_halves"]["half0_indices"] = list(range(roots // 2))
    plan["statistics"]["crossfit_root_halves"]["half1_indices"] = list(range(roots // 2, roots))
    return plan


def cubes_fixture(roots=4):
    cubes = {s: np.full((roots, 6, 4, 3), 0.5) for s in STATES}
    for run in run_matrix():
        if run["phase"] == "CONTINUE":
            v = cubes[run["start_state"]].copy()
            effect = {"a0": 0.05, "aP": 0.15, "aC": -0.05}[run["action"]]
            v[..., 0] += effect
            v[..., 1] += effect / 2
            cubes[run["run_id"]] = v
    return cubes


def packet(gold=False):
    return {
        "status": "MEASURED",
        "provenance": "gold_completion_answer_prefix_contains_gold_readings"
        if gold
        else "generated_tokens_actual_self_prefix",
        "distribution": "raw_model_before_temperature_or_top_p",
        "boundary_rule": "fixture",
        "field_nll": {
            f: {
                "token_count": 1,
                "token_indices": [i],
                "mean_nll": 2 if gold else 1,
                "mean_token_entropy": None if gold else 0.5,
            }
            for i, f in enumerate(("readings", "answer"))
        },
    }


def payload_fixture(panel="PROBE", roots=1):
    qs, outputs, selfs, golds = [], [], [], []
    for root in range(roots):
        for chart, op in STRATA:
            for cell in CELLS:
                qid = f"{root}-{chart}-{op}-{cell}"
                truth = [30, 20, 40] if op == "range" else [30, 20]
                qs.append(
                    {
                        "question_id": qid,
                        "root_index": root,
                        "root_family_id": f"root-{root}",
                        "numeric_root_id": f"root-{root}",
                        "split": panel,
                        "chart_type": chart,
                        "operation": op,
                        "visual_level": "low" if cell[0] == "L" else "high",
                        "numeric_level": "low" if cell[1] == "L" else "high",
                        "readset_id": f"root-{root}-{cell}-{'range' if op == 'range' else 'pair'}",
                        "image_hash": f"image-{root}-{chart}-{cell}",
                        "source_graph_hash": f"source-{root}",
                        "ordered_item_ids": [str(i) for i in range(len(truth))],
                        "true_values_decimal": list(map(str, truth)),
                        "delta_decimal": "1",
                        "quantity_units": "count",
                    }
                )
                golds.append({"question_id": qid, "field_scores": packet(True)})
                for k in range(4):
                    rid = f"r-{qid}-{k}"
                    outputs.append(
                        {
                            "request_id": rid,
                            "question_id": qid,
                            "stage": panel,
                            "sample_index": k,
                            "status": "completed",
                            "model_hash": "same-checkpoint",
                            "raw_text": json.dumps(
                                {
                                    "readings": truth,
                                    "answer": {"sum": 50, "difference": 10, "range": 20}[op],
                                }
                            ),
                            "tokens": [],
                            "truncated": False,
                        }
                    )
                    selfs.append({"request_id": rid, "field_scores": packet()})
    return {
        "questions": qs,
        "outputs": outputs,
        "self_scores": selfs if panel == "PROBE" else [],
        "gold_scores": golds if panel == "PROBE" else [],
        "complete": True,
        "identity": {"checkpoint_hash": "same-checkpoint"},
        "technical_events": [],
    }


class AnalysisTests(unittest.TestCase):
    def test_rational_and_missing(self):
        self.assertEqual(number("1/3"), 1 / 3)
        self.assertEqual(number({"numerator": 3, "denominator": 4}), 0.75)
        self.assertIsNone(number(None))
        for x in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                number(x)

    def test_normalization_preserves_actual_identity_and_input(self):
        payload = payload_fixture()
        original = copy.deepcopy(payload)
        scored = score_panel(payload, state_id="S0", panel="PROBE")
        self.assertEqual(payload, original)
        self.assertEqual(scored["scored_outputs"][0]["split"], "PROBE")
        self.assertEqual(scored["physical_field_measurements"], {"self": 96, "gold": 24})
        field_table = scored["tables"]["FIELD_METRICS"]["cohorts"][0]["overall"]
        self.assertEqual(field_table["gold_teacher_forced_field_nll"]["response_denominator"], 24)
        self.assertEqual(field_table["self_field_surprisal"]["response_denominator"], 96)
        self.assertEqual(panel_cube(scored["scored_outputs"], 1).shape, (1, 6, 4, 3))

    def test_slot_missing_duplicate_failure_and_orphan_rejected(self):
        for mode in ("missing", "duplicate", "failure", "orphan"):
            payload = payload_fixture()
            if mode == "missing":
                payload["outputs"].pop()
            elif mode == "duplicate":
                payload["outputs"].append(payload["outputs"][0])
            elif mode == "failure":
                payload["outputs"][0]["status"] = "failed"
            else:
                payload["gold_scores"].append(
                    {"question_id": "orphan", "field_scores": packet(True)}
                )
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                score_panel(payload, state_id="S0", panel="PROBE")

    def test_technical_slots_not_zero_but_missing_semantics_are_full_false(self):
        payload = payload_fixture("DEV_EVAL")
        payload["outputs"][0]["raw_text"] = '{"answer":50}'
        scored = score_panel(payload, state_id="S0", panel="DEV_EVAL")
        cube = panel_cube(scored["scored_outputs"], 1)
        self.assertEqual(cube[0, 0, 0, 0], 1)
        self.assertEqual(cube[0, 0, 0, 1], 0.75)
        self.assertEqual(cube[0, 0, 0, 2], 0.75)

    def test_endpoint_field_scoring_prohibited(self):
        payload = payload_fixture("DEV_EVAL")
        payload["self_scores"] = [{"request_id": "x"}]
        with self.assertRaises(ValueError):
            score_panel(payload, state_id="S0", panel="DEV_EVAL")

    def test_cube_rejects_missing_and_repeated_cells(self):
        rows = score_panel(payload_fixture(), state_id="S0", panel="PROBE")["scored_outputs"]
        for changed in (rows[:-1], [*rows, rows[0]]):
            with self.assertRaises(ValueError):
                panel_cube(changed, 1)

    def test_root_cluster_bootstrap_preserves_whole_pair(self):
        weights = bootstrap_weights(4, replicates=100)
        np.testing.assert_array_equal(weights, bootstrap_weights(4, replicates=100))
        np.testing.assert_allclose(weights.sum(axis=1), 1)
        left = np.array([0.1, 0.3, 0.8, 0.2])
        right = left + 0.25
        np.testing.assert_allclose(weights @ right - weights @ left, 0.25)

    def test_four_primary_and_future_effects_and_no_train_reference(self):
        result = response_analysis(cubes_fixture(), plan_fixture())
        effects = {x["contrast"]: x for x in result["primary"]["contrasts"]}
        self.assertEqual(len(effects), 4)
        self.assertAlmostEqual(effects["macro_A_aP_minus_a0"]["effect_probability"], 0.1)
        self.assertAlmostEqual(effects["macro_P_aC_minus_a0"]["effect_probability"], -0.05)
        for effect in effects.values():
            self.assertEqual(effect["CI95"]["defined_replicates"], 5000)
            self.assertEqual(effect["CI98_75"]["level"], 0.9875)
            self.assertAlmostEqual(effect["future_1_minus_0"], 0)
            self.assertIsNone(effect["future_training_SE"])
        notrain = next(r for r in result["responses"] if r["action"] == "no_train")
        self.assertEqual(notrain["absolute_D"]["A_full"], 0)
        self.assertAlmostEqual(notrain["relative_Delta_vs_a0"]["A_full"], -0.05)
        self.assertEqual(len(result["responses"]), 5 * 4 * 2 * 6)
        self.assertEqual(len(result["changes"]), 5 * 4 * 2 * 24)

    def test_all_35_ids_required_even_coincident_models(self):
        cubes = cubes_fixture()
        cubes.pop("cont_S0_a0_f0")
        with self.assertRaises(ValueError):
            response_analysis(cubes, plan_fixture())

    def test_fixed_bootstrap_recipe_not_tunable(self):
        p = plan_fixture()
        p["statistics"]["root_bootstrap_seed"] = BOOTSTRAP_SEED + 1
        with self.assertRaises(ValueError):
            response_analysis(cubes_fixture(), p)

    def test_utility_does_not_cancel_protected_stratum_harm(self):
        d = np.zeros((6, 3))
        d[:, 0] = 0.1
        d[0, 1], d[1, 1] = -0.12, 0.12
        u = utility_values(d)
        self.assertAlmostEqual(float(u["I"]), 0.02)
        self.assertAlmostEqual(float(u["U"]), 0.08)
        self.assertFalse(u["feasible"])

    def test_future_average_before_positive_part(self):
        cubes = cubes_fixture()
        for state in STATES:
            cubes[f"cont_{state}_aP_f0"][..., 1] = 0.4
            cubes[f"cont_{state}_aP_f1"][..., 1] = 0.6
        u = response_analysis(cubes, plan_fixture())["utilities"]["rows"]
        row = next(x for x in u if x["start_state"] == "S0" and x["action"] == "aP")
        self.assertEqual(row["I"], 0)
        self.assertAlmostEqual(row["per_future"][0]["I"], 0.1)

    def test_crossfit_selection_does_not_see_heldout_future_or_roots(self):
        d = np.zeros((5, 4, 2, 4, 6, 4, 3))
        d[:, 2, 0, :2, :, :, 0] = 0.1
        d[:, 3, 1, 2:, :, :, 0] = 0.9
        d[:, 2, 1, 2:, :, :, 0] = -0.3
        result = opportunity_analysis(d, plan_fixture())
        fold = result["crossfit"]["unconstrained"]["folds"][0]
        self.assertEqual(fold["selected_actions"], ["aP"] * 5)
        self.assertEqual(fold["fixed_action"], "aP")
        self.assertAlmostEqual(fold["evaluation_state_selected_U"], -0.3)
        self.assertAlmostEqual(fold["evaluation_gap"], 0)

    def test_constrained_fixed_action_requires_all_states_feasible(self):
        d = np.zeros((5, 4, 2, 4, 6, 4, 3))
        d[:, 2, ..., 0] = 0.2
        d[0, 2, ..., 1] = -0.1
        result = opportunity_analysis(d, plan_fixture())
        chosen = result["in_sample"]["constrained"]
        self.assertNotIn("aP", chosen["fixed_eligible_actions"])
        self.assertEqual(chosen["fixed_action_index"], 0)
        self.assertGreater(chosen["gap"], 0)

    def test_fixed_halves_cannot_overlap(self):
        plan = plan_fixture()
        plan["statistics"]["crossfit_root_halves"]["half1_indices"] = [0, 1]
        with self.assertRaises(ValueError):
            opportunity_analysis(np.zeros((5, 4, 2, 4, 6, 4, 3)), plan)

    def test_missing_S_preserved_with_support_and_undefined_bootstraps(self):
        rows = score_panel(payload_fixture(), state_id="S0", panel="PROBE")["scored_outputs"]
        result = _shape_diagnostics(rows, bootstrap_weights(1, replicates=20))
        self.assertIsNone(result["S"]["conditional_root_equal_mean"])
        self.assertIsNone(result["S"]["complete_fixed_weight_mean"])
        self.assertEqual(result["S"]["conditional_root_CI95"]["undefined_replicates"], 20)
        self.assertEqual(result["S"]["valid_response_count"], 0)

    def test_single_coordinate_shape_identity(self):
        payload = payload_fixture()
        payload["outputs"][0]["raw_text"] = '{"readings":[31,20],"answer":50}'
        rows = score_panel(payload, state_id="S0", panel="PROBE")["scored_outputs"]
        diagnostics = _shape_diagnostics(rows)
        self.assertEqual(diagnostics["single_coordinate_count"], 1)
        self.assertEqual(diagnostics["single_coordinate_max_identity_residual"], 0)
        self.assertIsNone(diagnostics["S"]["complete_fixed_weight_mean"])
        self.assertEqual(diagnostics["S"]["conditional_root_equal_mean"], 0.5)

    def test_hierarchical_weights_do_not_favor_supported_response_count(self):
        rows = [
            {"root_family_id": "a", "question_id": "q1", "v": 1},
            {"root_family_id": "a", "question_id": "q2", "v": None},
            {"root_family_id": "b", "question_id": "q3", "v": 0},
            {"root_family_id": "b", "question_id": "q3", "v": 0},
        ]
        value = _hierarchical(rows, lambda r: r["v"])
        self.assertEqual(value["conditional_root_equal_mean"], 0.5)
        self.assertIsNone(value["complete_fixed_weight_mean"])

    def test_feature_ladder_no_endpoint_leak_and_gold_denominator_once(self):
        scored = score_panel(payload_fixture(), state_id="S0", panel="PROBE")
        features = pretrain_features(scored, {"path_id": "SECRET_PATH", "future_rewards": [99]})
        text = json.dumps(features)
        self.assertNotIn("SECRET_PATH", text)
        inputs = features["strata"][0]["inputs"]
        self.assertNotIn("root_values", json.dumps(inputs))
        self.assertNotIn("root_supported_cells", json.dumps(inputs))
        self.assertIn("non_feature_root_diagnostics", features)
        self.assertEqual(
            set(inputs), {"B0", "BJ_addition", "BTab_addition", "BNum_addition", "BGeo_S_addition"}
        )
        self.assertEqual(set(inputs["BGeo_S_addition"]), {"S"})
        b0 = inputs["B0"]
        gold = b0["gold_teacher_forced_field_nll"]["answer"]["mean_nll"]
        self.assertEqual(gold["valid_response_count"], 4)
        self.assertEqual(
            b0["self_field_surprisal"]["answer"]["mean_nll"]["valid_response_count"], 16
        )
        for k in (
            "supported_root_count",
            "total_root_count",
            "valid_response_count",
            "missing_response_count",
        ):
            self.assertEqual(
                inputs["BNum_addition"]["shape_support"][k], inputs["BGeo_S_addition"]["S"][k]
            )
        with self.assertRaises(ValueError):
            pretrain_features({**scored, "state_id": "cont_S0_aP_f0"})

    def test_zero_contrast_group_retained_and_counterfactual_distinction(self):
        rows = [
            {
                "logical_step": 1,
                "question_id": "q",
                "sample_index": k,
                "rA": 0,
                "q_read": 0 if k == 0 else 1,
                "reward": 0,
                "chart_type": "line",
                "operation": "sum",
                "visual_level": "low",
                "numeric_level": "low",
            }
            for k in range(8)
        ]
        result = training_support({"prep_A_0": rows}, {"prep_A_0": [{"logical_step": 1}]})
        group = result["paths"][0]["groups"][0]
        self.assertTrue(group["zero_contrast"])
        self.assertTrue(group["all_wrong"])
        self.assertGreater(group["d_adv"], 1e-6)
        self.assertEqual(group["advantages"], [0] * 8)
        rows[0]["reward"] = 0.5
        with self.assertRaises(ValueError):
            training_support({"prep_A_0": rows}, {})

    def test_intervals_keep_undefined_not_zero(self):
        x = interval(np.array([np.nan, np.nan]))
        self.assertIsNone(x["lower"])
        self.assertEqual(x["undefined_replicates"], 2)

    def test_outputs_manifest_and_raw_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "raw").mkdir()
            raw = root / "raw" / "x.jsonl"
            raw.write_text("raw evidence\n")
            result = write_artifacts(
                root, {"summary/test.json": {"missing": None}, "features/test.jsonl": [{"v": 1}]}
            )
            self.assertEqual(raw.read_text(), "raw evidence\n")
            self.assertEqual(len(result["files"]), 2)
            self.assertFalse(result["next_stage_authorized"])
            repeat = write_artifacts(
                root, {"summary/test.json": {"missing": None}, "features/test.jsonl": [{"v": 1}]}
            )
            self.assertEqual(result, repeat)
            with self.assertRaises(FileExistsError):
                write_artifacts(root, {"summary/test.json": {"missing": 0}})
            with self.assertRaises(ValueError):
                write_artifacts(root, {"raw/overwrite.json": {}})

    def test_cli_help_requires_explicit_plan_and_runroot(self):
        spec = importlib.util.spec_from_file_location(
            "analysis_cli", ROOT / "scripts/mm_dev/score_and_analyze.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch("sys.argv", ["score_and_analyze.py"]), self.assertRaises(SystemExit) as error:
            module.main()
        self.assertEqual(error.exception.code, 2)

    def test_pipeline_gate_fails_before_loading_response_or_writing_files(self):
        spec = importlib.util.spec_from_file_location(
            "analysis_cli", ROOT / "scripts/mm_dev/score_and_analyze.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from types import SimpleNamespace
        from unittest.mock import Mock

        args = SimpleNamespace(plan=Path("unused"), run_root=Path("unused"))
        load = Mock(side_effect=AssertionError("must not reveal results"))
        gate = Mock(side_effect=PermissionError("not all paths complete"))
        with self.assertRaises(PermissionError):
            module.analyze(args, load, gate, load)
        load.assert_not_called()

    def test_complete_bounded_panels_build_all_outputs(self):
        plan = plan_fixture(2)
        plan["data"]["root_counts"]["PROBE"] = 1
        eval_panel = score_panel(payload_fixture("DEV_EVAL", 2), state_id="S0", panel="DEV_EVAL")
        evaluations = {
            m: {
                **eval_panel,
                "state_id": m,
                "scored_outputs": [{**r, "state_id": m} for r in eval_panel["scored_outputs"]],
            }
            for m in cubes_fixture(2)
        }
        probes = {s: score_panel(payload_fixture(), state_id=s, panel="PROBE") for s in STATES}
        paths = []
        for run in run_matrix():
            paths.append(
                {
                    **run,
                    "groups": [],
                    "group_count": 768,
                    "recorded_update_count": 32,
                    "updates": [{"logical_step": k} for k in range(1, 33)],
                    "exposure_by_stratum_cell": [],
                    "zero_contrast_fraction": 1,
                    "d_adv_above_1e_6_fraction": 0,
                }
            )
        # Reward recount is tested separately; this keeps the complete report
        # integration fixture bounded to 96/192 observations per model.
        with patch("mm_dev.analysis.training_support", return_value={"paths": paths}):
            files = build_artifacts(
                plan=plan,
                evaluations=evaluations,
                probes=probes,
                completeness={"status": "COMPLETE"},
                training_rollouts={},
                training_updates={},
                cost_accounting={"fixture": True},
            )
        required = {
            "FINAL_MM_DEV_REPORT_zh.md",
            "RUN_COMPLETENESS.json",
            "PRIMARY_EFFECTS.json",
            "RESPONSE_BY_STATE_ACTION_FUTURE_STRATUM.jsonl",
            "ABSOLUTE_AND_RELATIVE_CHANGES.jsonl",
            "PREPARATION_STATE_DIFFERENCES.json",
            "REWARD_AND_ADVANTAGE_SUPPORT.json",
            "JOINT_EVENT_TABLES.json",
            "NUMERIC_AND_GEOMETRY_SUPPORT.json",
            "CONTEXT_AUDIT.json",
            "UTILITY_AND_FEASIBILITY.json",
            "EXPLORATORY_OPPORTUNITY_CROSSFIT.json",
            "ACTUAL_COST_ACCOUNTING.json",
            "NEXT_STAGE_RECOMMENDATION_zh.md",
        }
        self.assertTrue({"summary/" + name for name in required}.issubset(files))
        self.assertEqual(len(files["features/PRETRAIN_FEATURES_B0_BJ_BTAB_BNUM_BGEO.jsonl"]), 5)
        json.dumps(files, allow_nan=False)
        for row in files["summary/RESPONSE_BY_ROOT.jsonl"]:
            self.assertIn("evaluation_root_family_id", row)
        self.assertIn("cont", str(files["summary/JOINT_EVENT_TABLES.json"]))
        self.assertEqual(sum(k.startswith("scoring/") for k in files), 40)

    def test_actual_gpu_cost_requires_hashed_terminal_source(self):
        from mm_dev.orchestration import digest, summarize_accounting

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "manifests").mkdir()
            (root / "orchestration").mkdir()
            permission = {"owner": "u", "account": "a", "qos": "q"}
            registration = {
                "plan_id": PLAN_ID,
                "tasks": {"t": {"gpus": 1}},
                "permission": permission,
            }
            attempt = {"attempt_id": "t-a1", "job_id": "1", "job_name": "job", "comment": "bound"}
            state = {"tasks": {"t": {"status": "COMPLETE", "attempts": [attempt]}}}
            (root / "orchestration/REGISTRATION.json").write_text(json.dumps(registration))
            (root / "orchestration/STATE.json").write_text(json.dumps(state))
            source = root / "sacct.json"
            accounting = [
                {
                    "JobIDRaw": "1",
                    "JobName": "job",
                    "Comment": "bound",
                    "User": "u",
                    "Account": "a",
                    "QOS": "q",
                    "State": "COMPLETED",
                    "ElapsedRaw": "180",
                    "AllocTRES": "gres/gpu=1,gres/gpu:pro6000=1",
                    "Start": "2026-10-09T00:00:00",
                    "End": "2026-10-09T00:03:00",
                    "ExitCode": "0:0",
                }
            ]
            source.write_text(json.dumps({"queue": [], "accounting": accounting}))
            final = {
                "plan_id": PLAN_ID,
                "registration_hash": digest(registration),
                "gpu_task_count": 1,
                "status": "COMPLETE",
                "policy": "ACCOUNTING_ONLY",
                "actual_gpu_seconds": 180,
                "actual_gpu_hours": 0.05,
                "allocations": [
                    {
                        "task_id": "t",
                        "attempt_id": "t-a1",
                        "job_id": "1",
                        **summarize_accounting(accounting, attempt, permission, 1),
                        "scheduler_observation": {
                            "path": "sacct.json",
                            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        },
                    }
                ],
            }
            (root / "manifests/GPU_ACCOUNTING_FINAL.json").write_text(json.dumps(final))
            cost = load_cost_accounting(root)
            self.assertEqual(cost["actual_allocated_gpu_seconds"], 180)
            self.assertIsNone(cost["queue_seconds"])
            final["allocations"] = []
            (root / "manifests/GPU_ACCOUNTING_FINAL.json").write_text(json.dumps(final))
            with self.assertRaises(ValueError):
                load_cost_accounting(root)
            final["allocations"] = [
                {
                    "task_id": "t",
                    "attempt_id": "t-a1",
                    "job_id": "1",
                    **summarize_accounting(accounting, attempt, permission, 1),
                    "scheduler_observation": {
                        "path": "sacct.json",
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    },
                }
            ]
            (root / "manifests/GPU_ACCOUNTING_FINAL.json").write_text(json.dumps(final))
            source.write_text("changed")
            with self.assertRaises(ValueError):
                load_cost_accounting(root)

    def test_fractional_AP_reward_is_compared_exactly_before_float(self):
        rows = [
            {
                "logical_step": 1,
                "question_id": "q",
                "sample_index": i,
                "rA": "1",
                "q_read": "2/3",
                "reward": "5/6",
            }
            for i in range(8)
        ]
        result = training_support({"prep_AP_0": rows}, {})
        path = next(p for p in result["paths"] if p["run_id"] == "prep_AP_0")
        group = path["groups"][0]
        self.assertEqual(group["rAP_counterfactual_exact"], ["5/6"] * 8)
        self.assertEqual(group["used_reward_exact"], ["5/6"] * 8)
        self.assertTrue(group["zero_contrast"])

    def test_saved_tokens_and_forward_targets_do_not_count_gold_as_generation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            raw = root / "raw/PROBE/S0"
            raw.mkdir(parents=True)
            partial = raw / "outputs_000000.jsonl"
            partial.write_bytes(b'{"request_id":')
            row = {
                "request_id": "q0",
                "panel": "PROBE",
                "state_id": "S0",
                "tokens": [1, 2],
                "prompt_token_count": 10,
            }
            (raw / "outputs_000001.jsonl").write_text(json.dumps(row))
            (raw / "gold_scores_000001.jsonl").write_text(json.dumps({"tokens": [1] * 100}) + "\n")
            result = saved_token_accounting(
                root,
                [
                    {
                        "kind": "extra_forward_sequences",
                        "count": 1,
                        "metadata": {"purpose": "gold", "completion_tokens": 100},
                    }
                ],
            )
            self.assertEqual(result["by_scope"]["PROBE/S0"]["known_completion_tokens"], 2)
            self.assertEqual(
                result["extra_forwards_by_purpose"]["gold"]["registered_target_tokens"], 100
            )
            self.assertIsNone(result["uncaptured_failed_generation_tokens"])
            self.assertEqual(partial.read_bytes(), b'{"request_id":')
            self.assertEqual(
                {e["event"] for e in result["retained_generation_tail_events"]},
                {"INCOMPLETE_WRITE_RETAINED", "VALID_JSON_WITHOUT_TERMINATOR_RETAINED"},
            )


if __name__ == "__main__":
    unittest.main()
