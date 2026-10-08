from __future__ import annotations

import copy
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from mm_core.contracts import parse_response, score
from mm_core.scoring import (
    build_format_report,
    discover_raw_files,
    join_field_scores,
    json_exact,
    normalize_question,
    score_records,
)


def question(qid="q", root="r", op="difference", split="AUDIT_MEASURE", **extra):
    return {
        "question_id": qid,
        "root_family_id": root,
        "numeric_root_id": root + "n",
        "split": split,
        "chart_type": "grouped_bar",
        "operation": op,
        "visual_level": "low",
        "numeric_level": "low",
        "readset_id": root + "read",
        "image_hash": root + "image",
        "source_graph_hash": root + "source",
        "ordered_item_ids": ["first", "second"],
        "true_values_decimal": ["30", "20"],
        "delta_decimal": "1",
        "quantity_units": "count",
        **extra,
    }


def output(qid="q", index=0, raw='{"readings":[30,20],"answer":10}', **extra):
    return {
        "request_id": f"request:{qid}:{index}",
        "question_id": qid,
        "stage": "MEASUREMENT_AUDIT",
        "sample_index": index,
        "status": "completed",
        "raw_text": raw,
        "tokens": [],
        "truncated": False,
        "model_hash": "model1",
        **extra,
    }


def fraction(value):
    return None if value is None else Fraction(value["numerator"], value["denominator"])


def field_packet(source="self", nll=2, entropy=3, count=2):
    return {
        "status": "MEASURED",
        "provenance": "generated_tokens_actual_self_prefix"
        if source == "self"
        else "gold_completion_answer_prefix_contains_gold_readings",
        "distribution": "raw_model_before_temperature_or_top_p",
        "boundary_rule": "largest_value_character_overlap_then_readings_tie",
        "field_nll": {
            field: {
                "token_count": count,
                "token_indices": list(range(count)),
                "mean_nll": nll,
                "mean_token_entropy": entropy,
            }
            for field in ("readings", "answer")
        },
    }


class ScoringTests(unittest.TestCase):
    def result(self, qs, outputs, **kwargs):
        return score_records(
            qs, outputs, bootstrap_replicates=kwargs.pop("bootstrap_replicates", 0), **kwargs
        )

    def test_independent_missing_fields_and_full_denominators(self):
        result = self.result(
            [question()],
            [output(index=0, raw='{"answer":10}'), output(index=1, raw='{"readings":[30,20]}')],
        )
        counts = result["tables"]["COUNTS"]["cohorts"][0]["overall"]
        self.assertEqual(counts["joint_scorable_denominator"], 0)
        self.assertEqual(sum(counts["exact_counts"].values()), 0)
        self.assertEqual(fraction(counts["P_full"]["rate"]), Fraction(1, 2))
        self.assertEqual(fraction(counts["A_full"]["rate"]), Fraction(1, 2))
        self.assertEqual(counts["missing_joint_fields_not_000"], 2)

    def test_no_outputs_not_passed_no_missing_rows_fabricated(self):
        result = self.result([question()], [], bootstrap_replicates=20)
        self.assertEqual(result["scored_outputs"], [])
        self.assertEqual(result["tables"]["SCORING_RECEIPT"]["status"], "NO_OUTPUTS")
        self.assertFalse(result["tables"]["FORMAT_REPORT"]["gate_passed"])
        self.assertEqual(result["tables"]["COVERAGE"]["cohorts"][0]["missing_sample_slots"], 4)
        boot = result["tables"]["ROOT_BOOTSTRAP"]["cohorts"][0]["metrics"]["S"]
        self.assertEqual(boot["undefined_replicates"], 20)
        self.assertIsNone(boot["percentile_95_low"])

    def test_all_exact_six_and_tolerance_eight_reconstructed(self):
        cases = [
            ("111", [30, 20], 10),
            ("100", [30, 20], 11),
            ("011", [35, 25], 10),
            ("010", [35, 20], 15),
            ("001", [35, 20], 10),
            ("000", [35, 20], 11),
        ]
        rows = [
            output(index=i, raw=json.dumps({"readings": r, "answer": a}))
            for i, (_, r, a) in enumerate(cases)
        ]
        result = self.result([question()], rows)
        counts = result["tables"]["COUNTS"]["cohorts"][0]["overall"]
        self.assertEqual(set(counts["exact_counts"].values()), {1})
        self.assertEqual(len(counts["tolerance_counts"]), 8)
        self.assertTrue(counts["six_cell_reconstruction_checked"])
        self.assertEqual(fraction(counts["conditional_on_L"]["z"]["rate"]), Fraction(1, 6))

    def test_all_eight_tolerance_cells_observable(self):
        cases = {
            "111": ([30, 20], 50),
            "100": ([30, 20], 52),
            "011": ([31, 19], 50),
            "010": ([31, 20], 51),
            "001": ([31, 20], 50),
            "000": ([31, 20], 52),
            "101": ([30.4, 20.4], 50),
            "110": ([30.4, 20.4], 50.8),
        }
        for cell, (readings, answer) in cases.items():
            self.assertEqual(
                score(json.dumps({"readings": readings, "answer": answer}), [30, 20], "sum")[
                    "cell_tolerance"
                ],
                cell,
            )

    def test_geometry_zero_error_missing_and_exact_decomposition(self):
        result = self.result(
            [question()], [output(), output(index=1, raw='{"readings":[35,20],"answer":10}')]
        )
        geometry = result["tables"]["GEOMETRY"]["cohorts"][0]["overall"]
        self.assertEqual(geometry["S_denominator"], 1)
        self.assertEqual(geometry["S_missing_zero_error_count"], 1)
        self.assertEqual(fraction(geometry["max_exact_residual_identity_error"]), 0)
        self.assertEqual(fraction(geometry["S_direct_error_response_mean"]), Fraction(1, 2))
        self.assertEqual(fraction(geometry["roles"][0]["MAE"]), Fraction(5, 2))

    def test_context_noise_controls_keep_negative_estimates(self):
        qs = [question("sum", op="sum"), question("diff")]
        rows = [
            output(qid, i, json.dumps({"readings": vector, "answer": 10}))
            for qid in ("sum", "diff")
            for i, vector in enumerate(([30, 20], [31, 21]))
        ]
        pair = self.result(qs, rows)["tables"]["CONTEXT_AUDIT"]["cohorts"][0]["pairs"][0]
        self.assertEqual(fraction(pair["C11"]), 0)
        self.assertEqual(fraction(pair["C12"]), Fraction(1, 2))
        self.assertEqual(fraction(pair["collision_contrast"]), Fraction(-1, 2))
        self.assertEqual(pair["C12_denominator"], 4)

    def test_context_units_roles_delta_and_arity_never_coerced(self):
        base = question("sum", op="sum")
        changes = [
            {"quantity_units": "kg"},
            {"ordered_item_ids": ["second", "first"]},
            {"delta_decimal": "2"},
            {"image_hash": "other"},
        ]
        for change in changes:
            result = self.result([base, question("diff", **change)], [])
            pairs = result["tables"]["CONTEXT_AUDIT"]["cohorts"][0]["pairs"]
            self.assertTrue(all(p["status"] == "PAIR_UNAVAILABLE" for p in pairs))

    def test_reward_support_per_question_not_pooled_and_smoothed(self):
        qs = [question("yes"), question("no")]
        rows = [
            output(q, i, '{"answer":' + ("10" if q == "yes" else "11") + "}")
            for q in ("yes", "no")
            for i in range(4)
        ]
        support = self.result(qs, rows)["tables"]["REWARD_SUPPORT"]["cohorts"][0]
        for row in support["questions"]:
            self.assertEqual(fraction(row["plugin_mixed_probability"]), 0)
            self.assertGreater(fraction(row["posterior_predictive_mixed_probability"]), 0)
            self.assertFalse(row["actual_nonconstant_reward"])
        layer = support["chart_operation"][1]
        self.assertEqual(fraction(layer["mean_per_question_plugin_mixed_probability"]), 0)

    def test_duplicate_request_id_dedup_and_conflict_rejected(self):
        row = output()
        result = self.result([question()], [row, dict(row)])
        self.assertEqual(len(result["scored_outputs"]), 1)
        self.assertEqual(
            result["tables"]["SCORING_RECEIPT"]["identical_request_records_deduplicated"], 1
        )
        with self.assertRaisesRegex(ValueError, "CONFLICTING_REQUEST_ID"):
            self.result([question()], [row, {**row, "raw_text": "different"}])

    def test_retry_cost_and_bad_outputs_never_discarded(self):
        rows = [output(status="failed", raw=""), output(request_id="retry")]
        result = self.result([question()], rows)
        coverage = result["tables"]["COVERAGE"]["cohorts"][0]
        self.assertEqual(coverage["recorded_attempts"], 2)
        self.assertEqual(coverage["duplicate_sample_slots"], 1)
        self.assertFalse(coverage["complete"])

    def test_aliases_must_not_conflict(self):
        q = question()
        for old, new in (
            ("true_values_decimal", "true_values"),
            ("delta_decimal", "delta"),
            ("image_hash", "image_sha256"),
            ("visual_level", "V"),
            ("numeric_level", "D"),
        ):
            q[new] = q.pop(old)
        self.assertEqual(normalize_question(q)["true_values_decimal"], ["30", "20"])
        with self.assertRaisesRegex(ValueError, "CONFLICTING_QUESTION_ALIASES"):
            normalize_question({**q, "true_values_decimal": ["30", "21"]})

    def test_input_objects_not_modified(self):
        qs, rows = [question()], [output()]
        before = copy.deepcopy((qs, rows))
        self.result(qs, rows)
        self.assertEqual((qs, rows), before)

    def test_provenance_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "OUTPUT_PROVENANCE_MISMATCH"):
            self.result([question()], [output(image_hash="wrong")])

    def test_unauthorized_split_rejected(self):
        with self.assertRaisesRegex(ValueError, "UNAUTHORIZED_QUESTION_SPLIT"):
            self.result([question(split="LOCK_EVAL")], [])

    def test_processed_image_split_clone_rejected(self):
        qs = [
            question(root="a", processed_image_hash="same"),
            question("q2", root="b", split="FORMAT_CHECK", processed_image_hash="same"),
        ]
        with self.assertRaisesRegex(ValueError, "CROSS_SPLIT_LEAK"):
            self.result(qs, [])

    def test_roles_and_units_validate_boundary(self):
        for changes in (
            {"ordered_item_ids": ["x", "x"]},
            {"quantity_units": ""},
            {"operation": "range"},
            {"delta_decimal": "0"},
        ):
            with self.assertRaises(ValueError):
                self.result([question(**changes)], [])
        parsed = parse_response('{"readings":["30 kg",20],"answer":10}', 2)
        self.assertFalse(parsed.L_P)
        self.assertTrue(parsed.L_A)

    def test_strict_format_gate_full_panel(self):
        qs = []
        for root_index in range(4):
            for chart in ("grouped_bar", "line"):
                for op in ("sum", "difference", "range"):
                    for v in ("low", "high"):
                        for d in ("low", "high"):
                            qid = f"{root_index}-{chart}-{op}-{v}-{d}"
                            qs.append(
                                question(
                                    qid,
                                    root=f"r{root_index}",
                                    op=op,
                                    split="FORMAT_TUNE",
                                    chart_type=chart,
                                    visual_level=v,
                                    numeric_level=d,
                                    ordered_item_ids=["a", "b", "c"]
                                    if op == "range"
                                    else ["a", "b"],
                                    true_values_decimal=["30", "20", "25"]
                                    if op == "range"
                                    else ["30", "20"],
                                )
                            )
        rows = [
            output(
                q["question_id"],
                i,
                json.dumps(
                    {
                        "readings": [30, 20, 25] if q["operation"] == "range" else [30, 20],
                        "answer": 10,
                    }
                ),
                stage="FORMAT_BASE_TEST",
            )
            for q in qs
            for i in range(4)
        ]
        report = build_format_report(qs, rows, stage="FORMAT_BASE_TEST")
        self.assertTrue(report["gate_passed"])
        self.assertFalse(
            build_format_report(qs, rows[:-1], stage="FORMAT_BASE_TEST")["gate_passed"]
        )
        # Failed decoding is a failed engineering result even with valid partial JSON.
        rows[0]["status"] = "failed"
        self.assertEqual(
            build_format_report(qs, rows, stage="FORMAT_BASE_TEST")["cohorts"][0]["overall"][
                "format_ok"
            ]["numerator"],
            383,
        )
        # More than 10% format failures in one layer fails its gate.
        bad_ids = {
            q["question_id"] for q in qs if q["operation"] == "sum" and q["chart_type"] == "line"
        }
        for row in [r for r in rows if r["question_id"] in bad_ids][:7]:
            row["truncated"] = True
        self.assertFalse(build_format_report(qs, rows, stage="FORMAT_BASE_TEST")["gate_passed"])

    def test_format_separates_base_bridge_models(self):
        q = question(split="FORMAT_TUNE")
        rows = [
            output(stage="FORMAT_BASE_TEST"),
            output(request_id="bridge", stage="FORMAT_TUNE_POST_BRIDGE", model_hash="model2"),
        ]
        report = build_format_report([q], rows)
        self.assertEqual(len(report["cohorts"]), 2)
        self.assertTrue(all(c["overall"]["recorded_attempts"] == 1 for c in report["cohorts"]))

    def test_root_equal_not_response_weighted_and_missing_exposed(self):
        qs = [question("small", root="small"), question("large", root="large")]
        rows = [output("small", 0)] + [output("large", i, '{"answer":11}') for i in range(4)]
        result = self.result(qs, rows)
        root = result["tables"]["ROOT_EQUAL"]["cohorts"][0]
        self.assertEqual(
            fraction(root["aggregate"]["A_full"]["root_equal_conditional_mean"]), Fraction(1, 2)
        )
        self.assertIsNone(root["aggregate"]["A_full"]["complete_fixed_weight_mean"])
        direct = result["tables"]["COUNTS"]["cohorts"][0]["overall"]["A_full"]["rate"]
        self.assertEqual(fraction(direct), Fraction(1, 5))

    def test_bootstrap_deterministic_and_undefined_preserved(self):
        qs = [question("a", root="a"), question("b", root="b")]
        rows = [output("a"), output("b", raw='{"readings":[31,21],"answer":10}')]
        first = self.result(qs, rows, bootstrap_replicates=80)
        second = self.result(qs, rows, bootstrap_replicates=80)
        a = first["tables"]["ROOT_BOOTSTRAP"]
        self.assertEqual(a, second["tables"]["ROOT_BOOTSTRAP"])
        shape = a["cohorts"][0]["metrics"]["S"]
        self.assertGreater(shape["undefined_replicates"], 0)
        self.assertEqual(shape["defined_replicates"] + shape["undefined_replicates"], 80)
        self.assertEqual(fraction(shape["percentile_95_low"]), 1)

    def test_serialized_rationals_exact(self):
        self.assertEqual(json_exact(Fraction(1, 3))["denominator"], 3)
        json.dumps(self.result([question()], [output()]), allow_nan=False)

    def test_large_finite_errors_keep_exact_values_without_float_overflow(self):
        number = "9" * 64 + "e128"
        result = self.result(
            [question()],
            [output(raw='{"readings":[' + number + ',20],"answer":10}')],
            bootstrap_replicates=10,
        )
        self.assertTrue(result["scored_outputs"][0]["L_P"])
        mse = result["tables"]["GEOMETRY"]["cohorts"][0]["overall"]["roles"][0]["MSE"]
        self.assertIsNone(mse["float"])
        self.assertGreater(fraction(mse), 10**300)
        json.dumps(result, allow_nan=False)

    def test_parser_deep_input_is_recorded_failure(self):
        raw = "[" * 1500 + "0" + "]" * 1500
        self.assertFalse(parse_response(raw, 2).L)
        with self.assertRaisesRegex(ValueError, "INVALID_AXIS"):
            score("invalid json", [30, 20], "difference", axis=(100, 0))

    def test_field_score_join_preserves_original_completion_exactly(self):
        raw = output(
            processor_hash="processor",
            seed=123,
            tokens=[1, 2, 3],
            raw_tokens=[1, 2, 3],
            image_routing={"grid": [1, 24, 32]},
        )
        sidecar = {
            **raw,
            "status": "scored",
            "self_field_surprisal": field_packet(),
            "gold_teacher_forced_field_nll": field_packet("gold", nll=7),
            "scoring_seconds": 1.5,
        }
        joined = join_field_scores([raw], [sidecar])[0]
        for key, value in raw.items():
            self.assertEqual(joined[key], value)
        self.assertEqual(joined["status"], "completed")
        self.assertNotIn("self_field_surprisal", raw)
        self.assertEqual(join_field_scores([raw], []), [raw])

    def test_field_score_join_rejects_identity_text_token_or_routing_change(self):
        raw = output(
            processor_hash="processor", seed=123, tokens=[1, 2], image_routing={"grid": [1, 24, 32]}
        )
        sidecar = {**raw, "status": "scored", "self_field_surprisal": field_packet()}
        changes = {
            "question_id": "other",
            "stage": "FORMAT_CHECK",
            "sample_index": 2,
            "model_hash": "other",
            "processor_hash": "other",
            "seed": 124,
            "raw_text": "changed",
            "tokens": [1, 3],
            "image_routing": {"grid": [1, 32, 32]},
        }
        for key, value in changes.items():
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(ValueError, "FIELD_SCORE_.*MISMATCH"),
            ):
                join_field_scores([raw], [{**sidecar, key: value}])
        with self.assertRaisesRegex(ValueError, "DUPLICATE_FIELD_SCORE_REQUEST"):
            join_field_scores([raw], [sidecar, dict(sidecar)])
        with self.assertRaisesRegex(ValueError, "ORPHAN_FIELD_SCORE_REQUEST"):
            join_field_scores([raw], [{**sidecar, "request_id": "absent"}])

    def test_nested_raw_discovery_never_loads_scores_as_completions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "raw" / "MEASUREMENT_AUDIT"
            folder.mkdir(parents=True)
            for name in (
                "outputs_0.jsonl",
                "scores_0.jsonl",
                "REQUESTS.jsonl",
                "SCORED_OUTPUTS.jsonl",
            ):
                (folder / name).write_text("", encoding="utf-8")
            outputs, scores = discover_raw_files(root)
            self.assertEqual([path.name for path in outputs], ["outputs_0.jsonl"])
            self.assertEqual([path.name for path in scores], ["scores_0.jsonl"])

    def test_missing_field_metrics_not_invented(self):
        result = self.result([question()], [output()])
        report = result["tables"]["FIELD_METRICS"]["cohorts"][0]["overall"]
        for source in report.values():
            self.assertEqual(source["status"], "NOT_MEASURED")
            self.assertEqual(source["source_status_counts"], {"MISSING": 1})
            field = source["fields"]["answer"]["mean_nll"]
            self.assertEqual(field["sequence_denominator"], 0)
            self.assertEqual(field["field_token_denominator"], 0)
            self.assertIsNone(field["sequence_mean"])

    def test_self_and_gold_metrics_have_separate_field_denominators(self):
        rows = [
            output(
                self_field_surprisal=field_packet(nll=2, count=1),
                gold_teacher_forced_field_nll=field_packet("gold", nll=7, count=3),
            ),
            output(index=1, self_field_surprisal=field_packet(nll=4, count=3)),
            output(index=2),
        ]
        result = self.result([question()], rows)
        report = result["tables"]["FIELD_METRICS"]["cohorts"][0]["overall"]
        own = report["self_field_surprisal"]["fields"]["answer"]
        gold = report["gold_teacher_forced_field_nll"]["fields"]["answer"]
        self.assertEqual(own["mean_nll"]["sequence_denominator"], 2)
        self.assertEqual(own["mean_nll"]["field_token_denominator"], 4)
        self.assertEqual(fraction(own["mean_nll"]["sequence_mean"]), 3)
        self.assertEqual(fraction(own["mean_nll"]["token_weighted_mean"]), Fraction(7, 2))
        self.assertEqual(gold["mean_nll"]["sequence_denominator"], 1)
        self.assertEqual(gold["mean_nll"]["field_token_denominator"], 3)
        self.assertEqual(fraction(gold["mean_nll"]["sequence_mean"]), 7)
        self.assertEqual(own["mean_token_entropy"]["field_token_denominator"], 4)
        root_nll = result["tables"]["ROOT_EQUAL"]["cohorts"][0]["aggregate"][
            "self_field_surprisal_answer_nll"
        ]["root_equal_conditional_mean"]
        self.assertEqual(fraction(root_nll), 3)

    def test_field_metric_prefix_or_zero_token_measurement_rejected(self):
        for packet in (field_packet("gold"), field_packet(count=0)):
            with self.assertRaises(ValueError):
                self.result([question()], [output(self_field_surprisal=packet)])

    def test_unverified_field_boundaries_remain_missing(self):
        packet = {
            "status": "TOKEN_CHARACTER_BOUNDARY_UNVERIFIED",
            "field_nll": None,
            "provenance": "generated_tokens_actual_self_prefix",
            "tokens": 10,
        }
        report = self.result([question()], [output(self_field_surprisal=packet)])["tables"][
            "FIELD_METRICS"
        ]["cohorts"][0]["overall"]["self_field_surprisal"]
        self.assertEqual(report["measured_packet_count"], 0)
        self.assertEqual(report["source_status_counts"]["TOKEN_CHARACTER_BOUNDARY_UNVERIFIED"], 1)

    def test_runner_failed_null_completion_is_retained_without_synthetic_text(self):
        result = self.result([question()], [output(raw=None, truncated=None, status="failed")])
        row = result["scored_outputs"][0]
        self.assertIsNone(row["raw_text"])
        self.assertIsNone(row["truncated"])
        self.assertFalse(row["L"])
        self.assertEqual(row["parse_errors"], ["NO_COMPLETION_TEXT_FAILED_ATTEMPT"])
        self.assertEqual(
            result["tables"]["COUNTS"]["cohorts"][0]["overall"]["A_full"]["denominator"], 1
        )


if __name__ == "__main__":
    unittest.main()
