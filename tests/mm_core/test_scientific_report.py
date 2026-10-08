"""Regression coverage for saved-table presentation; all completions are synthetic."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
import tempfile
import unittest
from itertools import product
from pathlib import Path

from mm_core.scoring import score_records

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/mm_core/build_scientific_report.py"
SPEC = importlib.util.spec_from_file_location("scientific_report", SCRIPT)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def synthetic_tables(*, roots=1, stage="MEASUREMENT_AUDIT", missing=False):
    questions, outputs = [], []
    for index, chart, op, visual, numeric in product(
        range(roots), report.CHARTS, report.OPERATIONS, report.LEVELS, report.LEVELS
    ):
        root = f"r{index}"
        image = f"{root}/{chart}/{visual}/{numeric}"
        qid = f"{image}/{op}"
        truth = [30, 20, 10] if op == "range" else [30, 20]
        split = "FORMAT_TUNE" if stage == "FORMAT_BASE_TEST" else "AUDIT_MEASURE"
        questions.append(
            {
                "question_id": qid,
                "root_family_id": root,
                "numeric_root_id": root + numeric,
                "split": split,
                "chart_type": chart,
                "operation": op,
                "visual_level": visual,
                "numeric_level": numeric,
                "readset_id": image,
                "image_hash": image,
                "source_graph_hash": image,
                "ordered_item_ids": list(map(str, range(len(truth)))),
                "true_values_decimal": list(map(str, truth)),
                "delta_decimal": "1",
                "quantity_units": "count",
            }
        )
        for sample in range(4):
            if missing and sample == 3:
                continue
            # Same stochastic reading pattern in both matched prompts gives a negative
            # finite-sample collision contrast; the report must preserve its sign.
            readings = [n + sample % 2 for n in truth]
            answer = sum(readings) if op == "sum" else readings[0] - readings[-1]
            outputs.append(
                {
                    "request_id": f"{qid}/{sample}",
                    "question_id": qid,
                    "sample_index": sample,
                    "stage": stage,
                    "status": "completed",
                    "model_hash": "frozen-model",
                    "truncated": False,
                    "tokens": [],
                    "raw_text": json.dumps({"readings": readings, "answer": answer}),
                }
            )
    return score_records(questions, outputs, stage=stage, bootstrap_replicates=4)["tables"]


class ScientificReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved = synthetic_tables()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def write_tables(self, tables=None, stage="MEASUREMENT_AUDIT"):
        for name, document in (tables or self.saved).items():
            self.write(f"tables/{stage}/{name}.json", document)

    def read_report(self, name="SCIENTIFIC_RESULTS_zh.md"):
        return (self.root / "report" / name).read_text(encoding="utf-8")

    def reject(self, mutate, message):
        tables = copy.deepcopy(self.saved)
        mutate(tables)
        self.write_tables(tables)
        with self.assertRaisesRegex(ValueError, message):
            report.build_report(self.root)
        self.assertFalse((self.root / "report").exists())

    def test_missing_measurement_does_not_borrow_format_success(self):
        self.write_tables(synthetic_tables(stage="FORMAT_BASE_TEST"), "FORMAT_BASE_TEST")
        result = report.build_report(self.root)
        text = self.read_report()
        self.assertEqual(result["status"], "MEASUREMENT_INCOMPLETE")
        self.assertIn("BASE/CHECK回答不代替测量回答", text)
        self.assertIn("没有测量覆盖表", text)
        self.assertNotIn("## 精确计数与成功率", text)
        self.assertFalse(result["dev_authorized"])

    def test_complete_table_panel_is_not_a_release_authorization(self):
        self.write_tables(synthetic_tables(roots=16))
        result = report.build_report(self.root)
        self.assertEqual(result["status"], "MEASUREMENT_TABLES_COMPLETE")
        self.assertFalse(result["scientific_stage_authorized"])
        self.assertEqual(result["delivery_interpretation"], "RELEASE_MISSING")

    def test_partial_measurement_has_observed_denominator_and_missing_slots(self):
        self.write_tables(synthetic_tables(missing=True))
        result = report.build_report(self.root)
        text = self.read_report()
        self.assertEqual(result["status"], "MEASUREMENT_INCOMPLETE")
        self.assertIn("missing_sample_slots | 24", text)
        self.assertIn("72/72", text)
        self.assertIn("预期但未生成槽位另见覆盖", text)

    def test_multi_cohort_even_identical_is_rejected(self):
        self.reject(
            lambda t: t["COUNTS"]["cohorts"].append(t["COUNTS"]["cohorts"][0]), "MULTIPLE_COHORTS"
        )

    def test_cross_table_model_identity_is_rejected(self):
        self.reject(
            lambda t: t["GEOMETRY"]["cohorts"][0].update(model_hash="other"),
            "CROSS_TABLE_MODEL_MISMATCH",
        )

    def test_wrong_stage_or_split_is_rejected(self):
        for key, value, message in (
            ("stage", "FORMAT_CHECK", "STAGE_MISMATCH"),
            ("split", "FORMAT_TUNE", "SPLIT_MISMATCH"),
        ):
            with self.subTest(key=key):
                self.reject(
                    lambda t, k=key, v=value: t["COUNTS"]["cohorts"][0].update({k: v}), message
                )

    def test_wrong_common_start_is_rejected(self):
        self.write_tables()
        self.write("manifests/COMMON_START.json", {"model_hash": "another"})
        with self.assertRaisesRegex(ValueError, "COMMON_START_MODEL_MISMATCH"):
            report.build_report(self.root)

    def test_duplicate_or_missing_factor_labels_rejected(self):
        self.reject(
            lambda t: t["COUNTS"]["cohorts"][0]["factor_cells"].append(
                t["COUNTS"]["cohorts"][0]["factor_cells"][0]
            ),
            "INVALID_OR_DUPLICATE_LABEL",
        )

    def test_conditional_denominator_mismatch_rejected(self):
        def mutate(tables):
            value = tables["COUNTS"]["cohorts"][0]["overall"]["conditional_on_L"]["p"]
            value.update(numerator=0, denominator=95, rate={"numerator": 0, "denominator": 1})

        self.reject(mutate, "CONDITIONAL_DENOMINATOR_OR_COUNT_MISMATCH")

    def test_fraction_and_rate_count_disagreement_rejected(self):
        def mutate(tables):
            value = tables["COUNTS"]["cohorts"][0]["overall"]["P_full"]
            value["rate"] = {"numerator": 1, "denominator": 7}

        self.reject(mutate, "COUNT_RATE_MISMATCH")

    def test_self_and_gold_prefix_cannot_be_swapped(self):
        def mutate(tables):
            row = tables["FIELD_METRICS"]["cohorts"][0]["overall"]
            row["self_field_surprisal"]["expected_prefix_provenance"] = "gold"

        self.reject(mutate, "FIELD_PREFIX_MISMATCH")

    def test_field_sequence_denominator_mismatch_rejected(self):
        def mutate(tables):
            row = tables["FIELD_METRICS"]["cohorts"][0]["overall"]
            row["self_field_surprisal"]["fields"]["answer"]["mean_nll"]["sequence_denominator"] = 1

        self.reject(mutate, "FIELD_SEQUENCE_DENOMINATOR_MISMATCH")

    def test_missing_metrics_not_invented_and_negative_context_retained(self):
        self.write_tables()
        report.build_report(self.root)
        main = self.read_report()
        field = self.read_report("SCIENTIFIC_FIELDS_zh.md")
        self.assertIn("gold teacher forcing", main)
        self.assertIn("不能称作自由生成能力", main)
        self.assertIn("missing | missing | 0 | 0", field)
        self.assertIn("-0.166667 &#91;-1/6&#93;", self.read_report("SCIENTIFIC_CONTEXT_zh.md"))
        self.assertIn("总体", main)
        for values in product(report.CHARTS, report.OPERATIONS, report.LEVELS, report.LEVELS):
            self.assertIn(" / ".join(values), self.read_report("SCIENTIFIC_COUNTS_zh.md"))

    def test_root_conditional_and_complete_means_are_distinct(self):
        tables = copy.deepcopy(self.saved)
        root = tables["ROOT_EQUAL"]["cohorts"][0]
        root["aggregate"]["S"]["complete_fixed_weight_mean"] = None
        self.write_tables(tables)
        report.build_report(self.root)
        self.assertIn("根等权条件均值 | 完整固定权重均值", self.read_report())
        self.assertIn("不能套在完整固定权重均值上", self.read_report())
        self.assertIn("undefined", self.read_report())

    def test_context_original_pair_denominator_is_validated(self):
        self.reject(
            lambda t: t["CONTEXT_AUDIT"]["cohorts"][0]["pairs"][0].update(C11_denominator=11),
            "CONTEXT_DENOMINATOR_MISMATCH",
        )

    def test_root_support_weight_cannot_change_its_denominator(self):
        def mutate(tables):
            value = tables["ROOT_EQUAL"]["cohorts"][0]["roots"][0]["metrics"]["P_full"]
            value["supported_weight"] = {"numerator": 1, "denominator": 2}

        self.reject(mutate, "ROOT_WEIGHT_DENOMINATOR_MISMATCH")

    def test_context_pair_cannot_belong_to_a_different_root(self):
        self.reject(
            lambda t: t["CONTEXT_AUDIT"]["cohorts"][0]["pairs"][0].update(
                root_family_id="other-root"
            ),
            "CONTEXT_QUESTION_IDENTITY_MISMATCH",
        )

    def test_nested_open_fields_only_change_ready_interpretation(self):
        self.write_tables()
        proposal = {
            "authorized": False,
            "status": "NOT_RUN",
            "recipe": {"lr": None},
            "open_fields": [],
        }
        self.write("report/DEV_FREEZE_PROPOSAL.json", proposal)
        for status, expected in (
            ("READY_FOR_DEV_REVIEW", "READY_WITH_OPEN_DEV_FIELDS"),
            ("ENGINE_REPRO_NOT_READY", "ENGINE_REPRO_NOT_READY"),
        ):
            self.write("report/FINAL_AUDIT_STATUS.json", {"status": status})
            original = (self.root / "report/FINAL_AUDIT_STATUS.json").read_bytes()
            result = report.build_report(self.root)
            self.assertEqual(result["delivery_interpretation"], expected)
            self.assertIn("recipe.lr", self.read_report())
            self.assertEqual((self.root / "report/FINAL_AUDIT_STATUS.json").read_bytes(), original)

    def test_engine_zero_contrast_and_actual_steps_are_separate_from_audit(self):
        self.write_tables()
        path = self.root / "engineering/engine/continuous/STEPS.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "logical_step": 1,
                    "zero_contrast_groups": [True, True],
                    "rewards": [1] * 16,
                    "gradient_norm": 0.0,
                }
            )
            + "\n"
        )
        report.build_report(self.root)
        evidence = self.read_report("SCIENTIFIC_EVIDENCE_zh.md")
        self.assertIn("continuous | 1 | 2/2", evidence)
        self.assertIn("可能不包含非零梯度更新", evidence)
        self.assertIn("CPU处理wall time: UNKNOWN", evidence)
        self.assertIn("resumed | UNKNOWN", evidence)

    def test_latency_fields_never_count_scoring_as_generation(self):
        self.write_tables()
        base = self.root / "raw/MEASUREMENT_AUDIT"
        base.mkdir(parents=True)
        (base / "outputs_0.jsonl").write_text(json.dumps({"generation_seconds": 2}) + "\n")
        (base / "scores_0.jsonl").write_text(
            json.dumps({"scoring_seconds": 3, "generation_seconds": 2}) + "\n"
        )
        report.build_report(self.root)
        text = self.read_report("SCIENTIFIC_EVIDENCE_zh.md")
        self.assertIn("outputs_0.jsonl | generation_seconds | 1/1 | 2", text)
        self.assertIn("scores_0.jsonl | scoring_seconds | 1/1 | 3", text)
        self.assertNotIn("scores_0.jsonl | generation_seconds", text)

    def test_read_only_inputs_manifest_hashes_and_all_links_resolve(self):
        self.write_tables()
        before = {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob("*.json")}
        result = report.build_report(self.root)
        for relative, data in before.items():
            self.assertEqual((self.root / relative).read_bytes(), data)
        for relative, digest in result["input_sha256"].items():
            self.assertEqual(
                hashlib.sha256((self.root / relative).read_bytes()).hexdigest(), digest
            )
        for relative, digest in result["output_sha256"].items():
            self.assertEqual(
                hashlib.sha256((self.root / relative).read_bytes()).hexdigest(), digest
            )
        future_evidence = {"FINAL_EVIDENCE_VERIFICATION.json", "FINAL_EVIDENCE_SHA256.json"}
        for path in (self.root / "report").glob("SCIENTIFIC_*.md"):
            for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
                if target not in future_evidence:
                    self.assertTrue((path.parent / target).is_file(), target)
        self.assertEqual(result["rescored_answers"], 0)

    def test_final_evidence_is_linked_without_circular_input_hash(self):
        self.write_tables()
        self.write("report/FINAL_EVIDENCE_VERIFICATION.json", {"prior": "stale"})
        self.write("report/FINAL_EVIDENCE_SHA256.json", {"old_report": "digest"})
        result = report.build_report(self.root)
        self.assertNotIn("report/FINAL_EVIDENCE_VERIFICATION.json", result["input_sha256"])
        self.assertNotIn("report/FINAL_EVIDENCE_SHA256.json", result["input_sha256"])
        self.assertNotIn("report/SCIENTIFIC_REPORT_MANIFEST.json", result["output_sha256"])
        self.assertIn(
            "FINAL_EVIDENCE_VERIFICATION.json", self.read_report("SCIENTIFIC_EVIDENCE_zh.md")
        )

    def test_undefined_bootstrap_has_no_broken_source_link(self):
        self.write_tables()
        (self.root / "tables/MEASUREMENT_AUDIT/ROOT_BOOTSTRAP.json").unlink()
        result = report.build_report(self.root)
        self.assertEqual(result["status"], "MEASUREMENT_INCOMPLETE")
        self.assertIn("ROOT_BOOTSTRAP missing", self.read_report())

    def test_table_text_cannot_inject_a_remote_image(self):
        value = report.table(["key"], [["![x](https://example.test/x.png)"]])
        self.assertNotIn("![", value)

    def test_rendering_uses_exact_count_fraction_not_float_display(self):
        value = {
            "numerator": 1,
            "denominator": 3,
            "rate": {"numerator": 1, "denominator": 3, "float": 0.999},
        }
        self.assertEqual(report.number(value), "1/3 (33.3333%)")
        self.assertEqual(
            report.number({"numerator": 0, "denominator": 0, "rate": None}), "0/0 (missing)"
        )

    def test_input_mutation_during_report_is_rejected(self):
        self.write("tables/input.json", {"value": 1})
        inputs = report.Inputs(self.root)
        inputs.json("tables/input.json")
        self.write("tables/input.json", {"value": 2})
        with self.assertRaisesRegex(ValueError, "INPUT_CHANGED_DURING_REPORT"):
            inputs.stable()


if __name__ == "__main__":
    unittest.main()
