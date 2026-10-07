"""Synthetic contract checks: these are not SER-J2 model observations."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.exposure_substitution.reports import summarize_cells, write_reports
from src.exposure_substitution.semantics import deletion_projection, parse, score_output
from src.exposure_substitution.statistics import (
    IncompleteDataError,
    analyze_confirmation,
    cross_macro,
    pass_at_k,
    primary_effect,
    safe_ratio,
    summarize_tasks,
    wilson,
)


def task(world=(10, 20, 30, 40), center=3, j=3, corrupt=45):
    observed = list(world)
    observed[j] = corrupt
    H, b = [], []
    for p, q in sorted(tuple(sorted((center, k))) for k in range(4) if k != center):
        row = [0] * 4
        row[p] = row[q] = 1
        H.append(row)
        b.append(world[p] + world[q])
    public = dict(
        task_id="t",
        root_id="r",
        family="cross_series",
        observed=observed,
        H_original=H,
        b_original=b,
        legal_domain=[0, 99],
        operation="sum4",
    )
    audit = dict(task_id="t", root_id="r", true_world=list(world), center=center, corrupted_index=j)
    return public, audit


def scored(raw, **extra):
    public, audit = task()
    return dict(
        checkpoint_id="parent",
        parent="S96",
        block=None,
        arm="PARENT",
        step=0,
        panel="E_DIAG",
        draw_index=0,
        raw_text=raw,
        **score_output(public, audit, raw),
        **extra,
    )


class SemanticsContracts(unittest.TestCase):
    def test_whole_json_exact_ints_and_domain_are_separate(self):
        for text in (
            "prefix [1,2,3,4]",
            "[1,2,3,4] tail",
            "[true,2,3,4]",
            "[1.0,2,3,4]",
            "[1,2,3]",
            '{"a":1}',
        ):
            self.assertIsNone(parse(text), text)
        self.assertEqual(parse("\n [1, 2, 3, 400]\t"), [1, 2, 3, 400])
        row = scored("[-1,20,30,40]")
        self.assertTrue(row["parse_ok"])
        self.assertFalse(row["domain_ok"])
        self.assertTrue(row["I"])
        self.assertTrue(row["value_present"])
        self.assertTrue(row["value_present_but_not_X"])
        self.assertIsNone(row["M"])
        self.assertTrue(row["projection_success"])

    def test_legal_edit_identity_does_not_confuse_location_and_correct_value(self):
        row = scored("[10,20,30,41]")
        self.assertEqual([row[k] for k in ("M", "B", "L", "F")], [1, 0, 1, 0])
        self.assertEqual(row["B"], row["M"] - 1)
        self.assertFalse(row["F"])

    def test_event_partition_and_complete_length_stop(self):
        for raw, expected in [
            ("[10,20,30,40]", "X"),
            ("[11,19,30,40]", "S"),
            ("[10,20,30,41]", "W"),
            ("error", "I"),
        ]:
            public, audit = task()
            row = score_output(public, audit, raw, "length")
            self.assertEqual(row["event"], expected)
            self.assertEqual(sum(row[k] for k in "XSWI"), 1)
            self.assertTrue(row["truncated"])
            self.assertEqual(
                int(row["X"]), int(row["value_present"]) - int(row["value_present_but_not_X"])
            )

    def test_public_anchor_does_not_require_correct_center(self):
        good = scored("[5,20,30,40]")
        wrong = scored("[5,20,30,41]")
        self.assertEqual(good["mixed_anchor_public_bits"], [True, False, False, False])
        self.assertTrue(good["mixed_anchor_audit"])
        self.assertTrue(wrong["mixed_anchor_public"])
        self.assertFalse(wrong["mixed_anchor_audit"])
        self.assertEqual(good["leaf_edited_and_center_correct"], [True, False, False, False])

    def test_operand_signatures_overlap(self):
        public, audit = task(world=(10, 12, 30, 20), center=3, j=0, corrupt=50)
        row = score_output(public, audit, "[12,12,30,20]")
        signature = next(s for s in row["operand_signatures"] if s["coordinate"] == 0)
        self.assertTrue(signature["other_relation_constant"])
        self.assertTrue(signature["off_by_2"])
        self.assertTrue(signature["forward_target"])
        self.assertTrue(signature["right_location_single_edit"])

    def test_projection_is_truth_free_and_uses_only_existing_values(self):
        public, audit = task()

        class PublicOnly(dict):
            def __getitem__(self, key):
                if key in ("truth", "true_world", "corrupted_index", "center"):
                    raise AssertionError("Audit read by public verifier")
                return super().__getitem__(key)

        result = deletion_projection(PublicOnly(public), [-500, 1000, 30, 40])
        self.assertEqual(result["projected_vector"], audit["true_world"])
        self.assertEqual(result["projection_verifier_calls"], 4)
        self.assertFalse(deletion_projection(public, [10, 20, 30, 41])["projection_success"])
        self.assertEqual(deletion_projection(public, None)["projection_verifier_calls"], 0)
        self.assertEqual(public["observed"], [10, 20, 30, 45])

    def test_ambiguous_projection_is_contract_failure(self):
        public, _ = task()
        public["H_original"] = [[1, 1, 0, 0]]
        public["b_original"] = [35]
        with self.assertRaisesRegex(ValueError, "Unique"):
            deletion_projection(public, [15, 25, 30, 45])

    def test_all_new_manifest_gold_targets_and_legacy_common_audit(self):
        root = Path(__file__).parents[1] / "docs/exposure_substitution/design/manifests"
        pairs = [
            ("donors_public.jsonl", "donors_audit.jsonl"),
            ("E_DIAG/tasks_public.jsonl", "E_DIAG/audit_only.jsonl"),
            ("E_CONFIRM/tasks_public.jsonl", "E_CONFIRM/audit_only.jsonl"),
        ]
        for public_name, audit_name in pairs:
            audits = {
                a["task_id"]: a
                for a in map(json.loads, (root / audit_name).read_text().splitlines())
            }
            for public in map(json.loads, (root / public_name).read_text().splitlines()):
                audit = audits[public["task_id"]]
                result = score_output(public, audit, json.dumps(audit["true_world"]))
                self.assertTrue(result["X"])
                self.assertTrue(result["projection_success"])
        audits = {
            a["task_id"]: a
            for a in map(json.loads, (root / "common_audit.jsonl").read_text().splitlines())
        }
        for row in map(json.loads, (root / "common_targets.jsonl").read_text().splitlines()):
            self.assertTrue(
                score_output(row["task"], audits[row["task"]["task_id"]], row["target"])["X"]
            )

    def test_zero_condition_denominator_and_sparse_joint(self):
        summary = summarize_cells([scored("error")])
        self.assertIsNone(summary["values"][0]["extra_given_parsed"])
        self.assertEqual(summary["values"][0]["denominator_parsed"], 0)
        self.assertEqual(summary["joints"][0]["event"], "I")
        self.assertEqual(summary["joints"][0]["count"], 1)
        self.assertIsNone(summary["anchors"][0]["p_leaf_edited_given_center_correct_but_not_X"])


class StatisticsContracts(unittest.TestCase):
    def test_primary_sign_secondary_and_shared_parent(self):
        p = np.full((2, 3, 3, 128, 2), 0.5)
        p[:, :, 1, :, 1], p[:, :, 2, :, 1] = 0.25, 0.75
        p[:, :, 1, :, 0], p[:, :, 2, :, 0] = 0.8, 0.2
        result = primary_effect(p, parent_probabilities=np.full((2, 128, 2), 0.5))
        self.assertAlmostEqual(result["Gamma"], -0.55)
        self.assertAlmostEqual(result["Delta4"], -0.5)
        self.assertAlmostEqual(result["Delta3"], 0.6)
        self.assertAlmostEqual(result["secondary"]["B4-A4"]["estimate"], -0.25)
        self.assertIn("ci98_75_bonferroni_four_secondary", result["secondary"]["B4-A4"])
        self.assertAlmostEqual(
            result["relative_to_parent"]["C_FORWARD_C3:c3j3-PARENT"]["estimate"], -0.3
        )
        self.assertEqual(result["seed"], 107079)
        self.assertEqual(result["bootstrap_replicates"], 5000)
        self.assertEqual(np.shape(result["per_parent_schedule_Gamma"]), (2, 3))

    def test_full_root_vector_bootstrap_matches_reference(self):
        p = np.full((2, 3, 3, 128, 2), 0.5)
        p[:, :, 1, :, 1] = np.arange(128) / 127
        result = primary_effect(p)
        roots = ((np.arange(128) / 127) - 0.5) / 2
        samples = np.random.default_rng(107079).integers(0, 128, size=(5000, 128))
        expected = np.quantile(roots[samples].mean(axis=1), [0.025, 0.975])
        np.testing.assert_allclose(result["ci95_root_conditional"], expected)
        broken = p[:, :, :, :127, :]
        with self.assertRaises(ValueError):
            primary_effect(broken)

    def test_macro_and_boundary_uncertainty(self):
        cells = {(c, j): float(c == j and c in (2, 3)) for c in range(4) for j in range(4)}
        self.assertEqual(cross_macro(cells), 0.125)
        self.assertNotEqual(cross_macro(cells), 256 / 704)
        self.assertGreater(wilson(0, 8)[1], 0)
        self.assertLess(wilson(8, 8)[0], 1)
        self.assertIsNone(safe_ratio(0, 0))
        self.assertEqual(pass_at_k(0, 8, 8), 0)
        self.assertEqual(pass_at_k(1, 8, 8), 1)

    def test_duplicate_formal_draw_and_missing_endpoint_fail(self):
        row = scored("[10,20,30,40]")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize_tasks([row, row])
        with self.assertRaises(IncompleteDataError):
            analyze_confirmation([row])

    def test_complete_manifest_matrix_macro_and_decomposition(self):
        root = Path(__file__).parents[1] / "docs/exposure_substitution/design/manifests/E_CONFIRM"
        public = {
            t["task_id"]: t
            for t in map(json.loads, (root / "tasks_public.jsonl").read_text().splitlines())
        }
        audit = list(map(json.loads, (root / "audit_only.jsonl").read_text().splitlines()))
        identities = [
            (s, b, a)
            for s in ("S96", "REP96")
            for b in range(3)
            for a in ("A_LOCAL_C1", "B_FORWARD_C4", "C_FORWARD_C3")
        ]
        identities += [(s, None, "PARENT") for s in ("S96", "REP96")]
        rows = []
        for parent, block, arm in identities:
            for a in audit:
                family = public[a["task_id"]]["family"]
                # Only the oversampled protected diagonal cells succeed. The
                # correct cross macro is 2/16 rather than the raw 256/704.
                x = (
                    family == "cross_series"
                    and a["center"] == a["corrupted_index"]
                    and a["center"] in (2, 3)
                )
                for draw in range(8):
                    rows.append(
                        dict(
                            checkpoint_id=f"{parent}.{block}.{arm}",
                            parent=parent,
                            block=block,
                            arm=arm,
                            step=0 if arm == "PARENT" else 256,
                            panel="E_CONFIRM",
                            task_id=a["task_id"],
                            root_id=a["root_id"],
                            root_cohort=a["root_cohort"],
                            family=family,
                            center=a["center"],
                            corrupted_index=a["corrupted_index"],
                            draw_index=draw,
                            X=x,
                            S=False,
                            W=not x,
                            I=False,
                            value_present=x,
                            value_present_but_not_X=False,
                        )
                    )
        result = analyze_confirmation(rows, bootstrap_replicates=32)
        self.assertEqual(result["primary"]["Gamma"], 0)
        self.assertEqual(len(result["task_metrics"]), 16000)
        for row in result["macro"]["metrics"]:
            if row["metric"] == "X" and row["macro"] == "cross_macro":
                self.assertEqual(row["estimate"], 0.125)
            if row["metric"] == "X" and row["macro"] == "cross_trend_50_50":
                self.assertEqual(row["estimate"], 0.0625)
        with self.assertRaises(IncompleteDataError):
            analyze_confirmation(rows[:-1], bootstrap_replicates=2)

    def test_release_required_and_incomplete_evidence_retained(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                write_reports(folder, [], release_receipt={})
            receipt = dict(
                all_registered_models_terminal=True,
                registered_training_jobs=18,
                analysis_code_sha256="a" * 64,
            )
            result = write_reports(
                folder,
                [scored("error")],
                release_receipt=receipt,
                execution_scope={"technical_missing": ["runB"]},
            )
            self.assertEqual(result["status"], "INCOMPLETE_NOT_CERTIFIED")
            primary = json.loads((Path(folder) / "PRIMARY_EFFECT.json").read_text())
            self.assertIsNone(primary["Gamma"])
            self.assertEqual(
                json.loads((Path(folder) / "EXECUTION_SCOPE.json").read_text())[
                    "technical_missing"
                ],
                ["runB"],
            )
            self.assertIn("没有从已完成子集", (Path(folder) / "FINAL_FINDINGS_zh.md").read_text())


if __name__ == "__main__":
    unittest.main()
