"""Pure bridge status regression tests; no CUDA execution or additional model draws."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from src.verified_discovery_transfer.bridge import assess_bridge_diagnostics


def complete_diagnostics():
    gradient = {
        "finite": True,
        "cosine": 1.0,
        "relative_l2": 0.0,
        "left_norm": 1.0,
        "right_norm": 1.0,
        "max_abs": 0.0,
    }
    return {
        "examples": [
            {
                "task_id": f"t{i}",
                "prompt_tokens": 3,
                "target_ids": [4, 9],
                "first_target_logit_index": 2,
                "full_token_nll": [1.0, 2.0],
                "batch_token_nll": [1.0, 2.0],
                "repeat_token_nll": [1.0, 2.0],
                "prefix_token_nll": [1.0, 2.0],
                "sequence_nll": 1.5,
                "batch_sequence_nll": 1.5,
                "future_logits_max_abs": 0.0,
                "same_path_logits_max_abs": 0.0,
            }
            for i in range(8)
        ],
        "padding_diagnostics": [
            {
                "tasks": [f"t{i}" for i in range(start, start + 4)],
                "padding_logits_max_abs": 0.0,
                "same_batch_logits_max_abs": 0.0,
            }
            for start in (0, 4)
        ],
        "gradient_same_path": copy.deepcopy(gradient),
        "gradient_microbatch": copy.deepcopy(gradient),
        "gradient_full_vs_prefix": copy.deepcopy(gradient),
        "causal_within_same_path_baseline": True,
        "resume_exact": True,
    }


class BridgeGateTests(unittest.TestCase):
    def test_zero_difference_passes_without_mutating_evidence(self):
        evidence = complete_diagnostics()
        before = copy.deepcopy(evidence)
        self.assertEqual(
            assess_bridge_diagnostics(evidence), {"status": "PASS", "diagnostic_gate_reasons": []}
        )
        self.assertEqual(before, evidence)

    def test_prefix_token_difference_alone_requires_review(self):
        evidence = complete_diagnostics()
        evidence["examples"][0]["prefix_token_nll"][1] += 0.25
        result = assess_bridge_diagnostics(evidence)
        self.assertEqual(result["status"], "NUMERICAL_REVIEW_REQUIRED")
        self.assertTrue(
            any("prefix/full token" in reason for reason in result["diagnostic_gate_reasons"])
        )

    def test_prefix_gradient_difference_alone_requires_review(self):
        for field, value in (("relative_l2", 0.1), ("cosine", 0.9)):
            with self.subTest(field=field):
                evidence = complete_diagnostics()
                evidence["gradient_full_vs_prefix"][field] = value
                self.assertEqual(
                    assess_bridge_diagnostics(evidence)["status"], "NUMERICAL_REVIEW_REQUIRED"
                )

    def test_prefix_token_uses_per_token_measured_noise(self):
        evidence = complete_diagnostics()
        row = evidence["examples"][0]
        row["repeat_token_nll"] = [1.25, 2.5]
        row["prefix_token_nll"] = [1.125, 2.5]
        self.assertEqual(assess_bridge_diagnostics(evidence)["status"], "PASS")
        row["prefix_token_nll"][0] = 1.5
        self.assertEqual(assess_bridge_diagnostics(evidence)["status"], "NUMERICAL_REVIEW_REQUIRED")

    def test_prefix_gradient_uses_dimensionless_same_path_noise(self):
        evidence = complete_diagnostics()
        evidence["gradient_same_path"].update(relative_l2=0.125, cosine=0.875, max_abs=0.01)
        # Raw max_abs has a different loss scale; compare relative error and cosine.
        evidence["gradient_full_vs_prefix"].update(
            relative_l2=0.0625, cosine=0.9375, left_norm=100.0, right_norm=100.0, max_abs=1.0
        )
        self.assertEqual(assess_bridge_diagnostics(evidence)["status"], "PASS")
        evidence["gradient_full_vs_prefix"]["relative_l2"] = 0.25
        self.assertEqual(assess_bridge_diagnostics(evidence)["status"], "NUMERICAL_REVIEW_REQUIRED")

    def test_nonfinite_tokens_block(self):
        for field in ("full_token_nll", "repeat_token_nll", "prefix_token_nll", "batch_token_nll"):
            for value in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(field=field, value=value):
                    evidence = complete_diagnostics()
                    evidence["examples"][0][field][0] = value
                    self.assertEqual(
                        assess_bridge_diagnostics(evidence)["status"], "BLOCKED_INVALID_DIAGNOSTICS"
                    )

    def test_nonfinite_gradient_fields_block_even_with_finite_flag(self):
        for key in ("gradient_same_path", "gradient_microbatch", "gradient_full_vs_prefix"):
            for field in ("cosine", "relative_l2", "left_norm", "right_norm", "max_abs"):
                with self.subTest(key=key, field=field):
                    evidence = complete_diagnostics()
                    evidence[key][field] = float("nan")
                    self.assertEqual(
                        assess_bridge_diagnostics(evidence)["status"], "BLOCKED_INVALID_DIAGNOSTICS"
                    )

    def test_missing_prefix_or_noise_blocks(self):
        for key in ("prefix_token_nll", "repeat_token_nll"):
            evidence = complete_diagnostics()
            evidence["examples"][0][key] = None
            self.assertEqual(
                assess_bridge_diagnostics(evidence)["status"], "BLOCKED_INVALID_DIAGNOSTICS"
            )
        for key in ("gradient_full_vs_prefix", "gradient_same_path", "padding_diagnostics"):
            evidence = complete_diagnostics()
            evidence.pop(key)
            self.assertEqual(
                assess_bridge_diagnostics(evidence)["status"], "BLOCKED_INVALID_DIAGNOSTICS"
            )

    def test_misaligned_prefix_tokens_and_zero_gradient_block(self):
        evidence = complete_diagnostics()
        evidence["examples"][0]["prefix_token_nll"] = [1.0]
        self.assertEqual(
            assess_bridge_diagnostics(evidence)["status"], "BLOCKED_INVALID_DIAGNOSTICS"
        )
        evidence = complete_diagnostics()
        evidence["gradient_full_vs_prefix"]["left_norm"] = 0.0
        self.assertEqual(
            assess_bridge_diagnostics(evidence)["status"], "BLOCKED_INVALID_DIAGNOSTICS"
        )

    def test_causal_failure_blocks_even_if_summary_flag_is_true(self):
        evidence = complete_diagnostics()
        evidence["examples"][0]["future_logits_max_abs"] = 0.5
        self.assertEqual(
            assess_bridge_diagnostics(evidence)["status"], "BLOCKED_CAUSALITY_OR_RESUME"
        )
        evidence = complete_diagnostics()
        evidence["padding_diagnostics"][0]["padding_logits_max_abs"] = 0.5
        self.assertEqual(
            assess_bridge_diagnostics(evidence)["status"], "BLOCKED_CAUSALITY_OR_RESUME"
        )

    def test_resume_failure_and_missing_boolean_do_not_pass(self):
        for value, status in (
            (False, "BLOCKED_CAUSALITY_OR_RESUME"),
            (None, "BLOCKED_INVALID_DIAGNOSTICS"),
        ):
            evidence = complete_diagnostics()
            evidence["resume_exact"] = value
            self.assertEqual(assess_bridge_diagnostics(evidence)["status"], status)


class BridgeEntryGateTests(unittest.TestCase):
    def exercise(self, receipt, *, acceptance=None):
        from src.verified_discovery_transfer.cli import _bridge_passed

        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "bridge.json").write_text(json.dumps(receipt))
            if acceptance is not None:
                (run / "BRIDGE_NUMERICAL_ACCEPTANCE.json").write_text(json.dumps(acceptance))
            return _bridge_passed(run)

    @staticmethod
    def receipt():
        return {
            "status": "PASS",
            "parents": {"S96": complete_diagnostics(), "REP96": complete_diagnostics()},
        }

    @staticmethod
    def acceptance(receipt):
        from src.verified_discovery_transfer.queue import digest

        return {
            "bridge_hash": digest(receipt),
            "status": "ACCEPTED",
            "rationale": "Reviewed measured finite BF16 differences against same-path noise",
        }

    def test_genuine_two_parent_pass_needs_no_manual_override(self):
        self.exercise(self.receipt())

    def test_both_numerical_reviews_need_current_acceptance(self):
        receipt = self.receipt()
        for parent in receipt["parents"].values():
            parent["gradient_full_vs_prefix"]["relative_l2"] = 0.125
        with self.assertRaisesRegex(RuntimeError, "numerical review"):
            self.exercise(receipt)
        self.exercise(receipt, acceptance=self.acceptance(receipt))

    def test_one_pass_one_review_can_be_accepted(self):
        receipt = self.receipt()
        receipt["parents"]["S96"]["examples"][0]["prefix_token_nll"][0] += 0.25
        self.exercise(receipt, acceptance=self.acceptance(receipt))

    def test_stale_pass_label_does_not_bypass_recomputed_prefix_review(self):
        receipt = self.receipt()
        receipt["parents"]["REP96"]["examples"][0]["prefix_token_nll"][0] += 0.25
        with self.assertRaisesRegex(RuntimeError, "numerical review"):
            self.exercise(receipt)

    def test_acceptance_cannot_override_missing_or_zero_gradient_evidence(self):
        for alteration in ("missing_prefix", "missing_gradient", "zero_gradient"):
            with self.subTest(alteration=alteration):
                receipt = self.receipt()
                parent = receipt["parents"]["S96"]
                if alteration == "missing_prefix":
                    parent["examples"][0]["prefix_token_nll"] = None
                elif alteration == "missing_gradient":
                    parent.pop("gradient_same_path")
                else:
                    parent["gradient_full_vs_prefix"]["left_norm"] = 0.0
                with self.assertRaisesRegex(RuntimeError, "non-overridable"):
                    self.exercise(receipt, acceptance=self.acceptance(receipt))

    def test_acceptance_cannot_override_causal_or_resume_failures(self):
        for alteration in ("causality", "padding", "resume"):
            with self.subTest(alteration=alteration):
                receipt = self.receipt()
                parent = receipt["parents"]["S96"]
                if alteration == "causality":
                    parent["examples"][0]["future_logits_max_abs"] = 0.25
                elif alteration == "padding":
                    parent["padding_diagnostics"][0]["padding_logits_max_abs"] = 0.25
                else:
                    parent["resume_exact"] = False
                with self.assertRaisesRegex(RuntimeError, "non-overridable"):
                    self.exercise(receipt, acceptance=self.acceptance(receipt))

    def test_nonfinite_evidence_is_blocked_before_acceptance_hash(self):
        for value in (float("nan"), float("inf")):
            receipt = self.receipt()
            receipt["parents"]["S96"]["gradient_full_vs_prefix"]["relative_l2"] = value
            # Nonfinite JSON cannot have a valid canonical digest; reject diagnostics first.
            with self.assertRaisesRegex(RuntimeError, "non-overridable"):
                self.exercise(
                    receipt,
                    acceptance={
                        "status": "ACCEPTED",
                        "bridge_hash": "forged",
                        "rationale": "Must not override invalid data",
                    },
                )

    def test_both_parent_evidence_is_mandatory(self):
        receipt = self.receipt()
        receipt["parents"].pop("REP96")
        with self.assertRaisesRegex(RuntimeError, "both S96 and REP96"):
            self.exercise(receipt, acceptance=self.acceptance(receipt))

    def test_changed_receipt_or_empty_rationale_invalidates_acceptance(self):
        receipt = self.receipt()
        receipt["parents"]["S96"]["gradient_full_vs_prefix"]["relative_l2"] = 0.125
        for change in ({"bridge_hash": "stale"}, {"rationale": " "}, {"status": "PENDING"}):
            acceptance = {**self.acceptance(receipt), **change}
            with self.assertRaisesRegex(RuntimeError, "numerical review"):
                self.exercise(receipt, acceptance=acceptance)


if __name__ == "__main__":
    unittest.main()
