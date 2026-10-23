import unittest

from sr_f4.development_review import assess_ai_reviews


class AIReviewTests(unittest.TestCase):
    def setUp(self):
        self.keys = [dict(review_id="x", row_id="r")]
        self.labels = {"r": dict(dimensions={d: dict(status="PASS") for d in "PS"})}
        self.first = dict(
            review_id="x",
            reviewer="AI_1",
            annotation_source="AI_REVIEW",
            P="PASS",
            S="PASS",
            rationale="explicit final set",
        )

    def test_complete_ai_review_does_not_become_human_certification(self):
        result = assess_ai_reviews(self.keys, [self.first], self.labels)
        self.assertTrue(result["complete"])
        self.assertFalse(result["human_audit_complete"])
        self.assertEqual(result["human_review_count"], 0)

    def test_disagreement_is_not_silently_adjudicated(self):
        second = dict(self.first, reviewer="AI_2", S="FAIL")
        result = assess_ai_reviews(self.keys, [self.first, second], self.labels)
        self.assertEqual(len(result["unresolved_disagreements"]), 1)
        self.assertNotIn("auto_PASS__ai_PASS", result["counts"]["S"])

    def test_reject_duplicate_and_false_source(self):
        for rows in ([self.first, self.first], [dict(self.first, annotation_source="HUMAN")]):
            with self.assertRaises(ValueError):
                assess_ai_reviews(self.keys, rows, self.labels)

    def test_explicit_adjudication_preserves_original_disagreement(self):
        second = dict(self.first, reviewer="AI_2", S="UNKNOWN")
        decision = dict(
            review_id="x",
            P="PASS",
            S="UNKNOWN",
            annotation_source="AI_ADJUDICATION",
            reason="invalid JSON",
        )
        result = assess_ai_reviews(self.keys, [self.first, second], self.labels, [decision])
        self.assertFalse(result["unresolved_disagreements"])
        self.assertEqual(len(result["resolved_disagreements"]), 1)
        self.assertEqual(result["counts"]["S"]["auto_PASS__ai_UNKNOWN"], 1)
