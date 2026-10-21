# ruff: noqa: RUF001
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from sr_f4 import task_contract
from sr_f4.replay import (
    assess_human_audit,
    prepare_blind_audit,
    replay_group,
    replay_stage,
    write_c_gate,
)
from sr_f4.verification import verify_semantics

WORLD = dict(
    categories=["Jan", "Feb", "Mar"],
    domain=[-20, 40],
    series=dict(Alpha=[10, 13, 19], Beta=[1, 3, 5]),
)
QUERY = dict(family="INTERVAL", start=0, end=2, include_start=False, aggregate="mean", value="diff")


def task(world=None, query=None):
    return dict(
        qid="q1",
        world_id="w1",
        world=copy.deepcopy(world or WORLD),
        query=copy.deepcopy(query or QUERY),
    )


def record(work, *, answer=12, world=None, evidence=None, row_id="r0"):
    w = world or WORLD
    if evidence is None:
        evidence = [
            dict(series=s, category=c, value=w["series"][s][i])
            for s in ("Alpha", "Beta")
            for i, c in enumerate(w["categories"])
        ]
    return dict(
        row_id=row_id,
        full_text=json.dumps(dict(evidence=evidence, work=work, answer=answer)),
        block="b0",
        model_id="test",
        generation_status="COMPLETE",
    )


def verify(rec, t=None):
    t = t or task()
    score = task_contract.score(rec["full_text"], t["world"], t["query"])
    return verify_semantics(t, rec, score)


def status(label, dimension):
    return label["dimensions"][dimension]["status"]


def native_std(values):
    mean = sum(values) / len(values)
    std = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))
    return [(x - mean) / (std + 1e-8) if std else 0.0 for x in values]


class VerifierTests(unittest.TestCase):
    def test_explicit_trace_and_exact_execution(self):
        label = verify(
            record(
                "interval_selection=Feb,Mar; diff operands=[10,14]; "
                "sum=24;count=2;reduce=mean(operands)"
            )
        )
        self.assertEqual([status(label, d) for d in "PSOE"], ["PASS"] * 4)
        self.assertTrue(all(label["dimensions"][d]["source_spans"] for d in "PSOE"))

    def test_correct_numeric_answer_never_infers_selection(self):
        label = verify(record("The result is 12."))
        self.assertEqual(label["native_score"]["value_correct"], 1)
        self.assertEqual(status(label, "S"), "UNKNOWN")

    def test_same_answer_wrong_process_can_fail_selection(self):
        self.assertEqual(status(verify(record("final selection=Jan,Mar")), "S"), "FAIL")

    def test_negated_or_provisional_selection_is_unknown(self):
        for work in (
            "not selected: Feb,Mar",
            "Initially selected: Feb,Mar",
            "hypothetical selection=Feb,Mar",
        ):
            self.assertEqual(status(verify(record(work)), "S"), "UNKNOWN")

    def test_decimal_facts_are_json_serializable(self):
        rec = record("selection=Feb,Mar")
        rec["full_text"] = rec["full_text"].replace('"value": 10', '"value": 10.0')
        label = verify(rec)
        self.assertEqual(status(label, "P"), "PASS")
        json.dumps(label, allow_nan=False)

    def test_empty_evidence_is_unknown_even_if_native_truth_vacuous(self):
        label = verify(record("", evidence=[]))
        self.assertEqual(status(label, "P"), "UNKNOWN")

    def test_truth_is_independent_of_certificate_sufficiency(self):
        label = verify(record("", evidence=[dict(series="Alpha", category="Jan", value=10)]))
        self.assertEqual(status(label, "P"), "PASS")
        self.assertFalse(label["normalization"]["sufficient"])

    def test_false_fact_is_fail(self):
        label = verify(record("", evidence=[dict(series="Alpha", category="Jan", value=11)]))
        self.assertEqual(status(label, "P"), "FAIL")

    def test_f9_repeated_operands_wrong_equation_not_selection(self):
        world = dict(
            categories=["Jan", "Feb", "Mar"],
            domain=[0, 20],
            series=dict(Alpha=[10, 10, 10], Beta=[1, 1, 1]),
        )
        label = verify(record("10-1=999, Count=3", world=world, answer=9), task(world))
        self.assertEqual(status(label, "S"), "UNKNOWN")
        self.assertEqual(status(label, "E"), "FAIL")

    def test_duplicate_numeric_mapping_abstains(self):
        world = copy.deepcopy(WORLD)
        world["series"]["Alpha"][0] = 11
        label = verify(
            record("diff operands=[10,14];reduce=mean(operands)", world=world), task(world)
        )
        self.assertEqual(status(label, "S"), "UNKNOWN")
        self.assertTrue(label["dimensions"]["S"]["ambiguous_candidates"])

    def test_explicit_category_resolves_repeated_numeric_mapping(self):
        world = copy.deepcopy(WORLD)
        world["series"]["Alpha"][0] = 11
        label = verify(
            record("final selection=Feb,Mar;diff operands=[10,14]", world=world), task(world)
        )
        self.assertEqual(status(label, "S"), "PASS")

    def test_unique_operand_mapping(self):
        self.assertEqual(status(verify(record("diff operands=[10,14]")), "S"), "PASS")

    def test_contradictory_final_sets_unknown(self):
        self.assertEqual(
            status(verify(record("selection=Jan,Feb,Mar;selection=Feb,Mar")), "S"), "UNKNOWN"
        )

    def test_declared_and_adopted_sets_conflict_unknown(self):
        self.assertEqual(
            status(verify(record("selection=Feb,Mar;diff operands=[9,14]")), "S"), "UNKNOWN"
        )

    def test_retracted_error_does_not_contaminate_final_work(self):
        label = verify(
            record(
                "selection=Jan,Feb,Mar;10-1=999; correction: final selection=Feb,Mar;"
                "diff operands=[10,14];reduce=mean(operands)"
            )
        )
        self.assertEqual(status(label, "S"), "PASS")
        self.assertEqual(status(label, "E"), "PASS")
        self.assertFalse(label["dimensions"]["E"]["observed"]["withdrawn_equations"][0]["correct"])

    def test_unrelated_denominator_does_not_establish_mean(self):
        label = verify(record("In another example there is /3. final selection=Feb,Mar"))
        self.assertEqual(status(label, "O"), "UNKNOWN")

    def test_truncation_abstains_even_with_balanced_json(self):
        rec = record("selection=Feb,Mar")
        rec["truncated"] = True
        self.assertEqual([status(verify(rec), d) for d in "PSOE"], ["UNKNOWN"] * 4)

    def test_escaped_braces_quotes_keep_valid_spans(self):
        rec = record('A note {"x":"quoted"}; final selection=Feb,Mar')
        label = verify(rec)
        self.assertEqual(status(label, "S"), "PASS")
        span = label["dimensions"]["S"]["source_spans"][0]
        self.assertEqual(
            json.loads(rec["full_text"][span[0] : span[1]]), json.loads(rec["full_text"])["work"]
        )

    def test_duplicate_keys_cannot_be_recovered_as_positive(self):
        rec = record("selection=Feb,Mar")
        rec["full_text"] = rec["full_text"][:-1] + ',"answer":12}'
        self.assertEqual([status(verify(rec), d) for d in "PSOE"], ["UNKNOWN"] * 4)

    def test_invalid_protocol_never_upgraded_by_semantics(self):
        rec = record("selection=Feb,Mar")
        rec["full_text"] += " trailing prose"
        label = verify(rec)
        self.assertFalse(label["normalization"]["protocol_valid"])
        self.assertEqual(label["native_score"]["value_correct"], 1)

    def test_undefined_query_does_not_gain_training_eligibility(self):
        query = dict(QUERY, start=2, end=2, include_start=False)
        label = verify(
            record("final selection=[];diff operands=[];reduce=mean(operands)", answer=None),
            task(query=query),
        )
        self.assertFalse(label["normalization"]["defined"])

    def test_decimal_fraction_minus_are_exact(self):
        label = verify(
            record(
                "−1.5+0.5=-1; correction: diff operands=[10,14];reduce=mean(operands)",
                answer="24/2",
            )
        )
        self.assertEqual(status(label, "E"), "PASS")

    def test_units_not_silently_stripped(self):
        label = verify(record("diff operands=[10,14];reduce=mean(operands)", answer="12 kg"))
        self.assertEqual(status(label, "E"), "UNKNOWN")

    def test_observation_does_not_mutate_native_score(self):
        rec = record("selection=Feb,Mar")
        score = task_contract.score(rec["full_text"], WORLD, QUERY)
        before = copy.deepcopy(score)
        label = verify_semantics(task(), rec, score)
        label["native_score"]["J"] = 900
        self.assertEqual(score, before)

    def test_numeric_answer_collision_does_not_prove_series(self):
        query = dict(
            family="TOPK",
            selection_series="Beta",
            k=2,
            direction="largest",
            aggregate="sum",
            value="Alpha",
        )
        label = verify(
            record("selection_series=Alpha;final selection=Feb,Mar", answer=32), task(query=query)
        )
        self.assertEqual(status(label, "S"), "FAIL")


class ReplayTests(unittest.TestCase):
    def group(self):
        rows = []
        for i in range(8):
            rec = record(
                "selection=" + ("Feb,Mar" if i == 0 else "Jan,Mar"), answer=0, row_id=f"r{i}"
            )
            score = task_contract.score(rec["full_text"], WORLD, QUERY)
            rows.append(dict(task=task(), record=rec, score=score))
        return dict(
            group_id="g0",
            source="C_TRAIN_PROBE",
            block="b0",
            family="INTERVAL",
            world_id="w1",
            rows=rows,
        )

    def test_residual_bound_shuffle_multiset_and_scalar_whitening(self):
        result = replay_group(self.group(), native_std)
        select = result["bounded"]["SELECT"]
        self.assertTrue(result["select_vs_evid_changed"])
        self.assertLessEqual(max(map(abs, select["residual"])), 0.25)
        self.assertAlmostEqual(sum(select["residual"]), 0)
        self.assertEqual(len(result["shuffle"]), 10)
        for shuffled in result["shuffle"]:
            self.assertEqual(sorted(select["residual"]), sorted(shuffled["residual"]))
        self.assertGreater(result["scalar"]["PS_0.01"]["linf"], 2)

    def test_unknown_is_shared_abstention_across_arms(self):
        group = self.group()
        group["rows"][1]["record"] = record("not stated", answer=0, row_id="r1")
        result = replay_group(group, native_std)
        self.assertFalse(result["select_vs_evid_changed"])
        for arm in ("EVID", "SELECT"):
            self.assertEqual(result["bounded"][arm]["residual"], [0.0] * 8)

    def test_audit_is_blinded_and_c_gate_stays_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            receipt = prepare_blind_audit([self.group()], out, representative_n=2, positive_n=1)
            self.assertFalse(receipt["human_audit_complete"])
            blind = [
                json.loads(line)
                for line in (out / "BLINDED_RECORDS.jsonl").read_text().splitlines()
            ]
            for row in blind:
                self.assertFalse(
                    set(row) & {"model_id", "block", "arm", "native_score", "dimensions", "cohort"}
                )
            human = assess_human_audit(out / "HUMAN_REVIEW.csv", out / "private/SAMPLE_KEY.jsonl")
            self.assertFalse(human["human_audit_complete"])
            gate = write_c_gate(out / "GATE.json", {}, human_audit=human)
            self.assertEqual(gate["decision"], "BLOCK_C")

    def test_confirmation_is_refused_for_audit(self):
        group = self.group()
        group["source"] = "B_CONFIRM"
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            prepare_blind_audit([group], tmp)

    def test_stage_joins_files_and_reports_missing_history_without_false_completion(self):
        def put(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            put(root / "CONFIG.json", dict(old_root=str(root / "old")))
            put(root / "DATA/DATA_READY.json", dict(status="DATA_READY"))
            (root / "DATA/TASKS.jsonl").write_text(json.dumps(task()) + "\n")
            for source in ("B0", "B2_DEV", "C_TRAIN_PROBE"):
                for block in range(3):
                    rows = []
                    group_id = f"{source}_b{block}"
                    for draw in range(8):
                        rec = record("selection=Feb,Mar", answer=0, row_id=f"{group_id}/{draw}")
                        rec.update(
                            qid="q1", score=task_contract.score(rec["full_text"], WORLD, QUERY)
                        )
                        rows.append(rec)
                    put(
                        root / f"PROBES/{source}/b{block}/groups/one.json",
                        dict(group_id=group_id, block=block, source=source, rows=rows),
                    )
            result = replay_stage(root)
            self.assertEqual(result["status"], "INCOMPLETE_TECHNICAL")
            self.assertEqual(result["c_gate"], "BLOCK_C")
            self.assertEqual(result["replay"]["answers"], 72)
            self.assertEqual(result["golden"], "INCOMPLETE_OR_MISMATCH")
            self.assertTrue((root / "VERIFIER/audit/HUMAN_REVIEW.csv").exists())


if __name__ == "__main__":
    unittest.main()
