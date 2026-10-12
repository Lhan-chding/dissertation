"""Independently verify released SR-F1.2 statistics without production analysis imports.

Usage: python independent_verify.py EVIDENCE_DIR ANALYSIS_DIR OUTPUT_JSON
Reads authenticated released scores only; never reads raw or sealed answers.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import shlex
import statistics
import sys

import numpy as np


FAMILIES = ("CROSS", "INTERVAL", "THRESHOLD", "TOPK")
SEEDS = (71001, 71002, 71003)
BASELINE = "SRF1_2_COMMON_ZERO"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_hash(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def verify(evidence, analysis_dir):
    complete = read(evidence / "endpoint_orchestration/COMPLETE.json")
    acceptance = read(evidence / "endpoint_orchestration/MAIN_ACCEPTANCE.json")
    state = read(evidence / "endpoint_orchestration/STATE.json")
    assert complete["status"] == "EVALUATIONS_COMPLETE_REPORT_PENDING"
    assert acceptance["status"] == "ALL_TWELVE_TERMINAL_AND_STATE_VERIFIED"
    assert state["endpoints_complete"] == 13 and state["scores_sealed"] is False
    expected = {BASELINE} | {f"SRF1_2_{a}_s{s}" for a in ("A", "J", "GATE", "DEC") for s in SEEDS}
    assert {item["model_id"] for item in complete["releases"]} == expected
    rows, inputs = {}, []
    for item in complete["releases"]:
        path = evidence / "released/evaluation" / f"{item['model_id']}.json"
        data = read(path)
        assert object_hash(data) == item["release_sha256"]
        assert data["status"] == "RELEASED_AFTER_ALL_MAIN_COMPLETE"
        assert data["authorization_sha256"] == object_hash(acceptance["main_completion"])
        assert data["count"] == item["records"] == len(data["records"])
        assert data["records_sha256"] == object_hash(data["records"])
        assert len({r["slot_id"] for r in data["records"]}) == data["count"]
        assert all(r["model_id"] == item["model_id"] for r in data["records"])
        rows[item["model_id"]] = data["records"]
        inputs.append(dict(file=str(path.relative_to(evidence)), sha256=sha(path), records=data["count"]))
    assert len(rows) == 13 and sum(map(len, rows.values())) == complete["endpoint_records"] == 227700
    analysis = read(analysis_dir / "ANALYSIS.json")
    primary_file = read(analysis_dir / "PRIMARY_CONTRASTS.json")
    assert analysis["primary"] == primary_file
    values, root_family, panels = {}, {}, defaultdict(list)
    for model, records in rows.items():
        questions = defaultdict(list)
        for row in records:
            panels[(model, row["pool"], row["protocol"], row["view"])].append(row)
            assert all(not isinstance(v, (float, int)) or math.isfinite(v) for v in row["score"].values())
            if row["pool"] == "TEST_ID" and row["protocol"] == "evidence_answer" and row["view"] is None:
                assert row["step"] == (0 if model == BASELINE else 96)
                questions[(row["root_id"], row["qid"])].append(row)
                assert root_family.setdefault(row["root_id"], row["family"]) == row["family"]
        roots = defaultdict(list)
        for (root, _), records in questions.items():
            assert sorted(r["draw"] for r in records) == [0, 1, 2, 3]
            roots[root].append(statistics.mean(r["score"]["J"] for r in records))
        assert len(roots) == 128 and all(len(v) == 16 for v in roots.values())
        values[model] = {root: statistics.mean(v) for root, v in roots.items()}
    assert Counter(root_family.values()) == {f: 32 for f in FAMILIES}
    roots = sorted(root_family)
    primary = []
    assert [x["contrast"] for x in primary_file] == ["J-A", "GATE-J", "GATE-DEC", "DEC-J"]
    for item in primary_file:
        left, right = item["contrast"].split("-")
        delta = np.array([[values[f"SRF1_2_{left}_s{s}"][r] - values[f"SRF1_2_{right}_s{s}"][r]
                           for r in roots] for s in SEEDS])
        effects = np.array([statistics.mean(statistics.mean(row[[root_family[r] == f for r in roots]])
                                             for f in FAMILIES) for row in delta]) * 100
        rng = np.random.default_rng(812901)
        draws = np.zeros(5000)
        for family in FAMILIES:
            indices = [i for i, root in enumerate(roots) if root_family[root] == family]
            samples = rng.choice(indices, (5000, len(indices)), replace=True)
            draws += delta.mean(axis=0)[samples].mean(axis=1) * 25
        ci95 = np.quantile(draws, [.025, .975])
        ci98 = np.quantile(draws, [.00625, .99375])
        assert math.isclose(effects.mean(), item["effect_pp"], abs_tol=1e-12)
        for actual, expected_values in [(effects, item["seed_effects_pp"]), (ci95, item["CI95_pp"]),
                                        (ci98, item["CI98_75_pp"])]:
            assert np.allclose(actual, expected_values, rtol=0, atol=1e-12)
        primary.append(dict(contrast=item["contrast"], effect_pp=float(effects.mean()),
                            seed_effects_pp=effects.tolist(), CI95_pp=ci95.tolist(), CI98_75_pp=ci98.tolist()))
    metric_count = 0
    for item in analysis["absolute"]:
        records = panels[(item["model"], item["pool"], item["protocol"], item["view"])]
        for metric, reported in item["metrics"].items():
            if item["pool"] == "CHARTQA":
                actual = statistics.mean(r["score"][metric] for r in records)
            else:
                questions = defaultdict(list)
                for row in records:
                    questions[(row["family"], row["root_id"], row["qid"])].append(row["score"][metric])
                grouped_roots, families = defaultdict(list), defaultdict(list)
                for (family, root, _), scores in questions.items():
                    grouped_roots[(family, root)].append(statistics.mean(scores))
                for (family, _), scores in grouped_roots.items():
                    families[family].append(statistics.mean(scores))
                actual = statistics.mean(statistics.mean(v) for v in families.values())
            assert math.isclose(actual, reported, rel_tol=0, abs_tol=1e-12)
            metric_count += 1
        support = [r for r in records if r["score"].get("P") == 1]
        denominator = len(support)
        numerator = sum(r["score"]["A"] == 0 for r in support)
        conditional = item["conditional"]
        assert conditional["P_support"] == denominator
        assert conditional["A_fail_given_P_count"] == numerator
        assert conditional["A_fail_given_P_rate"] == (numerator / denominator if denominator else None)
    for pool in ("TEST_COMPOSITION", "TEST_LANGUAGE", "TEST_RENDER"):
        reference = None
        for model in sorted(expected):
            records = panels[(model, pool, "evidence_answer", None)]
            keys = {(r["root_id"], r["qid"], r["draw"], r["family"]) for r in records}
            assert len(keys) == len(records) == 512 and len({k[0] for k in keys}) == 64
            if reference is None:
                reference = keys
            else:
                assert keys == reference
    chart = {m: panels[(m, "CHARTQA", "plain_answer", None)] for m in expected}
    for records in chart.values():
        assert len(records) == len({r["qid"] for r in records}) == 2500
        assert Counter(r["variant"] for r in records) == {"human": 1250, "augmented": 1250}
        assert len({r["root_id"] for r in records}) == 1509
    indexed = {m: {r["qid"]: r for r in records} for m, records in chart.items()}
    reference = indexed[BASELINE]
    clusters = defaultdict(list)
    for qid in sorted(reference):
        for model in expected:
            assert indexed[model][qid]["root_id"] == reference[qid]["root_id"]
        clusters[reference[qid]["root_id"]].append(qid)
    clusters = list(clusters.values())
    sizes = np.array([len(c) for c in clusters])
    sums = np.array([[sum(indexed[f"SRF1_2_GATE_s{s}"][q]["score"]["relaxed_accuracy"]
                         - reference[q]["score"]["relaxed_accuracy"] for q in c)
                      for c in clusters] for s in SEEDS])
    rng = np.random.default_rng(812901)
    sampled = rng.integers(len(clusters), size=(5000, len(clusters)))
    draws = sums.mean(axis=0)[sampled].sum(axis=1) / sizes[sampled].sum(axis=1) * 100
    point = float(sums.mean(axis=0).sum() / sizes.sum() * 100)
    ci = np.quantile(draws, [.025, .975])
    item = next(x for x in analysis["chartqa_changes"] if x["arm"] == "GATE"
                and x["split"] == "all" and x["metric"] == "relaxed_accuracy")
    assert math.isclose(point, item["effect_pp"], abs_tol=1e-12)
    assert np.allclose(ci, item["CI95_pp"], rtol=0, atol=1e-12)
    return dict(status="INDEPENDENT_RELEASED_ANALYSIS_VERIFIED", production_analysis_imported=False,
                models=13, records=227700, input_files=inputs, absolute_panels=len(analysis["absolute"]),
                absolute_metrics_verified=metric_count, conditional_denominators_verified=True,
                test_id=dict(roots_per_model=128, questions_per_root=16, draws_per_question=4, roots_per_family=32),
                shift_panels=dict(matched_models=13, records_per_model=512, roots_per_model=64),
                primary=primary, bootstrap_replicates=5000, bootstrap_seed=812901,
                chartqa=dict(questions_per_model=2500, images=1509, human=1250, augmented=1250,
                             verified_contrast="GATE-BASELINE", metric="relaxed_accuracy", split="all",
                             effect_pp=point, CI95_pp=ci.tolist()), tolerance_absolute=1e-12,
                analysis_sha256=sha(analysis_dir / "ANALYSIS.json"), primary_sha256=sha(analysis_dir / "PRIMARY_CONTRASTS.json"),
                limitations=["CI conditions on the three realised seeds; no training-seed population inference.",
                             "Conditional A failure given P uses raw-draw support, not balanced-family weighting.",
                             "This verifies frozen released scores, not the post-hoc ChartQA whitespace sensitivity."])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("analysis_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = verify(args.evidence, args.analysis_dir)
    result["verification_command"] = shlex.join([sys.executable, *sys.argv])
    result["verifier_sha256"] = sha(Path(__file__))
    result["numpy_version"] = np.__version__
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "models", "records", "absolute_panels", "absolute_metrics_verified")}))
