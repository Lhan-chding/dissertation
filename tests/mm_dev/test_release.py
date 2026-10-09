from __future__ import annotations

import contextlib
import copy
import importlib.util
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mm_core.contracts import score
from mm_core.scoring import json_exact
from mm_dev import release as release_module
from mm_dev.analysis import response_analysis, score_panel
from mm_dev.contract import CELLS, STRATA, run_matrix
from mm_dev.release import (
    STATES,
    Audit,
    RecountFailure,
    check_joint_table,
    check_responses,
    check_root_equal,
    derive_training_aggregates,
    derived_manifests,
    descriptor,
    encoded,
    event_counts,
    model_panels,
    path_inside,
    publish,
    recount_panel,
    recount_response,
    release,
    required_analysis_files,
    verify_analysis_manifest,
)

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "docs/mm_dev_f2/design/config/MM_DEV_F2.json"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        path.write_text("".join(json.dumps(json_exact(x), sort_keys=True) + "\n" for x in value))
    else:
        path.write_bytes(encoded(json_exact(value)))


def fixture_payload():
    questions, outputs = [], []
    for chart, operation in STRATA:
        truth = [30, 20, 40] if operation == "range" else [30, 20]
        for cell in CELLS:
            qid = f"{chart}-{operation}-{cell}"
            q = dict(
                question_id=qid,
                root_index=0,
                root_family_id="root-0",
                numeric_root_id="root-0",
                split="DEV_EVAL",
                chart_type=chart,
                operation=operation,
                true_values_decimal=list(map(str, truth)),
                delta_decimal="1",
                axis_min="0",
                axis_max="100",
                readset_id=qid,
                visual_level="low" if cell[0] == "L" else "high",
                numeric_level="low" if cell[1] == "L" else "high",
                image_hash=qid,
                source_graph_hash="source-0",
                ordered_item_ids=list(map(str, range(len(truth)))),
                quantity_units="count",
            )
            questions.append(q)
            for k in range(4):
                outputs.append(
                    dict(
                        request_id=f"{qid}-{k}",
                        question_id=qid,
                        stage="DEV_EVAL",
                        sample_index=k,
                        status="completed",
                        model_hash="checkpoint",
                        raw_text=json.dumps(
                            dict(
                                readings=truth,
                                answer={"sum": 50, "difference": 10, "range": 20}[operation],
                            )
                        )
                        if k != 3
                        else '{"answer":0}',
                        tokens=[],
                        truncated=False,
                    )
                )
    return dict(
        questions=questions,
        outputs=outputs,
        self_scores=[],
        gold_scores=[],
        complete=True,
        identity={"checkpoint_hash": "checkpoint"},
        technical_events=[],
    )


def response_fixture(root, roots=4):
    plan = json.loads(PLAN.read_text())
    plan["data"]["root_counts"]["DEV_EVAL"] = roots
    plan["statistics"]["crossfit_root_halves"] = {
        "half0_indices": list(range(roots // 2)),
        "half1_indices": list(range(roots // 2, roots)),
    }
    rng = np.random.default_rng(19)
    cubes = {s: rng.integers(0, 5, (roots, 6, 4, 3)) / 4 for s in STATES}
    for run in run_matrix():
        if run["phase"] == "CONTINUE":
            cubes[run["run_id"]] = rng.integers(0, 5, (roots, 6, 4, 3)) / 4
    primary = response_analysis(cubes, plan)
    families = {i: f"root-{i}" for i in range(roots)}
    for row in primary["root_responses"]:
        row["evaluation_root_family_id"] = families[row["evaluation_root_index"]]
    files = {
        "PRIMARY_EFFECTS.json": primary["primary"],
        "RESPONSE_BY_STATE_ACTION_FUTURE_STRATUM.jsonl": primary["responses"],
        "ABSOLUTE_AND_RELATIVE_CHANGES.jsonl": primary["changes"],
        "RESPONSE_BY_ROOT.jsonl": primary["root_responses"],
        "UTILITY_AND_FEASIBILITY.json": primary["utilities"],
        "PREPARATION_STATE_DIFFERENCES.json": primary["preparations"],
    }
    for name, value in files.items():
        save(root / "summary" / name, value)
    return cubes, families, plan


class ResponseRecountTests(unittest.TestCase):
    def test_matches_frozen_scorer_with_independent_arithmetic(self):
        question = fixture_payload()["questions"][0]
        examples = [
            '{"readings":[30,20],"answer":50}',
            '{"readings":[30,20],"answer":49}',
            '{"readings":[29,21],"answer":50}',
            '{"readings":[29,20],"answer":49}',
            '{"readings":[29,20],"answer":50}',
            '{"readings":[29,20],"answer":48}',
            '{"readings":[30.5,20.5],"answer":50.5}',
            '{"answer":50}',
            '{"readings":[30,20]}',
            '{"readings":[99999,-1000],"answer":0}',
            "not json",
        ]
        for operation in ("sum", "difference", "range"):
            question["operation"] = operation
            for text in examples:
                with self.subTest(operation=operation, text=text):
                    raw = dict(raw_text=text, status="completed", truncated=False)
                    actual = recount_response(question, raw)
                    old = score(text, [30, 20], operation, "1", (0, 100))
                    old.pop("parsed")
                    self.assertEqual(
                        {k: actual[k] for k in old},
                        {k: list(v) if isinstance(v, tuple) else v for k, v in old.items()},
                    )

    def test_missing_is_not_zero_cell_and_partial_full_denominator(self):
        q = fixture_payload()["questions"][0]
        r = recount_response(q, dict(raw_text='{"answer":50}', status="completed", truncated=False))
        self.assertTrue(r["A_full"])
        self.assertFalse(r["L"])
        for field in ("cell_exact", "cell_tolerance", "e", "S", "eps_O"):
            self.assertIsNone(r[field])
        self.assertEqual(event_counts([r])["missing_joint_fields_not_000"], 1)
        self.assertEqual(event_counts([r])["exact_counts"]["000"], 0)

    def test_incorrect_epsilon_is_detected_exactly(self):
        q = fixture_payload()["questions"][0]
        expected = recount_response(
            q,
            dict(raw_text='{"readings":[29,20],"answer":50}', status="completed", truncated=False),
        )
        saved = json_exact(expected)
        saved["eps_C"]["numerator"] += 1
        audit = Audit()
        audit.check(saved, expected, "response")
        with self.assertRaises(RecountFailure):
            audit.finish()
        self.assertIn("eps_C", audit.mismatches[0]["location"])

    def test_rational_zero_denominator_and_numeric_boolean_rejected(self):
        audit = Audit()
        audit.check({"numerator": 1, "denominator": 0, "float": 1}, Fraction(1), "bad")
        audit.check(True, 1, "bool_count")
        self.assertEqual(len(audit.mismatches), 2)

    def test_raw_scored_panel_binding_and_group_counts(self):
        payload = fixture_payload()
        scored = score_panel(payload, state_id="S0", panel="DEV_EVAL")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scored.jsonl"
            save(path, scored["scored_outputs"])
            audit = Audit()
            cube, rows, families = recount_panel(payload, path, "DEV_EVAL", "S0", 1, audit)
            check_joint_table(
                dict(counts=scored["tables"]["COUNTS"], coverage=scored["tables"]["COVERAGE"]),
                rows,
                audit,
                "counts",
            )
            check_root_equal(
                dict(root_equal=scored["tables"]["ROOT_EQUAL"]), rows, audit, "root_equal"
            )
            audit.finish()
            self.assertEqual(families, {0: "root-0"})
            self.assertTrue(np.all(cube == 0.75))
            bad = copy.deepcopy(scored["tables"]["COUNTS"])
            bad["cohorts"][0]["overall"]["exact_counts"]["111"] += 1
            check_joint_table(
                dict(counts=bad, coverage=scored["tables"]["COVERAGE"]), rows, audit, "counts"
            )
            with self.assertRaises(RecountFailure):
                audit.finish()

    def test_missing_raw_slot_duplicate_slot_and_incomplete_rejected(self):
        payload = fixture_payload()
        scored = score_panel(payload, state_id="S0", panel="DEV_EVAL")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scored.jsonl"
            save(path, scored["scored_outputs"])
            for mutation in ("missing", "duplicate", "incomplete"):
                with self.subTest(mutation=mutation):
                    modified = copy.deepcopy(payload)
                    if mutation == "missing":
                        modified["outputs"].pop()
                    elif mutation == "duplicate":
                        modified["outputs"].append(modified["outputs"][0])
                    else:
                        modified["complete"] = False
                    audit = Audit()
                    with self.assertRaises(RecountFailure):
                        recount_panel(modified, path, "DEV_EVAL", "S0", 1, audit)
                        audit.finish()

    def test_wrong_raw_text_is_not_hidden_by_preserved_indicator(self):
        payload = fixture_payload()
        scored = score_panel(payload, state_id="S0", panel="DEV_EVAL")
        scored["scored_outputs"][0]["raw_text"] = "changed"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scored.jsonl"
            save(path, scored["scored_outputs"])
            audit = Audit()
            recount_panel(payload, path, "DEV_EVAL", "S0", 1, audit)
            with self.assertRaises(RecountFailure):
                audit.finish()

    def test_all_response_points_and_shared_bootstrap_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cubes, families, plan = response_fixture(root)
            audit = Audit()
            result = check_responses(root, cubes, families, plan, audit)
            audit.finish()
            self.assertEqual(result["response_rows"], 240)
            self.assertEqual(result["cell_rows"], 960)
            self.assertEqual(result["root_rows"], 960)
            self.assertEqual(result["bootstrap_replicates"], 5000)

    def test_wrong_response_root_binding_and_ci_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cubes, families, plan = response_fixture(root)
            primary = json.loads((root / "summary/PRIMARY_EFFECTS.json").read_text())
            primary["contrasts"][0]["CI98_75"]["lower"] += 0.01
            primary["contrasts"][1]["future_training_SE"] = 0
            save(root / "summary/PRIMARY_EFFECTS.json", primary)
            families[0] = "wrong-family"
            audit = Audit()
            check_responses(root, cubes, families, plan, audit)
            self.assertTrue(any("CI98_75" in x["location"] for x in audit.mismatches))
            self.assertTrue(any("future_training_SE" in x["location"] for x in audit.mismatches))
            self.assertTrue(
                any("evaluation_root_family_id" in x["location"] for x in audit.mismatches)
            )
            with self.assertRaises(RecountFailure):
                audit.finish()


class ReleaseGateTests(unittest.TestCase):
    def test_readiness_change_during_recount_prevents_publication(self):
        before = {"status": "COMPLETE", "evidence_set_sha256": "before"}
        plan = json.loads(PLAN.read_text())
        panels = model_panels()
        identities = [dict(panel=panel, state_id=state, identity={}) for panel, state in panels]
        sample = dict(A_full=True, P_full=True, J_full=True, L=True, cell_exact="111")

        def panel_result(payload, path, panel, state, roots, audit):
            return (
                np.zeros((roots, 6, 4, 3)),
                [sample] * (6144 if panel == "PROBE" else 12288),
                {i: f"root-{i}" for i in range(roots)},
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save(root / "summary/RUN_COMPLETENESS.json", before)
            save(root / "summary/ANALYSIS_MANIFEST.json", {})
            for name in ("JOINT_EVENT_TABLES", "NUMERIC_AND_GEOMETRY_SUPPORT"):
                save(root / f"summary/{name}.json", identities)
            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    patch.object(
                        release_module.common,
                        "verify_analysis_readiness",
                        side_effect=[before, {**before, "evidence_set_sha256": "changed"}],
                    )
                )
                stack.enter_context(
                    patch.object(release_module.common, "load_plan", return_value=plan)
                )
                stack.enter_context(
                    patch.object(
                        release_module.common,
                        "verify_execution",
                        return_value={"source_hashes": {}},
                    )
                )
                for name in ("verify_analysis_manifest", "check_responses"):
                    stack.enter_context(patch.object(release_module, name, return_value={}))
                for name in (
                    "derived_manifests",
                    "derive_training_aggregates",
                    "evidence_inventory",
                ):
                    stack.enter_context(patch.object(release_module, name, return_value=[]))
                for name in ("check_joint_table", "check_root_equal"):
                    stack.enter_context(patch.object(release_module, name))
                stack.enter_context(
                    patch.object(release_module, "load_panel", return_value={"identity": {}})
                )
                stack.enter_context(
                    patch.object(release_module, "recount_panel", side_effect=panel_result)
                )
                with self.assertRaises(RecountFailure):
                    release(PLAN, root)
            self.assertFalse((root / "verification/RELEASE_MANIFEST.json").exists())
            report = json.loads(next((root / "verification/failed").glob("*.json")).read_text())
            self.assertIn("EVIDENCE_CHANGED_DURING_RECOUNT", str(report["mismatches"]))

    def test_manifest_requires_all_40_scored_files_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(len(model_panels()), 40)
            for name in required_analysis_files():
                path = path_inside(root, name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture\n")
            manifest = dict(
                status="ANALYZED",
                scientific_success_claimed=False,
                next_stage_authorized=False,
                files=[descriptor(root, name) for name in sorted(required_analysis_files())],
            )
            save(root / "summary/ANALYSIS_MANIFEST.json", manifest)
            audit = Audit()
            verify_analysis_manifest(root, audit)
            audit.finish()
            (root / manifest["files"][0]["path"]).write_text("tampered")
            verify_analysis_manifest(root, audit)
            with self.assertRaises(RecountFailure):
                audit.finish()

    def test_manifest_missing_file_or_success_claim_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save(
                root / "summary/ANALYSIS_MANIFEST.json",
                dict(
                    status="ANALYZED",
                    files=[],
                    scientific_success_claimed=True,
                    next_stage_authorized=True,
                ),
            )
            audit = Audit()
            verify_analysis_manifest(root, audit)
            self.assertEqual(len(audit.mismatches), 3)

    def test_failed_readiness_preserved_without_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(
                    release_module.common,
                    "verify_analysis_readiness",
                    side_effect=PermissionError("incomplete registered path"),
                ),
                self.assertRaises(PermissionError),
            ):
                release(PLAN, root)
            self.assertFalse((root / "verification/RELEASE_MANIFEST.json").exists())
            failures = list((root / "verification/failed").glob("*.json"))
            self.assertEqual(len(failures), 1)
            self.assertEqual(json.loads(failures[0].read_text())["status"], "FAIL")

    def test_missing_raw_preserved_without_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save(root / "summary/RUN_COMPLETENESS.json", {})
            save(root / "summary/JOINT_EVENT_TABLES.json", [])
            save(root / "summary/NUMERIC_AND_GEOMETRY_SUPPORT.json", [])
            with (
                patch.object(release_module.common, "verify_analysis_readiness", return_value={}),
                patch.object(
                    release_module.common, "load_plan", return_value=json.loads(PLAN.read_text())
                ),
                patch.object(release_module.common, "verify_execution", return_value={}),
                patch.object(release_module, "verify_analysis_manifest", return_value={}),
                patch.object(
                    release_module,
                    "load_panel",
                    side_effect=FileNotFoundError("missing raw segment"),
                ),
                self.assertRaises(FileNotFoundError),
            ):
                release(PLAN, root)
            self.assertFalse((root / "verification/RELEASE_MANIFEST.json").exists())
            self.assertFalse((root / "orchestration/completions/RELEASE.json").exists())

    def test_immutable_publication_and_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            publish(root, "verification/a.json", b"same")
            publish(root, "verification/a.json", b"same")
            with self.assertRaises(FileExistsError):
                publish(root, "verification/a.json", b"different")
            for name in ("../escape", "/absolute"):
                with self.assertRaises(ValueError):
                    path_inside(root, name)
            (root / "link").symlink_to(root / "verification")
            with self.assertRaises(ValueError):
                path_inside(root, "link/a.json")

    def test_checkpoint_milestones_required_and_rotation_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save(root / "manifests/F2_FREEZE.json", {"status": "fixture"})
            fake_run = dict(run_id="fixture", schedule_id="PREP_p0")
            for step in range(33):
                name = f"training/fixture/checkpoints/step-{step:02d}.pt"
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                data = f"checkpoint{step}".encode()
                sha = __import__("hashlib").sha256(data).hexdigest()
                if step in {0, 8, 16, 24, 31, 32}:
                    path.write_bytes(data)
                receipt = dict(step=step, path=path.name, sha256=sha, state_hash="recorded-only")
                save(path.parent / f"commit-{step:02d}.json", receipt)
                if step == 32:
                    save(path.parent / "LATEST.json", receipt)

            synthetic_plan = root / "design/config/MM_DEV_F2.json"
            save(synthetic_plan, {})
            save(synthetic_plan.parent / "run_matrix.json", [fake_run])
            for source in (PLAN.parent / "schedules").glob("*.jsonl"):
                target = synthetic_plan.parent / "schedules" / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
            with patch.object(release_module, "run_matrix", return_value=[fake_run]):
                derived_manifests(synthetic_plan, root, dict(input_hashes={}))
                index = json.loads((root / "manifests/CHECKPOINT_INDEX.json").read_text())
                self.assertEqual(
                    index["checkpoints"][1]["retention_status"],
                    "PLANNED_ROTATION_COMMIT_RECEIPT_RETAINED",
                )
                self.assertFalse(index["complete_weights_in_lightweight_package"])
                (root / "training/fixture/checkpoints/step-08.pt").unlink()
                with self.assertRaisesRegex(RecountFailure, "MISSING_RETAINED_CHECKPOINT"):
                    derived_manifests(synthetic_plan, root, dict(input_hashes={}))

    def test_derived_aggregates_keep_original_paths_hashes_and_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in range(1, 33):
                save(
                    root / f"training/fixture/steps/{step:02d}.json",
                    dict(logical_step=step, original="full update"),
                )
                for slot in range(24):
                    for k in range(8):
                        save(
                            root / f"training/fixture/rollouts/{step:02d}-{slot:02d}-{k}.json",
                            dict(
                                logical_step=step, raw_text="unfavorable", slot=slot, sample_index=k
                            ),
                        )
            save(root / "training/fixture/failures/keep.json", {"failure": True})
            with patch.object(release_module, "run_matrix", return_value=[dict(run_id="fixture")]):
                paths = derive_training_aggregates(root)
                derive_training_aggregates(root)
            self.assertEqual(len(paths), 2)
            rows = [json.loads(x) for x in (root / paths[0]).read_text().splitlines()]
            self.assertEqual(len(rows), 6144)
            self.assertEqual(rows[0]["record"]["raw_text"], "unfavorable")
            self.assertEqual(rows[0]["source"], descriptor(root, rows[0]["source"]["path"]))
            self.assertTrue((root / "training/fixture/failures/keep.json").exists())

    def test_cli_rejects_unregistered_identity_before_release(self):
        spec = importlib.util.spec_from_file_location(
            "verify_release_cli", ROOT / "scripts/mm_dev/verify_release.py"
        )
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save(root / "orchestration/REGISTRATION.json", {"tasks": {"RELEASE": {}}})
            with (
                patch("sys.argv", ["verify_release", "--plan", str(PLAN), "--run-root", str(root)]),
                patch.dict("os.environ", {}, clear=True),
                patch.object(cli, "release") as mocked,
            ):
                with self.assertRaises(PermissionError):
                    cli.main()
                mocked.assert_not_called()

    def test_cli_only_completes_successful_unpromoted_release(self):
        spec = importlib.util.spec_from_file_location(
            "verify_release_cli_completion", ROOT / "scripts/mm_dev/verify_release.py"
        )
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        registration = {"tasks": {"RELEASE": {}}}
        for status in ("VERIFIED_RELEASE", "FAIL"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                save(root / "orchestration/REGISTRATION.json", registration)
                receipt = dict(
                    status=status,
                    files=[],
                    scientific_success_claimed=False,
                    next_stage_authorized=False,
                )
                with (
                    patch(
                        "sys.argv", ["verify_release", "--plan", str(PLAN), "--run-root", str(root)]
                    ),
                    patch.dict(
                        "os.environ",
                        {
                            "MM_DEV_TASK_ID": "RELEASE",
                            "MM_DEV_ATTEMPT_ID": "RELEASE_attempt0001",
                            "MM_DEV_REGISTRATION_HASH": cli.digest(registration),
                        },
                    ),
                    patch.object(cli, "worker_lease", return_value=contextlib.nullcontext()),
                    patch.object(cli, "release", return_value=receipt),
                    patch.object(cli, "complete_task") as complete,
                    patch.object(cli, "fail_task") as fail,
                ):
                    if status == "FAIL":
                        with self.assertRaises(PermissionError):
                            cli.main()
                        complete.assert_not_called()
                        fail.assert_called_once()
                    else:
                        cli.main()
                        complete.assert_called_once()
                        fail.assert_not_called()


if __name__ == "__main__":
    unittest.main()
