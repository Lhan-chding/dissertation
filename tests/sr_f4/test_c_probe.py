import copy
import json

import pytest

from sr_f4 import c_probe, c_probe_data, task_contract
from sr_f4.common import read, rows, write


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    from sr_f3_ce import rendering

    monkeypatch.setattr(
        rendering,
        "render_chart",
        lambda task: (b"test-image", {"category_count": len(task["world"]["categories"])}),
    )
    source = tmp_path / "source"
    write(
        source / "CONFIG.json",
        dict(
            model_path="/registered/model",
            parents={str(i): f"/registered/b{i}" for i in range(3)},
            python="/registered/python",
            qos="soujanya-poria-startfund-2026-03",
            account="rose",
            partition="cluster02",
        ),
    )
    root = tmp_path / "probe"
    c_probe.prepare(root, source, "test-revision")
    return root, source


def test_draws_unique_shards_complete_and_live_prepare_refused(prepared):
    root, source = prepared
    plan = read(root / "DRAW_PLAN.json")
    groups = [g for s in plan for g in s["assignments"]]
    assert len(groups) == len({g["group_id"] for g in groups}) == 192
    assert len({seed for g in groups for seed in g["seeds"]}) == 1536
    for spec in c_probe.task_matrix()[:-1]:
        assert len(c_probe.assignments(root, spec["block"], spec["shard"])) == 16
    assert len(read(root / "TASKS.json")[-1]["dependencies"]) == 12
    assert read(root / "READY.json")["images"] == 32
    with pytest.raises(FileExistsError):
        c_probe.prepare(root, source, "different")
    changed = copy.deepcopy(read(root / "FROZEN.json"))
    changed["generation_seed"] += 1
    write(root / "FROZEN.json", changed)
    with pytest.raises(ValueError, match="prospective"):
        c_probe.require_frozen(root)


def test_full_budget_report_and_corrupted_resume_identity(prepared, monkeypatch):
    root, _ = prepared
    tasks = {t["qid"]: t for t in rows(root / "DATA/TASKS.jsonl")}
    incomplete = c_probe.report(root)
    assert incomplete["status"] == "INCOMPLETE_TECHNICAL"
    assert incomplete["counts"]["all"]["planned_answers"] == 1536
    for spec in read(root / "DRAW_PLAN.json"):
        for assignment in spec["assignments"]:
            task = tasks[assignment["qid"]]
            world = task["world"]
            text = json.dumps(
                dict(
                    evidence=[
                        dict(series=s, category=c, value=vs[i])
                        for s, vs in world["series"].items()
                        for i, c in enumerate(world["categories"])
                    ],
                    work="Selected: " + ", ".join(task["target_sets"]),
                    answer=task["gold_answer"],
                )
            )
            records = [
                dict(
                    full_text=text,
                    generation_status="COMPLETE",
                    finish_reason="eos",
                    qid=task["qid"],
                    row_id=f"{assignment['group_id']}/d{draw:02d}",
                    group_id=assignment["group_id"],
                    draw=draw,
                    seed=seed,
                    source=c_probe_data.SPLIT,
                    block=int(assignment["block"][1:]),
                    arm="PARENT",
                    score=task_contract.score(text, world, task["query"]),
                )
                for draw, seed in enumerate(assignment["seeds"])
            ]
            write(
                root
                / "PROBES"
                / assignment["block"]
                / "groups"
                / f"{assignment['group_index']:05d}.json",
                dict(group_id=assignment["group_id"], rows=records),
            )
    result = c_probe.report(root)
    assert result["answers"] == 1536 and result["groups"] == 192
    assert result["N_sig"] == 0
    summary = read(root / "ANALYSIS/SUMMARY.json")
    assert summary["strata"]["all"]["F_PRIME_KNOWN_75"]["candidate_rows"] == 0
    assert summary["native_statistics"]["all"]["rates"]["J"] == 1
    assert summary["native_statistics"]["all"]["rates"]["certificate_sufficient"] == 1
    assert not (root / "C_GATE_REPORT.json").exists()
    original = {
        p: (root / p).read_bytes()
        for p in ("REPORT_ZH.md", "ANALYSIS/SUMMARY.json", "ANALYSIS/COVERAGE.json")
    }
    from sr_f4 import verification_v3

    monkeypatch.setattr(
        verification_v3, "VERIFIER_VERSION", "srf4-observable-v3-programmatic-dev-r6"
    )
    sensitivity = root.parent / "r6"
    result6 = c_probe.report(root, measurement="r6", output=sensitivity)
    assert result6["status"] == "REMEASUREMENT_COMPLETE_NOT_C_TRAINING_AUTHORIZATION"
    summary6 = read(sensitivity / "ANALYSIS/SUMMARY.json")
    assert summary6["answers"] == 1536 and summary6["new_model_calls"] == 0
    assert summary6["verifier_version"].endswith("r6")
    assert summary6["frozen_verifier_version"].endswith("r5")
    assert all((root / p).read_bytes() == content for p, content in original.items())
    with pytest.raises(FileExistsError):
        c_probe.report(root, measurement="r6", output=sensitivity)
    with pytest.raises(ValueError, match="frozen run"):
        c_probe.report(root, measurement="r6", output=root / "ANALYSIS/r6")
    with pytest.raises(ValueError, match="separate output"):
        c_probe.report(root, measurement="r6")
    group = root / "PROBES/b0/groups/00000.json"
    saved = read(group)
    saved["rows"][0]["generation_status"] = "TECHNICAL_INVALID"
    write(group, saved)
    assert c_probe.report(root)["status"] == "INCOMPLETE_TECHNICAL"
    assert read(root / "ANALYSIS/COVERAGE.json")["technical_invalid_answers"] == 1
    assert read(root / "ANALYSIS/SUMMARY.json")["answers"] == 1536
    saved["rows"][0]["generation_status"] = "COMPLETE"
    saved["rows"][0]["seed"] += 1
    write(group, saved)
    with pytest.raises(ValueError, match="identity"):
        c_probe.report(root)


def test_changed_parent_reference_fails_before_model_load(prepared):
    root, _ = prepared
    config = read(root / "CONFIG.json")
    config["parents"]["0"] = "/different/parent"
    write(root / "CONFIG.json", config)
    with pytest.raises(ValueError, match="freeze"):
        c_probe.require_frozen(root)


def test_probe_stays_on_r5_when_development_verifier_changes(prepared, monkeypatch):
    from sr_f4 import verification_v3

    root, _ = prepared
    monkeypatch.setattr(verification_v3, "VERIFIER_VERSION", "future-development-version")
    assert c_probe.require_frozen(root)["measurement_version"].endswith("r5")
    assert c_probe.report(root)["status"] == "INCOMPLETE_TECHNICAL"
    with pytest.raises(ValueError, match="r6 development verifier"):
        c_probe.report(root, measurement="r6", output=root.parent / "future")


def test_slice_runs_registered_draws_and_resumes_without_new_calls(prepared, monkeypatch):
    from sr_f4 import runtime

    root, _ = prepared
    calls = []

    class FakeRuntime:
        def generate_group(self, model_input, run_root, seeds, **kwargs):
            assert set(model_input) == {"prompt", "image_file", "image_sha256"}
            calls.extend(seeds)
            return [
                dict(
                    seed=seed,
                    protocol="structured",
                    full_text='{"evidence":[],"answer":0}',
                    finish_reason="eos",
                    generation_status="COMPLETE",
                )
                for seed in seeds
            ]

    monkeypatch.setattr(runtime, "load", lambda *a, **kw: (FakeRuntime(), None, None))
    spec = c_probe.task_matrix()[0]
    result = c_probe.run_slice(root, spec)
    assert result["answers"] == len(calls) == 128
    assert result["groups"] == 16
    assert read(root / "PROBES/b0/PROGRESS_s0.json")["answers"] == 128
    again = c_probe.run_slice(root, spec)
    assert again == result and len(calls) == 128


def test_eight_category_chart_title_and_paired_image_identity():
    from sr_f3_ce.rendering import render_chart

    tasks = [t for t in c_probe_data.generate_tasks() if t["cell"] == "T8-SEPARATED"]
    first, metrics = render_chart(tasks[0])
    second, _ = render_chart(tasks[1])
    assert first == second
    assert "Eight-category quantities" in {label["text"] for label in metrics["labels"]}
    assert metrics["category_count"] == 8
