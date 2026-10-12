"""Prior evidence audits reject source mutation and keep syntax classification gold-blind."""

import importlib.util
import json
from pathlib import Path

import pytest

from mm_core.training import state_hash
from sr_f1.contract import gold_output
from sr_f12.protocol import BASELINE, object_hash

SPEC = importlib.util.spec_from_file_location(
    "audit_prior", Path(__file__).resolve().parents[2] / "scripts/sr_f2/audit_prior.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_sources_detect_mutation_and_retarget(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.write_text("original")
    second.write_text("other")
    link = tmp_path / "link"
    link.symlink_to(first)
    sources = audit.Sources()
    sources.add(link, role="test")
    link.unlink()
    link.symlink_to(second)
    with pytest.raises(ValueError, match="retargeted"):
        sources.stable()
    fresh = audit.Sources()
    fresh.add(first, role="test")
    first.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        fresh.stable()


def test_paths_reject_escape_and_existing_output(tmp_path):
    root = tmp_path / "old"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("a")
    (root / "escape").symlink_to(outside)
    for relative in ("../outside", str(outside), "escape"):
        with pytest.raises(PermissionError):
            audit.contained(root, relative)
    with pytest.raises(PermissionError):
        audit.new_output(root / "new", [root])
    target = tmp_path / "output"
    audit.new_output(target, [root])
    with pytest.raises(FileExistsError):
        audit.new_output(target, [root])


def test_checkpoints_follow_registered_directory_and_verify_all_bytes(tmp_path, monkeypatch):
    run = dict(model_id="MODEL", arm="A", seed=71001)
    monkeypatch.setattr(audit, "scientific_matrix", lambda: [run])
    directory = tmp_path / "prior/runs/MODEL"
    hdd = tmp_path / "hdd/MODEL"
    hdd.mkdir(parents=True)
    directory.mkdir(parents=True)
    (directory / "checkpoints").symlink_to(hdd, target_is_directory=True)
    manifest = dict(**run, technical_only=False)
    save(directory / "RUN_MANIFEST.json", manifest)
    for step in range(0, 97, 8):
        name = f"state-{step}.bin"
        (hdd / name).write_bytes(f"checkpoint-{step}".encode())
        commit = dict(
            step=step,
            path=name,
            sha256=audit.sha_file(hdd / name),
            state_hash="saved-full-state",
            field_hashes=dict(run_identity=state_hash(manifest)),
        )
        save(hdd / f"commit-{step:02d}.json", commit)
    save(hdd / "LATEST.json", commit)
    save(directory / "COMPLETE.json", dict(checkpoint=commit))
    authorization = dict(
        paths=[
            dict(
                model_id="MODEL",
                checkpoint_sha256=commit["sha256"],
                complete_sha256=audit.sha_file(directory / "COMPLETE.json"),
            )
        ]
    )
    rows = []
    result = audit.checkpoint_inventory(
        tmp_path / "prior", audit.Sources(), authorization, rows.append
    )
    assert result["verified"] == 13 and result["failures"] == []
    assert all(r["resolved_directory"] == str(hdd) for r in rows)
    (hdd / "state-8.bin").write_bytes(b"tampered")
    result = audit.checkpoint_inventory(
        tmp_path / "prior", audit.Sources(), authorization, lambda _: None
    )
    assert result["verified"] == 12
    assert result["failures"][0]["step"] == 8
    assert "SHA-256 differs" in result["failures"][0]["error"]


def endpoint_fixture(tmp_path):
    row = dict(
        zip(
            audit.FIELDS,
            [
                "slot",
                "MODEL",
                96,
                "TEST_COMPOSITION",
                "evidence_answer",
                "original",
                "q",
                "root",
                "TOPK",
                "line",
                "v0",
                0,
            ],
            strict=True,
        )
    )
    row.update(raw_text='{"answer":"12.5"}', tokens=[1])
    directory = tmp_path / "sealed/evaluation/MODEL"
    panel = directory / "composition"
    raw = dict(records=[row], records_sha256=object_hash([row]))
    save(panel / "groups/g.json", raw)
    plan = dict(assigned_groups=["g"], expected_records=1)
    save(panel / "PLAN.json", plan)
    finish = dict(
        plan_sha256=object_hash(plan),
        generated_records=1,
        groups=[dict(group_id="g", records=1, raw_sha256=audit.sha_file(panel / "groups/g.json"))],
    )
    save(panel / "COMPLETE.json", finish)
    authorization = dict(paths=[])
    done = dict(
        status="COMPLETE",
        main_completion_sha256=object_hash(authorization),
        panels=[
            dict(
                panel="composition",
                records=1,
                completion_sha256=audit.sha_file(panel / "COMPLETE.json"),
            )
        ],
    )
    save(directory / "COMPLETE.json", done)
    official = [{k: row[k] for k in audit.FIELDS} | dict(score=dict(L_answer=0))]
    release = dict(
        status="RELEASED_AFTER_ALL_MAIN_COMPLETE",
        model_id="MODEL",
        count=1,
        records=official,
        records_sha256=object_hash(official),
        evaluation_sha256=object_hash(done),
        authorization_sha256=object_hash(authorization),
    )
    save(tmp_path / "released/evaluation/MODEL.json", release)
    complete = dict(
        releases=[dict(model_id="MODEL", records=1, release_sha256=object_hash(release))]
    )
    return row, complete, authorization


def test_endpoint_hash_chain_and_blind_boundary(tmp_path, monkeypatch):
    row, complete, authorization = endpoint_fixture(tmp_path)
    monkeypatch.setattr(audit, "EXPECTED_ENDPOINT_RECORDS", 1)
    yielded = list(audit.endpoint_rows(tmp_path, audit.Sources(), complete, authorization))
    assert yielded[0][0] == row
    arguments, output = [], []

    def blind(raw_text, protocol):
        arguments.append((raw_text, protocol))
        return dict(answer_syntax="decimal_string")

    sources = audit.Sources()
    result = audit.syntax_census(tmp_path, sources, complete, authorization, output.append, blind)
    assert result["gold_read"] is False
    assert arguments == [(row["raw_text"], row["protocol"])]
    assert not any("gold" in e["roles"] for e in sources.files.values())
    raw_path = tmp_path / "sealed/evaluation/MODEL/composition/groups/g.json"
    raw_path.write_text("{}")
    with pytest.raises(ValueError, match="SHA-256 differs"):
        list(audit.endpoint_rows(tmp_path, audit.Sources(), complete, authorization))


def task():
    return dict(
        world=dict(
            categories=["a", "b", "c", "d"],
            series=dict(Alpha=[90, 80, 20, 10], Beta=[10, 20, 30, 40]),
        ),
        query=dict(family="TOPK", direction="largest", k=2, aggregate="sum"),
    )


def test_six_certificate_key_patterns_are_not_all_missing_two():
    t = task()
    obj = gold_output(t["world"], t["query"])
    obj["evidence"] = [
        e for e in obj["evidence"] if e["series"] == "Alpha" or e["category"] in ("a", "b")
    ]
    result = audit.evidence_pattern(json.dumps(obj), t)
    assert result["omission_pattern"] == {"Beta:unselected": 2}
    obj = gold_output(t["world"], t["query"])
    obj["evidence"] = [
        e for e in obj["evidence"] if e["series"] == "Beta" or e["category"] in ("a", "b")
    ]
    assert audit.evidence_pattern(json.dumps(obj), t)["omission_pattern"] == {"Alpha:unselected": 2}
    assert audit.evidence_pattern("garbage", t)["status"] == "UNEXTRACTABLE_KEYS"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "null"),
        ("12", "integer"),
        (" 12.5\n", "decimal"),
        ("7/3", "fraction_string"),
        ("yes", "nonnumeric_text"),
    ],
)
def test_gold_type(value, expected):
    assert audit.gold_type(value) == expected


def test_gate_original_numerical_anomaly_preserves_original_components():
    import copy

    import numpy as np

    from sr_f1.contract import reference_module, score

    t = task()
    t["query"] = dict(family="INTERVAL", start=1, end=3, include_start=True, aggregate="max")
    correct = gold_output(t["world"], t["query"])
    wrong = copy.deepcopy(correct)
    wrong["evidence"][0]["value"] += 1
    records = []
    for obj in [correct] * 5 + [wrong] * 3:
        text = json.dumps(obj)
        records.append(dict(raw_text=text, tokens=[1], score=score(text, t["world"], t["query"])))
    old = reference_module("semantic_contract")
    scores = [r["score"] for r in records]
    advantage, branch = old.gate_one(
        *[np.array([s[k] for s in scores]) for k in ("J", "E", "p_read")]
    )
    group = dict(identity=dict(slot=0), records=records)
    metrics = dict(
        reward_advantage_audit=dict(final_advantage=[advantage.tolist()], branches=[branch])
    )
    result = audit.gate_evidence(group, metrics, t, location=dict(path="fixture"))
    assert result["exact_reproduction"] is True
    assert result["failed_h_exactly_equal"] is True
    assert result["saved_advantage_sum"] > 0.9
    assert all(
        r["p_read_numerator"] == 5 and r["p_read_denominator"] == 6
        for r in result["original_records"][5:]
    )
    metrics["reward_advantage_audit"]["final_advantage"][0][0] += 0.01
    with pytest.raises(ValueError, match="differs from saved"):
        audit.gate_evidence(group, metrics, t, location={})


def test_rescore_preserves_v1_and_does_not_invent_122_cohort(tmp_path, monkeypatch):
    from sr_f1.chartqa import score_chartqa
    from sr_f2.scoring import classify_answer_syntax, score_record_v2
    from sr_f12.evaluation import score_record

    monkeypatch.setattr(audit, "scientific_matrix", lambda: [dict(model_id="SRF1_2_A_s71001")])
    monkeypatch.setattr(audit, "GATE_GROUPS", ())
    t = task()
    t.update(qid="q", answer=30)
    tasks = tmp_path / "manifests/TASKS_GOLD_AUDIT_ONLY.jsonl"
    save(tasks, t)
    chart = tmp_path / "external/chartqa/TEST_MANIFEST.jsonl"
    chart.parent.mkdir(parents=True)
    chart.write_text(
        "\n".join(
            json.dumps(dict(qid=f"chart-{i}", answer="yes", image_id=f"im-{i}"))
            for i in range(2500)
        )
    )
    save(
        chart.parent / "IDENTITY.json",
        dict(status="AVAILABLE_VERIFIED", manifest_sha256=audit.sha_file(chart)),
    )
    base = {k: None for k in audit.FIELDS}
    base.update(
        model_id="SRF1_2_A_s71001",
        pool="TEST_ID",
        protocol="evidence_answer",
        family="TOPK",
        qid="q",
        slot_id="s",
        raw_text="",
        tokens=[1],
    )
    obj = gold_output(t["world"], t["query"])
    obj["evidence"] = [
        e for e in obj["evidence"] if e["series"] == "Alpha" or e["category"] in ("a", "b")
    ]
    base["raw_text"] = json.dumps(obj)
    rows = [(base, score_record(base, t), {})]
    for model, answer in ((BASELINE, "yes\n"), ("SRF1_2_A_s71001", "no")):
        row = dict(
            base,
            model_id=model,
            pool="CHARTQA",
            family="CHARTQA",
            protocol="plain_answer",
            qid="chart-0",
            raw_text=answer,
            slot_id=model,
        )
        rows.append((row, score_chartqa("yes", answer), {}))
    monkeypatch.setattr(audit, "endpoint_rows", lambda *_: iter(rows))
    output = []
    result = audit.rescore(
        tmp_path, audit.Sources(), {}, {}, output.append, score_record_v2, classify_answer_syntax, 3
    )
    endpoint = output[0]
    assert endpoint["score_v1"]["E"] == endpoint["score_v2"]["E"] == 0
    assert endpoint["score_v2"]["E6"] == endpoint["score_v2"]["J6"] == 1
    assert result["v1_score_mismatches"] == 0
    assert result["requested_chartqa_cohort"]["status"] == "UNRESOLVED"
    assert result["chartqa_loss_candidates"]["count"] == 1
    assert output[-1]["scope"] == "DIAGNOSTIC_NOT_THE_REQUESTED_UNRESOLVED_122_COHORT"
    assert output[1]["score_v1"]["relaxed_accuracy"] == 0
    assert output[1]["score_v2"]["relaxed_accuracy"] == 1
