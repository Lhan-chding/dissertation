from collections import Counter

import pytest

from sr_f4 import c_local
from sr_f4.common import read, rows, write, write_rows


def test_task_matrix_pipelines_each_endpoint_without_global_training_barrier():
    matrix = c_local.task_matrix()
    counts = Counter(s["kind"] for s in matrix)
    assert counts == {"c_local_train": 12, "c_local_eval": 60, "c_local_report": 1}
    ids = {s["task_id"] for s in matrix}
    assert len(ids) == len(matrix)
    for position, spec in enumerate(matrix):
        if spec["kind"] == "c_local_train":
            ready_evals = matrix[position + 1 : position + 5]
            assert all(e["dependencies"] == [spec["task_id"]] for e in ready_evals)
        if spec["kind"] == "c_local_eval" and spec["arm"] == "PARENT":
            assert spec["dependencies"] == []
    assert set(matrix[-1]["dependencies"]) == ids - {"CLC-REPORT"}
    scheduled = c_local.pool_task_matrix()
    assert sum(s.get("resume_supported") is True for s in scheduled) == 72
    assert [{k: v for k, v in s.items() if k != "resume_supported"} for s in scheduled] == matrix


def test_frozen_protocol_changes_fail_before_model_loading(tmp_path):
    config = dict(
        parents={str(b): f"parent{b}" for b in range(3)}, model_path="base", code_revision="abc"
    )
    frozen = dict(
        c_local.data.plan(),
        **c_local.protocol(),
        **{"parent_references": config["parents"], "model_path": "base", "code_revision": "abc"},
    )
    write(tmp_path / "FROZEN.json", frozen)
    write(tmp_path / "CONFIG.json", config)
    write(tmp_path / "TASKS.json", c_local.task_matrix())
    frozen["reward"]["lambda_selection"] = 0.5
    write(tmp_path / "FROZEN.json", frozen)
    with pytest.raises(ValueError, match="freeze"):
        c_local.require_frozen(tmp_path)


def test_prepare_freezes_full_budget_with_original_retention_routes(tmp_path, monkeypatch):
    from sr_f3_ce import rendering

    monkeypatch.setattr(c_local, "prelaunch_gate", lambda: {"status": "PASS", "fixture": True})

    monkeypatch.setattr(rendering, "render_chart", lambda task: (b"image-placeholder", {}))
    source, root, old = tmp_path / "source", tmp_path / "pilot", tmp_path / "old"
    config = dict(
        model_path="base",
        parents={str(b): f"parent{b}" for b in range(3)},
        python="python",
        qos="soujanya-poria-startfund-2026-03",
        account="rose",
        partition="cluster02",
        old_root=str(old),
    )
    write(source / "CONFIG.json", config)
    retained = []
    for family in ("CROSS", "INTERVAL", "THRESHOLD", "TOPK"):
        for variant in range(8):
            for i in range(2):
                for chart in ("grouped_bar", "line"):
                    rid = f"{family}-{variant}-{i}"
                    retained.append(
                        dict(
                            qid=f"{rid}-{chart}",
                            root_id=rid,
                            world_id=rid,
                            pool="TEST_ID",
                            split="OLD_REGRESSION",
                            family=family,
                            variant=variant,
                            chart=chart,
                            world={"x": i},
                            query={},
                        )
                    )
    external = [
        dict(qid=f"chartqa-{i:04d}", pool="CHARTQA", answer="3", question="q") for i in range(250)
    ]
    write_rows(source / "DATA/TASKS.jsonl", retained)
    write_rows(
        source / "DATA/IMAGE_MANIFEST.jsonl",
        [
            dict(qid=t["qid"], image_file=f"images/{t['qid']}.png", image_sha256="original")
            for t in retained
        ],
    )
    write_rows(old / "external/chartqa/TASKS_AUDIT_ONLY.jsonl", external)
    write_rows(
        source / "DATA/CHARTQA_IMAGES.jsonl",
        [
            dict(
                qid=t["qid"],
                prompt="original prompt",
                image_file=f"external/{t['qid']}.png",
                image_sha256="original",
            )
            for t in external
        ],
    )
    for path in ("DATA/IMAGE_MANIFEST.jsonl", "DATA/CHARTQA_IMAGES.jsonl"):
        for image in rows(source / path):
            file = source / "DATA" / image["image_file"]
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(b"registered-image-pixels")
    c_local.prepare(root, source, "revision")
    c_local.require_frozen(root)
    assert sum(t["spec"].get("resume_supported") is True
               for t in read(root / "pool/STATE.json")["tasks"].values()) == 72
    assert read(root / "READY.json")["images_created"] == 216
    assert (
        sum(
            sum(g["G"] for g in rows(root / "DATA" / f"EVAL_GROUPS_b{b}.jsonl")) * 5
            for b in range(3)
        )
        == 31970
    )
    images = {r["qid"]: r for r in rows(root / "DATA/IMAGE_MANIFEST.jsonl")}
    assert images["chartqa-0000"]["image_file"] == "retained_images/external/chartqa-0000.png"
    for image in images.values():
        path = root / "DATA" / image["image_file"]
        assert path.resolve().is_relative_to((root / "DATA").resolve())
        assert path.is_file()
    assert images["chartqa-0000"]["image_sha256"] == "original"
    for b in range(3):
        plan = read(root / "DATA" / f"TRAIN_PLAN_b{b}.json")
        assert sum(g["G"] for groups in plan.values() for g in groups) == 1536
    mutated = list(rows(root / "DATA/TASKS.jsonl"))
    mutated[0]["prompt"] = "same qid but altered prompt"
    write_rows(root / "DATA/TASKS.jsonl", mutated)
    with pytest.raises(ValueError, match="manifest differs"):
        c_local.require_frozen(root)
    with pytest.raises(FileExistsError):
        c_local.prepare(root, source, "revision")


@pytest.mark.parametrize("failure", ["status", "measurement_version", "protocol", "review"])
def test_startup_requires_matching_completed_gate(monkeypatch, failure):
    gate = dict(schema="C_LOCAL_PRELAUNCH_V1", status="PASS", data_version=c_local.data.VERSION,
                measurement_version=c_local.data.MEASUREMENT_VERSION,
                protocol=c_local.protocol(), data_plan=c_local.data.plan(),
                offline_replay={"counts": {"answers": 6144}},
                checks=dict(cpu_gradient="PASS", integration="PASS", review="PASS"))
    monkeypatch.setattr(c_local, "read", lambda path: gate)
    assert c_local.prelaunch_gate() == gate
    if failure == "review":
        gate["checks"]["review"] = "PENDING"
    else:
        gate[failure] = "CHANGED_OR_INCOMPLETE"
    with pytest.raises(PermissionError, match="CPU/replay gate"):
        c_local.prelaunch_gate()
