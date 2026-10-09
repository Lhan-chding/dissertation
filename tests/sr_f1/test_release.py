import copy
import json
from pathlib import Path

import pytest

from sr_f1.release import ReleaseBlocked, _write_once, terminal_matrix

PACKAGE = Path(__file__).resolve().parents[2] / "docs/sr_f1/package"


def registered():
    return json.loads((PACKAGE / "manifests/RUN_MATRIX.json").read_text())


def test_no_release_before_every_registered_arm_terminal_or_with_changed_design():
    expected = registered()
    rows = [dict(r, status="COMPLETE") for r in expected]
    rows[0]["status"] = "RUNNING"
    with pytest.raises(ReleaseBlocked, match="terminal"):
        terminal_matrix({"runs": rows}, expected)
    rows[0]["status"] = "TECHNICAL_FAILED"
    with pytest.raises(ReleaseBlocked, match="reason"):
        terminal_matrix(rows, expected)
    rows[0]["failure"] = "retained CUDA incident"
    assert terminal_matrix(rows, expected)[rows[0]["run_id"]]["status"] == "TECHNICAL_FAILED"
    bad = copy.deepcopy(rows)
    bad[0]["H"] = 128
    with pytest.raises(ReleaseBlocked, match="registration"):
        terminal_matrix(bad, expected)
    with pytest.raises(ReleaseBlocked, match="15"):
        terminal_matrix(rows[:-1], expected)


def test_release_is_idempotent_and_never_overwrites_changed_result(tmp_path):
    path = tmp_path / "artifact.json"
    _write_once(path, '{"actual":1}\n')
    _write_once(path, '{"actual":1}\n')
    with pytest.raises(ReleaseBlocked, match="changed"):
        _write_once(path, '{"actual":2}\n')
    assert path.read_text() == '{"actual":1}\n'


def test_raw_recount_requires_real_slots_and_frozen_prompt(tmp_path, monkeypatch):
    import hashlib

    import sr_f1.release as release_module
    from sr_f1.contract import gold_output
    from sr_f1.evaluation import encoded

    tasks = {
        r["qid"]: r
        for r in map(
            json.loads, (PACKAGE / "manifests/TASKS_GOLD_AUDIT_ONLY.jsonl").read_text().splitlines()
        )
    }
    task = next(iter(tasks.values()))
    slot = dict(
        slot_id="fixed",
        model_id="SRF1_COMMON_START",
        step=0,
        pool="MONITOR",
        protocol="evidence_answer",
        view=None,
        qid=task["qid"],
        root_id=task["root_id"],
        family=task["family"],
        chart=task["chart"],
        variant="v0",
        draw=0,
        monitor_block=0,
    )
    monkeypatch.setattr(release_module, "iter_evaluation_slots", lambda *_: iter([slot]))
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "MODEL_INPUTS.jsonl").write_text(
        encoded(dict(qid=task["qid"], text="frozen question", plain_text="plain", image_file=None))
        + "\n"
    )
    (manifests / "DIAGNOSTIC_INPUTS_AUDIT_ONLY.jsonl").write_text("")
    raw = tmp_path / "raw/evaluation"
    raw.mkdir(parents=True)
    with pytest.raises(ReleaseBlocked, match="Missing 1"):
        release_module.audit_evaluation(tmp_path, tasks, [], {})
    row = dict(
        slot,
        raw_text=encoded(gold_output(task["world"], task["query"])),
        status="GENERATED",
        tokens=[1],
        old_logprobs=[-1],
        truncated=False,
        model_identity="actual-model",
        adapter_identity="actual-adapter",
        image_file=None,
        image_hash=None,
        text="frozen question",
    )
    row["input_hash"] = hashlib.sha256(
        encoded(dict(text=row["text"], image_hash=None)).encode()
    ).hexdigest()
    (raw / "model.jsonl").write_text(encoded(row) + "\n")
    rows, _, audit = release_module.audit_evaluation(tmp_path, tasks, [], {})
    assert audit["synthetic_complete"] and rows[0]["independent_score"]["J"] == 1
    row["text"] = "Injected gold answer"
    row["input_hash"] = hashlib.sha256(
        encoded(dict(text=row["text"], image_hash=None)).encode()
    ).hexdigest()
    (raw / "model.jsonl").write_text(encoded(row) + "\n")
    with pytest.raises(ReleaseBlocked, match="gold-free prompt"):
        release_module.audit_evaluation(tmp_path, tasks, [], {})
