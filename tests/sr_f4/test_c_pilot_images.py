import copy
import os

import pytest

from sr_f4 import c_pilot, pool
from sr_f4 import c_pilot_images as routing
from sr_f4.common import read, rows, write, write_rows


@pytest.fixture
def legacy(tmp_path, monkeypatch):
    source, root = tmp_path / "source", tmp_path / "pilot"
    image = source / "DATA/images/old.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"registered-pixels")
    old = dict(qid="old", image_file=str(image), image_sha256="unchanged", prompt="same")
    inputs = dict(tasks=[dict(qid="old", pool="TEST_ID")], images=[old])
    write(root / "CONFIG.json", dict(source_root=str(source), code_revision="baseline"))
    write(root / "FROZEN.json", dict(code_revision="baseline", budget=48866))
    write(root / "FROZEN_INPUTS.json", inputs)
    write_rows(root / "DATA/IMAGE_MANIFEST.jsonl", inputs["images"])
    write(root / "EVAL/cached.json", {"rows": ["completed-answer"]})
    write_rows(root / "technical_events.jsonl", [])
    specs = [dict(task_id=f"eval{i}", kind="c_pilot_eval", dependencies=[]) for i in range(4)]
    specs.append(dict(task_id="train", kind="c_pilot_train", dependencies=[]))
    pool.init_pool(root, specs)

    def validate(root):
        assert (
            list(rows(root / "DATA/IMAGE_MANIFEST.jsonl"))
            == read(root / "FROZEN_INPUTS.json")["images"]
        )
        assert read(root / "FROZEN_INPUTS.json")["tasks"] == inputs["tasks"]

    monkeypatch.setattr(c_pilot, "require_frozen", validate)
    return root, source, inputs


def failed(root, key, error="Image path escapes run root", attempt=1):
    event = dict(
        task_id=key, at="2026-01-01T00:00:01+00:00", error_type="PermissionError", error=error
    )
    with pool._locked(root) as state:
        row = state["tasks"][key]
        row.update(
            status="FAILED",
            reason="child_exit=1",
            attempts=[
                dict(
                    owner=dict(started_at="2026-01-01T00:00:00+00:00"),
                    status="FAILED",
                    finished_at="2026-01-01T00:00:02+00:00",
                    returncode=1,
                )
                for _ in range(attempt)
            ],
        )
        pool._save(root, state)
    events = list(rows(root / "technical_events.jsonl"))
    write_rows(root / "technical_events.jsonl", [*events, event])


def test_repair_preserves_pixels_frozen_identity_and_cached_answers(legacy):
    root, source, inputs = legacy
    originals = {
        p: (root / p).read_bytes() for p in ("CONFIG.json", "FROZEN.json", "EVAL/cached.json")
    }
    receipt = routing.repair_routes(root, "fix-revision")
    updated = read(root / "FROZEN_INPUTS.json")["images"][0]
    assert {k: v for k, v in updated.items() if k != "image_file"} == {
        k: v for k, v in inputs["images"][0].items() if k != "image_file"
    }
    assert updated["image_file"] == "retained_images/images/old.png"
    assert (root / "DATA" / updated["image_file"]).read_bytes() == (
        source / "DATA/images/old.png"
    ).read_bytes()
    assert read(root / routing.RECOVERY / "FROZEN_INPUTS.before.json") == inputs
    assert routing.repair_routes(root, "fix-revision") == receipt
    assert all((root / p).read_bytes() == content for p, content in originals.items())


def test_resume_partial_pair_write_from_recorded_before_and_after(legacy, monkeypatch):
    root, _, _ = legacy
    original_write = routing.write

    def interrupt(path, value):
        if path == root / "FROZEN_INPUTS.json":
            raise OSError("interrupted")
        original_write(path, value)

    monkeypatch.setattr(routing, "write", interrupt)
    with pytest.raises(OSError, match="interrupted"):
        routing.repair_routes(root, "fix")
    assert read(root / routing.RECOVERY / "RECEIPT.json")["status"] == "PREPARED"
    monkeypatch.setattr(routing, "write", original_write)
    assert routing.repair_routes(root, "fix")["status"] == "APPLIED"
    c_pilot.require_frozen(root)


def test_repair_waits_for_inflight_child_to_read_manifests(legacy):
    root, _, inputs = legacy
    with pool._locked(root) as state:
        state["tasks"]["train"].update(
            status="RUNNING", owner=dict(started_at="2026-01-01T00:00:00+00:00")
        )
        pool._save(root, state)
    with pytest.raises(RuntimeError, match="still initializing"):
        routing.repair_routes(root, "fix")
    assert read(root / "FROZEN_INPUTS.json") == inputs
    identity = root / "runtime_identity/train.json"
    write(identity, {})
    os.utime(identity, (1800000000, 1800000000))
    assert routing.repair_routes(root, "fix")["status"] == "APPLIED"


def test_retry_is_exact_attempt_only_preserves_history_and_ignores_other_errors(legacy):
    root, _, _ = legacy
    failed(root, "eval0")
    failed(root, "eval1", "different failure")
    failed(root, "eval2")
    failed(root, "train")
    write(root / "tasks/eval2/COMPLETE.json", dict(status="COMPLETE"))
    before = copy.deepcopy(read(root / "pool/STATE.json")["tasks"]["eval0"]["attempts"])
    routing.repair_routes(root, "fix")
    assert routing.retry_route_failures(root) == ["eval0"]
    state = read(root / "pool/STATE.json")
    assert state["tasks"]["eval0"]["attempts"] == before
    assert state["tasks"]["eval0"]["recoveries"][0]["failed_attempt"] == 1
    failed(root, "eval0", attempt=2)
    failed(root, "eval3")  # First started after the migration snapshot.
    assert routing.retry_route_failures(root) == []


def test_late_failure_from_running_pre_repair_attempt_can_resume(legacy):
    root, _, _ = legacy
    failed(root, "eval0")
    with pool._locked(root) as state:
        row = state["tasks"]["eval0"]
        row.update(status="RUNNING", owner=row["attempts"][0]["owner"])
        row["attempts"][0]["status"] = "RUNNING"
        pool._save(root, state)
    identity = root / "runtime_identity/eval0.json"
    write(identity, {})
    os.utime(identity, (1800000000, 1800000000))
    routing.repair_routes(root, "fix")
    failed(root, "eval0")
    assert routing.retry_route_failures(root) == ["eval0"]


def test_refuse_external_source_and_destination_symlink(legacy, tmp_path):
    root, source, inputs = legacy
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"foreign")
    with pytest.raises(PermissionError, match="source"):
        routing.local_image(root, source, dict(inputs["images"][0], image_file=str(outside)))
    (root / "DATA/retained_images").symlink_to(source / "DATA", target_is_directory=True)
    with pytest.raises(PermissionError, match="destination"):
        routing.local_image(root, source, inputs["images"][0])
