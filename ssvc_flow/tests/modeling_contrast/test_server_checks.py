import hashlib
import json
import os
import signal
import threading

import numpy as np
import pytest

from src.modeling_contrast import server_checks as checks


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _raw(root, *, fresh):
    root.mkdir()
    config = {
        "dataset": {"seed": 20260915},
        "toy_model": {"model_initialization_seed": 7001},
    }
    _write(root / "resolved_config.json", config)
    metadata = [
        {
            "prompt_id": "fixed",
            "actions": [[1, 2], "INVALID_A"],
            "event_categories": ["X", "I"],
            "group": 0,
            "weight": 1.0 / 72,
        }
    ]
    for split in ("train", "probe"):
        _write(root / f"{split}_metadata.json", metadata)
    _write(root / "parameter_layout.json", [{"name": "weight", "numel": 737}])
    arrays = {
        f"{split}_features": np.full((2, 2, 3), 0.2, dtype=np.float64)
        for split in ("train", "probe")
    }
    arrays.update(
        {
            f"{split}_categories": np.array([[0, 3], [1, 2]], dtype=np.int8)
            for split in ("train", "probe")
        }
    )
    np.savez(root / "dataset.npz", **arrays)
    for split in ("train", "control"):
        path = root / "generated_dataset" / f"{split}.jsonl"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"base_scene_id": split, "truth_world": [1, 2]}) + "\n")
    specifications = (
        [
            (seed, arm, "fresh_calibration" if seed < 600 else "fresh_locked_test")
            for seed in [*range(501, 507), *range(601, 611)]
            for arm in ("X_BASE", "X_VALID")
        ]
        if fresh
        else [(101, "X_BASE", "development_fit"), (101, "X_VALID", "development_fit")]
    )
    entries = []
    for seed, arm, split in specifications:
        tid = f"seed{seed}_{arm}"
        relative = f"observations/{tid}.npz"
        (root / relative).parent.mkdir(exist_ok=True)
        # A second row is intentionally different: only the initial parameters are identities.
        np.savez(root / relative, theta=np.stack((np.arange(737) * 0.001, np.ones(737))))
        entries.append(dict(id=tid, seed=seed, arm=arm, split=split, observations_file=relative))
    _write(
        root / "manifest.json",
        {
            "events": ["X", "S", "W", "I"],
            "parameter_count": 737,
            "probe_count": 72,
            "trajectories": entries,
        },
    )
    return root


def test_fresh_identity_compares_semantics_and_all_32_initializations(tmp_path):
    parent, fresh = _raw(tmp_path / "parent", fresh=False), _raw(tmp_path / "fresh", fresh=True)
    with np.load(fresh / "dataset.npz", allow_pickle=False) as data:
        values = dict(data)
    values["probe_features"][0, 0, 0] += 1e-15
    # Different archive layout/compression and one tolerated FP64 roundoff are valid.
    np.savez_compressed(fresh / "dataset.npz", **dict(reversed(list(values.items()))))
    before = hashlib.sha256((fresh / "dataset.npz").read_bytes()).hexdigest()
    receipt = checks.verify_fresh_identity(parent, fresh)
    assert receipt["passed"], receipt
    assert receipt["fresh_initializations_checked"] == 32
    assert receipt["parent_initializations_checked"] == 2
    feature = next(row for row in receipt["checks"] if row["name"] == "dataset.probe_features")
    assert 0 < feature["max_abs_error"] < feature["atol"]
    assert hashlib.sha256((fresh / "dataset.npz").read_bytes()).hexdigest() == before


@pytest.mark.parametrize(
    "change", ["last_initialization", "action_order", "categories", "seed", "missing_trajectory"]
)
def test_identity_rejects_scientific_changes(tmp_path, change):
    parent, fresh = _raw(tmp_path / "parent", fresh=False), _raw(tmp_path / "fresh", fresh=True)
    if change == "last_initialization":
        p = fresh / "observations/seed610_X_VALID.npz"
        with np.load(p, allow_pickle=False) as data:
            theta = data["theta"]
        theta[0, -1] += 1e-6
        np.savez(p, theta=theta)
    elif change == "action_order":
        p = fresh / "probe_metadata.json"
        meta = json.loads(p.read_text())
        meta[0]["actions"].reverse()
        _write(p, meta)
    elif change == "categories":
        with np.load(fresh / "dataset.npz", allow_pickle=False) as data:
            values = dict(data)
        values["train_categories"][0, 0] = 1
        np.savez(fresh / "dataset.npz", **values)
    elif change == "seed":
        p = fresh / "resolved_config.json"
        config = json.loads(p.read_text())
        config["toy_model"]["model_initialization_seed"] = 7002
        _write(p, config)
    else:
        p = fresh / "manifest.json"
        manifest = json.loads(p.read_text())
        manifest["trajectories"].pop()
        _write(p, manifest)
    receipt = checks.verify_fresh_identity(parent, fresh)
    assert not receipt["passed"]
    assert receipt["reasons"]


def test_identity_refuses_manifest_escape_and_object_arrays(tmp_path):
    parent, fresh = _raw(tmp_path / "parent", fresh=False), _raw(tmp_path / "fresh", fresh=True)
    p = fresh / "manifest.json"
    manifest = json.loads(p.read_text())
    manifest["trajectories"][-1]["observations_file"] = "../outside.npz"
    _write(p, manifest)
    assert not checks.verify_fresh_identity(parent, fresh)["passed"]
    manifest["trajectories"][-1]["observations_file"] = "observations/seed610_X_VALID.npz"
    _write(p, manifest)
    np.savez(fresh / manifest["trajectories"][-1]["observations_file"], theta=np.array([object()]))
    assert not checks.verify_fresh_identity(parent, fresh)["passed"]


def test_runtime_preflight_uses_fixed_seeds_and_cleans_temporary_files(tmp_path, monkeypatch):
    parent = _raw(tmp_path / "parent", fresh=False)
    scratch = tmp_path / "scratch"
    calls = []

    def build(out, dataset_seed, initialization_seed):
        calls.append((out, dataset_seed, initialization_seed))
        out.mkdir()
        (out / "proof.txt").write_text("temporary construction only")
        with np.load(parent / "dataset.npz", allow_pickle=False) as data:
            arrays = dict(data)
        return {
            "arrays": arrays,
            "metadata": {
                split: json.loads((parent / f"{split}_metadata.json").read_text())
                for split in ("train", "probe")
            },
            "layout": json.loads((parent / "parameter_layout.json").read_text()),
            "scenes": {
                split: [{"base_scene_id": split, "truth_world": [1, 2]}]
                for split in ("train", "control")
            },
            "theta": np.arange(737) * 0.001,
        }

    monkeypatch.setattr(checks, "_build_runtime_identity", build)
    receipt = checks.verify_parent_runtime_identity(parent, temporary_root=scratch)
    assert receipt["passed"], receipt
    assert len(calls) == 1 and calls[0][1:] == (20260915, 7001)
    assert calls[0][0].is_relative_to(scratch)
    assert not calls[0][0].exists()
    assert not list(scratch.iterdir())


def _config():
    return {
        "resources": {
            "max_ram_gib": 8,
            "max_added_output_gib": 6,
            "max_temporary_gib": 1,
            "max_total_walltime_hours": 4,
        }
    }


def _snapshot():
    return {"wall_seconds": 1, "peak_ram_gib": 0.2, "added_output_bytes": 20, "temporary_bytes": 2}


def test_resource_evaluation_and_rss_units_are_explicit():
    values = _snapshot()
    values["added_output_bytes"] = 6 * 2**30
    assert checks.evaluate_resource_snapshot(values, _config())["passed"]
    values["added_output_bytes"] += 1
    assert not checks.evaluate_resource_snapshot(values, _config())["passed"]
    values["added_output_bytes"] = float("nan")
    assert not checks.evaluate_resource_snapshot(values, _config())["passed"]
    assert checks.peak_rss_gib(2**30, "darwin") == 1
    assert checks.peak_rss_gib(2**20, "linux") == 1


def test_watchdog_counts_real_files_without_following_external_links(tmp_path, monkeypatch):
    run, control, temporary, repository = [
        tmp_path / name for name in ("run", "control", "tmp", "repo")
    ]
    for directory in (run, temporary, repository / "docs/modeling_contrast/results"):
        directory.mkdir(parents=True)
    (run / "ten").write_bytes(b"1" * 10)
    (temporary / "seven").write_bytes(b"1" * 7)
    results = repository / "docs/modeling_contrast/results"
    (results / "seventeen").write_bytes(b"1" * 17)
    os.link(run / "ten", results / "hardlink")
    (run / "external").symlink_to(temporary, target_is_directory=True)
    monkeypatch.setattr(checks, "ROOT", repository)
    with checks.RuntimeResourceWatchdog(
        _config(), run, control, temporary, prior_wall_seconds=11
    ) as watcher:
        receipt = watcher.check_now()
        assert receipt["snapshot"]["added_output_bytes"] == 27
        assert receipt["snapshot"]["temporary_bytes"] == 7
        assert receipt["snapshot"]["wall_seconds"] >= 11
    assert len((control / "resource_watchdog.jsonl").read_text().splitlines()) >= 3
    assert not watcher._thread.is_alive()


def test_watchdog_flushes_violation_before_signaling_only_current_process(tmp_path, monkeypatch):
    control = tmp_path / "control"
    stopped = []

    def terminate(pid, sig):
        violation = json.loads((control / "resource_violation.json").read_text())
        assert violation["status"] == "RESOURCE_REVIEW_REQUIRED"
        assert (control / "resource_watchdog.jsonl").read_text().endswith("\n")
        stopped.append((pid, sig))

    watcher = checks.RuntimeResourceWatchdog(
        _config(), tmp_path / "run", control, tmp_path / "tmp", terminate=terminate
    )
    bad = _snapshot() | {"wall_seconds": 14401}
    monkeypatch.setattr(watcher, "_snapshot", lambda: bad)
    with pytest.raises(checks.ResourceLimitExceeded), watcher:
        pytest.fail("startup violation must refuse work")
    assert stopped == [(os.getpid(), signal.SIGTERM)]


def test_watchdog_daemon_detects_limit_during_work(tmp_path, monkeypatch):
    stopped = threading.Event()
    watcher = checks.RuntimeResourceWatchdog(
        _config(),
        tmp_path / "run",
        tmp_path / "control",
        tmp_path / "tmp",
        interval_seconds=0.01,
        terminate=lambda pid, sig: stopped.set(),
    )
    snapshots = iter([_snapshot(), _snapshot() | {"temporary_bytes": 2**30 + 1}])
    monkeypatch.setattr(watcher, "_snapshot", lambda: next(snapshots))
    with pytest.raises(checks.ResourceLimitExceeded), watcher:
        assert stopped.wait(1)
    assert watcher.violation_receipt["snapshot"]["temporary_bytes"] == 2**30 + 1


def test_watchdog_refuses_control_log_inside_sealed_stage(tmp_path):
    run = tmp_path / "run"
    _write(run / "N4" / "RUN_MANIFEST.json", {"status": "COMPLETE"})
    with pytest.raises(ValueError, match="stage"):
        checks.RuntimeResourceWatchdog(_config(), run, run / "N4/control", tmp_path / "tmp")


def test_watchdog_fails_closed_if_measurement_or_receipt_io_fails(tmp_path, monkeypatch, capsys):
    stopped = []
    watcher = checks.RuntimeResourceWatchdog(
        _config(),
        tmp_path / "run",
        tmp_path / "control",
        tmp_path / "tmp",
        terminate=lambda pid, sig: stopped.append((pid, sig)),
    )

    def unreadable():
        raise PermissionError("unreadable experiment output")

    def full_disk(receipt, *, fatal):
        raise OSError("no space left for control receipt")

    monkeypatch.setattr(watcher, "_snapshot", unreadable)
    monkeypatch.setattr(watcher, "_persist", full_disk)
    with pytest.raises(checks.ResourceLimitExceeded), watcher:
        pytest.fail("unmeasured resource use must refuse work")
    assert stopped == [(os.getpid(), signal.SIGTERM)]
    receipt = json.loads(capsys.readouterr().err)
    assert "unreadable" in receipt["reasons"][0]
    assert "persistence failed" in receipt["reasons"][1]
