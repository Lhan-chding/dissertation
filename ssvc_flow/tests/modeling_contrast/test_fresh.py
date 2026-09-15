"""Gate refusal tests never invoke training or generate new trajectories."""

import json
from unittest.mock import Mock

import pytest

from src.modeling_contrast.collect_fresh import (
    build_fresh_parent_config,
    run_collect_fresh,
    scan_seed_collisions,
    validate_fresh_gate,
)
from src.modeling_contrast.io import RunWriter, canonical_hash, sha256_file
from src.modeling_contrast.protocol import load_config
from src.modeling_contrast.selection import freeze_selection


def frozen_fixture(tmp_path, mutate=None):
    config = load_config()
    source = tmp_path / "source.py"
    source.write_text("# fixture source, no model\n")
    data = tmp_path / "old_data.txt"
    data.write_text("old fixture\n")
    packet = tmp_path / "packet.txt"
    packet.write_text("paid fixture\n")
    candidate = {
        "configuration_id": "finite",
        "model": "C3_CONTRAST_GLS_FULL",
        "method": "O_LR_ORIGIN",
        "access_regime": "SAMPLE_AND_LOGP",
        "n": 64,
        "actual_rank": 1,
        "nonalias_eval_units": 12,
        "predicted_difference_norm": 0.1,
        "pX_nrmse": 0.4,
        "v_nrmse": 0.3,
        "stable_improvement_C0": True,
        "stable_improvement_C1": True,
    }
    cost = {
        "configuration_id": "finite",
        "access_regime": "SAMPLE_AND_LOGP",
        "includes_fit_score_label": True,
        "score_to_generation_ratio": 0.25,
        "within_domain": True,
        "cost_feasible": True,
        "comparison_baseline": "D_DIRECT",
    }
    lock = freeze_selection(
        [candidate],
        protocol=config,
        input_hashes={str(data): sha256_file(data)},
        source_hashes={str(source): sha256_file(source)},
        packet_hashes={str(packet): sha256_file(packet)},
        resource_forecast={
            "total_walltime_hours": 1,
            "peak_ram_gib": 1,
            "added_output_gib": 0.1,
            "temporary_gib": 0.1,
        },
        data_gate={"status": "PASS", "legacy_complete": True, "fresh_seed_collision": False},
        cost_frontier=[cost],
    )
    if mutate:
        mutate(lock)
        lock["lock_sha256"] = canonical_hash(
            {key: value for key, value in lock.items() if key != "lock_sha256"}
        )
    out = tmp_path / "runs/modeling_contrast_v2/N3"
    with RunWriter(out, lock["binding"], project_root=tmp_path) as writer:
        writer.write_json("MODEL_SELECTION_LOCK.json", lock)
    kwargs = dict(
        existing_run_roots=[tmp_path / "runs"],
        project_root=tmp_path,
        resource_forecast=dict(
            wall_seconds=3600, peak_ram_gib=1, added_output_bytes=100, temporary_bytes=100
        ),
    )
    return config, out / "MODEL_SELECTION_LOCK.json", lock, kwargs


def test_fresh_parent_adapter_preserves_optimizer_and_fixed_budget():
    adapted = build_fresh_parent_config(load_config())
    profile = adapted["profiles"]["contrast_fresh"]
    assert profile["seeds_by_role"] == {
        "fresh_calibration": list(range(501, 507)),
        "fresh_locked_test": list(range(601, 611)),
    }
    assert profile["steps"] == 64 and profile["B"] == 4 and profile["K"] == 8
    assert profile["arms"] == ["X_BASE", "X_VALID"]
    assert adapted["budget_derived"]["total_optimizer_steps"] == 6080
    assert adapted["dataset"]["seed"] == 20260915
    assert adapted["toy_model"]["model_initialization_seed"] == 7001


def test_gate_failure_does_not_call_collector_or_create_output(tmp_path):
    config = load_config()
    path = tmp_path / "runs/modeling_contrast_v2/N3"
    binding = {
        key: canonical_hash(key) for key in ("source", "config", "data", "selector", "packet")
    }
    with RunWriter(path, binding, project_root=tmp_path) as writer:
        writer.write_json("MODEL_SELECTION_LOCK.json", {"allow_fresh_cpu": False})
    collector = Mock(side_effect=AssertionError("new CPU training was entered"))
    output = tmp_path / "runs/modeling_contrast_v2/N4"
    with pytest.raises(ValueError, match="fresh gate"):
        run_collect_fresh(
            config,
            path / "MODEL_SELECTION_LOCK.json",
            output,
            binding=binding,
            existing_run_roots=[tmp_path / "runs"],
            resource_forecast={},
            collector=collector,
            project_root=tmp_path,
        )
    collector.assert_not_called()
    assert not output.exists()


def test_missing_lock_and_changed_outputs_are_rejected(tmp_path):
    config = load_config()
    with pytest.raises(ValueError, match="fresh gate"):
        validate_fresh_gate(
            config,
            tmp_path / "missing.json",
            existing_run_roots=[tmp_path],
            resource_forecast={},
            project_root=tmp_path,
        )


def test_existing_training_seeds_are_not_silently_renumbered(tmp_path):
    root = tmp_path / "runs/legacy"
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps({"trajectories": [{"seed": 601, "id": "seed601_X_BASE"}]})
    )
    collisions = scan_seed_collisions([tmp_path / "runs"], set(range(601, 611)))
    assert collisions[0]["seed"] == 601
    assert "manifest.json" in collisions[0]["evidence"]


def test_seed_inventory_rejects_missing_scan_roots(tmp_path):
    with pytest.raises(ValueError, match="inventory"):
        scan_seed_collisions([tmp_path / "missing"], {601})


def test_seed_inventory_skips_only_verified_appledouble_sidecars(tmp_path):
    # AppleDouble metadata is not a raw collection or a JSON inventory.
    sidecar = b"\x00\x05\x16\x07\x00\x02\x00\x00" + b"\xff" * 24
    for name in ("._manifest.json", "._seed601_X_BASE.npz"):
        (tmp_path / name).write_bytes(sidecar)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"trajectories": [{"seed": 601}]}))
    assert scan_seed_collisions([tmp_path], {601}) == [
        {"seed": 601, "evidence": str(manifest)}
    ]
    assert (tmp_path / "._manifest.json").read_bytes() == sidecar


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("manifest.json", b"\x00\x05\x16\x07\xff"),
        ("._manifest.json", b"\xffnot AppleDouble"),
        ("._manifest.json", b"\x00\x05\x16"),
        ("._manifest.json", b"\x00\x05\x16\x00\xff"),  # AppleSingle is not exempt.
    ],
)
def test_seed_inventory_refuses_unverified_or_real_corrupt_manifests(tmp_path, name, content):
    (tmp_path / name).write_bytes(content)
    with pytest.raises(ValueError, match="manifest unreadable"):
        scan_seed_collisions([tmp_path], {601})


def test_seed_inventory_checks_json_even_with_sidecar_style_name(tmp_path):
    manifest = tmp_path / "._manifest.json"
    manifest.write_text(json.dumps({"trajectories": [{"seed": 601}]}))
    assert scan_seed_collisions([tmp_path], {601}) == [
        {"seed": 601, "evidence": str(manifest)}
    ]


def test_valid_frozen_gate_can_be_checked_without_training(tmp_path):
    config, path, lock, kwargs = frozen_fixture(tmp_path)
    receipt = validate_fresh_gate(config, path, **kwargs)
    assert receipt["status"] == "PASS"
    assert receipt["binding"] == lock["binding"]


@pytest.mark.parametrize("kind", ["source", "data", "packet", "selector"])
def test_mutated_bound_files_refuse_before_collector(tmp_path, kind):
    config, path, lock, kwargs = frozen_fixture(tmp_path)
    if kind == "selector":
        path.write_text("{}")
    else:
        field = {"source": "source_hashes", "data": "input_hashes", "packet": "packet_hashes"}[kind]
        from pathlib import Path

        Path(next(iter(lock[field]))).write_text("changed")
    collector = Mock(side_effect=AssertionError("new training entered"))
    with pytest.raises(ValueError, match="fresh gate"):
        run_collect_fresh(
            config,
            path,
            tmp_path / "runs/modeling_contrast_v2/N4",
            binding=lock["binding"],
            collector=collector,
            **kwargs,
        )
    collector.assert_not_called()


@pytest.mark.parametrize("kind", ["science_gate", "cost_gate", "resource_gate", "data_gate"])
def test_one_failed_gate_refuses_new_training(tmp_path, kind):
    config, path, lock, kwargs = frozen_fixture(
        tmp_path, lambda lock: lock[kind].update(status="FAIL")
    )
    collector = Mock(side_effect=AssertionError("new training entered"))
    with pytest.raises(ValueError, match="fresh gate"):
        run_collect_fresh(
            config,
            path,
            tmp_path / "runs/modeling_contrast_v2/N4",
            binding=lock["binding"],
            collector=collector,
            **kwargs,
        )
    collector.assert_not_called()


def test_seed_scan_ignores_explicit_test_fixtures_but_checks_actual_runs(tmp_path):
    fixture = tmp_path / "fixtures/example"
    fixture.mkdir(parents=True)
    (fixture / "manifest.json").write_text("{broken fixture JSON")
    (fixture / "seed601_fake.npz").write_bytes(b"fixture")
    actual = tmp_path / "actual"
    actual.mkdir()
    (actual / "manifest.json").write_text(json.dumps({"trajectories": [{"seed": 602}]}))
    hits = scan_seed_collisions([tmp_path], {601, 602})
    assert len(hits) == 1 and hits[0]["seed"] == 602


def test_post_collection_gate_exempts_only_complete_bound_raw_files(tmp_path):
    config, path, lock, kwargs = frozen_fixture(tmp_path)
    collection = tmp_path / "runs/modeling_contrast_v2/N4"
    with RunWriter(collection, lock["binding"], project_root=tmp_path) as writer:
        writer.write_json("raw/manifest.json", {"trajectories": [{"seed": 601}]})
        writer.write_bytes("raw/observations/seed601_X_BASE.npz", b"hash-only raw fixture")
    # The original pre-collection gate must continue refusing existing seeds.
    with pytest.raises(ValueError, match="seed collision"):
        validate_fresh_gate(config, path, **kwargs)
    receipt = validate_fresh_gate(config, path, completed_collection_root=collection, **kwargs)
    assert receipt["status"] == "PASS"
    assert len(receipt["verified_collection"]["excluded_raw_paths"]) == 2
    other = tmp_path / "runs/other_collection"
    other.mkdir()
    (other / "seed601_X_BASE.npz").write_bytes(b"conflicting run")
    with pytest.raises(ValueError, match="seed collision"):
        validate_fresh_gate(config, path, completed_collection_root=collection, **kwargs)


@pytest.mark.parametrize("corruption", ["bytes", "unregistered", "incomplete", "binding"])
def test_post_collection_exemption_requires_complete_binding_and_bytes(tmp_path, corruption):
    config, path, lock, kwargs = frozen_fixture(tmp_path)
    collection = tmp_path / "runs/modeling_contrast_v2/N4"
    with RunWriter(collection, lock["binding"], project_root=tmp_path) as writer:
        writer.write_json("raw/manifest.json", {"trajectories": [{"seed": 601}]})
    if corruption == "bytes":
        (collection / "raw/manifest.json").write_text("{}")
    elif corruption == "unregistered":
        (collection / "raw/seed601_extra.npz").write_bytes(b"unregistered")
    else:
        manifest_path = collection / "RUN_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text())
        if corruption == "incomplete":
            manifest["status"] = "INTERRUPTED"
        else:
            manifest["binding"]["selector"] = "b" * 64
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        validate_fresh_gate(config, path, completed_collection_root=collection, **kwargs)


@pytest.mark.parametrize("server_amendment", [False, True])
def test_n3_server_stage_requires_exact_approved_server_amendment(tmp_path, server_amendment):
    from src.modeling_contrast.protocol import SERVER_CONFIG, config_sha256

    config, _path, lock, kwargs = frozen_fixture(tmp_path)
    if server_amendment:
        config = load_config(SERVER_CONFIG)
        lock["protocol_sha256"] = config_sha256(config)
        lock["binding"]["config"] = config_sha256(config)
        lock["lock_sha256"] = canonical_hash(
            {key: value for key, value in lock.items() if key != "lock_sha256"}
        )
    out = tmp_path / "runs/modeling_contrast_v2/N3_server"
    with RunWriter(out, lock["binding"], project_root=tmp_path) as writer:
        writer.write_json("MODEL_SELECTION_LOCK.json", lock)
    if server_amendment:
        result = validate_fresh_gate(config, out / "MODEL_SELECTION_LOCK.json", **kwargs)
        assert result["status"] == "PASS"
    else:
        with pytest.raises(ValueError, match="N3"):
            validate_fresh_gate(config, out / "MODEL_SELECTION_LOCK.json", **kwargs)
