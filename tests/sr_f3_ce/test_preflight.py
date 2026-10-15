import pytest
import torch

from sr_f3_ce.preflight import (
    CHECKPOINT_FIELDS,
    bounded_file,
    copy_verified,
    file_hash,
    load_verified_checkpoint,
    object_hash,
    verify_file,
    verify_tensor_inventory,
)


def test_file_paths_reject_traversal_and_escaping_symlink(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"data")
    (root / "escape").symlink_to(outside)
    for name in ("../outside", str(outside), "escape"):
        with pytest.raises(PermissionError):
            bounded_file(root, name)


def test_hash_mismatch_before_pickle_loader(tmp_path):
    checkpoint = tmp_path / "state.pt"
    checkpoint.write_bytes(b"not a trusted checkpoint")
    called = []
    marker = {"path": "state.pt", "sha256": "0" * 64}
    with pytest.raises(PermissionError, match="bytes changed"):
        load_verified_checkpoint(tmp_path, marker, {}, loader=lambda path: called.append(path))
    assert called == []


def test_verified_copy_is_idempotent_and_never_overwrites(tmp_path):
    source, root = tmp_path / "source", tmp_path / "new"
    source.write_bytes(b"immutable")
    root.mkdir()
    before = file_hash(source)
    one = copy_verified(source, root, "parents/adapter.bin", before)
    assert copy_verified(source, root, "parents/adapter.bin", before) == one
    source.write_bytes(b"changed")
    with pytest.raises(PermissionError, match="Existing output differs"):
        copy_verified(source, root, "parents/adapter.bin")
    assert (root / "parents/adapter.bin").read_bytes() == b"immutable"
    with pytest.raises(PermissionError):
        verify_file(root, "parents/adapter.bin", "0" * 64)


def tensor_fixture():
    module = "model.language_model.layers.0.mlp.up_proj"
    params = {
        f"base_model.model.{module}.lora_A.default.weight": torch.arange(24).reshape(8, 3).float(),
        f"base_model.model.{module}.lora_B.default.weight": torch.arange(32).reshape(4, 8).float(),
    }
    saved = {
        name.replace(".default.weight", ".weight"): value.clone() for name, value in params.items()
    }
    config = {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0,
        "bias": "none",
        "peft_type": "LORA",
        "target_modules": ["up_proj"],
    }
    return params, saved, [module], config


def test_exact_adapter_key_mapping_and_tensor_content():
    params, saved, modules, config = tensor_fixture()
    result = verify_tensor_inventory(params, saved, modules, config)
    assert len(result) == 2
    assert sum(row["numel"] for row in result) == 56
    next(iter(saved.values()))[0, 0] += 1
    with pytest.raises(PermissionError, match="tensor differs"):
        verify_tensor_inventory(params, saved, modules, config)


def test_incomplete_parameter_or_adapter_targets_are_rejected():
    params, saved, modules, config = tensor_fixture()
    config["target_modules"] = ["q_proj", "v_proj"]
    with pytest.raises(PermissionError, match="target suffixes"):
        verify_tensor_inventory(params, saved, modules, config)
    config["target_modules"] = ["up_proj"]
    saved.pop(next(iter(saved)))
    with pytest.raises(PermissionError, match="one-to-one"):
        verify_tensor_inventory(params, saved, modules, config)


def test_copy_does_not_follow_output_symlink(tmp_path):
    root, elsewhere = tmp_path / "root", tmp_path / "elsewhere"
    root.mkdir()
    elsewhere.mkdir()
    (root / "parents").symlink_to(elsewhere)
    source = tmp_path / "source"
    source.write_bytes(b"parent")
    with pytest.raises(PermissionError, match="escapes"):
        copy_verified(source, root, "parents/adapter.bin")
    assert list(elsewhere.iterdir()) == []


def checkpoint_fixture(tmp_path):
    from mm_core.training import state_hash

    params, _, _, _ = tensor_fixture()
    reference = {name: value + 1 for name, value in params.items()}
    manifest = {"model_id": "SRF2_J6_s71001", "reference_hash": state_hash(reference)}
    state = dict.fromkeys(CHECKPOINT_FIELDS)
    state.update(
        plan_id="SR-F2",
        run_identity=manifest,
        committed_logical_step=96,
        cursor={"next_logical_step": 97, "next_slot": 0, "next_sample_index": 0},
        parameters=params,
        reference=reference,
        reference_hash=state_hash(reference),
        optimizer={"state": {0: {"step": torch.tensor(96.0)}}},
    )
    path = tmp_path / "checkpoint.pt"
    torch.save(state, path)
    marker = {
        "path": path.name,
        "sha256": file_hash(path),
        "step": 96,
        "state_hash": state_hash(state),
        "field_hashes": {k: state_hash(v) for k, v in state.items()},
    }
    return state, marker, manifest


def test_verified_checkpoint_recomputes_complete_state_and_all_fields(tmp_path):
    from mm_core.training import state_hash

    state, marker, manifest = checkpoint_fixture(tmp_path)
    loaded = load_verified_checkpoint(tmp_path, marker, manifest)
    assert state_hash(loaded) == state_hash(state)
    marker["field_hashes"]["parameters"] = "0" * 64
    with pytest.raises(PermissionError, match="state/field/reference"):
        load_verified_checkpoint(tmp_path, marker, manifest)


def test_trusted_checkpoint_manifest_still_must_match(tmp_path):
    _, marker, manifest = checkpoint_fixture(tmp_path)
    manifest = {**manifest, "model_id": "another-parent"}
    with pytest.raises(PermissionError, match="state/field/reference"):
        load_verified_checkpoint(tmp_path, marker, manifest)


def test_object_hash_ignores_json_whitespace_but_not_values():
    assert object_hash({"a": 1, "b": 2}) == object_hash({"b": 2, "a": 1})
    assert object_hash({"a": 1}) != object_hash({"a": 2})
