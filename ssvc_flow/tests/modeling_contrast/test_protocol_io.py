"""Safety and reproducibility contracts, independent of model outcomes."""

import json

import numpy as np
import pytest

from src.modeling_contrast.io import (
    RunWriter,
    canonical_hash,
    checked_output_path,
    sha256_file,
    source_snapshot,
    write_json,
    write_npz,
)
from src.modeling_contrast.protocol import (
    DESIGN,
    cpu_environment,
    load_config,
    resource_gate,
    validate_config,
)


def test_config_byte_and_semantic_freeze(tmp_path):
    config = load_config(DESIGN)
    assert validate_config(config)["total_optimizer_updates"] == 6080
    changed = json.loads(json.dumps(config))
    changed["statistics"]["nrmse_pX_max"] = 0.8
    with pytest.raises(ValueError, match="locked protocol"):
        validate_config(changed)
    altered = tmp_path / "config.json"
    altered.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="byte"):
        load_config(altered)


def test_cpu_environment():
    environment = cpu_environment()
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert environment["OMP_NUM_THREADS"] == "1"
    assert environment["HF_HUB_OFFLINE"] == "1"


def test_resource_gate_checks_all_dimensions_and_missing():
    forecast = dict(wall_seconds=100, peak_ram_gib=1, added_output_bytes=100, temporary_bytes=100)
    assert resource_gate(forecast)["passed"]
    for field, high in [
        ("wall_seconds", 14401),
        ("peak_ram_gib", 8.01),
        ("added_output_bytes", 3 * 2**30 + 1),
        ("temporary_bytes", 2**30 + 1),
    ]:
        assert not resource_gate({**forecast, field: high})["passed"]
    assert not resource_gate({})["passed"]
    assert not resource_gate({**forecast, "wall_seconds": float("nan")})["passed"]


def test_output_root_and_symlink_escape(tmp_path):
    allowed = tmp_path / "runs/modeling_contrast_v2/N0"
    assert checked_output_path(allowed, project_root=tmp_path) == allowed
    with pytest.raises(ValueError, match="output"):
        checked_output_path(tmp_path / "runs/modeling_qualification/M2", project_root=tmp_path)
    root = tmp_path / "runs/modeling_contrast_v2"
    root.mkdir(parents=True)
    (root / "escape").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="output"):
        checked_output_path(root / "escape/old", project_root=tmp_path)


def binding():
    return {key: canonical_hash(key) for key in ("source", "config", "data", "selector", "packet")}


def test_run_writer_lock_resume_hash_and_no_clobber(tmp_path):
    out = tmp_path / "runs/modeling_contrast_v2/N0"
    with RunWriter(out, binding(), project_root=tmp_path) as writer:
        writer.write_json("sample.json", {"x": 1})
        with pytest.raises(FileExistsError):
            writer.write_json("sample.json", {"x": 2})
        with (
            pytest.raises(RuntimeError, match="writer"),
            RunWriter(out, binding(), resume=True, project_root=tmp_path),
        ):
            pass
    with RunWriter(out, binding(), resume=True, project_root=tmp_path) as writer:
        assert writer.resumed
        assert writer.output_hashes["sample.json"] == sha256_file(out / "sample.json")
    with (
        pytest.raises(ValueError, match="binding"),
        RunWriter(
            out,
            {**binding(), "packet": canonical_hash("changed")},
            resume=True,
            project_root=tmp_path,
        ),
    ):
        pass
    (out / "sample.json").write_text("corruption")
    with (
        pytest.raises(ValueError, match="hash"),
        RunWriter(out, binding(), resume=True, project_root=tmp_path),
    ):
        pass


def test_atomic_writes_reject_objects_and_preserve_existing(tmp_path):
    out = tmp_path / "runs/modeling_contrast_v2/test"
    out.mkdir(parents=True)
    write_json(out / "result.json", {"x": 1}, project_root=tmp_path)
    with pytest.raises(FileExistsError):
        write_json(out / "result.json", {"x": 2}, project_root=tmp_path)
    with pytest.raises(ValueError, match="object"):
        write_npz(out / "bad.npz", {"x": np.array([{}], dtype=object)}, project_root=tmp_path)
    with pytest.raises(ValueError, match="finite"):
        write_npz(out / "bad.npz", {"x": np.array([np.nan])}, project_root=tmp_path)
    write_npz(out / "valid.npz", {"x": np.arange(3)}, project_root=tmp_path)
    with np.load(out / "valid.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["x"], [0, 1, 2])


def test_source_snapshot_is_byte_identical(tmp_path):
    src = tmp_path / "src/modeling_contrast"
    src.mkdir(parents=True)
    (src / "example.py").write_text("x = 1\n")
    out = tmp_path / "runs/modeling_contrast_v2/snapshot"
    result = source_snapshot(src, out, project_root=tmp_path)
    assert result["files"]["example.py"] == sha256_file(src / "example.py")
    assert (out / "files/example.py").read_bytes() == (src / "example.py").read_bytes()


def test_completed_resume_retains_original_status(tmp_path):
    out = tmp_path / 'runs/modeling_contrast_v2/completed'
    with RunWriter(out, binding(), project_root=tmp_path) as writer:
        writer.write_json('x.json', {'value': 1})
    with RunWriter(out, binding(), resume=True, project_root=tmp_path) as writer:
        assert writer.previous_status == 'COMPLETE'
