from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from compensability_v4.data.splits import DatasetSplit
from compensability_v4.qwen.phase5_runtime import verify_phase5_package_lock
from compensability_v4.qwen.phase5_support import (
    CheckpointSceneMeasurement,
    PolicyCheckpoint,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts/v4"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PREPARE = _load("test_phase5_prepare", "09_prepare_phase5_support_dev.py")
MEASURE = _load("test_phase5_measure", "10_measure_policy_support.py")
SHA = "a" * 64


def _row() -> CheckpointSceneMeasurement:
    return CheckpointSceneMeasurement(
        scene_id="scene-a",
        family="cross_series",
        split=DatasetSplit.SUPPORT_DEV,
        checkpoint=PolicyCheckpoint.BASE,
        checkpoint_sha256=SHA,
        truth=(2, 3, 4, 5),
        observed=(9, 3, 4, 5),
        greedy_raw_output="2,3,4,5",
        greedy_token_ids=(2, 3, 4, 5),
        greedy_output=(2, 3, 4, 5),
        greedy_parse_success=True,
        greedy_success=True,
        greedy_observation_copy=False,
        candidate_logp_true=-1.0,
        candidate_logp_observed=-2.0,
        candidate_margin_true_observed=1.0,
        sample_raw_outputs=("2,3,4,5", "9,3,4,5"),
        sample_token_ids=((2, 3, 4, 5), (9, 3, 4, 5)),
        sample_seeds=(1, 2),
        sample_outputs=((2, 3, 4, 5), (9, 3, 4, 5)),
        sample_parse_success=(True, True),
        sample_success=(True, False),
        sample_observation_copy=(False, True),
    )


def test_phase5_checkpoint_cache_round_trips_and_is_hash_bound(tmp_path: Path) -> None:
    rows = (_row(),)
    MEASURE._write_cache(
        tmp_path,
        PolicyCheckpoint.BASE,
        rows,
        checkpoint_sha256=SHA,
        support_dev_sha256="b" * 64,
        config_sha256="c" * 64,
    )
    loaded = MEASURE._load_cache(
        tmp_path,
        PolicyCheckpoint.BASE,
        checkpoint_sha256=SHA,
        support_dev_sha256="b" * 64,
        config_sha256="c" * 64,
    )
    assert loaded == rows


def test_phase5_scripts_are_measurement_only_and_have_resume_surface() -> None:
    prepare_text = (SCRIPT_DIR / "09_prepare_phase5_support_dev.py").read_text(encoding="utf-8")
    measure_text = (SCRIPT_DIR / "10_measure_policy_support.py").read_text(encoding="utf-8")
    assert "generate_observation_with_cache" in prepare_text
    assert "optimizer" not in measure_text.lower()
    assert "Trainer" not in measure_text
    assert "_load_cache" in measure_text
    assert "_write_cache" in measure_text


def test_phase5_package_lock_closes_both_server_entrypoints() -> None:
    assert PREPARE._LOCKED_PATHS == MEASURE._LOCKED_PATHS
    digest = verify_phase5_package_lock(
        lock_path=ROOT / "configs/recoverability/v4/server_package_lock_phase_5.yaml",
        repository_root=ROOT,
        expected_paths=PREPARE._LOCKED_PATHS,
    )
    assert len(digest) == 64
