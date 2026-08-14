from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from compbias.gpu_pilot.analysis import main as analysis_main
from compbias.gpu_pilot.chart_data import generate_dataset
from compbias.gpu_pilot.collection import calibration_gate
from compbias.gpu_pilot.config import PilotDataConfig
from compbias.gpu_pilot.training import outcome_reward


def _small_data_config() -> PilotDataConfig:
    return PilotDataConfig(
        dataset_id="CVA-Chart-Pilot-v0.1",
        seed=20260814,
        image_size=(320, 240),
        chart_types=("grouped_bar", "line"),
        operations=("difference", "sum", "max_minus_min"),
        split_counts={
            "calibration": 4,
            "smoke_train": 4,
            "pilot_train": 6,
            "dev": 4,
            "iid_test": 4,
            "mechanism_ood": 4,
        },
        counterfactual_pairs=2,
        natural_audit=2,
    )


def test_dataset_generation_is_deterministic_and_no_clobber(tmp_path: Path) -> None:
    output = tmp_path / "dataset"
    manifest = generate_dataset(_small_data_config(), output)

    assert manifest["record_count"] == 26
    assert manifest["counterfactual_pairs"] == 2
    assert len(manifest["natural_audit_ids"]) == 2
    assert manifest["images_generated"] == 28
    assert len(manifest["records_sha256"]) == 64
    assert len(manifest["counterfactual_sha256"]) == 64

    manifest_bytes = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        generate_dataset(_small_data_config(), output)
    assert (output / "manifest.json").read_bytes() == manifest_bytes


@pytest.mark.parametrize(
    ("metrics", "expected"),
    [
        (
            {
                "answer_accuracy": 0.55,
                "natural_perception_error_rate": 0.25,
                "parse_rate": 0.98,
                "error_counts": {"visual": 20, "reasoning": 20, "parse": 20},
            },
            True,
        ),
        (
            {
                "answer_accuracy": 0.90,
                "natural_perception_error_rate": 0.25,
                "parse_rate": 0.98,
                "error_counts": {"visual": 20, "reasoning": 20, "parse": 20},
            },
            False,
        ),
        (
            {
                "answer_accuracy": 0.55,
                "natural_perception_error_rate": 0.25,
                "parse_rate": 0.94,
                "error_counts": {"visual": 20, "reasoning": 20, "parse": 20},
            },
            False,
        ),
    ],
)
def test_calibration_gate_is_closed(metrics: dict[str, object], expected: bool) -> None:
    failures = calibration_gate(metrics)
    assert (not failures) is expected


def test_outcome_reward_requires_a_strict_structured_answer() -> None:
    rewards = outcome_reward(
        completions=[
            [
                {
                    "content": (
                        '<perception>{"values":[4,3]}</perception>'
                        '<reasoning>{"operation":"sum"}</reasoning>'
                        "<answer>7</answer>"
                    )
                }
            ],
            [{"content": "7"}],
            [{"content": "<answer>8</answer>"}],
        ],
        answer=["7", "7", "7"],
    )
    assert rewards == [1.0, 0.0, 0.0]


def test_server_scripts_parse_and_gpu_requirements_do_not_replace_torch() -> None:
    root = Path(__file__).resolve().parents[1]
    scripts = sorted((root / "scripts" / "server").glob("*.sh"))
    assert {path.name for path in scripts} == {
        "bootstrap_env.sh",
        "preflight.sh",
        "run_smoke.sh",
        "setup_paths.sh",
    }
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)

    requirements = (root / "requirements-gpu.in").read_text(encoding="utf-8")
    assert "torch" not in {
        line.partition("==")[0].strip()
        for line in requirements.splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert "transformers==5.14.1" in requirements
    assert "trl==1.9.0" in requirements
    for dependency in (
        "pandas==2.3.3",
        "pyarrow==25.0.0",
        "scikit-learn==1.9.0",
        "Pillow==12.3.0",
        "PyYAML==6.0.3",
        "pytest==9.0.3",
    ):
        assert dependency in requirements


def test_pilot_a_consumes_the_natural_collection_output() -> None:
    root = Path(__file__).resolve().parents[1]
    stage = (root / "configs" / "train" / "pilot_a.yaml").read_text(encoding="utf-8")
    assert "natural_records: trajectories/natural/pilot_train_records.jsonl" in stage


def test_publication_docs_and_ignore_boundaries_are_explicit() -> None:
    root = Path(__file__).resolve().parents[1]
    required_docs = {
        "COMPENSABILITY_V2.md",
        "CPU_EVIDENCE.md",
        "GPU_PILOT_PROTOCOL.md",
        "RESEARCH_QUESTION.md",
        "SERVER_SETUP.md",
        "THEORY.md",
    }
    assert required_docs <= {path.name for path in (root / "docs").glob("*.md")}

    ignored = (root / ".gitignore").read_text(encoding="utf-8")
    for boundary in (
        "*.safetensors",
        "/data/generated/",
        "/outputs/",
        "/checkpoints/",
        "/trajectories/",
        "configs/paths.yaml",
    ):
        assert boundary in ignored


def test_analysis_is_blocked_without_gpu_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = tmp_path / "paths.yaml"
    paths.write_text(
        "\n".join(
            (
                "schema_version: 1",
                f"project_root: {tmp_path}",
                "model:",
                f"  qwen25vl3b:\n    path: {tmp_path / 'model'}",
                "storage:",
                f"  data: {tmp_path / 'data'}",
                f"  outputs: {tmp_path / 'outputs'}",
                f"  checkpoints: {tmp_path / 'checkpoints'}",
                f"  trajectories: {tmp_path / 'trajectories'}",
                f"  cache: {tmp_path / 'cache'}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    exit_code = analysis_main(["--paths", str(paths)])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["ready"] is False
    assert payload["claims_permitted"] == []
