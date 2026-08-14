from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from compbias.gpu_pilot.analysis import main as analysis_main
from compbias.gpu_pilot.chart_data import generate_dataset
from compbias.gpu_pilot.collection import calibration_gate
from compbias.gpu_pilot.config import PilotDataConfig
from compbias.gpu_pilot.structured_generation import StructuredGeneration
from compbias.gpu_pilot.training import outcome_reward
from compbias.models.structured_parser import parse_trajectory


def test_natural_error_type_distinguishes_exact_perception_from_reasoning_error() -> None:
    from compbias.gpu_pilot.collection import _error_type

    record = {"values": [3, 7, 5], "answer": 4}
    correct = parse_trajectory(
        '<perception>{"values":[3,7,5]}</perception>'
        '<reasoning>{"operation":"max_minus_min"}</reasoning><answer>4</answer>',
        sample_id="correct",
    )
    wrong = parse_trajectory(
        '<perception>{"values":[3,7,5]}</perception>'
        '<reasoning>{"operation":"max_minus_min"}</reasoning><answer>3</answer>',
        sample_id="wrong",
    )

    assert _error_type(record, correct) == "none"
    assert _error_type(record, wrong) == "reasoning_error"


def test_pilot_a_rows_skip_parse_failures_and_use_canonical_mediator() -> None:
    from compbias.gpu_pilot.training import _pilot_a_rows

    parse_failure = {
        "error_type": "parse_failure",
        "parsed": {"perceived_scene": None},
        "question": "ignored",
        "answer": 0,
        "operation": "sum",
        "values": [1, 2, 3, 4],
    }
    visual_error = {
        "sample_id": "pilot_train-000000",
        "error_type": "visual_error",
        "parsed": {"perceived_scene": {"values": [4, 7, 5]}},
        "question": "What is max minus min?",
        "answer": 3,
        "operation": "max_minus_min",
        "values": [3, 7, 5],
    }

    rows = _pilot_a_rows([parse_failure, visual_error])

    assert len(rows) == 1
    assert isinstance(rows[0]["prompt"], list)
    prompt = rows[0]["prompt"][0]
    assert prompt["role"] == "user"
    content = prompt["content"]
    assert 'Evidence: {"values":[4,7,5]}' in content
    assert '<perception>{"values":[INTEGER,INTEGER,INTEGER]}</perception>' in content
    assert '<reasoning>{"operation":"max_minus_min"}</reasoning>' in content
    assert "exactly 3 integers" in content
    assert content.endswith("</answer>")
    assert "</answer>." not in content


def test_pilot_b_prompt_reuses_closed_structured_grammar() -> None:
    from compbias.gpu_pilot.training import _prompt_for_b

    prompt = _prompt_for_b(
        {
            "question": "What is the sum of the first two values?",
            "operation": "sum",
            "values": [4, 7, 5, 6],
        }
    )

    assert prompt[0]["role"] == "user"
    assert prompt[0]["content"][0] == {"type": "image"}
    text = prompt[0]["content"][1]["text"]
    assert (
        '<perception>{"values":[INTEGER,INTEGER,INTEGER,INTEGER]}</perception>' in text
    )
    assert '<reasoning>{"operation":"sum"}</reasoning>' in text
    assert "exactly 4 integers" in text
    assert text.endswith("</answer>")
    assert "</answer>." not in text


def test_audited_reward_retains_raw_completion_and_reward(tmp_path: Path) -> None:
    from compbias.gpu_pilot.training import _audited_reward

    raw = (
        '<perception>{"values":[3,7,5]}</perception>'
        '<reasoning>{"operation":"max_minus_min"}</reasoning><answer>4</answer>'
    )
    target = tmp_path / "rollouts.jsonl"
    reward = _audited_reward(target)

    assert reward(
        [raw],
        [4],
        ["max_minus_min"],
        [3],
        ["sample-1"],
    ) == [1.0]
    saved = json.loads(target.read_text(encoding="utf-8"))
    assert saved["raw_completion"] == raw
    assert saved["reward"] == 1.0


def test_gpu_artifact_publication_rejects_target_and_parent_symlinks(tmp_path: Path) -> None:
    from compbias.gpu_pilot.safe_io import atomic_write_json_text

    root = tmp_path / "outputs"
    root.mkdir()
    victim = tmp_path / "victim.json"
    victim.write_text("unchanged", encoding="utf-8")
    target = root / "report.json"
    target.symlink_to(victim)
    with pytest.raises(RuntimeError, match="not a regular file"):
        atomic_write_json_text(root, target, "changed")
    assert victim.read_text(encoding="utf-8") == "unchanged"

    target.unlink()
    escaped = tmp_path / "escaped"
    escaped.mkdir()
    (root / "nested").symlink_to(escaped, target_is_directory=True)
    with pytest.raises(RuntimeError, match="parent is not a regular directory"):
        atomic_write_json_text(root, root / "nested/report.json", "changed")
    assert not (escaped / "report.json").exists()


def test_gpu_artifact_publication_rejects_parent_component_escape(tmp_path: Path) -> None:
    from compbias.gpu_pilot.safe_io import atomic_write_json_text

    root = tmp_path / "outputs"
    root.mkdir()
    victim = tmp_path / "victim.json"
    victim.write_text("unchanged", encoding="utf-8")

    with pytest.raises(RuntimeError, match="escapes its approved root"):
        atomic_write_json_text(root, root / "../victim.json", "changed")
    assert victim.read_text(encoding="utf-8") == "unchanged"


def test_pilot_paths_normalize_parent_components_before_disjoint_check(tmp_path: Path) -> None:
    from compbias.gpu_pilot.config import PilotPaths

    with pytest.raises(ValueError, match="disjoint storage roots"):
        PilotPaths(
            project_root=tmp_path / "project",
            model_path=tmp_path / "model",
            data=tmp_path / "storage/data",
            outputs=tmp_path / "storage/data/../data",
            checkpoints=tmp_path / "storage/checkpoints",
            trajectories=tmp_path / "storage/trajectories",
            cache=tmp_path / "storage/cache",
        )


def test_gpu_stage_can_allocate_a_new_run_after_an_incomplete_run(tmp_path: Path) -> None:
    from compbias.gpu_pilot.safe_io import prepare_new_output_directory

    root = tmp_path / "outputs"
    old = root / "pilot_a/runs/run-old"
    old.mkdir(parents=True)
    (old / "partial.txt").write_text("interrupted", encoding="utf-8")
    new = root / "pilot_a/runs/run-new"

    assert prepare_new_output_directory(root, new) == new
    assert not new.exists()
    assert (old / "partial.txt").is_file()


def test_training_executor_requires_ack_before_any_heavy_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from compbias.gpu_pilot.training import run_grpo_stage

    monkeypatch.delenv("COMPBIAS_GPU_EXECUTION_ACK", raising=False)
    with pytest.raises(RuntimeError, match="ACK is missing"):
        run_grpo_stage({})


def test_training_executor_rejects_mutated_in_memory_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from compbias.gpu_pilot.training import run_grpo_stage

    config = yaml.safe_load(Path("configs/train/pilot_a.yaml").read_text(encoding="utf-8"))
    config["training"]["max_steps"] = 999
    config.update(
        {
            "validated_stage_config_path": str(Path("configs/train/pilot_a.yaml").resolve()),
            "validated_paths_config_path": str(Path("configs/paths.yaml").resolve()),
        }
    )
    monkeypatch.setenv(
        "COMPBIAS_GPU_EXECUTION_ACK",
        "I_UNDERSTAND_THIS_STARTS_GPU_TRAINING",
    )

    with pytest.raises(RuntimeError, match="differs from the validated stage config"):
        run_grpo_stage(config)


def test_training_executor_requires_live_hardware_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from compbias.gpu_pilot import preflight, training

    project_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(Path("configs/train/pilot_a.yaml").read_text(encoding="utf-8"))
    config.update(
        {
            "validated_stage_config_path": str(Path("configs/train/pilot_a.yaml").resolve()),
            "validated_paths_config_path": str(Path("configs/paths.yaml").resolve()),
        }
    )
    paths = SimpleNamespace(project_root=project_root, outputs=project_root / "outputs")
    monkeypatch.setattr(training, "load_pilot_paths", lambda _path: paths)
    monkeypatch.setattr(
        training,
        "capture_environment",
        lambda **_kwargs: {
            "git_commit": "c" * 40,
            "git_dirty": False,
            "package_versions": {},
            "cuda_available": False,
            "gpu_devices": [],
        },
    )
    monkeypatch.setattr(
        preflight,
        "audit_server",
        lambda _paths: (_ for _ in ()).throw(RuntimeError("live audit invoked")),
    )
    monkeypatch.setenv(
        "COMPBIAS_GPU_EXECUTION_ACK",
        "I_UNDERSTAND_THIS_STARTS_GPU_TRAINING",
    )

    with pytest.raises(RuntimeError, match="live audit invoked"):
        training.run_grpo_stage(config)


def test_structured_generation_retries_without_echoing_invalid_output() -> None:
    from compbias.gpu_pilot.structured_generation import generate_with_format_retries

    invalid = '```json\n{"values":[4,7,5]}\n```'
    valid = (
        '<perception>{"values":[3,7,5]}</perception>'
        '<reasoning>{"operation":"max_minus_min"}</reasoning>'
        "<answer>4</answer>"
    )
    responses = iter((invalid, valid))
    prompts: list[tuple[dict[str, object], ...]] = []

    def generate(messages: tuple[dict[str, object], ...]) -> str:
        prompts.append(messages)
        return next(responses)

    result = generate_with_format_retries(
        generate,
        question="Read the chart and compute max minus min.",
        operation="max_minus_min",
        sample_id="smoke-000001",
        expected_value_count=3,
    )

    assert result.parsed.status.value == "ok"
    assert result.raw_text == valid
    assert len(result.attempts) == 2
    assert [attempt["status"] for attempt in result.attempts] == ["malformed", "ok"]
    assert invalid not in json.dumps(prompts[1], sort_keys=True)
    assert "previous attempt failed" in json.dumps(prompts[1]).lower()


def test_structured_prompt_does_not_teach_a_trailing_period() -> None:
    from compbias.gpu_pilot.structured_generation import build_structured_messages

    messages = build_structured_messages(
        question="Read the chart and compute max minus min.",
        operation="max_minus_min",
        retry_index=1,
        expected_value_count=3,
    )
    rendered = " ".join(str(message["content"]) for message in messages)

    assert "</answer>." not in rendered
    assert "final character must be >" in rendered.lower()


def test_structured_prompt_example_matches_expected_value_count() -> None:
    from compbias.gpu_pilot.structured_generation import build_structured_messages

    three = json.dumps(
        build_structured_messages(
            question="question",
            operation="max_minus_min",
            retry_index=0,
            expected_value_count=3,
        ),
        sort_keys=True,
    )
    four = json.dumps(
        build_structured_messages(
            question="question",
            operation="sum",
            retry_index=0,
            expected_value_count=4,
        ),
        sort_keys=True,
    )

    assert '\\"values\\":[2,8,5]' in three
    assert '\\"values\\":[2,8,5,5]' in four
    assert "exactly 3 integers" in three
    assert "exactly 4 integers" in four


def test_structured_prompt_requires_full_scene_transcription_for_partial_operations() -> None:
    from compbias.gpu_pilot.structured_generation import build_structured_messages

    messages = build_structured_messages(
        question="What is the sum of the first two values?",
        operation="sum",
        retry_index=0,
        expected_value_count=4,
    )
    rendered = " ".join(str(message["content"]) for message in messages)

    assert "transcribe all 4 labeled values" in rendered.lower()
    assert "even when the question uses only some of them" in rendered.lower()
    assert "do not insert \\n or any escape sequence" in rendered.lower()


@pytest.mark.parametrize("operation", ["sum", "difference", "max_minus_min"])
def test_structured_prompt_binds_exact_array_delimiters(operation: str) -> None:
    from compbias.gpu_pilot.structured_generation import build_structured_instruction

    instruction = build_structured_instruction(
        operation=operation,
        expected_value_count=4,
    )

    assert (
        '<perception>{"values":[INTEGER,INTEGER,INTEGER,INTEGER]}</perception>'
        in instruction
    )
    assert (
        "the values field is one json array: keep the opening [ and closing ] around all four "
        "integers"
        in instruction.lower()
    )


def test_unbracketed_value_lists_remain_strictly_rejected() -> None:
    from compbias.gpu_pilot.structured_generation import validate_pilot_trajectory

    raw = (
        '<perception>{"values":12,18,3,13}</perception>'
        '<reasoning>{"operation":"difference"}</reasoning>'
        "<answer>-6</answer>"
    )
    parsed = validate_pilot_trajectory(
        parse_trajectory(raw, sample_id="calibration-000000"),
        operation="difference",
        expected_value_count=4,
    )

    assert parsed.status.value == "invalid_json"
    assert parsed.error_code == "invalid_perception_json"


def test_structured_generation_stops_after_two_format_retries() -> None:
    from compbias.gpu_pilot.structured_generation import generate_with_format_retries

    calls = 0

    def generate(_messages: tuple[dict[str, object], ...]) -> str:
        nonlocal calls
        calls += 1
        return '{"values":[4,7,5]}'

    result = generate_with_format_retries(
        generate,
        question="Read the chart and compute max minus min.",
        operation="max_minus_min",
        sample_id="smoke-000001",
        expected_value_count=3,
    )

    assert calls == 3
    assert result.parsed.status.value == "malformed"
    assert len(result.attempts) == 3


def test_structured_generation_rejects_invalid_prompt_and_budget_inputs() -> None:
    from compbias.gpu_pilot.structured_generation import (
        build_structured_messages,
        generate_with_format_retries,
    )

    with pytest.raises(ValueError, match="question must be a non-empty string"):
        build_structured_messages(
            question="", operation="sum", retry_index=0, expected_value_count=4
        )
    with pytest.raises(ValueError, match="question must not contain NUL"):
        build_structured_messages(
            question="unsafe\x00text",
            operation="sum",
            retry_index=0,
            expected_value_count=4,
        )
    with pytest.raises(ValueError, match="unsupported operation"):
        build_structured_messages(
            question="question", operation="product", retry_index=0, expected_value_count=4
        )
    with pytest.raises(TypeError, match="retry_index must be an integer"):
        build_structured_messages(
            question="question", operation="sum", retry_index=True, expected_value_count=4
        )
    with pytest.raises(ValueError, match="retry_index must be between"):
        build_structured_messages(
            question="question", operation="sum", retry_index=3, expected_value_count=4
        )
    with pytest.raises(ValueError, match="max_format_retries must be between"):
        generate_with_format_retries(
            lambda _messages: "unused",
            question="question",
            operation="sum",
            sample_id="sample",
            expected_value_count=4,
            max_format_retries=3,
        )
    with pytest.raises(TypeError, match="model decoder must return a string"):
        generate_with_format_retries(
            lambda _messages: None,  # type: ignore[return-value]
            question="question",
            operation="sum",
            sample_id="sample",
            expected_value_count=4,
        )


def test_pilot_schema_failure_retries_and_rejects_string_answer() -> None:
    from compbias.gpu_pilot.structured_generation import generate_with_format_retries

    invalid = '<perception>{}</perception><reasoning>{}</reasoning><answer>"4"</answer>'
    valid = (
        '<perception>{"values":[3,7,5]}</perception>'
        '<reasoning>{"operation":"max_minus_min"}</reasoning>'
        "<answer>4.0</answer>"
    )
    responses = iter((invalid, valid))
    result = generate_with_format_retries(
        lambda _messages: next(responses),
        question="Read the chart and compute max minus min.",
        operation="max_minus_min",
        sample_id="smoke-000001",
        expected_value_count=3,
    )

    assert [attempt["status"] for attempt in result.attempts] == ["invalid_type", "ok"]
    assert result.parsed.status.value == "ok"
    assert result.parsed.answer == 4.0


@pytest.mark.parametrize(
    ("raw", "error_code"),
    [
        (
            '<perception>{"values":[3,7,5],"extra":1}</perception>'
            '<reasoning>{"operation":"max_minus_min"}</reasoning><answer>4</answer>',
            "pilot_perception_schema",
        ),
        (
            '<perception>{"values":[3,7]}</perception>'
            '<reasoning>{"operation":"max_minus_min"}</reasoning><answer>4</answer>',
            "pilot_values_shape",
        ),
        (
            '<perception>{"values":[3,true,5]}</perception>'
            '<reasoning>{"operation":"max_minus_min"}</reasoning><answer>4</answer>',
            "pilot_values_type",
        ),
        (
            '<perception>{"values":[3,7,5]}</perception>'
            '<reasoning>{"operation":"sum"}</reasoning><answer>4</answer>',
            "pilot_operation_mismatch",
        ),
        (
            '<perception>{"values":[3,7,5]}</perception>'
            '<reasoning>{"operation":"max_minus_min","extra":1}</reasoning>'
            "<answer>4</answer>",
            "pilot_reasoning_schema",
        ),
        (
            '<perception>{"values":[3,7,5]}</perception>'
            '<reasoning>{"operation":"max_minus_min"}</reasoning><answer>true</answer>',
            "pilot_answer_type",
        ),
    ],
)
def test_pilot_schema_is_closed(raw: str, error_code: str) -> None:
    from compbias.gpu_pilot.structured_generation import validate_pilot_trajectory

    parsed = validate_pilot_trajectory(
        parse_trajectory(raw, sample_id="fixture"),
        operation="max_minus_min",
        expected_value_count=3,
    )

    assert parsed.status.value == "invalid_type"
    assert parsed.error_code == error_code


def test_pilot_numeric_validation_rejects_huge_integers_without_overflow() -> None:
    from compbias.gpu_pilot.structured_generation import (
        numeric_answer_matches,
        validate_pilot_trajectory,
    )

    huge = 10**1_000
    raw = (
        f'<perception>{{"values":[3,{huge},5]}}</perception>'
        '<reasoning>{"operation":"max_minus_min"}</reasoning>'
        f"<answer>{huge}</answer>"
    )
    parsed = validate_pilot_trajectory(
        parse_trajectory(raw, sample_id="fixture"),
        operation="max_minus_min",
        expected_value_count=3,
    )

    assert parsed.status.value == "invalid_type"
    assert numeric_answer_matches(huge, 4) is False


def test_smoke_main_returns_nonzero_for_malformed_protocol(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from compbias.gpu_pilot import qwen_smoke

    monkeypatch.setattr(
        qwen_smoke,
        "load_pilot_paths",
        lambda _path: SimpleNamespace(model_path=tmp_path, outputs=tmp_path),
    )
    monkeypatch.setattr(
        qwen_smoke,
        "run_smoke",
        lambda _model, _output: {
            "parsed": {"status": "malformed"},
            "smoke_passed": False,
            "answer_correct": False,
        },
    )

    assert qwen_smoke.main(["--paths", str(tmp_path / "paths.yaml")]) == 3

    monkeypatch.setattr(
        qwen_smoke,
        "run_smoke",
        lambda _model, _output: {
            "parsed": {"status": "ok"},
            "smoke_passed": True,
            "answer_correct": False,
        },
    )
    assert qwen_smoke.main(["--paths", str(tmp_path / "paths.yaml")]) == 3


def test_natural_collection_never_resamples_a_parse_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from compbias.gpu_pilot import collection

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    record = {
        "sample_id": "calibration-000001",
        "split": "calibration",
        "operation": "max_minus_min",
        "question": "What is the maximum value minus the minimum value?",
        "values": [3, 7, 5],
        "answer": 4,
        "image": "images/calibration-000001.png",
    }
    (dataset / "records.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (dataset / "manifest.json").write_text(
        json.dumps({"images_sha256": "a" * 64}), encoding="utf-8"
    )
    monkeypatch.setattr(collection, "load_local_qwen", lambda _path: (object(), object()))
    monkeypatch.setattr(collection, "model_snapshot_sha256", lambda _path: "b" * 64)
    monkeypatch.setattr(
        "compbias.gpu_pilot.execution_gate._validate_canonical_dataset",
        lambda *_args: None,
    )
    observed_retry_budgets: list[int] = []

    def fake_generate(
        _generate_once: object,
        **kwargs: object,
    ) -> StructuredGeneration:
        observed_retry_budgets.append(int(kwargs["max_format_retries"]))
        raw = '{"values":[4,7,5]}'
        parsed = parse_trajectory(raw, sample_id=str(kwargs["sample_id"]))
        return StructuredGeneration(
            raw_text=raw,
            parsed=parsed,
            attempts=(
                {
                    "attempt_index": 0,
                    "raw_text": raw,
                    "status": parsed.status.value,
                    "error_code": parsed.error_code,
                },
            ),
        )

    monkeypatch.setattr(collection, "generate_with_format_retries", fake_generate)
    report = collection.collect_split(
        dataset,
        tmp_path / "model",
        tmp_path / "natural.jsonl",
        split="calibration",
        data_config_path=tmp_path / "data.yaml",
    )

    assert observed_retry_budgets == [0]
    assert report["parse_rate"] == 0.0
    saved = json.loads((tmp_path / "natural.jsonl").read_text(encoding="utf-8"))
    assert saved["format_retries"] == 0
    assert saved["parsed"]["status"] == "malformed"


def test_natural_collection_failure_does_not_publish_partial_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from compbias.gpu_pilot import collection

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    records = [
        {
            "sample_id": f"calibration-{index:06d}",
            "split": "calibration",
            "operation": "sum",
            "question": "What is the sum of the first two values?",
            "values": [3, 7, 5, 2],
            "answer": 10,
            "image": f"images/calibration-{index:06d}.png",
        }
        for index in range(2)
    ]
    (dataset / "records.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    (dataset / "manifest.json").write_text(
        json.dumps({"images_sha256": "a" * 64}), encoding="utf-8"
    )
    monkeypatch.setattr(collection, "load_local_qwen", lambda _path: (object(), object()))
    monkeypatch.setattr(collection, "model_snapshot_sha256", lambda _path: "b" * 64)
    monkeypatch.setattr(
        "compbias.gpu_pilot.execution_gate._validate_canonical_dataset",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        collection,
        "generate_with_format_retries",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("decoder failed")),
    )
    target = tmp_path / "natural" / "calibration_records.jsonl"

    with pytest.raises(RuntimeError, match="decoder failed"):
        collection.collect_split(
            dataset,
            tmp_path / "model",
            target,
            split="calibration",
            data_config_path=tmp_path / "data.yaml",
            output_root=tmp_path,
        )

    assert not target.exists()
    assert not list(target.parent.glob(f".{target.name}.*"))


def test_failed_calibration_can_be_archived_before_reviewed_rerun(tmp_path: Path) -> None:
    from compbias.gpu_pilot.collection import _archive_failed_collection

    root = tmp_path / "trajectories"
    records = root / "natural/calibration_records.jsonl"
    summary = root / "natural/calibration_records.summary.json"
    records.parent.mkdir(parents=True)
    records.write_text("{}\n", encoding="utf-8")
    summary.write_text(json.dumps({"gate_passed": False}), encoding="utf-8")

    archive = _archive_failed_collection(root, records, summary)

    assert not records.exists()
    assert not summary.exists()
    assert (archive / records.name).is_file()
    assert (archive / summary.name).is_file()


def _small_data_config() -> PilotDataConfig:
    return PilotDataConfig(
        dataset_id="CVA-Chart-Pilot-v0.2",
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


@pytest.mark.parametrize("chart_type", ["grouped_bar", "line"])
def test_chart_renderer_uses_axis_ticks_instead_of_direct_value_labels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    chart_type: str,
) -> None:
    from compbias.gpu_pilot import chart_data

    labels: list[str] = []

    def capture_text(_draw: object, _position: object, text: object, **_kwargs: object) -> None:
        labels.append(str(text))

    monkeypatch.setattr(chart_data.ImageDraw.ImageDraw, "text", capture_text)
    chart_data._draw_chart(
        tmp_path / "chart.png",
        chart_type=chart_type,
        values=(3, 7, 13, 17),
        size=(512, 384),
        ood=False,
        render_mode="axis_scale_v0_2",
    )

    numeric_labels = {label for label in labels if label.isdigit()}
    assert numeric_labels == {str(value) for value in range(0, 21, 2)}
    assert {"3", "7", "13", "17"}.isdisjoint(numeric_labels)


def test_versioned_renderer_preserves_legacy_labels_and_scales_value_21(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from compbias.gpu_pilot import chart_data

    labels: list[str] = []

    def capture_text(_draw: object, _position: object, text: object, **_kwargs: object) -> None:
        labels.append(str(text))

    monkeypatch.setattr(chart_data.ImageDraw.ImageDraw, "text", capture_text)
    chart_data._draw_chart(
        tmp_path / "legacy.png",
        chart_type="line",
        values=(3, 7, 13, 17),
        size=(512, 384),
        ood=False,
        render_mode="direct_labels_v0_1",
    )
    assert {"3", "7", "13", "17"}.issubset(labels)

    labels.clear()
    chart_data._draw_chart(
        tmp_path / "v0_2.png",
        chart_type="line",
        values=(21, 7, 13, 17),
        size=(512, 384),
        ood=True,
        render_mode="axis_scale_v0_2",
    )
    assert "22" in labels
    assert "21" not in labels


def test_dataset_generation_is_deterministic_and_no_clobber(tmp_path: Path) -> None:
    output = tmp_path / "dataset"
    manifest = generate_dataset(_small_data_config(), output)

    assert manifest["record_count"] == 26
    assert manifest["counterfactual_pairs"] == 2
    assert len(manifest["natural_audit_ids"]) == 2
    assert manifest["images_generated"] == 28
    assert len(manifest["records_sha256"]) == 64
    assert len(manifest["counterfactual_sha256"]) == 64
    assert len(manifest["images_sha256"]) == 64

    manifest_bytes = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        generate_dataset(_small_data_config(), output)
    assert (output / "manifest.json").read_bytes() == manifest_bytes


def test_v0_1_and_v0_2_coexist_without_semantic_task_drift(tmp_path: Path) -> None:
    current = _small_data_config()
    legacy = replace(current, dataset_id="CVA-Chart-Pilot-v0.1")
    legacy_root = tmp_path / legacy.output_slug
    current_root = tmp_path / current.output_slug
    replay_root = tmp_path / "legacy-replay"

    legacy_manifest = generate_dataset(legacy, legacy_root)
    current_manifest = generate_dataset(current, current_root)
    replay_manifest = generate_dataset(legacy, replay_root)

    def normalized_records(path: Path) -> list[dict[str, object]]:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        return [{**row, "dataset_id": "normalized"} for row in rows]

    assert legacy_root.is_dir()
    assert current_root.is_dir()
    assert normalized_records(legacy_root / "records.jsonl") == normalized_records(
        current_root / "records.jsonl"
    )
    assert (legacy_root / "counterfactual_pairs.jsonl").read_bytes() == (
        current_root / "counterfactual_pairs.jsonl"
    ).read_bytes()
    assert legacy_manifest["images_sha256"] != current_manifest["images_sha256"]
    assert legacy_manifest["images_sha256"] == replay_manifest["images_sha256"]


def test_execution_gate_replays_seeded_dataset_pixels(tmp_path: Path) -> None:
    from compbias.gpu_pilot.execution_gate import _validate_canonical_dataset

    output = tmp_path / "dataset"
    config = _small_data_config()
    generate_dataset(config, output)
    config_path = tmp_path / "data.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "dataset_id": config.dataset_id,
                "seed": config.seed,
                "image_size": list(config.image_size),
                "chart_types": list(config.chart_types),
                "operations": list(config.operations),
                "split_counts": dict(config.split_counts),
                "counterfactual_pairs": config.counterfactual_pairs,
                "natural_audit": config.natural_audit,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    cache = tmp_path / "cache"

    _validate_canonical_dataset(output / "manifest.json", config_path, cache)
    image = next((output / "images").glob("*.png"))
    image.write_bytes(image.read_bytes() + b"tampered")

    with pytest.raises(RuntimeError, match="committed seed"):
        _validate_canonical_dataset(output / "manifest.json", config_path, cache)


def test_execution_gate_replays_the_full_registered_dataset(tmp_path: Path) -> None:
    from compbias.gpu_pilot.config import load_pilot_data_config
    from compbias.gpu_pilot.execution_gate import (
        _validate_dataset_bundle,
        _validate_natural_records,
    )

    output = tmp_path / "dataset"
    manifest = generate_dataset(
        load_pilot_data_config(Path("configs/data/cva_chart_pilot_v0_2.yaml")),
        output,
    )

    records = _validate_dataset_bundle(output / "manifest.json", manifest)
    assert len(records) == 2_800
    assert set(record["split"] for record in records.values()) == {
        "calibration",
        "smoke_train",
        "pilot_train",
        "dev",
        "iid_test",
        "mechanism_ood",
    }
    natural_path = tmp_path / "calibration_records.jsonl"
    with natural_path.open("x", encoding="utf-8") as stream:
        for index in range(200):
            sample_id = f"calibration-{index:06d}"
            source = records[sample_id]
            raw = (
                f'<perception>{{"values":{json.dumps(source["values"])}}}</perception>'
                f'<reasoning>{{"operation":{json.dumps(source["operation"])}}}</reasoning>'
                f"<answer>{json.dumps(source['answer'])}</answer>"
            )
            parsed = parse_trajectory(raw, sample_id=sample_id)
            payload = {
                **source,
                "rollout_id": f"calibration-rollout-{index:06d}",
                "raw_text": raw,
                "parsed": parsed.to_mapping(),
                "format_attempts": [
                    {
                        "attempt_index": 0,
                        "raw_text": raw,
                        "status": "ok",
                        "error_code": None,
                    }
                ],
                "format_retries": 0,
                "reward": 1,
                "error_type": "none",
            }
            stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
    replay = _validate_natural_records(
        natural_path,
        split="calibration",
        dataset_records=records,
    )
    assert replay == {
        "records": 200,
        "answer_accuracy": 1.0,
        "parse_rate": 1.0,
        "natural_perception_error_rate": 0.0,
        "error_counts": {"none": 200},
    }


@pytest.mark.parametrize(
    ("metrics", "expected"),
    [
        (
            {
                "answer_accuracy": 0.55,
                "natural_perception_error_rate": 0.25,
                "parse_rate": 0.98,
                "error_counts": {
                    "visual_error": 20,
                    "compensated_visual_error": 20,
                    "reasoning_error": 20,
                },
            },
            True,
        ),
        (
            {
                "answer_accuracy": 0.90,
                "natural_perception_error_rate": 0.25,
                "parse_rate": 0.98,
                "error_counts": {
                    "visual_error": 20,
                    "compensated_visual_error": 20,
                    "reasoning_error": 20,
                },
            },
            False,
        ),
        (
            {
                "answer_accuracy": 0.55,
                "natural_perception_error_rate": 0.25,
                "parse_rate": 0.94,
                "error_counts": {
                    "visual_error": 20,
                    "compensated_visual_error": 20,
                    "reasoning_error": 20,
                },
            },
            False,
        ),
        (
            {
                "answer_accuracy": 0.55,
                "natural_perception_error_rate": 0.25,
                "parse_rate": 0.98,
                "error_counts": {
                    "visual_error": 20,
                    "reasoning_error": 20,
                    "parse_failure": 20,
                },
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
            [
                {
                    "content": (
                        '<perception>{}</perception><reasoning>{}</reasoning><answer>"7"</answer>'
                    )
                }
            ],
            [
                {
                    "content": (
                        '<perception>{"values":[4,3]}</perception>'
                        '<reasoning>{"operation":"sum"}</reasoning>'
                        "<answer>7.0</answer>"
                    )
                }
            ],
        ],
        answer=[7, 7, 7, 7, 7],
        operation=["sum"] * 5,
        expected_value_count=[2] * 5,
    )
    assert rewards == [1.0, 0.0, 0.0, 0.0, 1.0]


def test_pilot_a_prompt_uses_validated_perception_instead_of_missing_raw_field() -> None:
    from compbias.gpu_pilot.training import _prompt_for_a

    prompt = _prompt_for_a(
        {
            "question": "What is the maximum minus the minimum?",
            "operation": "max_minus_min",
            "values": [3, 7, 5],
            "parsed": {"perceived_scene": {"values": [4, 7, 5]}},
        }
    )

    assert 'Evidence: {"values":[4,7,5]}' in prompt[0]["content"]
    assert "Evidence: None" not in prompt[0]["content"]


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


def test_gpu_lock_and_runbook_bind_portable_security_review() -> None:
    root = Path(__file__).resolve().parents[1]
    lock_path = root / "requirements-gpu.lock.txt"
    lock_bytes = lock_path.read_bytes()
    lock_lines = lock_bytes.decode("utf-8").splitlines()
    parsed = [Requirement(line) for line in lock_lines]
    normalized_names = [canonicalize_name(requirement.name) for requirement in parsed]
    locked = {
        canonicalize_name(requirement.name): str(requirement.specifier) for requirement in parsed
    }

    assert len(lock_lines) == 125
    assert lock_lines == sorted(lock_lines, key=str.casefold)
    assert len(normalized_names) == len(set(normalized_names))
    assert all(requirement.url is None for requirement in parsed)
    assert hashlib.sha256(lock_bytes).hexdigest() == (
        "d928379a590e5071d9b5042fe99d480f57ab187f0cb3a74e13af219a6048aeb3"
    )

    security_overlay = {
        "filelock": "==3.20.3",
        "jinja2": "==3.1.6",
        "pip": "==26.1.2",
        "protobuf": "==6.33.5",
        "pygments": "==2.20.0",
        "setuptools": "==84.0.0",
        "tornado": "==6.5.7",
        "uv": "==0.11.15",
        "wheel": "==0.46.3",
    }
    assert {name: locked.get(name) for name in security_overlay} == security_overlay
    assert locked["torch"] == "==2.8.0+cu128"

    candidate = (root / "requirements-gpu.in").read_text(encoding="utf-8")
    candidate_exact = {
        canonicalize_name(requirement.name): str(requirement.specifier)
        for requirement in (
            Requirement(line)
            for line in candidate.splitlines()
            if line.strip() and not line.startswith("#")
        )
    }
    assert {name: locked.get(name) for name in candidate_exact} == candidate_exact
    for name, version in security_overlay.items():
        assert f"{name}=={version.removeprefix('==')}" in candidate.casefold()

    runbook = (root / "docs" / "SERVER_SETUP.md").read_text(encoding="utf-8")
    assert "python -m pip list --format=freeze --exclude-editable" in runbook
    assert "pip freeze --all" not in runbook
    assert "records resolved versions but not" in runbook
    assert "original installation provenance" in runbook
    assert "not a standalone proof that no package originally came from a" in runbook
    assert "skipped the" in runbook
    assert "local-version `torch`, `torchaudio`, and `torchvision` builds" in runbook
    assert "`+cu128` builds are not published on PyPI" in runbook
    assert "pip-audit==2.10.1" in runbook
    assert "d928379a590e5071d9b5042fe99d480f57ab187f0cb3a74e13af219a6048aeb3" in runbook


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
