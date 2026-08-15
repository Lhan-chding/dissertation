from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from compbias.recoverability.stage1_v2 import (
    Stage1V2Scene,
    build_stage1_v2_messages,
    run_stage1_v2_probe,
    select_stage1_v2_probe_scenes,
)


def _source_rows(*, per_stratum: int = 5) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for chart_type in ("grouped_bar", "line"):
        for operation in ("difference", "max_minus_min", "sum"):
            for index in range(per_stratum):
                sample_id = f"dev-{chart_type}-{operation}-{index:02d}"
                rows.append(
                    {
                        "sample_id": sample_id,
                        "split": "dev",
                        "chart_type": chart_type,
                        "operation": operation,
                        "values": [3, 7, 5, 2],
                        "image": f"images/{sample_id}.png",
                        "question": "What is the sum of the first two values?",
                        "answer": 10,
                    }
                )
    return tuple(rows)


def test_stage1_v2_prompt_is_question_free_and_binds_four_literal_slots() -> None:
    assert tuple(inspect.signature(build_stage1_v2_messages).parameters) == ()

    messages = build_stage1_v2_messages()
    rendered = json.dumps(messages, ensure_ascii=False)
    exact_grammar = (
        '{"target_facts":[INTEGER,INTEGER,INTEGER,INTEGER],'
        '"redundant_facts":[],"axis_facts":["integer_ticks"]}'
    )

    assert len(messages) == 2
    assert exact_grammar in rendered
    assert "A, B, C, and D" in rendered
    assert "all four" in rendered.lower()
    assert "even if" not in rendered.lower()
    assert "sum of the first two" not in rendered.lower()
    assert "maximum" not in rendered.lower()
    assert "difference" not in rendered.lower()
    assert "question" not in rendered.lower()
    assert "markdown" in rendered.lower()
    assert "do not compute" in rendered.lower()


def test_stage1_v2_probe_selection_is_fixed_balanced_and_order_independent(
    tmp_path: Path,
) -> None:
    rows = _source_rows()
    selected = select_stage1_v2_probe_scenes(rows, dataset_root=tmp_path)
    reversed_selected = select_stage1_v2_probe_scenes(
        tuple(reversed(rows)), dataset_root=tmp_path
    )

    assert len(selected) == 24
    assert tuple(scene.scene_id for scene in selected) == tuple(
        scene.scene_id for scene in reversed_selected
    )
    counts: dict[tuple[str, str], int] = {}
    for scene in selected:
        key = (scene.chart_type, scene.operation)
        counts[key] = counts.get(key, 0) + 1
        assert scene.image_path.is_absolute()
        assert scene.values == (3, 7, 5, 2)
    assert set(counts.values()) == {4}

    with pytest.raises(ValueError, match="underfilled"):
        select_stage1_v2_probe_scenes(
            _source_rows(per_stratum=3), dataset_root=tmp_path
        )


def test_stage1_v2_probe_makes_one_call_per_scene_and_never_retries() -> None:
    scenes = tuple(
        Stage1V2Scene(
            scene_id=f"dev-{index:03d}",
            image_path=Path(f"/dataset/images/dev-{index:03d}.png"),
            chart_type="line" if index % 2 else "grouped_bar",
            operation=("difference", "max_minus_min", "sum")[index % 3],
            values=(3, 7, 5, 2),
        )
        for index in range(24)
    )
    calls = 0

    def generate(
        _scene: Stage1V2Scene,
        messages: tuple[dict[str, object], ...],
    ) -> str:
        nonlocal calls
        calls += 1
        assert messages == build_stage1_v2_messages()
        return (
            '{"target_facts":[3,7,5,2],"redundant_facts":[],'
            '"axis_facts":["integer_ticks"]}'
        )

    report, records = run_stage1_v2_probe(scenes, generate=generate)

    assert calls == 24
    assert report.scenes == 24
    assert report.model_calls == 24
    assert report.parse_rate == 1.0
    assert report.exact_transcription_rate == 1.0
    assert report.probe_passed is True
    assert report.format_retries == 0
    assert report.training_invoked is False
    assert len(records) == 24
    assert all(record.parse_success for record in records)


def test_stage1_v2_probe_keeps_fences_and_partial_arrays_as_failures() -> None:
    scenes = (
        Stage1V2Scene(
            scene_id="dev-000",
            image_path=Path("/dataset/images/dev-000.png"),
            chart_type="grouped_bar",
            operation="sum",
            values=(3, 7, 5, 2),
        ),
        Stage1V2Scene(
            scene_id="dev-001",
            image_path=Path("/dataset/images/dev-001.png"),
            chart_type="line",
            operation="difference",
            values=(3, 7, 5, 2),
        ),
    )
    outputs = iter(
        (
            "```json\n{\"target_facts\":[3,7,5,2],\"redundant_facts\":[],"
            '\"axis_facts\":[\"integer_ticks\"]}\n```',
            '{"target_facts":[3,7],"redundant_facts":[],'
            '"axis_facts":["integer_ticks"]}',
        )
    )

    report, records = run_stage1_v2_probe(
        scenes,
        generate=lambda _scene, _messages: next(outputs),
    )

    assert report.model_calls == 2
    assert report.parse_rate == 0.0
    assert report.probe_passed is False
    assert dict(report.error_counts) == {
        "not_exact_json_object": 1,
        "target_facts_not_four_integers": 1,
    }
    assert all(record.parse_success is False for record in records)


def test_stage1_v2_scene_rejects_invalid_or_mutable_inputs() -> None:
    with pytest.raises(TypeError, match="exact integers"):
        Stage1V2Scene(
            scene_id="dev-000",
            image_path=Path("/dataset/images/dev-000.png"),
            chart_type="line",
            operation="sum",
            values=(True, 7, 5, 2),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="chart_type"):
        Stage1V2Scene(
            scene_id="dev-000",
            image_path=Path("/dataset/images/dev-000.png"),
            chart_type="pie",
            operation="sum",
            values=(3, 7, 5, 2),
        )
