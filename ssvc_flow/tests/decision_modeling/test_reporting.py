import json

import pytest

from src.decision_modeling.reporting import REPORTS, report
from src.modeling_v3.vlm_observation import digest


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def config():
    return {
        "panels": {"prompts_per_panel": 6, "scenes_per_panel": 3, "interfaces": ["image", "text"]},
        "observations": {"looks": [32, 128, 512], "reference_looks": [32, 128]},
        "blocks": {"origins": {"O1": {}}, "O1_recipes": ["R0", "R4"]},
        "statistics": {
            "key_groups": [
                [family, interface] for family in ("a", "b", "c") for interface in ("image", "text")
            ],
            "alpha": 0.05,
            "tau_X": 0.01,
            "epsilon_J": 0.01,
            "decision_episode": {"candidates": 2},
            "evaluation_preferences": {
                "w0": [1, 0, 0],
                "w1": [0.8, 0.1, 0.1],
                "w2": [0.6, 0.2, 0.2],
            },
        },
    }


def _stream(
    root,
    *,
    candidate="R0",
    event="X",
    role="endpoint",
    missing_prompt=False,
    execution="REAL_CUDA_MODEL",
    origin="O1",
    horizon=8,
):
    stream = f"{candidate}:{role}:{origin}:{horizon}"
    identity = {
        "stream_id": stream,
        "role": role,
        "policies": [{"candidate_id": candidate}],
        "prompts": "panelhash",
        "cache_namespace": "cache",
    }
    _save(root / "samples" / "IDENTITY.json", identity)
    count = 0
    for family in ("a", "b", "c"):
        for interface in ("image", "text"):
            prompt = f"{family}-{interface}"
            if missing_prompt and prompt == "c-text":
                continue
            rows = [
                {
                    "prompt_id": prompt,
                    "base_scene_id": family,
                    "family": family,
                    "interface": interface,
                    "prompt_record_hash": digest(prompt),
                    "inference_fingerprint": candidate,
                    "sample_id": digest([stream, prompt, i]),
                    "draw_index": i,
                    "role": role,
                    "rng_stream_id": stream,
                    "scoring_usable": True,
                    "event": event,
                    "answer_correct": int(event in ("X", "S")),
                    "valid": int(event != "I"),
                    "single_edit": int(event == "X"),
                    "relation_numerator": int(event != "I"),
                    "relation_denominator": 1,
                    "relation_score": int(event != "I"),
                }
                for i in range(32)
            ]
            _save(
                root / "samples" / prompt / "0000_0032.json",
                {"identity": digest(identity), "rows": rows, "rows_hash": digest(rows)},
            )
            count += len(rows)
    _save(
        root / "LOOK_32.json",
        {
            "origin_id": origin,
            "horizon": horizon,
            "candidate_id": candidate,
            "panel": "P",
            "panel_identity": "panelhash",
            "role": role,
            "look": 32,
            "total_rows": count,
            "execution_kind": execution,
        },
    )


def test_empty_report_is_honestly_not_run(tmp_path, config):
    root = tmp_path / "source"
    root.mkdir()
    out = tmp_path / "report"
    result = report(root, out, config)
    assert result["new_model_calls"] == result["new_training_steps"] == 0
    assert result["decision_episodes"] == []
    assert all((out / name).exists() for name in REPORTS)
    assert "NOT_RUN" in (out / "MULTIREWARD_BLOCK_RESULTS_zh.md").read_text()
    with pytest.raises(FileExistsError):
        report(root, out, config)


def test_completed_endpoints_analyze_harm_and_same_prefix(tmp_path, config):
    root = tmp_path / "source"
    _stream(root / "R0", event="X")
    _stream(root / "R4", candidate="R4", event="I")
    result = report(root, tmp_path / "report", config)
    assert not result["observation_issues"]
    episode = result["decision_episodes"][0]
    assert episode["status"] == "ANALYZED_FIXED_PANEL"
    look = episode["looks"][0]
    assert look["comparisons"]["R4"]["status"] == "HARM_DETECTED"
    uncertainty = look["comparisons"]["R4"]["descriptive_uncertainty"]["pX"]
    assert uncertainty["scene_sampling"]["replicates"] == 5000
    assert uncertainty["scene_sampling"]["nested_completion_resampling"] is False
    assert uncertainty["fixed_panel_generation_mc"]["empirical_standard_error"] == 0
    assert uncertainty["fixed_panel_generation_mc"]["zero_empirical_se_is_certainty"] is False
    assert look["comparisons"]["R0"]["group_differences"]["a|image"] == [0, 0]
    assert look["selections"]["exact:w0"]["selected"] == "R0"
    assert look["selections"]["exact:w0"]["regret_upper"] == 0
    replay = episode["same_prefix_replay"][0]
    assert replay["total_observations"] == 12 * 32
    assert all(row["candidate_set"] == ["R0"] for row in replay["decisions"].values())
    assert episode["independent_reference"] == [{"status": "NOT_RUN"}]


def test_missing_arm_preserved_unknown_and_reference_not_truth(tmp_path, config):
    config["statistics"]["decision_episode"]["candidates"] = 10
    root = tmp_path / "source"
    _stream(root / "R0")
    _stream(root / "ref", role="reference")
    result = report(root, tmp_path / "report", config)
    episode = result["decision_episodes"][0]
    selection = episode["looks"][0]["selections"]["exact:w0"]
    assert selection["unknown_candidates"] == ["R4"]
    assert selection["regret_upper"] > 0
    targets = episode["looks"][0]["additional_observation_targets"]
    assert any(row["candidate_id"] == "R4" and row["next_look"] == 32 for row in targets)
    assert all(not row["uses_reference_or_test"] for row in targets)
    reference = episode["independent_reference"][0]
    assert reference["status"] == "INDEPENDENT_REFERENCE_MEASURED"
    assert reference["comparison"]["reference_is_exact_truth"] is False
    ref_selection = episode["looks"][0]["independent_reference_selection"]["exact:w0"]
    assert ref_selection["regret_interval"] is None
    assert "R4" in ref_selection["reference_unresolved_candidates"]


def test_partial_fixture_and_corrupted_sources_never_become_real_evidence(tmp_path, config):
    root = tmp_path / "source"
    _stream(root / "partial", missing_prompt=True)
    _save(
        root / "partial" / "samples" / "a-image" / "0000_0032_FAILURE_1.json",
        {"error": "interrupted"},
    )
    _stream(root / "fixture", candidate="R4", execution="CPU_FIXTURE")
    _stream(root / "corrupt", candidate="R4")
    path = root / "corrupt" / "samples" / "a-image" / "0000_0032.json"
    data = json.loads(path.read_text())
    data["rows"][0]["event"] = "I"
    _save(path, data)
    result = report(root, tmp_path / "report", config)
    assert not result["decision_episodes"]
    assert result["observation_issues"][0]["status"] == "INVALID_EVIDENCE"
    blockers = [item["decision_blockers"] for item in result["observation_looks"]]
    assert any("INCOMPLETE_PANEL_PROMPTS" in reasons for reasons in blockers)
    assert any("NOT_REAL_CUDA_MODEL" in reasons for reasons in blockers)


def test_origin_horizon_separation_and_immutable_cost_receipts(tmp_path, config):
    root = tmp_path / "source"
    _stream(root / "R0-h8")
    _stream(root / "R4-h32", candidate="R4", horizon=32)
    _save(
        root / "R0-h8" / "costs" / "invocation_1.json", {"wall_seconds": 2, "generated_tokens": 20}
    )
    _save(
        root / "R0-h8" / "costs" / "invocation_2.json", {"wall_seconds": 3, "generated_tokens": 0}
    )
    result = report(root, tmp_path / "report", config)
    assert len(result["decision_episodes"]) == 2
    assert len(result["costs"]) == 2
    h32 = next(item for item in result["decision_episodes"] if item["horizon"] == 32)
    assert h32["looks"][0]["comparisons"]["R4"]["reason"] == "MATCHED_BASELINE_NOT_MEASURED"


def test_independent_e_panel_is_evaluation_only(tmp_path, config):
    root = tmp_path / "source"
    _stream(root / "R0")
    path = root / "R0" / "LOOK_32.json"
    look = json.loads(path.read_text())
    look["panel"] = "E"
    _save(path, look)
    result = report(root, tmp_path / "report", config)
    episode = result["decision_episodes"][0]
    assert episode["looks"][0]["comparisons"]["R0"]["status"] == "NONINFERIOR_AT_SCALE"
    assert episode["looks"][0]["selections"] == {}
    assert episode["same_prefix_replay"] == []
    assert episode["looks"][0]["additional_observation_targets"] == []
