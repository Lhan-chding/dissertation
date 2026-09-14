"""S2 driver: immutable requests, exact resume, and isolated fixed evaluations."""

import copy
import json

import pytest


def _design():
    from pathlib import Path

    return json.loads(
        (
            Path(__file__).parents[1]
            / "docs/mechanism_followup/design/configs/followup_design.json"
        ).read_text()
    )


def _panel_plan():
    def prompts(n, track):
        return [
            {
                "prompt_id": f"{track}{i}",
                "prompt_hash": f"h{track}{i}",
                "base_scene_id": f"s{track}{i}",
                "family": "trend",
                "interface": "SYMBOLIC_FRESH",
                "track": track,
                "split": "dev",
                "max_new_tokens": 48 if track == "L" else 64,
            }
            for i in range(n)
        ]

    full = prompts(288, "N")
    return {
        "design": _design(),
        "dev_panel_prompts": full[:72],
        "dev_prompts": full,
        "legacy_prompts": prompts(176, "L"),
        "ood_prompts": prompts(72, "OOD"),
        "diagnostic_prompts": prompts(216, "R2"),
    }


def test_prescribed_fixed_evaluation_budget_and_endpoint_alias():
    from src.followup_train import evaluation_plan

    plan = _panel_plan()
    panels = {step: evaluation_plan(plan, step) for step in (0, 16, 32, 48, 64)}
    assert (
        sum(
            len(p["prompts"]) * p["samples_per_prompt"]
            for values in panels.values()
            for p in values
            if not p.get("alias_of")
        )
        == 8896
    )
    assert {p["track"] for p in panels[0]} == {"N_PANEL", "OOD", "R2"}
    assert {p["track"] for p in panels[64]} == {"N", "N_PANEL", "L", "OOD", "R2"}
    assert next(p for p in panels[64] if p["track"] == "N_PANEL")["alias_of"] == "N"
    assert evaluation_plan(plan, 3) == []


def test_request_stream_is_semantic_directory_independent_and_arm_owned():
    from src.followup_train import build_training_requests

    prompt = _panel_plan()["dev_prompts"][0]
    identity = {"model_hash": "m", "parser_hash": "p", "config_hash": "c", "data_hash": "d"}
    args = dict(
        prompts=[prompt],
        identity=identity,
        seed=29,
        arm="X_BASE",
        step=3,
        role="train",
        policy_hash="state",
        samples_per_prompt=8,
    )
    first = build_training_requests(**args)
    assert first == build_training_requests(**copy.deepcopy(args))
    other = build_training_requests(**{**args, "arm": "X_VALID"})
    assert {r["sample_key"] for r in first}.isdisjoint(r["sample_key"] for r in other)
    assert [r["sample_seed"] for r in first] == [r["sample_seed"] for r in other]
    assert len(set(r["sample_key"] for r in first)) == 8
    assert all(r["train_seed"] == 29 and r["arm"] == "X_BASE" for r in first)
    assert first != build_training_requests(**{**args, "seed": 41})


def test_training_rejects_historical_anchor_and_unapproved_real_execution(tmp_path):
    from src.followup_train import run_training_arm

    plan = {"design": _design(), "execution_kind": "REAL_CUDA_FOLLOWUP"}
    with pytest.raises(ValueError, match="historical"):
        run_training_arm(plan, seed=17, arm="X_BASE", output_root=tmp_path)
    with pytest.raises(ValueError, match=r"gate|authoriz"):
        run_training_arm(plan, seed=29, arm="X_BASE", output_root=tmp_path)


def _fixture_plan(seed=29, steps=2):
    from followup_test_adapter import fake_prompts

    plan = {
        "design": _design(),
        "execution_kind": "CPU_FAKE_TORCH",
        "cpu_fixture": True,
        "identity": {
            "model_hash": "fake",
            "parser_hash": "fixture",
            "config_hash": "locked",
            "data_hash": "fixture",
        },
        "sampler_seed": seed,
        "execution_identity": {"source_hash": "fixture-v1"},
    }
    spec = plan["design"]["S2"]
    spec["steps"] = steps
    spec["checkpoint_steps"] = list(range(steps + 1))
    spec["dev_panel"]["steps"] = list(range(steps + 1))
    spec["dev_panel"]["samples_per_prompt"] = 2
    spec["final_eval"]["samples_per_prompt"] = 2
    spec["interface_diagnostic"]["samples_per_condition"] = 2
    spec["interface_diagnostic"]["steps"] = [0, steps]

    def prompts(n, split, track="N"):
        values = fake_prompts(n, split=split)
        for p in values:
            p.update(track=track, max_new_tokens=48 if track == "L" else 64)
            if track == "L":
                p["scene"].update(
                    truth=[2, 3, 4, 5],
                    observation=[9, 3, 4, 5],
                    error_index=0,
                    facts=[],
                    scene_id=p["base_scene_id"],
                    operation={"operator": "sum", "indices": [0, 1]},
                )
        return values

    plan["train_prompts"] = prompts(4 * steps, "train")
    plan["train_steps"] = [
        [p["prompt_id"] for p in plan["train_prompts"][i : i + 4]] for i in range(0, steps * 4, 4)
    ]
    plan["dev_prompts"] = prompts(2, "dev")
    plan["dev_panel_prompts"] = plan["dev_prompts"]
    plan["legacy_prompts"] = prompts(2, "legacy", "L")
    plan["ood_prompts"] = prompts(2, "ood", "OOD")
    plan["diagnostic_prompts"] = prompts(2, "diagnostic", "R2")
    for prompt, condition in zip(
        plan["diagnostic_prompts"], ("SYM_ORIGINAL", "IMAGE_CUE"), strict=True
    ):
        prompt["diagnostic_condition"] = condition
    return plan


def _model(seed):
    import random

    import numpy as np
    import torch
    from followup_test_adapter import make_fake_adapter

    random.seed(seed)
    np.random.seed(seed)
    adapter = make_fake_adapter(seed)
    optimizer = torch.optim.AdamW(
        [p for p in adapter.model.parameters() if p.requires_grad],
        lr=1e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    )
    return adapter, optimizer


def test_evaluation_restores_complete_state_even_after_exception():
    import random

    import torch

    from src.followup_train import isolated_evaluation
    from src.followup_updates import capture_state
    from src.optimizer_fork import state_hash

    adapter, optimizer = _model(29)
    sampler = {"position": 7}
    before = capture_state(adapter.model, optimizer, adapter=adapter, sampler=sampler)
    with (
        pytest.raises(RuntimeError, match="injected"),
        isolated_evaluation(adapter, optimizer, sampler=sampler),
    ):
        random.random()
        torch.rand(3)
        adapter.model.child.train()
        adapter.model.scale.add_(1)
        sampler["position"] = 99
        raise RuntimeError("injected evaluator failure")
    after = capture_state(adapter.model, optimizer, adapter=adapter, sampler=sampler)
    assert state_hash(after) == state_hash(before)


def test_two_seeds_three_arms_real_adam_and_ledger_chain(tmp_path):
    from src.followup_train import run_training_arm

    results, keys = {}, {}
    for seed in (29, 41):
        for arm in ("X_BASE", "X_VALID", "X_VALID_NO_X_OFF"):
            adapter, optimizer = _model(seed)
            out = tmp_path / f"{seed}" / arm
            result = run_training_arm(
                _fixture_plan(seed),
                seed=seed,
                arm=arm,
                output_root=out,
                adapter=adapter,
                optimizer=optimizer,
            )
            assert result["status"] == "PASS"
            assert result["completed_steps"] == result["cpu_optimizer_updates"] == 2
            assert result["training_outputs"] == 64
            assert result["evaluation_outputs"] == 32
            manifest = json.loads((out / "checkpoint_manifest.json").read_text())
            assert [c["step"] for c in manifest["checkpoints"]] == [0, 1, 2]
            summaries = json.loads((out / "training_steps.json").read_text())
            assert all(s["update"]["optimizer_updates"] == 1 for s in summaries)
            assert all(s["loss"]["backward_calls"] == 32 for s in summaries)
            assert summaries[0]["parameter_hash"] != summaries[1]["parameter_hash"]
            rows = [
                json.loads(line)
                for path in out.glob("step_*/rollouts/samples.jsonl")
                for line in path.read_text().splitlines()
            ]
            assert len(rows) == 64 and any(r["category"] == "I" for r in rows)
            assert len({r["sample_key"] for r in rows}) == 64
            assert {r["train_seed"] for r in rows} == {seed}
            assert rows[0]["policy_state_hash"] != rows[32]["policy_state_hash"]
            keys[seed, arm] = {r["sample_key"] for r in rows}
            results[seed, arm] = result
        assert keys[seed, "X_BASE"].isdisjoint(keys[seed, "X_VALID"])
        assert keys[seed, "X_VALID"].isdisjoint(keys[seed, "X_VALID_NO_X_OFF"])
    assert len(results) == 6


def test_interrupted_training_resume_is_bitwise_uninterrupted_and_no_clobber(tmp_path):
    from src.followup_train import run_training_arm

    plan = _fixture_plan()
    adapter, optimizer = _model(29)
    complete = run_training_arm(
        plan,
        seed=29,
        arm="X_VALID",
        output_root=tmp_path / "full",
        adapter=adapter,
        optimizer=optimizer,
    )
    adapter, optimizer = _model(29)
    adapter.fail_after = 49
    with pytest.raises(RuntimeError, match="interruption"):
        run_training_arm(
            plan,
            seed=29,
            arm="X_VALID",
            output_root=tmp_path / "resume",
            adapter=adapter,
            optimizer=optimizer,
        )
    partial = tmp_path / "resume/step_02/rollouts/samples.jsonl"
    prefix = partial.read_bytes()
    adapter, optimizer = _model(29)
    resumed = run_training_arm(
        plan,
        seed=29,
        arm="X_VALID",
        output_root=tmp_path / "resume",
        adapter=adapter,
        optimizer=optimizer,
        resume=True,
    )
    assert complete == resumed
    assert partial.read_bytes().startswith(prefix)
    count = adapter.generation_calls
    again = run_training_arm(
        plan,
        seed=29,
        arm="X_VALID",
        output_root=tmp_path / "resume",
        adapter=adapter,
        optimizer=optimizer,
        resume=True,
    )
    assert again == resumed and adapter.generation_calls == count
    with pytest.raises(FileExistsError):
        run_training_arm(
            plan,
            seed=29,
            arm="X_VALID",
            output_root=tmp_path / "resume",
            adapter=adapter,
            optimizer=optimizer,
        )
    changed = copy.deepcopy(plan)
    changed["identity"]["parser_hash"] = "changed"
    with pytest.raises(ValueError, match=r"evidence|identity"):
        run_training_arm(
            changed,
            seed=29,
            arm="X_VALID",
            output_root=tmp_path / "resume",
            adapter=adapter,
            optimizer=optimizer,
            resume=True,
        )


def test_resume_rejects_changed_initial_adapter_and_completed_counts(tmp_path):
    import torch

    from src.followup_train import run_training_arm

    plan = _fixture_plan(steps=1)
    adapter, optimizer = _model(29)
    run_training_arm(
        plan, seed=29, arm="X_BASE", output_root=tmp_path, adapter=adapter, optimizer=optimizer
    )
    adapter, optimizer = _model(29)
    with torch.no_grad():
        adapter.model.logits[0] += 0.1
    with pytest.raises(ValueError, match=r"evidence|initial"):
        run_training_arm(
            plan,
            seed=29,
            arm="X_BASE",
            output_root=tmp_path,
            adapter=adapter,
            optimizer=optimizer,
            resume=True,
        )
    adapter, optimizer = _model(29)
    summary = tmp_path / "evaluation_summary.json"
    summary.write_text(summary.read_text() + "\n")
    with pytest.raises(ValueError, match="manifest"):
        run_training_arm(
            plan,
            seed=29,
            arm="X_BASE",
            output_root=tmp_path,
            adapter=adapter,
            optimizer=optimizer,
            resume=True,
        )
    assert adapter.generation_calls == 0


def test_cold_initial_and_source_resume_identity_are_enforced(tmp_path):
    import torch

    from src.followup_train import run_training_arm

    plan = _fixture_plan(steps=1)
    adapter, optimizer = _model(29)
    for parameter in adapter.model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    with pytest.raises(ValueError, match="cold initial"):
        run_training_arm(
            plan,
            seed=29,
            arm="X_BASE",
            output_root=tmp_path / "warm",
            adapter=adapter,
            optimizer=optimizer,
        )
    adapter, optimizer = _model(29)
    run_training_arm(
        plan,
        seed=29,
        arm="X_BASE",
        output_root=tmp_path / "cold",
        adapter=adapter,
        optimizer=optimizer,
    )
    plan["execution_identity"]["source_hash"] = "changed"
    with pytest.raises(ValueError, match="execution_identity"):
        run_training_arm(
            plan,
            seed=29,
            arm="X_BASE",
            output_root=tmp_path / "cold",
            adapter=adapter,
            optimizer=optimizer,
            resume=True,
        )


def test_fixture_label_cannot_bypass_real_adapter_training_gates(tmp_path):
    from src.followup_train import run_training_arm

    adapter, optimizer = _model(29)
    adapter.execution_kind = "REAL_CUDA_FOLLOWUP"
    with pytest.raises(ValueError, match=r"fixture.*adapter"):
        run_training_arm(
            _fixture_plan(steps=1),
            seed=29,
            arm="X_BASE",
            output_root=tmp_path,
            adapter=adapter,
            optimizer=optimizer,
        )
    assert adapter.generation_calls == 0


@pytest.mark.parametrize("resume", [False, True])
def test_unbound_output_and_missing_resume_identity_are_rejected(tmp_path, resume):
    from src.followup_train import run_training_arm

    adapter, optimizer = _model(29)
    if not resume:
        (tmp_path / "progress.json").write_text('{"existing": true}\n')
    with pytest.raises(ValueError, match=r"unbound|resume.*identity"):
        run_training_arm(
            _fixture_plan(steps=1),
            seed=29,
            arm="X_BASE",
            output_root=tmp_path,
            adapter=adapter,
            optimizer=optimizer,
            resume=resume,
        )
    assert adapter.generation_calls == 0


def test_evaluation_preserves_primary_error_if_restore_also_fails():
    import torch

    from src.followup_train import isolated_evaluation

    adapter, optimizer = _model(29)
    with (
        pytest.raises(RuntimeError, match="primary execution failure"),
        isolated_evaluation(adapter, optimizer),
    ):
        with torch.no_grad():
            adapter.model.frozen.add_(1)
        raise RuntimeError("primary execution failure")
    assert adapter._followup_unusable


def test_r2_counts_keep_three_conditions_when_image_interfaces_are_shared():
    from src.followup_train import _counts

    rows = [
        {
            "prompt_id": f"{family}/{condition}",
            "family": family,
            "interface": "SYMBOLIC_FRESH" if condition == "SYM_ORIGINAL" else "IMAGE_CUE_FRESH",
            "condition": condition,
            "track": "R2",
            "category": category,
        }
        for family in ("trend", "cross_series")
        for condition, category in (("SYM_ORIGINAL", "S"), ("IMAGE_CUE", "I"), ("IMAGE_ONLY", "X"))
    ]
    result = _counts(rows)
    assert result["overall"]["n"] == 6
    assert result["groups"]["trend/IMAGE_CUE_FRESH"]["n"] == 2
    assert result["groups"]["trend/IMAGE_CUE_FRESH"]["pX"] == 0.5
    assert result["conditions"]["IMAGE_ONLY"]["counts"] == {"X": 2, "S": 0, "W": 0, "I": 0}
    assert result["conditions"]["IMAGE_ONLY"]["pX"] == 1.0
    assert result["conditions"]["IMAGE_CUE"]["v"] == 0.0
    assert result["conditions"]["IMAGE_CUE"]["qX"] is None
    assert result["conditions"]["IMAGE_CUE"]["qS"] is None
    assert result["conditions"]["IMAGE_CUE"]["qS_support"] == "UNDEFINED_ZERO_VALID"
    assert result["conditions"]["SYM_ORIGINAL"]["v"] == 1.0
    assert result["conditions"]["SYM_ORIGINAL"]["pX"] == 0.0
    assert result["conditions"]["SYM_ORIGINAL"]["qS"] == 1.0
    assert result["conditions"]["SYM_ORIGINAL"]["qS_support"] == "DEFINED"
    assert result["conditions"]["IMAGE_ONLY"]["qS"] == 0.0
    for family in ("trend", "cross_series"):
        assert result["condition_groups"][f"{family}/IMAGE_ONLY"]["pX"] == 1.0
        assert result["condition_groups"][f"{family}/IMAGE_CUE"]["counts"]["I"] == 1
        assert result["condition_groups"][f"{family}/SYM_ORIGINAL"]["counts"]["S"] == 1
    ordinary = [{k: v for k, v in row.items() if k != "condition"} | {"track": "N"} for row in rows]
    assert set(_counts(ordinary)) == {"overall", "groups"}


def test_r2_condition_counts_are_published_without_extra_generation(tmp_path):
    from src.core import canonical_hash
    from src.followup_train import run_training_arm

    plan = _fixture_plan(steps=1)
    symbolic, image_cue = plan["diagnostic_prompts"]
    image_only = copy.deepcopy(image_cue)
    image_only["prompt_id"] += "-image-only"
    image_only["prompt"]["user"] += " image only"
    image_only["prompt_hash"] = canonical_hash(image_only["prompt"]["user"])
    image_only["prompt"]["prompt_hash"] = image_only["prompt_hash"]
    plan["diagnostic_prompts"] = [symbolic, image_cue, image_only]
    forced = {}
    for prompt, condition, token in zip(
        plan["diagnostic_prompts"],
        ("SYM_ORIGINAL", "IMAGE_CUE", "IMAGE_ONLY"),
        (2, 3, 0),
        strict=True,
    ):
        prompt["diagnostic_condition"] = condition
        forced[prompt["prompt_hash"]] = token
    adapter, optimizer = _model(29)
    original_generate = adapter.generate

    def generate(prepared, **kwargs):
        result = original_generate(prepared, **kwargs)
        key = prepared["audit"]["final_prompt_hash"]
        if key in forced:
            tokens = [forced[key], 4]
            result.update(
                token_ids=tokens,
                raw_completion=adapter.decode(tokens),
                behavior_token_logprobs=adapter.logprobs(prepared, tokens).tolist(),
            )
        return result

    adapter.generate = generate
    result = run_training_arm(
        plan, seed=29, arm="X_BASE", output_root=tmp_path, adapter=adapter, optimizer=optimizer
    )
    assert result["training_outputs"] == 32
    assert result["evaluation_outputs"] == 32
    assert adapter.generation_calls == 64
    reports = json.loads((tmp_path / "evaluation_summary.json").read_text())
    diagnostics = [row for row in reports if row["track"] == "R2"]
    assert len(diagnostics) == 2
    for report in diagnostics:
        stored = json.loads(
            (tmp_path / "evaluation" / f"step_{report['step']:02d}" / "R2/counts.json").read_text()
        )
        assert stored == report
        assert report["conditions"]["IMAGE_CUE"]["counts"]["I"] == 2
        assert report["conditions"]["IMAGE_ONLY"]["counts"]["X"] == 2
        assert report["conditions"]["IMAGE_ONLY"]["pX"] == 1.0
        assert report["conditions"]["IMAGE_CUE"]["qX"] is None
        assert report["conditions"]["IMAGE_CUE"]["qS"] is None
        assert report["condition_groups"]["duplicate_encoding/IMAGE_ONLY"]["pX"] == 1.0
    manifest = json.loads((tmp_path / "completion_manifest.json").read_text())
    paths = {item["path"] for item in manifest["files"]}
    assert "evaluation_summary.json" in paths
    assert "evaluation/step_00/R2/counts.json" in paths
    assert "evaluation/step_01/R2/counts.json" in paths
