"""S1 actual CPU autograd execution, durable observation and resume checks."""

import json
from collections import Counter
from pathlib import Path

import pytest

from src.core import canonical_hash, file_hash


def test_followup_runtime_api_exists():
    from src.followup_runtime import build_direct_requests, collect_samples, run_followup_bank

    assert (
        callable(build_direct_requests)
        and callable(collect_samples)
        and callable(run_followup_bank)
    )


def fixture_context():
    from followup_test_adapter import fake_prompts, make_fake_adapter, warm_origin

    from src.optimizer_fork import state_hash

    adapter = make_fake_adapter()
    optimizer, origin = warm_origin(adapter)
    design = json.loads(
        Path("docs/mechanism_followup/design/configs/followup_design.json").read_text()
    )
    compositions = {
        0: ["XXXXXXXX", "XXXXXXXX", "XXXXWWWW", "XXXXXXXW"],
        3: ["XXXXXXXX", "XXXXXXXX", "WWWWIIII", "XXXXXXXX"],
        4: ["WWWWWWWI", "XXXXXXXX", "XXXXXXXX", "XXXXXXXX"],
        5: ["WWWWWWWW", "XXXXXXXX", "WWWWWWWI", "XXXXXXXX"],
        11: ["WWWWWWWW", "XXXXXXXX", "XXXXWIII", "XXXXXXXX"],
    }
    banks = {}
    for bank, patterns in compositions.items():
        groups = []
        prompts = fake_prompts(4, split=f"train{bank}")
        for prompt, pattern in zip(prompts, patterns, strict=True):
            prepared = adapter.prepare(prompt["prompt"], ".")
            group = []
            for i, category in enumerate(pattern):
                token = {"X": 0, "S": 1, "W": 2, "I": 3}[category]
                tokens = [token, 4]
                group.append(
                    {
                        **prompt,
                        "sample_key": f"{bank}-{prompt['prompt_id']}-{i}",
                        "sample_index": i,
                        "category": category,
                        "token_ids": tokens,
                        "old_logprobs": adapter.logprobs(prepared, tokens).tolist(),
                        "prepared": prepared,
                    }
                )
            groups.append(group)
        banks[bank] = groups
    units = []
    for spec in design["S1"]["units"]:
        groups = (
            banks[spec["parent_bank_index"]]
            if "parent_bank_index" in spec
            else [banks[3][2], banks[11][2], banks[0][0], banks[0][1]]
        )
        units.append(
            {
                "bank_id": spec["id"],
                "groups": groups,
                "bank_hash": state_hash(groups),
                "source_groups": spec.get("source_groups_zero_based", []),
            }
        )
    plan = {
        "design": design,
        "units": units,
        "control_prompts": fake_prompts(4),
        "identity": {
            "model_hash": "fixture-model",
            "data_hash": "fixture-data",
            "config_hash": canonical_hash(design),
            "parser_version_hash": "fixture-parser",
            "protocol_version": design["protocol_version"],
        },
        "execution_kind": "CPU_FAKE_TORCH",
        "fixture_limits": {"samples_per_prompt": 2, "bootstrap_replicates": 10},
    }
    return plan, adapter, optimizer, origin


def test_all_six_banks_actual_adam_checkpoint_raw_counts_response_and_resume(tmp_path, monkeypatch):
    from src.followup_runtime import run_followup_bank
    from src.followup_updates import capture_state
    from src.optimizer_fork import state_hash

    plan, adapter, optimizer, origin = fixture_context()
    steps = []
    original_step = optimizer.step

    def measured_step(*args, **kwargs):
        steps.append(1)
        return original_step(*args, **kwargs)

    monkeypatch.setattr(optimizer, "step", measured_step)
    result = run_followup_bank(
        plan,
        output_root=tmp_path,
        adapter=adapter,
        optimizer=optimizer,
        origin=origin,
        data_root=".",
        fixture=True,
    )
    assert result["status"] == "CPU_TESTED"
    assert result["candidate_updates"] == 41 and result["replay_updates"] == 6
    assert result["logical_direct_candidates"] == 18
    assert len(steps) == 47
    observed = [
        json.loads(line)
        for path in tmp_path.glob("*/candidates/*/direct/samples.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert Counter(row["category"] for row in observed)["I"] > 0
    assert all(row["execution_checks"]["passed"] for row in observed)
    assert state_hash(
        capture_state(adapter.model, optimizer, origin["metadata"], adapter=adapter)
    ) == state_hash(origin)
    for spec in plan["design"]["S1"]["units"]:
        root = tmp_path / spec["id"]
        assert (root / "paired_response.json").is_file()
        assert (root / "policy_aliases.json").is_file()
        for candidate in spec["candidates"]:
            assert (root / "candidates" / candidate["id"] / "checkpoint.pt").is_file()
    hashes = {str(p): file_hash(p) for p in tmp_path.rglob("*") if p.is_file()}
    calls = adapter.generation_calls
    assert (
        run_followup_bank(
            plan,
            output_root=tmp_path,
            adapter=adapter,
            optimizer=optimizer,
            origin=origin,
            data_root=".",
            fixture=True,
            resume=True,
        )
        == result
    )
    assert calls == adapter.generation_calls
    assert hashes == {str(p): file_hash(p) for p in tmp_path.rglob("*") if p.is_file()}
    assert len(steps) == 47


def test_generation_interruption_reuses_prefix_and_identity_changes_fail(tmp_path):
    import copy

    from src.followup_runtime import run_followup_bank

    plan, adapter, optimizer, origin = fixture_context()
    plan["units"] = plan["units"][:1]
    adapter.fail_after = 3
    kwargs = dict(
        output_root=tmp_path,
        adapter=adapter,
        optimizer=optimizer,
        origin=origin,
        data_root=".",
        fixture=True,
    )
    with pytest.raises(RuntimeError, match="injected generation"):
        run_followup_bank(plan, **kwargs)
    path = next(tmp_path.rglob("samples.jsonl"))
    prefix = path.read_bytes()
    assert len(prefix.splitlines()) == 3
    failures = {str(p): file_hash(p) for p in tmp_path.rglob("failure.json")}
    assert failures
    adapter.fail_after = None
    assert run_followup_bank(plan, **kwargs, resume=True)["status"] == "CPU_TESTED"
    assert path.read_bytes().startswith(prefix)
    assert all(file_hash(p) == h for p, h in failures.items())
    for field in (
        "model_hash",
        "data_hash",
        "config_hash",
        "parser_version_hash",
        "protocol_version",
    ):
        changed = copy.deepcopy(plan)
        changed["identity"][field] = "changed-identity"
        with pytest.raises(ValueError, match="identity"):
            run_followup_bank(changed, **kwargs, resume=True)
    changed = copy.deepcopy(plan)
    changed["design"]["S1"]["direct_evaluation"]["seed_root"] += 1
    with pytest.raises(ValueError, match="identity"):
        run_followup_bank(changed, **kwargs, resume=True)
    different_origin = copy.deepcopy(origin)
    different_origin["metadata"]["train_seed"] = 29
    with pytest.raises(ValueError, match="identity"):
        run_followup_bank(plan, **{**kwargs, "origin": different_origin}, resume=True)


def test_semantic_invalid_is_observation_bad_probability_is_fault(tmp_path):
    from src.followup_runtime import run_followup_bank

    plan, adapter, optimizer, origin = fixture_context()
    plan["units"] = plan["units"][:1]
    adapter.bad_scores = True
    kwargs = dict(
        output_root=tmp_path,
        adapter=adapter,
        optimizer=optimizer,
        origin=origin,
        data_root=".",
        fixture=True,
    )
    with pytest.raises(RuntimeError, match="execution fault"):
        run_followup_bank(plan, **kwargs)
    row = json.loads(next(tmp_path.rglob("samples.jsonl")).read_text())
    assert not row["execution_checks"]["passed"]
    assert row["behavior_token_logprobs"][0] == "nan"
    adapter.bad_scores = False
    with pytest.raises(ValueError, match="failed execution"):
        run_followup_bank(plan, **kwargs, resume=True)


def test_candidate_rng_excludes_candidate_and_path_escape_rejected(tmp_path):
    from followup_test_adapter import fake_prompts

    from src.followup_runtime import build_direct_requests, run_followup_bank

    plan, adapter, optimizer, origin = fixture_context()
    identity = {**plan["identity"], "origin_state_hash": "o", "execution_kind": "CPU_FAKE_TORCH"}
    a = build_direct_requests(
        plan["control_prompts"],
        identity,
        "bank00",
        {"candidate_id": "a", "candidate_policy_fingerprint": "x"},
        samples_per_prompt=2,
    )
    b = build_direct_requests(
        plan["control_prompts"],
        identity,
        "bank00",
        {"candidate_id": "b", "candidate_policy_fingerprint": "y"},
        samples_per_prompt=2,
    )
    assert [r["sample_seed"] for r in a] == [r["sample_seed"] for r in b]
    assert a[0]["sample_key"] != b[0]["sample_key"]
    full_requests = build_direct_requests(
        fake_prompts(48),
        identity,
        "bank00",
        {"candidate_id": "a", "candidate_policy_fingerprint": "x"},
    )
    assert len(full_requests) == 48 * 16 == 768
    assert len(full_requests) * 18 == 13824
    plan["units"][0]["bank_id"] = "../escape"
    with pytest.raises(ValueError, match=r"bank|path"):
        run_followup_bank(
            plan,
            output_root=tmp_path,
            adapter=adapter,
            optimizer=optimizer,
            origin=origin,
            data_root=".",
            fixture=True,
        )


def test_single_bank_cli_layout_and_completed_tamper(tmp_path):
    from src.followup_runtime import run_followup_bank

    plan, adapter, optimizer, origin = fixture_context()
    plan["units"] = plan["units"][:1]
    root = tmp_path / "bank00"
    kwargs = dict(
        output_root=root,
        adapter=adapter,
        optimizer=optimizer,
        origin=origin,
        data_root=".",
        fixture=True,
    )
    result = run_followup_bank(plan, **kwargs)
    assert result["candidate_updates"] == 2
    assert (root / "candidates" / "joint_0" / "checkpoint.pt").is_file()
    assert not (root / "bank00").exists()
    assert json.loads((root / "identity.json").read_text())["design_hash"] == canonical_hash(
        plan["design"]
    )
    (root / "paired_response.json").write_text("[]\n")
    with pytest.raises(ValueError, match=r"manifest.*mismatch"):
        run_followup_bank(plan, **kwargs, resume=True)


def test_forward_alias_requires_exact_all_forward_fields_not_optimizer(tmp_path):
    import copy

    from src.followup_runtime import _fingerprint, run_followup_bank

    plan, adapter, optimizer, origin = fixture_context()
    changed = copy.deepcopy(origin)
    changed["optimizer"]["state"][0]["exp_avg"].add_(1)
    assert _fingerprint(origin, plan["identity"], adapter, plan) == _fingerprint(
        changed, plan["identity"], adapter, plan
    )
    changed["buffers"]["scale"].add_(1e-7)
    assert _fingerprint(origin, plan["identity"], adapter, plan) != _fingerprint(
        changed, plan["identity"], adapter, plan
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "link"
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="symbolic link"):
        run_followup_bank(
            plan,
            output_root=link,
            adapter=adapter,
            optimizer=optimizer,
            origin=origin,
            data_root=".",
            fixture=True,
        )


def test_interruption_at_final_completion_marker_reuses_sealed_measurements(tmp_path, monkeypatch):
    from src import followup_runtime

    plan, adapter, optimizer, origin = fixture_context()
    plan["units"] = plan["units"][:1]
    write = followup_runtime.write_json
    failed = []

    def interrupt_marker(path, value):
        if Path(path) == tmp_path / "completed.json" and not failed:
            failed.append(True)
            raise RuntimeError("interrupted final marker")
        return write(path, value)

    monkeypatch.setattr(followup_runtime, "write_json", interrupt_marker)
    kwargs = dict(
        output_root=tmp_path,
        adapter=adapter,
        optimizer=optimizer,
        origin=origin,
        data_root=".",
        fixture=True,
    )
    with pytest.raises(RuntimeError, match="final marker"):
        followup_runtime.run_followup_bank(plan, **kwargs)
    profile_hash = file_hash(tmp_path / "runtime_profile.json")
    calls = adapter.generation_calls
    result = followup_runtime.run_followup_bank(plan, **kwargs, resume=True)
    assert result["status"] == "CPU_TESTED" and calls == adapter.generation_calls
    assert file_hash(tmp_path / "runtime_profile.json") == profile_hash


def test_loader_timing_does_not_change_exact_forward_fingerprint():
    from src.followup_runtime import _fingerprint

    plan, adapter, _, origin = fixture_context()
    before = _fingerprint(origin, plan["identity"], adapter, plan)
    adapter.audit.update(load_seconds=12.5, load_peak_cuda_bytes=2048)
    assert _fingerprint(origin, plan["identity"], adapter, plan) == before
    adapter.audit["processor_hash"] = "different processor"
    assert _fingerprint(origin, plan["identity"], adapter, plan) != before


def test_interrupted_adam_attempt_cannot_silently_expand_physical_budget(tmp_path, monkeypatch):
    from src import followup_updates
    from src.followup_runtime import run_followup_bank

    plan, adapter, optimizer, origin = fixture_context()
    plan["units"] = plan["units"][:1]
    engine = followup_updates.fork_one_candidate
    calls = []

    def interrupt_after_real_step(*args, **kwargs):
        result = engine(*args, **kwargs)
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("interrupted after real Adam before checkpoint")
        return result

    monkeypatch.setattr(followup_updates, "fork_one_candidate", interrupt_after_real_step)
    kwargs = dict(
        output_root=tmp_path,
        adapter=adapter,
        optimizer=optimizer,
        origin=origin,
        data_root=".",
        fixture=True,
    )
    with pytest.raises(RuntimeError, match="after real Adam"):
        run_followup_bank(plan, **kwargs)
    with pytest.raises(RuntimeError, match="budget"):
        run_followup_bank(plan, **kwargs, resume=True)
    assert len(calls) == 3  # bank00: two candidates plus one replay physical ceiling.
    assert not (tmp_path / "completed.json").exists()
    with pytest.raises(RuntimeError, match="budget"):
        run_followup_bank(plan, **kwargs, resume=True)
    assert len(calls) == 3
