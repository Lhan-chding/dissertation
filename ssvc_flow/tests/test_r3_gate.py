"""Cold evidence checks are model-free; fixtures never stand in for CUDA."""

import json
from pathlib import Path

import pytest

from src.core import canonical_hash, write_json


def test_gate_rejects_uncompleted_or_non_cuda_before_artifact_use(tmp_path):
    from src.next_stage_runtime import validate_r3_cold_gate

    write_json(tmp_path / "manifest.json", {"files": []})
    for state, kind in [("RUNNING", "REAL_CUDA_FORK"), ("PASS", "CPU_FAKE_ADAPTER_FIXTURE")]:
        write_json(
            tmp_path / "status.json", {"status": state, "phase": "R3-cold", "execution_kind": kind}
        )
        with pytest.raises(ValueError):
            validate_r3_cold_gate(tmp_path, {}, {})


def test_ledger_rejects_duplicates_truncated_tail_and_failed_measurements(tmp_path):
    from src.r3_gate import _ledger

    p = tmp_path / "samples.jsonl"
    row = {
        "sample_key": "a",
        "execution_kind": "REAL_CUDA_FORK",
        "execution_checks": {"passed": True},
    }
    row["record_hash"] = canonical_hash(row)
    p.write_text(json.dumps(row) + "\n")
    assert list(_ledger(p)) == ["a"]
    p.write_text((json.dumps(row) + "\n") * 2)
    with pytest.raises(ValueError, match="duplicate"):
        _ledger(p)
    p.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="truncated"):
        _ledger(p)
    row["execution_checks"]["passed"] = False
    row["record_hash"] = canonical_hash({k: v for k, v in row.items() if k != "record_hash"})
    p.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="failed"):
        _ledger(p)


def test_reuse_permission_requires_both_informative_banks():
    from src.r3_gate import _check_reuse

    validation = {
        str(i): {
            "adoptable": False,
            "validation_optimizer_updates": 4,
            "origin_state_hash": "o",
            "checks": {"0.0": {"passed": False}, "1.0": {"passed": False}},
        }
        for i in (1, 11)
    }
    decision = {
        "reuse_authorized": False,
        "validation_bank_indices": [1, 11],
        "validation": validation,
    }
    assert _check_reuse(decision, "o") is False
    decision["reuse_authorized"] = True
    with pytest.raises(ValueError):
        _check_reuse(decision, "o")
    for result in validation.values():
        result["adoptable"] = True
        for check in result["checks"].values():
            check["passed"] = True
    assert _check_reuse(decision, "o") is True
    del validation["11"]
    with pytest.raises(ValueError):
        _check_reuse(decision, "o")


def test_control_score_cannot_reseal_wrong_proposal_or_missing_output(tmp_path):
    from src.r3_gate import _check_scores

    identity = {"phase": "R3-cold"}
    candidate = {
        "candidate_id": "bank_00_lambda_0",
        "parameter_hash": "p",
        "optimizer_state_hash": "a",
    }
    proposal = {}
    for index in range(768):
        row = {"sample_key": f"p{index}", "record_hash": f"h{index}", "token_ids": [1, 2]}
        proposal[row["sample_key"]] = row
    sid = {
        **identity,
        "unit": "control_likelihood",
        "candidate_id": candidate["candidate_id"],
        "candidate_parameter_hash": "p",
        "candidate_optimizer_hash": "a",
        "proposal_hash": canonical_hash([r["record_hash"] for r in proposal.values()]),
    }
    root = tmp_path / "control_scores" / candidate["candidate_id"]
    root.mkdir(parents=True)
    write_json(root / "identity.json", sid)
    rows = []
    for row in proposal.values():
        value = {
            "sample_key": canonical_hash([sid, row["sample_key"]]),
            "proposal_sample_key": row["sample_key"],
            "candidate_id": candidate["candidate_id"],
            "candidate_parameter_hash": "p",
            "proposal_record_hash": row["record_hash"],
            "token_ids": [1, 2],
            "candidate_token_logprobs": [-1.0, -2.0],
            "execution_kind": "REAL_CUDA_FORK",
            "execution_checks": {"passed": True},
        }
        value["record_hash"] = canonical_hash(value)
        rows.append(value)
    p = root / "samples.jsonl"

    def save():
        p.write_text("".join(json.dumps(row) + "\n" for row in rows))

    save()
    assert _check_scores(tmp_path, identity, candidate, proposal) == 768
    rows[0]["proposal_record_hash"] = "wrong"
    rows[0]["record_hash"] = canonical_hash(
        {k: v for k, v in rows[0].items() if k != "record_hash"}
    )
    save()
    with pytest.raises(ValueError):
        _check_scores(tmp_path, identity, candidate, proposal)
    rows.pop(0)
    save()
    with pytest.raises(ValueError):
        _check_scores(tmp_path, identity, candidate, proposal)


class _Tokenizer:
    def decode(self, tokens, **kwargs):
        return "".join(chr(t - 1) for t in tokens)


@pytest.fixture
def completed_cold(tmp_path, monkeypatch):
    import torch

    from src.core import file_hash
    from src.fork_gradients import _bank_binding
    from src.model_adapters.base import _hash_json
    from src.optimizer_fork import capture_state, parameter_hash, save_checkpoint, state_hash
    from src.r2_runtime import annotate_diagnostic
    from src.r3_runtime import _load_plan, _manifest, _save_tensor_payload, build_requests

    monkeypatch.setattr(
        "src.r3_gate._load_tokenizer", lambda config, audit: _Tokenizer(), raising=False
    )
    # Statistical recomputation has its own real-data tests; keep raw/fork fault
    # cases focused and avoid repeating 5,000 bootstrap draws for every mutation.
    monkeypatch.setattr(
        "src.r3_report_gate.validate_response_artifacts",
        lambda root, *, proposal_records, bank_summaries: {
            "status": "PASS",
            "fixture_only": True,
            "proposal_sequences": len(proposal_records),
            "banks": len(bank_summaries),
        },
    )

    root = tmp_path / "cold"
    root.mkdir()
    data_root = Path("data/generated").resolve()
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0)
    origin = capture_state(model, optimizer, {"checkpoint_step": 0, "state": "cold"})
    initial = parameter_hash(model, trainable=True)
    config = {
        "model": {"id": "Qwen/Qwen3.5-9B", "revision": "fixture"},
        "data_root": str(data_root),
        "protocol_version": "fixture",
        "generation_proposed_N": {"do_sample": True, "max_new_tokens": 64},
    }
    gate = {
        "config": config,
        "binding": {
            "data_manifest_sha256": file_hash(data_root / "manifest.json"),
            "dataset_files": json.loads((data_root / "manifest.json").read_text())["files"],
        },
        "certificate": {
            "initial_adapter_hash": initial,
            "model_audit": {"frozen_parameter_hash": "frozen", "eos_token_ids": [0]},
        },
    }
    r2 = {"status": "PASS", "environment": {"fixture": True}}
    plan, data = _load_plan(data_root, gate)
    source = {"source_commit": "fixture"}
    identity = {
        "phase": "R3-cold",
        "execution_kind": "REAL_CUDA_FORK",
        "checkpoint_step": 0,
        "model_hash": canonical_hash(config["model"]),
        "config_hash": canonical_hash(config),
        "data_hash": canonical_hash(data),
        "initial_adapter_hash": initial,
        "gate_hash": canonical_hash(gate["binding"]),
        "r2_gate_hash": canonical_hash(r2),
        "source_hash": canonical_hash(source),
        "plan_hash": plan["plan_hash"],
    }
    origin_hash = state_hash(origin)
    save_checkpoint(root / "origin.pt", origin, {**identity, "unit": "initial_origin"})
    write_json(root / "identity.json", identity)
    write_json(
        root / "runtime_lock.json",
        {
            "identity": identity,
            "config": config,
            "environment": r2["environment"],
            "source": source,
            "origin_hash": origin_hash,
            "origin_file_sha256": file_hash(root / "origin.pt"),
            "frozen_base_hash": "frozen",
            "initial_adapter_hash": initial,
            "optimizer_initial_hash": state_hash(origin["optimizer"]),
            "selected_probability_path": "uncached_prefix_recompute",
        },
    )
    write_json(root / "gate_binding.json", {"r0_r1": gate["binding"], "r2": r2})
    request_identity = {**identity, "origin_hash": origin_hash}
    requests = build_requests(plan, request_identity)
    write_json(
        root / "bank_manifest.json",
        {
            "identity": identity,
            "request_identity": request_identity,
            "plan": plan,
            "data_binding": data,
            "requests": requests,
        },
    )
    rows = []
    prepared_by_prompt = {}
    prompts = {r["prompt_id"]: r for r in [*plan["train_prompts"], *plan["control_prompts"]]}
    for request in requests:
        prompt = prompts[request["prompt_id"]]
        scene = prompt["scene"]
        if request["prompt_id"] not in prepared_by_prompt:
            text = (
                prompt["prompt"]["system"]
                + "\n"
                + prompt["prompt"]["user"]
                + "\n<think>\n\n</think>\n\n"
            )
            ids = [ord(c) + 1 for c in text]
            inputs = {
                "input_ids": torch.tensor([ids]),
                "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
            }
            image = request["interface"] == "IMAGE_CUE_FRESH"
            audit = {
                "final_prompt": text,
                "final_prompt_token_ids": ids,
                "final_prompt_hash": _hash_json(text),
                "tokenized_prompt_hash": state_hash(inputs["input_ids"]),
                "prompt_token_count": len(ids),
                "image_token_count": int(image),
                "enable_thinking": False,
                "input_tensor_hash": state_hash(inputs),
                "pixel_values_hash": canonical_hash("pixels") if image else None,
            }
            prepared_by_prompt[request["prompt_id"]] = {"inputs": inputs, "audit": audit}
        prepared = prepared_by_prompt[request["prompt_id"]]
        audit = prepared["audit"]
        raw = "bad"
        tokens = [ord(c) + 1 for c in raw] + [0]
        old = [-1.25] * len(tokens)
        row = {
            **request,
            **audit,
            **annotate_diagnostic(raw, scene),
            "run_id": canonical_hash(identity),
            "model_id": config["model"]["id"],
            "model_revision": config["model"]["revision"],
            "adapter_hash": initial,
            "optimizer_state_hash": state_hash(origin["optimizer"]),
            "protocol_version": config["protocol_version"],
            "train_seed": 17,
            "truth_world": scene["truth_world"],
            "observed_world": scene["observed_world"],
            "changed_index": scene["changed_index"],
            "cue": scene["cue"],
            "chart_type": scene["chart_type"],
            "operation": scene["operation"],
            "solution_count": 1,
            "image_hash": scene["image_hash"] if audit["image_token_count"] else None,
            "token_ids": tokens,
            "raw_token_ids": tokens,
            "raw_completion": raw,
            "raw_text": raw,
            "completion_length": len(tokens),
            "n_generated_tokens": len(tokens),
            "stop_reason": "eos",
            "behavior_token_logprobs": old,
            "old_logprobs": old,
            "per_token_logprob_behavior": old,
            "logprob_sequence": sum(old),
            "generation_config_hash": canonical_hash(config["generation_proposed_N"]),
            "input_ids_hash": audit["tokenized_prompt_hash"],
            "actual_image_tokens": audit["image_token_count"],
            "vision_forward_calls": int(bool(audit["image_token_count"])),
            "prepared_hash": state_hash(prepared),
            "execution_kind": "REAL_CUDA_FORK",
            "execution_checks": {"passed": True, "faults": []},
        }
        row["record_hash"] = canonical_hash(row)
        rows.append(row)
    (root / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    proposal = [r for r in rows if r["bank_role"] == "control_proposal"]
    validation = {
        str(i): {
            "adoptable": False,
            "validation_optimizer_updates": 4,
            "origin_state_hash": origin_hash,
            "checks": {"0.0": {"passed": False}, "1.0": {"passed": False}},
        }
        for i in (1, 11)
    }
    write_json(
        root / "reuse_decision.json",
        {"reuse_authorized": False, "validation_bank_indices": [1, 11], "validation": validation},
    )
    model(torch.ones(1, 1)).sum().backward()
    optimizer.step()
    candidate_state = capture_state(model, optimizer, origin["metadata"])
    banks = {}
    for index in range(12):
        groups = [
            [{**r, "prepared": prepared_by_prompt[pid]} for r in rows if r["prompt_id"] == pid]
            for pid in plan["banks"][index]
        ]
        sample_hash = canonical_hash([r["record_hash"] for group in groups for r in group])
        bank_hash = _bank_binding(groups)["bank_hash"]
        if index in (1, 11):
            attempt = root / "validation" / f"bank_{index:02d}" / "attempt_0000"
            attempt.mkdir(parents=True)
            validation_identity = {
                **identity,
                "unit": "reuse_validation",
                "bank_index": index,
                "origin_hash": origin_hash,
                "sample_hash": sample_hash,
            }
            write_json(attempt / "identity.json", validation_identity)
            binding = _save_tensor_payload(
                attempt / "score_bank.pt",
                {"audit": {"bank_hash": bank_hash, "B": 4, "K": 8, "sequences": 32}},
                validation_identity,
            )
            result = {**validation[str(index)], "score_bank_binding": binding}
            write_json(attempt / "result.json", result)
            validation[str(index)] = {
                **result,
                "attempt": str(attempt),
                "reused_completed_unit": False,
            }
        unit_identity = {
            **identity,
            "unit": "bank_forks",
            "bank_index": index,
            "origin_hash": origin_hash,
            "sample_hash": sample_hash,
            "reuse_authorized": False,
        }
        candidates = []
        for lam in (0.0, 0.01, 0.25, 1.0, 2.0):
            cid = f"bank_{index:02d}_lambda_{lam:g}"
            cid_identity = {
                **unit_identity,
                "candidate_id": cid,
                "lambda": lam,
                "origin_state_hash": origin_hash,
                "bank_hash": bank_hash,
            }
            p = root / "forks" / f"bank_{index:02d}" / "attempt_0000" / "engine" / f"{cid}.pt"
            save_checkpoint(p, candidate_state, cid_identity)
            c = {
                "candidate_id": cid,
                "lambda": lam,
                "checkpoint_identity": cid_identity,
                "checkpoint_path": str(p),
                "checkpoint_sha256": file_hash(p),
                "state_hash": state_hash(candidate_state),
                "parameter_hash": parameter_hash(model, trainable=True),
                "optimizer_state_hash": state_hash(candidate_state["optimizer"]),
            }
            candidates.append(c)
            if index in (0, 6):
                sid = {
                    **identity,
                    "unit": "control_likelihood",
                    "candidate_id": cid,
                    "candidate_parameter_hash": c["parameter_hash"],
                    "candidate_optimizer_hash": c["optimizer_state_hash"],
                    "proposal_hash": canonical_hash([r["record_hash"] for r in proposal]),
                }
                score_root = root / "control_scores" / cid
                score_root.mkdir(parents=True)
                write_json(score_root / "identity.json", sid)
                scores = []
                for row in proposal:
                    s = {
                        "sample_key": canonical_hash([sid, row["sample_key"]]),
                        "proposal_sample_key": row["sample_key"],
                        "candidate_id": cid,
                        "candidate_parameter_hash": c["parameter_hash"],
                        "proposal_record_hash": row["record_hash"],
                        "token_ids": row["token_ids"],
                        "candidate_token_logprobs": [-1.0] * len(row["token_ids"]),
                        "execution_kind": "REAL_CUDA_FORK",
                        "execution_checks": {"passed": True},
                    }
                    s["record_hash"] = canonical_hash(s)
                    scores.append(s)
                (score_root / "samples.jsonl").write_text(
                    "".join(json.dumps(s) + "\n" for s in scores)
                )
        banks[str(index)] = {
            "status": "PASS",
            "passed": True,
            "candidates": candidates,
            "origin_state_hash": origin_hash,
            "identity": unit_identity,
            "candidate_optimizer_updates": 5,
            "replay_optimizer_updates": 1,
            "origin_restored": True,
            "reuse_adopted": False,
            "lambda_zero_replay": {"passed": True, "bitwise_equal": True},
            "all_valid_invariant": {"applicable": False},
            "answer_diagnostics": {
                "A_BASE": {"optimizer_updates": 0},
                "A_VALID": {"optimizer_updates": 0},
            },
        }
    write_json(
        root / "reuse_decision.json",
        {"reuse_authorized": False, "validation_bank_indices": [1, 11], "validation": validation},
    )
    write_json(
        root / "candidate_manifest.json",
        {
            "distinct_candidates": 60,
            "banks": banks,
            "committed_candidates": 0,
            "response_bank_indices": [0, 6],
        },
    )
    write_json(
        root / "joint_advantage_checks.json",
        {"reuse_validation": validation, "bank_advantage_and_gradient_audits": banks},
    )
    counts = {"optimizer_steps_observed_at_least": 80, "backward_calls_observed_at_least": 1}
    write_json(root / "runtime_profile.json", counts)
    details = {
        "raw_sample_count": 1152,
        "train_rollouts": 384,
        "control_proposal_rollouts": 768,
        "distinct_candidate_optimizer_updates": 60,
        "distinct_lambda_zero_replays": 12,
        "response_candidates": 10,
        "control_candidate_sequences_scored": 7680,
        "direct_resample_cold": False,
        "candidate_commit": False,
        "scratch_restoration": {
            k: True
            for k in ("full_origin_state_and_rng", "initial_adapter", "empty_adam", "frozen_base")
        },
        "runtime_counts": counts,
    }
    write_json(
        root / "status.json",
        {
            "status": "PASS",
            "phase": "R3-cold",
            "execution_kind": "REAL_CUDA_FORK",
            "details": details,
        },
    )
    for name in (
        "paired_response.csv",
        "gradients_summary.parquet",
        "support_control_report.md",
        "response_bank_00.json",
        "response_bank_06.json",
    ):
        (root / name).write_text("{}\n")
    write_json(root / "manifest.json", _manifest(root, exclude=("manifest.json", "status.json")))
    return root, gate, r2


def test_completed_bound_cold_evidence_accepts_without_reuse(completed_cold):
    from src.next_stage_runtime import validate_r3_cold_gate

    result = validate_r3_cold_gate(*completed_cold)
    assert result["status"] == "PASS"
    assert result["response_artifact_audit"] == {
        "status": "PASS",
        "fixture_only": True,
        "proposal_sequences": 768,
        "banks": 12,
    }
    assert (
        result["initial_adapter_hash"] == completed_cold[1]["certificate"]["initial_adapter_hash"]
    )


@pytest.mark.parametrize(
    "mutation", ["origin", "duplicate", "replay", "counts", "gate", "missing_sample"]
)
def test_resealed_stage_cannot_hide_broken_cold_gate(completed_cold, mutation):
    from src.next_stage_runtime import validate_r3_cold_gate
    from src.r3_runtime import _manifest

    root, gate, r2 = completed_cold
    if mutation == "gate":
        write_json(root / "gate_binding.json", {})
    if mutation == "origin":
        (root / "origin.pt").write_bytes(b"corrupt checkpoint")
    if mutation == "missing_sample":
        p = root / "samples.jsonl"
        p.write_text("\n".join(p.read_text().splitlines()[:-1]) + "\n")
    if mutation in ("duplicate", "replay"):
        p = root / "candidate_manifest.json"
        v = json.loads(p.read_text())
        b = v["banks"]["0"]
        if mutation == "duplicate":
            b["candidates"][1]["candidate_id"] = b["candidates"][0]["candidate_id"]
        else:
            b["lambda_zero_replay"]["passed"] = False
        write_json(p, v)
        j = json.loads((root / "joint_advantage_checks.json").read_text())
        j["bank_advantage_and_gradient_audits"] = v["banks"]
        write_json(root / "joint_advantage_checks.json", j)
    if mutation == "counts":
        p = root / "status.json"
        v = json.loads(p.read_text())
        v["details"]["runtime_counts"]["optimizer_steps_observed_at_least"] = 60
        write_json(p, v)
    write_json(root / "manifest.json", _manifest(root, exclude=("manifest.json", "status.json")))
    with pytest.raises(ValueError):
        validate_r3_cold_gate(root, gate, r2)


def test_checkpoint_adam_groups_and_steps_are_validated():
    import copy

    import torch

    from src.optimizer_fork import capture_state
    from src.r3_gate import _check_optimizer_state

    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0)
    initial = capture_state(model, optimizer)
    _check_optimizer_state(initial, step=0)
    wrong = copy.deepcopy(initial)
    wrong["optimizer"]["param_groups"][0]["lr"] = 0.02
    with pytest.raises(ValueError):
        _check_optimizer_state(wrong, step=0)
    model(torch.ones(1, 1)).sum().backward()
    optimizer.step()
    one = capture_state(model, optimizer)
    _check_optimizer_state(one, step=1)
    wrong = copy.deepcopy(one)
    wrong["optimizer"]["state"][0]["step"] = torch.tensor(2.0)
    with pytest.raises(ValueError):
        _check_optimizer_state(wrong, step=1)
    wrong = copy.deepcopy(one)
    wrong["optimizer"]["state"][0]["exp_avg"].fill_(float("nan"))
    with pytest.raises(ValueError):
        _check_optimizer_state(wrong, step=1)


@pytest.mark.parametrize(
    "fault",
    [
        "missing_raw",
        "raw_decode",
        "old_logp",
        "behavior_logp",
        "category",
        "input_hash",
        "prompt_ids",
        "prepared_hash",
        "missing_vision",
    ],
)
def test_resealed_incomplete_or_inconsistent_raw_rollout_is_rejected(completed_cold, fault):
    from src.r3_gate import validate_r3_cold_gate
    from src.r3_runtime import _manifest

    root, gate, r2 = completed_cold
    path = root / "samples.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    row = next(r for r in rows if r["bank_role"] == "train" and r["interface"] == "IMAGE_CUE_FRESH")
    if fault == "missing_raw":
        del row["raw_completion"]
    elif fault == "raw_decode":
        row["raw_completion"] = row["raw_text"] = "other invalid text"
    elif fault == "old_logp":
        row["old_logprobs"] = []
    elif fault == "behavior_logp":
        row["behavior_token_logprobs"][0] = 0.5
    elif fault == "category":
        row["category"] = "X"
    elif fault == "input_hash":
        del row["input_tensor_hash"]
    elif fault == "prompt_ids":
        row["final_prompt_token_ids"][0] += 1
    elif fault == "prepared_hash":
        del row["prepared_hash"]
    else:
        row["vision_forward_calls"] = 0
    row["record_hash"] = canonical_hash({k: v for k, v in row.items() if k != "record_hash"})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    write_json(root / "manifest.json", _manifest(root, exclude=("manifest.json", "status.json")))
    with pytest.raises(ValueError):
        validate_r3_cold_gate(root, gate, r2)


@pytest.mark.parametrize(
    "fault", ["candidate_sample_hash", "candidate_bank_hash", "summary_sample_hash"]
)
def test_resealed_candidate_must_bind_the_same_32_frozen_rollouts(completed_cold, fault):
    from src.core import file_hash
    from src.optimizer_fork import load_checkpoint, save_checkpoint
    from src.r3_gate import validate_r3_cold_gate
    from src.r3_runtime import _manifest

    root, gate, r2 = completed_cold
    manifest = json.loads((root / "candidate_manifest.json").read_text())
    summary = manifest["banks"]["0"]
    if fault == "summary_sample_hash":
        summary["identity"]["sample_hash"] = "0" * 64
    else:
        candidate = summary["candidates"][0]
        state = load_checkpoint(candidate["checkpoint_path"], candidate["checkpoint_identity"])
        key = "sample_hash" if fault == "candidate_sample_hash" else "bank_hash"
        candidate["checkpoint_identity"][key] = "0" * 64
        save_checkpoint(candidate["checkpoint_path"], state, candidate["checkpoint_identity"])
        candidate["checkpoint_sha256"] = file_hash(candidate["checkpoint_path"])
    write_json(root / "candidate_manifest.json", manifest)
    joint = json.loads((root / "joint_advantage_checks.json").read_text())
    joint["bank_advantage_and_gradient_audits"] = manifest["banks"]
    write_json(root / "joint_advantage_checks.json", joint)
    write_json(root / "manifest.json", _manifest(root, exclude=("manifest.json", "status.json")))
    with pytest.raises(ValueError):
        validate_r3_cold_gate(root, gate, r2)


def test_resealed_reuse_probe_must_bind_its_preregistered_bank(completed_cold):
    from src.r3_gate import validate_r3_cold_gate
    from src.r3_runtime import _manifest

    root, gate, r2 = completed_cold
    path = root / "validation" / "bank_01" / "attempt_0000" / "identity.json"
    identity = json.loads(path.read_text())
    identity["sample_hash"] = "0" * 64
    write_json(path, identity)
    write_json(root / "manifest.json", _manifest(root, exclude=("manifest.json", "status.json")))
    with pytest.raises(ValueError, match="reuse"):
        validate_r3_cold_gate(root, gate, r2)


def test_tokenizer_gate_uses_only_pinned_local_processor_and_certified_vocab(monkeypatch):
    from types import SimpleNamespace

    from transformers import AutoProcessor

    from src.model_adapters.base import _hash_json
    from src.r3_gate import _load_tokenizer

    calls = []
    vocab = {"x": 1, "</s>": 0}
    tokenizer = SimpleNamespace(get_vocab=lambda: vocab)

    def load(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return SimpleNamespace(tokenizer=tokenizer)

    monkeypatch.setattr(AutoProcessor, "from_pretrained", load)
    config = {"model": {"id": "Qwen/Qwen3.5-9B", "revision": "a" * 40}}
    assert _load_tokenizer(config, {"tokenizer_hash": _hash_json(vocab)}) is tokenizer
    assert calls == [
        (
            config["model"]["id"],
            {"revision": "a" * 40, "trust_remote_code": False, "local_files_only": True},
        )
    ]
    with pytest.raises(ValueError, match="vocabulary"):
        _load_tokenizer(config, {"tokenizer_hash": "0" * 64})
    with pytest.raises(ValueError, match="pinned"):
        _load_tokenizer({"model": {"id": config["model"]["id"], "revision": "main"}}, {})


def test_response_artifact_failure_blocks_the_complete_cold_gate(completed_cold, monkeypatch):
    from src.r3_gate import validate_r3_cold_gate

    def fail(root, *, proposal_records, bank_summaries):
        assert len(proposal_records) == 768
        assert set(bank_summaries) == {str(i) for i in range(12)}
        raise ValueError("response artifact mismatch")

    monkeypatch.setattr("src.r3_report_gate.validate_response_artifacts", fail)
    with pytest.raises(ValueError, match="response artifact mismatch"):
        validate_r3_cold_gate(*completed_cold)
