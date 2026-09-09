"""R4 acceptance tests never substitute CPU fixtures for real CUDA evidence."""

import copy
import json
from pathlib import Path

import pytest

from src.core import canonical_hash, write_json
from src.r4_gate import _committed_sources as _committed_source_reader


def test_gate_rejects_running_fixture_and_missing_full_evidence(tmp_path):
    from src.r4_gate import validate_r4_gate

    write_json(tmp_path / "manifest.json", {"files": []})
    for status, kind in (
        ("RUNNING", "REAL_CUDA_TRAINING"),
        ("PASS", "CPU_FAKE_ADAPTER_FIXTURE"),
        ("PASS", "REAL_CUDA_TRAINING"),
    ):
        write_json(
            tmp_path / "status.json", {"phase": "R4", "status": status, "execution_kind": kind}
        )
        with pytest.raises(ValueError):
            validate_r4_gate(tmp_path, {}, {}, {})


class _Tokenizer:
    def decode(self, ids, **kwargs):
        return "".join(chr(token - 1) for token in ids)


@pytest.fixture
def raw_case():
    import torch

    from src.model_adapters.base import _hash_json
    from src.optimizer_fork import state_hash
    from src.r4_inputs import build_train_schedule
    from src.r4_runtime import _annotate_rollout, _scene_fields, build_requests

    train = [
        json.loads(line) for line in Path("data/generated/train.jsonl").read_text().splitlines()
    ]
    prompt = next(
        p
        for p in build_train_schedule(train)["train_prompts"]
        if p["interface"] == "IMAGE_CUE_FRESH"
    )
    config = {
        "model": {"id": "fixture", "revision": "fixture"},
        "protocol_version": "fixture",
        "generation_proposed_N": {"max_new_tokens": 64},
    }
    identity = {"model_hash": "m", "data_hash": "d", "config_hash": canonical_hash(config)}
    policy = {"state_hash": "a" * 64, "parameter_hash": "b" * 64, "optimizer_state_hash": "c" * 64}
    row = build_requests([prompt], identity, "X_BASE", 0, "train", policy["state_hash"])[0]
    text = (
        prompt["prompt"]["system"] + "\n" + prompt["prompt"]["user"] + "\n<think>\n\n</think>\n\n"
    )
    ids = [ord(c) + 1 for c in text]
    audit = {
        "final_prompt": text,
        "final_prompt_token_ids": ids,
        "prompt_token_count": len(ids),
        "final_prompt_hash": _hash_json(text),
        "tokenized_prompt_hash": state_hash(torch.tensor([ids])),
        "input_tensor_hash": "d" * 64,
        "prepared_hash": "e" * 64,
        "image_token_count": 1,
        "actual_image_tokens": 1,
        "pixel_values_hash": "f" * 64,
        "enable_thinking": False,
    }
    raw, tokens = "bad", [99, 98, 101, 0]
    old = [-1.25] * 4
    row.update(
        {
            **audit,
            **_annotate_rollout(raw, prompt),
            **_scene_fields(prompt, audit),
            "input_ids_hash": audit["tokenized_prompt_hash"],
            "raw_completion": raw,
            "raw_text": raw,
            "token_ids": tokens,
            "raw_token_ids": tokens,
            "completion_length": 4,
            "n_generated_tokens": 4,
            "old_logprobs": old,
            "behavior_token_logprobs": old,
            "per_token_logprob_behavior": old,
            "logprob_sequence": -5.0,
            "stop_reason": "eos",
            "vision_forward_calls": 1,
            "model_id": "fixture",
            "model_revision": "fixture",
            "adapter_hash": policy["parameter_hash"],
            "optimizer_state_hash": policy["optimizer_state_hash"],
            "protocol_version": "fixture",
            "train_seed": 17,
            "generation_config_hash": canonical_hash(config["generation_proposed_N"]),
            "execution_kind": "REAL_CUDA_TRAINING",
            "execution_checks": {"passed": True, "faults": []},
            "elapsed": 0.01,
            "elapsed_seconds": 0.01,
            "elapsed_unit": "seconds",
            "elapsed_scope": "adapter.generate",
            "elapsed_status": "MEASURED",
            "peak_memory_status": "CUDA_PHASE_PEAK_AND_PROCESS_RSS_HIGH_WATERMARK",
            "peak_memory": {
                "peak_cuda_bytes": 1,
                "peak_cuda_reserved_bytes": 1,
                "peak_cpu_rss_bytes": 1,
            },
            "runtime_forward_by_reason": {"generation": 4},
        }
    )
    return row, prompt, identity, policy, config


def test_raw_validator_accepts_complete_controlled_record_without_mutation(raw_case):
    from src.r4_gate import _validate_raw_row

    before = copy.deepcopy(raw_case)
    _validate_raw_row(*raw_case, {}, _Tokenizer(), [0], {})
    assert raw_case == before


@pytest.mark.parametrize(
    "fault",
    [
        "raw",
        "tokens",
        "eos",
        "old",
        "annotation",
        "prepared",
        "input_ids",
        "policy",
        "image",
        "elapsed",
    ],
)
def test_raw_validator_rejects_missing_or_changed_real_evidence(raw_case, fault):
    from src.r4_gate import _validate_raw_row

    row = raw_case[0]
    key, value = {
        "raw": ("raw_completion", "different"),
        "tokens": ("raw_token_ids", [9]),
        "eos": ("stop_reason", "length"),
        "old": ("old_logprobs", []),
        "annotation": ("category", "X"),
        "prepared": ("prepared_hash", None),
        "input_ids": ("tokenized_prompt_hash", "0" * 64),
        "policy": ("adapter_hash", "0" * 64),
        "image": ("vision_forward_calls", 0),
        "elapsed": ("elapsed_status", "CPU_FIXTURE"),
    }[fault]
    row[key] = value
    with pytest.raises(ValueError):
        _validate_raw_row(*raw_case, {}, _Tokenizer(), [0], {})


def test_step64_state_requires_mature_adam_and_exact_own_arm_sampler_keys():
    import torch

    from src.optimizer_fork import capture_state
    from src.r4_gate import _check_state

    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0)
    origin = capture_state(
        model,
        optimizer,
        {
            "arm": "INITIAL",
            "checkpoint_step": 0,
            "completed_sample_keys": [],
            "sampler": {"plan_hash": "p", "position": 0},
            "scheduler": None,
            "grad_scaler": None,
        },
    )
    keys = [f"own-{i}" for i in range(2048)]
    for _ in range(64):
        optimizer.zero_grad(set_to_none=True)
        model(torch.ones(1, 1)).sum().backward()
        optimizer.step()
    current = capture_state(
        model,
        optimizer,
        {
            "arm": "X_BASE",
            "checkpoint_step": 64,
            "completed_sample_keys": keys,
            "completed_step_sample_keys": keys[-32:],
            "sampler": {"plan_hash": "p", "position": 64},
            "scheduler": None,
            "grad_scaler": None,
        },
    )
    _check_state(current, origin, "X_BASE", 64, "p", keys, keys[-32:])
    for fault in ("keys", "arm", "step", "adam"):
        value = copy.deepcopy(current)
        if fault == "keys":
            value["metadata"]["completed_sample_keys"][0] = "other-arm"
        elif fault == "arm":
            value["metadata"]["arm"] = "X_VALID"
        elif fault == "step":
            value["metadata"]["sampler"]["position"] = 63
        else:
            value["optimizer"]["state"][0]["step"] = torch.tensor(63.0)
        with pytest.raises(ValueError):
            _check_state(value, origin, "X_BASE", 64, "p", keys, keys[-32:])


def test_processor_reconstruction_binds_full_prepared_pixels_and_reuses_only_audits(raw_case):
    from types import SimpleNamespace

    import torch

    from src.optimizer_fork import state_hash
    from src.r4_gate import _PreparedInputs, _validate_raw_row

    row, _prompt, *_ = raw_case
    audit = {
        k: row[k]
        for k in (
            "final_prompt",
            "final_prompt_hash",
            "final_prompt_token_ids",
            "prompt_token_count",
            "tokenized_prompt_hash",
            "input_tensor_hash",
            "image_token_count",
            "pixel_values_hash",
            "enable_thinking",
        )
    }
    prepared = {
        "inputs": {"input_ids": torch.tensor([[4]]), "pixels": torch.tensor([2.0])},
        "audit": audit,
    }
    row["prepared_hash"] = state_hash(prepared)
    calls = []

    def prepare(value, root):
        calls.append((value, root))
        return prepared

    bindings = _PreparedInputs(SimpleNamespace(prepare=prepare), "root")
    _validate_raw_row(*raw_case, {}, _Tokenizer(), [0], bindings)
    _validate_raw_row(*raw_case, {}, _Tokenizer(), [0], bindings)
    assert len(calls) == 1
    before = state_hash(prepared)
    row["prepared_hash"] = "0" * 64
    with pytest.raises(ValueError, match="reconstructed"):
        _validate_raw_row(*raw_case, {}, _Tokenizer(), [0], bindings)
    assert state_hash(prepared) == before


def _seal(root):
    from src.core import file_hash

    write_json(
        root / "manifest.json",
        {
            "files": [
                {
                    "path": str(p.relative_to(root)),
                    "sha256": file_hash(p),
                    "bytes": p.stat().st_size,
                }
                for p in sorted(root.rglob("*"))
                if p.is_file() and p != root / "manifest.json"
            ]
        },
    )


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            row["record_hash"] = canonical_hash(
                {k: v for k, v in row.items() if k != "record_hash"}
            )
            stream.write(json.dumps(row, sort_keys=True) + "\n")


@pytest.fixture(scope="module")
def complete_evidence(tmp_path_factory):
    """Complete synthetic artifact format: CPU fixtures do not establish CUDA provenance."""
    from types import SimpleNamespace

    import torch
    from test_r4_inputs import _plan, data

    from src import r2_result_audit, r4_gate, r4_report_gate, r4_runtime
    from src.core import file_hash
    from src.grpo_update import grouped_advantages
    from src.model_adapters.base import _hash_json
    from src.next_stage_common import load_yaml
    from src.optimizer_fork import capture_state, save_checkpoint, state_hash
    from src.r3_inputs import build_r3_plan
    from src.r4_metrics import control_kl_diagnostic

    root = tmp_path_factory.mktemp("r4-gate-synthetic")
    patch = pytest.MonkeyPatch()
    dataset = data.__wrapped__()
    plan = _plan(dataset, data_root=str(Path("data/generated").resolve()))
    config = load_yaml(Path("configs/next_stage.yaml"))
    config["data_root"] = str(Path("data/generated").resolve())
    source = {
        "source_commit": "1" * 40,
        "source_files": {"src/r4_runtime.py": file_hash(Path("src/r4_runtime.py"))},
    }
    patch.setattr(r4_gate, "_committed_sources", lambda revision: source["source_files"].copy())
    data_binding = {"fixture_data_hash": canonical_hash(dataset["train"])}
    legacy_lock = dataset["legacy_lock"]
    patch.setattr(r4_runtime, "_load_plan", lambda *a: (plan, data_binding, legacy_lock))
    model = torch.nn.Module()
    model.register_parameter("lora_B", torch.nn.Parameter(torch.zeros(1, 1)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0)
    origin = capture_state(
        model,
        optimizer,
        {
            "arm": "INITIAL",
            "checkpoint_step": 0,
            "completed_sample_keys": [],
            "sampler": {"plan_hash": plan["plan_hash"], "position": 0},
            "scheduler": None,
            "grad_scaler": None,
        },
    )
    # Saved RNG bytes are synthetic unit fixtures, not evidence of executed GPU work.
    origin["rng"]["cuda"] = [torch.tensor([1, 2, 3], dtype=torch.uint8)]
    audit = {
        "execution_kind": "REAL_CUDA_INFERENCE",
        "model_id": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "processor_hash": "a" * 64,
        "tokenizer_hash": "b" * 64,
        "chat_template_hash": "c" * 64,
        "frozen_parameter_hash": "f" * 64,
        "eos_token_ids": [0],
        "processor_max_pixels": 786432,
        "processor_patch_size": 16,
        "processor_merge_size": 2,
        "requested_image_token_limit": 768,
        "base_dtype": "bfloat16",
        "probability_execution": "uncached_prefix_recompute",
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0,
        "trainable_parameters": 1,
        "trainable_dtypes": ["torch.float32"],
    }
    environment = {
        "status": "PASS",
        "python": "3.fixture",
        "observed_packages": {"transformers": "fixture"},
        "historical_version_coverage": ["python", "transformers"],
    }
    gate = {
        "config": config,
        "binding": {
            "r0_dir": str(root / "unused-r0"),
            "source_implementation_hashes": source["source_files"].copy(),
        },
        "certificate": {
            "initial_adapter_hash": r4_gate._parameter_hash(origin),
            "model_audit": audit,
        },
    }
    r2 = {"status": "PASS", "environment": environment}
    r3dir = root.parent / (root.name + "-r3")
    r3identity = {"phase": "R3", "fixture": True}
    write_json(r3dir / "runtime_lock.json", {"identity": r3identity})
    save_checkpoint(r3dir / "origin.pt", origin, {**r3identity, "unit": "initial_origin"})
    control = build_r3_plan(dataset["train"], dataset["control"])["control_prompts"]
    write_json(r3dir / "bank_manifest.json", {"plan": {"control_prompts": control}})
    probe = [
        dict(
            sample_key=f"probe-{i}",
            prompt_id=p["prompt_id"],
            base_scene_id=p["base_scene_id"],
            family=p["family"],
            interface=p["interface"],
            token_ids=[99, 98, 101, 0],
            old_logprobs=[-1.25] * 4,
            stop_reason="eos",
            bank_role="control_proposal",
            sample_index=0,
            execution_kind="REAL_CUDA_FORK",
            execution_checks={"passed": True},
        )
        for i, p in enumerate(control)
    ]
    _write_rows(r3dir / "samples.jsonl", probe)
    r3 = {
        "status": "PASS",
        "environment": environment,
        "r3_cold_dir": str(r3dir),
        "plan_hash": "cold-plan",
        "r3_files": {
            n: file_hash(r3dir / n)
            for n in ("samples.jsonl", "origin.pt", "runtime_lock.json", "bank_manifest.json")
        },
    }
    identity = dict(
        phase="R4",
        model_hash=canonical_hash(config["model"]),
        config_hash=canonical_hash(config),
        data_hash=canonical_hash(data_binding),
        source_hash=canonical_hash(source),
        gate_hash=canonical_hash(gate["binding"]),
        r2_gate_hash=canonical_hash(r2),
        r3_cold_gate_hash=canonical_hash(r3),
        plan_hash=plan["plan_hash"],
        initial_adapter_hash=gate["certificate"]["initial_adapter_hash"],
        execution_kind="REAL_CUDA_TRAINING",
    )
    initial_identity = {**identity, "unit": "initial_origin"}
    save_checkpoint(root / "origin.pt", origin, initial_identity)
    authorization = {
        "allow_training_flag": True,
        "canonical_config_allow_training": config["R4"]["allow_training"],
    }
    runtime = dict(
        identity=identity,
        config=config,
        environment=environment,
        source=source,
        model_audit=audit,
        origin_hash=state_hash(origin),
        origin_file_sha256=file_hash(root / "origin.pt"),
        frozen_base_hash="f" * 64,
        initial_adapter_hash=identity["initial_adapter_hash"],
        optimizer_initial_hash=state_hash(origin["optimizer"]),
        selected_probability_path="uncached_prefix_recompute",
        scheduler=None,
        gradient_scaler=None,
        authorization=authorization,
    )
    write_json(root / "runtime_lock.json", runtime)
    write_json(root / "gate_binding.json", {"r0_r1": gate["binding"], "r2": r2, "r3_cold": r3})
    write_json(
        root / "two_arm_training_config.json",
        dict(
            canonical_config=config,
            plan=plan,
            data_binding=data_binding,
            authorization=authorization,
            control_probe={
                "source_r3_gate_hash": canonical_hash(r3),
                "source_samples_sha256": r3["r3_files"]["samples.jsonl"],
                "sample_keys": [r["sample_key"] for r in probe],
                "record_hashes": [r["record_hash"] for r in probe],
                "new_control_generations": 0,
                "sequences_per_step": 48,
                "planned_sequence_scores": 6144,
            },
        ),
    )
    prepared_cache = {}

    def prepare(prompt, data_root):
        key = canonical_hash(prompt)
        if key not in prepared_cache:
            text = (
                (prompt.get("system") or "") + "\n" + prompt["user"] + "\n<think>\n\n</think>\n\n"
            )
            ids = torch.tensor([[ord(c) + 1 for c in text]], dtype=torch.long)
            image = bool(prompt.get("image_path"))
            tensors = {"input_ids": ids}
            if image:
                tensors["pixel_values"] = torch.ones(1)
            prepared_cache[key] = {
                "inputs": tensors,
                "audit": {
                    "final_prompt": text,
                    "final_prompt_token_ids": ids[0].tolist(),
                    "prompt_token_count": ids.numel(),
                    "final_prompt_hash": _hash_json(text),
                    "tokenized_prompt_hash": state_hash(ids),
                    "input_tensor_hash": state_hash(tensors),
                    "image_token_count": int(image),
                    "pixel_values_hash": state_hash(tensors["pixel_values"]) if image else None,
                    "enable_thinking": False,
                    "image_grid_thw": None,
                },
            }
        return prepared_cache[key]

    adapter = SimpleNamespace(prepare=prepare, processor=SimpleNamespace(tokenizer=_Tokenizer()))
    patch.setattr(r2_result_audit, "load_processor_adapter", lambda runtime: adapter)
    patch.setattr(
        r4_gate,
        "_initial_rows",
        lambda root, gate, runtime, plan, origin, adapter, shared: {"N": shared},
    )
    report_calls = []

    def report(root, **kwargs):
        report_calls.append({k: len(v) for k, v in kwargs.items() if k != "initial_rows_by_track"})
        return {"status": "PASS", "execution_kind": "CPU_MATH", "fixture_report_boundary": True}

    patch.setattr(r4_report_gate, "validate_r4_response_artifacts", report)
    counts = []

    def sample(path, prompts, state, arm, step, role):
        policy = r4_gate._policy(state)
        requests = r4_runtime.build_requests(
            prompts, identity, arm, step, role, policy["state_hash"]
        )
        write_json(
            path / "identity.json",
            {
                **identity,
                "arm": arm,
                "step": step,
                "role": role,
                "policy_state_hash": policy["state_hash"],
                "request_hash": canonical_hash(requests),
            },
        )
        by_prompt = {p["prompt_id"]: p for p in prompts}
        rows = []
        for req in requests:
            p = by_prompt[req["prompt_id"]]
            prepared = prepare(p["prompt"], config["data_root"])
            a = prepared["audit"]
            row = {
                **req,
                **a,
                **r4_runtime._annotate_rollout("bad", p),
                **r4_runtime._scene_fields(p, a),
                "input_ids_hash": a["tokenized_prompt_hash"],
                "prepared_hash": state_hash(prepared),
                "raw_completion": "bad",
                "raw_text": "bad",
                "token_ids": [99, 98, 101, 0],
                "raw_token_ids": [99, 98, 101, 0],
                "completion_length": 4,
                "n_generated_tokens": 4,
                "old_logprobs": [-1.25] * 4,
                "behavior_token_logprobs": [-1.25] * 4,
                "per_token_logprob_behavior": [-1.25] * 4,
                "logprob_sequence": -5.0,
                "stop_reason": "eos",
                "vision_forward_calls": a["image_token_count"],
                "model_id": config["model"]["id"],
                "model_revision": config["model"]["revision"],
                "adapter_hash": policy["parameter_hash"],
                "optimizer_state_hash": policy["optimizer_state_hash"],
                "protocol_version": config["protocol_version"],
                "train_seed": 17,
                "generation_config_hash": canonical_hash(
                    legacy_lock["generation"]
                    if p["track"] == "L"
                    else config["generation_proposed_N"]
                ),
                "execution_kind": "REAL_CUDA_TRAINING",
                "execution_checks": {"passed": True, "faults": []},
                "actual_image_tokens": a["image_token_count"],
                "elapsed": 0.01,
                "elapsed_seconds": 0.01,
                "elapsed_unit": "seconds",
                "elapsed_scope": "adapter.generate",
                "elapsed_status": "MEASURED",
                "peak_memory_status": "CUDA_PHASE_PEAK_AND_PROCESS_RSS_HIGH_WATERMARK",
                "peak_memory": {
                    "peak_cuda_bytes": 1,
                    "peak_cuda_reserved_bytes": 1,
                    "peak_cpu_rss_bytes": 1,
                },
                "runtime_forward_by_reason": {"generation": 4},
            }
            rows.append(row)
        _write_rows(path / "samples.jsonl", rows)
        counts.append(len(rows))
        return rows

    def scores(path, rows, ident, state):
        policy = r4_gate._policy(state)
        ident = {
            **ident,
            "scored_parameter_hash": policy["parameter_hash"],
            "scored_optimizer_hash": policy["optimizer_state_hash"],
        }
        write_json(path / "identity.json", ident)
        _write_rows(
            path / "samples.jsonl",
            [
                dict(
                    sample_key=canonical_hash([ident, r["sample_key"]]),
                    proposal_sample_key=r["sample_key"],
                    proposal_record_hash=r["record_hash"],
                    token_ids=r["token_ids"],
                    scored_parameter_hash=policy["parameter_hash"],
                    scored_optimizer_hash=policy["optimizer_state_hash"],
                    new_token_logprobs=r["old_logprobs"],
                    execution_kind="REAL_CUDA_TRAINING",
                    execution_checks={"passed": True},
                )
                for r in rows
            ],
        )

    sample(
        root / "evaluation/INITIAL/step_00/N",
        plan["dev_panel_prompts"],
        origin,
        "INITIAL",
        0,
        "evaluation",
    )
    entries = []
    lookup = {p["prompt_id"]: p for p in plan["train_prompts"]}
    for arm in r4_gate.ARMS:
        entries.append(
            dict(
                arm=arm,
                step=0,
                checkpoint_path=str(root / "origin.pt"),
                checkpoint_identity=initial_identity,
                state_hash=state_hash(origin),
                milestone=True,
            )
        )
        state = copy.deepcopy(origin)
        keys = []
        for step, prompt_ids in enumerate(plan["train_steps"], 1):
            rows = sample(
                root / arm / f"step_{step:02d}/rollouts",
                [lookup[k] for k in prompt_ids],
                state,
                arm,
                step - 1,
                "train",
            )
            keys.extend(r["sample_key"] for r in rows)
            unit = {
                **identity,
                "unit": "training_step",
                "arm": arm,
                "step": step,
                "prestate_hash": state_hash(state),
                "sample_hash": canonical_hash([r["record_hash"] for r in rows]),
                "probe_hash": canonical_hash([r["record_hash"] for r in probe]),
            }
            attempt = root / arm / f"step_{step:02d}/updates/attempt_0000"
            post = copy.deepcopy(state)
            post["optimizer"]["state"] = {
                0: {
                    "step": torch.tensor(float(step)),
                    "exp_avg": torch.zeros(1, 1),
                    "exp_avg_sq": torch.zeros(1, 1),
                }
            }
            post["metadata"] = dict(
                arm=arm,
                checkpoint_step=step,
                sampler={"plan_hash": plan["plan_hash"], "position": step},
                completed_sample_keys=keys.copy(),
                completed_step_sample_keys=[r["sample_key"] for r in rows],
                scheduler=None,
                grad_scaler=None,
            )
            checkpoint_identity = {**unit, "unit": "post_update_checkpoint"}
            save_checkpoint(attempt / "checkpoint.pt", post, checkpoint_identity)
            groups = [
                dict(
                    prompt_id=pid,
                    group_id=pid,
                    K=8,
                    **{"lambda": 0.0 if arm == "X_BASE" else 1.0},
                    category_counts={"X": 0, "S": 0, "W": 0, "I": 8},
                    reward_vector=[0.0] * 8,
                    sample_keys=[r["sample_key"] for r in rows if r["prompt_id"] == pid],
                    **grouped_advantages([0.0] * 8),
                )
                for pid in prompt_ids
            ]
            update = dict(
                optimizer_updates=1,
                backward_calls=32,
                sequences=32,
                Lnorm=64,
                post_update_likelihood_sequences=32,
                post_update_likelihood_forwards=32,
                loss=0.0,
                grad_norm_preclip=0.0,
                grad_norm_postclip=0.0,
                actual_step_norm=0.0,
                clip_fraction=0.0,
                empirical_same_training_bank_kl=0.0,
                token_records=[
                    dict(
                        advantage=0.0,
                        Lnorm=64,
                        masked_token_count=4,
                        loss_reduction="sum_generated_tokens/(B*K*64)",
                        clip_fraction=0.0,
                        loss=0.0,
                    )
                    for _ in rows
                ],
            )
            parity = dict(mean_abs_token_error=0.0, p99_abs_token_error=0.0, tokens=128)
            diagnostic = control_kl_diagnostic(
                probe, {r["sample_key"]: r["old_logprobs"] for r in probe}, eos_token_ids=[0]
            )
            result = dict(
                status="PASS",
                arm=arm,
                step=step,
                checkpoint_identity=checkpoint_identity,
                checkpoint_path="checkpoint.pt",
                checkpoint_sha256=file_hash(attempt / "checkpoint.pt"),
                **r4_gate._policy(post),
                update=update,
                group_statistics=groups,
                parity=parity,
                control_diagnostic=diagnostic,
                training_category_counts={"X": 0, "S": 0, "W": 0, "I": 32},
                zero_advantage_groups=4,
                control_sequences_scored=48,
                sample_keys=[r["sample_key"] for r in rows],
            )
            for name, value in [
                ("identity", unit),
                ("result", result),
                ("update", update),
                ("group_statistics", groups),
                ("parity", parity),
                ("control_diagnostic", diagnostic),
            ]:
                write_json(attempt / f"{name}.json", value)
            scores(attempt / "preupdate_parity", rows, {**unit, "unit": "preupdate_parity"}, state)
            scores(
                attempt / "control_scores",
                probe,
                {**unit, "unit": "fixed_step0_control", "poststate_hash": state_hash(post)},
                post,
            )
            _seal(attempt)
            write_json(
                attempt / "completed.json",
                {
                    "status": "PASS",
                    "identity_hash": canonical_hash(unit),
                    "manifest_sha256": file_hash(attempt / "manifest.json"),
                },
            )
            entries.append(
                dict(
                    arm=arm,
                    step=step,
                    checkpoint_identity=checkpoint_identity,
                    checkpoint_path=str(attempt / "checkpoint.pt"),
                    checkpoint_sha256=result["checkpoint_sha256"],
                    **r4_gate._policy(post),
                    milestone=step in (16, 32, 64),
                )
            )
            state = post
            if step == 32:
                sample(
                    root / "evaluation" / arm / "step_32/N",
                    plan["dev_panel_prompts"],
                    state,
                    arm,
                    step,
                    "evaluation",
                )
            if step == 64:
                for track, key in [
                    ("N", "dev_prompts"),
                    ("L", "legacy_prompts"),
                    ("OOD", "ood_prompts"),
                ]:
                    sample(
                        root / "evaluation" / arm / f"step_64/{track}",
                        plan[key],
                        state,
                        arm,
                        step,
                        "evaluation",
                    )
    assert sum(counts) == 14400
    warm = next(e for e in entries if e["arm"] == "X_BASE" and e["step"] == 64)
    write_json(
        root / "checkpoint_manifest.json",
        {"identity": identity, "checkpoints": entries, "X_BASE_step64_for_R3_warm": warm},
    )
    observed = dict(
        optimizer_step_calls_observed=128,
        backward_calls_observed=4096,
        language_layers_instrumented=32,
        top_hook_installed=True,
        vision_hooks_installed=1,
        internal_recompute_status="MEASURED",
    )
    invocation = root / "invocations/attempt_0000"
    write_json(invocation / "runtime_profile.json", observed)
    write_json(
        invocation / "completion.json",
        {"status": "PASS", "details": {"final_scratch_origin_restored": True}},
    )
    write_json(
        invocation / "model_load_audit.json",
        {**audit, "load_seconds": 1.0, "load_peak_cuda_bytes": 1},
    )
    cost = dict(
        optimizer_steps_observed_at_least=128,
        backward_calls_observed_at_least=4096,
        incomplete_invocations_with_unknown_extra_cost=0,
        invocations=[dict(invocation="attempt_0000", completed=True, observed=observed)],
    )
    write_json(root / "runtime_profile.json", cost)
    details = dict(
        distinct_optimizer_updates=128,
        training_rollouts=4096,
        shared_step0_outputs=576,
        step32_outputs=1152,
        step64_outputs=8576,
        new_outputs=14400,
        fixed_control_sequence_scores=6144,
        preupdate_parity_sequence_scores=4096,
        postupdate_training_sequence_scores=4096,
        new_control_outputs=0,
        arms={a: 64 for a in r4_gate.ARMS},
        final_scratch_origin_restored=True,
        runtime_counts={k: v for k, v in cost.items() if k != "invocations"},
    )
    write_json(
        root / "status.json",
        {
            "phase": "R4",
            "status": "PASS",
            "execution_kind": "REAL_CUDA_TRAINING",
            "details": details,
            **source,
        },
    )
    for name in (
        "initial_alignment.json",
        "step32_metrics.json",
        "endpoint_metrics.json",
        "L_family_equal_sensitivity.json",
    ):
        write_json(root / name, {"fixture_report_boundary": True})
    for name in ("N_L_OOD_effects.csv", "learning_curves.csv", "pilot_report.md", "report_zh.md"):
        (root / name).write_text("CPU synthetic gate boundary fixture\n")
    _seal(root)
    yield root, gate, r2, r3, report_calls
    patch.undo()


def test_complete_14400_record_gate_binds_both_64_step_chains_and_full_warm_state(
    complete_evidence,
):
    from src.r4_gate import validate_r4_gate

    root, gate, r2, r3, reports = complete_evidence
    before = (root / "manifest.json").read_bytes()
    binding = validate_r4_gate(root, gate, r2, r3)
    assert binding["status"] == "PASS"
    assert binding["r3_cold_plan_hash"] == r3["plan_hash"]
    warm = binding["warm_checkpoint"]
    assert warm["arm"] == "X_BASE" and warm["step"] == 64
    assert warm["path"] == warm["checkpoint_path"]
    assert warm["identity"] == warm["checkpoint_identity"]
    assert warm["file_sha256"] == warm["checkpoint_sha256"]
    assert all(
        len(warm[k]) == 64
        for k in ("file_sha256", "state_hash", "parameter_hash", "optimizer_state_hash")
    )
    assert reports[-1] == dict(
        endpoint_rows=8576, step32_rows=1152, shared_initial_rows=576, step_summaries=128
    )
    assert (root / "manifest.json").read_bytes() == before


def test_commit_source_requires_exact_committed_python_tree_and_certified_frozen_bytes(monkeypatch):
    from src import r4_gate

    committed = {"src/r4_runtime.py": "a" * 64, "src/model_adapters/base.py": "b" * 64}
    calls = []
    monkeypatch.setattr(r4_gate, "_committed_sources", lambda rev: calls.append(rev) or committed)
    runtime = {"source": {"source_commit": "1" * 40, "source_files": committed.copy()}}
    gate = {"binding": {"source_implementation_hashes": {"src/model_adapters/base.py": "b" * 64}}}
    r4_gate._check_source(runtime, gate)
    assert calls == ["1" * 40]
    for fault in ("omit", "change", "extra", "revision", "frozen"):
        value = copy.deepcopy(runtime)
        if fault == "omit":
            del value["source"]["source_files"]["src/r4_runtime.py"]
        elif fault == "change":
            value["source"]["source_files"]["src/r4_runtime.py"] = "c" * 64
        elif fault == "extra":
            value["source"]["source_files"]["src/extra.py"] = "c" * 64
        elif fault == "revision":
            value["source"]["source_commit"] = "not-a-commit"
        else:
            gate = copy.deepcopy(gate)
            gate["binding"]["source_implementation_hashes"]["src/model_adapters/base.py"] = "d" * 64
        with pytest.raises(ValueError):
            r4_gate._check_source(value, gate)


def test_raw_validator_rejects_fabricated_cuda_peak(raw_case):
    from src.r4_gate import _validate_raw_row

    raw_case[0]["peak_memory"] = {
        "peak_cuda_bytes": None,
        "peak_cuda_reserved_bytes": None,
        "peak_cpu_rss_bytes": 1,
    }
    with pytest.raises(ValueError, match="runtime"):
        _validate_raw_row(*raw_case, {}, _Tokenizer(), [0], {})


def test_runtime_profile_requires_instrumented_invocations_not_only_aggregate_pass(tmp_path):
    from src.r4_gate import _check_runtime_profile

    cost = {
        "optimizer_steps_observed_at_least": 128,
        "backward_calls_observed_at_least": 4096,
        "invocations": [],
    }
    with pytest.raises(ValueError):
        _check_runtime_profile(tmp_path, {}, cost, {})


def test_git_source_reader_uses_recorded_objects_without_current_checkout_comparison(monkeypatch):
    import hashlib
    from types import SimpleNamespace

    from src import r4_gate

    object_id = b"a" * 40
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        if args[1] == "ls-tree":
            return SimpleNamespace(
                stdout=b"100644 blob " + object_id + b"\tssvc_flow/src/older.py\0"
            )
        assert args == ["git", "cat-file", "--batch"]
        assert kwargs["input"] == object_id + b"\n"
        return SimpleNamespace(stdout=object_id + b" blob 4\nold\n\n")

    monkeypatch.setattr(r4_gate.subprocess, "run", run)
    assert _committed_source_reader("1" * 40) == {
        "src/older.py": hashlib.sha256(b"old\n").hexdigest()
    }
    assert len(calls) == 2


def _change_file(root, relative, transform):
    """Reseal outer checksums: regressions must test semantic evidence validation."""
    from contextlib import contextmanager

    @contextmanager
    def changed():
        target = root / relative
        old = target.read_bytes() if target.exists() else None
        manifest = (root / "manifest.json").read_bytes()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(transform(old))
        _seal(root)
        try:
            yield
        finally:
            if old is None:
                target.unlink()
            else:
                target.write_bytes(old)
            (root / "manifest.json").write_bytes(manifest)

    return changed()


@pytest.mark.parametrize(
    "fault",
    [
        "raw_missing",
        "prepared_resealed",
        "prompt_resealed",
        "wrong_arm",
        "checkpoint_coverage",
        "state_chain",
        "atomic_marker",
        "persistent_fault",
        "model_cuda",
        "eos_drift",
        "aggregate_cost",
        "source_tree",
    ],
)
def test_complete_gate_rejects_resealed_evidence_tampering(complete_evidence, fault):
    from src.r4_gate import validate_r4_gate

    root, gate, r2, r3, _ = complete_evidence
    relative = "runtime_lock.json"

    def transform(old):
        if fault in ("raw_missing", "prepared_resealed", "prompt_resealed", "wrong_arm"):
            rows = [json.loads(line) for line in old.decode().splitlines()]
            if fault == "raw_missing":
                rows.pop()
            elif fault == "prepared_resealed":
                rows[0]["prepared_hash"] = "0" * 64
            elif fault == "prompt_resealed":
                rows[0]["pixel_values_hash"] = "0" * 64
            else:
                rows[0]["arm"] = "X_VALID"
            for row in rows:
                row["record_hash"] = canonical_hash(
                    {k: v for k, v in row.items() if k != "record_hash"}
                )
            return ("".join(json.dumps(r) + "\n" for r in rows)).encode()
        if fault == "persistent_fault":
            return b"{}\n"
        value = json.loads(old)
        if fault == "checkpoint_coverage":
            value["checkpoints"].pop()
        elif fault == "state_chain":
            value["checkpoints"][1]["checkpoint_identity"]["prestate_hash"] = "0" * 64
        elif fault == "atomic_marker":
            value["status"] = "FAIL"
        elif fault == "model_cuda":
            value["model_audit"]["execution_kind"] = "CPU_FAKE_ADAPTER_FIXTURE"
        elif fault == "eos_drift":
            value["model_audit"]["eos_token_ids"].append(999)
        elif fault == "aggregate_cost":
            value["optimizer_steps_observed_at_least"] = 999
        elif fault == "source_tree":
            value["source"]["source_files"]["src/r4_runtime.py"] = "0" * 64
        return json.dumps(value).encode()

    if fault in ("raw_missing", "prepared_resealed", "prompt_resealed", "wrong_arm"):
        relative = "evaluation/INITIAL/step_00/N/samples.jsonl"
    elif fault in ("checkpoint_coverage", "state_chain"):
        relative = "checkpoint_manifest.json"
    elif fault == "atomic_marker":
        relative = "X_BASE/step_01/updates/attempt_0000/completed.json"
    elif fault == "persistent_fault":
        relative = "measurement_fault.json"
    elif fault == "aggregate_cost":
        relative = "runtime_profile.json"
    with _change_file(root, relative, transform), pytest.raises(ValueError):
        validate_r4_gate(root, gate, r2, r3)


def test_checkpoint_step_rejects_second_completed_attempt(complete_evidence):
    from src.r4_gate import validate_r4_gate

    root, gate, r2, r3, _ = complete_evidence
    relative = "X_BASE/step_01/updates/attempt_0001/completed.json"
    try:
        with (
            _change_file(root, relative, lambda old: b'{"status":"PASS"}\n'),
            pytest.raises(ValueError, match="multiple"),
        ):
            validate_r4_gate(root, gate, r2, r3)
    finally:
        (root / relative).parent.rmdir()


def test_warm_pointer_and_report_failure_cannot_bypass_full_gate(complete_evidence, monkeypatch):
    from src import r4_report_gate
    from src.r4_gate import validate_r4_gate

    root, gate, r2, r3, _ = complete_evidence

    def changed(old):
        value = json.loads(old)
        value["X_BASE_step64_for_R3_warm"] = next(
            e for e in value["checkpoints"] if e["arm"] == "X_VALID" and e["step"] == 64
        )
        return json.dumps(value).encode()

    with (
        _change_file(root, "checkpoint_manifest.json", changed),
        pytest.raises(ValueError, match="warm pointer"),
    ):
        validate_r4_gate(root, gate, r2, r3)

    def bad_report(*args, **kwargs):
        raise ValueError("statistical recomputation fixture failure")

    monkeypatch.setattr(r4_report_gate, "validate_r4_response_artifacts", bad_report)
    with pytest.raises(ValueError, match="statistical recomputation"):
        validate_r4_gate(root, gate, r2, r3)


def test_measured_update_loss_aggregation_cannot_be_relabelled(complete_evidence):
    from src.optimizer_fork import load_checkpoint
    from src.r4_gate import _check_update

    root, *_ = complete_evidence
    attempt = root / "X_BASE/step_01/updates/attempt_0000"
    result = json.loads((attempt / "result.json").read_text())
    runtime = json.loads((root / "runtime_lock.json").read_text())
    pre = load_checkpoint(root / "origin.pt", {**runtime["identity"], "unit": "initial_origin"})
    post = load_checkpoint(attempt / "checkpoint.pt", result["checkpoint_identity"])
    rows = [
        json.loads(line)
        for line in (root / "X_BASE/step_01/rollouts/samples.jsonl").read_text().splitlines()
    ]
    result["update"]["token_records"][0]["loss"] = 1.0
    with pytest.raises(ValueError, match="aggregate"):
        _check_update(result, rows, "X_BASE", pre, post)


def test_source_end_binding_rejects_code_changed_during_the_run():
    from src.r4_gate import _check_finished_source

    runtime = {
        "source": {"source_commit": "1" * 40, "source_files": {"src/r4_runtime.py": "a" * 64}}
    }
    status = {**runtime["source"], "status": "PASS"}
    _check_finished_source(status, runtime)
    for key, value in [("source_commit", "2" * 40), ("source_files", {})]:
        with pytest.raises(ValueError, match="execution"):
            _check_finished_source({**status, key: value}, runtime)


def test_git_source_reader_resolves_repository_paths_from_project_subdirectory(
    tmp_path, monkeypatch
):
    import hashlib
    import subprocess

    from src import r4_gate

    project = tmp_path / "ssvc_flow"
    (project / "src/nested").mkdir(parents=True)
    sources = {
        "src/example.py": b"print('committed')\n",
        "src/nested/second.py": b"VALUE = 2\n",
    }
    for name, content in sources.items():
        (project / name).write_bytes(content)
    (project / "src/notes.txt").write_text("not Python source\n")
    (tmp_path / "outside.py").write_text("not in the project source tree\n")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Gate Fixture",
            "-c",
            "user.email=gate@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "test: committed source fixture",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    # The reader must use committed objects even when checkout bytes have drifted.
    (project / "src/example.py").write_text("uncommitted checkout bytes\n")
    monkeypatch.setattr(r4_gate, "PROJECT_ROOT", project)
    assert _committed_source_reader(revision) == {
        name: hashlib.sha256(content).hexdigest() for name, content in sources.items()
    }
