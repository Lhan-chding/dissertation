"""Conditional common-format SFT and isolated native multimodal GRPO resume audit.

No entrypoint authorizes scientific preparation, response runs, or MM-DEV. All
recipes and input ordering must be frozen before constructing a training model.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import re
import tempfile
from pathlib import Path

from .vl_runtime import FULL_ATTENTION_LAYERS, QwenRuntime, gold_completion, hash_json, seed_all

OPTIMIZER_RECIPE = dict(
    type="AdamW",
    learning_rate=1e-5,
    betas=[0.9, 0.999],
    eps=1e-8,
    weight_decay=0.0,
    clip_grad_norm=1.0,
    scheduler="constant",
    warmup=0,
)
LORA_RECIPE = dict(
    rank=8,
    alpha=16,
    dropout=0.0,
    scope="qwen3_5_language_full_attention_only_q_proj_v_proj",
    layer_indices=list(FULL_ATTENTION_LAYERS),
    module_count=16,
    linear_attention_frozen=True,
    q_proj_semantics="native_full_attention_query_and_gate_projection",
    vision_encoder_frozen=True,
    projector_frozen=True,
)
BRIDGE_RECIPE = dict(
    optimizer_updates=16,
    effective_batch_sequences=8,
    microbatch_sequences=1,
    sequence_exposures=128,
    objective="mean_per_sequence_completion_NLL_then_equal_sequence_mean",
    gold_supervision_disclosed=True,
    lora=LORA_RECIPE,
    optimizer=OPTIMIZER_RECIPE,
)
ENGINE_RECIPE = dict(
    logical_steps=4,
    physical_updates=8,
    prompts_per_step=2,
    group_size=8,
    rollout_count=128,
    reward="exact_answer_only_missing_is_zero",
    group_normalization="population_std_plus_1e-8",
    clip_epsilon=0.2,
    kl_coefficient=0.0,
    reference="frozen_common_start_base_plus_starting_adapter",
    loss_normalization="completion_token_mean_then_equal_sequence_mean",
    generation_reuse=1,
    optimizer_updates_per_rollout_set=1,
    probability_distribution="raw_model_ratio_under_frozen_temperature_top_p_sampling",
    precision_comparison="exact_no_tolerance",
    optimizer=OPTIMIZER_RECIPE,
    lora=LORA_RECIPE,
)


def language_qv_modules(names, full_attention_layers=FULL_ATTENTION_LAYERS):
    pattern = re.compile(
        r"(?:model\.)?(?:language_model\.)?layers\.(\d+)\.self_attn\.(q_proj|v_proj)"
    )
    selected = [(name, pattern.fullmatch(name)) for name in names]
    selected = [(name, match) for name, match in selected if match]
    actual = {(int(match[1]), match[2]) for _, match in selected}
    expected = {(i, kind) for i in full_attention_layers for kind in ("q_proj", "v_proj")}
    if actual != expected or len(selected) != 2 * len(full_attention_layers):
        raise ValueError("Exact language q/v module enumeration failed")
    return sorted(name for name, _ in selected)


def group_advantages(rewards, epsilon=1e-8):
    if len(rewards) != 8 or any(x not in (0, 1) for x in rewards):
        raise ValueError("Engine requires eight real binary answer rewards")
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((x - mean) ** 2 for x in rewards) / len(rewards))
    return [(x - mean) / (std + epsilon) for x in rewards]


def answer_reward(raw_text, row):
    """Registered strict parser; readout correctness never contributes to reward."""
    from .contracts import operate, parse_response, rational

    values = row.get("true_values", row.get("true_values_decimal"))
    parsed = parse_response(raw_text, len(values))
    return int(
        parsed.L_A and parsed.answer == operate([rational(v) for v in values], row["operation"])
    )


def deterministic_order(rows, seed, count):
    if len(rows) < count or len({r["question_id"] for r in rows}) != len(rows):
        raise ValueError("Registered unique pool is too small")
    ordered = sorted(rows, key=lambda row: row["question_id"])
    random.Random(seed).shuffle(ordered)
    return ordered[:count]


def state_hash(value):
    """Content hash for tensors, nested optimizer states, and Python/NumPy RNG."""
    digest = hashlib.sha256()

    def visit(obj):
        if hasattr(obj, "detach"):
            import torch

            t = obj.detach().cpu().contiguous()
            digest.update(str((str(t.dtype), tuple(t.shape))).encode())
            digest.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif hasattr(obj, "dtype") and hasattr(obj, "tobytes"):
            digest.update(str((str(obj.dtype), obj.shape)).encode())
            digest.update(obj.tobytes())
        elif isinstance(obj, dict):
            digest.update(b"dict")
            for key in sorted(obj, key=lambda key: (type(key).__name__, str(key))):
                visit(key)
                visit(obj[key])
        elif isinstance(obj, (tuple, list)):
            digest.update(type(obj).__name__.encode())
            for item in obj:
                visit(item)
        else:
            digest.update((type(obj).__name__ + ":" + repr(obj)).encode())

    visit(value)
    return digest.hexdigest()


def capture_rng():
    import numpy as np
    import torch

    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        cpu=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def restore_rng(state):
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def trainable_state(model):
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def frozen_hash(model):
    return state_hash(
        {
            name: param.detach()
            for name, param in model.named_parameters()
            if not param.requires_grad
        }
    )


def _cpu_tree(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def save_checkpoint(path, runtime, optimizer, scheduler, step, reference, *, stream_hash):
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = dict(
        committed_logical_step=step,
        parameters=trainable_state(runtime.model),
        optimizer=_cpu_tree(optimizer.state_dict()),
        scheduler=scheduler.state_dict(),
        rng=capture_rng(),
        reference=_cpu_tree(reference),
        input_stream_hash=stream_hash,
    )
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=path.parent)
    os.close(fd)
    try:
        with open(temporary, "wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return state


def load_checkpoint(path, runtime, optimizer, scheduler, stream_hash):
    import torch

    # Only load this run's locally created full-state checkpoint, never untrusted pickle.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["input_stream_hash"] != stream_hash or state["committed_logical_step"] != 2:
        raise ValueError("Resume checkpoint stream or committed step mismatch")
    actual = {name: p for name, p in runtime.model.named_parameters() if p.requires_grad}
    if actual.keys() != state["parameters"].keys():
        raise ValueError("Resume trainable parameter set changed")
    with torch.no_grad():
        for name, param in actual.items():
            param.copy_(state["parameters"][name].to(param.device))
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    restore_rng(state["rng"])
    return state


def configure_training(runtime):
    import torch
    from peft import LoraConfig, get_peft_model

    model = runtime.model
    layer_types = model.config.text_config.layer_types
    actual_full = tuple(i for i, kind in enumerate(layer_types) if kind == "full_attention")
    if actual_full != FULL_ATTENTION_LAYERS or len(layer_types) != 32:
        raise ValueError("Training model hybrid attention layout changed")
    if not hasattr(model, "peft_config"):
        modules = language_qv_modules([name for name, _ in model.named_modules()])
        for param in model.parameters():
            param.requires_grad_(False)
        model = get_peft_model(
            model,
            LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
                target_modules=modules,
                task_type="CAUSAL_LM",
            ),
        )
        runtime.model = model
    else:
        config = model.peft_config[model.active_adapter]
        if config.r != 8 or config.lora_alpha != 16 or config.lora_dropout != 0:
            raise ValueError("Existing adapter differs from registered default recipe")
        modules = sorted(config.target_modules)
        language_qv_modules(modules)
        for name, param in model.named_parameters():
            param.requires_grad_("lora_" in name)
    trainable_names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "lora_" not in name or any(
                x in name.lower() for x in ("visual", "vision", "merger", "projector")
            ):
                raise RuntimeError("Non-language parameter unexpectedly trainable")
            param.data = param.data.float()
            trainable_names.append(name)
    if not trainable_names:
        raise ValueError("No trainable language adapters")
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    model.eval()  # Gradients remain enabled; explicit dropout=0 for exact engineering repeat.
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return (
        optimizer,
        scheduler,
        dict(
            target_modules=modules,
            trainable_names=trainable_names,
            trainable_dtype="float32",
            model_forward_mode="eval_dropout_zero_grad_enabled",
        ),
    )


def _runtime(root, freeze, adapter, common=None):
    from .execution import BudgetLedger

    ledger = BudgetLedger(root)
    runtime = QwenRuntime(
        freeze["model_path"],
        adapter_path=adapter,
        dtype=freeze.get("dtype", "bfloat16"),
        attention_backend=freeze.get("attention_backend", "eager"),
        account=lambda kind, count, metadata: ledger.reserve(kind, count, metadata),
    )
    runtime.verify_identity(freeze, common)
    return runtime


def _training_step(runtime, optimizer, scheduler, losses, *, identity):
    import torch

    runtime.reserve("physical_optimizer_updates", 1, **identity)
    optimizer.zero_grad(set_to_none=True)
    values = []
    for loss in losses:
        value = loss()
        if not bool(torch.isfinite(value)):
            raise RuntimeError("Nonfinite training loss; reserved physical cost retained")
        values.append(float(value.detach()))
        (value / len(losses)).backward()
    norm = torch.nn.utils.clip_grad_norm_(
        [p for p in runtime.model.parameters() if p.requires_grad], 1.0
    )
    if not bool(torch.isfinite(norm)):
        raise RuntimeError("Nonfinite gradients; optimizer not applied")
    optimizer.step()
    scheduler.step()
    return dict(sequence_losses=values, loss=sum(values) / len(values), gradient_norm=float(norm))


def run_bridge(run_root):
    from .execution import (
        append_jsonl,
        atomic_json,
        exclusive_json,
        read_json,
        read_jsonl,
        sha256_file,
        verify_gate,
    )

    root = Path(run_root)
    freeze = verify_gate(root, "BRIDGE")
    from .allocations import require_allocation

    require_allocation(root, "BRIDGE")
    trigger = read_json(root / "manifests/BRIDGE_TRIGGER.json")
    if trigger.get("recipe") != BRIDGE_RECIPE or type(trigger.get("seed")) is not int:
        raise PermissionError("Exact bridge recipe and seed must be frozen in trigger")
    exclusive_json(
        root / "engineering/BRIDGE_STARTED.json",
        dict(status="RUNNING", trigger_hash=sha256_file(root / "manifests/BRIDGE_TRIGGER.json")),
    )
    seed_all(trigger["seed"])
    runtime = _runtime(root, freeze, freeze.get("adapter_path"))
    optimizer, scheduler, module_receipt = configure_training(runtime)
    initial_frozen = frozen_hash(runtime.model)
    rows = [r for r in read_jsonl(root / "data/questions.jsonl") if r["split"] == "BRIDGE_TRAIN"]
    stream = deterministic_order(rows, trigger["seed"], 128)
    atomic_json(
        root / "engineering/BRIDGE_INPUT_STREAM.json",
        dict(
            seed=trigger["seed"], question_ids=[r["question_id"] for r in stream], **module_receipt
        ),
    )
    for step in range(16):
        selected = stream[step * 8 : (step + 1) * 8]
        losses = []
        for row in selected:

            def loss(row=row):
                prepared = runtime.prepare(row, root)
                tokens = runtime.encode_completion(gold_completion(row))
                scored = runtime.sequence_forward(
                    prepared, tokens, purpose="bridge_gold_completion", grad=True
                )
                return -scored["logprobs"].mean()

            losses.append(loss)
        result = _training_step(
            runtime,
            optimizer,
            scheduler,
            losses,
            identity=dict(stage="BRIDGE", logical_step=step + 1),
        )
        append_jsonl(root / "engineering/BRIDGE_STEPS.jsonl", dict(logical_step=step + 1, **result))
    if frozen_hash(runtime.model) != initial_frozen:
        raise RuntimeError("Frozen vision/projector/base parameters changed")
    bridge_hash = state_hash(trainable_state(runtime.model))
    protocol_hash = freeze.get("protocol_hash", hash_json(freeze["chat_template_hash"]))
    endpoint = root / "engineering" / f"COMMON_START_MM_{protocol_hash[:12]}_{bridge_hash[:12]}"
    endpoint.mkdir(exist_ok=False)
    runtime.model.save_pretrained(endpoint, safe_serialization=True)
    hashes = {p.name: sha256_file(p) for p in sorted(endpoint.iterdir()) if p.is_file()}
    receipt = dict(
        status="COMPLETED",
        executed=True,
        physical_updates=16,
        sequence_exposures=128,
        gold_supervision_disclosed=True,
        common_start_is_untouched_base=False,
        adapter_path=str(endpoint),
        adapter_hash=hash_json(hashes),
        file_hashes=hashes,
        trainable_parameter_hash=bridge_hash,
        frozen_parameters_unchanged=True,
        recipe=BRIDGE_RECIPE,
        module_audit=module_receipt,
        pre_freeze_hash=sha256_file(root / "manifests/PRE_INFERENCE_FREEZE.json"),
        trigger_hash=sha256_file(root / "manifests/BRIDGE_TRIGGER.json"),
    )
    exclusive_json(root / "engineering/BRIDGE_RECEIPT.json", receipt)
    from .execution import create_common_start

    create_common_start(root)
    return receipt


def engine_segment(run_root, branch, start, stop):
    """Called in a fresh Python process for each continuous/interrupt/resume segment."""
    import torch

    from .execution import (
        append_jsonl,
        atomic_json,
        exclusive_json,
        read_json,
        read_jsonl,
        sha256_file,
        verify_gate,
    )

    root = Path(run_root)
    freeze = verify_gate(root, "ENGINE")
    from .allocations import require_allocation

    require_allocation(root, "ENGINE")
    engine = read_json(root / "manifests/ENGINE_TEST_FREEZE.json")
    if engine.get("recipe") != ENGINE_RECIPE or type(engine.get("seed")) is not int:
        raise PermissionError("Exact native GRPO definition and seed must be frozen")
    if (branch, start, stop) not in {("continuous", 0, 4), ("resumed", 0, 2), ("resumed", 2, 4)}:
        raise PermissionError("Only registered four-step recovery segments are permitted")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python starts")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    seed_all(engine["seed"])
    destination = root / "engineering/engine" / branch
    exclusive_json(
        destination / f"SEGMENT_{start}_{stop}_STARTED.json",
        dict(
            pid=os.getpid(),
            start=start,
            stop=stop,
            freeze_hash=sha256_file(root / "manifests/ENGINE_TEST_FREEZE.json"),
        ),
    )
    common = read_json(root / "manifests/COMMON_START.json")
    runtime = _runtime(root, freeze, common.get("adapter_path"), common)
    optimizer, scheduler, modules = configure_training(runtime)
    initial_frozen = frozen_hash(runtime.model)
    reference = dict(parameters=trainable_state(runtime.model), base_hash=initial_frozen)
    rows = [r for r in read_jsonl(root / "data/questions.jsonl") if r["split"] == "ENGINE_TEST"]
    stream = deterministic_order(rows, engine["seed"], 8)
    stream_hash = hash_json([r["question_id"] for r in stream])
    if start:
        checkpoint = load_checkpoint(
            destination / "checkpoint_2.pt", runtime, optimizer, scheduler, stream_hash
        )
        reference = checkpoint["reference"]
        if reference["base_hash"] != initial_frozen:
            raise ValueError("Resume frozen reference base changed")
    else:
        seed_all(engine["seed"])
    reference_hash = state_hash(reference)
    atomic_json(
        destination / f"SEGMENT_{start}_{stop}_IDENTITY.json",
        dict(reference_hash=reference_hash, stream_hash=stream_hash, **modules),
    )
    for logical_step in range(start, stop):
        rollout_group = []
        for slot, row in enumerate(stream[logical_step * 2 : (logical_step + 1) * 2]):
            group = []
            for sample in range(8):
                # Generated from restored global RNG, independently regenerated on each branch.
                seed = random.randrange(2**63 - 1)
                request_id = hash_json(["ENGINE", branch, logical_step, slot, sample])
                request = dict(
                    stage="ENGINE",
                    branch=branch,
                    logical_step=logical_step + 1,
                    slot=slot,
                    sample_index=sample,
                    request_id=request_id,
                    seed=seed,
                    question_id=row["question_id"],
                    image_sha256=row["image_sha256"],
                    model_hash=common["model_hash"],
                    processor_hash=freeze["processor_hash"],
                )
                append_jsonl(root / "raw/REQUESTS.jsonl", {**request, "status": "REQUESTED"})
                raw = runtime.generate(
                    row,
                    root,
                    seed,
                    score_fields=False,
                    on_completion=lambda record, request=request: append_jsonl(
                        destination / "RAW_COMPLETIONS.jsonl", {**request, **record}
                    ),
                )
                reward = answer_reward(raw["raw_text"], row)
                prepared = runtime.prepare(row, root)
                old = runtime.sequence_forward(
                    prepared, raw["tokens"], purpose="engine_behavior_logprob"
                )["logprobs"]
                group.append(dict(row=row, tokens=raw["tokens"], old=old.detach(), reward=reward))
                append_jsonl(destination / "ROLLOUTS.jsonl", {**request, **raw, "reward": reward})
            advantages = group_advantages([g["reward"] for g in group])
            for item, advantage in zip(group, advantages, strict=True):
                item["advantage"] = advantage
                rollout_group.append(item)
        losses = []
        for item in rollout_group:

            def loss(item=item):
                current = runtime.sequence_forward(
                    runtime.prepare(item["row"], root),
                    item["tokens"],
                    purpose="engine_policy_gradient",
                    grad=True,
                )["logprobs"]
                ratio = (current - item["old"]).exp()
                unclipped = ratio * item["advantage"]
                clipped = ratio.clamp(0.8, 1.2) * item["advantage"]
                return -torch.minimum(unclipped, clipped).mean()

            losses.append(loss)
        result = _training_step(
            runtime,
            optimizer,
            scheduler,
            losses,
            identity=dict(stage="ENGINE", branch=branch, logical_step=logical_step + 1),
        )
        rewards = [item["reward"] for item in rollout_group]
        append_jsonl(
            destination / "STEPS.jsonl",
            dict(
                logical_step=logical_step + 1,
                question_ids=[item["row"]["question_id"] for item in rollout_group],
                rewards=rewards,
                zero_contrast_groups=[len(set(rewards[i : i + 8])) == 1 for i in (0, 8)],
                **result,
            ),
        )
    if state_hash(reference) != reference_hash or frozen_hash(runtime.model) != initial_frozen:
        raise RuntimeError("Reference or frozen base changed")
    state = save_checkpoint(
        destination / f"checkpoint_{stop}.pt",
        runtime,
        optimizer,
        scheduler,
        stop,
        reference,
        stream_hash=stream_hash,
    )
    atomic_json(
        destination / f"SEGMENT_{start}_{stop}_COMPLETE.json",
        dict(
            status="COMPLETED",
            physical_updates=stop - start,
            final_state_hash=state_hash(state),
            reference_unchanged=True,
            frozen_base_unchanged=True,
            image_calls=runtime.image_calls,
        ),
    )


def compare_engine(run_root):
    import torch

    from .execution import atomic_json, read_jsonl

    root = Path(run_root)
    states = [
        torch.load(
            root / "engineering/engine" / branch / "checkpoint_4.pt",
            map_location="cpu",
            weights_only=False,
        )
        for branch in ("continuous", "resumed")
    ]
    comparisons = {
        key: state_hash(states[0][key]) == state_hash(states[1][key]) for key in states[0]
    }
    rollouts = [
        read_jsonl(root / "engineering/engine" / branch / "ROLLOUTS.jsonl")
        for branch in ("continuous", "resumed")
    ]
    keys = ("logical_step", "slot", "sample_index", "seed", "question_id", "tokens", "reward")
    comparisons["raw_tokens_slots_seeds_rewards"] = [
        {key: row[key] for key in keys} for row in rollouts[0]
    ] == [{key: row[key] for key in keys} for row in rollouts[1]]
    comparisons["rollout_count"] = len(rollouts[0]) == len(rollouts[1]) == 64
    comparisons["image_calls"] = all(
        row["image_routing"]["generation_vision_forward_calls"] > 0
        for branch in rollouts
        for row in branch
    )
    receipt = dict(
        status="PASS" if all(comparisons.values()) else "ENGINE_REPRO_NOT_READY",
        executed=True,
        comparisons=comparisons,
        physical_updates=8,
        rollouts=128,
        exact_comparison=True,
        tolerance=None,
        engine_model_replaces_common_start=False,
        scientific_response_evaluation=False,
        recipe=ENGINE_RECIPE,
    )
    atomic_json(root / "engineering/ENGINE_COMPARE.json", receipt)
    return receipt
