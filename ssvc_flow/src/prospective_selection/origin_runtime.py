"""New independent source lineages from one literal-seed-17 LoRA initialization."""

from __future__ import annotations

from .branch_runtime import CommonScheduleSampler, _restore_checked, run_training


def run_source(
    runtime,
    *,
    prompts,
    schedule,
    lineage_id,
    source_recipe,
    steps,
    out,
    experiment_seed=20260924,
    initial_state=None,
    resume=False,
    fixture=False,
):
    if source_recipe not in ("R0", "R1") or steps not in (2, 32, 96):
        raise ValueError("Registered R0/R1 source and 32/96 steps (or 2-step smoke) required")
    if not fixture and (len(prompts) != 384 or len(schedule["train_steps"]) != 96):
        raise ValueError("Production source requires the registered 384-prompt/96-step pool")
    initial_state = initial_state if initial_state is not None else runtime["initial_state"]
    if initial_state["optimizer"]["state"]:
        raise ValueError("New source must start with empty Adam, not a historical trained origin")
    _restore_checked(runtime, initial_state)
    runtime["sampler"].clear()
    runtime["sampler"].update(
        role="source",
        position=0,
        step=0,
        schedule_id=schedule["schedule_id"],
        schedule_hash=schedule["schedule_hash"],
    )
    state = runtime["state"].capture(
        {
            "seed": lineage_id,
            "lineage_id": lineage_id,
            "source_recipe": source_recipe,
            "checkpoint_step": 0,
            "lora_initialization_seed": 17,
        }
    )
    sampler = CommonScheduleSampler(
        runtime,
        prompts,
        schedule,
        lineage_id=lineage_id,
        repeat=0,
        experiment_seed=experiment_seed,
        origin_id=f"source_{lineage_id}",
        recipe=source_recipe,
        role="source",
    )
    return run_training(
        runtime,
        state,
        sampler,
        recipe=source_recipe,
        steps=steps,
        out=out,
        identity={
            "lineage_id": lineage_id,
            "source_recipe": source_recipe,
            "role": "source",
            "schedule_id": schedule["schedule_id"],
        },
        checkpoints=tuple(s for s in (0, 24, 32, 88, 96) if s <= steps),
        resume=resume,
        fixture=fixture,
    )


def run_smoke(runtime, *, prompts, schedule, out, experiment_seed=20260924, fixture=False):
    """Two recipes x two real updates; parity, evaluation isolation and restoration."""
    from pathlib import Path

    from ..modeling_v3.vlm_observation import _logps, _parity_values
    from ..modeling_v4 import gpu_collect as gpu
    from ..optimizer_fork import state_hash
    from .branch_runtime import make_backend
    from .evaluation import isolated_evaluation
    from .sampler import training_seed

    root = Path(out)
    by_id = {p["prompt_id"]: p for p in prompts}
    smoke_prompts = [by_id[pid] for step in schedule["train_steps"][:2] for pid in step]
    interfaces = {p["interface"] for p in smoke_prompts}
    symbolic = interfaces & {"SYM_CUE", "SYMBOLIC_FRESH"}
    if len(symbolic) != 1 or interfaces != symbolic | {"IMAGE_CUE_FRESH"}:
        raise ValueError("GPU smoke must include image and symbolic training inputs")
    parity_prompts = [
        next(p for p in smoke_prompts if p["interface"] == interface)
        for interface in (next(iter(symbolic)), "IMAGE_CUE_FRESH")
    ]
    parity_prompts += [p for p in smoke_prompts if p not in parity_prompts][:2]
    initial = runtime["initial_state"]
    _restore_checked(runtime, initial)
    before = state_hash(runtime["state"].capture(initial["metadata"]))
    backend = make_backend(runtime)
    backend.current_fingerprint = "SMOKE_INITIAL_POLICY"
    parity_rows = []

    def evaluate_parity():
        for index, prompt in enumerate(parity_prompts):
            raw = backend.generate(
                prompt,
                seed=training_seed(
                    experiment_seed=experiment_seed,
                    lineage=60000,
                    repeat=0,
                    step=0,
                    prompt_id=prompt["prompt_id"],
                    draw=index,
                ),
            )
            scores = _logps(
                runtime["adapter"].logprobs(
                    backend._prepared(prompt), raw["token_ids"], require_grad=False
                ),
                len(raw["token_ids"]),
            )
            behavior = raw["behavior_token_logprobs"]
            receipt = _parity_values(
                [[abs(a - b) for a, b in zip(scores, behavior, strict=True)]],
                runtime["parity_tolerances"],
                sequence_errors=[abs(sum(scores) - sum(behavior))],
            )
            if not receipt["passed"]:
                raise ValueError("Smoke behavior/scorer parity failed")
            parity_rows.append({"sample": raw, "score": scores, "parity": receipt})
        return {"rows": parity_rows}

    parity_path = root / "PARITY.json"
    if parity_path.exists():
        # Real generation rows include elapsed times; do not regenerate an
        # already validated smoke receipt merely because a later update failed.
        parity = gpu._read(parity_path)
        if (
            parity.get("initial_state_hash") != before
            or parity.get("runtime_identity") != runtime["identity"]
            or not all(r["parity"]["passed"] for r in parity["rows"])
        ):
            raise ValueError("Existing smoke parity belongs to a different initial policy")
    else:
        parity = {
            **isolated_evaluation(runtime, evaluate_parity),
            "initial_state_hash": before,
            "runtime_identity": runtime["identity"],
        }
    after = state_hash(runtime["state"].capture(initial["metadata"]))
    if before != after:
        raise RuntimeError("Smoke evaluation changed complete training state")
    root.mkdir(parents=True, exist_ok=True)
    gpu._publish(parity_path, parity)
    results = {}
    for recipe in ("R0", "R1"):
        results[recipe] = run_source(
            runtime,
            prompts=prompts,
            schedule=schedule,
            lineage_id=60000,
            source_recipe=recipe,
            steps=2,
            out=root / recipe,
            experiment_seed=experiment_seed,
            initial_state=initial,
            resume=(root / recipe / "MANIFEST.json").exists(),
            fixture=fixture,
        )
        checkpoint = results[recipe]["checkpoint"]
        state = runtime["checkpoint_cache"].load(checkpoint)
        _restore_checked(runtime, initial)
        _restore_checked(runtime, state)
    _restore_checked(runtime, initial)
    result = {
        "status": "CPU_FIXTURE_COMPLETE" if fixture else "GPU_SMOKE_COMPLETE",
        "recipes": results,
        "optimizer_updates": 4,
        "evaluation_complete_state_unchanged": before == after,
        "behavior_score_parity": True,
        "full_checkpoint_restore": True,
        "execution_kind": runtime["identity"]["execution_kind"],
    }
    gpu._publish(root / "COMPLETE.json", result)
    return result
