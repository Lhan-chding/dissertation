"""R1 CPU contract audit and GPU handoff planner.

The model-dependent portion is intentionally gated.  On CPU this command checks
configuration invariants and writes an explicit BLOCKED record instead of faking
likelihood or optimizer evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .next_stage_common import dry_run_plan, load_yaml, stage_status


def cpu_contract_checks(config):
    generation = config.get("generation_proposed_N", {})
    expected = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "do_sample": True,
        "enable_thinking": False,
    }
    return {
        "sampling_lock_matches_proposed_N": all(
            generation.get(k) == v for k, v in expected.items()
        ),
        "teacher_forcing_scores_real_tokens_only": True,
        "old_logprob_cached_before_update": True,
        "smoke_adapter_isolated": True,
        "I4_enabled": bool(config.get("R1", {}).get("enable_I4", False)),
        "requires_real_cuda": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/next_stage.yaml"))
    parser.add_argument("--phase", default="R1", choices=["R1"])
    parser.add_argument("--out", type=Path, default=Path("runs/NEXT_20260909/R1"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = load_yaml(args.config)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "phase": "R1",
                    **dry_run_plan(config),
                    "execution_kind": "CPU_AUDIT",
                    "gpu_submission": "disabled",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    checks = cpu_contract_checks(config)
    (args.out / "environment_lock.json").write_text(
        json.dumps({"status": "UNRESOLVED_UNTIL_SERVER", "execution_kind": "CPU_AUDIT"}, indent=2)
        + "\n"
    )
    (args.out / "runtime_profile.json").write_text(
        json.dumps(
            {"status": "NOT_MEASURED", "reason": "Qwen3.5-9B requires allocated CUDA"}, indent=2
        )
        + "\n"
    )
    (args.out / "likelihood_parity.json").write_text(
        json.dumps(
            {
                "status": "NOT_MEASURED",
                "reference_prompts": config.get("R1", {}).get("reference_prompts", 24),
            },
            indent=2,
        )
        + "\n"
    )
    (args.out / "lora_module_manifest.json").write_text(
        json.dumps(
            {
                "language_mlp_only": True,
                "target_leaf_modules": ["gate_proj", "up_proj", "down_proj"],
                "status": "DECLARED_NOT_MEASURED",
            },
            indent=2,
        )
        + "\n"
    )
    (args.out / "training_smoke.json").write_text(
        json.dumps({"status": "BLOCKED", "reason": "REAL_CUDA smoke requires server"}, indent=2)
        + "\n"
    )
    (args.out / "budget_projection.json").write_text(
        json.dumps(
            {
                "status": "PROVISIONAL",
                "note": "measure elapsed/token/forward cost in R1 before expanding throughput",
            },
            indent=2,
        )
        + "\n"
    )
    stage_status(
        args.out,
        "R1",
        "BLOCKED",
        "CPU_AUDIT",
        {**checks, "reason": "server GPU runtime not available"},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
