"""Forced saved-token cache replay, never a new scientific sample or update.

Two bounded trials per precision: original G8 and original row 3, through token
162 inclusive. Observe raw model probabilities before forcing each saved token.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import traceback
from pathlib import Path

_helper_spec = importlib.util.spec_from_file_location(
    "srf5_probability_diagnostic_helpers", Path(__file__).with_name("diagnose_probability.py")
)
helper = importlib.util.module_from_spec(_helper_spec)
_helper_spec.loader.exec_module(helper)
TOKENS = 163


def tensor_summary(value, *, values=False):
    if value is None:
        return None
    if hasattr(value, "shape"):
        result = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
        if not value.is_floating_point() and (values or value.numel() <= 128):
            result["values"] = value.detach().cpu().tolist()
        return result
    result = {"type": type(value).__name__}
    if hasattr(value, "get_seq_length"):
        result["sequence_length"] = int(value.get_seq_length())
    return result


class ForcedReplay:
    """Observe original scores; force saved IDs using a fresh score tensor."""

    def __init__(self, torch, records, prompt_length, *, reserve=None):
        self.torch, self.records, self.prompt_length = torch, records, prompt_length
        self.reserve = reserve
        self.forwards, self.probabilities, self.score_checks = [], [[] for _ in records], []
        self.raw = None

    def before_forward(self, module, args, kwargs):
        if args:
            raise PermissionError("Expected native generation to call forward with keyword inputs")
        index = len(self.forwards)
        if index >= TOKENS:
            raise PermissionError("Cached replay exceeded its fixed diagnostic budget")
        snapshot = {
            "index": index,
            "phase": "prefill" if index == 0 else "decode",
            "inputs": {
                key: tensor_summary(kwargs.get(key), values=index in (0, 151, 162))
                for key in (
                    "input_ids",
                    "attention_mask",
                    "position_ids",
                    "cache_position",
                    "mm_token_type_ids",
                    "image_grid_thw",
                    "pixel_values",
                    "past_key_values",
                )
            },
            "use_cache": kwargs.get("use_cache"),
        }
        self.forwards.append(snapshot)
        if self.reserve:
            self.reserve(
                "extra_forward_sequences",
                len(self.records),
                purpose="FORCED_TOKEN_REPLAY_ONLY",
                token_position=index,
                batch_size=len(self.records),
            )

    def before_language_forward(self, module, args, kwargs):
        if not self.forwards:
            raise PermissionError("Language model forward lacks an enclosing native forward")
        index = len(self.forwards) - 1
        self.forwards[-1]["language_model_inputs"] = {
            key: tensor_summary(kwargs.get(key), values=index in (0, 151, 162))
            for key in ("position_ids", "cache_position", "attention_mask", "inputs_embeds")
        }
        self.forwards[-1]["language_position_ids_exposed"] = kwargs.get("position_ids") is not None

    def after_forward(self, module, args, kwargs, output):
        if not hasattr(output, "logits") or output.logits.ndim != 3:
            raise PermissionError("Native forward did not expose three-dimensional raw logits")
        self.raw = output.logits[:, -1, :].detach().float().clone()
        self.forwards[-1]["output_logits"] = tensor_summary(output.logits)
        self.forwards[-1]["output_cache"] = tensor_summary(getattr(output, "past_key_values", None))

    def __call__(self, input_ids, scores):
        index = input_ids.shape[-1] - self.prompt_length
        if index != len(self.score_checks) or not 0 <= index < TOKENS:
            raise PermissionError("Generation cursor differs from saved forced-token replay")
        if len(self.forwards) != index + 1 or self.raw is None or self.raw.shape != scores.shape:
            raise PermissionError("Raw model forward and logits processor are not one-to-one")
        if input_ids.shape[0] != len(self.records):
            raise PermissionError("Replay batch size changed")
        expected = input_ids.new_tensor([row["tokens"][:index] for row in self.records])
        if not self.torch.equal(input_ids[:, self.prompt_length :], expected):
            raise PermissionError("Forced replay prefix departed from the original token IDs")
        identical = bool(self.torch.equal(self.raw, scores.float()))
        self.score_checks.append({"position": index, "raw_logits_scores_equal": identical})
        if not identical:
            raise PermissionError(
                "A pre-existing logits processor changed the observed distribution"
            )
        forced = input_ids.new_tensor([row["tokens"][index] for row in self.records])
        logps = self.raw.log_softmax(-1).gather(-1, forced[:, None]).squeeze(-1)
        if not bool(self.torch.isfinite(logps).all()):
            raise FloatingPointError("Nonfinite original forced-path token logprob")
        for result, logp in zip(self.probabilities, logps.detach().cpu().tolist(), strict=True):
            result.append(logp)
        # Never mutate scores: transformers may retain an alias to raw logits.
        selected = self.torch.full_like(scores, float("-inf"))
        selected.scatter_(1, forced[:, None], 0.0)
        return selected


def replay_config(saved):
    """Disable only random selection/output retention; raw-score identity is checked live."""
    from transformers import GenerationConfig

    config = GenerationConfig.from_dict(copy.deepcopy(saved))
    if config.temperature != 1 or config.top_p != 1 or config.top_k != 0:
        raise PermissionError("Original distribution has nontrivial sampling transforms")
    if getattr(config, "forced_eos_token_id", None) is not None:
        raise PermissionError("Length-dependent forced EOS prevents a short faithful replay")
    if getattr(config, "max_time", None) is not None or getattr(config, "stop_strings", None):
        raise PermissionError("Unregistered early stop in original generation config")
    if config.use_cache is not True:
        raise PermissionError("Original generation did not explicitly use the native cache")
    config.do_sample = False
    config.max_new_tokens = TOKENS
    config.num_return_sequences = 1
    config.return_dict_in_generate = True
    config.output_logits = False
    config.output_scores = False
    return config


def run(args):
    root = helper.output_path(args.root, args.output)
    if "PROBABILITY_CACHE_226283" not in args.output.relative_to(root).parts:
        raise PermissionError("Cache replay requires its dedicated diagnostic directory")
    with args.output.with_suffix(".started.json").open("x") as stream:
        json.dump({"at": helper.now(), "pid": os.getpid(), "precision": args.mode}, stream)
    result = {
        "status": "RUNNING",
        "kind": "FORCED_TOKEN_REPLAY_ONLY",
        "started_at": helper.now(),
        "precision": args.mode,
        "run_id": helper.RUN_ID,
        "checkpoint_step": 6,
        "token_prefix_length": TOKENS,
        "new_scientific_samples": 0,
        "optimizer_updates": 0,
        "gate_decision": "NOT_AUTHORIZED_BY_DIAGNOSTIC",
        "trials": [],
    }
    helper.save(args.output, result)
    try:
        import torch
        from transformers import LogitsProcessorList

        import sr_f5.pilot as pilot
        from mm_dev.runtime import actual_cuda_identity
        from sr_f4.training import load_checkpoint
        from sr_f5.runtime import load
        from sr_f12.runtime import _repeat_prompt

        frozen = helper.read(root / "G3/FROZEN.json")
        if Path(pilot.__file__).resolve().parents[2] != args.code.resolve(strict=True):
            raise PermissionError("Imported science differs from requested immutable deployment")
        if (
            frozen["code_revision"] != args.revision
            or pilot.source_identity() != frozen["source_identity"]
        ):
            raise PermissionError("Scientific revision/source differs from frozen G3")
        result.update(
            science_revision=args.revision,
            science_code=str(args.code),
            hardware=actual_cuda_identity(),
        )
        lane = root / "G3/RUNS/b1/SHUF/primary"
        if (lane / "updates/007.json").exists() or (
            lane / "checkpoints/step-007/COMMITTED.json"
        ).exists():
            raise PermissionError("Failed step has subsequently been committed")
        records = helper.select_records(helper.read(lane / "rollouts/007/06.json"))
        if any(len(row["tokens"]) < TOKENS for row in records):
            raise PermissionError("A saved row ended before the diagnostic prefix")
        identity = helper.read(lane / "IDENTITY.json")
        checkpoint = lane / "checkpoints/step-006"
        if helper.read(checkpoint / "COMMITTED.json")["identity"] != identity:
            raise PermissionError("Checkpoint marker identity differs")
        pilot.checkpoint_header(checkpoint / "state.pt", identity, 6)
        task = f"LH-G3-CACHE-{args.mode}-{os.environ.get('SLURM_JOB_ID', 'local')}"
        runtime, optimizer, scheduler = load(root, "b1", task, training=True)
        step, cursor = load_checkpoint(
            checkpoint / "state.pt", runtime, optimizer, scheduler, identity=identity
        )
        if step != 6 or cursor != {"next_step": 7, "committed_log_step": 6}:
            raise PermissionError("Unexpected checkpoint step/cursor")
        runtime.identity.update(run_id=helper.RUN_ID, step=6, diagnostic_only=True)
        originals = {}
        for line in (root / "NUMERICAL_TOKEN_AUDIT.jsonl").open():
            audit = json.loads(line)
            if (
                audit.get("run_id") == helper.RUN_ID
                and audit.get("step") == 7
                and audit.get("status") == "UNCOMMITTED_FAILURE"
            ):
                originals = {row["row_id"]: row for row in audit["probability_vectors"]}
        for row in records:
            old = originals[row["row_id"]]
            if old["tokens"] != row["tokens"] or old["sampler_logprobs"] != row["sampler_logprobs"]:
                raise PermissionError(
                    "Original failed audit differs from saved tokens/probabilities"
                )
        prepared = runtime.prepare_input(
            records[0]["model_input"], root, protocol=records[0]["protocol"]
        )
        result["prompt_checks"] = [helper.verify_prompt(prepared, row) for row in records]
        if args.mode == "fp32":
            runtime.model.float()
            prepared = dict(
                prepared,
                inputs={
                    key: value.float() if value.is_floating_point() else value
                    for key, value in prepared["inputs"].items()
                },
            )
        result["parameter_dtypes"] = sorted({str(p.dtype) for p in runtime.model.parameters()})
        base = runtime.model.get_base_model()
        language_model = getattr(getattr(base, "model", None), "language_model", None)
        if language_model is None:
            raise PermissionError("Native language model hook target missing")
        for name, selected in [("G8", records), ("single_row3", [records[3]])]:
            inputs = _repeat_prompt(prepared["inputs"], len(selected))
            prompt_length = inputs["input_ids"].shape[-1]
            config = replay_config(selected[0]["sampling_parameters"])
            if any(
                row["sampling_parameters"] != selected[0]["sampling_parameters"] for row in selected
            ):
                raise PermissionError("Original group generation configurations differ")
            eos = config.eos_token_id
            eos = set(eos if isinstance(eos, list) else ([] if eos is None else [eos]))
            if any(eos.intersection(row["tokens"][:TOKENS]) for row in selected):
                raise PermissionError("Original saved prefix contains an early EOS")
            runtime.model.eval()
            for module in (base, getattr(base, "model", None)):
                if module is not None and hasattr(module, "rope_deltas"):
                    module.rope_deltas = None
            observer = ForcedReplay(torch, selected, prompt_length, reserve=runtime.reserve)
            trial = {
                "name": name,
                "batch_size": len(selected),
                "status": "RUNNING",
                "original_sampling_config": selected[0]["sampling_parameters"],
                "diagnostic_generation_config": config.to_dict(),
                "rows": [],
            }
            result["trials"].append(trial)
            helper.save(args.output, result)
            before = base.register_forward_pre_hook(observer.before_forward, with_kwargs=True)
            after = base.register_forward_hook(observer.after_forward, with_kwargs=True)
            language_hook = language_model.register_forward_pre_hook(
                observer.before_language_forward, with_kwargs=True
            )
            try:
                with torch.no_grad():
                    generated = runtime.model.generate(
                        **inputs,
                        generation_config=config,
                        logits_processor=LogitsProcessorList([observer]),
                    )
                wanted = inputs["input_ids"].new_tensor(
                    [row["tokens"][:TOKENS] for row in selected]
                )
                if not torch.equal(generated.sequences[:, prompt_length:], wanted):
                    raise PermissionError(
                        "Returned replay sequence differs from exact saved prefix"
                    )
                if len(observer.forwards) != TOKENS or len(observer.score_checks) != TOKENS:
                    raise PermissionError("Incomplete cached diagnostic replay")
                for row, logps in zip(selected, observer.probabilities, strict=True):
                    trial["rows"].append(
                        {
                            "row_id": row["row_id"],
                            "tokens": row["tokens"][:TOKENS],
                            "raw_cached_logprobs": logps,
                            "sampler_logprobs": row["sampler_logprobs"][:TOKENS],
                            "original_train_logprobs": originals[row["row_id"]]["train_logprobs"][
                                :TOKENS
                            ],
                            "versus_sampler": helper.differences(
                                logps, row["sampler_logprobs"][:TOKENS]
                            ),
                            "versus_original_train": helper.differences(
                                logps, originals[row["row_id"]]["train_logprobs"][:TOKENS]
                            ),
                        }
                    )
                trial["status"] = "REPLAY_COMPLETE"
            finally:
                before.remove()
                after.remove()
                language_hook.remove()
                trial["forward_inputs_outputs"] = observer.forwards
                trial["raw_logits_score_checks"] = observer.score_checks
                trial["observed_probability_vectors"] = observer.probabilities
                helper.save(args.output, result)
        result.update(
            status="DIAGNOSTIC_COMPLETE",
            finished_at=helper.now(),
            native_forward_calls=sum(len(t["forward_inputs_outputs"]) for t in result["trials"]),
            extra_forward_sequences=sum(
                len(t["forward_inputs_outputs"]) * t["batch_size"] for t in result["trials"]
            ),
        )
        helper.save(args.output, result)
    except BaseException as exc:
        result.update(
            status="DIAGNOSTIC_FAILED",
            error=repr(exc),
            traceback=traceback.format_exc(),
            finished_at=helper.now(),
        )
        helper.save(args.output, result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--mode", choices=["bf16", "fp32"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
