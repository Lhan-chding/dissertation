"""SR-F2 runtime: preserve original training; add unprefilled FREE/LIST.

FREE and LIST continue through ordinary EOS or the registered token budget.
They never use first-JSON stopping or trim a response to its first JSON object.
"""

from __future__ import annotations

import copy
import math
import time

from mm_core.training import capture_rng, restore_rng
from mm_core.vl_runtime import hash_json, seed_all
from sr_f1.json_protocol import decode_generated, decoded_protocol, first_balanced_token
from sr_f1.runtime import SRRuntime
from sr_f1.runtime import generation_recipe as historical_generation_recipe
from sr_f12.runtime import (
    BatchedBalancedJSONStop,
    SRF12Runtime,
    _repeat_prompt,
    common_zero_check,
    configure_training,
    enable_gradient_checkpointing,
)

from .diagnostics import protocol_prompt

__all__ = [
    "SRF2Runtime",
    "common_zero_check",
    "configure_training",
    "enable_gradient_checkpointing",
]


def generation_recipe(*, max_new_tokens=768, do_sample=True):
    if max_new_tokens not in (128, 256, 768):
        raise ValueError("Unregistered SR-F2 token budget")
    recipe = historical_generation_recipe(max_new_tokens=768, do_sample=do_sample)
    return dict(recipe, max_new_tokens=max_new_tokens)


class SRF2Runtime(SRF12Runtime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.identity["runtime_recipe"] = "SR-F2-preserved-training-free-list-eos"

    def prepare(self, row, run_root, *, protocol="evidence_answer"):
        if protocol not in ("free", "list"):
            return super().prepare(row, run_root, protocol=protocol)
        return self.prepare_text(
            protocol_prompt(row, protocol),
            row["image_file"],
            run_root,
            expected_hash=row.get("image_sha256"),
            protocol=protocol,
        )

    def prepare_text(
        self, text, image_path, run_root, *, expected_hash=None, protocol="evidence_answer"
    ):
        if protocol not in ("free", "list"):
            return super().prepare_text(
                text, image_path, run_root, expected_hash=expected_hash, protocol=protocol
            )
        # The base plain-answer path uses the same native non-thinking chat template,
        # but never appends the historical opening-brace assistant prefill.
        prepared = SRRuntime.prepare_text(
            self, text, image_path, run_root, expected_hash=expected_hash, protocol="plain_answer"
        )
        prepared["routing"].update(
            protocol=protocol,
            sr_f2_protocol=True,
            assistant_prefill="",
            answer_extraction="last_balanced_json",
        )
        return prepared

    def generate_group(
        self,
        row,
        run_root,
        seeds,
        *,
        protocol="evidence_answer",
        text=None,
        image_path=None,
        generation=None,
        on_completion=None,
    ):
        if protocol not in ("free", "list"):
            return super().generate_group(
                row,
                run_root,
                seeds,
                protocol=protocol,
                text=text,
                image_path=image_path,
                generation=generation,
                on_completion=on_completion,
            )
        if not 1 <= len(seeds) <= 8 or any(type(seed) is not int for seed in seeds):
            raise ValueError("Same-question group needs 1..8 integer slot seeds")
        if row is not None and text is not None:
            raise ValueError("Use one prompt interface")
        prepared = (
            self.prepare(row, run_root, protocol=protocol)
            if row is not None
            else self.prepare_text(text, image_path, run_root, protocol=protocol)
        )
        count = len(seeds)
        config = copy.deepcopy(self.generation_config)
        opts = generation or {}
        token_cap = 256 if protocol == "free" else 128
        default_sample = True
        if opts.get("do_sample", True) is not True:
            raise PermissionError("FREE/LIST require registered stochastic sampling")
        if protocol == "plain_answer" and (count != 1 or opts.get("do_sample", False)):
            raise PermissionError("ChartQA plain-answer channel is single-row greedy")
        if set(opts) - {
            "max_new_tokens",
            "do_sample",
            "temperature",
            "top_p",
            "top_k",
            "num_return_sequences",
        }:
            raise PermissionError("Unregistered generation options")
        if any(
            opts.get(key, expected) != expected
            for key, expected in (
                ("temperature", 1),
                ("top_p", 1),
                ("top_k", 0),
                ("max_new_tokens", token_cap),
            )
        ):
            raise PermissionError("Registered sampling recipe or protocol token cap changed")
        if opts.get("num_return_sequences", count) != count:
            raise PermissionError("Explicit group size disagrees with registered seeds")
        for key, value in generation_recipe(
            max_new_tokens=token_cap, do_sample=opts.get("do_sample", default_sample)
        ).items():
            setattr(config, key, value)
        config.num_return_sequences = 1  # Inputs are manually expanded, including image patches.
        inputs = _repeat_prompt(prepared["inputs"], count)
        prompt_len = inputs["input_ids"].shape[-1]
        amended = protocol in ("evidence_answer", "answer_only")
        stop = (
            BatchedBalancedJSONStop(self.processor.tokenizer, prompt_len, count)
            if amended
            else None
        )
        kwargs = {}
        if stop is not None:
            from transformers import StoppingCriteriaList

            kwargs["stopping_criteria"] = StoppingCriteriaList([stop])
        self.reserve("completion_attempts", count, group_seed=seeds[0], qid=(row or {}).get("qid"))
        rng, before, started = capture_rng(), self.image_calls, time.perf_counter()
        try:
            seed_all(seeds[0])
            self.model.eval()
            base = (
                self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
            )
            for module in (base, getattr(base, "model", None)):
                if module is not None and hasattr(module, "rope_deltas"):
                    module.rope_deltas = None
            with self.torch.no_grad():
                result = self.model.generate(**inputs, generation_config=config, **kwargs)
            if result.sequences.shape[0] != count:
                raise RuntimeError("Generation returned wrong number of group rows")
            records = []
            adapter_identity = self.current_adapter_identity()
            model_identity = self.stable_model_identity()
            for index, seed in enumerate(seeds):
                full_tokens = result.sequences[index, prompt_len:].cpu().tolist()
                balanced_n, _ = (
                    first_balanced_token(self.processor.tokenizer, full_tokens)
                    if amended
                    else (None, None)
                )
                eos_n = next(
                    (i + 1 for i, token in enumerate(full_tokens) if token in self.eos_ids), None
                )
                n = min(x for x in (balanced_n, eos_n, len(full_tokens)) if x is not None)
                tokens, errors, logps = full_tokens[:n], [], []
                if not tokens or len(result.logits) < n or len(result.scores) < n:
                    errors.append("INCOMPLETE_RAW_SAMPLER_LOGITS")
                for token, raw, transformed in zip(
                    tokens, result.logits, result.scores, strict=False
                ):
                    if not self.torch.equal(raw[index].float(), transformed[index].float()):
                        errors.append("UNREGISTERED_LOGITS_TRANSFORM")
                    logps.append(float(raw[index].float().log_softmax(-1)[token]))
                if not all(map(math.isfinite, logps)):
                    errors.append("NONFINITE_SAMPLER_LOGPROB")
                finish = "balanced" if balanced_n == n else "eos" if eos_n == n else "length"
                if finish == "length" and n != token_cap:
                    errors.append("UNREGISTERED_TERMINATION")
                if stop is not None and finish == "balanced" and stop.counts[index] != n:
                    errors.append("BALANCED_STOP_REPLAY_MISMATCH")
                if prepared["routing"].get("image_token_count", 0) and self.image_calls <= before:
                    errors.append("NATIVE_VISUAL_ENCODER_NOT_CALLED")
                text_out = decode_generated(self.processor.tokenizer, tokens)
                record = dict(
                    row=row,
                    root=str(run_root),
                    protocol=protocol,
                    qid=(row or {}).get("qid"),
                    tokens=tokens,
                    raw_tokens=tokens,
                    raw_text=text_out,
                    generated_text=text_out,
                    assistant_prefill="",
                    answer_extraction="last_balanced_json",
                    seed=seed,
                    group_seed=seeds[0],
                    group_row_index=index,
                    group_size=count,
                    sampler_logprobs=logps,
                    sampler_logprob_source="actual_generate_raw_logits_selected_token",
                    prompt_token_count=prompt_len,
                    completion_token_count=n,
                    prompt_tensor_hash=prepared["routing"]["input_tensor_hash"],
                    input_routing=copy.deepcopy(prepared["routing"]),
                    image_routing=copy.deepcopy(prepared["routing"]),
                    sampling_parameters=config.to_dict(),
                    sampling_hash=hash_json(config.to_dict()),
                    generation_seconds=time.perf_counter() - started,
                    finish_reason=finish,
                    truncated=finish == "length",
                    generation_status="TECHNICAL_INVALID" if errors else "COMPLETE",
                    technical_validation_errors=errors,
                    adapter_identity=copy.deepcopy(adapter_identity),
                    model_identity=copy.deepcopy(model_identity),
                )
                if amended:
                    record.update(
                        decoded_protocol(text_out),
                        balanced_token_count=balanced_n if finish == "balanced" else None,
                    )
                if on_completion:
                    on_completion(record)
                records.append(record)
                self.reserve("generated_tokens", n, group_seed=seeds[0], row_index=index)
            if any(record["technical_validation_errors"] for record in records):
                raise RuntimeError(
                    "Invalid group generation preserved: "
                    + str([record["technical_validation_errors"] for record in records])
                )
            return records
        finally:
            restore_rng(rng)
