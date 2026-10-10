"""SR-F1.2 native whole-sequence training and same-question generation.

This module deliberately does not modify SR-F1's cached replay or narrow LoRA.
The caller owns registered input/release gates and the frozen microbatch choice.
"""

from __future__ import annotations

import copy
import inspect
import math
import re
import time

from mm_core.training import capture_rng, restore_rng, state_hash
from mm_core.vl_runtime import hash_json, seed_all
from mm_dev.runtime import reference_parameters
from sr_f1.json_protocol import (
    AMENDMENT_ID,
    PREFILL,
    balanced_cut,
    decode_generated,
    decoded_protocol,
    first_balanced_token,
)
from sr_f1.runtime import SRRuntime, generation_recipe


class BatchedBalancedJSONStop:
    """HF per-row stop flags; finished rows retain their first token boundary."""

    def __init__(self, tokenizer, prompt_len, rows):
        if prompt_len <= 0 or rows <= 0:
            raise ValueError("Positive prompt length and row count required")
        self.tokenizer, self.prompt_len = tokenizer, prompt_len
        self.counts, self.cuts = [None] * rows, [None] * rows

    def __call__(self, input_ids, scores, **kwargs):
        import torch

        if input_ids.shape[0] != len(self.counts):
            raise PermissionError("Generation row count changed")
        for row in range(len(self.counts)):
            if self.counts[row] is not None:
                continue
            tokens = input_ids[row, self.prompt_len :].tolist()
            cut = balanced_cut(PREFILL + decode_generated(self.tokenizer, tokens))
            if cut is not None:
                self.counts[row], self.cuts[row] = len(tokens), cut
        return input_ids.new_tensor([n is not None for n in self.counts], dtype=torch.bool)


def language_linear_modules(model):
    """Discover actual native decoder Linear modules, including gated-delta projections."""
    import torch

    names = sorted(
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and re.search(r"(?:^|\.)language_model\.layers\.\d+\.", name)
        and not any(
            x in name.lower()
            for x in ("visual", "vision", "merger", "projector", "lm_head", "embed", "lora_")
        )
    )
    if not names:
        raise ValueError("No native language decoder linear layers discovered")
    return names


def configure_training(runtime, learning_rate=5e-5):
    """Independent SR-F1.2 all-language-linear LoRA recipe; no old defaults mutated."""
    import torch
    from peft import LoraConfig, get_peft_model

    if learning_rate not in (5e-5, 2e-5, 1e-4):
        raise PermissionError("Learning rate outside the preregistered pilot choices")
    model = runtime.model
    if hasattr(model, "peft_config"):
        cfg = model.peft_config[model.active_adapter]
        targets = sorted(cfg.target_modules)
        actual = sorted(
            name.removesuffix(".base_layer")
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear)
            and name.endswith(".base_layer")
            and re.search(r"(?:^|\.)language_model\.layers\.\d+\.", name)
        )
        if (cfg.r, cfg.lora_alpha, cfg.lora_dropout, cfg.bias) != (8, 16, 0, "none"):
            raise PermissionError("Existing adapter differs from SR-F1.2")
        # All decoder linears must be wrapped: no narrow old adapter is accepted.
        unwrapped = [
            name for name in language_linear_modules(model) if not name.endswith(".base_layer")
        ]
        if unwrapped or len(actual) != len(targets):
            raise PermissionError("Existing adapter omits language decoder linear modules")
    else:
        targets = language_linear_modules(model)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model = get_peft_model(
            model,
            LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
                target_modules=targets,
                task_type="CAUSAL_LM",
                init_lora_weights=True,
            ),
        )
        runtime.model = model
        if any(
            bool((p.detach() != 0).any())
            for name, p in model.named_parameters()
            if ".lora_B." in name
        ):
            raise RuntimeError("Common LoRA B must initialize exactly to zero")
    trainable = []
    for name, parameter in model.named_parameters():
        enabled = ".lora_A." in name or ".lora_B." in name
        parameter.requires_grad_(enabled)
        if enabled:
            if not re.search(r"(?:^|\.)language_model\.layers\.\d+\.", name) or any(
                x in name.lower()
                for x in ("visual", "vision", "merger", "projector", "lm_head", "embed")
            ):
                raise PermissionError("Trainable tensor outside registered language decoder")
            parameter.data = parameter.data.float()
            trainable.append(name)
    if not trainable:
        raise RuntimeError("No trainable LoRA parameters")
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    model.eval()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    receipt = dict(
        target_modules=targets,
        trainable_names=trainable,
        r=8,
        lora_alpha=16,
        lora_dropout=0,
        trainable_dtype="float32",
        learning_rate=learning_rate,
        target_module_count=len(targets),
        common_start="SRF1_2_COMMON_ZERO",
    )
    runtime.identity["sr_f12_lora"] = receipt
    return optimizer, scheduler, receipt


def _repeat_prompt(inputs, rows):
    """Repeat one image's flattened patches and grid together, in row order."""
    output = {}
    for name, value in inputs.items():
        if name in ("position_ids", "past_key_values", "cache_position"):
            continue
        if not hasattr(value, "shape"):
            raise TypeError("Unexpected native processor non-tensor input: " + name)
        if name not in ("pixel_values", "image_grid_thw") and value.shape[0] != 1:
            raise PermissionError("Only one same-question prompt may be expanded")
        output[name] = value.repeat((rows,) + (1,) * (value.ndim - 1))
    return output


class SRF12Runtime(SRRuntime):
    def __init__(self, *args, **kwargs):
        if kwargs.get("protocol_amendment") not in (None, {"id": AMENDMENT_ID}):
            raise PermissionError("SR-F1.2 requires its inherited registered JSON prefill")
        kwargs["protocol_amendment"] = {"id": AMENDMENT_ID}
        super().__init__(*args, **kwargs)
        self.identity["runtime_recipe"] = "SR-F1.2-batched-full-sequence"

    @classmethod
    def processor_only(cls, model_path, *, protocol_amendment=None):
        return super().processor_only(model_path, protocol_amendment={"id": AMENDMENT_ID})

    def prepare_text(
        self, text, image_path, run_root, *, expected_hash=None, protocol="evidence_answer"
    ):
        # The prompt text is already chosen by model_input for its real protocol.
        # Passing evidence here only reuses SR-F1.1's audited prefill boundary.
        if protocol not in ("evidence_answer", "answer_only", "plain_answer"):
            raise PermissionError("Unknown output protocol")
        prepared = super().prepare_text(
            text,
            image_path,
            run_root,
            expected_hash=expected_hash,
            protocol="evidence_answer" if protocol == "answer_only" else protocol,
        )
        prepared["routing"]["protocol"] = protocol
        prepared["routing"]["sr_f12_protocol"] = True
        return prepared

    def _prepared_records(self, records):
        if not records:
            raise ValueError("Empty teacher-forcing microbatch")
        from sr_f1.data import model_input

        prepared, same_call = [], {}
        for record in records:
            item = record.get("prepared")
            if item is None:
                protocol = record.get("protocol", "evidence_answer")
                root = record.get("run_root", record.get("root"))
                row = record["row"]
                key = hash_json(
                    dict(
                        model_input=model_input(row, protocol=protocol),
                        root=str(root),
                        image_sha256=row.get("image_sha256"),
                        protocol=protocol,
                    )
                )
                if key not in same_call:
                    same_call[key] = self.prepare(row, root, protocol=protocol)
                item = same_call[key]
            prepared.append(item)
        # Exact inputs, not qid alone: diagnostic variants must never share a batch.
        first = prepared[0]["inputs"]
        for item in prepared[1:]:
            other = item["inputs"]
            if first.keys() != other.keys() or any(
                not self.torch.equal(first[k], other[k]) for k in first
            ):
                raise PermissionError("Microbatch must have byte-identical same-question prompts")
        return prepared[0]

    def batch_sequence_forward(self, records, *, purpose="training_policy", grad=True):
        return self._batch_forward(
            self._prepared_records(records),
            [record["tokens"] for record in records],
            purpose=purpose,
            grad=grad,
        )

    def batch_reference_forward(self, records, reference):
        with reference_parameters(self.model, reference):
            return self.batch_sequence_forward(records, purpose="training_reference", grad=False)

    def sequence_forward(self, prepared, tokens, *, purpose, grad=False):
        return dict(
            logprobs=self._batch_forward(prepared, [tokens], purpose=purpose, grad=grad)[0],
            entropy=None,
        )

    def reference_forward(self, prepared, tokens, reference):
        with reference_parameters(self.model, reference):
            return self.sequence_forward(prepared, tokens, purpose="training_reference", grad=False)

    def _batch_forward(self, prepared, token_rows, *, purpose, grad):
        torch = self.torch
        if not token_rows or any(not row or len(row) > 768 for row in token_rows):
            raise ValueError("Each completion must contain 1..768 generated tokens")
        rows, longest = len(token_rows), max(map(len, token_rows))
        inputs = _repeat_prompt(prepared["inputs"], rows)
        prompt_len = inputs["input_ids"].shape[-1]
        pad = self.processor.tokenizer.pad_token_id
        if pad is None:
            pad = self.eos_ids[0]
        completion = inputs["input_ids"].new_full((rows, longest), pad)
        mask = completion.new_zeros(completion.shape)
        for index, tokens in enumerate(token_rows):
            completion[index, : len(tokens)] = completion.new_tensor(tokens)
            mask[index, : len(tokens)] = 1
        inputs["input_ids"] = torch.cat([inputs["input_ids"], completion], dim=-1)
        inputs["attention_mask"] = torch.cat(
            [inputs.get("attention_mask", mask.new_ones((rows, prompt_len))), mask], dim=-1
        )
        if "mm_token_type_ids" not in inputs:
            raise RuntimeError("Native multimodal token types missing")
        inputs["mm_token_type_ids"] = torch.cat(
            [inputs["mm_token_type_ids"], torch.zeros_like(completion)], dim=-1
        )
        inputs.pop("token_type_ids", None)
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        native = getattr(base, "model", base)
        for module in (base, native):
            if hasattr(module, "rope_deltas"):
                module.rope_deltas = None
        if "logits_to_keep" not in inspect.signature(base.forward).parameters:
            raise RuntimeError("Native model must support answer-only logits_to_keep")
        # Explicit independent mRoPE on each row; no stale shared rope_deltas.
        positions = []
        for index in range(rows):
            position, _ = native.get_rope_index(
                input_ids=inputs["input_ids"][index : index + 1],
                mm_token_type_ids=inputs["mm_token_type_ids"][index : index + 1],
                image_grid_thw=prepared["inputs"].get("image_grid_thw"),
                attention_mask=inputs["attention_mask"][index : index + 1],
            )
            positions.append(position)
        inputs.update(
            position_ids=torch.cat(positions, dim=1), use_cache=False, logits_to_keep=longest + 1
        )
        self.reserve(
            "extra_forward_sequences",
            rows,
            purpose=purpose,
            completion_tokens=sum(map(len, token_rows)),
        )
        before = self.image_calls
        self.model.train(bool(grad and getattr(self, "_sr_f12_checkpointing", False)))
        with torch.enable_grad() if grad else torch.no_grad():
            output = self.model(**inputs)
            if output.logits.shape[:2] != (rows, longest + 1):
                raise RuntimeError("Native answer-only logits shape changed")
            # Every row starts at the same prompt boundary. Right padded logits
            # are excluded before fp32 log_softmax and never enter the loss.
            chosen = [
                output.logits[i, : len(tokens)]
                .float()
                .log_softmax(-1)
                .gather(-1, completion[i, : len(tokens), None])
                .squeeze(-1)
                for i, tokens in enumerate(token_rows)
            ]
        if prepared["routing"].get("image_token_count", 0) and self.image_calls <= before:
            raise RuntimeError("Whole-sequence forward did not invoke vision encoder")
        return chosen

    def generate_training(self, row, run_root, seed, *, on_completion=None):
        return self.generate(row, run_root, seed, on_completion=on_completion)

    def generate(
        self,
        row=None,
        run_root=None,
        seed=None,
        *,
        protocol="evidence_answer",
        text=None,
        image_path=None,
        generation=None,
        on_completion=None,
        **unused,
    ):
        if unused:
            raise TypeError("Unknown generation options: " + ",".join(unused))
        return self.generate_group(
            row,
            run_root,
            [seed],
            protocol=protocol,
            text=text,
            image_path=image_path,
            generation=generation,
            on_completion=on_completion,
        )[0]

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
        token_cap = 128 if protocol == "plain_answer" else 768
        default_sample = protocol != "plain_answer"
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


def common_zero_check(runtime, cases):
    """Exactly 32 registered fixed answers; compare raw base versus B=0 adapter."""
    if len(cases) != 32:
        raise ValueError("Common zero check requires exactly 32 answer trajectories")
    if not hasattr(runtime.model, "disable_adapter"):
        raise PermissionError("Common zero check requires configured LoRA")
    if any(
        bool((p.detach() != 0).any())
        for name, p in runtime.model.named_parameters()
        if ".lora_B." in name
    ):
        raise PermissionError("Common start B differs from zero")
    differences, exact = [], True
    for case in cases:
        with runtime.model.disable_adapter():
            baseline = runtime.batch_sequence_forward(
                [case], purpose="common_zero_base", grad=False
            )[0]
        adapted = runtime.batch_sequence_forward([case], purpose="common_zero_adapter", grad=False)[
            0
        ]
        exact = exact and runtime.torch.equal(baseline, adapted)
        differences.append(float((baseline - adapted).abs().max()))
    return dict(
        status="PASS" if exact else "FAIL",
        answer_count=32,
        maximum_logprob_difference=max(differences),
        bitwise_equal=exact,
        parameter_hash=state_hash(
            {name: p.detach() for name, p in runtime.model.named_parameters() if p.requires_grad}
        ),
    )


def enable_gradient_checkpointing(runtime, cases):
    """Only called after microbatch one OOM; verify mode equivalence before enabling.

    This is a technical resource choice that must be frozen for every arm. It
    never changes dropout, dtype, token cap, or the full-sequence objective.
    """
    if not cases:
        raise ValueError("Checkpoint mode requires fixed validation cases")
    if any(
        isinstance(module, runtime.torch.nn.Dropout) and module.p != 0
        for module in runtime.model.modules()
    ):
        raise PermissionError("Gradient checkpointing requires all dropout to be zero")
    before = [
        runtime.batch_sequence_forward([case], purpose="checkpoint_eval", grad=False)[0]
        for case in cases
    ]
    runtime.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False, "preserve_rng_state": True}
    )
    runtime._sr_f12_checkpointing = True
    differences, exact = [], True
    try:
        for case, baseline in zip(cases, before, strict=True):
            current = runtime.batch_sequence_forward([case], purpose="checkpoint_train", grad=True)[
                0
            ]
            exact = exact and runtime.torch.equal(current.detach(), baseline)
            differences.append(float((current.detach() - baseline).abs().max()))
            del current
        if not exact:
            raise RuntimeError("Checkpoint train/eval forward probabilities differ")
    except BaseException:
        runtime.model.gradient_checkpointing_disable()
        runtime._sr_f12_checkpointing = False
        runtime.model.eval()
        raise
    return dict(
        status="PASS",
        non_reentrant=True,
        dropout=0,
        train_eval_bitwise_equal=True,
        maximum_logprob_difference=max(differences),
    )
