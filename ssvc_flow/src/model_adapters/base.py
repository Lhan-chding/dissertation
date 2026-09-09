"""Official Hugging Face multimodal adapter with an auditable sampling distribution.

Live compatibility is a P1 GPU measurement. Importing this module downloads nothing.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import re
import time
from pathlib import Path


def pure_generation_options(max_new_tokens=64):
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    return {
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "typical_p": 1.0,
        "epsilon_cutoff": 0.0,
        "eta_cutoff": 0.0,
        "repetition_penalty": 1.0,
        "encoder_repetition_penalty": 1.0,
        "num_beams": 1,
        "num_beam_groups": 1,
        "num_return_sequences": 1,
        "max_new_tokens": max_new_tokens,
        "min_length": 0,
        "min_new_tokens": 0,
        "no_repeat_ngram_size": 0,
        "encoder_no_repeat_ngram_size": 0,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "bad_words_ids": None,
        "force_words_ids": None,
        "suppress_tokens": None,
        "begin_suppress_tokens": None,
        "sequence_bias": None,
        "renormalize_logits": False,
        "remove_invalid_values": False,
        "penalty_alpha": None,
        "constraints": None,
        "watermarking_config": None,
        "guidance_scale": None,
        "dola_layers": None,
        "token_healing": False,
        "use_cache": False,
        "return_dict_in_generate": True,
        "output_scores": True,
    }


def select_language_mlp_modules(module_names, expected_layers):
    pattern = re.compile(
        r"(?:model\.)?(?:language_model\.)?layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)"
    )
    matched = [(name, pattern.fullmatch(name)) for name in module_names]
    chosen = [(name, match) for name, match in matched if match is not None]
    actual = {(int(match[1]), match[2]) for _, match in chosen}
    expected = {
        (layer, matrix)
        for layer in range(expected_layers)
        for matrix in ("gate_proj", "up_proj", "down_proj")
    }
    if actual != expected or len(chosen) != 3 * expected_layers:
        raise ValueError(
            f"Language MLP module audit failed: expected {3 * expected_layers}, found {len(chosen)}"
        )
    return sorted(name for name, _ in chosen)


def _hash_json(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()


class HuggingFaceAdapter:
    """One CUDA model; microbatch=1; no offload paths or server assumptions."""

    model_class = ""
    requires_thinking_switch = False

    def __init__(self, model, processor, model_id, revision, device="cuda"):
        self.model = model
        self.processor = processor
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.forward_calls = 0
        self.vision_forward_calls = 0
        self.last_vision_hash = None
        self.generation_calls = 0
        self._vision_hook = None
        self.audit = {}

    @classmethod
    def load(cls, model_spec, *, image_token_limit=768):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("NTU CUDA GPU required for P1; no weights were requested")
        import transformers
        from huggingface_hub import HfApi
        from peft import LoraConfig, get_peft_model

        from src.optimizer_fork import parameter_hash

        model_id = model_spec["id"]
        revision = HfApi().model_info(model_id, revision=model_spec.get("revision") or "main").sha
        if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
            raise RuntimeError("Could not resolve model revision to an immutable commit")
        model_cls = getattr(transformers, cls.model_class, None)
        if model_cls is None:
            raise RuntimeError(
                f"{cls.model_class} absent from transformers {transformers.__version__}"
            )
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        model = model_cls.from_pretrained(
            model_id,
            revision=revision,
            dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation="eager",
            trust_remote_code=False,
        )
        processor = transformers.AutoProcessor.from_pretrained(
            model_id, revision=revision, trust_remote_code=False
        )
        ip = processor.image_processor
        patch = int(getattr(ip, "patch_size", 16))
        merge = int(getattr(ip, "merge_size", getattr(ip, "spatial_merge_size", 2)))
        pixels = image_token_limit * (patch * merge) ** 2
        # New processors use size edges as area bounds; older Qwen2.5 uses max_pixels.
        if hasattr(ip, "max_pixels"):
            ip.max_pixels = pixels
        if isinstance(getattr(ip, "size", None), dict) and "longest_edge" in ip.size:
            ip.size = {**ip.size, "longest_edge": pixels}
        elif dataclasses.is_dataclass(getattr(ip, "size", None)):
            ip.size = dataclasses.replace(ip.size, longest_edge=pixels)
        names = select_language_mlp_modules(
            (name for name, _ in model.named_modules()), model_spec["expected_layers"]
        )
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model = get_peft_model(
            model,
            LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
                target_modules=names,
                task_type="CAUSAL_LM",
            ),
        )
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                if "lora_" not in name or not any(target in name for target in names):
                    raise RuntimeError(f"Unexpected trainable parameter: {name}")
                parameter.data = parameter.data.float()
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.eval()
        adapter = cls(model, processor, model_id, revision)
        adapter._install_hooks()
        torch.cuda.synchronize()
        adapter.audit = {
            "execution_kind": "REAL_CUDA_INFERENCE",
            "model_id": model_id,
            "model_revision": revision,
            "model_class": cls.model_class,
            "transformers_version": transformers.__version__,
            "base_dtype": "bfloat16",
            "probability_execution": "uncached_prefix_recompute",
            "eos_token_ids": sorted(adapter.eos_ids),
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0,
            "lora_modules": names,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "trainable_dtypes": sorted(
                {str(p.dtype) for p in model.parameters() if p.requires_grad}
            ),
            "model_tensor_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
            "load_seconds": time.perf_counter() - start,
            "load_peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "requested_image_token_limit": image_token_limit,
            "processor_max_pixels": pixels,
            "processor_patch_size": patch,
            "processor_merge_size": merge,
            "processor_hash": _hash_json(processor.to_dict()),
            "tokenizer_hash": _hash_json(processor.tokenizer.get_vocab()),
            "chat_template_hash": _hash_json(processor.chat_template),
            "frozen_parameter_hash": parameter_hash(model, trainable=False),
        }
        return adapter

    def _install_hooks(self):
        from src.optimizer_fork import state_hash

        forward_model = (
            self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        )
        # PEFT can call the conditional-generation wrapper's .forward directly,
        # bypassing its module hooks. Both entry paths call the shared backbone
        # once; decoder checkpoint replays remain below this boundary.
        forward_model.model.register_forward_pre_hook(
            lambda *args: setattr(self, "forward_calls", self.forward_calls + 1)
        )
        matches = [
            (name, module)
            for name, module in self.model.named_modules()
            if name == "model.visual" or name.endswith(".model.visual")
        ]
        if len(matches) != 1:
            raise RuntimeError("Cannot uniquely identify official visual encoder for forward audit")

        def hook(module, inputs, output):
            self.vision_forward_calls += 1
            tensors = output if isinstance(output, (tuple, list)) else (output,)
            # Official vision outputs may be ModelOutput objects.
            tensor = next((item for item in tensors if hasattr(item, "detach")), None)
            if tensor is None:
                tensor = getattr(output, "last_hidden_state", None)
            self.last_vision_hash = state_hash(tensor) if tensor is not None else None

        self._vision_hook = matches[0][1].register_forward_hook(hook)

    def prepare(self, prompt, data_root):
        from PIL import Image

        from src.optimizer_fork import state_hash

        image = None
        content = [{"type": "text", "text": prompt["user"]}]
        if prompt.get("image_path"):
            root = Path(data_root).resolve()
            path = (root / prompt["image_path"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Image path escapes dataset root")
            with Image.open(path) as source:
                image = source.convert("RGB")
            content = [{"type": "image", "image": image}, *content]
        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": content},
        ]
        if "messages" in prompt:
            expected = [{"role": "user", "content": prompt["user"]}]
            if image is not None or prompt["system"] is not None or prompt["messages"] != expected:
                raise ValueError("Legacy template requires the original single user message")
            messages = copy.deepcopy(expected)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        alternate = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        if self.requires_thinking_switch and (
            text == alternate or not text.rstrip().endswith("</think>")
        ):
            raise RuntimeError(
                "enable_thinking=False did not produce the audited completed thinking prefix"
            )
        kwargs = {"text": [text], "return_tensors": "pt", "padding": False}
        if image is not None:
            kwargs["images"] = [image]
        inputs = dict(self.processor(**kwargs))
        image_id = getattr(self.model.config, "image_token_id", None)
        image_tokens = int((inputs["input_ids"] == image_id).sum()) if image_id is not None else 0
        if image is not None and ("pixel_values" not in inputs or not image_tokens):
            raise RuntimeError("Image input disappeared in processor/tokenization")
        if image_tokens > self.audit.get("requested_image_token_limit", 768):
            raise RuntimeError("Actual visual token count exceeds P1 configured cap")
        metadata = {
            "final_prompt": text,
            "final_prompt_token_ids": inputs["input_ids"][0].tolist(),
            "final_prompt_hash": _hash_json(text),
            "tokenized_prompt_hash": state_hash(inputs["input_ids"]),
            "prompt_token_count": inputs["input_ids"].shape[-1],
            "image_token_count": image_tokens,
            "enable_thinking": False,
            "thinking_template_changed": text != alternate,
            "input_tensor_hash": state_hash(inputs),
            "pixel_values_hash": state_hash(inputs["pixel_values"]) if image is not None else None,
            "original_image_size": list(image.size) if image is not None else None,
            "image_grid_thw": inputs["image_grid_thw"].tolist()
            if "image_grid_thw" in inputs
            else None,
        }
        grid = inputs.get("image_grid_thw")
        patch_size = int(getattr(self.processor.image_processor, "patch_size", 16))
        metadata = {
            **metadata,
            "processor_height": int(grid[0, 1]) * patch_size if grid is not None else None,
            "processor_width": int(grid[0, 2]) * patch_size if grid is not None else None,
            "token_count_status": "P1_MEASURED",
            "numeric_token_count": sum(
                len(self.processor.tokenizer.encode(value, add_special_tokens=False))
                for value in re.findall(r"\d+", prompt["user"])
            ),
            "numeric_token_count_definition": (
                "sum of standalone tokenizer lengths of every decimal numeral in user text; "
                "excludes system and template"
            ),
        }
        return {"inputs": inputs, "audit": metadata}

    def _device_inputs(self, prepared):
        dtype = next(self.model.parameters()).dtype
        return {
            key: value.to(self.device, dtype=dtype if value.is_floating_point() else value.dtype)
            for key, value in prepared["inputs"].items()
        }

    def _reset_positions(self):
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        if hasattr(base, "model") and hasattr(base.model, "rope_deltas"):
            base.model.rope_deltas = None

    @property
    def eos_ids(self):
        # Some official chat tokenizers end turns before the model's corpus EOS.
        # Both stop sampling; only the final sampled EOS is removed when decoding.
        terminal = set()
        for configured in (
            getattr(getattr(self.model, "generation_config", None), "eos_token_id", None),
            getattr(self.processor.tokenizer, "eos_token_id", None),
        ):
            ids = configured if isinstance(configured, (tuple, list)) else [configured]
            terminal.update(token for token in ids if token is not None)
        return terminal

    @property
    def pad_id(self):
        value = self.processor.tokenizer.pad_token_id
        return self.processor.tokenizer.eos_token_id if value is None else value

    def generate(self, prepared, *, seed, max_new_tokens=64, do_sample=True):
        import torch
        from transformers import GenerationConfig

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.model.eval()
        self._reset_positions()
        inputs = self._device_inputs(prepared)
        if type(do_sample) is not bool:
            raise ValueError("do_sample must be a boolean")
        settings = GenerationConfig(
            **{**pure_generation_options(max_new_tokens), "do_sample": do_sample},
            eos_token_id=sorted(self.eos_ids),
            pad_token_id=self.pad_id,
            bos_token_id=self.processor.tokenizer.bos_token_id,
        )
        before_vision = self.vision_forward_calls
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            result = self.model.generate(**inputs, generation_config=settings)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        self.generation_calls += 1
        ids = result.sequences[0, inputs["input_ids"].shape[1] :].tolist()
        from src.likelihood import completion_mask

        mask = completion_mask(ids, 0, eos_ids=self.eos_ids, pad_id=self.pad_id)
        ids = [token for token, active in zip(ids, mask, strict=False) if active]
        scores = [
            float(logits[0].float().log_softmax(-1)[token])
            for logits, token in zip(result.scores, ids, strict=False)
        ]
        # Strip the terminal EOS only; retain all generated special/thinking text.
        raw_ids = ids[:-1] if ids and ids[-1] in self.eos_ids else ids
        raw = self.processor.tokenizer.decode(
            raw_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if prepared["audit"]["image_token_count"] and self.vision_forward_calls == before_vision:
            raise RuntimeError("Image did not enter an actual visual forward")
        return {
            "raw_completion": raw,
            "token_ids": ids,
            "completion_length": len(ids),
            "stop_reason": "eos" if ids and ids[-1] in self.eos_ids else "length",
            "behavior_token_logprobs": scores,
            "behavior_top1_token_ids": [
                int(logits[0].argmax()) for logits in result.scores[: len(ids)]
            ],
            "elapsed_seconds": elapsed,
            "vision_forward_calls": self.vision_forward_calls - before_vision,
            "vision_representation_hash": self.last_vision_hash,
        }

    def full_inputs(self, prepared, completion):
        import torch

        inputs = self._device_inputs(prepared)
        ids = torch.tensor([completion], device=self.device, dtype=inputs["input_ids"].dtype)
        result = {
            **inputs,
            "input_ids": torch.cat((inputs["input_ids"], ids), -1),
            "attention_mask": torch.cat((inputs["attention_mask"], torch.ones_like(ids)), -1),
        }
        if "mm_token_type_ids" in result:
            result["mm_token_type_ids"] = torch.cat(
                (result["mm_token_type_ids"], torch.zeros_like(ids)), -1
            )
        return result

    def logprobs(self, prepared, completion, *, require_grad=False):
        """Score the exact prefixes used by uncached generation, with gradients.

        BF16 full-sequence and cached/chunked kernels need not define the same
        numerical policy. Recompute each prefix with the same last-row LM head
        as generate(), preserving the original on-policy tolerance. No gradient
        cache is created; model gradient checkpointing still bounds activations.
        """
        import torch

        from src.likelihood import completion_mask

        self.model.train(require_grad)
        mask = completion_mask(completion, 0, eos_ids=self.eos_ids, pad_id=self.pad_id)
        scores = []
        with torch.set_grad_enabled(require_grad):
            for index, active in enumerate(mask):
                if not active:
                    break
                self._reset_positions()
                inputs = self.full_inputs(prepared, completion[:index])
                output = self.model(**inputs, use_cache=False, logits_to_keep=1)
                scores.append(output.logits[0, -1].float().log_softmax(-1)[completion[index]])
            if not scores:
                raise ValueError("At least one generated token required")
            return torch.stack(scores)

    def reference_logprobs(self, prepared, completion):
        with self.model.disable_adapter():
            return self.logprobs(prepared, completion, require_grad=False)

    def make_prefix_state(self, prepared):
        self.model.eval()
        self._reset_positions()
        inputs = self._device_inputs(prepared)
        output = self.model(**inputs, use_cache=True)
        if output.past_key_values is None:
            raise RuntimeError("Official forward returned no complete cache")
        # Position continuation from the framework cache itself.  Qwen2.5 can
        # account for multimodal placeholder expansion internally, so the
        # processor input length is not a safe cache position.
        cache_length = output.past_key_values.get_seq_length()
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        return {
            "cache": output.past_key_values,
            "logits": output.logits[:, -1:],
            "attention_mask": inputs["attention_mask"],
            "offset": cache_length,
            "has_mm_token_type_ids": "mm_token_type_ids" in inputs,
            "rope_deltas": copy.deepcopy(getattr(base.model, "rope_deltas", None)),
            "metadata": {
                key: value
                for key, value in inputs.items()
                if key not in ("input_ids", "attention_mask", "pixel_values", "mm_token_type_ids")
            },
        }

    def continue_from_state(self, state, completion, *, chunk_size=1):
        import torch

        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        if hasattr(base.model, "rope_deltas"):
            base.model.rope_deltas = state["rope_deltas"]
        cache, offset = state["cache"], state["offset"]
        mask = state["attention_mask"]
        pieces = [state["logits"]]
        for start in range(0, max(0, len(completion) - 1), chunk_size):
            tokens = completion[start : min(start + chunk_size, len(completion) - 1)]
            ids = torch.tensor([tokens], device=self.device, dtype=torch.long)
            mask = torch.cat((mask, torch.ones_like(ids)), -1)
            multimodal = (
                {"mm_token_type_ids": torch.zeros_like(ids)}
                if state["has_mm_token_type_ids"]
                else {}
            )
            positions = torch.arange(offset, offset + len(tokens), device=self.device)
            if state["rope_deltas"] is not None:
                multimodal = {
                    **multimodal,
                    "position_ids": positions[None, None, :].expand(3, 1, -1)
                    + state["rope_deltas"].to(self.device).reshape(1, 1, 1),
                }
            output = self.model(
                input_ids=ids,
                attention_mask=mask,
                past_key_values=cache,
                cache_position=positions,
                use_cache=True,
                **state["metadata"],
                **multimodal,
            )
            pieces.append(output.logits)
            cache, offset = output.past_key_values, offset + len(tokens)
        return self._summarize_scores(torch.cat(pieces, dim=1)[0], completion)

    @staticmethod
    def _summarize_scores(logits, completion):
        import torch

        selected = (
            logits.float()
            .log_softmax(-1)
            .gather(-1, torch.tensor(completion, device=logits.device)[:, None])[:, 0]
        )
        return logits.argmax(-1).tolist(), selected.tolist()

    def continuation_scores(self, prepared, completion, *, mode):
        if mode == "full":
            self._reset_positions()
            inputs = self.full_inputs(prepared, completion)
            output = self.model(**inputs, use_cache=False)
            start = prepared["audit"]["prompt_token_count"] - 1
            return self._summarize_scores(
                output.logits[0, start : start + len(completion)], completion
            )
        if mode not in ("chunk", "token"):
            raise ValueError("Unknown cache comparison mode")
        return self.continue_from_state(
            self.make_prefix_state(prepared),
            completion,
            chunk_size=max(2, len(completion) // 2) if mode == "chunk" else 1,
        )
