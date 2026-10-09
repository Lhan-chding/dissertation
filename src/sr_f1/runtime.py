"""SR-F1 native Qwen runtime. Importing this module never loads model weights.

The old F2 modules are reused only for parameter ownership and native cache
mechanics; their prompts, samplers, gates, adapters and scientific state are not used.
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

from mm_core.training import capture_rng, restore_rng, state_hash, trainable_state
from mm_core.vl_runtime import (
    CHAT_TEMPLATE_KWARGS,
    GENERATION,
    QwenRuntime,
    hash_json,
    linear_kernel_identity,
    seed_all,
    validate_model_config,
)
from mm_dev.runtime import (
    actual_cuda_identity,
    atomic_json,
    bounded_path,
    cache_compatible_autograd,
    configure_audited_backend,
    file_hash,
    functional_training_cache,
    read_json,
    reference_parameters,
)
from mm_dev.runtime import (
    copy_parameters as copy_parameters,
)

CANVAS = (1024, 768)
TARGET_PIXELS = 786432
PLAN_ID = "SR-F1-20261009"


def generation_recipe(*, max_new_tokens=768, do_sample=True):
    if max_new_tokens not in (128, 768) or type(do_sample) is not bool:
        raise ValueError("Only registered 128/768 token output channels are permitted")
    return {
        **GENERATION,
        "do_sample": do_sample,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "max_new_tokens": max_new_tokens,
        "output_logits": True,
        "output_scores": True,
    }


class SRRuntime(QwenRuntime):
    def __init__(
        self,
        model_path,
        adapter_path=None,
        *,
        device="cuda:0",
        dtype="bfloat16",
        attention_backend="eager",
        account=None,
    ):
        if dtype != "bfloat16" or attention_backend != "eager":
            raise PermissionError("SR-F1 requires BF16 base and native eager attention")
        super().__init__(
            model_path,
            adapter_path,
            device=device,
            dtype=dtype,
            attention_backend=attention_backend,
            account=account,
        )
        from transformers import AutoProcessor, GenerationConfig

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=False,
            min_pixels=TARGET_PIXELS,
            max_pixels=TARGET_PIXELS,
        )
        self.generation_config = GenerationConfig(
            **generation_recipe(),
            eos_token_id=self.eos_ids,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            bos_token_id=self.processor.tokenizer.bos_token_id,
        )
        self.identity.update(
            processor_hash=hash_json(self.processor.to_dict()),
            tokenizer_hash=hash_json(self.processor.tokenizer.get_vocab()),
            chat_template_hash=hash_json(self.processor.chat_template),
            generation_config_expanded=self.generation_config.to_dict(),
            canvas_pixels=list(CANVAS),
            processor_target_pixels=TARGET_PIXELS,
        )

    @classmethod
    def processor_only(cls, model_path):
        import torch
        import transformers
        from transformers import AutoConfig, AutoProcessor, GenerationConfig

        instance = cls.__new__(cls)
        instance.device = "cpu"
        instance.model_path = str(Path(model_path).resolve())
        architecture = validate_model_config(read_json(Path(model_path) / "config.json"))
        instance.processor = AutoProcessor.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=False,
            min_pixels=TARGET_PIXELS,
            max_pixels=TARGET_PIXELS,
        )
        instance.model = SimpleNamespace(
            config=AutoConfig.from_pretrained(
                str(model_path), local_files_only=True, trust_remote_code=False
            )
        )
        kernels = linear_kernel_identity()
        instance.identity = {
            **architecture,
            "linear_kernel_identity": kernels,
            "linear_kernel_identity_hash": hash_json(kernels),
            "processor_hash": hash_json(instance.processor.to_dict()),
            "tokenizer_hash": hash_json(instance.processor.tokenizer.get_vocab()),
            "chat_template_hash": hash_json(instance.processor.chat_template),
            "canvas_pixels": list(CANVAS),
            "processor_target_pixels": TARGET_PIXELS,
        }
        generation_file = Path(model_path) / "generation_config.json"
        stored_generation = (
            GenerationConfig.from_pretrained(str(model_path), local_files_only=True)
            if generation_file.is_file()
            else GenerationConfig.from_model_config(instance.model.config)
        )
        eos = stored_generation.eos_token_id
        instance.eos_ids = eos if isinstance(eos, list) else [eos]
        instance.generation_config = GenerationConfig(
            **generation_recipe(),
            eos_token_id=instance.eos_ids,
            pad_token_id=instance.processor.tokenizer.pad_token_id,
            bos_token_id=instance.processor.tokenizer.bos_token_id,
        )
        instance.identity.update(
            generation_config_expanded=instance.generation_config.to_dict(),
            torch_version=torch.__version__,
            transformers_version=transformers.__version__,
            chat_template_kwargs=dict(CHAT_TEMPLATE_KWARGS),
            chat_template_kwargs_hash=hash_json(CHAT_TEMPLATE_KWARGS),
        )
        return instance

    def prepare(self, row, run_root, *, protocol="evidence_answer"):
        from .data import model_input

        safe = model_input(row, protocol=protocol)
        return self.prepare_text(
            safe["text"], safe["image_file"], run_root, expected_hash=row.get("image_sha256")
        )

    def prepare_text(self, text, image_path, run_root, *, expected_hash=None):
        """Only explicit text and pixels enter the processor; no gold sidecar is read."""
        from PIL import Image

        if not isinstance(text, str) or not text:
            raise ValueError("Nonempty registered prompt text required")
        root = Path(run_root).resolve()
        rgb, source_size, image_hash = None, None, None
        content = []
        if image_path is not None:
            path = Path(image_path)
            path = path.resolve() if path.is_absolute() else bounded_path(root, path)
            if not path.is_relative_to(root):
                raise PermissionError("Image path escapes run root")
            image_hash = file_hash(path)
            if expected_hash is not None and image_hash != expected_hash:
                raise PermissionError("Image bytes differ from registered identity")
            with Image.open(path) as original:
                source_size = original.size
                rgb = original.convert("RGB")
            # PIL object is represented by an image token by the native template.
            # A filesystem path or qid is never placed in the template's text.
            content.append({"type": "image", "image": rgb})
        content.append({"type": "text", "text": text})
        chat = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            **CHAT_TEMPLATE_KWARGS,
        )
        if not chat.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"):
            raise PermissionError("Native non-thinking assistant template differs")
        kwargs = dict(text=[chat], padding=False, return_tensors="pt")
        if rgb is not None:
            kwargs["images"] = [rgb]
        inputs = self.processor(**kwargs)
        ids = inputs["input_ids"]
        if ids.shape[0] != 1:
            raise PermissionError("One independent context per generation is required")
        image_id = getattr(self.model.config, "image_token_id", None)
        mm_types = inputs.get("mm_token_type_ids")
        if rgb is None and mm_types is None:
            # Native text-only processor may omit this field. The hybrid cached
            # path requires explicit zeros to preserve the same generation API.
            inputs["mm_token_type_ids"] = ids.new_zeros(ids.shape)
            mm_types = inputs["mm_token_type_ids"]
        if mm_types is None or mm_types.shape != ids.shape:
            raise RuntimeError("Native multimodal token types are missing")
        grid, processed_size, pixel_hash, visual_tokens = [], None, None, 0
        if rgb is not None:
            if "pixel_values" not in inputs or "image_grid_thw" not in inputs:
                raise RuntimeError("Processor did not route real image pixels")
            grid = inputs["image_grid_thw"].tolist()
            ip = self.processor.image_processor
            patch, merge = int(ip.patch_size), int(ip.merge_size)
            if len(grid) != 1 or grid[0][0] != 1:
                raise PermissionError("Exactly one still image is required")
            processed_size = [grid[0][2] * patch, grid[0][1] * patch]
            if source_size == CANVAS and tuple(processed_size) != CANVAS:
                raise PermissionError("Native processor resized the fixed synthetic canvas")
            visual_tokens = int((ids == image_id).sum()) if image_id is not None else 0
            if visual_tokens <= 0 or visual_tokens != math.prod(grid[0]) // merge**2:
                raise RuntimeError("Image grid and visual token counts disagree")
            if not bool(((mm_types == 1) == (ids == image_id)).all()):
                raise RuntimeError("Native image token flags disagree")
            pixels = inputs["pixel_values"].detach().cpu().contiguous()
            pixel_hash = hashlib.sha256(pixels.numpy().tobytes()).hexdigest()
        elif bool((mm_types != 0).any()) or "pixel_values" in inputs:
            raise RuntimeError("Text-only diagnostic unexpectedly contains visual inputs")
        tensor_hashes = {
            name: state_hash(value) for name, value in inputs.items() if hasattr(value, "detach")
        }
        routing = dict(
            source_image_sha256=image_hash,
            source_size=list(source_size) if source_size else None,
            processed_size=processed_size,
            image_grid_thw=grid,
            processed_pixel_sha256=pixel_hash,
            image_token_count=visual_tokens,
            image_token_id=image_id,
            processor_target_pixels=TARGET_PIXELS,
            native_input_tensor_hashes=tensor_hashes,
            input_tensor_hash=hash_json(tensor_hashes),
            chat_text_sha256=hashlib.sha256(chat.encode()).hexdigest(),
            input_ids_sha256=hash_json(ids.tolist()),
            processor_hash=self.identity["processor_hash"],
            chat_template_hash=self.identity["chat_template_hash"],
            chat_template_kwargs=dict(CHAT_TEMPLATE_KWARGS),
            pixels_per_count=(515 / 100 * processed_size[1] / 768)
            if source_size == CANVAS
            else None,
        )
        return dict(inputs=inputs.to(self.device), routing=routing, chat_text=chat)

    def current_adapter_identity(self):
        parameters = {
            name: p.detach() for name, p in self.model.named_parameters() if "lora_" in name
        }
        return dict(
            model_id=self.identity.get("model_id"),
            step=self.identity.get("step"),
            adapter_path=self.adapter_path,
            base_model_weights_hash=self.identity.get("base_model_weights_hash"),
            adapter_parameter_hash=state_hash(parameters) if parameters else None,
            published_adapter_hash=self.identity.get("adapter_hash"),
        )

    def stable_model_identity(self):
        result = {
            key: value
            for key, value in self.identity.items()
            if key not in {"hardware", "active_adapter"}
        }
        hardware = self.identity.get("hardware", {})
        result["hardware"] = {
            key: hardware[key]
            for key in (
                "cuda_name",
                "cuda_total_memory_bytes",
                "compute_capability",
                "cuda_runtime",
                "driver_version",
            )
            if key in hardware
        }
        return result

    def training_generation_config(self):
        return copy.deepcopy(self.generation_config)

    def generate_training(self, row, run_root, seed, *, on_completion=None):
        return self.generate(row, run_root, seed, on_completion=on_completion)

    def generate(
        self,
        row=None,
        run_root=None,
        seed=None,
        *,
        text=None,
        image_path=None,
        generation=None,
        protocol="evidence_answer",
        on_completion=None,
        **unused,
    ):
        if unused:
            raise TypeError("Unknown generation options: " + ",".join(unused))
        if run_root is None or type(seed) is not int:
            raise ValueError("Run root and explicit integer sample seed are required")
        if row is not None and text is not None:
            raise ValueError("Use one prompt interface per generation")
        prepared = (
            self.prepare(row, run_root, protocol=protocol)
            if row is not None
            else (self.prepare_text(text, image_path, run_root))
        )
        config = self.training_generation_config()
        if generation:
            permitted = {
                "max_new_tokens",
                "do_sample",
                "temperature",
                "top_p",
                "top_k",
                "num_return_sequences",
            }
            if set(generation) - permitted:
                raise PermissionError("Unregistered generation options")
            if generation.get("num_return_sequences", 1) != 1:
                raise PermissionError("Exactly one independent completion per call is required")
            for key in ("temperature", "top_p", "top_k"):
                if (
                    key in generation
                    and generation[key] != {"temperature": 1, "top_p": 1, "top_k": 0}[key]
                ):
                    raise PermissionError("Sampling channel differs from preregistration")
            recipe = generation_recipe(
                max_new_tokens=generation.get("max_new_tokens", 768),
                do_sample=generation.get("do_sample", True),
            )
            for key, value in recipe.items():
                setattr(config, key, value)
        self.reserve("completion_attempts", 1, seed=seed, qid=(row or {}).get("qid"))
        rng, before, started = capture_rng(), self.image_calls, time.perf_counter()
        try:
            seed_all(seed)
            self.model.eval()
            with self.torch.no_grad():
                result = self.model.generate(**prepared["inputs"], generation_config=config)
            prompt_len = prepared["inputs"]["input_ids"].shape[-1]
            tokens = result.sequences[0, prompt_len:].cpu().tolist()
            errors, logps, sampler = [], [], []
            if not tokens or len(result.logits) != len(tokens) or len(result.scores) != len(tokens):
                errors.append("INCOMPLETE_RAW_SAMPLER_LOGITS")
            for token, raw, transformed in zip(tokens, result.logits, result.scores, strict=False):
                if not self.torch.equal(raw.float(), transformed.float()):
                    errors.append("UNREGISTERED_LOGITS_TRANSFORM")
                logps.append(float(raw[0].float().log_softmax(-1)[token]))
                sampler.append(float(transformed[0].float().log_softmax(-1)[token]))
            if not all(math.isfinite(v) for v in logps + sampler):
                errors.append("NONFINITE_SAMPLER_LOGPROB")
            requires_image = prepared["routing"]["image_token_count"] > 0
            if requires_image and self.image_calls <= before:
                errors.append("NATIVE_VISUAL_ENCODER_NOT_CALLED")
            routing = {
                **prepared["routing"],
                "generation_vision_forward_calls": self.image_calls - before,
            }
            record = dict(
                raw_text=self.processor.tokenizer.decode(
                    tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
                ),
                tokens=tokens,
                raw_tokens=tokens,
                seed=seed,
                old_logprobs=[v if math.isfinite(v) else None for v in logps],
                sampler_logprobs=[v if math.isfinite(v) else None for v in sampler],
                old_logprob_source="actual_generate_raw_logits_selected_token",
                generation_status="TECHNICAL_INVALID" if errors else "COMPLETE",
                technical_validation_errors=errors,
                sampling_hash=hash_json(config.to_dict()),
                sampling_parameters=config.to_dict(),
                prompt_token_count=prompt_len,
                completion_token_count=len(tokens),
                generation_seconds=time.perf_counter() - started,
                truncated=bool(
                    tokens
                    and tokens[-1] not in self.eos_ids
                    and len(tokens) == config.max_new_tokens
                ),
                finish_reason="eos" if tokens and tokens[-1] in self.eos_ids else "length",
                image_routing=routing,
                input_routing=routing,
                adapter_identity=self.current_adapter_identity(),
                model_identity=self.stable_model_identity(),
                runtime_hardware=self.identity.get("hardware"),
            )
            if on_completion:
                on_completion(record)
            self.reserve("generated_tokens", len(tokens), seed=seed)
            if errors:
                raise RuntimeError("Invalid generation preserved: " + ";".join(errors))
            return record
        finally:
            restore_rng(rng)

    def reference_forward(self, prepared, tokens, reference):
        with reference_parameters(self.model, reference):
            return self.sequence_forward(prepared, tokens, purpose="training_reference")

    def sequence_forward(self, prepared, tokens, *, purpose, grad=False):
        """The identical native cache path is used for old/current/reference probabilities."""
        return self.cached_training_forward(prepared, tokens, purpose=purpose, grad=grad)

    def cached_training_forward(self, prepared, tokens, *, purpose, grad=False):
        """Teacher-force saved tokens through the sampler's native cached path.

        All prefix states retain their computation graph. Only cache containers
        change ownership; no kernel, precision, token, or sampling setting changes.
        Measurement/PROBE teacher forcing continues to use the audited core path.
        """
        torch = self.torch
        if not tokens:
            raise ValueError("Cannot score empty completion")
        if self.model.training:
            raise PermissionError("Training policy forward must retain eval/dropout-zero mode")
        self.reserve("extra_forward_sequences", 1, purpose=purpose, completion_tokens=len(tokens))
        kwargs = dict(prepared["inputs"])
        input_ids = kwargs.pop("input_ids")
        if input_ids.shape[0] != 1:
            raise PermissionError("SR-F1 teacher forcing requires one complete sequence")
        if kwargs.get("mm_token_type_ids") is None:
            raise RuntimeError("Teacher forcing requires original native multimodal token types")
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        cache = functional_training_cache(base.config)
        kwargs.update(past_key_values=cache, use_cache=True, logits_to_keep=1)
        kwargs["position_ids"] = base._prepare_position_ids_for_generation(input_ids, kwargs)
        before = self.image_calls
        selected, entropies = [], []
        calls = dict(
            prefill_model_forward_calls=0,
            decode_model_forward_calls=0,
            prefill_input_tokens=0,
            decode_input_tokens=0,
            completed_model_forward_calls=0,
        )
        complete = False
        try:
            with (
                cache_compatible_autograd(self.model),
                torch.enable_grad() if grad else torch.no_grad(),
            ):
                for index, token in enumerate(tokens):
                    # Native one-token conv updates write their input cache in place.
                    # Clone without detach so older graph nodes own unchanged storage.
                    for layer in cache.layers:
                        for state_idx, state in getattr(layer, "conv_states", {}).items():
                            if state is not None:
                                layer.conv_states[state_idx] = state.clone()
                    model_inputs = base.prepare_inputs_for_generation(
                        input_ids,
                        next_sequence_length=None if index == 0 else 1,
                        is_first_iteration=index == 0,
                        **kwargs,
                    )
                    phase = "prefill" if index == 0 else "decode"
                    calls[phase + "_model_forward_calls"] += 1
                    calls[phase + "_input_tokens"] += model_inputs["input_ids"].numel()
                    output = self.model(**model_inputs, return_dict=True)
                    calls["completed_model_forward_calls"] += 1
                    if (
                        output.past_key_values is not cache
                        or cache.get_seq_length() != input_ids.shape[1]
                    ):
                        raise RuntimeError("Native teacher-forcing cache was dropped or misaligned")
                    logp = output.logits[0, -1].float().log_softmax(-1)
                    selected.append(logp[token])
                    if not grad:
                        entropies.append(-(logp.exp() * logp).sum())
                    kwargs = base._update_model_kwargs_for_generation(
                        output, kwargs, is_encoder_decoder=False
                    )
                    input_ids = torch.cat([input_ids, input_ids.new_tensor([[token]])], dim=1)
            if prepared["routing"]["image_token_count"] and self.image_calls <= before:
                raise RuntimeError("Forward did not call visual encoder")
            result = dict(
                logprobs=torch.stack(selected),
                entropy=None if grad else torch.stack(entropies),
                vision_forward_calls=self.image_calls - before,
                cached_model_forward_calls=len(tokens),
            )
            complete = True
            return result
        finally:
            # One durable accounting row per sequence, including failed forwards.
            # The sequence was reserved before execution; these are actual calls,
            # with no GPU-hour or token-budget acceptance threshold.
            self.reserve(
                "cached_training_model_forward_calls",
                calls["prefill_model_forward_calls"] + calls["decode_model_forward_calls"],
                purpose=purpose,
                **calls,
                requested_completion_tokens=len(tokens),
                scored_completion_tokens=len(selected),
                vision_forward_calls=self.image_calls - before,
                status="COMPLETE" if complete else "TECHNICAL_FAILED",
            )


def runtime_account(root, task_id):
    """Durably record real attempted work; there is no GPU-hour stop budget."""
    import fcntl
    import json

    path = Path(root) / "accounting" / (task_id + ".jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)

    def reserve(kind, count, metadata):
        if type(count) is not int or count < 0:
            raise ValueError("Nonnegative integer accounting count required")
        record = dict(
            plan_id=PLAN_ID,
            task_id=task_id,
            kind=kind,
            count=count,
            metadata=metadata,
            pid=os.getpid(),
            time_ns=time.time_ns(),
            attempt_id=os.environ.get("SR_F1_ATTEMPT_ID"),
        )
        with path.open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle, fcntl.LOCK_UN)

    return reserve


def verified_adapter(root, model_id, step=96):
    root = Path(root)
    path = root / (
        "COMMON_START.json"
        if model_id == "SRF1_COMMON_START"
        else str(Path("states") / model_id / f"step-{step:02d}" / "IDENTITY.json")
    )
    entry = read_json(path)
    if entry.get("plan_id") != PLAN_ID or entry.get("status") != "VERIFIED":
        raise PermissionError("No verified SR-F1 adapter identity")
    if entry.get("freeze_sha256") != file_hash(root / "EXECUTION_FREEZE.json"):
        raise PermissionError("Adapter belongs to another execution freeze")
    observed = adapter_identity(root, entry["adapter_path"])
    if any(entry.get(key) != value for key, value in observed.items()):
        raise PermissionError("Published adapter bytes changed")
    return entry


def load_runtime(plan, root, *, state_id=None, step=96, account=None):
    from peft import PeftModel

    root = Path(root)
    identity = read_json(root / "MODEL_ENVIRONMENT_IDENTITY.json")
    if (
        identity.get("model_revision") != plan["model"]["revision"]
        or identity.get("model_weights_hash") != plan["model"]["prior_composite_weight_hash"]
    ):
        raise PermissionError("SR-F1 requires the exact original untrained 9B snapshot")
    determinism = configure_audited_backend()
    hardware = actual_cuda_identity()
    runtime = SRRuntime(identity["model_path"], account=account)
    runtime.verify_identity(identity)
    runtime.identity.update(
        hardware=hardware,
        determinism=determinism,
        base_model_weights_hash=identity["model_weights_hash"],
    )
    if state_id is not None:
        entry = verified_adapter(root, state_id, step)
        runtime.model = PeftModel.from_pretrained(
            runtime.model, bounded_path(root, entry["adapter_path"]), is_trainable=False
        )
        runtime.adapter_path = str(bounded_path(root, entry["adapter_path"]))
        runtime.identity.update(entry)
    return runtime


def load_for_evaluation(plan, root, model_id, step=96):
    return load_runtime(
        plan,
        root,
        state_id=model_id,
        step=step,
        account=runtime_account(root, f"eval-{model_id}-{step}"),
    )


def publish_adapter(root, runtime, model_id, *, step=None, extra=None):
    import shutil
    import tempfile

    from .contract import digest

    root = Path(root)
    relative = Path("states") / model_id
    if step is not None:
        relative /= f"step-{step:02d}"
    destination = bounded_path(root, relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".pending-adapter-", dir=destination.parent))
    try:
        runtime.model.save_pretrained(staging, safe_serialization=True)
        if destination.exists():
            # IDENTITY.json is a receipt, never part of the adapter payload hash.
            existing = {
                p.name: file_hash(p)
                for p in destination.iterdir()
                if p.is_file() and p.name != "IDENTITY.json"
            }
            new = {p.name: file_hash(p) for p in staging.iterdir() if p.is_file()}
            if existing != new:
                raise PermissionError("Existing published adapter differs from committed state")
        else:
            os.rename(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    files = {
        p.name: file_hash(p)
        for p in destination.iterdir()
        if p.is_file() and p.name != "IDENTITY.json"
    }
    entry = dict(
        plan_id=PLAN_ID,
        status="VERIFIED",
        model_id=model_id,
        step=step,
        freeze_sha256=file_hash(root / "EXECUTION_FREEZE.json"),
        base_model_weights_hash=read_json(root / "MODEL_ENVIRONMENT_IDENTITY.json")[
            "model_weights_hash"
        ],
        adapter_path=str(relative),
        adapter_file_hashes=files,
        adapter_hash=hash_json(files),
        trainable_state_hash=state_hash(trainable_state(runtime.model)),
        **(extra or {}),
    )
    # match the generic adapter_identity hash convention.
    from mm_dev.contract import digest as adapter_digest

    entry["adapter_hash"] = adapter_digest(files)
    entry["model_hash"] = digest({"base": entry["base_model_weights_hash"], "adapter": files})
    marker = destination / "IDENTITY.json"
    if marker.exists():
        if read_json(marker) != entry:
            raise PermissionError("Published adapter receipt differs")
    else:
        atomic_json(marker, entry, exclusive=True)
    runtime.identity.update(entry)
    runtime.adapter_path = str(destination)
    return entry


def adapter_identity(root, relative):
    from mm_dev.contract import digest as adapter_digest

    path = bounded_path(root, relative)
    files = {
        p.name: file_hash(p) for p in path.iterdir() if p.is_file() and p.name != "IDENTITY.json"
    }
    if not {"adapter_config.json", "adapter_model.safetensors"}.issubset(files):
        raise PermissionError("Missing serialized adapter payload")
    return dict(
        adapter_path=str(relative), adapter_file_hashes=files, adapter_hash=adapter_digest(files)
    )


def _coverage(record, task):
    from .contract import score

    result = score(record["raw_text"], task["world"], task["query"])
    # The reference scorer separates legal entities/values from semantic coverage.
    return bool(result["L_json"] and result["L_answer"] and result["L_evidence"])


def format_panel(runtime, root, pool, label, boundary=None):
    from .contract import digest, score, stable_seed
    from .data import load_inputs, load_tasks

    root = Path(root)
    inputs, tasks = load_inputs(root), load_tasks(root)
    qids = sorted(q for q, t in tasks.items() if t["pool"] == pool)
    if len(qids) != 32:
        raise PermissionError("FORMAT panel must contain exactly 32 registered prompts")
    directory = root / "engineering" / "format" / label
    records = []
    policy = state_hash(trainable_state(runtime.model))
    for qid in qids:
        for index in range(8):
            if boundary and boundary["requested"]:
                from .training import LeaseEnding

                raise LeaseEnding("PREEMPTION_DURING_FORMAT_PANEL")
            path = directory / f"{qid}-{index}.json"
            seed = stable_seed(PLAN_ID, "FORMAT", pool, qid, index)
            expected = dict(
                qid=qid,
                sample_index=index,
                seed=seed,
                policy_hash=policy,
                input_hash=digest(inputs[qid]),
            )
            if path.exists():
                record = read_json(path)
                if any(record.get(k) != v for k, v in expected.items()):
                    raise PermissionError("FORMAT raw identity changed")
            else:

                def persist(raw, path=path, expected=expected):
                    value = {**expected, **raw}
                    value["record_hash"] = digest(value)
                    atomic_json(path, value, exclusive=True)

                runtime.generate_training(inputs[qid], root, seed, on_completion=persist)
                record = read_json(path)
            if record.get("record_hash") != digest(
                {k: v for k, v in record.items() if k != "record_hash"}
            ):
                raise PermissionError("FORMAT raw response bytes changed")
            if record["generation_status"] != "COMPLETE":
                raise RuntimeError("FORMAT contains technical generation failure")
            record["score"] = score(record["raw_text"], tasks[qid]["world"], tasks[qid]["query"])
            records.append(record)
    covered = sum(_coverage(r, tasks[r["qid"]]) for r in records)
    receipt = dict(
        panel=pool,
        label=label,
        policy_hash=policy,
        responses=len(records),
        covered=covered,
        coverage=covered / len(records),
        truncated=sum(r["truncated"] for r in records),
        coverage_uses_gold_accuracy=False,
        component_means={
            k: sum(r["score"][k] for r in records) / len(records) for k in ("A", "E", "P", "J")
        },
    )
    atomic_json(directory / "COVERAGE.json", receipt)
    return receipt


def _establish_common_start(plan, root, boundary):
    """Zero-output init, format-only trigger, at most one fixed shared bridge."""
    import gc

    from mm_core.training import configure_training, frozen_hash

    from .data import load_inputs, load_tasks
    from .training import run_common_bridge

    root = Path(root)
    final_receipt = root / "FORMAT_AND_BRIDGE_RECEIPT.json"
    if final_receipt.exists():
        result = read_json(final_receipt)
        if result["status"] == "COMPLETE":
            verified_adapter(root, "SRF1_COMMON_START")
            return result
        raise PermissionError("Protocol is blocked; repeated bridge is forbidden")
    runtime = load_runtime(plan, root, account=runtime_account(root, "COMMON_START"))
    inputs, tasks = load_inputs(root), load_tasks(root)
    zero_path = root / "COMMON_ZERO_LORA.json"
    if zero_path.exists():
        from peft import PeftModel

        zero = read_json(zero_path)
        actual = adapter_identity(root, zero["adapter_path"])
        if any(zero.get(k) != v for k, v in actual.items()):
            raise PermissionError("Common zero LoRA changed")
        runtime.model = PeftModel.from_pretrained(
            runtime.model, bounded_path(root, zero["adapter_path"]), is_trainable=False
        )
        configure_training(runtime)
    else:
        qid = sorted(q for q, t in tasks.items() if t["pool"] == "ENGINE")[0]
        prepared = runtime.prepare(inputs[qid], root)
        kwargs = dict(prepared["inputs"], logits_to_keep=1, use_cache=False)
        runtime.reserve("extra_forward_sequences", 1, purpose="zero_lora_base")
        with runtime.torch.no_grad():
            before = runtime.model(**kwargs).logits.detach().cpu()
        seed_all(plan["model"]["lora"]["init_seed"])
        configure_training(runtime)
        for name, p in runtime.model.named_parameters():
            if "lora_B" in name and bool(p.detach().ne(0).any()):
                raise RuntimeError("Common initialization did not have exact zero B matrices")
        runtime.reserve("extra_forward_sequences", 1, purpose="zero_lora_invariance")
        with runtime.torch.no_grad():
            after = runtime.model(**kwargs).logits.detach().cpu()
        if not runtime.torch.equal(before, after):
            raise RuntimeError("Zero-output LoRA changed native logits")
        zero = publish_adapter(
            root,
            runtime,
            "SRF1_COMMON_ZERO",
            extra={
                "initialization_seed": plan["model"]["lora"]["init_seed"],
                "zero_output_exact": True,
                "base_logits_hash": state_hash(before),
                "frozen_base_hash": frozen_hash(runtime.model),
            },
        )
        atomic_json(zero_path, zero, exclusive=True)
    runtime.identity.update(zero)
    runtime.adapter_path = str(bounded_path(root, zero["adapter_path"]))
    initial = format_panel(runtime, root, "FORMAT", "before", boundary)
    bridge = None
    confirmation, repeated = None, None
    if initial["coverage"] < 0.95:
        if initial["truncated"]:
            blocked = dict(
                status="PROTOCOL_BLOCKED",
                reason="FORMAT_TRUNCATION_REQUIRES_TECHNICAL_REVIEW",
                format_before=initial,
                bridge_executed=False,
                artifacts=[str(zero_path.relative_to(root))],
            )
            atomic_json(final_receipt, blocked, exclusive=True)
            return blocked
        # Native template, parser CPU contract and image routing are separately gated.
        preflight = read_json(root / "PROCESSOR_PREFLIGHT.json")
        if preflight.get("status") not in ("PASS", "VERIFIED", "COMPLETE"):
            raise PermissionError("Protocol coverage cannot trigger bridge before input preflight")
        bridge = run_common_bridge(runtime, root, plan, boundary)
        if bridge["status"] == "CHECKPOINTED":
            return bridge
        confirmation = format_panel(runtime, root, "FORMAT_CONFIRM", "confirm", boundary)
        repeated = format_panel(runtime, root, "FORMAT", "after", boundary)
        if confirmation["coverage"] < 0.95:
            blocked = dict(
                status="PROTOCOL_BLOCKED",
                reason="ONE_BRIDGE_CONFIRMATION_BELOW_95_PERCENT",
                format_before=initial,
                bridge=bridge,
                format_confirmation=confirmation,
                format_after=repeated,
                artifacts=[str(zero_path.relative_to(root))],
            )
            atomic_json(final_receipt, blocked, exclusive=True)
            return blocked
    entry = publish_adapter(
        root,
        runtime,
        "SRF1_COMMON_START",
        extra={
            "bridge_executed": bridge is not None,
            "zero_output_initialization": zero["trainable_state_hash"],
        },
    )
    common_path = root / "COMMON_START.json"
    if common_path.exists():
        if read_json(common_path) != entry:
            raise PermissionError("Partially published common start identity differs")
    else:
        atomic_json(common_path, entry, exclusive=True)
    result = dict(
        status="COMPLETE",
        format_before=initial,
        bridge=bridge,
        format_confirmation=confirmation,
        format_after=repeated,
        artifacts=["COMMON_START.json", "COMMON_ZERO_LORA.json", "FORMAT_AND_BRIDGE_RECEIPT.json"],
        metadata={"common_start": "SRF1_COMMON_START", "bridge_executed": bridge is not None},
    )
    atomic_json(final_receipt, result, exclusive=True)
    del runtime
    gc.collect()
    return result


def establish_common_start(plan, root):
    from .training import LeaseEnding, stop_at_committed_boundary

    root = Path(root)
    with stop_at_committed_boundary() as boundary:
        try:
            return _establish_common_start(plan, root, boundary)
        except LeaseEnding:
            # FORMAT has independently seeded immutable raw slots and the already
            # persisted common zero adapter. Neither needs a gradient/Adam state.
            zero = read_json(root / "COMMON_ZERO_LORA.json")
            artifacts = ["COMMON_ZERO_LORA.json"]
            artifacts.extend(
                str(p.relative_to(root))
                for p in bounded_path(root, zero["adapter_path"]).iterdir()
                if p.is_file()
            )
            artifacts.extend(
                str(p.relative_to(root)) for p in (root / "engineering/format").rglob("*.json")
            )
            bridge_latest = root / "engineering/bridge/checkpoints/LATEST.json"
            if bridge_latest.exists():
                latest = read_json(bridge_latest)
                artifacts.extend(
                    [
                        str(bridge_latest.relative_to(root)),
                        str((bridge_latest.parent / latest["path"]).relative_to(root)),
                    ]
                )
            return dict(
                status="CHECKPOINTED",
                reason="PREEMPTION",
                artifacts=artifacts,
                metadata=dict(
                    full_state=True,
                    identity_verified=True,
                    next_update=None,
                    run_id="SRF1_COMMON_START",
                ),
            )
