"""Native Qwen3.5-9B runtime; importing does not load weights or contact a network."""

from __future__ import annotations

import hashlib
import inspect
import json
import random
import time
from pathlib import Path

MODEL_ID = "Qwen/Qwen3.5-9B"
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}
FULL_ATTENTION_LAYERS = tuple(range(3, 32, 4))
CANVAS = (896, 672)
GENERATION = dict(
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    top_k=0,
    repetition_penalty=1.0,
    max_new_tokens=192,
    num_beams=1,
    num_return_sequences=1,
    min_length=0,
    min_new_tokens=0,
    typical_p=1.0,
    epsilon_cutoff=0.0,
    eta_cutoff=0.0,
    min_p=None,
    no_repeat_ngram_size=0,
    encoder_repetition_penalty=1.0,
    forced_bos_token_id=None,
    forced_eos_token_id=None,
    bad_words_ids=None,
    force_words_ids=None,
    suppress_tokens=None,
    begin_suppress_tokens=None,
    sequence_bias=None,
    constraints=None,
    penalty_alpha=None,
    renormalize_logits=False,
    remove_invalid_values=False,
    guidance_scale=None,
    watermarking_config=None,
    token_healing=False,
    use_cache=True,
    return_dict_in_generate=True,
    output_scores=False,
    output_logits=False,
)


def hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _callable_identity(value):
    if value is None:
        return None
    result = dict(
        module=getattr(value, "__module__", type(value).__module__),
        qualified_name=getattr(value, "__qualname__", type(value).__qualname__),
    )
    try:
        source = inspect.getsourcefile(value)
    except (TypeError, OSError):
        source = None
    if source and Path(source).is_file():
        result["source_sha256"] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    wrapped = getattr(value, "__wrapped__", None)
    if wrapped is not None and wrapped is not value:
        result["wrapped_callable"] = _callable_identity(wrapped)
    return result


def linear_kernel_identity(model=None, *, expected_linear_count=None):
    """Inspect optional native kernel selection without loading weights or executing it.

    Eager attention selects full attention only. Gated DeltaNet independently
    chooses optional FLA/causal-conv1d/norm implementations, which must also be
    frozen. With a loaded model, verify its actual module callables match the
    no-weight inspection used before inference.
    """
    from importlib.metadata import PackageNotFoundError, version

    import torch
    from transformers.integrations import hub_kernels
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native

    versions = {}
    for package in (
        "flash-linear-attention",
        "fla-core",
        "causal-conv1d",
        "triton",
        "kernels",
        "mamba-ssm",
    ):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    chosen = dict(
        causal_conv1d_fn=native.causal_conv1d_fn,
        causal_conv1d_update=native.causal_conv1d_update or native.torch_causal_conv1d_update,
        chunk_gated_delta_rule=native.chunk_gated_delta_rule or native.torch_chunk_gated_delta_rule,
        recurrent_gated_delta_rule=native.fused_recurrent_gated_delta_rule
        or native.torch_recurrent_gated_delta_rule,
        norm_class=native.FusedRMSNormGated or native.Qwen3_5RMSNormGated,
        forward=native.Qwen3_5GatedDeltaNet.forward,
    )
    selected = {key: _callable_identity(value) for key, value in chosen.items()}
    if model is not None:
        modules = [
            module for module in model.modules() if isinstance(module, native.Qwen3_5GatedDeltaNet)
        ]
        if not modules or (
            expected_linear_count is not None and len(modules) != expected_linear_count
        ):
            raise RuntimeError("Loaded native linear-attention module count differs")
        for module in modules:
            actual = {
                name: getattr(module, name)
                for name in (
                    "causal_conv1d_fn",
                    "causal_conv1d_update",
                    "chunk_gated_delta_rule",
                    "recurrent_gated_delta_rule",
                    "forward",
                )
            }
            actual["norm_class"] = type(module.norm)
            if {key: _callable_identity(value) for key, value in actual.items()} != selected:
                raise RuntimeError(
                    "Loaded linear-attention kernel differs from inspected native selection"
                )
    return dict(
        optional_package_versions=versions,
        selected_callables=selected,
        unfused_prefill_conv=_callable_identity(torch.nn.Conv1d.forward),
        unfused_prefill_activation=_callable_identity(torch.nn.functional.silu),
        all_fast_kernels_available=bool(native.is_fast_path_available),
        hub_kernels_enabled=bool(hub_kernels._kernels_enabled),
        hub_kernels_environment=hub_kernels._TRANSFORMERS_USE_HUB_KERNELS,
    )


def validate_model_config(config):
    """Reject other Qwen sizes/families and changed hybrid-layer definitions."""
    text = config.get("text_config", {})
    expected_layers = [
        "full_attention" if i in FULL_ATTENTION_LAYERS else "linear_attention" for i in range(32)
    ]
    expected = dict(
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=32,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=256,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    if config.get("model_type") != "qwen3_5" or any(text.get(k) != v for k, v in expected.items()):
        raise ValueError("Required local Qwen3.5-9B architecture does not match")
    if text.get("layer_types") != expected_layers:
        raise ValueError("Qwen3.5-9B hybrid attention layout differs from the frozen definition")
    vision = config.get("vision_config", {})
    if (
        vision.get("patch_size"),
        vision.get("spatial_merge_size"),
        vision.get("out_hidden_size"),
    ) != (16, 2, 4096):
        raise ValueError("Qwen3.5-9B native vision architecture does not match")
    return dict(
        model_id=MODEL_ID,
        model_type="qwen3_5",
        architecture_verified=True,
        full_attention_layers=list(FULL_ATTENTION_LAYERS),
        chat_template_kwargs=dict(CHAT_TEMPLATE_KWARGS),
        chat_template_kwargs_hash=hash_json(CHAT_TEMPLATE_KWARGS),
    )


def model_file_path(model_path, relative):
    """Resolve local files, permitting HF snapshot links only inside their repository.

    Hugging Face snapshots link into the sibling blobs directory. A direct model
    directory remains confined to itself, and neither form permits traversal names
    or symlinks that escape its allowed repository boundary.
    """
    name = Path(relative)
    if not str(relative) or name.is_absolute() or ".." in name.parts or name == Path("."):
        raise ValueError("Model filename must be a nonempty safe relative path")
    snapshot = Path(model_path).resolve(strict=True)
    boundary = snapshot.parent.parent if snapshot.parent.name == "snapshots" else snapshot
    target = (snapshot / name).resolve(strict=True)
    if not target.is_relative_to(boundary) or not target.is_file():
        raise ValueError("Model file resolves outside the permitted repository or is not a file")
    return target


def seed_all(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def field_value_spans(text):
    """Exact full-object JSON value spans; explanatory prose and duplicate keys fail."""
    decoder = json.JSONDecoder()
    i = len(text) - len(text.lstrip())
    if i >= len(text) or text[i] != "{":
        return {}
    i += 1
    spans = {}
    try:
        while True:
            while text[i].isspace():
                i += 1
            if text[i] == "}":
                i += 1
                break
            key, i = decoder.raw_decode(text, i)
            if not isinstance(key, str) or key in spans:
                return {}
            while text[i].isspace():
                i += 1
            if text[i] != ":":
                return {}
            i += 1
            while text[i].isspace():
                i += 1
            start = i
            _, i = decoder.raw_decode(text, i)
            spans[key] = (start, i)
            while text[i].isspace():
                i += 1
            if text[i] == "}":
                i += 1
                break
            if text[i] != ",":
                return {}
            i += 1
        if text[i:].strip():
            return {}
    except (ValueError, IndexError):
        return {}
    return {k: spans[k] for k in ("readings", "answer") if k in spans}


def assign_field_tokens(text, token_character_spans):
    """Largest value-character overlap wins; readings wins ties. EOS is unassigned."""
    fields = field_value_spans(text)
    assigned = {name: [] for name in fields}
    overlaps = []
    for index, (start, stop) in enumerate(token_character_spans):
        choices = [
            (max(0, min(stop, hi) - max(start, lo)), key) for key, (lo, hi) in fields.items()
        ]
        choices = [choice for choice in choices if choice[0] > 0]
        if not choices:
            continue
        chosen = sorted(choices, key=lambda x: (-x[0], x[1] != "readings"))[0][1]
        assigned[chosen].append(index)
        if len(choices) > 1:
            overlaps.append(dict(token_index=index, overlaps=choices, assigned=chosen))
    return assigned, overlaps


def gold_completion(row):
    from decimal import Decimal

    values = row.get("true_values", row.get("true_values_decimal"))
    if values is None:
        raise ValueError("Gold field NLL requires registered true_values")
    values = [Decimal(str(x)) for x in values]
    answer = {
        "sum": lambda: sum(values),
        "difference": lambda: values[0] - values[1],
        "range": lambda: max(values) - min(values),
    }[row["operation"]]()
    return (
        '{"readings": ['
        + ", ".join(format(v, "f") for v in values)
        + '], "answer": '
        + format(answer, "f")
        + "}"
    )


class QwenRuntime:
    """Callers own gates; account(kind,count,metadata) reserves before all model work."""

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
        import torch
        import transformers
        from transformers import AutoProcessor, GenerationConfig, Qwen3_5ForConditionalGeneration

        self.torch, self.device, self.account = torch, device, account
        self.model_path = str(Path(model_path).resolve())
        config = json.loads(Path(self.model_path, "config.json").read_text())
        architecture = validate_model_config(config)
        if str(device).startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; no model execution occurred")
            if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
                raise RuntimeError("bf16 hardware support is not verified")
        if dtype not in {"bfloat16", "float32"}:
            raise ValueError("Only frozen bf16 or float32 precision is supported")
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=False,
            min_pixels=896 * 672,
            max_pixels=896 * 672,
        )
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=getattr(torch, dtype),
            attn_implementation=attention_backend,
        ).to(device)
        self.adapter_path = str(Path(adapter_path).resolve()) if adapter_path else None
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path, is_trainable=False)
        self.model.eval()
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = [int(x) for x in (eos if isinstance(eos, list) else [eos]) if x is not None]
        if not self.eos_ids:
            raise ValueError("Original model EOS set missing")
        self.generation_config = GenerationConfig(
            **GENERATION,
            eos_token_id=self.eos_ids,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            bos_token_id=self.processor.tokenizer.bos_token_id,
        )
        self.image_calls = 0
        self._vision_hook = self._visual_module().register_forward_pre_hook(self._mark_vision)
        kernels = linear_kernel_identity(self.model, expected_linear_count=24)
        self.identity = dict(
            **architecture,
            linear_kernel_identity=kernels,
            linear_kernel_identity_hash=hash_json(kernels),
            transformers_version=transformers.__version__,
            torch_version=torch.__version__,
            dtype=dtype,
            attention_backend=attention_backend,
            processor_hash=hash_json(self.processor.to_dict()),
            tokenizer_hash=hash_json(self.processor.tokenizer.get_vocab()),
            chat_template_hash=hash_json(self.processor.chat_template),
            generation_config_expanded=self.generation_config.to_dict(),
            canvas_pixels=list(CANVAS),
            active_adapter=self.adapter_path,
        )

    def verify_identity(self, freeze, common=None):
        """Verify loaded local bytes/configuration against the preregistered identity."""
        from .execution import checked_path, object_hash, sha256_file

        files = freeze.get("model_files")
        if not isinstance(files, list) or not files:
            raise PermissionError("Freeze lacks actual model file identities")
        if freeze.get("model_id") != MODEL_ID:
            raise PermissionError("Freeze specifies a different model family/size")
        if freeze.get("chat_template_kwargs") != CHAT_TEMPLATE_KWARGS:
            raise PermissionError("Non-thinking chat-template kwargs must be frozen")
        if freeze.get("chat_template_kwargs_hash") != hash_json(CHAT_TEMPLATE_KWARGS):
            raise PermissionError("Chat-template kwargs identity mismatch")
        kernels = self.identity.get("linear_kernel_identity")
        if not kernels or freeze.get("linear_kernel_identity") != kernels:
            raise PermissionError("Native linear-attention kernel identity differs from freeze")
        if freeze.get("linear_kernel_identity_hash") != hash_json(kernels):
            raise PermissionError("Native linear-attention kernel identity hash differs")
        observed = {}
        for entry in files:
            path = model_file_path(self.model_path, entry["name"])
            if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
                raise PermissionError("Model/processor file identity changed: " + entry["name"])
            observed[entry["name"]] = entry["sha256"]
        weights = {
            name: sha
            for name, sha in observed.items()
            if name.endswith(".safetensors") or name == "config.json" or name.endswith("index.json")
        }
        if object_hash(weights) != freeze["model_weights_hash"]:
            raise PermissionError("Model aggregate identity changed")
        for name in ("processor_hash", "chat_template_hash"):
            if self.identity[name] != freeze[name]:
                raise PermissionError("Runtime identity differs: " + name)
        if self.identity["generation_config_expanded"] != freeze["generation_config_expanded"]:
            raise PermissionError("Expanded generation configuration differs from freeze")
        for name in ("torch_version", "transformers_version"):
            if name in freeze and self.identity[name] != freeze[name]:
                raise PermissionError("Runtime library differs: " + name)
        if self.adapter_path:
            adapter_files = (common or {}).get("adapter_file_hashes")
            if not adapter_files:
                raise PermissionError("Active adapter lacks frozen file identities")
            for name, expected in adapter_files.items():
                if sha256_file(checked_path(self.adapter_path, name)) != expected:
                    raise PermissionError("Active adapter identity changed")
            expected_common = object_hash(
                dict(
                    base_model_weights_hash=freeze["model_weights_hash"],
                    adapter_file_hashes=adapter_files,
                )
            )
            if common.get("model_hash") != expected_common:
                raise PermissionError("Common-start model identity differs")
        elif common and common.get("model_hash") != freeze["model_weights_hash"]:
            raise PermissionError("Base common-start identity differs")
        return dict(
            verified=True,
            model_files=len(files),
            active_adapter=self.adapter_path,
            model_hash=(common or {}).get("model_hash", freeze["model_weights_hash"]),
        )

    @classmethod
    def processor_only(cls, model_path):
        """CPU-only actual processor inspection; no weight loading or generation."""
        from types import SimpleNamespace

        from transformers import AutoConfig, AutoProcessor

        instance = cls.__new__(cls)
        instance.device = "cpu"
        architecture = validate_model_config(
            json.loads(Path(model_path, "config.json").read_text())
        )
        instance.processor = AutoProcessor.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=False,
            min_pixels=896 * 672,
            max_pixels=896 * 672,
        )
        instance.model = SimpleNamespace(
            config=AutoConfig.from_pretrained(
                str(model_path), local_files_only=True, trust_remote_code=False
            )
        )
        kernels = linear_kernel_identity()
        instance.identity = dict(
            **architecture,
            linear_kernel_identity=kernels,
            linear_kernel_identity_hash=hash_json(kernels),
            processor_hash=hash_json(instance.processor.to_dict()),
            chat_template_hash=hash_json(instance.processor.chat_template),
        )
        return instance

    def _visual_module(self):
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        visual = getattr(getattr(base, "model", base), "visual", None)
        if visual is None:
            visual = getattr(base, "visual", None)
        if visual is None:
            raise RuntimeError("Actual visual encoder module cannot be verified")
        return visual

    def _mark_vision(self, *args):
        self.image_calls += 1

    def reserve(self, kind, count, **metadata):
        if self.account is None:
            raise RuntimeError("Budget reservation callback required before model execution")
        self.account(kind, count, metadata)

    def prepare(self, row, run_root):
        from PIL import Image

        root = Path(run_root).resolve()
        image_path = (root / row["image_path"]).resolve()
        if not image_path.is_relative_to(root):
            raise ValueError("Image escapes registered run root")
        image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
        expected = row.get("image_sha256", row.get("image_hash"))
        if expected and image_hash != expected:
            raise ValueError("Image no longer matches registered hash")
        with Image.open(image_path) as original:
            if original.size != CANVAS:
                raise ValueError("Frozen 896x672 image contract changed")
            rgb = original.convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": row["prompt"]},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **CHAT_TEMPLATE_KWARGS
        )
        if not text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"):
            raise RuntimeError("Native chat template did not apply enable_thinking=False")
        inputs = self.processor(text=[text], images=[rgb], padding=False, return_tensors="pt")
        if "pixel_values" not in inputs or "image_grid_thw" not in inputs:
            raise RuntimeError("Processor did not route real pixels and image_grid_thw")
        grid = inputs["image_grid_thw"].tolist()
        ip = self.processor.image_processor
        patch, merge = int(getattr(ip, "patch_size", 16)), int(getattr(ip, "merge_size", 2))
        if len(grid) != 1 or grid[0][0] != 1:
            raise ValueError("Exactly one still image per request required")
        processed_size = (int(grid[0][2]) * patch, int(grid[0][1]) * patch)
        if processed_size != CANVAS:
            raise ValueError(f"Processor resized frozen image to {processed_size}")
        image_id = getattr(self.model.config, "image_token_id", None)
        if image_id is None:
            raise ValueError("Model image token ID missing")
        actual = int((inputs["input_ids"] == image_id).sum())
        if actual <= 0 or actual != grid[0][0] * grid[0][1] * grid[0][2] // merge**2:
            raise RuntimeError("Image grid and model image-token count differ")
        mm_types = inputs.get("mm_token_type_ids")
        if mm_types is None or mm_types.shape != inputs["input_ids"].shape:
            raise RuntimeError("Qwen3.5 native multimodal token type IDs are missing")
        if not bool(((mm_types == 1) == (inputs["input_ids"] == image_id)).all()):
            raise RuntimeError("Image token flags disagree with native multimodal token types")
        pixels = inputs["pixel_values"].detach().cpu().contiguous()
        pixel_hash = hashlib.sha256(pixels.numpy().tobytes()).hexdigest()
        if row.get("processed_pixel_sha256") and row["processed_pixel_sha256"] != pixel_hash:
            raise ValueError("Actual processor tensor changed since freeze")
        routing = dict(
            processed_pixel_sha256=pixel_hash,
            pixels_per_delta=488 / 100 * processed_size[1] / 672,
            source_image_sha256=image_hash,
            source_size=list(CANVAS),
            processed_size=list(processed_size),
            image_grid_thw=grid,
            pixel_values_shape=list(inputs["pixel_values"].shape),
            input_ids_shape=list(inputs["input_ids"].shape),
            mm_token_type_ids_shape=list(mm_types.shape),
            mm_token_type_ids_sha256=hash_json(mm_types.tolist()),
            chat_template_kwargs=dict(CHAT_TEMPLATE_KWARGS),
            chat_template_kwargs_hash=hash_json(CHAT_TEMPLATE_KWARGS),
            image_token_count=actual,
            image_token_id=image_id,
            processor_hash=self.identity["processor_hash"],
            chat_template_hash=self.identity["chat_template_hash"],
            chat_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
            input_ids_sha256=hash_json(inputs["input_ids"].tolist()),
        )
        return dict(inputs=inputs.to(self.device), routing=routing, chat_text=text)

    def encode_completion(self, text, *, eos=True):
        ids = self.processor.tokenizer.encode(text, add_special_tokens=False)
        return [*ids, self.eos_ids[0]] if eos else ids

    def sequence_forward(self, prepared, tokens, *, purpose, grad=False):
        """Raw model log probabilities on the actual supplied completion prefixes."""
        torch = self.torch
        if not tokens:
            raise ValueError("Cannot score empty completion")
        self.reserve("extra_forward_sequences", 1, purpose=purpose, completion_tokens=len(tokens))
        inputs = dict(prepared["inputs"])
        prompt_len = inputs["input_ids"].shape[1]
        completion = torch.tensor([tokens], device=self.device, dtype=torch.long)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], completion], dim=1)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        inputs.pop("token_type_ids", None)
        if "mm_token_type_ids" not in inputs or inputs["mm_token_type_ids"].shape[1] != prompt_len:
            raise RuntimeError("Teacher forcing requires original native multimodal token types")
        inputs["mm_token_type_ids"] = torch.cat(
            [inputs["mm_token_type_ids"], torch.zeros_like(completion)], dim=1
        )
        inputs["use_cache"] = False
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        for module in (base, getattr(base, "model", None)):
            if module is not None and hasattr(module, "rope_deltas"):
                module.rope_deltas = None
        keep = "logits_to_keep" in inspect.signature(base.forward).parameters
        if keep:
            inputs["logits_to_keep"] = len(tokens) + 1
        before = self.image_calls
        with torch.enable_grad() if grad else torch.no_grad():
            output = self.model(**inputs)
            logits = (
                output.logits[0, :-1]
                if keep
                else output.logits[0, prompt_len - 1 : prompt_len + len(tokens) - 1]
            )
            logp = logits.float().log_softmax(-1)
            chosen = logp.gather(-1, completion[0, :, None]).squeeze(-1)
            entropy = -(logp.exp() * logp).sum(-1) if not grad else None
        if self.image_calls <= before:
            raise RuntimeError("Forward did not call visual encoder")
        return dict(
            logprobs=chosen, entropy=entropy, vision_forward_calls=self.image_calls - before
        )

    def field_scores(self, prepared, tokens, *, provenance):
        scored = self.sequence_forward(prepared, tokens, purpose=provenance)
        tok = self.processor.tokenizer
        prefixes = [
            tok.decode(tokens[:i], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for i in range(len(tokens) + 1)
        ]
        if any(not prefixes[i + 1].startswith(prefixes[i]) for i in range(len(tokens))):
            return dict(
                status="TOKEN_CHARACTER_BOUNDARY_UNVERIFIED",
                provenance=provenance,
                field_nll=None,
                tokens=len(tokens),
            )
        assigned, overlaps = assign_field_tokens(
            prefixes[-1], [(len(prefixes[i]), len(prefixes[i + 1])) for i in range(len(tokens))]
        )
        logps, entropy = scored["logprobs"].cpu().tolist(), scored["entropy"].cpu().tolist()
        fields = {}
        for field in ("readings", "answer"):
            indices = assigned.get(field, [])
            fields[field] = dict(
                token_count=len(indices),
                token_indices=indices,
                mean_nll=-sum(logps[i] for i in indices) / len(indices) if indices else None,
                mean_token_entropy=sum(entropy[i] for i in indices) / len(indices)
                if indices
                else None,
            )
        return dict(
            status="MEASURED",
            provenance=provenance,
            distribution="raw_model_before_temperature_or_top_p",
            field_nll=fields,
            token_logprobs=logps,
            token_entropy=entropy,
            boundary_rule="largest_value_character_overlap_then_readings_tie",
            boundary_overlaps=overlaps,
            vision_forward_calls=scored["vision_forward_calls"],
        )

    def generate(self, row, run_root, seed, *, score_fields=True, on_completion=None):
        prepared = self.prepare(row, run_root)
        self.reserve("completion_attempts", 1, question_id=row.get("question_id"), seed=seed)
        seed_all(seed)
        self.model.eval()
        before = self.image_calls
        if str(self.device).startswith("cuda"):
            self.torch.cuda.synchronize()
        start = time.perf_counter()
        with self.torch.no_grad():
            result = self.model.generate(
                **prepared["inputs"], generation_config=self.generation_config
            )
        if str(self.device).startswith("cuda"):
            self.torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if self.image_calls <= before:
            raise RuntimeError("Generate did not call visual encoder")
        prompt_len = prepared["inputs"]["input_ids"].shape[-1]
        tokens = result.sequences[0, prompt_len:].cpu().tolist()
        raw = self.processor.tokenizer.decode(
            tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        record = dict(
            raw_text=raw,
            tokens=tokens,
            raw_tokens=tokens,
            prompt_token_count=prompt_len,
            completion_token_count=len(tokens),
            truncated=bool(
                tokens
                and tokens[-1] not in self.eos_ids
                and len(tokens) >= GENERATION["max_new_tokens"]
            ),
            finish_reason="eos" if tokens and tokens[-1] in self.eos_ids else "length",
            generation_seconds=elapsed,
            seed=seed,
            image_routing={
                **prepared["routing"],
                "generation_vision_forward_calls": self.image_calls - before,
            },
            field_origin="both_model_generated_in_one_completion",
        )
        if on_completion is not None:
            on_completion(record)
        if score_fields:
            start = time.perf_counter()
            record["self_field_surprisal"] = (
                self.field_scores(
                    prepared, tokens, provenance="generated_tokens_actual_self_prefix"
                )
                if tokens
                else None
            )
            record["gold_teacher_forced_field_nll"] = self.field_scores(
                prepared,
                self.encode_completion(gold_completion(row)),
                provenance="gold_completion_answer_prefix_contains_gold_readings",
            )
            record["scoring_seconds"] = time.perf_counter() - start
        return record
