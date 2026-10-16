"""SR-F3-CE native Qwen runtime; import never loads weights or changes backends.

Model calls remain subject to the caller's stage/pool and accounting gates. The
only accepted public input is the three-field output of ``ModelInputStore``.
All teacher masks use offsets from the actual complete processor invocation.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import math
import re
import time
from collections.abc import Mapping
from pathlib import Path

from mm_core.training import capture_rng, restore_rng, state_hash
from mm_core.vl_runtime import GENERATION, hash_json, seed_all
from mm_dev.runtime import bounded_path, file_hash
from sr_f1.json_protocol import AMENDMENT_ID, balanced_cut, decode_generated
from sr_f1.runtime import SRRuntime
from sr_f12.runtime import SRF12Runtime, _repeat_prompt, enable_gradient_checkpointing

from .masks import (
    REVISED_SERIALIZATION,
    SERIALIZATIONS,
    MaskAlignmentError,
    field_masks,
    freeze_serialization,
    serialize_target,
)
from .prompts import REVISIONS, prefill_for

EXPERIMENT_ID = "SR-F3-CE-20261016"
PROTOCOLS = ("structured", "answer_only", "natural_work", "chartqa_plain", "chartqapro_official")
PREFILLED = ("structured", "answer_only")
EXTERNAL = ("chartqa_plain", "chartqapro_official")
__all__ = [
    "BatchedProtocolStop",
    "SRF3CERuntime",
    "configure_parent_training",
    "decoded_protocol",
    "enable_gradient_checkpointing",
    "first_boundary_token",
    "generation_recipe",
    "native_readability_measurements",
    "validate_model_input",
]


def _prefill(protocol, revision):
    if revision not in REVISIONS:
        raise ValueError("Unregistered protocol revision")
    if protocol in EXTERNAL:
        return ""
    return prefill_for(protocol, revision=revision)


def generation_recipe(*, protocol="structured"):
    if protocol not in PROTOCOLS:
        raise PermissionError("Unknown SR-F3-CE output protocol")
    return {
        **GENERATION,
        "do_sample": protocol not in EXTERNAL,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "max_new_tokens": 256 if protocol == "chartqa_plain" else 1024,
        "output_logits": True,
        "output_scores": True,
    }


def validate_model_input(model_input):
    """Reject sidecars at the runtime boundary instead of trying to ignore them."""
    if not isinstance(model_input, Mapping) or set(model_input) != {
        "prompt",
        "image_file",
        "image_sha256",
    }:
        raise PermissionError("Model input must contain only prompt/image_file/image_sha256")
    if not isinstance(model_input["prompt"], str) or not model_input["prompt"]:
        raise ValueError("A nonempty frozen prompt is required")
    image, digest = model_input["image_file"], model_input["image_sha256"]
    if (image is None) != (digest is None):
        raise ValueError("Image path and image SHA must both be present or absent")
    if image is not None and (
        not isinstance(image, str) or not image or not re.fullmatch(r"[0-9a-f]{64}", digest or "")
    ):
        raise ValueError("A registered image path and SHA256 are required")
    return dict(model_input)


def closed_final_cut(text):
    """First registered closing marker; malformed opening/suffix remains scoreable failure."""
    position = text.find("</final>")
    return None if position < 0 else position + len("</final>")


def protocol_boundary(text, protocol):
    if protocol in PREFILLED:
        return balanced_cut(text)
    if protocol == "natural_work":
        return closed_final_cut(text)
    if protocol not in EXTERNAL:
        raise PermissionError("Unknown output protocol")
    return None


def decoded_protocol(generated_text, protocol, *, revision="v1"):
    prefill = _prefill(protocol, revision)
    full = prefill + generated_text
    cut = protocol_boundary(full, protocol)
    if protocol in PREFILLED:
        error = (
            "unbalanced"
            if cut is None
            else ("trailing_after_close" if full[cut:].strip() else None)
        )
    elif protocol == "natural_work":
        error = (
            "missing_closed_final_tag"
            if cut is None
            else (
                "trailing_after_close"
                if full[cut:].strip()
                else (
                    "invalid_final_tag_count_or_order"
                    if full.count("<final>") != 1
                    or full.count("</final>") != 1
                    or full.index("<final>") > full.index("</final>")
                    else None
                )
            )
        )
    else:
        error = None
    return {
        "generated_text": generated_text,
        "assistant_prefill": prefill,
        "prefill": prefill,
        "full_decoded_text": full,
        "full_text": full,
        # Preserve every decoded character, including token-attached suffixes.
        "raw_text": full,
        "boundary_char": cut,
        "balanced_cut_char": cut if protocol in PREFILLED else None,
        "final_tag_cut_char": cut if protocol == "natural_work" else None,
        "format_protocol_error": error,
    }


def first_boundary_token(tokenizer, tokens, protocol, *, revision="v1"):
    prefill = _prefill(protocol, revision)
    if protocol in EXTERNAL:
        return None, None
    for count in range(1, len(tokens) + 1):
        text = prefill + decode_generated(tokenizer, tokens[:count])
        cut = protocol_boundary(text, protocol)
        if cut is not None:
            return count, cut
    return None, None


class BatchedProtocolStop:
    """Per-row stopping flags with independently retained first token boundaries."""

    def __init__(self, tokenizer, prompt_len, rows, protocol, *, revision="v1"):
        if prompt_len <= 0 or rows <= 0 or protocol not in (*PREFILLED, "natural_work"):
            raise ValueError("Invalid registered stopping context")
        self.tokenizer, self.prompt_len, self.protocol = tokenizer, prompt_len, protocol
        self.prefill = _prefill(protocol, revision)
        self.counts, self.cuts = [None] * rows, [None] * rows

    def __call__(self, input_ids, scores, **kwargs):
        import torch

        if input_ids.shape[0] != len(self.counts):
            raise PermissionError("Generation row count changed")
        for row in range(len(self.counts)):
            if self.counts[row] is None:
                tokens = input_ids[row, self.prompt_len :].tolist()
                cut = protocol_boundary(
                    self.prefill + decode_generated(self.tokenizer, tokens), self.protocol
                )
                if cut is not None:
                    self.counts[row], self.cuts[row] = len(tokens), cut
        return input_ids.new_tensor([n is not None for n in self.counts], dtype=torch.bool)


def configure_parent_training(runtime, *, parent_adapter_config, expected_adapter_state_hash=None):
    """Enable the exact already-loaded J6 adapter; never construct a new LoRA."""
    torch, model = runtime.torch, runtime.model
    if not hasattr(model, "peft_config") or not getattr(runtime, "adapter_path", None):
        raise PermissionError("A verified J6 parent adapter must already be loaded")
    active = model.active_adapter
    if not isinstance(active, str) or set(model.peft_config) != {active}:
        raise PermissionError("Exactly one inherited parent adapter is required")
    config = model.peft_config[active]
    expected = dict(parent_adapter_config)
    if any(
        expected.get(k) != v
        for k, v in (("r", 8), ("lora_alpha", 16), ("lora_dropout", 0), ("bias", "none"))
    ) or (config.r, config.lora_alpha, config.lora_dropout, config.bias) != (8, 16, 0, "none"):
        raise PermissionError("Inherited J6 LoRA configuration differs")
    targets = sorted(expected.get("target_modules", ()))
    if not targets or sorted(config.target_modules) != targets:
        raise PermissionError("Inherited parent target module list differs")
    full_modules, unwrapped = [], []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or ".lora_" in name:
            continue
        if re.search(r"(?:^|\.)language_model\.layers\.\d+\.", name):
            if not name.endswith(".base_layer"):
                unwrapped.append(name)
            else:
                full_modules.append(name.removesuffix(".base_layer"))
    if unwrapped or not full_modules:
        raise PermissionError("J6 adapter omits actual language decoder linear modules")
    if any(
        not any(name == target or name.endswith("." + target) for target in targets)
        for name in full_modules
    ):
        raise PermissionError("Actual wrapped module is absent from the parent targets")
    if any(
        not any(name == target or name.endswith("." + target) for name in full_modules)
        for target in targets
    ):
        raise PermissionError("Parent target does not resolve to an actual decoder module")
    parameters = {}
    for name, parameter in model.named_parameters():
        enabled = ".lora_A." in name or ".lora_B." in name
        if enabled and (
            not re.search(r"(?:^|\.)language_model\.layers\.\d+\.", name)
            or any(
                x in name.lower()
                for x in ("visual", "vision", "merger", "projector", "lm_head", "embed")
            )
        ):
            raise PermissionError("Parent LoRA parameter outside registered language decoder")
        if enabled:
            parameters[name] = parameter.detach()
    if not parameters:
        raise PermissionError("Loaded parent has no language LoRA tensors")
    before_hash = state_hash(parameters)
    if expected_adapter_state_hash is not None and before_hash != expected_adapter_state_hash:
        raise PermissionError("Loaded parent adapter tensor identity differs from checkpoint")
    if any(parameter.dtype != torch.float32 for parameter in parameters.values()):
        raise PermissionError("Inherited parent LoRA must already be float32")
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout) and module.p != 0:
            raise PermissionError("Inherited J6 contains nonzero dropout")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in parameters)
    model.eval()
    receipt = dict(
        status="PASS",
        parent_adapter_path=str(runtime.adapter_path),
        target_modules=targets,
        expanded_target_modules=sorted(full_modules),
        trainable_names=sorted(parameters),
        parent_parameter_hash=before_hash,
        r=8,
        lora_alpha=16,
        lora_dropout=0,
        trainable_dtype="float32",
        new_adapter_initialized=False,
        optimizer_inherited=False,
    )
    runtime.identity["sr_f3_ce_lora"] = receipt
    return receipt


def native_readability_measurements(prepared, *, source_pixels_per_count, source_text_heights):
    """Scale renderer-measured geometry through the actual processor image size.

    This verifies the registered geometric thresholds, not semantic recognition.
    The renderer supplies actual glyph bounding-box heights, not font point size.
    """
    source, processed = (
        prepared["routing"].get("source_size"),
        prepared["routing"].get("processed_size"),
    )
    if not source or not processed or not source_text_heights:
        raise ValueError("Actual source/processor sizes and glyph measurements are required")
    if source_pixels_per_count <= 0 or any(height <= 0 for height in source_text_heights):
        raise ValueError("Invalid renderer readability measurements")
    scale = processed[1] / source[1]
    unit = source_pixels_per_count * scale
    text = min(source_text_heights) * scale
    return dict(
        status="PASS" if unit >= 3 and text >= 9 else "FAIL",
        source_size=source,
        processed_size=processed,
        vertical_scale=scale,
        processed_pixels_per_count=unit,
        minimum_processed_text_height=text,
        minimum_pixels_per_count=3,
        minimum_text_height=9,
        processed_pixel_sha256=prepared["routing"].get("processed_pixel_sha256"),
        processor_hash=prepared["routing"].get("processor_hash"),
        claim="registered_geometric_readability_only",
    )


class SRF3CERuntime(SRF12Runtime):
    # Also preserves the old __new__-based CPU fixture interface.
    protocol_revision = "v1"

    def __init__(self, *args, protocol_revision="v1", **kwargs):
        _prefill("structured", protocol_revision)
        self.protocol_revision = protocol_revision
        super().__init__(*args, **kwargs)
        self._set_recipe()

    def _set_recipe(self):
        for key, value in generation_recipe().items():
            setattr(self.generation_config, key, value)
        self.identity.update(
            runtime_recipe="SR-F3-CE-native-full-sequence-1024",
            experiment_id=EXPERIMENT_ID,
            generation_config_expanded=self.generation_config.to_dict(),
            support_mask_source="actual_complete_native_processor_offsets",
            protocol_revision=self.protocol_revision,
            assistant_prefills={p: _prefill(p, self.protocol_revision) for p in PROTOCOLS},
            teacher_serialization="json.dumps(indent=2)"
            if self.protocol_revision != "v1"
            else "legacy_uniform_token_alignment_freeze",
        )

    @classmethod
    def processor_only(cls, model_path, *, protocol_amendment=None, protocol_revision="v1"):
        _prefill("structured", protocol_revision)
        instance = super().processor_only(model_path, protocol_amendment={"id": AMENDMENT_ID})
        import torch

        instance.torch = torch
        instance.protocol_revision = protocol_revision
        instance._set_recipe()
        return instance

    def prepare_input(self, model_input, run_root, *, protocol="structured"):
        safe = validate_model_input(model_input)
        if protocol not in PROTOCOLS:
            raise PermissionError("Unknown SR-F3-CE protocol")
        prefill = _prefill(protocol, self.protocol_revision)
        # Reuse the previously validated native image/template/prefill routing.
        prepared = SRRuntime.prepare_text(
            self,
            safe["prompt"],
            safe["image_file"],
            run_root,
            expected_hash=safe["image_sha256"],
            protocol="evidence_answer" if prefill == "{" else "plain_answer",
        )
        if prefill and prefill != "{":
            prepared = self._append_native_prefill(prepared, safe, run_root, prefill)
        prepared["routing"].update(
            protocol=protocol,
            protocol_revision=self.protocol_revision,
            experiment_id=EXPERIMENT_ID,
            assistant_prefill=prefill,
            prefill_is_context_not_loss_target=True,
            stop_rule="first_balanced_json"
            if protocol in PREFILLED
            else ("first_closed_final_tag" if protocol == "natural_work" else "eos_or_length"),
        )
        prepared["model_input"] = safe
        return prepared

    def _processor_arguments(self, safe, run_root, chat):
        kwargs = dict(text=[chat], padding=False, return_tensors="pt")
        if safe["image_file"] is not None:
            from PIL import Image

            root = Path(run_root).resolve()
            path = Path(safe["image_file"])
            path = path.resolve() if path.is_absolute() else bounded_path(root, path)
            if not path.is_relative_to(root) or file_hash(path) != safe["image_sha256"]:
                raise PermissionError("Native image differs from frozen model input")
            with Image.open(path) as image:
                kwargs["images"] = [image.convert("RGB")]
        return kwargs

    def _append_native_prefill(self, prepared, safe, run_root, prefill):
        """Process a complete registered context; never append hand-encoded IDs."""
        chat = prepared["chat_text"] + prefill
        inputs = self.processor(
            **self._processor_arguments(safe, run_root, chat), return_offsets_mapping=True
        )
        if "offset_mapping" not in inputs:
            raise MaskAlignmentError("Native prefill requires actual full-sequence offsets")
        offsets = inputs.pop("offset_mapping")
        offsets = offsets.tolist() if hasattr(offsets, "tolist") else offsets
        ids, base = inputs["input_ids"], prepared["inputs"]["input_ids"]
        base_count = base.shape[-1]
        if (
            ids.shape[0] != 1
            or ids.shape[-1] <= base_count
            or not self.torch.equal(ids[:, :base_count].cpu(), base.cpu())
            or len(offsets) != 1
            or len(offsets[0]) != ids.shape[-1]
        ):
            raise MaskAlignmentError("Registered prefill changed the native context boundary")
        actual_prefill_ids = ids[0, base_count:].tolist()
        actual_prefill_offsets = offsets[0][base_count:]
        native_end = max(end for _, end in offsets[0])
        native_start = native_end - len(prefill)
        decoded = self.processor.tokenizer.decode(
            actual_prefill_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if (
            decoded != prefill
            or actual_prefill_offsets[0][0] != native_start
            or actual_prefill_offsets[-1][1] != native_end
            or any(not native_start <= lo < hi <= native_end for lo, hi in actual_prefill_offsets)
        ):
            raise MaskAlignmentError("Native full-context IDs/offsets do not reproduce prefill")
        for key in ("pixel_values", "image_grid_thw"):
            if key in prepared["inputs"] and (
                key not in inputs
                or not self.torch.equal(prepared["inputs"][key].cpu(), inputs[key].cpu())
            ):
                raise PermissionError("Registered prefill changed native image tensors")
        if "mm_token_type_ids" not in inputs:
            if safe["image_file"] is not None:
                raise RuntimeError("Native prefill lacks multimodal token types")
            inputs["mm_token_type_ids"] = self.torch.zeros_like(ids)
        mm_types = inputs["mm_token_type_ids"]
        if (
            mm_types.shape != ids.shape
            or not self.torch.equal(
                mm_types[:, :base_count].cpu(), prepared["inputs"]["mm_token_type_ids"].cpu()
            )
            or bool(mm_types[:, base_count:].any())
        ):
            raise PermissionError("Registered prefill changed native image token routing")
        tensor_hashes = {
            name: state_hash(value) for name, value in inputs.items() if hasattr(value, "detach")
        }
        routing = dict(prepared["routing"])
        routing.update(
            native_input_tensor_hashes=tensor_hashes,
            input_tensor_hash=hash_json(tensor_hashes),
            chat_text_sha256=hashlib.sha256(chat.encode()).hexdigest(),
            input_ids_sha256=hash_json(ids.tolist()),
            assistant_prefill_token_ids=actual_prefill_ids,
            assistant_prefill_native_offsets=actual_prefill_offsets,
            assistant_prefill_native_char_span=[native_start, native_end],
            base_prompt_token_count=base_count,
        )
        return dict(inputs=inputs.to(self.device), routing=routing, chat_text=chat)

    def prepare(self, model_input, run_root, *, protocol="structured"):
        return self.prepare_input(model_input, run_root, protocol=protocol)

    def prepare_text(self, *args, **kwargs):
        raise PermissionError("Use the ModelInputStore three-field prepare_input interface")

    def encode_completion(self, *args, **kwargs):
        raise PermissionError("Teacher support requires complete native processor offsets")

    def _prepared_records(self, records):
        if not records:
            raise ValueError("Empty teacher-forcing microbatch")
        prepared, same_call = [], {}
        for record in records:
            self._validate_record_revision(record)
            item = record.get("prepared")
            if item is None:
                safe = validate_model_input(record["model_input"])
                protocol = record.get("protocol", "structured")
                root = record.get("run_root", record.get("root"))
                key = hash_json(dict(model_input=safe, protocol=protocol, root=str(root)))
                if key not in same_call:
                    same_call[key] = self.prepare_input(safe, root, protocol=protocol)
                item = same_call[key]
            self._validate_prepared_revision(item)
            if record.get("prompt_tensor_hash", item["routing"].get("input_tensor_hash")) != item[
                "routing"
            ].get("input_tensor_hash"):
                raise PermissionError("Restored native prompt tensor identity changed")
            prepared.append(item)
        first = prepared[0]["inputs"]
        for item in prepared[1:]:
            if first.keys() != item["inputs"].keys() or any(
                not self.torch.equal(first[key], item["inputs"][key]) for key in first
            ):
                raise PermissionError("Microbatch must have byte-identical same-question prompts")
        return prepared[0]

    def _validate_record_revision(self, record):
        if record.get("protocol_revision", "v1") != self.protocol_revision:
            raise PermissionError("Record protocol revision differs from runtime")
        expected = _prefill(record.get("protocol", "structured"), self.protocol_revision)
        if record.get("assistant_prefill", expected) != expected:
            raise PermissionError("Record assistant prefill differs from registered revision")

    def _validate_prepared_revision(self, prepared):
        route = prepared.get("routing", {})
        if route.get("protocol_revision", "v1") != self.protocol_revision:
            raise PermissionError("Prepared context protocol revision differs from runtime")
        expected = _prefill(route.get("protocol", "structured"), self.protocol_revision)
        if route.get("assistant_prefill", expected) != expected:
            raise PermissionError("Prepared context assistant prefill differs from revision")

    def _native_forward(self, inputs, *, completion_start, lengths, purpose, grad):
        """One native full-sequence call with explicit per-row mRoPE and no cache."""
        torch = self.torch
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        native = getattr(base, "model", base)
        for module in (base, native):
            if hasattr(module, "rope_deltas"):
                module.rope_deltas = None
        if "logits_to_keep" not in inspect.signature(base.forward).parameters:
            raise RuntimeError("Native model must support completion-only logits_to_keep")
        inputs = dict(inputs)
        inputs.pop("token_type_ids", None)
        if "mm_token_type_ids" not in inputs:
            raise RuntimeError("Native multimodal token types missing")
        rows, total = inputs["input_ids"].shape
        if not 1 <= completion_start < total or len(lengths) != rows:
            raise ValueError("Invalid native completion boundary")
        positions = []
        for row in range(rows):
            grid = inputs.get("image_grid_thw")
            row_grid = grid[row : row + 1] if grid is not None else None
            position, _ = native.get_rope_index(
                input_ids=inputs["input_ids"][row : row + 1],
                mm_token_type_ids=inputs["mm_token_type_ids"][row : row + 1],
                image_grid_thw=row_grid,
                attention_mask=inputs["attention_mask"][row : row + 1],
            )
            positions.append(position)
        kept = total - completion_start + 1
        inputs.update(
            position_ids=torch.cat(positions, dim=1), use_cache=False, logits_to_keep=kept
        )
        self.reserve(
            "extra_forward_sequences", rows, purpose=purpose, completion_tokens=sum(lengths)
        )
        before, old_training = self.image_calls, self.model.training
        self.model.train(bool(grad and getattr(self, "_sr_f12_checkpointing", False)))
        try:
            with torch.enable_grad() if grad else torch.no_grad():
                output = self.model(**inputs)
                if output.logits.shape[:2] != (rows, kept):
                    raise RuntimeError("Native completion logits shape changed")
                selected = [
                    output.logits[row, :length]
                    .float()
                    .log_softmax(-1)
                    .gather(
                        -1,
                        inputs["input_ids"][
                            row, completion_start : completion_start + length, None
                        ],
                    )
                    .squeeze(-1)
                    for row, length in enumerate(lengths)
                ]
            if bool((inputs["mm_token_type_ids"] == 1).any()) and self.image_calls <= before:
                raise RuntimeError("Whole-sequence forward did not invoke vision encoder")
            return selected
        finally:
            self.model.train(old_training)

    def _batch_forward(self, prepared, token_rows, *, purpose, grad):
        self._validate_prepared_revision(prepared)
        torch = self.torch
        if not token_rows or any(not tokens or len(tokens) > 1024 for tokens in token_rows):
            raise ValueError("Each completion must contain 1..1024 actual generated tokens")
        inputs = _repeat_prompt(prepared["inputs"], len(token_rows))
        prompt_len, longest = inputs["input_ids"].shape[-1], max(map(len, token_rows))
        pad = self.processor.tokenizer.pad_token_id
        if pad is None:
            pad = self.eos_ids[0]
        completion = inputs["input_ids"].new_full((len(token_rows), longest), pad)
        mask = self.torch.zeros_like(completion)
        for index, tokens in enumerate(token_rows):
            completion[index, : len(tokens)] = completion.new_tensor(tokens)
            mask[index, : len(tokens)] = 1
        inputs["input_ids"] = torch.cat([inputs["input_ids"], completion], -1)
        inputs["attention_mask"] = torch.cat(
            [inputs.get("attention_mask", mask.new_ones((len(token_rows), prompt_len))), mask], -1
        )
        inputs["mm_token_type_ids"] = torch.cat(
            [inputs["mm_token_type_ids"], torch.zeros_like(completion)], -1
        )
        return self._native_forward(
            inputs,
            completion_start=prompt_len,
            lengths=list(map(len, token_rows)),
            purpose=purpose,
            grad=grad,
        )

    def _support_serializations(self):
        return SERIALIZATIONS if self.protocol_revision == "v1" else (REVISED_SERIALIZATION,)

    def prepare_support(self, model_input, run_root, target, arm, *, serialization=None):
        if arm not in ("GOLD", "PREFIX", "EXEC"):
            raise PermissionError("RL has no teacher support sequence")
        if serialization is None:
            serialization = self._support_serializations()[0]
        if serialization not in self._support_serializations():
            raise PermissionError("Support serialization differs from registered protocol revision")
        safe = validate_model_input(model_input)
        selected = serialize_target(target, serialization)
        if ("work" in selected["spans"]) is not (arm != "GOLD"):
            raise PermissionError("GOLD omits work; PREFIX and EXEC require the same work target")
        prepared = self.prepare_input(safe, run_root, protocol="structured")
        prefill = _prefill("structured", self.protocol_revision)
        if not prepared["chat_text"].endswith(prefill):
            raise PermissionError("Support prompt lacks its registered assistant prefill")
        if not selected["text"].startswith(prefill):
            raise MaskAlignmentError("Teacher does not begin with the registered context prefill")
        complete_chat = prepared["chat_text"][: -len(prefill)] + selected["text"]
        encoded = self.processor(
            **self._processor_arguments(safe, run_root, complete_chat), return_offsets_mapping=True
        )
        if "offset_mapping" not in encoded:
            raise MaskAlignmentError("Native processor must return actual full-sequence offsets")
        offsets = encoded.pop("offset_mapping")
        offsets = offsets.tolist() if hasattr(offsets, "tolist") else offsets
        if len(offsets) != 1 or encoded["input_ids"].shape[0] != 1:
            raise MaskAlignmentError("Exactly one native teacher sequence is required")
        offsets = offsets[0]
        ids = encoded["input_ids"][0].tolist()
        # Native processors expand image placeholders before tokenization. Their
        # offsets therefore use expanded-chat coordinates. The complete teacher
        # is the verified terminal suffix, ending at the final positive offset.
        native_end = max(end for _, end in offsets)
        target_start = native_end - len(selected["text"])
        if target_start < len(prepared["chat_text"]) - len(prefill):
            raise MaskAlignmentError("Native offsets do not cover the complete teacher suffix")
        if not offsets or offsets[-1][1] != native_end or offsets[-1][0] == offsets[-1][1]:
            raise MaskAlignmentError("Native processor appended a token beyond the teacher text")
        # The actual native token suffix must decode to the identical teacher,
        # allowing only a token containing adjacent assistant-context characters.
        first = next((i for i, (_, end) in enumerate(offsets) if end > target_start), None)
        if first is None:
            raise MaskAlignmentError("Native teacher target has no tokens")
        decoded = self.processor.tokenizer.decode(
            ids[first:], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if not decoded.endswith(selected["text"]):
            raise MaskAlignmentError("Full processor IDs do not reproduce exact teacher bytes")
        audit = field_masks(selected, offsets, target_start=target_start, token_ids=ids)
        # Every prefill character is context. A token spanning its final delimiter
        # and subsequent evidence stays unsupervised; its actual native IDs and
        # offsets are retained instead of hand-splitting or separately encoding it.
        prefill_end = target_start + len(prefill)
        completion_start = next(
            (i for i, (_, end) in enumerate(offsets) if end > prefill_end), None
        )
        if completion_start is None or completion_start == 0:
            raise MaskAlignmentError("Invalid teacher completion token boundary")
        prefill_token_indices = [
            i
            for i, (start, end) in enumerate(offsets)
            if start < prefill_end and end > target_start
        ]
        if any(
            audit[field][i] for field in ("answer_mask", "work_mask") for i in prefill_token_indices
        ):
            raise MaskAlignmentError("A registered prefill token cannot carry teacher loss")
        tokens = ids[completion_start:]
        answer, work = (
            audit["answer_mask"][completion_start:],
            audit["work_mask"][completion_start:],
        )
        if not any(answer) or (arm != "GOLD" and not any(work)):
            raise MaskAlignmentError("Teacher masks disappeared at the context boundary")
        if len(tokens) > 1024:
            raise PermissionError("Teacher support exceeds the registered 1024 token channel")
        for key in ("pixel_values", "image_grid_thw"):
            if key in prepared["inputs"] and (
                key not in encoded
                or not self.torch.equal(prepared["inputs"][key].cpu(), encoded[key].cpu())
            ):
                raise PermissionError("Support processor changed registered image tensors")
        if "mm_token_type_ids" not in encoded:
            if safe["image_file"] is not None:
                raise RuntimeError("Native teacher lacks multimodal token type IDs")
            encoded["mm_token_type_ids"] = self.torch.zeros_like(encoded["input_ids"])
        tensor_hashes = {name: state_hash(value) for name, value in encoded.items()}
        audit.update(
            serialization=serialization,
            source_text_sha256=selected["source_text_sha256"],
            target_text_sha256=selected["text_sha256"],
            completion_start_index=completion_start,
            native_full_input_tensor_hash=hash_json(tensor_hashes),
            native_full_input_tensor_hashes=tensor_hashes,
            offset_coordinate_system="native_processor_expanded_chat_text",
            assistant_prefill=prefill,
            protocol_revision=self.protocol_revision,
            prefill_native_char_span=[target_start, prefill_end],
            prefill_token_indices=prefill_token_indices,
            prefill_token_ids=[ids[i] for i in prefill_token_indices],
            prefill_cross_boundary_tokens=[
                dict(token_index=i, token_id=ids[i], offset=list(offsets[i]), owner="context")
                for i in prefill_token_indices
                if offsets[i][1] > prefill_end
            ],
            prefill_is_context_not_loss_target=True,
        )
        return dict(
            model_input=safe,
            run_root=str(run_root),
            root=str(run_root),
            arm=arm,
            protocol="structured",
            protocol_revision=self.protocol_revision,
            assistant_prefill=prefill,
            teacher_target=copy.deepcopy(target),
            serialized_teacher_target=selected,
            serialization=serialization,
            supervised_fields=["answer", "work"] if arm == "EXEC" else ["answer"],
            support_field_weights={"answer": 0.5, "work": 0.5}
            if arm == "EXEC"
            else {"answer": 1.0},
            tokens=tokens,
            answer_mask=answer,
            work_mask=work,
            mask_audit=audit,
            prepared=prepared,
            full_inputs=encoded.to(self.device),
            completion_start_index=completion_start,
        )

    def freeze_support_serialization(self, cases, run_root):
        """CPU/processor-only audit of every registered GOLD and shared traced target."""
        receipt = freeze_serialization(
            [case["target"] for case in cases],
            lambda selected, index: self.prepare_support(
                cases[index]["model_input"],
                run_root,
                cases[index]["target"],
                cases[index]["arm"],
                serialization=selected["serialization"],
            )["mask_audit"],
            serializations=self._support_serializations(),
        )
        receipt.update(
            protocol_revision=self.protocol_revision,
            assistant_prefill=_prefill("structured", self.protocol_revision),
            serialization_candidates=list(self._support_serializations()),
        )
        return receipt

    def support_forward(self, record, *, purpose="training_support", grad=True):
        self._validate_record_revision(record)
        if "full_inputs" not in record:
            rebuilt = self.prepare_support(
                record["model_input"],
                record.get("run_root", record.get("root")),
                record["teacher_target"],
                record["arm"],
                serialization=record.get("serialization"),
            )
            for key in ("tokens", "answer_mask", "work_mask", "mask_audit"):
                if record.get(key) != rebuilt[key]:
                    raise PermissionError("Restored support native IDs/mask audit changed: " + key)
            record = rebuilt
        ids = record["full_inputs"]["input_ids"][0, record["completion_start_index"] :].tolist()
        if ids != record["tokens"]:
            raise PermissionError("Support token IDs differ from actual forward sequence")
        return self._native_forward(
            record["full_inputs"],
            completion_start=record["completion_start_index"],
            lengths=[len(record["tokens"])],
            purpose=purpose,
            grad=grad,
        )[0]

    def batch_support_forward(self, records, *, purpose="training_support", grad=True):
        if not records:
            raise ValueError("Empty support microbatch")
        # Different support questions cannot use same-image prompt expansion.
        return [self.support_forward(record, purpose=purpose, grad=grad) for record in records]

    def generate(
        self,
        model_input,
        run_root,
        seed,
        *,
        protocol="structured",
        generation=None,
        on_completion=None,
    ):
        return self.generate_group(
            model_input,
            run_root,
            [seed],
            protocol=protocol,
            generation=generation,
            on_completion=on_completion,
        )[0]

    def generate_group(
        self,
        model_input,
        run_root,
        seeds,
        *,
        protocol="structured",
        generation=None,
        on_completion=None,
    ):
        if not 1 <= len(seeds) <= 8 or any(type(seed) is not int for seed in seeds):
            raise ValueError("Same-question group requires 1..8 integer slot seeds")
        safe = validate_model_input(model_input)
        recipe, opts = generation_recipe(protocol=protocol), generation or {}
        permitted = {
            "max_new_tokens",
            "do_sample",
            "temperature",
            "top_p",
            "top_k",
            "num_return_sequences",
        }
        if set(opts) - permitted or any(
            opts.get(key, recipe[key]) != recipe[key]
            for key in permitted - {"num_return_sequences"}
        ):
            raise PermissionError("Registered generation recipe or token cap changed")
        if opts.get("num_return_sequences", len(seeds)) != len(seeds):
            raise PermissionError("Group size differs from registered slot seeds")
        if protocol in EXTERNAL and len(seeds) != 1:
            raise PermissionError("External direct protocols require single-row greedy generation")
        prepared = self.prepare_input(safe, run_root, protocol=protocol)
        config = copy.deepcopy(self.generation_config)
        for key, value in recipe.items():
            setattr(config, key, value)
        config.num_return_sequences = 1
        inputs = _repeat_prompt(prepared["inputs"], len(seeds))
        prompt_len = inputs["input_ids"].shape[-1]
        stop, kwargs = None, {}
        if protocol not in EXTERNAL:
            from transformers import StoppingCriteriaList

            stop = BatchedProtocolStop(
                self.processor.tokenizer,
                prompt_len,
                len(seeds),
                protocol,
                revision=self.protocol_revision,
            )
            kwargs["stopping_criteria"] = StoppingCriteriaList([stop])
        self.reserve("completion_attempts", len(seeds), group_seed=seeds[0], protocol=protocol)
        rng, before, started = capture_rng(), self.image_calls, time.perf_counter()
        old_training = self.model.training
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
            if result.sequences.shape[0] != len(seeds):
                raise RuntimeError("Native generation returned the wrong number of rows")
            records = []
            adapter_identity, model_identity = (
                self.current_adapter_identity(),
                self.stable_model_identity(),
            )
            for index, seed in enumerate(seeds):
                backend_tokens = result.sequences[index, prompt_len:].cpu().tolist()
                boundary_n, boundary_char = first_boundary_token(
                    self.processor.tokenizer,
                    backend_tokens,
                    protocol,
                    revision=self.protocol_revision,
                )
                eos_n = next(
                    (i + 1 for i, token in enumerate(backend_tokens) if token in self.eos_ids), None
                )
                n = min(
                    value for value in (boundary_n, eos_n, len(backend_tokens)) if value is not None
                )
                tokens, errors, logps = backend_tokens[:n], [], []
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
                finish = (
                    ("balanced" if protocol in PREFILLED else "final_tag")
                    if boundary_n == n
                    else ("eos" if eos_n == n else "length")
                )
                if finish == "length" and n != recipe["max_new_tokens"]:
                    errors.append("UNREGISTERED_TERMINATION")
                if (
                    boundary_n == n
                    and stop is not None
                    and (stop.counts[index] != n or stop.cuts[index] != boundary_char)
                ):
                    errors.append("PROTOCOL_STOP_REPLAY_MISMATCH")
                pad = self.processor.tokenizer.pad_token_id
                if n < len(backend_tokens) and any(token != pad for token in backend_tokens[n:]):
                    errors.append("NONPAD_TOKENS_AFTER_ROW_STOP")
                if prepared["routing"].get("image_token_count", 0) and self.image_calls <= before:
                    errors.append("NATIVE_VISUAL_ENCODER_NOT_CALLED")
                text = decode_generated(self.processor.tokenizer, tokens)
                prefill = _prefill(protocol, self.protocol_revision)
                full_text = prefill + text
                prefix_chars, prefix_stable, prefix_views = [], True, {}
                for count in range(1, len(tokens) + 1):
                    prefix_text = prefill + decode_generated(
                        self.processor.tokenizer, tokens[:count]
                    )
                    prefix_chars.append(len(prefix_text))
                    prefix_stable = prefix_stable and full_text.startswith(prefix_text)
                    if count in (128, 256, 512, 1024):
                        prefix_views[str(count)] = prefix_text
                for budget in (128, 256, 512, 1024):
                    if len(tokens) < budget:
                        prefix_views[str(budget)] = full_text
                record = dict(
                    model_input=safe,
                    prompt=safe["prompt"],
                    run_root=str(run_root),
                    root=str(run_root),
                    protocol=protocol,
                    protocol_revision=self.protocol_revision,
                    tokens=tokens,
                    raw_tokens=tokens,
                    backend_sequence_tokens=backend_tokens,
                    backend_padding_count=len(backend_tokens) - n,
                    seed=seed,
                    group_seed=seeds[0],
                    group_row_index=index,
                    group_size=len(seeds),
                    seed_semantics="group_seed_initializes_one_batched_rng_slot_seeds_are_identity",
                    sampler_logprobs=logps,
                    sampler_logprob_source="actual_generate_raw_logits_selected_token",
                    generated_text_with_special_tokens=self.processor.tokenizer.decode(
                        tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False
                    ),
                    prompt_token_count=prompt_len,
                    completion_token_count=n,
                    prompt_tensor_hash=prepared["routing"]["input_tensor_hash"],
                    input_routing=copy.deepcopy(prepared["routing"]),
                    image_routing=copy.deepcopy(prepared["routing"]),
                    sampling_parameters=config.to_dict(),
                    sampling_hash=hash_json(config.to_dict()),
                    max_new_tokens=recipe["max_new_tokens"],
                    generation_seconds=time.perf_counter() - started,
                    finish_reason=finish,
                    truncated=finish == "length",
                    boundary_token_count=n if boundary_n == n else None,
                    balanced_token_count=n if boundary_n == n and protocol in PREFILLED else None,
                    token_prefix_character_ends=prefix_chars,
                    prefix_character_offsets_stable=prefix_stable,
                    decoded_prefix_views=prefix_views,
                    prefix_diagnostic_claim="observed_full_trajectory_only_pending_prefix_invariance_gate",
                    generation_status="TECHNICAL_INVALID" if errors else "COMPLETE",
                    technical_validation_errors=errors,
                    adapter_identity=copy.deepcopy(adapter_identity),
                    model_identity=copy.deepcopy(model_identity),
                    **decoded_protocol(text, protocol, revision=self.protocol_revision),
                )
                if on_completion is not None:
                    on_completion(record)
                records.append(record)
                self.reserve("generated_tokens", n, group_seed=seeds[0], row_index=index)
            if any(record["technical_validation_errors"] for record in records):
                raise RuntimeError(
                    "Invalid generation preserved: "
                    + str([record["technical_validation_errors"] for record in records])
                )
            return records
        finally:
            self.model.train(old_training)
            restore_rng(rng)
