"""SR-F6 native token trajectory adapter with an explicit 1536-token recipe.

The copied sampling body is the frozen SR-F3/F5 native path. Its local recipe
changes only the registered token limit. It does not monkeypatch legacy globals.
"""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path

from mm_core.training import capture_rng, restore_rng, state_hash, trainable_state
from mm_core.vl_runtime import hash_json, seed_all
from sr_f1.json_protocol import decode_generated
from sr_f3_ce.runtime import (
    EXTERNAL,
    PREFILLED,
    BatchedProtocolStop,
    _prefill,
    configure_parent_training,
    decoded_protocol,
    first_boundary_token,
    validate_model_input,
)
from sr_f3_ce.runtime import (
    generation_recipe as legacy_generation_recipe,
)
from sr_f4.common import append, now, read, write
from sr_f4.runtime import independent_draw_rng
from sr_f5.runtime import NUMERICS_V2
from sr_f5.runtime import Runtime as FP32Runtime
from sr_f12.runtime import _repeat_prompt
from srf6_ref.core import seed_for

PROTOCOL_VERSION = "structured_v4-srf6-1.0"


def backend_protocol(protocol):
    if protocol in ("structured_v4", "structured"):
        return "structured"
    raise PermissionError("This census adapter only supports the registered structured protocol")


def generation_recipe(*, protocol="structured"):
    backend_protocol(protocol)
    return {**legacy_generation_recipe(protocol="structured"), "max_new_tokens": 1536}


def validate_generation_config(config):
    expected = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "max_new_tokens": 1536}
    if any(
        type(config["training"].get(k)) is bool or config["training"].get(k) != v
        for k, v in expected.items()
    ):
        raise PermissionError("Registered SR-F6 sampling recipe differs")
    if any(
        config["numerics"].get(k) != "float32"
        for k in ("model_dtype", "input_float_dtype", "probability_dtype")
    ):
        raise PermissionError("All registered numerical paths must be float32")


def verify_binding_inputs(root, config, parent=None):
    """Validate the bound file identity before processor/model reads any bytes."""
    from .source_binding import (
        PROJECT,
        content_inventory,
        file_sha256,
        object_sha256,
        verify_bound_files,
    )

    root = Path(root)
    bound = read(root / "SOURCE_BINDING.json")
    if bound["config_file_sha256"] != file_sha256(root / "CONFIG.json"):
        raise PermissionError("Source binding configuration identity changed before load")
    model = Path(config["model"]["model_path"])
    actual_files = {str(p) for p in model.rglob("*") if p.is_file()}
    if actual_files != {entry["path"] for entry in bound["model_files"]}:
        raise PermissionError("Bound model/tokenizer file inventory changed")
    model_check = verify_bound_files(bound["model_files"], PROJECT)
    parent_check = None
    if parent is not None:
        record = bound["parents"][parent]
        adapter = Path(config["model"]["parents"][parent])
        expected = {
            str(adapter / name) for name in ("adapter_config.json", "adapter_model.safetensors")
        }
        if (
            record["path"] != str(adapter)
            or {f["path"] for f in record["files"]} != expected
            or object_sha256(content_inventory(record["files"])) != record["content_sha256"]
        ):
            raise PermissionError("Parent binding differs from its registered content inventory")
        parent_check = verify_bound_files(record["files"], PROJECT)
    return bound, dict(model=model_check, parent=parent_check)


class Runtime(FP32Runtime):
    def _set_recipe(self):
        super()._set_recipe()
        for key, value in generation_recipe().items():
            setattr(self.generation_config, key, value)
        self.identity.update(
            runtime_recipe="SR-F6-native-full-sequence-1536",
            experiment_id="SR-F6-SEMANTIC-LEARNING-20261027",
            generation_config_expanded=self.generation_config.to_dict(),
        )

    def prepare_input(self, model_input, run_root, *, protocol="structured_v4"):
        prepared = super().prepare_input(model_input, run_root, protocol=backend_protocol(protocol))
        prepared["routing"]["sr_f6_protocol_version"] = PROTOCOL_VERSION
        return prepared

    def _validate_record_revision(self, record):
        if record.get("sr_f6_protocol_version") != PROTOCOL_VERSION:
            raise PermissionError("SR-F6 record protocol identity differs")
        super()._validate_record_revision(
            dict(record, protocol=backend_protocol(record["protocol"]))
        )

    def stable_model_identity(self):
        return dict(
            super().stable_model_identity(),
            sr_f6_protocol_version=PROTOCOL_VERSION,
            registered_max_new_tokens=1536,
        )

    def policy_tensor_identity(self):
        """Hash once at load/committed-step transition, not once per question."""
        step = self.identity["step"]
        if type(step) is not int or step < 0:
            raise PermissionError("A committed nonnegative policy step is required")
        if not getattr(self, "_srf6_training", False) and step != 0:
            raise PermissionError("Frozen parent census cannot advance model parameters")
        cached = getattr(self, "_srf6_policy_identity", None)
        if cached is None or cached[0] != step:
            value = state_hash(trainable_state(self.model))
            cached = (step, value)
            self._srf6_policy_identity = cached
            self.identity["policy_hash"] = value
        return cached[1]

    def generate_group(
        self,
        model_input,
        run_root,
        seeds,
        *,
        protocol="structured_v4",
        generation=None,
        on_completion=None,
    ):
        backend = backend_protocol(protocol)
        policy_tensor_hash = self.policy_tensor_identity()

        def capture(row):
            row["protocol"] = "structured_v4"
            row["backend_protocol"] = backend
            row["sr_f6_protocol_version"] = PROTOCOL_VERSION
            row["seed_semantics"] = "independent_per_draw_torch_generator"
            row["policy_hash"] = policy_tensor_hash
            row["policy_tensor_hash"] = policy_tensor_hash
            row["parent_content_hash"] = self.identity["parent_content_hash"]
            row["source_sha256"] = self.identity["source_sha256"]
            row["source_hash"] = self.identity["source_sha256"]
            row["data_hash"] = self.identity["data_hash"]
            row["run_id"] = self.identity["run_id"]
            row["policy_step"] = self.identity["step"]
            row["step"] = self.identity["step"] + int(getattr(self, "_srf6_training", False))
            import hashlib

            row["prompt_sha256"] = hashlib.sha256(row["model_input"]["prompt"].encode()).hexdigest()
            inputs = self._srf5_prepared["inputs"]
            row["native_prompt_inputs"] = {
                k: v.detach().cpu().tolist()
                for k, v in inputs.items()
                if k != "pixel_values" and hasattr(v, "detach")
            }
            row["position_ids_semantics"] = (
                "native_generate_mrope_from_saved_input_ids_attention_mask_and_image_grid"
            )
            if on_completion is not None:
                on_completion(row)

        with independent_draw_rng(self.torch, seeds):
            return self._generate_native_group(
                model_input,
                run_root,
                seeds,
                protocol=backend,
                generation=generation,
                on_completion=capture,
            )

    def _batch_forward(self, prepared, token_rows, *, purpose, grad):
        self._validate_prepared_revision(prepared)
        torch = self.torch
        if not token_rows or any(not tokens or len(tokens) > 1536 for tokens in token_rows):
            raise ValueError("Each completion must contain 1..1536 actual generated tokens")
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

    def _generate_native_group(
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


def load(root, parent, task_id, *, training=False):
    import torch

    from mm_dev.runtime import actual_cuda_identity, configure_audited_backend
    from sr_f3_ce.training import fresh_optimizer

    root = Path(root)
    config = read(root / "CONFIG.json")
    validate_generation_config(config)
    if parent not in config["model"]["parents"]:
        raise ValueError("Unknown registered parent")
    adapter = Path(config["model"]["parents"][parent])
    if not adapter.is_dir():
        raise FileNotFoundError(adapter)
    bound, bound_files_check = verify_binding_inputs(root, config, parent)
    configure_audited_backend()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Each SR-F6 worker must own exactly one visible GPU")
    seed_all(seed_for(config["master_seed"], "load", parent))

    def account(kind, count, metadata):
        if type(count) is not int or count < 0:
            raise ValueError("Invalid model-call accounting")
        append(
            root / "accounting" / f"{task_id}.jsonl",
            dict(
                at=now(), task_id=task_id, parent=parent, kind=kind, count=count, metadata=metadata
            ),
        )

    runtime = Runtime(
        config["model"]["model_path"],
        adapter_path=str(adapter),
        output_root=root,
        protocol_revision="v2",
        account=account,
    )
    configure_parent_training(runtime, parent_adapter_config=read(adapter / "adapter_config.json"))
    source_dtype = runtime.identity["dtype"]
    runtime.model = runtime.model.float()
    parameter_dtypes = sorted(
        {
            str(v.dtype).removeprefix("torch.")
            for v in runtime.model.parameters()
            if v.is_floating_point()
        }
    )
    buffer_dtypes = sorted(
        {
            str(v.dtype).removeprefix("torch.")
            for v in runtime.model.buffers()
            if v.is_floating_point()
        }
    )
    if parameter_dtypes != ["float32"] or any(v != "float32" for v in buffer_dtypes):
        raise RuntimeError("Full FP32 conversion did not take effect")
    runtime._srf5_numerics = dict(NUMERICS_V2)
    runtime.identity.update(
        dtype="float32",
        numerics=dict(NUMERICS_V2),
        numerics_recipe_version=2,
        model_cast_from_dtype=source_dtype,
        forward_parameter_dtypes=parameter_dtypes,
        forward_float_buffer_dtypes=buffer_dtypes,
        input_float_dtype="float32",
        probability_dtype="float32",
        parent_id=parent,
        run_id=task_id,
        step=0,
        snapshot=config["model"]["snapshot"],
        sr_f6_protocol_version=PROTOCOL_VERSION,
    )
    _, post_load_files_check = verify_binding_inputs(root, config, parent)
    identity = read(root / "IDENTITY.json")
    tensor_hash = state_hash(trainable_state(runtime.model))
    runtime._srf6_training = bool(training)
    runtime._srf6_policy_identity = (0, tensor_hash)
    runtime.identity.update(
        run_root=str(root.resolve(strict=True)),
        source_sha256=identity["source_manifest"]["sha256"],
        source_hash=identity["source_manifest"]["sha256"],
        data_hash=identity["data_manifest"]["sha256"],
        parent_content_hash=bound["parents"][parent]["content_sha256"],
        policy_hash=tensor_hash,
        reference_hash=tensor_hash,
        reference_parent=parent,
    )
    trainable = [
        {"name": n, "shape": list(p.shape), "dtype": str(p.dtype)}
        for n, p in runtime.model.named_parameters()
        if p.requires_grad
    ]
    write(
        root / "runtime_identity" / f"{task_id}.json",
        dict(
            copy.deepcopy(runtime.identity),
            hardware=actual_cuda_identity(),
            trainable_parameters=trainable,
            fresh_optimizer=bool(training),
            source_binding_note="content binding and loaded identity are separate receipts",
            bound_files_before_load=bound_files_check,
            bound_files_after_load=post_load_files_check,
        ),
    )
    optimizer, scheduler = fresh_optimizer(runtime.model) if training else (None, None)
    return runtime, optimizer, scheduler


def processor_preflight(root):
    """Use the real frozen processor and save its actual reconstructed pixels.

    These are program checks. The returned receipt explicitly awaits a visual
    reviewer and actual generation alignment; it cannot authorize a census.
    """
    import hashlib

    from PIL import Image

    from mm_dev.data import reconstruct_rgb

    from .source_binding import file_sha256

    root = Path(root)
    config = read(root / "CONFIG.json")
    validate_generation_config(config)
    _, bound_files_check = verify_binding_inputs(root, config)
    runtime = Runtime.processor_only(config["model"]["model_path"], protocol_revision="v2")
    runtime._srf5_numerics = dict(NUMERICS_V2)
    public = {r["qid"]: r for r in _jsonl(root / "DATA/DEV_CAL/MODEL_INPUTS.jsonl")}
    plan = read(root / "DATA/PROCESSOR_REVIEW_PLAN.json")
    if plan.get("count") != 48 or len(plan["items"]) != 48:
        raise ValueError("Exactly 48 registered development images required")
    destination = root / "PREFLIGHT/processor"
    if destination.exists():
        raise FileExistsError("Preserve existing processor evidence")
    destination.mkdir(parents=True)
    results = []
    for item in plan["items"]:
        task = public[item["qid"]]
        prepared = runtime.prepare_input(task["model_input"], root, protocol="structured_v4")
        route = prepared["routing"]
        rgb = reconstruct_rgb(
            prepared["inputs"]["pixel_values"],
            route["image_grid_thw"],
            runtime.processor.image_processor,
        )
        output = destination / f"{item['qid']}.png"
        rgb.save(output)
        with Image.open(root / task["model_input"]["image_file"]) as source:
            if source.size != (1280, 896):
                raise ValueError("Unregistered original image dimensions")
        results.append(
            dict(
                qid=item["qid"],
                family=item["family"],
                source_file=task["model_input"]["image_file"],
                source_sha256=task["model_input"]["image_sha256"],
                processed_file=str(output.relative_to(root)),
                processed_file_sha256=file_sha256(output),
                processed_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                routing=route,
                reviewer_type="PROGRAM",
                visual_review="NOT_RUN",
            )
        )
    fixtures = [
        '{"evidence":[],"selected":["Mug"],"bindings":[{"category":"Mug","operand":18}],"aggregate":"sum","answer":18}',
        '{"evidence":[],"selected":[],"bindings":[],"aggregate":"mean","answer":"1/2","work":"选择与计算"}',
        '{"evidence":[],"selected":["M\\u0075g"],"bindings":[],"aggregate":"sum","answer":1}',
    ]
    tokenizer = runtime.processor.tokenizer
    token_checks = []
    for text in fixtures:
        ids = tokenizer.encode(text, add_special_tokens=False)
        recovered = tokenizer.decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if recovered != text:
            raise RuntimeError("Real tokenizer fixture cannot reproduce registered JSON bytes")
        token_checks.append(dict(text=text, token_ids=ids, exact_roundtrip=True))
    receipt = dict(
        status="PROGRAM_COMPLETE_VISUAL_REVIEW_PENDING",
        execution_origin="SERVER_ACTUAL",
        config_file_sha256=file_sha256(root / "CONFIG.json"),
        data_manifest_sha256=file_sha256(root / "DATA/MANIFEST.json"),
        identity=runtime.identity,
        bound_files_before_processor=bound_files_check,
        images=results,
        tokenizer_fixtures=token_checks,
        visual_review="NOT_RUN",
        actual_generation_alignment="NOT_RUN",
        model_calls=0,
        optimizer_updates=0,
    )
    write(root / "PREFLIGHT/PROCESSOR.json", receipt)
    return receipt


def _jsonl(path):
    import json

    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["processor-preflight"])
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = processor_preflight(args.root)
    print(
        json.dumps({"status": result["status"], "images": len(result["images"]), "model_calls": 0})
    )


if __name__ == "__main__":
    main()
