"""F2 native visual runtime; the audited MM-CORE implementation stays immutable."""

from __future__ import annotations

import contextlib
import copy
import csv
import hashlib
import json
import math
import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path

from mm_core.training import capture_rng, restore_rng, trainable_state
from mm_core.vl_runtime import QwenRuntime, seed_all

from .contract import PLAN_ID, digest


def read_json(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def bounded_path(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Artifact path must be bounded and relative")
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError("Artifact path escapes run root")
    return result


def adapter_identity(root, adapter_path):
    path = bounded_path(root, adapter_path)
    files = {p.name: file_hash(p) for p in path.iterdir() if p.is_file()}
    if not {"adapter_config.json", "adapter_model.safetensors"}.issubset(files):
        raise ValueError("Adapter has no exact config and safetensors identity")
    return dict(
        adapter_path=str(Path(adapter_path)), adapter_file_hashes=files, adapter_hash=digest(files)
    )


def verified_state(root, state_id):
    root = Path(root)
    common = read_json(root / "manifests/COMMON_LORA.json")
    if (
        common.get("plan_id") != PLAN_ID
        or common.get("status") != "VERIFIED"
        or common.get("freeze_sha256") != file_hash(root / "manifests/F2_FREEZE.json")
    ):
        raise PermissionError("No verified common zero-output LoRA")
    if state_id == "S0":
        entry = common
    else:
        entry = read_json(root / "manifests/STATE_REGISTRY.json")["states"][state_id]
    freeze = read_json(root / "manifests/F2_FREEZE.json")
    if (
        entry.get("freeze_sha256") != file_hash(root / "manifests/F2_FREEZE.json")
        or entry.get("base_model_weights_hash") != freeze["model_identity"]["model_weights_hash"]
    ):
        raise PermissionError("Published state belongs to a different freeze/base")
    gate_path = root / "manifests/ENGINE_GATE.json"
    if gate_path.exists() and read_json(gate_path)["common_lora_sha256"] != file_hash(
        root / "manifests/COMMON_LORA.json"
    ):
        raise PermissionError("Common initialization changed after ENGINE gate")
    actual = adapter_identity(root, entry["adapter_path"])
    if any(actual[key] != entry[key] for key in actual):
        raise PermissionError("Published state adapter identity changed")
    return entry


def copy_parameters(model, parameters):
    """Copy values in place, preserving the optimizer's Parameter identities."""
    import torch

    actual = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if actual.keys() != parameters.keys():
        raise ValueError("Trainable parameter names changed")
    with torch.no_grad():
        for name, parameter in actual.items():
            value = parameters[name]
            if value.shape != parameter.shape or value.dtype != parameter.dtype:
                raise ValueError("Trainable parameter shape/dtype changed")
            parameter.copy_(value.to(parameter.device))


@contextlib.contextmanager
def reference_parameters(model, reference):
    current = trainable_state(model)
    before_ids = [id(p) for p in model.parameters() if p.requires_grad]
    copy_parameters(model, reference)
    try:
        yield
    finally:
        copy_parameters(model, current)
        if before_ids != [id(p) for p in model.parameters() if p.requires_grad]:
            raise RuntimeError("Reference switch replaced optimizer parameters")


class F2Runtime(QwenRuntime):
    """Two explicit sampling channels; training records actual generation logits."""

    def prepare(self, row, run_root):
        from .data import model_input

        root = Path(run_root).resolve()
        cached_root, routes = getattr(self, "_frozen_routes", (None, None))
        if cached_root != root:
            path = root / "data/processor_routing.jsonl"
            freeze = read_json(root / "manifests/F2_FREEZE.json")
            if file_hash(path) != freeze["input_hashes"]["data/processor_routing.jsonl"]:
                raise PermissionError("Frozen native processor-routing manifest changed")
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            routes = {entry["question_id"]: entry for entry in rows}
            if len(routes) != len(rows):
                raise PermissionError("Duplicate frozen processor route")
            self._frozen_routes = root, routes
        safe = model_input(row)
        expected = routes[safe["question_id"]]
        prepared = super().prepare(safe, root)
        if safe["image_id"] != expected["image_id"]:
            raise PermissionError("Frozen image routing identity differs")
        for key, value in prepared["routing"].items():
            if expected.get(key) != value:
                raise PermissionError("Actual native processor routing differs: " + key)
        return prepared

    def reserve(self, kind, count, **metadata):
        if kind == "extra_forward_sequences":
            kind = {
                "training_reference": "reference_forward_sequences",
                "training_gradient": "gradient_forward_sequences",
            }.get(metadata.get("purpose"), kind)
        super().reserve(kind, count, **metadata)

    def training_generation_config(self):
        config = copy.deepcopy(self.generation_config)
        config.temperature = 1.0
        config.top_p = 1.0
        config.top_k = 0
        config.repetition_penalty = 1.0
        config.num_beams = 1
        config.output_logits = True
        config.output_scores = True
        config.return_dict_in_generate = True
        if config.max_new_tokens != 192 or not config.do_sample:
            raise PermissionError("Unregistered training generation recipe")
        return config

    def generate_training(self, row, run_root, seed, *, on_completion=None):
        prepared = self.prepare(row, run_root)
        config = self.training_generation_config()
        self.reserve(
            "completion_attempts", 1, channel="train", question_id=row["question_id"], seed=seed
        )
        rng = capture_rng()
        before = self.image_calls
        start = time.perf_counter()
        try:
            seed_all(seed)
            self.model.eval()
            with self.torch.no_grad():
                result = self.model.generate(**prepared["inputs"], generation_config=config)
            prompt_len = prepared["inputs"]["input_ids"].shape[-1]
            tokens = result.sequences[0, prompt_len:].cpu().tolist()
            errors, logps, sampled_logps = [], [], []
            try:
                if not tokens or len(result.logits) != len(tokens):
                    raise RuntimeError("Sampler did not expose each selected-token raw logit")
                for token, raw, processed in zip(tokens, result.logits, result.scores, strict=True):
                    if not self.torch.equal(raw.float(), processed.float()):
                        errors.append("UNREGISTERED_LOGITS_TRANSFORM")
                    logps.append(float(raw[0].float().log_softmax(-1)[token]))
                    sampled_logps.append(float(processed[0].float().log_softmax(-1)[token]))
                if not all(math.isfinite(v) for v in [*logps, *sampled_logps]):
                    errors.append("NONFINITE_SAMPLER_LOGPROB")
                if self.image_calls <= before:
                    errors.append("NATIVE_VISUAL_ENCODER_NOT_CALLED")
            except Exception as error:
                errors.append(type(error).__name__ + ": " + str(error))
            logps = [v if math.isfinite(v) else None for v in logps]
            sampled_logps = [v if math.isfinite(v) else None for v in sampled_logps]
            record = dict(
                raw_text=self.processor.tokenizer.decode(
                    tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
                ),
                tokens=tokens,
                raw_tokens=tokens,
                generation_status="TECHNICAL_INVALID" if errors else "COMPLETE",
                technical_validation_errors=errors,
                seed=seed,
                old_logprobs=logps,
                sampler_logprobs=sampled_logps,
                old_logprob_source="actual_generate_raw_logits_selected_token",
                sampling_hash=digest(config.to_dict()),
                prompt_token_count=prompt_len,
                completion_token_count=len(tokens),
                truncated=bool(tokens and tokens[-1] not in self.eos_ids and len(tokens) == 192),
                finish_reason="eos" if tokens and tokens[-1] in self.eos_ids else "length",
                generation_seconds=time.perf_counter() - start,
                image_routing={
                    **prepared["routing"],
                    "generation_vision_forward_calls": self.image_calls - before,
                },
            )
            if on_completion is not None:
                on_completion(record)
            self.reserve("generated_tokens", len(tokens), channel="train", seed=seed)
            if errors:
                raise RuntimeError("Recorded invalid training generation: " + "; ".join(errors))
            return record
        finally:
            # Each request has its registered independent seed; replay never advances
            # the surrounding training RNG differently from a fresh completion.
            restore_rng(rng)

    def reference_forward(self, prepared, tokens, reference):
        with reference_parameters(self.model, reference):
            return self.sequence_forward(prepared, tokens, purpose="training_reference", grad=False)


def actual_cuda_identity():
    """Inspect the one visible allocated device before loading model weights."""
    import torch

    if torch.cuda.device_count() != 1:
        raise PermissionError("F2 worker requires exactly one CUDA-visible GPU")
    properties = torch.cuda.get_device_properties(0)
    if "RTX PRO 6000" not in properties.name.upper():
        raise PermissionError("Actual CUDA device is not the audited RTX PRO 6000 class")
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,pci.bus_id,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    devices = [
        dict(
            zip(
                ("name", "uuid", "pci_bus_id", "driver_version", "memory_mib"),
                (part.strip() for part in row),
                strict=True,
            )
        )
        for row in csv.reader(query.stdout.splitlines())
        if row
    ]
    cuda_uuid = str(getattr(properties, "uuid", ""))

    def normalize(value):
        return value.lower().removeprefix("gpu-").replace("-", "")

    matches = [device for device in devices if normalize(device["uuid"]) == normalize(cuda_uuid)]
    if not cuda_uuid or len(matches) != 1:
        raise PermissionError("Cannot match actual CUDA UUID to a unique NVIDIA driver device")
    device = matches[0]
    if device["name"] != properties.name:
        raise PermissionError("CUDA and NVIDIA driver device identities differ")
    return dict(
        cuda_visible_device_count=1,
        cuda_name=properties.name,
        cuda_total_memory_bytes=properties.total_memory,
        cuda_uuid=cuda_uuid,
        compute_capability=[properties.major, properties.minor],
        cuda_runtime=torch.version.cuda,
        hostname=socket.gethostname(),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        **device,
    )


def configure_audited_backend():
    """Match the audited deterministic CUDA settings before allocating model tensors."""
    import torch

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise PermissionError("Audited CUBLAS_WORKSPACE_CONFIG=:4096:8 is required")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return dict(
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
    )


def load_runtime(plan, run_root, *, state_id=None, account=None, freeze=None):
    from peft import PeftModel

    root = Path(run_root)
    freeze = freeze or read_json(root / "manifests/F2_FREEZE.json")
    determinism = configure_audited_backend()
    hardware = actual_cuda_identity()
    runtime = F2Runtime(freeze["model_path"], account=account)
    runtime.identity["hardware"] = hardware
    runtime.identity["determinism"] = determinism
    runtime.verify_identity(freeze["model_identity"])
    for key in (
        "tokenizer_hash",
        "processor_hash",
        "chat_template_hash",
        "linear_kernel_identity_hash",
    ):
        if runtime.identity[key] != plan["model"][key]:
            raise PermissionError("F2 runtime identity differs: " + key)
    if state_id is not None:
        entry = verified_state(root, state_id)
        runtime.model = PeftModel.from_pretrained(
            runtime.model, bounded_path(root, entry["adapter_path"]), is_trainable=False
        )
        runtime.adapter_path = str(bounded_path(root, entry["adapter_path"]))
        runtime.identity.update(get_state_identity(plan, root, state_id))
    return runtime


def get_state_identity(plan, run_root, state_id):
    entry = verified_state(run_root, state_id)
    return dict(
        **{**entry, "state_id": state_id},
        model_hash=digest(
            dict(
                base_model_weights_hash=plan["model"]["model_weights_hash"],
                adapter_file_hashes=entry["adapter_file_hashes"],
            )
        ),
        state_manifest_hash=digest(entry),
        processor_hash=plan["model"]["processor_hash"],
    )


def create_evaluation_runtime(plan, run_root, state_id, account=None):
    runtime = load_runtime(plan, run_root, state_id=state_id, account=account)
    return runtime, runtime.identity


def tensor_summary(values):
    values = values.detach().float()
    return dict(mean=float(values.mean()), minimum=float(values.min()), maximum=float(values.max()))
