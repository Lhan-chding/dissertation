"""SR-F6 native token trajectory adapter with an explicit 1536-token recipe.

The copied sampling body is the frozen SR-F3/F5 native path. Its local recipe
changes only the registered token limit. It does not monkeypatch legacy globals.
"""

from __future__ import annotations

import copy
import math
import re
import sys
import time
import weakref
from contextlib import nullcontext
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
EXECUTION_MODES = frozenset({"CENSUS", "TRAINING", "EVALUATION", "REPLAY"})
MODEL_PHASES = frozenset({"PILOT_V1", "PILOT_V3", "FORMAL_V1"})
_LOADED_RUNTIMES = weakref.WeakKeyDictionary()
_LOADED_ENDPOINTS = weakref.WeakKeyDictionary()
EXECUTION_IDENTITY_FIELDS = (
    "run_root",
    "audit_root",
    "task_id",
    "run_id",
    "phase",
    "execution_mode",
    "phase_identity_sha256",
)
V3_INPUT_IDENTITY_FIELDS = (
    "pilot_spec_sha256",
    "input_overlay",
    "input_overlay_sha256",
    "prompt_version",
)


def _execution_identity_fields(phase):
    return EXECUTION_IDENTITY_FIELDS + (V3_INPUT_IDENTITY_FIELDS if phase == "PILOT_V3" else ())


def _pilot_input_binding(root, source, pilot_spec_sha256):
    """Bind the real V3 plan before loading; this does not grant admission."""
    if source["phase"] != "PILOT_V3":
        if pilot_spec_sha256 is not None:
            raise PermissionError("The V3 input pin requires its explicit PILOT_V3 phase")
        return {}
    if not isinstance(pilot_spec_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", pilot_spec_sha256
    ):
        raise PermissionError("PILOT_V3 requires its actual frozen pilot specification")
    from .pilot_contract import verify_registration

    context = {k: v for k, v in source.items() if k != "origin_identity_sha256"}
    registered = verify_registration(
        root, phase_context=context, pilot_spec_sha256=pilot_spec_sha256
    )
    overlay = registered["overlay"]
    return dict(
        pilot_spec_sha256=pilot_spec_sha256,
        input_overlay=copy.deepcopy(overlay["manifest_ref"]),
        input_overlay_sha256=overlay["manifest_ref"]["sha256"],
        prompt_version=overlay["manifest"]["prompt_version"],
    )


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


def _actual_deployment():
    """The caller cannot substitute a different deployment for loaded Python."""
    return Path(__file__).resolve(strict=True).parents[2]


def _verify_module_origins(deployment, manifest, modules):
    members = {item["path"] for item in manifest["files"]}
    packages = {
        Path(path).parts[1].removesuffix(".py") for path in members if path.startswith("src/")
    }
    for module in modules:
        name = getattr(module, "__name__", "")
        path = getattr(module, "__file__", None)
        if name.split(".")[0] not in packages or path is None:
            continue
        path = Path(path).resolve(strict=True)
        if not path.is_relative_to(deployment):
            raise PermissionError("Imported source dependency is outside the verified deployment")
        if str(path.relative_to(deployment)) not in members:
            raise PermissionError("Imported source dependency is missing from the phase manifest")


def verify_runtime_source(root, *, phase=None, phase_identity_sha256=None):
    """Bind actual imported source only; this function never grants execution.

    Phase scientific prerequisite validation and evaluation access locks are not
    implemented by this adapter. New-phase generation therefore fails closed.
    """
    from .phases import ORIGIN, verify_origin, verify_phase

    deployment = _actual_deployment()
    if phase is None:
        if phase_identity_sha256 is not None:
            raise PermissionError("A phase pin requires an explicit registered phase")
        source = verify_origin(root, deployment=deployment)
    else:
        if (
            phase not in MODEL_PHASES
            or not isinstance(phase_identity_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", phase_identity_sha256)
        ):
            raise PermissionError("Model phases require a pinned full phase identity SHA256")
        source = verify_phase(
            root,
            phase,
            deployment=deployment,
            expected_identity_sha256=phase_identity_sha256,
        )
    identity = source["identity"]
    manifest = read(Path(root) / identity["source_manifest"]["path"])
    _verify_module_origins(deployment, manifest, tuple(sys.modules.values()))
    if source["status"] != "BOUND_NOT_AUTHORIZED" or source["training_allowed"] is not False:
        raise PermissionError("Source verification cannot masquerade as scientific authorization")
    return {
        **source,
        "origin_identity_sha256": source["identity_sha256"]
        if source["phase"] == ORIGIN
        else identity["origin"]["identity"]["sha256"],
    }


def _execution_mode(mode, training):
    if training is not None and type(training) is not bool:
        raise ValueError("training must be a boolean compatibility option")
    if mode is not None and (not isinstance(mode, str) or mode not in EXECUTION_MODES):
        raise PermissionError("Unknown runtime execution mode")
    selected = mode if mode is not None else ("TRAINING" if training else "CENSUS")
    if selected not in EXECUTION_MODES or (
        training is not None and mode is not None and training != (mode == "TRAINING")
    ):
        raise PermissionError("Conflicting or unknown runtime execution mode")
    return selected


def _audit_directory(root, phase, audit_root):
    """New-phase writes are confined to their explicit per-path namespace."""
    root = Path(root).resolve(strict=True)
    if phase == "CENSUS_S0" and audit_root is None:
        return root
    if audit_root is None:
        raise PermissionError("A new model phase requires its independent audit_root")
    audit = Path(audit_root)
    base = root / "PHASE_RUNS" / phase
    if (
        not audit.is_absolute()
        or ".." in audit.parts
        or not audit.is_relative_to(base)
        or audit == base
        or audit.resolve(strict=False) != audit
    ):
        raise PermissionError("Audit root must be a canonical descendant of its phase namespace")
    for path in (audit, *audit.parents):
        if path == root:
            break
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise PermissionError("Unsafe phase audit directory")
    return audit


def _runtime_artifact(audit, category, name):
    if audit.is_symlink() or audit.resolve(strict=False) != audit:
        raise PermissionError("Runtime audit root changed or became a symlink")
    path = audit / category / name
    for item in (path, *path.parents):
        if item == audit:
            break
        if item.is_symlink():
            raise PermissionError("Symlinked runtime evidence is forbidden")
        if item.exists() and not (item.is_file() if item == path else item.is_dir()):
            raise PermissionError("Runtime evidence has an invalid file type")
    return path


def _freeze_evaluation(runtime, step, policy_hash):
    """Bind a read-only adapter after actual parent/COMMIT content validation."""
    names = runtime._srf6_adapter_parameter_names
    parameters = dict(runtime.model.named_parameters())
    for value in parameters.values():
        value.requires_grad_(False)
        value.grad = None
    runtime.model.eval()
    runtime._srf6_readonly_policy = (step, policy_hash)
    runtime._srf6_readonly_versions = {
        name: (id(parameters[name]), parameters[name]._version) for name in names
    }
    runtime._srf6_policy_identity = (step, policy_hash)


def _check_loaded_source_identity(runtime):
    phase = runtime.identity.get("phase", "CENSUS_S0")
    bound = getattr(runtime, "_srf6_source_context", None)
    loaded = _LOADED_RUNTIMES.get(runtime) if isinstance(runtime, Runtime) else None
    if loaded is not None and bound != loaded["source"]:
        raise PermissionError("Actual loaded source is missing or changed; cannot downgrade to S0")
    if bound is not None:
        expected = {
            "phase": bound["phase"],
            "source_sha256": bound["identity"]["source_manifest"]["sha256"],
            "source_hash": bound["identity"]["source_manifest"]["sha256"],
            "data_hash": bound["identity"]["data_manifest"]["sha256"],
            "origin_identity_sha256": bound["origin_identity_sha256"],
            "phase_identity_sha256": bound["identity_sha256"]
            if bound["phase"] != "CENSUS_S0"
            else None,
        }
        if any(runtime.identity.get(key) != value for key, value in expected.items()):
            raise PermissionError("Runtime identity differs from its actual loaded source")
        phase = bound["phase"]
        if phase == "PILOT_V3" and (
            loaded is None
            or any(
                runtime.identity.get(key) != loaded["execution"].get(key)
                or key not in loaded["execution"]
                for key in V3_INPUT_IDENTITY_FIELDS
            )
        ):
            raise PermissionError("Actual loaded V3 specification or input overlay changed")
    return phase


def _generation_authorized(runtime):
    phase = _check_loaded_source_identity(runtime)
    if phase != "CENSUS_S0":
        from .access import active_session

        return active_session(runtime)
    if hasattr(runtime, "_srf6_execution_identity") and not hasattr(
        runtime, "_srf6_source_context"
    ):
        raise PermissionError("Actual loaded source is missing; cannot downgrade to S0")
    return None


def _session_access(session):
    if session.runtime.execution_mode() == "REPLAY":
        from .replay import _RUNTIME_ACCESS
    else:
        from .access import _RUNTIME_ACCESS
    return _RUNTIME_ACCESS


class Runtime(FP32Runtime):
    def _replay_session(self):
        if self in _LOADED_RUNTIMES:
            _check_loaded_source_identity(self)
        if self.execution_mode() == "REPLAY":
            from .replay import active_session

            return active_session(self)
        source = getattr(self, "_srf6_source_context", None)
        if source is not None:
            if _check_loaded_source_identity(self) != "CENSUS_S0":
                from .access import active_session

                return active_session(self)
        elif hasattr(self, "_srf6_execution_identity"):
            raise PermissionError("Actual loaded source is missing; cannot downgrade to S0")
        return None

    def reserve(self, kind, count, **metadata):
        # The unchanged native sampler calls this before any model.generate.
        # This also closes direct calls to its private sampling entrypoint.
        if self.execution_mode() == "REPLAY" and kind != "extra_forward_sequences":
            raise PermissionError("REPLAY forbids generation and completion accounting")
        session = self._replay_session()
        if session is not None and self.execution_mode() != "REPLAY":
            session.check_reservation(kind, count, metadata)
        return super().reserve(kind, count, **metadata)

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
        session = self._replay_session()
        if session is not None:
            session.check_input(model_input, run_root, protocol)
        prepared = super().prepare_input(model_input, run_root, protocol=backend_protocol(protocol))
        prepared["routing"]["sr_f6_protocol_version"] = PROTOCOL_VERSION
        if session is not None:
            session.bind_prepared(prepared, access=_session_access(session))
        return prepared

    def _prepared_records(self, records):
        session = self._replay_session()
        if session is not None:
            session.check_records(records)
        prepared = super()._prepared_records(records)
        if session is not None:
            session.check_prepared(prepared)
        return prepared

    def batch_reference_forward(self, records, reference):
        session = self._replay_session()
        if session is None or self.execution_mode() != "REPLAY":
            if session is not None:
                session.check_reference(reference)
            return super().batch_reference_forward(records, reference)
        # At T03 step0 current and reference are the same verified parent.
        # Avoid copying identical parameters and changing tensor version counters.
        session.check_reference(reference)
        return self.batch_sequence_forward(records, purpose="t03_reference", grad=False)

    def reference_forward(self, prepared, tokens, reference):
        session = self._replay_session()
        if session is None or self.execution_mode() != "REPLAY":
            if session is not None:
                session.check_reference(reference)
            return super().reference_forward(prepared, tokens, reference)
        session.check_reference(reference)
        return self.sequence_forward(prepared, tokens, purpose="t03_reference", grad=False)

    def _native_forward(self, inputs, *, completion_start, lengths, purpose, grad):
        session = self._replay_session()
        if session is not None:
            session.check_native(
                inputs,
                completion_start=completion_start,
                lengths=lengths,
                purpose=purpose,
                grad=grad,
            )
        return super()._native_forward(
            inputs, completion_start=completion_start, lengths=lengths, purpose=purpose, grad=grad
        )

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
            **{
                key: copy.deepcopy(self.identity[key])
                for key in V3_INPUT_IDENTITY_FIELDS
                if self.identity.get("phase") == "PILOT_V3"
            },
        )

    def policy_tensor_identity(self):
        """Hash once at load/committed-step transition, not once per question."""
        step = self.identity["step"]
        if type(step) is not int or step < 0:
            raise PermissionError("A committed nonnegative policy step is required")
        mode = self.execution_mode()
        if mode == "REPLAY":
            if step != 0:
                raise PermissionError("REPLAY requires the frozen step0 parent")
            self._replay_session()
            return self._srf6_policy_identity[1]
        if mode == "EVALUATION":
            expected = getattr(self, "_srf6_readonly_policy", None)
            parameters = dict(self.model.named_parameters())
            versions = {
                name: (id(parameters[name]), parameters[name]._version)
                for name in self._srf6_adapter_parameter_names
            }
            if (
                expected is None
                or expected[0] != step
                or self._srf6_policy_identity != expected
                or versions != self._srf6_readonly_versions
                or any(value.requires_grad for value in parameters.values())
            ):
                raise PermissionError(
                    "Read-only evaluation policy no longer matches its bound content"
                )
            return expected[1]
        if mode == "CENSUS" and step != 0:
            raise PermissionError("Frozen parent census cannot advance model parameters")
        cached = getattr(self, "_srf6_policy_identity", None)
        if cached is None or cached[0] != step:
            value = state_hash(trainable_state(self.model))
            cached = (step, value)
            self._srf6_policy_identity = cached
            self.identity["policy_hash"] = value
        return cached[1]

    def execution_mode(self):
        mode = getattr(self, "identity", {}).get(
            "execution_mode", "TRAINING" if getattr(self, "_srf6_training", False) else "CENSUS"
        )
        if mode not in EXECUTION_MODES or mode != getattr(self, "_srf6_execution_mode", mode):
            raise PermissionError("Unknown runtime mode")
        return mode

    def _annotate_runtime_row(self, row, backend, policy_tensor_hash):
        row.update(
            protocol="structured_v4",
            backend_protocol=backend,
            sr_f6_protocol_version=PROTOCOL_VERSION,
            seed_semantics="independent_per_draw_torch_generator",
            policy_hash=policy_tensor_hash,
            policy_tensor_hash=policy_tensor_hash,
            parent_content_hash=self.identity["parent_content_hash"],
            source_sha256=self.identity["source_sha256"],
            source_hash=self.identity["source_sha256"],
            data_hash=self.identity["data_hash"],
            run_id=self.identity["run_id"],
            policy_step=self.identity["step"],
            step=self.identity["step"] + int(self.execution_mode() == "TRAINING"),
            execution_mode=self.execution_mode(),
        )
        for key in (
            "phase",
            "task_id",
            "origin_identity_sha256",
            "phase_identity_sha256",
            "checkpoint_run_id",
            "checkpoint_commit_sha256",
            "checkpoint_step",
            *V3_INPUT_IDENTITY_FIELDS,
        ):
            if key in self.identity:
                row[key] = self.identity[key]

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
        session = _generation_authorized(self)
        backend = backend_protocol(protocol)
        policy_tensor_hash = self.policy_tensor_identity()

        def capture(row):
            self._annotate_runtime_row(row, backend, policy_tensor_hash)
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

        permit = (
            session.generation(
                model_input, run_root, seeds, protocol=protocol, access=_session_access(session)
            )
            if session is not None
            else nullcontext()
        )
        with permit, independent_draw_rng(self.torch, seeds):
            return self._generate_native_group(
                model_input,
                run_root,
                seeds,
                protocol=backend,
                generation=generation,
                on_completion=capture,
            )

    def _batch_forward(self, prepared, token_rows, *, purpose, grad):
        session = self._replay_session()
        if session is not None:
            session.check_batch(prepared, token_rows)
        if self.execution_mode() == "EVALUATION":
            if grad:
                raise PermissionError("Evaluation runtime cannot request gradient forwards")
            self.policy_tensor_identity()
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
        forward = dict(
            completion_start=prompt_len,
            lengths=list(map(len, token_rows)),
            purpose=purpose,
            grad=grad,
        )
        if session is not None:
            with session.native_call(inputs, **forward, access=_session_access(session)):
                return self._native_forward(inputs, **forward)
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


def load(
    root,
    parent,
    task_id,
    *,
    training=None,
    mode=None,
    phase=None,
    phase_identity_sha256=None,
    audit_root=None,
    run_id=None,
    pilot_spec_sha256=None,
):
    """Load a bound parent, with separate provenance and execution semantics.

    Default S0 loading retains its interface but verifies the actual historical
    deployment. New phases require an explicit full identity pin and audit root.
    They remain NOT_AUTHORIZED: phase scientific gates, evaluation-panel access
    locks, and orchestration are not implemented here, so generation is blocked.
    EVALUATION creates no optimizer and cannot perform gradient forwards.
    PILOT REPLAY keeps adapter gradients, creates no optimizer, and requires an
    active original-token session for every forward. It cannot sample/update.
    """
    import torch

    from mm_dev.runtime import actual_cuda_identity, configure_audited_backend
    from sr_f3_ce.training import fresh_optimizer

    from .common import write_once

    selected_mode = _execution_mode(mode, training)
    if selected_mode == "REPLAY" and phase != "PILOT_V1":
        raise PermissionError("REPLAY is restricted to the registered PILOT_V1 phase")
    if not isinstance(task_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", task_id
    ):
        raise PermissionError("Runtime task_id must be one safe execution identifier")
    if selected_mode == "EVALUATION" and phase is None:
        raise PermissionError("Evaluation requires an explicit registered execution phase")
    if phase is None and run_id is not None:
        raise PermissionError("Legacy S0 run_id remains its task_id")
    if phase is not None and selected_mode == "TRAINING" and run_id is None:
        raise PermissionError("Phase training requires its stable scientific run_id")
    if run_id is not None and (not isinstance(run_id, str) or not run_id.strip()):
        raise PermissionError("Explicit run_id must be nonempty")
    effective_run_id = task_id if run_id is None else run_id
    root = Path(root)
    source = verify_runtime_source(root, phase=phase, phase_identity_sha256=phase_identity_sha256)
    input_binding = _pilot_input_binding(root, source, pilot_spec_sha256)
    audit = _audit_directory(root, source["phase"], audit_root)
    identity_path = _runtime_artifact(audit, "runtime_identity", task_id + ".json")
    if phase is not None and identity_path.exists():
        raise FileExistsError("Preserve an existing runtime execution receipt; use a new task_id")
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

    def construct():
        seed_all(seed_for(config["master_seed"], "load", parent))

        def account(kind, count, metadata):
            if type(count) is not int or count < 0:
                raise ValueError("Invalid model-call accounting")
            append(
                _runtime_artifact(audit, "accounting", task_id + ".jsonl"),
                dict(
                    at=now(),
                    task_id=task_id,
                    parent=parent,
                    kind=kind,
                    count=count,
                    metadata=metadata,
                ),
            )

        runtime = Runtime(
            config["model"]["model_path"],
            adapter_path=str(adapter),
            output_root=root,
            protocol_revision="v2",
            account=account,
        )
        configure_parent_training(
            runtime, parent_adapter_config=read(adapter / "adapter_config.json")
        )
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
            run_id=effective_run_id,
            step=0,
            snapshot=config["model"]["snapshot"],
            sr_f6_protocol_version=PROTOCOL_VERSION,
        )
        _, post_load_files_check = verify_binding_inputs(root, config, parent)
        if (
            verify_runtime_source(root, phase=phase, phase_identity_sha256=phase_identity_sha256)
            != source
            or _pilot_input_binding(root, source, pilot_spec_sha256) != input_binding
        ):
            raise PermissionError("Source identity changed during model loading")
        identity = source["identity"]
        tensor_hash = state_hash(trainable_state(runtime.model))
        runtime._srf6_adapter_parameter_names = tuple(
            name for name, value in runtime.model.named_parameters() if value.requires_grad
        )
        runtime._srf6_training = selected_mode == "TRAINING"
        runtime._srf6_execution_mode = selected_mode
        runtime._srf6_policy_identity = (0, tensor_hash)
        runtime._srf6_source_context = copy.deepcopy(source)
        runtime.identity.update(
            run_root=str(root.resolve(strict=True)),
            audit_root=str(audit),
            task_id=task_id,
            phase=source["phase"],
            execution_mode=selected_mode,
            source_sha256=identity["source_manifest"]["sha256"],
            source_hash=identity["source_manifest"]["sha256"],
            data_hash=identity["data_manifest"]["sha256"],
            config_sha256=identity["config_sha256"],
            origin_identity_sha256=source["origin_identity_sha256"],
            phase_identity_sha256=source["identity_sha256"] if phase is not None else None,
            source_binding_status="BOUND_NOT_AUTHORIZED",
            execution_authorization="NOT_AUTHORIZED"
            if phase is not None
            else "CALLER_CENSUS_GATE_REQUIRED",
            parent_content_hash=bound["parents"][parent]["content_sha256"],
            policy_hash=tensor_hash,
            reference_hash=tensor_hash,
            reference_parent=parent,
            **input_binding,
        )
        runtime._srf6_execution_identity = {
            key: runtime.identity[key] for key in _execution_identity_fields(source["phase"])
        }
        trainable = [
            {"name": n, "shape": list(p.shape), "dtype": str(p.dtype)}
            for n, p in runtime.model.named_parameters()
            if p.requires_grad
        ]
        if selected_mode == "EVALUATION":
            _freeze_evaluation(runtime, 0, tensor_hash)
        optimizer, scheduler = (
            fresh_optimizer(runtime.model) if selected_mode == "TRAINING" else (None, None)
        )
        writer = write_once if phase is not None else write
        writer(
            identity_path,
            dict(
                copy.deepcopy(runtime.identity),
                hardware=actual_cuda_identity(),
                inherited_adapter_parameters=trainable,
                trainable_parameters=[] if selected_mode == "EVALUATION" else trainable,
                fresh_optimizer=selected_mode == "TRAINING",
                source_binding_note="content binding and loaded identity are separate receipts",
                bound_files_before_load=bound_files_check,
                bound_files_after_load=post_load_files_check,
            ),
        )
        _LOADED_RUNTIMES[runtime] = dict(
            execution=copy.deepcopy(runtime._srf6_execution_identity),
            source=copy.deepcopy(source),
            model=runtime.model,
            parent={
                key: runtime.identity[key]
                for key in (
                    "parent_id",
                    "reference_parent",
                    "reference_hash",
                    "parent_content_hash",
                    "config_sha256",
                )
            },
        )
        return runtime, optimizer, scheduler

    # Configure the audited backend above before touching CUDA RNG state.
    # Evaluation loading preserves its caller even when construction fails.
    if selected_mode in {"EVALUATION", "REPLAY"}:
        rng = capture_rng()
        try:
            return construct()
        finally:
            restore_rng(rng)
    return construct()


def load_committed_endpoint(runtime, tx, *, expected_commit_sha256):
    """Copy a pinned, published adapter into an EVALUATION runtime, read-only.

    ``tx`` must already hold its exclusive RunTransactions lock. This does not
    create an optimizer, restore training RNG, run evaluation, or authorize any
    panel access. PILOT generation still requires a separately verified active
    evaluation access session after successful loading; FORMAL remains closed.
    """
    import json

    import torch

    from sr_f3_ce.training import finite_tree

    from .common import write_once
    from .transactions import RunTransactions, _cursor, _file_hash, _read_bytes

    if runtime.execution_mode() != "EVALUATION" or not isinstance(tx, RunTransactions):
        raise PermissionError("Endpoint loading requires EVALUATION and a real locked transaction")
    if not isinstance(expected_commit_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_commit_sha256
    ):
        raise PermissionError("An exact committed endpoint identity is required")
    identity = runtime.identity
    _check_loaded_source_identity(runtime)
    if identity.get("phase") not in MODEL_PHASES:
        raise PermissionError("Endpoint evaluation requires its registered execution phase")
    loaded_execution = getattr(runtime, "_srf6_execution_identity", None)
    if loaded_execution is None or any(
        identity.get(key) != loaded_execution.get(key)
        for key in _execution_identity_fields(identity.get("phase"))
    ):
        raise PermissionError("Endpoint runtime execution identity changed after loading")
    if not isinstance(identity.get("task_id"), str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", identity["task_id"]
    ):
        raise PermissionError("Endpoint task_id is not a safe execution filename")
    audit = _audit_directory(identity["run_root"], identity["phase"], identity["audit_root"])
    source = verify_runtime_source(
        identity["run_root"],
        phase=identity["phase"],
        phase_identity_sha256=identity["phase_identity_sha256"],
    )
    if source != runtime._srf6_source_context:
        raise PermissionError("Endpoint runtime source binding changed")
    _audit_directory(identity["run_root"], identity["phase"], tx.directory)
    tx._locked()
    for key, expected in (
        ("phase", identity["phase"]),
        ("parent", identity["parent_id"]),
        ("source_sha256", identity["source_sha256"]),
        ("data_sha256", identity["data_hash"]),
        ("config_sha256", identity["config_sha256"]),
        ("reference_sha256", identity["reference_hash"]),
        ("experiment_id", identity["experiment_id"]),
    ):
        if tx.identity.get(key) != expected:
            raise PermissionError("Endpoint transaction differs from runtime " + key)
    if identity["phase"] == "PILOT_V3":
        for key in ("input_overlay", "input_overlay_sha256", "prompt_version"):
            if tx.identity.get(key) != identity[key]:
                raise PermissionError("Endpoint transaction V3 input binding differs")
    if tx.identity["run_id"] == identity["run_id"]:
        raise PermissionError(
            "Evaluation run and checkpoint producer must have distinct identities"
        )
    markers = tx.published()
    matched = [marker for marker in markers if marker["commit_sha256"] == expected_commit_sha256]
    if len(matched) != 1:
        raise PermissionError("Pinned endpoint is not a unique published commit")
    marker = matched[0]
    step = marker["step"]
    checkpoint = tx.checkpoints / f"step-{step:03d}"
    receipt_path = _runtime_artifact(
        audit, "runtime_identity", identity["task_id"] + "-endpoint.json"
    )
    if receipt_path.exists():
        raise FileExistsError("Endpoint execution receipt already exists")
    rng = capture_rng()
    try:
        path = checkpoint / "state.pt"
        before = path.stat()
        if _file_hash(path) != marker["state_sha256"]:
            raise PermissionError("Endpoint checkpoint state content differs")
        state = torch.load(path, map_location="cpu", weights_only=True)
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise PermissionError("Endpoint state changed while loading")
        if (
            state.get("identity") != tx.identity
            or state.get("step") != step
            or state.get("sampler_state") != marker["sampler_state"]
        ):
            raise PermissionError("Endpoint serialized metadata differs from its commit")
        parameters = dict(runtime.model.named_parameters())
        names = runtime._srf6_adapter_parameter_names
        policy = state.get("policy")
        if (
            not isinstance(policy, dict)
            or set(policy) != set(names)
            or set(marker.get("parameter_names", [])) != set(names)
            or not finite_tree(policy)
            or state_hash(policy) != marker["policy_sha256"]
        ):
            raise PermissionError("Endpoint adapter content/names differ")
        for name, value in policy.items():
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != torch.float32
                or parameters[name].dtype != torch.float32
                or value.shape != parameters[name].shape
            ):
                raise PermissionError("Endpoint adapter shape/dtype differs")
        if step:
            update_bytes = _read_bytes(checkpoint / "UPDATE.json")
            import hashlib

            if hashlib.sha256(update_bytes).hexdigest() != marker["update_sha256"]:
                raise PermissionError("Endpoint update receipt changed")
            update = json.loads(update_bytes)
            qids = tx._evidence(
                marker["references"],
                update,
                step,
                marker["policy_before_sha256"],
                marker["rng_sha256"],
            )
            if update.get("policy_after_sha256") != marker["policy_sha256"]:
                raise PermissionError("Endpoint update policy differs from commit")
            _cursor(
                state["sampler_state"],
                step,
                markers[step - 1]["sampler_state"],
                qids,
                marker["references"]["rollout"],
            )
        with torch.no_grad():
            for name, value in policy.items():
                parameters[name].copy_(value)
        actual = state_hash({name: parameters[name].detach().cpu().clone() for name in names})
        if actual != marker["policy_sha256"]:
            raise PermissionError("Actual loaded endpoint parameters differ")
        _freeze_evaluation(runtime, step, actual)
        identity.update(
            step=step,
            policy_hash=actual,
            checkpoint_step=step,
            checkpoint_run_id=tx.identity["run_id"],
            checkpoint_commit_sha256=expected_commit_sha256,
            checkpoint_state_sha256=marker["state_sha256"],
            execution_authorization="NOT_AUTHORIZED",
        )
        report = {
            "status": "ENDPOINT_LOADED_NOT_AUTHORIZED",
            "runtime_identity": copy.deepcopy(identity),
            "checkpoint_identity": copy.deepcopy(tx.identity),
            "checkpoint": str(checkpoint),
            "optimizer_created": False,
            "training_rng_restored": False,
            "model_calls": 0,
            "optimizer_updates": 0,
        }
        write_once(receipt_path, report)
        _LOADED_ENDPOINTS[runtime] = dict(
            tx=tx,
            run_id=tx.identity["run_id"],
            step=step,
            commit_sha256=expected_commit_sha256,
            policy_sha256=actual,
        )
        return report
    finally:
        restore_rng(rng)


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
