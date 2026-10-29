"""Small-source correspondence to original T03, never a new numerical PASS.

The original 24 sequences measured shared aux=0 objective/gradient arithmetic.
They did not measure optimizer updates, nonzero aux/KL after updates, recovery,
new pilot prompts, or the 1536/EOS cases they did not happen to contain.
"""

from __future__ import annotations

import ast
import copy
import importlib
import inspect
from pathlib import Path

from . import historical_confirmation as historical
from .common import digest, file_hash

SCHEMA = "srf6-pilot-v3-kernel-correspondence-v1"
STATUS = "SAME_NUMERICAL_IMPLEMENTATION_BOUND_TO_ORIGINAL_T03"
# Whole-module equality preserves imported targets, globals, dtype, modes and
# lower-level tensor construction, not merely a selected objective expression.
WHOLE_MODULES = (
    "sr_f6.interface_v3_protocol",
    "sr_f6.source_binding",
    "sr_f6.common",
    "sr_f6.planning",
    "sr_f6.numerical_replay",
    "sr_f6.replay",
    "sr_f6.objective",
    "sr_f6.credit",
    "sr_f12.runtime",
    "sr_f12.training",
    "sr_f5.runtime",
    "sr_f5.training",
    "sr_f4.runtime",
    "sr_f4.common",
    "sr_f3_ce.runtime",
    "sr_f3_ce.training",
    "sr_f1.runtime",
    "sr_f1.json_protocol",
    "sr_f1.training",
    "sr_f1.contract",
    "srf5_ref.generator",
    "mm_dev.contract",
    "mm_dev.runtime",
    "mm_core.training",
    "mm_core.vl_runtime",
    "srf6_ref.core",
)
FUNCTIONS = {
    "sr_f6.runtime": (
        "_session_access",
        "backend_protocol",
        "generation_recipe",
        "validate_generation_config",
        "Runtime._replay_session",
        "Runtime.reserve",
        "Runtime.execution_mode",
        "Runtime.policy_tensor_identity",
        "Runtime.generate_group",
        "Runtime._set_recipe",
        "Runtime.prepare_input",
        "Runtime._prepared_records",
        "Runtime._validate_record_revision",
        "Runtime.batch_reference_forward",
        "Runtime.reference_forward",
        "Runtime._batch_forward",
        "Runtime._native_forward",
        "Runtime._generate_native_group",
    ),
    "sr_f6.training": (
        "trajectory_hash",
        "_score_row",
        "build_credit_group",
        "_verify_prepared",
        "_validate_reference",
        "_validate_records",
        "_diagnostic_indices",
        "_validate_optimizer_cursor",
        "_safe",
        "_numerical_audit",
        "_norm",
        "_channel_summary",
        "_update",
    ),
}
LIMITATIONS = [
    "Original T03 only: original first qid, three parents, draws 0..7; zero new sampling",
    "No new trainer numerical replay or actual optimizer update was performed by this check",
    "Original T03 does not measure /128 accumulation, nonzero auxiliary credit, "
    "clipping, Adam or recovery",
    "Original step0 reference=current does not measure updated-policy reference switching "
    "or nonzero KL",
    "Original unobserved length/EOS coverage remains unobserved; "
    "no replacement trajectories selected",
    "Current actual model/processor/library/FP32 identity is separately enforced "
    "by the unchanged loader",
]


def _require(value, message):
    if not value:
        raise PermissionError(message)


def _nodes(source):
    tree = ast.parse(source)
    result = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            result[node.name] = node
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        result[node.name + "." + child.name] = child
    return tree, result


def _dump(node):
    return ast.dump(node, include_attributes=False)


# Exact prospective source-only transformation from the frozen producer. Each
# replacement must match once; the complete resulting module AST must equal the
# deployed module AST. Unlisted functions, added statements, duplicated defs,
# global/import changes and extra numerical loader operations therefore fail.
_V3_HELPERS = '''
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
'''
_V3_LOADED_GUARD = """        if phase == "PILOT_V3" and (
            loaded is None
            or any(
                runtime.identity.get(key) != loaded["execution"].get(key)
                or key not in loaded["execution"]
                for key in V3_INPUT_IDENTITY_FIELDS
            )
        ):
            raise PermissionError("Actual loaded V3 specification or input overlay changed")
"""
_V3_STABLE_FIELDS = """            **{
                key: copy.deepcopy(self.identity[key])
                for key in V3_INPUT_IDENTITY_FIELDS
                if self.identity.get("phase") == "PILOT_V3"
            },
"""
_V3_ENDPOINT_GUARD = """    if identity["phase"] == "PILOT_V3":
        for key in ("input_overlay", "input_overlay_sha256", "prompt_version"):
            if tx.identity.get(key) != identity[key]:
                raise PermissionError("Endpoint transaction V3 input binding differs")
"""
ROUTING_TRANSFORMS = {
    "sr_f6.runtime": (
        (
            'MODEL_PHASES = frozenset({"PILOT_V1", "FORMAL_V1"})',
            'MODEL_PHASES = frozenset({"PILOT_V1", "PILOT_V3", "FORMAL_V1"})',
        ),
        (
            "\n\ndef backend_protocol(protocol):",
            _V3_HELPERS + "\n\ndef backend_protocol(protocol):",
        ),
        (
            '        phase = bound["phase"]\n    return phase',
            '        phase = bound["phase"]\n' + _V3_LOADED_GUARD + "    return phase",
        ),
        (
            "            registered_max_new_tokens=1536,\n        )",
            "            registered_max_new_tokens=1536,\n" + _V3_STABLE_FIELDS + "        )",
        ),
        (
            '            "checkpoint_step",\n        ):',
            '            "checkpoint_step",\n            *V3_INPUT_IDENTITY_FIELDS,\n        ):',
        ),
        ("    run_id=None,\n):", "    run_id=None,\n    pilot_spec_sha256=None,\n):"),
        (
            "    source = verify_runtime_source(root, phase=phase, "
            "phase_identity_sha256=phase_identity_sha256)\n",
            "    source = verify_runtime_source(root, phase=phase, "
            "phase_identity_sha256=phase_identity_sha256)\n"
            "    input_binding = _pilot_input_binding(root, source, pilot_spec_sha256)\n",
        ),
        (
            "            != source\n        ):",
            "            != source\n"
            "            or _pilot_input_binding(root, source, pilot_spec_sha256) "
            "!= input_binding\n"
            "        ):",
        ),
        (
            "            reference_parent=parent,\n        )",
            "            reference_parent=parent,\n            **input_binding,\n        )",
        ),
        (
            "key: runtime.identity[key] for key in EXECUTION_IDENTITY_FIELDS",
            'key: runtime.identity[key] for key in _execution_identity_fields(source["phase"])',
        ),
        (
            "identity.get(key) != loaded_execution.get(key) for key in EXECUTION_IDENTITY_FIELDS",
            "identity.get(key) != loaded_execution.get(key)\n"
            '        for key in _execution_identity_fields(identity.get("phase"))',
        ),
        (
            '    if tx.identity["run_id"] == identity["run_id"]:',
            _V3_ENDPOINT_GUARD + '    if tx.identity["run_id"] == identity["run_id"]:',
        ),
    ),
    "sr_f6.training": (
        (
            'registered_phase = identity.get("phase") in ("PILOT_V1", "FORMAL_V1")',
            'registered_phase = identity.get("phase") in ("PILOT_V1", "PILOT_V3", "FORMAL_V1")',
        ),
    ),
}


def _approved_module(old_text, new_text, module):
    """Compare the entire module, permitting only exact reviewed V3 insertions."""
    original, actual = ast.parse(old_text), ast.parse(new_text)
    if _dump(original) == _dump(actual):
        return dict(
            kind="UNCHANGED_MODULE_AST",
            applied_transforms=0,
            producer_ast_sha256=digest(_dump(original)),
            current_ast_sha256=digest(_dump(actual)),
        )
    expected = old_text
    for before, after in ROUTING_TRANSFORMS[module]:
        _require(expected.count(before) == 1, "Frozen producer routing anchor differs: " + module)
        expected = expected.replace(before, after, 1)
    _require(
        _dump(ast.parse(expected)) == _dump(actual),
        "Unapproved module change outside exact V3 identity routing: " + module,
    )
    return dict(
        kind="EXACT_V3_IDENTITY_TRANSFORMATION",
        applied_transforms=len(ROUTING_TRANSFORMS[module]),
        producer_ast_sha256=digest(_dump(original)),
        current_ast_sha256=digest(_dump(actual)),
    )


def _function_map(old_path, new_path, module):
    old_text, new_text = old_path.read_text(), new_path.read_text()
    _, old_nodes = _nodes(old_text)
    _, new_nodes = _nodes(new_text)
    projection = _approved_module(old_text, new_text, module)
    result = []
    for name in FUNCTIONS[module]:
        before, after = old_nodes[name], new_nodes[name]
        _require(
            _dump(before) == _dump(after),
            "Numerical function changed; original T03 cannot cover " + module + "." + name,
        )
        old_source = ast.get_source_segment(old_text, before)
        new_source = ast.get_source_segment(new_text, after)
        result.append(
            dict(
                function=module + "." + name,
                ast_sha256=digest(_dump(before)),
                producer_source_sha256=digest(old_source),
                current_source_sha256=digest(new_source),
            )
        )
    if module == "sr_f6.runtime":
        _require(
            [_dump(n) for n in old_nodes["Runtime"].bases]
            == [_dump(n) for n in new_nodes["Runtime"].bases],
            "Runtime base class changed",
        )
    changed = sorted(
        name
        for name in old_nodes
        if name != "Runtime"
        and name in new_nodes
        and _dump(old_nodes[name]) != _dump(new_nodes[name])
    )
    return dict(
        functions=result,
        reviewed_routing_changed_symbols=changed,
        complete_module_projection=projection,
    )


def _loaded_origins(deployment):
    """Confirm callable origins and the actual MRO without constructing a model."""
    output = []
    for module_name in (*WHOLE_MODULES, *FUNCTIONS):
        module = importlib.import_module(module_name)
        expected = deployment / "src" / (module_name.replace(".", "/") + ".py")
        _require(
            Path(module.__file__).resolve(strict=True) == expected,
            "Current numerical module comes from a different deployment",
        )
        for name in FUNCTIONS.get(module_name, ()):
            value = module
            for component in name.split("."):
                value = getattr(value, component)
            _require(
                Path(inspect.getfile(value)).resolve(strict=True) == expected,
                "Numerical callable was replaced",
            )
        output.append(dict(module=module_name, path=str(expected)))
    from sr_f5.runtime import Runtime as FP32Runtime

    from .runtime import Runtime

    _require(Runtime.__bases__ == (FP32Runtime,), "Unexpected actual numerical MRO")
    return output


def verify_kernel_compatibility(root, *, phase_context, confirmation_binding):
    """Reject any unproven numerical delta before sampling; do not execute T03."""
    root = historical.phases._root(root)
    context = historical._phase(root, phase_context)
    checked = historical.require_confirmation_binding(
        root, phase_context=context, confirmation_binding_ref=confirmation_binding["binding_ref"]
    )
    _require(checked["binding"] == confirmation_binding["binding"], "Confirmation binding changed")
    facts = checked["verification"]["facts"]
    producer = historical.PRODUCER_DEPLOYMENT
    current = Path(context["deployment"])
    _require(current != producer, "V3 requires a new actual producer deployment")
    rows = []
    for module in (*WHOLE_MODULES, *FUNCTIONS):
        relative = "src/" + module.replace(".", "/") + ".py"
        old_path = historical.phases._file(producer, relative)
        new_path = historical.phases._file(current, relative)
        old_hash, new_hash = file_hash(old_path), file_hash(new_path)
        row = dict(
            module=module, relative_path=relative, producer_sha256=old_hash, current_sha256=new_hash
        )
        if module in WHOLE_MODULES:
            _require(
                old_hash == new_hash,
                "Numerical dependency changed; original T03 does not cover " + module,
            )
            row["comparison"] = "BYTE_IDENTICAL"
        else:
            row.update(_function_map(old_path, new_path, module))
            row["comparison"] = "EXACT_MODULE_AST_WITH_FIXED_V3_IDENTITY_ROUTING"
        rows.append(row)
    origins = _loaded_origins(current)
    numerical = facts["numerical"]
    spec_ref = dict(numerical["spec_ref"])
    spec_ref["path"] = str(historical.FOLLOWUP_ROOT / spec_ref["path"])
    spec = historical._absolute_ref(spec_ref, historical.FOLLOWUP_ROOT / "T03/SPEC.json")
    from .replay import RULES

    expected_rules = dict(
        RULES,
        selection="first qid in sorted frozen DEV_SCHEMA_CONFIRM public; all original draws 0..7",
    )
    _require(
        spec["rules"] == expected_rules and set(spec["groups"]) == {"b0", "b1", "b2"},
        "Original T03 rules or parent selection differ",
    )
    for parent, group in spec["groups"].items():
        _require(
            group["row_ids"] == [f"{parent}/{group['qid']}/{draw}" for draw in range(8)],
            "Original T03 fixed G8 selection differs",
        )
    _require(
        spec["context"]["consumer"] == facts["consumer"]
        and spec["context"]["producer"] == facts["producer"]
        and spec["generated_responses"] == spec["optimizer_updates"] == 0,
        "Original T03 execution identities or zero-call scope differ",
    )
    historical._phase(root, context)
    after = historical.require_confirmation_binding(
        root, phase_context=context, confirmation_binding_ref=checked["binding_ref"]
    )
    _require(
        after["binding"] == checked["binding"], "Historical source changed during correspondence"
    )
    return dict(
        schema=SCHEMA,
        status=STATUS,
        phase_identity_sha256=context["identity_sha256"],
        current_deployment=str(current),
        producer=facts["producer"],
        consumer=facts["consumer"],
        confirmation_binding_ref=checked["binding_ref"],
        original_t03_spec_ref=spec_ref,
        original_numerical_sha256=digest(numerical),
        original_parent_tensors={
            p: group["policy_tensor_hash"] for p, group in spec["groups"].items()
        },
        correspondence=rows,
        loaded_modules=origins,
        call_graph=dict(
            original=[
                "sr_f6_followup.t03.run",
                "sr_f6.numerical_replay._execute",
                "sr_f6_followup.t03_runtime.ReplayRuntime._batch_forward",
                "sr_f6.runtime.Runtime._batch_forward",
                "sr_f3_ce.runtime.SRF3CERuntime._native_forward",
            ],
            current=[
                "sr_f6.training._update",
                "sr_f6.runtime.Runtime._batch_forward",
                "sr_f6.runtime.Runtime._native_forward",
                "sr_f3_ce.runtime.SRF3CERuntime._native_forward",
            ],
        ),
        limitations=copy.deepcopy(LIMITATIONS),
        new_model_calls=0,
        optimizer_updates=0,
        new_trainer_numerical_pass=False,
        training_authorized=False,
        generation_authorized=False,
    )
