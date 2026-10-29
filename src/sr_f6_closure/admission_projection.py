"""Read-only admission projection for the separately registered closure repair.

The copied predicates retain the original admission schema and execution
consumer. They are inner facts only: the new closure wrapper records the repair
verifier, immutable UNKNOWN history and separate registration. This module has
no seal, receipt writer, sampling, training or historical authorization entrypoint.
"""

from __future__ import annotations

import ast
import copy as ast_copy
import hashlib
from pathlib import Path
from types import CodeType

from sr_f6_followup import admission as original

ORIGINAL_SHA256 = "f0cd5602c71270d032f7f964d00ca62e827901df86c3542cd00320e165031114"
FUNCTIONS = ("_facts", "assess")
ADDITIONAL_FUNCTIONS = ("compose_assessment", "evaluate")
source = original.source
_require = original._require
_synthetic = original._synthetic
SCHEMA = original.SCHEMA
STATUS = original.STATUS
SCOPE = original.SCOPE
copy = original.copy
digest = original.digest


def _functions(text, names=FUNCTIONS):
    tree = ast.parse(text)
    result = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            _require(node in tree.body, "Projected admission functions must remain top-level")
            _require(node.name not in result, "Duplicate projected admission function")
            _require(
                isinstance(node, ast.FunctionDef) and not node.decorator_list,
                "Projected admission function decorators or async execution forbidden",
            )
            result[node.name] = node
    _require(set(result) == set(names), "Missing projected admission function")
    return result


def _prove_sources(original_text, projected_text):
    expected = _functions(original_text)
    actual = _functions(projected_text, FUNCTIONS + ADDITIONAL_FUNCTIONS)
    expected = ast_copy.deepcopy(expected)
    imports = [
        node
        for node in ast.walk(expected["_facts"])
        if isinstance(node, ast.ImportFrom)
        and node.module is None
        and node.level == 1
        and len(node.names) == 1
        and node.names[0].name == "t03"
        and node.names[0].asname is None
    ]
    _require(len(imports) == 1, "Original admission numerical import differs")
    imports[0].module = "sr_f6_closure"
    imports[0].level = 0
    imports[0].names = [ast.alias(name="projection", asname="t03")]
    for name in FUNCTIONS:
        _require(
            ast.dump(expected[name], include_attributes=False)
            == ast.dump(actual[name], include_attributes=False),
            "Admission projection changed a frozen predicate: " + name,
        )
    # The extra API reads the facts only once and delivers the complete
    # numerical report alongside exactly the original admission composition.
    original_assess = expected["assess"]
    _require(
        len(original_assess.body) == 5
        and isinstance(original_assess.body[0], ast.Expr)
        and isinstance(original_assess.body[1], ast.Try)
        and len(original_assess.body[1].handlers) == 1
        and len(original_assess.body[1].handlers[0].body) == 1
        and isinstance(original_assess.body[1].handlers[0].body[0], ast.Return),
        "Original admission assessment structure differs",
    )
    compose = ast.parse("def compose_assessment(context, review, numerical):\n    pass\n").body[0]
    compose.body = ast_copy.deepcopy(original_assess.body[2:])
    evaluate = ast_copy.deepcopy(original_assess)
    evaluate.name = "evaluate"
    unknown = evaluate.body[1].handlers[0].body[0]
    unknown.value = ast.Call(
        func=ast.Name(id="dict", ctx=ast.Load()),
        args=[],
        keywords=[
            ast.keyword(arg="assessment", value=unknown.value),
            ast.keyword(arg="numerical", value=ast.Constant(value=None)),
        ],
    )
    evaluate.body[2:] = ast.parse(
        "result = dict(assessment=compose_assessment(context, review, numerical), "
        "numerical=numerical)"
    ).body
    evaluate.body[2] = ast.Return(value=evaluate.body[2].value)
    for name, node in (("compose_assessment", compose), ("evaluate", evaluate)):
        _require(
            ast.dump(node, include_attributes=False)
            == ast.dump(actual[name], include_attributes=False),
            "Admission single-read composition changed: " + name,
        )
    return dict(
        functions=list(FUNCTIONS + ADDITIONAL_FUNCTIONS),
        changed_imports=1,
        changed_predicates=0,
        single_read_composition=True,
    )


def verify_projection(original_path=None):
    """Prove the only copied-function change is the numerical reader import.

    This is a source correspondence check, not evidence validation or scientific
    admission. The caller separately verifies the registered package inventories
    and module origins before and after lazy imports.
    """
    old_path = Path(original.__file__).resolve(strict=True)
    if original_path is not None:
        _require(
            Path(original_path).resolve(strict=True) == old_path, "Original module path differs"
        )
    payload = old_path.read_bytes()
    _require(
        hashlib.sha256(payload).hexdigest() == ORIGINAL_SHA256, "Frozen admission source changed"
    )
    new_path = Path(__file__).resolve(strict=True)
    projected_source = new_path.read_text(encoding="utf-8")
    result = _prove_sources(payload.decode("utf-8"), projected_source)
    compiled = compile(projected_source, str(new_path), "exec", dont_inherit=True)
    for name in FUNCTIONS + ADDITIONAL_FUNCTIONS:
        function = globals()[name]
        expected = [
            code
            for code in compiled.co_consts
            if isinstance(code, CodeType) and code.co_name == name
        ]
        _require(
            len(expected) == 1
            and function.__module__ == __name__
            and function.__name__ == name
            and Path(function.__code__.co_filename).resolve(strict=True) == new_path
            and function.__code__ == expected[0],
            "Loaded admission function differs from new source: " + name,
        )
    for name in ("source", "_require", "_synthetic", "copy", "digest"):
        _require(globals()[name] is getattr(original, name), "Frozen admission dependency replaced")
    for name in ("SCHEMA", "STATUS", "SCOPE"):
        _require(globals()[name] == getattr(original, name), "Frozen admission constant changed")
    return dict(
        result,
        original_source=str(old_path),
        original_sha256=ORIGINAL_SHA256,
        repair_source=str(new_path),
        execution_permissions_granted=False,
    )


def _facts(root):
    # Imports stay lazy: source verifies the actual dual-package deployment
    # before any evidence can be promoted. No GPU or sampling entrypoint runs.
    from sr_f6_closure import projection as t03

    context = source.require_support(root, purpose="admission")
    review = context["review"]
    support = context["support_binding"]
    _require(
        support["status"] == "VERIFIED_SUPPORT_PASS_REVIEW_PENDING"
        and support["source_rows"] == 9216
        and len(context["records"]) == len(context["raw_by_row"]) == 9216,
        "Complete independently verified confirmation required",
    )
    _require(
        review["status"] == "REVIEW_COMPLETE_NOT_GATE_SIGNED",
        "Complete actual independent review required",
    )
    numerical = t03.verify_completed(root)
    _require(
        numerical["scope"] == "SERVER_ACTUAL",
        "Only actual server numerical evidence can close confirmation",
    )
    _require(
        set(numerical["parents"]) == {"b0", "b1", "b2"},
        "All three registered numerical parents required",
    )
    _require(
        not _synthetic((support, review, numerical)),
        "Synthetic fixtures cannot produce actual confirmation admission",
    )
    # Revalidate after reading downstream facts. The producer's private warm
    # cache reuses unchanged facts and checks the complete evidence membership.
    after = source.verify_registered(root, require_support=True)
    _require(
        after["consumer"] == context["consumer"]
        and after["support_binding_ref"] == context["support_binding_ref"],
        "Source or support binding changed during admission",
    )
    return context, review, numerical


def assess(root):
    """Read actual evidence; never run a model, write a receipt, or authorize."""
    try:
        context, review, numerical = _facts(root)
    except (OSError, ValueError, KeyError, TypeError, ImportError, PermissionError) as exc:
        return dict(
            schema=SCHEMA,
            status="UNKNOWN",
            scope=SCOPE,
            training_allowed=False,
            generation_allowed=False,
            evaluation_allowed=False,
            gate_status="NOT_SIGNED",
            error=type(exc).__name__ + ": " + str(exc),
        )
    states = [numerical["status"]] + [numerical["parents"][p]["status"] for p in ("b0", "b1", "b2")]
    status = (
        STATUS
        if all(state == "PASS" for state in states)
        else "FAIL"
        if "FAIL" in states
        else "INCONCLUSIVE"
        if "INCONCLUSIVE" in states
        else "UNKNOWN"
    )
    return dict(
        schema=SCHEMA,
        status=status,
        scope=SCOPE,
        execution_origin="ACTUAL_EVIDENCE_VALIDATION",
        consumer=copy.deepcopy(context["consumer"]),
        producer=copy.deepcopy(context["support_binding"]["producer"]),
        support_binding=copy.deepcopy(context["support_binding_ref"]),
        review_sha256=digest(review),
        numerical_sha256=digest(numerical),
        checks=dict(
            source_data_processor_visual="VERIFIED_BY_FROZEN_PRODUCER",
            complete_confirmation_support="PASS",
            independent_response_review=review["status"],
            numerical={p: numerical["parents"][p]["status"] for p in ("b0", "b1", "b2")},
        ),
        training_allowed=False,
        generation_allowed=False,
        evaluation_allowed=False,
        gate_status="NOT_SIGNED",
        new_model_calls=0,
        optimizer_updates=0,
        limitations=[
            "Confirmation exposure and measurement are not demonstrated learning benefit",
            "Independent reviewer identities are recorded, not cryptographically authenticated",
            "A new pilot input overlay, producer and runtime admission remain required",
        ],
    )


def compose_assessment(context, review, numerical):
    states = [numerical["status"]] + [numerical["parents"][p]["status"] for p in ("b0", "b1", "b2")]
    status = (
        STATUS
        if all(state == "PASS" for state in states)
        else "FAIL"
        if "FAIL" in states
        else "INCONCLUSIVE"
        if "INCONCLUSIVE" in states
        else "UNKNOWN"
    )
    return dict(
        schema=SCHEMA,
        status=status,
        scope=SCOPE,
        execution_origin="ACTUAL_EVIDENCE_VALIDATION",
        consumer=copy.deepcopy(context["consumer"]),
        producer=copy.deepcopy(context["support_binding"]["producer"]),
        support_binding=copy.deepcopy(context["support_binding_ref"]),
        review_sha256=digest(review),
        numerical_sha256=digest(numerical),
        checks=dict(
            source_data_processor_visual="VERIFIED_BY_FROZEN_PRODUCER",
            complete_confirmation_support="PASS",
            independent_response_review=review["status"],
            numerical={p: numerical["parents"][p]["status"] for p in ("b0", "b1", "b2")},
        ),
        training_allowed=False,
        generation_allowed=False,
        evaluation_allowed=False,
        gate_status="NOT_SIGNED",
        new_model_calls=0,
        optimizer_updates=0,
        limitations=[
            "Confirmation exposure and measurement are not demonstrated learning benefit",
            "Independent reviewer identities are recorded, not cryptographically authenticated",
            "A new pilot input overlay, producer and runtime admission remain required",
        ],
    )


def evaluate(root):
    """Read actual evidence; never run a model, write a receipt, or authorize."""
    try:
        context, review, numerical = _facts(root)
    except (OSError, ValueError, KeyError, TypeError, ImportError, PermissionError) as exc:
        return dict(
            assessment=dict(
                schema=SCHEMA,
                status="UNKNOWN",
                scope=SCOPE,
                training_allowed=False,
                generation_allowed=False,
                evaluation_allowed=False,
                gate_status="NOT_SIGNED",
                error=type(exc).__name__ + ": " + str(exc),
            ),
            numerical=None,
        )
    return dict(assessment=compose_assessment(context, review, numerical), numerical=numerical)
