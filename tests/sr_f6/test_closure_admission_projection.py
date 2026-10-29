"""Synthetic control-flow only; no actual evidence or scientific gate is certified."""

import ast
import copy
from pathlib import Path
from types import CodeType, FunctionType

import pytest

from sr_f6_closure import admission_projection as repair
from sr_f6_followup import admission as original
from sr_f6_followup import source


@pytest.fixture
def facts(tmp_path, monkeypatch):
    from sr_f6_closure import projection
    from sr_f6_followup import t03

    root = tmp_path / "original-immutable-root"
    root.mkdir()
    original_bytes = b"Original UNKNOWN must remain unchanged.\n"
    (root / "UNKNOWN.txt").write_bytes(original_bytes)
    context = dict(
        root=root,
        consumer={"revision": "a" * 40, "deployment": "/synthetic/consumer"},
        support_binding=dict(
            status="VERIFIED_SUPPORT_PASS_REVIEW_PENDING",
            source_rows=9216,
            producer={"revision": "b" * 40},
        ),
        support_binding_ref={"path": str(root / "SUPPORT_BINDING.json"), "sha256": "c" * 64},
        records=[{}] * 9216,
        raw_by_row={str(i): {} for i in range(9216)},
        review={"status": "REVIEW_COMPLETE_NOT_GATE_SIGNED", "reviewed": 360},
    )
    numerical = dict(
        status="PASS",
        scope="SERVER_ACTUAL",
        parents={p: {"status": "PASS"} for p in ("b0", "b1", "b2")},
    )
    calls = []

    def require_support(actual, *, purpose):
        assert actual == root and purpose == "admission"
        calls.append("support")
        return copy.deepcopy(context)

    def verify_registered(actual, require_support=False):
        assert actual == root and require_support
        calls.append("recheck")
        return copy.deepcopy(context)

    def verify_completed(actual):
        assert actual == root
        calls.append("numerical")
        return copy.deepcopy(numerical)

    monkeypatch.setattr(source, "require_support", require_support)
    monkeypatch.setattr(source, "verify_registered", verify_registered)
    monkeypatch.setattr(projection, "verify_completed", verify_completed)
    monkeypatch.setattr(t03, "verify_completed", verify_completed)
    yield root, context, numerical, calls
    assert {p.name for p in root.iterdir()} == {"UNKNOWN.txt"}
    assert (root / "UNKNOWN.txt").read_bytes() == original_bytes


def assert_permissions_closed(value):
    assert value["gate_status"] == "NOT_SIGNED"
    assert all(
        value[name] is False
        for name in ("generation_allowed", "training_allowed", "evaluation_allowed")
    )


def test_exact_frozen_function_projection_and_new_identity():
    proof = repair.verify_projection()
    assert proof["changed_imports"] == 1
    assert proof["changed_predicates"] == 0
    assert proof["execution_permissions_granted"] is False
    assert proof["original_source"] != proof["repair_source"]
    assert Path(repair.assess.__code__.co_filename).resolve() == Path(repair.__file__).resolve()
    assert repair.assess is not original.assess
    assert not hasattr(repair, "seal")
    assert not hasattr(repair, "require_confirmation")


@pytest.mark.parametrize(
    ("overall", "parent_state", "expected"),
    [
        ("PASS", "PASS", "CONFIRMATION_GATES_PASS"),
        ("UNKNOWN", "PASS", "UNKNOWN"),
        ("INCOMPLETE", "PASS", "UNKNOWN"),
        ("TECHNICAL_INVALID", "INCOMPLETE", "UNKNOWN"),
        ("PASS", "INCOMPLETE", "UNKNOWN"),
        ("PASS", "UNKNOWN", "UNKNOWN"),
        ("PASS", "INCONCLUSIVE", "INCONCLUSIVE"),
        ("INCONCLUSIVE", "PASS", "INCONCLUSIVE"),
        ("PASS", "FAIL", "FAIL"),
        ("FAIL", "PASS", "FAIL"),
        ("INCONCLUSIVE", "FAIL", "FAIL"),
        ("FAIL", "INCONCLUSIVE", "FAIL"),
    ],
)
def test_all_classifications_and_original_inner_identity(facts, overall, parent_state, expected):
    root, context, numerical, calls = facts
    numerical["status"] = overall
    numerical["parents"]["b2"]["status"] = parent_state
    actual = repair.assess(root)
    assert calls == ["support", "numerical", "recheck"]
    assert actual == original.assess(root)
    assert actual["status"] == expected
    assert actual["schema"] == original.SCHEMA
    assert actual["consumer"] == context["consumer"]
    assert actual["new_model_calls"] == actual["optimizer_updates"] == 0
    assert_permissions_closed(actual)


@pytest.mark.parametrize(
    "error", [PermissionError, OSError, ValueError, KeyError, TypeError, ImportError]
)
def test_source_rejection_stays_unknown_before_numerical(facts, monkeypatch, error):
    root, _, _, calls = facts

    def rejected(*args, **kwargs):
        raise error("source rejected")

    monkeypatch.setattr(source, "require_support", rejected)
    actual = repair.assess(root)
    assert actual == original.assess(root)
    assert actual["status"] == "UNKNOWN"
    assert actual["error"].startswith(error.__name__ + ":")
    assert calls == []
    assert_permissions_closed(actual)


@pytest.mark.parametrize("field", ["records", "raw_by_row", "source_rows", "support", "review"])
def test_original_support_denominator_and_review_gates(facts, field):
    root, context, _, calls = facts
    if field == "records":
        context["records"].pop()
    elif field == "raw_by_row":
        context["raw_by_row"].pop("0")
    elif field == "source_rows":
        context["support_binding"]["source_rows"] = 9215
    elif field == "support":
        context["support_binding"]["status"] = "UNKNOWN"
    else:
        context["review"]["status"] = "REVIEW_INCOMPLETE"
    actual = repair.assess(root)
    assert actual == original.assess(root)
    assert actual["status"] == "UNKNOWN"
    assert "numerical" not in calls


@pytest.mark.parametrize("where", ["review", "support", "numerical"])
def test_synthetic_sources_never_promoted(facts, where):
    root, context, numerical, _ = facts
    item = {
        "review": context["review"],
        "support": context["support_binding"],
        "numerical": numerical,
    }[where]
    item["nested"] = [{"execution_origin": "SYNTHETIC_FIXTURE"}]
    actual = repair.assess(root)
    assert actual == original.assess(root)
    assert actual["status"] == "UNKNOWN"
    assert "Synthetic fixtures" in actual["error"]


@pytest.mark.parametrize("mutation", ["scope", "missing_parent", "extra_parent"])
def test_actual_three_parent_scope_required(facts, mutation):
    root, _, numerical, _ = facts
    if mutation == "scope":
        numerical["scope"] = "LOCAL_ONLY"
    elif mutation == "missing_parent":
        numerical["parents"].pop("b2")
    else:
        numerical["parents"]["b3"] = {"status": "PASS"}
    assert repair.assess(root) == original.assess(root)
    assert repair.assess(root)["status"] == "UNKNOWN"


@pytest.mark.parametrize("field", ["consumer", "support_binding_ref"])
def test_source_recheck_mutation_stays_unknown(facts, monkeypatch, field):
    root, context, _, _ = facts
    after = copy.deepcopy(context)
    after[field] = {"changed": True}
    monkeypatch.setattr(source, "verify_registered", lambda *a, **k: after)
    actual = repair.assess(root)
    assert actual == original.assess(root)
    assert actual["status"] == "UNKNOWN"
    assert "changed during" in actual["error"]


@pytest.mark.parametrize(
    "change",
    [
        lambda text: text.replace("== 9216", ">= 9216"),
        lambda text: text.replace("training_allowed=False", "training_allowed=True"),
        lambda text: text.replace(
            "from sr_f6_closure import projection as t03", "from sr_f6_followup import t03"
        ),
        lambda text: text + '\ndef assess(root):\n    return {"status": "PASS"}\n',
        lambda text: text.replace("def _facts(root):", "def missing(root):"),
        lambda text: text.replace("def assess(root):", "@staticmethod\ndef assess(root):"),
        lambda text: text.replace("def assess(root):", "async def assess(root):"),
    ],
)
def test_unapproved_predicate_declaration_or_dependency_change_rejected(change):
    before = Path(original.__file__).read_text()
    projected = Path(repair.__file__).read_text()
    with pytest.raises(PermissionError):
        repair._prove_sources(before, change(projected))


def test_original_numerical_import_shape_must_match():
    before = (
        Path(original.__file__)
        .read_text()
        .replace("from . import t03", "from . import source as t03")
    )
    with pytest.raises(PermissionError, match="import differs"):
        repair._prove_sources(before, Path(repair.__file__).read_text())


def test_changed_original_source_hash_rejected(tmp_path, monkeypatch):
    other = tmp_path / "admission.py"
    other.write_text(Path(original.__file__).read_text() + "\n# Changed source\n")
    monkeypatch.setattr(original, "__file__", str(other))
    with pytest.raises(PermissionError, match="source changed"):
        repair.verify_projection()


def test_original_path_argument_cannot_replace_loaded_module(tmp_path):
    other = tmp_path / "admission.py"
    other.write_text(Path(original.__file__).read_text())
    with pytest.raises(PermissionError, match="module path differs"):
        repair.verify_projection(other)


@pytest.mark.parametrize("dependency", ["source", "_require", "_synthetic", "copy", "digest"])
def test_frozen_dependencies_cannot_be_replaced(monkeypatch, dependency):
    # _require is itself the guard, so replace it with an equivalent independent
    # guard, not a bypass; correspondence still must reject the changed identity.
    def guard(condition, message):
        if not condition:
            raise PermissionError(message)

    monkeypatch.setattr(repair, dependency, guard if dependency == "_require" else object())
    with pytest.raises(PermissionError, match="dependency replaced"):
        repair.verify_projection()


@pytest.mark.parametrize("constant", ["SCHEMA", "STATUS", "SCOPE"])
def test_frozen_constants_cannot_be_replaced(monkeypatch, constant):
    monkeypatch.setattr(repair, constant, "OTHER")
    with pytest.raises(PermissionError, match="constant changed"):
        repair.verify_projection()


@pytest.mark.parametrize("state", ["PASS", "FAIL", "INCONCLUSIVE", "INCOMPLETE", "UNKNOWN"])
def test_evaluate_reads_once_and_keeps_complete_numerical_evidence(facts, state):
    root, _, numerical, calls = facts
    numerical["status"] = numerical["parents"]["b2"]["status"] = state
    numerical["errors"] = [{"detail": "retained unchanged"}]
    result = repair.evaluate(root)
    assert calls == ["support", "numerical", "recheck"]
    assert set(result) == {"assessment", "numerical"}
    assert result["numerical"] == numerical
    assert result["assessment"] == original.assess(root)
    assert_permissions_closed(result["assessment"])


@pytest.mark.parametrize(
    "error", [PermissionError, OSError, ValueError, KeyError, TypeError, ImportError]
)
def test_evaluate_original_unknown_on_incomplete_facts(facts, monkeypatch, error):
    root, _, _, calls = facts

    def rejected(*args, **kwargs):
        raise error("source rejected")

    monkeypatch.setattr(source, "require_support", rejected)
    result = repair.evaluate(root)
    assert result["numerical"] is None
    assert result["assessment"] == original.assess(root)
    assert result["assessment"]["status"] == "UNKNOWN"
    assert calls == []


@pytest.mark.parametrize("function", ["compose_assessment", "evaluate"])
def test_single_read_composition_change_rejected(function):
    before = Path(original.__file__).read_text()
    projected = Path(repair.__file__).read_text()
    prefix, suffix = projected.rsplit("def " + function + "(", 1)
    changed = (
        prefix
        + "def "
        + function
        + "("
        + suffix.replace("training_allowed=False", "training_allowed=True", 1)
    )
    with pytest.raises(PermissionError, match="single-read composition changed"):
        repair._prove_sources(before, changed)


def test_evaluate_must_not_add_a_second_fact_read():
    before = Path(original.__file__).read_text()
    projected = Path(repair.__file__).read_text()
    prefix, suffix = projected.rsplit("def evaluate(", 1)
    changed = (
        prefix
        + "def evaluate("
        + suffix.replace(
            "context, review, numerical = _facts(root)",
            "context, review, numerical = _facts(root)\n        _facts(root)",
            1,
        )
    )
    with pytest.raises(PermissionError, match="single-read composition changed"):
        repair._prove_sources(before, changed)


@pytest.mark.parametrize("name", repair.FUNCTIONS + repair.ADDITIONAL_FUNCTIONS)
def test_loaded_altered_body_with_same_filename_and_identity_rejected(monkeypatch, name):
    path = Path(repair.__file__).resolve()
    tree = ast.parse(path.read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    function.body = [ast.Return(value=ast.Constant(value=None))]
    compiled = compile(ast.fix_missing_locations(tree), str(path), "exec", dont_inherit=True)
    changed = next(
        code for code in compiled.co_consts if isinstance(code, CodeType) and code.co_name == name
    )
    replacement = FunctionType(changed, repair.__dict__, name)
    assert replacement.__module__ == repair.__name__
    assert replacement.__name__ == name
    assert replacement.__code__.co_filename == str(path)
    monkeypatch.setattr(repair, name, replacement)
    with pytest.raises(PermissionError, match="Loaded admission function differs"):
        repair.verify_projection()
