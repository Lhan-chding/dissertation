"""CPU boundary/structural tests; these are not actual DEV or scientific evidence."""

import ast
import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sr_f6 import access, evaluation, runtime

REPO = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = load("dev_adapter_test", REPO / "scripts/sr_f6_dev_eval/runtime_adapter.py")
a = load("dev_access_fixture", Path(__file__).with_name("test_access.py"))
e = load("dev_evaluation_fixture", Path(__file__).with_name("test_evaluation.py"))
case, evaluation_case = a.case, a.evaluation_case
prototype, fixture = e.prototype, e.fixture


@pytest.mark.parametrize(
    "module,old,new,alias",
    [
        ("evaluation", "_inputs", "_inputs", "evaluation"),
        ("evaluation", "run_endpoint", "_run_endpoint", "evaluation"),
        ("evaluation", "verify_evaluation", "_verify_evaluation", "evaluation"),
        ("runtime", "load", "_load", "native"),
        ("runtime", "load_committed_endpoint", "_load_committed_endpoint", "native"),
        ("access", "open_evaluation_access", "_open_evaluation_access", "access"),
        ("pilot_v3_inputs", "build_overlay", "_build_overlay_readonly", "overlay"),
        ("pilot_v3_inputs", "verify_overlay", "_verify_overlay_readonly", "overlay"),
    ],
)
def test_static_core_exact_except_explicit_authority_providers(module, old, new, alias):
    original = next(
        n
        for n in ast.parse((REPO / "src/sr_f6" / (module + ".py")).read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == old
    )
    actual = ast.parse(inspect.getsource(getattr(adapter, new))).body[0]
    names = {
        "_runtime_source": "verify_runtime_source",
        "_input_binding": "_pilot_input_binding",
        "_access_authorization": "_admission",
        "_load": "native.load",
        "_load_committed_endpoint": "native.load_committed_endpoint",
        "_open_evaluation_access": "open_evaluation_access",
        "_context_readonly": "_context",
        "_build_overlay_readonly": "build_overlay",
        "_verify_overlay_readonly": "verify_overlay",
    }

    class Normalize(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            node = self.generic_visit(node)
            node.name = "core" if node.name in (old, new) else node.name
            keep = [
                i
                for i, arg in enumerate(node.args.kwonlyargs)
                if arg.arg not in ("execution_source", "authorization_ref")
            ]
            node.args.kwonlyargs = [node.args.kwonlyargs[i] for i in keep]
            node.args.kw_defaults = [node.args.kw_defaults[i] for i in keep]
            return node

        def visit_Expr(self, node):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return None
            if isinstance(node.value, ast.Call) and any(
                isinstance(arg, ast.Constant)
                and arg.value == "This adapter authorizes DEV evaluation only"
                for arg in node.value.args
            ):
                return None
            return self.generic_visit(node)

        def visit_ImportFrom(self, node):
            module = (node.module or "").removeprefix("sr_f6.")
            if module in ("pilot_v3_inputs", "access", "evaluation") or (
                not module and any(x.name == "runtime" for x in node.names)
            ):
                return None
            if node.module and node.module.startswith("sr_f6."):
                node.module = node.module.removeprefix("sr_f6.")
                node.level = 1
            return node

        def visit_Subscript(self, node):
            if module == "pilot_v3_inputs" and ast.unparse(node) == "sys.modules[__name__]":
                return ast.Name(id="overlay", ctx=ast.Load())
            return self.generic_visit(node)

        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == alias:
                return ast.Name(id=node.attr, ctx=node.ctx)
            if (
                module == "access"
                and isinstance(node.value, ast.Name)
                and node.value.id == "evaluation"
                and node.attr == "EvaluationLedger"
            ):
                return ast.Name(id="EvaluationLedger", ctx=node.ctx)
            return self.generic_visit(node)

        def visit_Call(self, node):
            target = ast.unparse(node.func)
            node = self.generic_visit(node)
            if target in names:
                node.func = ast.parse(names[target], mode="eval").body
            node.keywords = [
                k for k in node.keywords if k.arg not in ("execution_source", "authorization_ref")
            ]
            return node

    assert ast.dump(Normalize().visit(actual), include_attributes=False) == ast.dump(
        Normalize().visit(original), include_attributes=False
    )


def test_no_dangerous_historical_entrypoint_or_original_cache_write():
    tree = ast.parse(inspect.getsource(adapter))
    forbidden = {
        "verify_phase",
        "_verify_phase",
        "verify_origin",
        "_verify_origin",
        "verify_registration",
        "require_admission",
        "verify_recovery",
        "verify_comparison",
        "_verify_run",
        "verify_runtime_pilot",
        "require_confirmation_binding",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            assert ast.unparse(node.func).split(".")[-1] not in forbidden
        if isinstance(node, ast.Assign):
            assert all("_CACHE" not in ast.unparse(t) for t in node.targets)
    assert adapter.native is runtime and adapter.access is access
    assert adapter.evaluation is evaluation


def test_bounded_bytes_reject_mutation_and_symlink(tmp_path):
    path = tmp_path / "item.json"
    path.write_text('{"ok":true}')
    ref = dict(path="item.json", sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert adapter._bound(tmp_path, ref) == {"ok": True}
    path.write_text('{"ok":false}')
    with pytest.raises(PermissionError, match="Pinned bytes"):
        adapter._bound(tmp_path, ref)
    link = tmp_path / "linked"
    link.symlink_to(path)
    with pytest.raises(PermissionError, match="Symlink"):
        adapter._bytes(link)


def test_public_entrypoints_no_injected_runtime():
    for name in ("run_endpoint", "aggregate", "prepare_context"):
        assert "runtime" not in inspect.signature(getattr(adapter, name)).parameters
    with pytest.raises(PermissionError, match="actual independent deployment"):
        adapter._operator(
            dict(revision="a" * 40, deployment=str(REPO.parent), binding_sha256="b" * 64)
        )


@pytest.mark.parametrize("mode", ["TRAINING", "REPLAY"])
def test_load_cannot_create_training_or_replay_runtime(mode):
    with pytest.raises(PermissionError, match="DEV evaluation only"):
        adapter._load(
            "/unused",
            "b0",
            "test",
            mode=mode,
            phase="PILOT_V3",
            execution_source={},
            authorization_ref={},
        )


def test_inputs_retain_original_qid_metadata_checks(fixture):
    root, spec, _ = fixture
    assert adapter._inputs(root, spec["evaluation"]) == evaluation._inputs(root, spec["evaluation"])
    private = root / spec["evaluation"]["inputs"]["private"]["path"]
    rows = [json.loads(row) for row in private.read_text().splitlines()]
    rows[0]["qid"] = rows[1]["qid"]
    private.write_text("".join(json.dumps(row) + "\n" for row in rows))
    spec["evaluation"]["inputs"]["private"] = evaluation._reference(root, private)
    with pytest.raises(PermissionError):
        adapter._inputs(root, spec["evaluation"])


def open_fixture(case, monkeypatch):
    # Isolate authority discovery, retaining the real Runtime, ledger and PilotSession.
    monkeypatch.setattr(
        adapter, "_access_authorization", lambda *a, **k: copy.deepcopy(case["grant"])
    )
    return adapter._open_evaluation_access(
        case["runtime"],
        experiment_root=case["experiment"],
        pilot_spec_sha256="9" * 64,
        admission_ref=case["admission_ref"],
        ledger=case["ledger"],
        execution_source={},
        authorization_ref={},
    )


def test_actual_session_preserves_seed_budget_and_freeze(evaluation_case, monkeypatch):
    case = evaluation_case
    r = case["runtime"]
    with case["ledger"] as ledger, open_fixture(case, monkeypatch) as session:
        assert type(session) is access.PilotSession
        assert access._ACTIVE[id(r)] is session
        item = ledger.schedule[0]
        safe = ledger.public[item["qid"]]["model_input"]
        ledger.begin(item)
        with (
            pytest.raises(PermissionError, match="frozen DEV_EVAL"),
            session.generation(
                safe,
                case["experiment"],
                [1, 2],
                protocol="structured_v4",
                access=access._RUNTIME_ACCESS,
            ),
        ):
            pass
        with session.generation(
            safe,
            case["experiment"],
            item["seeds"],
            protocol="structured_v4",
            access=access._RUNTIME_ACCESS,
        ):
            prepared = r.prepare_input(safe, case["experiment"])
            with pytest.raises(PermissionError, match="teacher forcing"):
                r._batch_forward(prepared, [[1]], purpose="training_policy", grad=True)
            r.reserve("completion_attempts", 2, group_seed=item["seeds"][0])
            for draw in range(2):
                r.reserve("generated_tokens", 1, group_seed=item["seeds"][0], row_index=draw)
        with (
            pytest.raises(PermissionError),
            session.generation(
                safe,
                case["experiment"],
                item["seeds"],
                protocol="structured_v4",
                access=access._RUNTIME_ACCESS,
            ),
        ):
            pass
        assert all(not p.requires_grad and p.grad is None for p in r.model.parameters())
    assert id(r) not in access._ACTIVE
    with pytest.raises(PermissionError, match="NOT_AUTHORIZED"):
        r.prepare_input(safe, case["experiment"])


@pytest.mark.parametrize(
    "field", ["policy_tensor_sha256", "phase_identity_sha256", "evaluation_run_id"]
)
def test_actual_ledger_rejects_identity_mutation(evaluation_case, monkeypatch, field):
    case = evaluation_case
    case["ledger"].identity[field] = "changed"
    with (
        case["ledger"],
        pytest.raises(PermissionError, match="loaded identity"),
        open_fixture(case, monkeypatch),
    ):
        pass


def test_fake_ledger_and_unlocked_ledger_rejected(evaluation_case, monkeypatch):
    case = evaluation_case
    with pytest.raises(PermissionError, match="lock"), open_fixture(case, monkeypatch):
        pass
    case["ledger"] = SimpleNamespace()
    with (
        pytest.raises(PermissionError, match="Actual held EvaluationLedger"),
        open_fixture(case, monkeypatch),
    ):
        pass


@pytest.fixture
def authorization_fixture(tmp_path, monkeypatch):
    """Real frozen DEV plan, synthetic source authority; no historical verification."""
    op = load("dev_ops_fixture", REPO / "scripts/sr_f6_dev_eval/operations.py")
    root = tmp_path / "root"
    ops = tmp_path / "ops"
    dep = tmp_path / "new-source"
    root.mkdir()
    ops.mkdir()
    (dep / op.PREFIX).mkdir(parents=True)
    (dep / op.DRIVER).write_text("# synthetic driver\n")
    monkeypatch.setattr(op, "ROOT", root)
    monkeypatch.setattr(op, "OPS", ops)
    monkeypatch.setattr(adapter, "ROOT", root)
    actual = dict(revision="1" * 40, deployment=str(dep), binding_sha256="2" * 64)
    source_root = (
        REPO / ".local/pilot_v3_closure_20261029/delivery/sr_f6_semantic_factorial_20261027"
    )
    if not source_root.exists():
        pytest.skip("Optional local fixed-receipt fixture unavailable")

    def save(relative, value):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(op.encode(value))
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    phase = json.loads((source_root / "PHASES/PILOT_V3/IDENTITY.json").read_bytes())
    spec = json.loads(
        (source_root / "PHASE_RUNS/PILOT_V3/PILOT_FROZEN/PILOT_SPEC_FROZEN.json").read_bytes()
    )
    # Overlay pin remains a real hash reference, not mocked success.
    overlay = json.loads((source_root / spec["input_overlay"]["path"]).read_bytes())
    overlay_ref = save(spec["input_overlay"]["path"], overlay)
    spec["input_overlay"]["sha256"] = overlay_ref["sha256"]
    source_ref = save(
        "SOURCE_BINDING.json", {"fixture": "SYNTHETIC_BOUND_SOURCE_NOT_MODEL_EVIDENCE"}
    )
    spec["source_binding"] = dict(path="SOURCE_BINDING.json", sha256=source_ref["sha256"])
    spec_ref = save("PHASE_RUNS/PILOT_V3/PILOT_FROZEN/PILOT_SPEC_FROZEN.json", spec)
    admission = dict(
        root=str(root),
        phase_identity_sha256=adapter.PHASE_SHA,
        pilot_spec_sha256=spec_ref["sha256"],
        input_overlay=spec["input_overlay"],
        status="EXISTING_RECEIPT_NOT_REASSESSED",
        training_authorized=True,
    )
    admission_ref = save("PHASES/PILOT_V3/ADMISSION.json", admission)
    rel = dict(path="PHASES/PILOT_V3/ADMISSION.json", sha256=admission_ref["sha256"])
    jobs_ref = save(
        "PHASE_RUNS/PILOT_V3/JOBS/PLAN.json",
        dict(
            pins=dict(
                phase="PILOT_V3",
                phase_identity_sha256=adapter.PHASE_SHA,
                pilot_spec_sha256=spec_ref["sha256"],
                admission_ref=rel,
            )
        ),
    )
    auth = copy.deepcopy(op.USER_AUTHORIZATION)
    auth.update(
        spec_ref=spec_ref,
        admission_ref=admission_ref,
        plan_ref=jobs_ref,
        phase_ref=dict(path=str(root / "PHASES/PILOT_V3/IDENTITY.json"), sha256=adapter.PHASE_SHA),
    )
    monkeypatch.setattr(op, "USER_AUTHORIZATION", auth)
    context = dict(
        phase="PILOT_V3",
        status="BOUND_NOT_AUTHORIZED",
        training_allowed=False,
        deployment=str(adapter.CODE),
        identity=phase,
        identity_sha256=adapter.PHASE_SHA,
    )
    monkeypatch.setattr(
        adapter, "_context_readonly", lambda root, supplied=None: copy.deepcopy(context)
    )
    monkeypatch.setattr(adapter, "_operator", lambda source: op)
    contract_path = ops / "CONTRACT.json"
    contract_path.write_bytes(op.encode(op.contract_value(actual)))
    authorization_ref = op.reference(contract_path)
    (ops / "SOURCE_BINDING.json").write_bytes(
        op.encode(dict(source=actual, contract_ref=authorization_ref))
    )
    return root, op, actual, authorization_ref


def test_user_authority_does_not_inherit_historical_training_permission(authorization_fixture):
    root, op, actual, ref = authorization_fixture
    result = adapter.prepare_context(root, execution_source=actual, authorization_ref=ref)
    assert result["status"] == "USER_AUTHORIZED_REGISTERED_DEV_EVAL"
    assert result["historical_receipts_scope"] == "EXISTING_PINNED_RECEIPTS_NOT_REASSESSED"
    assert result["generation_authorized"] is True
    assert result["full_delivery_verified"] is False
    assert result["delivery_audit_waived_by_user"] is True
    for key in ("training_authorized", "formal_authorized", "test_authorized"):
        assert result[key] is False
    assert len(result["plan"]["endpoints"]) == 12
    assert result["plan"]["responses"] == 2304
    assert op.read(op.OPS / "CONTRACT.json")["user_authorization"] == op.USER_AUTHORIZATION


@pytest.mark.parametrize(
    "key,value",
    [
        ("full_delivery_verified", True),
        ("formal_authorized", True),
        ("test_authorized", True),
        ("delivery_audit_waived_by_user", False),
        ("full_delivery_verified", 0),
    ],
)
def test_mutated_contract_is_not_user_authority(authorization_fixture, key, value):
    root, op, actual, ref = authorization_fixture
    path = Path(ref["path"])
    contract = op.read(path)
    contract[key] = value
    path.write_bytes(op.encode(contract))
    updated = op.reference(path)
    (op.OPS / "SOURCE_BINDING.json").write_bytes(
        op.encode(dict(source=actual, contract_ref=updated))
    )
    with pytest.raises(PermissionError):
        adapter.prepare_context(root, execution_source=actual, authorization_ref=updated)


@pytest.mark.parametrize("key", ["spec_ref", "admission_ref", "plan_ref"])
def test_fixed_original_receipt_bytes_cannot_change(authorization_fixture, key):
    root, op, actual, ref = authorization_fixture
    Path(op.USER_AUTHORIZATION[key]["path"]).write_text("{}")
    with pytest.raises(PermissionError, match="Pinned bytes"):
        adapter.prepare_context(root, execution_source=actual, authorization_ref=ref)


def test_model_source_binding_must_match_original_spec_pin(authorization_fixture):
    root, _, actual, ref = authorization_fixture
    (root / "SOURCE_BINDING.json").write_text("{}")
    with pytest.raises(PermissionError, match="Pinned bytes"):
        adapter.prepare_context(root, execution_source=actual, authorization_ref=ref)
