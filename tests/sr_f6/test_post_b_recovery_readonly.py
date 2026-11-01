"""Frozen-core equivalence and read-only boundaries for recovery delivery."""

import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import re
from pathlib import Path

import pytest
import torch

from sr_f6 import recovery
from sr_f6.transactions import RunTransactions


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[2]
consumer = load(
    "post_b_recovery_consumer_test", ROOT / "scripts/sr_f6_post_b_delivery/recovery_readonly.py"
)
fixture = load("post_b_recovery_physical_fixture", Path(__file__).with_name("test_recovery.py"))
prototype, factory, case = fixture.prototype, fixture.factory, fixture.case


@pytest.mark.parametrize(
    "source, old, new, alias",
    [
        ("pilot_v3_inputs", "build_overlay", "_build_overlay_readonly", "overlay"),
        ("pilot_v3_inputs", "verify_overlay", "_verify_overlay_readonly", "overlay"),
        (
            "pilot_v3_registration",
            "build_pilot_v3_registration",
            "_build_registration_readonly",
            "v3_registration",
        ),
        (
            "pilot_v3_registration",
            "verify_pilot_v3_registration",
            "_verify_registration_readonly",
            "v3_registration",
        ),
        ("recovery", "_registration", "_registration_readonly", "frozen"),
        ("recovery", "verify_recovery", "_verify_recovery_readonly", "frozen"),
        ("recovery", "_comparison_facts", "_comparison_facts_readonly", "frozen"),
    ],
)
def test_static_copies_preserve_all_original_assertions(source, old, new, alias):
    """Only provenance entrypoints, global qualification and cold-cache change."""
    original_tree = ast.parse((ROOT / "src/sr_f6" / (source + ".py")).read_text())
    original = next(
        n for n in original_tree.body if isinstance(n, ast.FunctionDef) and n.name == old
    )
    actual = ast.parse(inspect.getsource(getattr(consumer, new))).body[0]
    renames = {
        "_context_readonly": "_context",
        "_build_overlay_readonly": "build_overlay",
        "_verify_overlay_readonly": "verify_overlay",
        "_build_registration_readonly": "build_pilot_v3_registration",
        "_verify_registration_readonly": "verify_pilot_v3_registration",
        "_registration_readonly": "_registration",
        "_verify_recovery_readonly": "verify_recovery",
        "_comparison_facts_readonly": "_comparison_facts",
        "_confirmation_readonly": "require_confirmation_binding",
    }

    class Normalize(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if node.module == "historical_closure":
                return None
            return node

        def visit_Assign(self, node):
            if old == "verify_recovery":
                if any(isinstance(t, ast.Name) and t.id == "cache_key" for t in node.targets):
                    return None
                if any(isinstance(t, ast.Name) and t.id == "previous_stamps" for t in node.targets):
                    return ast.Assign(targets=node.targets, value=ast.Constant(None))
                if any(
                    isinstance(t, ast.Subscript)
                    and isinstance(t.value, ast.Name)
                    and t.value.id == "_PREFIX_CACHE"
                    for t in node.targets
                ):
                    return None
            return self.generic_visit(node)

        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == alias:
                return ast.Name(id=node.attr, ctx=ast.Load())
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == "pilot_v3_inputs"
                and node.attr in ("_context", "verify_overlay")
            ):
                return ast.Name(id=node.attr, ctx=ast.Load())
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == "pilot_contract"
                and node.attr == "verify_registration"
            ):
                return ast.Name(id="_registration_bundle_readonly", ctx=ast.Load())
            return self.generic_visit(node)

        def visit_Name(self, node):
            if node.id == alias and isinstance(node.ctx, ast.Load):
                return ast.Subscript(
                    ast.Attribute(ast.Name("sys", ast.Load()), "modules", ast.Load()),
                    ast.Name("__name__", ast.Load()),
                    ast.Load(),
                )
            return ast.Name(id=renames.get(node.id, node.id), ctx=node.ctx)

    expected = Normalize().visit(copy.deepcopy(original))
    normalized = Normalize().visit(copy.deepcopy(actual))
    expected.name = normalized.name = "same_function"
    # Qualification removal can expose a replaced nested pilot_v3_inputs entry.
    expected = Normalize().visit(expected)
    normalized = Normalize().visit(normalized)
    assert ast.dump(normalized, include_attributes=False) == ast.dump(
        expected, include_attributes=False
    )


def test_provider_never_adopts_cache_and_hashes_each_time(tmp_path, monkeypatch):
    path = tmp_path / "payload"
    path.write_bytes(b"first")
    monkeypatch.setattr(consumer.phases, "_path", lambda p, **_: Path(p))
    calls = []
    monkeypatch.setattr(
        consumer.phases,
        "file_hash",
        lambda p: calls.append(p) or hashlib.sha256(p.read_bytes()).hexdigest(),
    )
    hashes = consumer.ReadOnlyHashes(tmp_path)
    hashes.adopt_verified(path, "0" * 64, {"sha256": "0" * 64})
    assert hashes.hash(path) == hashlib.sha256(b"first").hexdigest()
    path.write_bytes(b"second")
    assert hashes.hash(path) == hashlib.sha256(b"second").hexdigest()
    assert calls == [path, path]
    assert not (tmp_path / ".verification-cache").exists()


def test_provider_rejects_file_changed_during_read(tmp_path, monkeypatch):
    path = tmp_path / "payload"
    path.write_bytes(b"first")
    monkeypatch.setattr(consumer.phases, "_path", lambda p, **_: Path(p))

    def mutate(p):
        p.write_bytes(b"longer changed payload")
        return "0" * 64

    monkeypatch.setattr(consumer.phases, "file_hash", mutate)
    with pytest.raises(PermissionError, match="changed while hashing"):
        consumer.ReadOnlyHashes(tmp_path).hash(path)


def snapshot(root):
    return {
        str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }


@pytest.mark.parametrize("mutation", [None, "raw", "update_hash", "environment", "rng", "tensor"])
def test_physical_torch_rng_core_matches_frozen_verifier(case, monkeypatch, mutation):
    pref = fixture.primary(case)
    imported = fixture.prepare(case, pref)
    mutate = None
    if mutation == "raw":

        def mutate(cached):
            cached["records"][0].update(full_text='{"time":"different model text"}')

    if mutation == "rng":
        original_rollout = fixture._t.rollout

        def changed_rollout(*args, **kwargs):
            if args[3] == 7:
                torch.rand(3)
            return original_rollout(*args, **kwargs)

        monkeypatch.setattr(fixture._t, "rollout", changed_rollout)
    if mutation == "tensor":
        original_update = fixture._t.tiny_update
        counts = []

        def changed_update(runtime, *args, **kwargs):
            value = original_update(runtime, *args, **kwargs)
            counts.append(1)
            if len(counts) == 3:
                with torch.no_grad():
                    next(runtime.model.parameters()).add_(0.125)
                value["policy_after_sha256"] = fixture.state_hash(
                    fixture.trainable_state(runtime.model)
                )
            return value

        monkeypatch.setattr(fixture._t, "tiny_update", changed_update)
    rref = fixture.recover(case, imported, mutate_step=7 if mutate else None, mutate=mutate)
    directory = Path(imported["directory"])
    if mutation == "update_hash":
        path = directory / "checkpoints/step-007/UPDATE.json"
        value = json.loads(path.read_bytes())
        value["training_seconds"] = 999  # Excluded execution metadata, equality must stay true.
        path.write_text(json.dumps(value))
        # UPDATE is bound into checkpoint marker, so malformed rehash is rejected by both.
    if mutation == "environment":
        path = directory / "runtime_identity/independent-recovery.json"
        value = json.loads(path.read_bytes())
        value["torch_version"] = "different-version"
        path.write_text(json.dumps(value))
    # Registration was genuinely constructed by the existing CPU fixture; the
    # differential isolates the copied core, not production provenance consumer.
    registered = recovery._registration(case.root, case.context, case.spec_hash)
    monkeypatch.setattr(consumer, "_registration_readonly", lambda *args: registered)
    kwargs = dict(
        root=case.root,
        phase_context=case.context,
        pilot_spec_sha256=case.spec_hash,
        binding_ref=imported["binding_ref"],
        primary_execution_ref=pref,
        recovery_execution_ref=rref,
    )
    with (
        RunTransactions(case.primary, case.identity) as ptx,
        RunTransactions(directory, case.identity) as rtx,
    ):
        try:
            expected = recovery._comparison_facts(ptx, rtx, **kwargs)
        except PermissionError as exc:
            with pytest.raises(PermissionError, match=re.escape(str(exc))):
                consumer._comparison_facts_readonly(ptx, rtx, **kwargs)
            return
        before = snapshot(case.root)
        old_cache = copy.deepcopy(recovery._PREFIX_CACHE)
        actual = consumer._comparison_facts_readonly(ptx, rtx, **kwargs)
        assert actual == expected
        assert snapshot(case.root) == before
        assert old_cache == recovery._PREFIX_CACHE
        if mutation == "raw":
            assert actual["status"] == "FAIL" and any(
                "original_rows/0/full_text" in p for p in actual["differences"]
            )
        elif mutation in ("rng", "tensor"):
            assert actual["status"] == "FAIL" and actual["differences"]
        else:
            assert actual["exact"]


def test_prefix_cold_hash_rejects_tamper_even_with_original_cache(case, monkeypatch):
    pref = fixture.primary(case)
    imported = fixture.prepare(case, pref)
    registered = recovery._registration(case.root, case.context, case.spec_hash)
    monkeypatch.setattr(consumer, "_registration_readonly", lambda *args: registered)
    target = Path(imported["directory"]) / "checkpoints/step-004/state.pt"
    target.write_bytes(b"tampered preserved original checkpoint")
    with pytest.raises(PermissionError, match="prefix content differs"):
        consumer._verify_recovery_readonly(
            case.root,
            phase_context=case.context,
            pilot_spec_sha256=case.spec_hash,
            binding_ref=imported["binding_ref"],
        )


@pytest.mark.parametrize(
    "left,right",
    [
        (1, True),
        (torch.tensor([1], dtype=torch.int64), torch.tensor([1], dtype=torch.int32)),
        (torch.tensor([1, 2]), torch.tensor([[1, 2]])),
        ({"rng": [1, 2]}, {"rng": [1, 3]}),
    ],
)
def test_frozen_typed_difference_semantics_preserved(left, right):
    assert consumer.frozen._differences(left, right, "evidence")


def test_no_calls_to_historical_reexecution_or_original_registration():
    tree = ast.parse(inspect.getsource(consumer))
    forbidden = {
        "verify_corrected_confirmation",
        "require_confirmation_binding",
        "verify_phase",
        "verify_recovery",
        "verify_comparison",
        "compare_recovery",
        "prepare_recovery",
        "freeze_confirmation_binding",
    }
    calls = {
        n.func.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    assert not (calls & forbidden)
    assert not any(
        isinstance(n, ast.Name) and n.id in {"_CACHE", "_PREFIX_CACHE"} for n in ast.walk(tree)
    )


historical_fixture = load(
    "post_b_historical_transport_fixture", Path(__file__).with_name("test_historical_closure.py")
)
historical_case = historical_fixture.case


def proof_fixture(historical_case, monkeypatch, mutation=None):
    """Actual-shaped local proof to reach transport checks, never admission."""
    h = consumer.historical
    root = historical_case["root"]
    request = historical_case["request"]
    context = dict(identity_sha256="c" * 64, scope="SYNTHETIC_TEST")
    monkeypatch.setattr(consumer, "_context_readonly", lambda *args: context)
    folder = root / h.DIRECTORY / request["nonce"]
    folder.mkdir(parents=True)
    envelope = copy.deepcopy(historical_case["envelope"])
    # As in the original historical unit tests, remove the fixture marker only
    # to exercise deeper assertions. No production context accepts this fixture.
    envelope["result"]["review"]["execution_origin"] = "ACTUAL_REVIEW"
    if mutation == "nonce":
        envelope["nonce"] = "b" * 32
    elif mutation == "request_digest":
        envelope["request_sha256"] = "0" * 64
    elif mutation == "duplicate_input":
        envelope["result"]["inputs"].append(copy.deepcopy(envelope["result"]["inputs"][0]))
    elif mutation == "input_hash":
        envelope["result"]["inputs"][0]["sha256"] = "0" * 64
    envelope["result_sha256"] = consumer.digest(envelope["result"])
    if mutation == "result_digest":
        envelope["result_sha256"] = "0" * 64
    execution = dict(
        command=["/fixture/python", "-I", "-B", "-c", h.BOOTSTRAP],
        cwd=str(h.FOLLOWUP_DEPLOYMENT),
        started_at="fixture-start",
        python="/fixture/python",
        bootstrap_sha256=hashlib.sha256(h.BOOTSTRAP.encode()).hexdigest(),
    )
    if mutation == "command":
        execution["command"][1] = "-S"

    def write(path, value):
        historical_fixture.write(path, value)
        return recovery._ref(root, path)

    transcript = [
        write(folder / "REQUEST.json", dict(schema=h.SCHEMA, request=request, execution=execution)),
        write(folder / "stdout.json", envelope),
    ]
    (folder / "stderr.log").write_bytes(b"")
    transcript.append(recovery._ref(root, folder / "stderr.log"))
    if mutation == "missing_stderr":
        transcript.pop()
    original = dict(
        schema=h.SCHEMA,
        status="CONFIRMATION_RECOMPUTED_NOT_AUTHORIZED",
        facts=envelope["result"],
        facts_sha256=consumer.digest(envelope["result"]),
        request=request,
        execution=dict(
            execution, completed_at="fixture-end", child_pid=envelope["child_pid"], returncode=0
        ),
        transcript=transcript,
        training_authorized=False,
        generation_authorized=False,
        formal_authorized=False,
    )
    if mutation == "pid":
        original["execution"]["child_pid"] += 1
    original_ref = write(folder / "RESULT.json", original)
    binding = h._binding(
        context,
        historical_case["registration_ref"],
        historical_case["receipt_ref"],
        dict(original, report_ref=original_ref),
    )
    reference = write(root / h.BINDING_PATH, binding)
    if mutation == "stderr_bytes":
        (folder / "stderr.log").write_bytes(b"changed preserved stderr")
    if mutation == "source_bytes":
        source = h.CLOSURE_DEPLOYMENT / "src/sr_f6_closure/source.py"
        source.write_bytes(source.read_bytes() + b"\n# tampered frozen source\n")
    return root, context, reference


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "nonce",
        "request_digest",
        "result_digest",
        "duplicate_input",
        "input_hash",
        "command",
        "missing_stderr",
        "pid",
        "stderr_bytes",
        "source_bytes",
    ],
)
def test_existing_proof_transport_and_inputs_are_rechecked(historical_case, monkeypatch, mutation):
    root, context, reference = proof_fixture(historical_case, monkeypatch, mutation)

    def forbidden(*args, **kwargs):
        pytest.fail("Read-only consumer must not start historical recomputation")

    monkeypatch.setattr(consumer.historical, "verify_corrected_confirmation", forbidden)
    monkeypatch.setattr(consumer.historical, "require_confirmation_binding", forbidden)
    monkeypatch.setattr(consumer.historical.subprocess, "Popen", forbidden)
    before = snapshot(root)
    if mutation:
        with pytest.raises(PermissionError):
            consumer._confirmation_readonly(
                root, phase_context=context, confirmation_binding_ref=reference
            )
    else:
        value = consumer._confirmation_readonly(
            root, phase_context=context, confirmation_binding_ref=reference
        )
        assert value["binding_ref"] == reference
        assert value["verification"]["facts"]["old_admission_status"] == "UNKNOWN_NOT_MODIFIED"
    assert snapshot(root) == before


@pytest.mark.parametrize(
    "mutation", [None, "unbound_failure", "earlier_different", "technical_unknown"]
)
def test_entire_comparison_inventory_is_retained(case, mutation):
    # Inventory-only fixture; numerical equivalence is tested above with real
    # tensors. No synthetic inventory is accepted as full delivery.
    folder = case.primary / "COMPARISONS"
    folder.mkdir()
    expected = dict(binding_ref={"path": "binding", "sha256": "a" * 64}, status="PASS")
    selected = folder / ("a" * 32 + ".json")
    selected.write_text(json.dumps(expected))
    compared = dict(
        expected,
        receipt_reference=recovery._ref(case.root, selected),
        verified_from_saved_facts=True,
    )
    if mutation == "earlier_different":
        (folder / ("b" * 32 + ".json")).write_text(json.dumps(dict(expected, status="FAIL")))
    if mutation in ("technical_unknown", "unbound_failure"):
        failure = dict(
            format=recovery.FORMAT,
            status="INVALID_OR_INCOMPLETE_EVIDENCE",
            binding_ref=expected["binding_ref"],
            gate_status="NOT_SIGNED",
            training_authorized=False,
        )
        if mutation == "unbound_failure":
            failure["binding_ref"] = {"path": "other", "sha256": "b" * 64}
        (folder / ("b" * 32 + "-FAILURE.json")).write_text(json.dumps(failure))
    before = snapshot(folder)
    if mutation in ("unbound_failure", "earlier_different"):
        with pytest.raises(PermissionError):
            consumer.pilot_audit._comparison_inventory(case.root, case.primary, compared)
    else:
        result = consumer.pilot_audit._comparison_inventory(case.root, case.primary, compared)
        assert result["status"] == (
            "HISTORICAL_TECHNICAL_UNKNOWN" if mutation else "ALL_SAVED_FACTS_VERIFIED"
        )
        assert len(result["receipts"]) == (2 if mutation else 1)
    assert snapshot(folder) == before


def test_public_consumer_requires_both_existing_transaction_locks(case):
    with pytest.raises(PermissionError, match="lock"):
        consumer.verify_saved_recovery(
            RunTransactions(case.primary, case.identity),
            RunTransactions(case.primary, case.identity),
            root=case.root,
            phase_context=case.context,
            pilot_spec_sha256=case.spec_hash,
            binding_ref={},
            comparison_ref={},
            recovery_execution_ref={},
        )


def test_cold_proof_read_rejects_transport_change_without_legacy_cache(tmp_path, monkeypatch):
    path = tmp_path / "proof.json"
    path.write_text('{"original":true}')
    reference = recovery._ref(tmp_path, path)
    old_read = recovery._read_bytes
    cache_before = copy.deepcopy(recovery._CONTENT_CACHE)

    def changed_read(target):
        payload = old_read(target)
        target.write_text('{"changed_during_read":true}')
        return payload

    monkeypatch.setattr(recovery, "_read_bytes", changed_read)
    with pytest.raises(PermissionError, match="Proof changed while reading"):
        consumer._proof_bound(tmp_path, reference)
    assert cache_before == recovery._CONTENT_CACHE


def test_frozen_helper_modules_are_actual_imports():
    from sr_f6 import (
        historical_closure,
        phases,
        pilot_audit,
        pilot_v3_inputs,
        pilot_v3_registration,
    )

    assert consumer.frozen is recovery
    assert consumer.historical is historical_closure
    assert consumer.phases is phases
    assert consumer.pilot_audit is pilot_audit
    assert consumer.overlay is pilot_v3_inputs
    assert consumer.v3_registration is pilot_v3_registration
