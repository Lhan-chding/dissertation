"""Synthetic control-flow tests; no actual review, replay or science certificate."""

import copy
import json

import pytest

from sr_f6_followup import admission, source


@pytest.fixture
def facts(tmp_path, monkeypatch):
    from sr_f6_followup import t03

    root = tmp_path / "followup"
    root.mkdir()
    monkeypatch.setattr(source, "PROJECT", tmp_path)
    monkeypatch.setattr(source, "ROOT", root)
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
    numeric = dict(
        status="PASS",
        scope="SERVER_ACTUAL",
        parents={p: {"status": "PASS"} for p in ("b0", "b1", "b2")},
    )
    calls = []

    def support(_root, *, purpose):
        assert _root == root and purpose == "admission"
        calls.append("source-review")
        return copy.deepcopy(context)

    def verify(_root, require_support=False):
        assert _root == root
        calls.append("source-recheck" if require_support else "source")
        return copy.deepcopy(context)

    def numerical(_root):
        assert _root == root
        calls.append("numerical")
        return copy.deepcopy(numeric)

    monkeypatch.setattr(source, "require_support", support)
    monkeypatch.setattr(source, "verify_registered", verify)
    monkeypatch.setattr(t03, "verify_completed", numerical)
    return root, context, numeric, calls


def test_fixed_receipt_recomputes_and_never_grants_model_access(facts):
    root, _, _, calls = facts
    sealed = admission.seal(root)
    assert sealed["status"] == admission.STATUS
    checked = admission.require_confirmation(root)
    assert checked["scope"] == admission.SCOPE
    assert all(
        checked[key] is False
        for key in ("training_allowed", "generation_allowed", "evaluation_allowed")
    )
    assert calls.count("numerical") == 2
    assert calls.count("source-review") == 2


def test_missing_review_blocks_before_numerical(facts):
    root, context, _, calls = facts
    context["review"]["status"] = "REVIEW_INCOMPLETE"
    assert admission.assess(root)["status"] == "UNKNOWN"
    assert "numerical" not in calls


@pytest.mark.parametrize("field", ["records", "raw_by_row"])
def test_full_denominator_required_before_numerical(facts, field):
    root, context, _, calls = facts
    context[field].pop() if field == "records" else context[field].pop("0")
    assert admission.assess(root)["status"] == "UNKNOWN"
    assert "numerical" not in calls


@pytest.mark.parametrize("state", ["FAIL", "INCONCLUSIVE", "INCOMPLETE"])
def test_adverse_numerical_states_preserved(facts, state):
    root, _, numeric, _ = facts
    numeric["status"] = numeric["parents"]["b2"]["status"] = state
    result = admission.seal(root)
    assert result["status"] == ("UNKNOWN" if state == "INCOMPLETE" else state)
    assert result["gate_status"] == "NOT_SIGNED"
    with pytest.raises(PermissionError):
        admission.require_confirmation(root)
    with pytest.raises(PermissionError, match="Preserve"):
        admission.seal(root)


def test_does_not_trust_overall_pass_or_omitted_parent(facts):
    root, _, numeric, _ = facts
    numeric["parents"]["b2"]["status"] = "FAIL"
    assert admission.assess(root)["status"] == "FAIL"
    numeric["parents"].pop("b2")
    assert admission.assess(root)["status"] == "UNKNOWN"


@pytest.mark.parametrize("where", ["review", "numerical"])
def test_synthetic_evidence_never_seals_pass(facts, where):
    root, context, numeric, _ = facts
    value = context["review"] if where == "review" else numeric
    value["execution_origin"] = "SYNTHETIC_TEST"
    assert admission.seal(root)["status"] == "UNKNOWN"


def test_changed_review_after_seal_invalidates_receipt(facts):
    root, context, _, _ = facts
    admission.seal(root)
    context["review"]["reviewed"] = 359
    with pytest.raises(PermissionError, match="facts changed"):
        admission.require_confirmation(root)


def test_forged_training_permission_is_rejected(facts):
    root, _, _, _ = facts
    admission.seal(root)
    path = root / "ADMISSION/RECEIPT.json"
    receipt = json.loads(path.read_text())
    receipt["training_allowed"] = True
    path.write_text(json.dumps(receipt))
    with pytest.raises(PermissionError, match="receipt required"):
        admission.require_confirmation(root)


def test_unknown_attempt_or_symlink_is_not_silently_ignored(facts):
    root, _, _, _ = facts
    admission.seal(root)
    extra = root / "ADMISSION/FAILURE.json"
    extra.write_text('{"status":"UNKNOWN"}')
    with pytest.raises(PermissionError, match="execution evidence"):
        admission.require_confirmation(root)
    extra.unlink()
    receipt = root / "ADMISSION/RECEIPT.json"
    replacement = root / "receipt-copy.json"
    receipt.rename(replacement)
    receipt.symlink_to(replacement)
    with pytest.raises(PermissionError, match="Symlink"):
        admission.require_confirmation(root)


def test_source_changed_between_verification_steps_blocks(facts, monkeypatch):
    root, _, _, _ = facts
    original = source.verify_registered

    def changed(*args, **kwargs):
        context = original(*args, **kwargs)
        context["consumer"]["revision"] = "f" * 40
        return context

    monkeypatch.setattr(source, "verify_registered", changed)
    result = admission.assess(root)
    assert result["status"] == "UNKNOWN"
    assert "changed during" in result["error"]
