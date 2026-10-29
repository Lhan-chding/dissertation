"""Synthetic CPU boundary tests; no model loading or scientific authorization."""

import copy
from dataclasses import FrozenInstanceError

import pytest

from sr_f6 import pilot_contract, runtime


def test_contract_is_closed_and_does_not_grant_access():
    old = pilot_contract.for_phase("PILOT_V1")
    new = pilot_contract.for_phase("PILOT_V3")
    assert old.registration_directory == "PHASE_RUNS/PILOT_V1/PILOT_FROZEN"
    assert new.admission_scope == "PILOT_V3_REGISTERED_ONLY"
    assert old.admission_path != new.admission_path
    assert not hasattr(new, "training_authorized")
    with pytest.raises(FrozenInstanceError):
        new.phase = "PILOT_V1"
    for phase in (None, "FORMAL_V1", "PILOT_V2", {"phase": "PILOT_V3"}):
        with pytest.raises(PermissionError):
            pilot_contract.for_phase(phase)


def test_v3_input_pin_is_mandatory_before_any_loader_work(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Unpinned input must be rejected before registration or model loading")

    monkeypatch.setattr(pilot_contract, "verify_registration", forbidden)
    for pin in (None, "a" * 63, True):
        with pytest.raises(PermissionError, match="specification"):
            runtime._pilot_input_binding("/synthetic", {"phase": "PILOT_V3"}, pin)
    for phase in ("PILOT_V1", "FORMAL_V1", "CENSUS_S0"):
        assert runtime._pilot_input_binding("/synthetic", {"phase": phase}, None) == {}
        with pytest.raises(PermissionError, match="explicit PILOT_V3"):
            runtime._pilot_input_binding("/synthetic", {"phase": phase}, "a" * 64)


def test_loader_binding_uses_verified_overlay_and_owns_its_copy(monkeypatch):
    context = dict(phase="PILOT_V3", identity_sha256="b" * 64)
    source = dict(context, origin_identity_sha256="c" * 64)
    reference = dict(path="PHASE_RUNS/PILOT_V3/INPUTS_V3/OVERLAY_MANIFEST.json", sha256="d" * 64)
    response = dict(
        overlay=dict(manifest_ref=reference, manifest=dict(prompt_version="srf6-interface-v3"))
    )
    calls = []

    def verify(root, **kwargs):
        calls.append((root, kwargs))
        return response

    monkeypatch.setattr(pilot_contract, "verify_registration", verify)
    binding = runtime._pilot_input_binding("/synthetic", source, "a" * 64)
    assert calls == [("/synthetic", dict(phase_context=context, pilot_spec_sha256="a" * 64))]
    assert binding["input_overlay"] == reference
    assert binding["pilot_spec_sha256"] == "a" * 64
    response["overlay"]["manifest_ref"]["sha256"] = "e" * 64
    assert binding["input_overlay_sha256"] == binding["input_overlay"]["sha256"] == "d" * 64


def _bound_runtime():
    model = object.__new__(runtime.Runtime)
    source = dict(
        phase="PILOT_V3",
        identity_sha256="a" * 64,
        origin_identity_sha256="b" * 64,
        identity=dict(source_manifest=dict(sha256="c" * 64), data_manifest=dict(sha256="d" * 64)),
    )
    extra = dict(
        pilot_spec_sha256="e" * 64,
        input_overlay=dict(path="fixture/overlay.json", sha256="f" * 64),
        input_overlay_sha256="f" * 64,
        prompt_version="srf6-interface-v3",
    )
    model._srf6_source_context = copy.deepcopy(source)
    model.identity = dict(
        phase="PILOT_V3",
        source_sha256="c" * 64,
        source_hash="c" * 64,
        data_hash="d" * 64,
        origin_identity_sha256="b" * 64,
        phase_identity_sha256="a" * 64,
        **extra,
    )
    runtime._LOADED_RUNTIMES[model] = dict(source=source, execution=copy.deepcopy(extra))
    return model


@pytest.mark.parametrize("field", runtime.V3_INPUT_IDENTITY_FIELDS)
def test_changed_loaded_spec_or_overlay_cannot_enter_forward(field):
    model = _bound_runtime()
    assert runtime._check_loaded_source_identity(model) == "PILOT_V3"
    model.identity[field] = "substituted"
    with pytest.raises(PermissionError, match="overlay changed"):
        runtime._check_loaded_source_identity(model)


def test_identity_dictionary_cannot_replace_actual_v3_load():
    model = _bound_runtime()
    del runtime._LOADED_RUNTIMES[model]
    with pytest.raises(PermissionError, match="overlay changed"):
        runtime._check_loaded_source_identity(model)
