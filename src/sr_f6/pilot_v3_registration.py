"""Reconstruct and freeze the complete nine-path PILOT_V3 plan.

The fixed scientific plan uses the original DATA, parents, schedules and random
streams. Only its phase/source namespace and public v3 input references differ.
A historical confirmation binding is independently required; this registration
never grants execution or relabels any historical license.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sys
import uuid

from . import pilot_v3_inputs, registration
from .transactions import _fsync_dir, _plain_path, _read_bytes, _require, _sha

PHASE = "PILOT_V3"
DIRECTORY = "PHASE_RUNS/PILOT_V3/PILOT_FROZEN"
SPEC_FILE = registration.SPEC_FILE
SCHEMA = "srf6-pilot-v3-registration-v1"


def build_pilot_v3_registration(
    root,
    *,
    phase_context,
    parent_references,
    overlay_ref,
    confirmation_binding_ref,
):
    """Build new-source artifacts from original inputs, without publishing."""
    from .historical_closure import require_confirmation_binding

    root = _plain_path(root)
    context = pilot_v3_inputs._context(root, phase_context)
    pilot_v3_inputs._source_members(root, context, (sys.modules[__name__],))
    overlay = pilot_v3_inputs.verify_overlay(root, phase_context=context, overlay_ref=overlay_ref)
    confirmation = require_confirmation_binding(
        root,
        phase_context=context,
        confirmation_binding_ref=confirmation_binding_ref,
    )
    _require(
        confirmation.get("binding_ref") == confirmation_binding_ref,
        "Historical confirmation verifier returned another binding",
    )
    _, config, _, data_ref, order_ref, order, parents, binding_ref = registration._inputs(
        root,
        context,
        parent_references,
    )
    bundle = registration._assemble_registration(
        root,
        context,
        parent_references,
        config,
        data_ref,
        order_ref,
        order,
        parents,
        binding_ref,
        shared_inputs=overlay["inputs_by_split"]["DEV_PILOT"],
        evaluation_inputs=overlay["inputs_by_split"]["DEV_EVAL"],
        phase=PHASE,
        directory=DIRECTORY,
        schema=SCHEMA,
        run_namespace=PHASE,
        identity_bindings={
            "input_overlay": overlay["manifest_ref"],
            "input_overlay_sha256": overlay["manifest_ref"]["sha256"],
            "prompt_version": pilot_v3_inputs.PROMPT_VERSION,
        },
        spec_bindings={
            "input_overlay": overlay["manifest_ref"],
            "prompt_version": pilot_v3_inputs.PROMPT_VERSION,
            "confirmation_binding_ref": confirmation_binding_ref,
        },
    )
    return {
        **bundle,
        "gate_status": "NOT_SIGNED",
        "overlay": overlay,
        "confirmation_binding": confirmation,
    }


def _result(bundle, status):
    return {
        "status": status,
        **{
            key: bundle[key]
            for key in (
                "directory",
                "spec_sha256",
                "spec",
                "gate_status",
                "training_authorized",
                "generation_authorized",
                "overlay",
                "confirmation_binding",
            )
        },
    }


def verify_pilot_v3_registration(root, *, phase_context, expected_spec_sha256):
    """Rebuild every artifact from verified v3 source, original DATA and parents."""
    root = _plain_path(root)
    _require(_sha(expected_spec_sha256), "Pinned PILOT_V3 specification required")
    destination = _plain_path(root / DIRECTORY)
    payload = _read_bytes(destination / SPEC_FILE)
    _require(
        hashlib.sha256(payload).hexdigest() == expected_spec_sha256,
        "PILOT_V3 specification pin differs",
    )
    spec = json.loads(payload)
    _require(
        spec.get("schema") == SCHEMA and spec.get("phase") == PHASE,
        "Explicit PILOT_V3 specification required",
    )
    bundle = build_pilot_v3_registration(
        root,
        phase_context=phase_context,
        parent_references=spec.get("parent_references"),
        overlay_ref=spec.get("input_overlay"),
        confirmation_binding_ref=spec.get("confirmation_binding_ref"),
    )
    _require(
        bundle["spec_sha256"] == expected_spec_sha256,
        "PILOT_V3 specification differs from fixed reconstruction",
    )
    registration._check_files(destination, bundle)
    return _result(bundle, "VERIFIED_REGISTRATION_NOT_AUTHORIZED")


def freeze_pilot_v3_registration(
    root,
    *,
    phase_context,
    parent_references,
    overlay_ref,
    confirmation_binding_ref,
):
    """Atomically publish all nineteen new-source files, never replace a tree."""
    root = _plain_path(root)
    bundle = build_pilot_v3_registration(
        root,
        phase_context=phase_context,
        parent_references=parent_references,
        overlay_ref=overlay_ref,
        confirmation_binding_ref=confirmation_binding_ref,
    )
    destination = _plain_path(root / DIRECTORY)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(
        _plain_path(destination.parent / ".pilot-v3-registration.lock"),
        os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
        0o600,
    )
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if destination.exists():
            registration._check_files(destination, bundle)
        else:
            staging = destination.with_name(f".pilot-v3-registration-{uuid.uuid4().hex}")
            staging.mkdir()
            for relative, value in bundle["files"].items():
                path = registration._path(staging, relative)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as stream:
                    stream.write(registration._payload(value))
                    stream.flush()
                    os.fsync(stream.fileno())
            for directory, _, _ in os.walk(staging, topdown=False):
                _fsync_dir(directory)
            _require(
                not destination.exists() and not destination.is_symlink(),
                "Never replace existing PILOT_V3 registration",
            )
            os.rename(staging, destination)
            _fsync_dir(destination.parent)
        registration._check_files(destination, bundle)
    finally:
        os.close(fd)
    return _result(bundle, "FROZEN_NOT_AUTHORIZED")
