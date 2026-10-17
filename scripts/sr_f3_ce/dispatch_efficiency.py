"""Bound, instance-local identity checks for an already-frozen CPU dispatcher.

Startup performs the original full gates. Subsequent identity calls rehash the
small binding files, rather than replaying historical scientific validation.
Submission, release, completion, recovery, and delivery verification stay in the
original implementation. This module never installs a worker or module patch.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

_OWNER = "_srf3ce_dispatch_efficiency_owner"


class Efficiency:
    """Install a light identity check on exactly one authenticated controller."""

    def __init__(self, value, api, io):
        self.value = copy.deepcopy(value)
        self.api, self.io = api, io
        self.root = Path(value["root"])
        self.controller = api.controller
        self._started = False
        self._starting = False
        self._receipt = None
        self._entry = self.check

    def _paths(self):
        root = self.root.resolve(strict=True)
        code = (root / "code").resolve(strict=True)
        operation = root / "operations" / self.value["operation_commit"]
        identity = Path(self.value.get("identity_path", operation / "OPERATION_IDENTITY.json"))
        manifest = Path(self.value["extension_manifest_path"])
        source = Path(__file__)
        if (
            not root.is_dir()
            or not code.is_dir()
            or (
                Path(self.value["code_root"]).resolve(strict=True) != code
                or Path(self.controller.root).resolve(strict=True) != root
                or Path(self.controller.code).resolve(strict=True) != code
                or identity.resolve(strict=True) != operation / "OPERATION_IDENTITY.json"
                or source.resolve(strict=True) != operation / "dispatch_efficiency.py"
            )
        ):
            raise PermissionError("Efficiency root/code/operation path differs")
        required = {
            "run_identity": root / "RUN_IDENTITY.json",
            "config": root / "orchestration/CONFIG.json",
            "execution_freeze": root / "EXECUTION_FREEZE.json",
            "source_manifest": root / "deployment/SOURCE_MANIFEST.json",
            "extension_manifest": manifest,
            "amendment": manifest.parent / "AMENDMENT.json",
            "operation_identity": identity,
            "operation_sources": operation / "OPERATION_SOURCES.json",
            "efficiency_source": source,
            "scientific_config": root / "plan/config/SR_F3_CE.json",
            "typed_publication": root / "external/revisions/typed_v1/PUBLICATION.json",
            "typed_receipt": root / "external/revisions/typed_v1/TYPE_AUDIT_RECEIPT.json",
        }
        for label in ("submission_repair", "dispatch_efficiency"):
            binding = self.value.get(label, {})
            if not isinstance(binding, dict):
                raise PermissionError("Efficiency authorization binding is invalid: " + label)
            if "authorization_path" in binding or "authorization_sha256" in binding:
                if not isinstance(binding.get("authorization_path"), str) or not isinstance(
                    binding.get("authorization_sha256"), str
                ):
                    raise PermissionError(
                        "Efficiency authorization path/hash is incomplete: " + label
                    )
                required[label + "_authorization"] = Path(binding["authorization_path"])
        optional = {
            "package_manifest": root / "plan/PACKAGE_SHA256.json",
            "model_inputs": root / "plan/manifests/MODEL_INPUTS.jsonl",
            "protocol_selection": root / "PROTOCOL_SELECTION.json",
            "protocol_request": root / "authorization/REQUEST.md",
        }
        # Selected revisions must carry these inputs. For legacy roots, absence
        # is itself bound and a later addition invalidates this startup snapshot.
        revision = self.io.read(required["scientific_config"]).get("protocol_revision", "v1")
        if revision in {"v2", "c2"}:
            required.update(optional)
            optional = {}
        return root, code, required, optional

    def _binding(self, path, *, required):
        path = Path(path).absolute()
        try:
            canonical = path.resolve(strict=required)
        except (OSError, RuntimeError) as error:
            raise PermissionError("Efficiency binding path is unavailable: " + str(path)) from error
        if not canonical.is_relative_to(self._root):
            raise PermissionError("Efficiency binding escapes this run: " + str(path))
        exists = path.exists()
        if path.is_symlink() and not exists:
            raise PermissionError("Efficiency binding is a dangling symlink: " + str(path))
        if (required and not exists) or (exists and not path.is_file()):
            raise PermissionError("Efficiency binding is not a file: " + str(path))
        result = dict(path=str(path), canonical_path=str(canonical), exists=exists)
        if exists:
            result.update(sha256=self.io.sha(path), bytes=path.stat().st_size)
        return result

    def _check_paths(self):
        try:
            if self.root.resolve(strict=True) != self._root or (
                not self._root.is_dir()
                or (self._root / "code").resolve(strict=True) != self._code
                or not self._code.is_dir()
                or Path(self.value["code_root"]).resolve(strict=True) != self._code
                or Path(self.controller.root).resolve(strict=True) != self._root
                or Path(self.controller.code).resolve(strict=True) != self._code
                or self.api.controller is not self.controller
            ):
                raise PermissionError("Efficiency root/code/controller identity changed")
        except (OSError, RuntimeError) as error:
            raise PermissionError("Efficiency root/code path is unavailable") from error
        stop = self._root / "STOP"
        if stop.exists() or stop.is_symlink():
            raise PermissionError("SR-F3-CE STOP exists")

    def _check_bindings(self):
        self._check_paths()
        for label, expected in self._files.items():
            current = self._binding(expected["path"], required=expected["exists"])
            if current != expected:
                raise PermissionError("Efficiency bound file changed: " + label)

    def start(self):
        """Verify full gates once, then return the receipt for the caller to save.

        ``full_verifications`` counts this helper's explicit calls only; the
        original functions may perform additional nested validation. Calling
        start again on this same instance checks the bindings without repeating
        those gates. Another Efficiency instance cannot replace this owner.
        """
        if self._started:
            self.check()
            return copy.deepcopy(self._receipt)
        if self._starting or getattr(self.controller, _OWNER, None) is not None:
            raise PermissionError("Efficiency is already starting or installed on this controller")
        authorization = self.value.get("dispatch_efficiency")
        if not isinstance(authorization, dict) or not authorization:
            raise PermissionError("Registered dispatcher efficiency authorization is required")
        self._starting = True
        installed = False
        started_at, began = time.time(), time.perf_counter()
        try:
            self._root, self._code, required, optional = self._paths()
            self._check_paths()
            self._files = {
                label: self._binding(path, required=True) for label, path in required.items()
            }
            self._files.update(
                {label: self._binding(path, required=False) for label, path in optional.items()}
            )
            for label in ("submission_repair", "dispatch_efficiency"):
                bound = self._files.get(label + "_authorization")
                if (
                    bound is not None
                    and bound["sha256"] != self.value[label]["authorization_sha256"]
                ):
                    raise PermissionError("Efficiency authorization original differs: " + label)
            config = self.io.read(required["config"])
            if config.get("pool_size") != 1 or self.controller.pool_size != 1:
                raise PermissionError(
                    "Efficiency preserves the registered single-GPU configuration"
                )
            sources = self.io.read(required["operation_sources"])
            if (
                sources.get("files", {}).get("dispatch_efficiency.py")
                != (self._files["efficiency_source"]["sha256"])
            ):
                raise PermissionError("Efficiency source is not bound by OPERATION_SOURCES")
            operation = self.io.read(required["operation_identity"])
            if operation.get("dispatch_efficiency") != authorization or (
                operation.get("operation_sources_sha256")
                != self._files["operation_sources"]["sha256"]
            ):
                raise PermissionError("Efficiency operation authorization/source binding differs")
            for field, label in (
                ("run_identity_sha256", "run_identity"),
                ("config_sha256", "config"),
                ("source_manifest_sha256", "source_manifest"),
                ("extension_manifest_sha256", "extension_manifest"),
            ):
                if self.value.get(field) != self._files[label]["sha256"]:
                    raise PermissionError("Efficiency registered identity differs: " + field)
            self._original_identity = self.controller.identity
            calls = {}

            def verify(label, function):
                before = time.perf_counter()
                result = function()
                calls[label] = dict(calls=1, seconds=time.perf_counter() - before)
                return result

            original_identity = verify("controller_identity", self._original_identity)
            verify(
                "protocol_science", lambda: self.api.protocol.verify_run(self._root, science=True)
            )
            checked = verify(
                "extension_science",
                lambda: self.api.extension.verify_extension(
                    self._root, self.value["extension_manifest_path"], science=True
                ),
            )
            if original_identity != self._files["run_identity"]["sha256"] or (
                checked.get("run_identity_sha256") != original_identity
                or checked.get("manifest_sha256") != self._files["extension_manifest"]["sha256"]
                or checked.get("amendment_sha256") != self._files["amendment"]["sha256"]
                or checked.get("execution_freeze_sha256")
                != self._files["execution_freeze"]["sha256"]
            ):
                raise PermissionError("Full efficiency startup gates returned different identities")
            self._check_bindings()
            if getattr(self.controller, _OWNER, None) is not None or (
                self.controller.identity != self._original_identity
            ):
                raise PermissionError("Controller identity changed during efficiency startup")
            self._identity = original_identity
            self._receipt = dict(
                schema_version=1,
                status="ACTIVE",
                scope="CPU_CONTROLLER_INSTANCE_IDENTITY_ONLY",
                started_at=started_at,
                startup_seconds=time.perf_counter() - began,
                run_identity_sha256=original_identity,
                explicit_calls_only=True,
                full_verifications=calls,
                files=copy.deepcopy(self._files),
                unchanged_verification=["submit", "release", "verify_job", "resume", "delivery"],
            )
            setattr(self.controller, _OWNER, self)
            self.controller.identity = self._entry
            installed = True
            self._started = True
            self.check()
            return copy.deepcopy(self._receipt)
        except BaseException:
            if installed:
                self.controller.identity = self._original_identity
                if getattr(self.controller, _OWNER, None) is self:
                    delattr(self.controller, _OWNER)
            self._started = False
            self._receipt = None
            raise
        finally:
            self._starting = False

    def check(self):
        """Rehash the frozen bindings; never replay scientific validation here."""
        if not self._started:
            raise PermissionError("Efficiency check requires a completed startup")
        if getattr(self.controller, _OWNER, None) is not self or (
            self.controller.identity is not self._entry
        ):
            raise PermissionError("Efficiency controller owner or identity entry changed")
        self._check_bindings()
        return self._identity
