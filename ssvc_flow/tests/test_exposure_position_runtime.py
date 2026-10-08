"""CPU-only tests: the fixture bridge can never authorize CUDA experiment work."""

import copy
import json
import random
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from test_exposure_training_runtime import PLAN, TinyAdapter

from src.exposure_position.bridge import (
    PhysicalBudget,
    normalized_state,
    normalized_update,
    preregister_semantics,
    run_bridge,
    run_bridge_suite,
    validate_g0,
)
from src.exposure_position.runtime import SERRuntime
from src.exposure_position.training import (
    REQUIRED_BRIDGE_CHECKS,
    require_bridge,
    require_training_job,
)
from src.exposure_substitution.runtime import SERRuntime as LegacyRuntime
from src.exposure_substitution.schedule import ExplicitScheduleSampler
from src.optimizer_fork import state_hash
from src.verified_discovery_transfer.sft_loss import EncodedExample

torch.set_num_threads(1)


class MappedSampler:
    def __init__(self, sampler, arm):
        self.source, self.arm, self.schedule_id = copy.deepcopy(sampler), arm, "new-schedule"
        self.late_error = False

    @property
    def step(self):
        return self.source.step

    def peek(self):
        rows = self.source.peek()
        for row in rows:
            row.update(arm=self.arm, task_id="new-" + row["task_id"])
            if "donor_root" in row:
                row["donor_root"] = "new-" + row["donor_root"]
        if self.late_error and self.step == 192:
            rows[0]["task_id"] = rows[1]["task_id"]
        return rows

    def commit(self, update):
        self.source.commit(update)

    def state_dict(self):
        return {
            "schema": "new-cursor",
            "schedule_id": self.schedule_id,
            "arm": self.arm,
            "committed_step": self.step,
        }

    def load_state_dict(self, state):
        expected = self.state_dict()
        expected["committed_step"] = state["committed_step"]
        if expected != state:
            raise ValueError("different cursor identity")
        source = self.source.state_dict()
        source["committed_step"] = state["committed_step"]
        self.source.load_state_dict(source)


def make_runtime(arm="A3_LOCAL_C1_J3", *, new=True, parent="S96"):
    torch.manual_seed(79)
    adapter = TinyAdapter()
    old_arm = "B_FORWARD_C4" if arm.startswith("B") else "A_LOCAL_C1"
    old_sampler = ExplicitScheduleSampler(PLAN, old_arm)
    metadata, encoded = {}, {}
    cursor = copy.deepcopy(old_sampler)
    slot_by_id = {}
    for update in range(1, 257):
        slot_by_id.update({s["task_id"]: s for s in cursor.peek()})
        cursor.commit(update)
    for index, (source, slot) in enumerate(sorted(slot_by_id.items())):
        key = ("new-" if new else "") + source
        root = slot.get("donor_root", "root-" + source)
        metadata[key] = dict(
            task_id=key,
            role=slot["role"],
            weight=1,
            prompt={"system": "s", "user": f"u{index}"},
            target="[1,2,3,4]",
            root_id=("new-" if new else "") + root,
        )
        if new:
            metadata[key].update(source_task_id=source, source_root_id=root)
        encoded[key] = EncodedExample(key, (1, 2) + (3,) * (index % 3), (4, 5, 16))
    sampler = MappedSampler(old_sampler, arm) if new else old_sampler
    identity = dict(
        experiment_id="SER_J23_20261008" if new else "SER_J2_20261007",
        arm=arm,
        parent=parent,
        block=0,
        seed=108701,
        parent_checkpoint_sha256="a" * 64,
        loss="completion_sequence_mean_slots16_donor_isolated_44314",
        steps=256,
        technical_only=True,
        frozen_plan_hash=None if new else "old-plan",
        schedule_id=sampler.schedule_id,
        encoded_training_hash="new" if new else "old",
    )
    if new:
        identity.update(logical_arm_id=arm, phase_id="SER_J23_20261008", source_bound_hash="source")
    cls = SERRuntime if new else LegacyRuntime
    return cls(
        adapter,
        seed=108701,
        identity=identity,
        sampler=sampler,
        encoded=encoded,
        parameter_names=[n for n, p in adapter.model.named_parameters() if p.requires_grad],
        metadata=metadata,
    )


def pair(old, new):
    return make_runtime(old, new=False), make_runtime(new)


def inference_fixture(runtime, origin, final):
    runtime.resume(origin["path"])
    runtime.resume(final["path"])
    return runtime.step == 2


class SemanticsTests(unittest.TestCase):
    def test_complete_mapping_preserves_every_numerical_field(self):
        old, new = pair("A_LOCAL_C1", "A2_LOCAL_C1_J2")
        contract = preregister_semantics(old, new)
        self.assertEqual(len(contract["task_proofs"]), 278)
        self.assertTrue(
            all(r["token_type_ids_sha256"] == state_hash(None) for r in contract["task_proofs"])
        )
        self.assertEqual(
            state_hash(old.capture()), state_hash(normalized_state(new.capture(), contract))
        )
        self.assertEqual(normalized_update(old.update()), normalized_update(new.update(), contract))
        self.assertEqual(
            state_hash(old.capture()), state_hash(normalized_state(new.capture(), contract))
        )
        for mutate in (
            lambda s: s["parameters"]["lora_A.weight"].add_(0.1),
            lambda s: next(iter(s["optimizer"]["state"].values()))["exp_avg"].add_(0.1),
            lambda s: s["rng"]["torch"].fill_(0),
            lambda s: s["buffers"].update(unexpected=torch.tensor([1])),
            lambda s: s["sampler"].update(committed_step=2),
            lambda s: s["exposures"].update({next(iter(s["exposures"])): 2}),
            lambda s: s["position_state"].update(unexpected=1),
        ):
            changed = copy.deepcopy(new.capture())
            mutate(changed)
            self.assertNotEqual(
                state_hash(old.capture()), state_hash(normalized_state(changed, contract))
            )

    def test_prompt_token_loss_and_late_slot_differences_cannot_be_hidden(self):
        for change in ("target", "token", "token_types", "loss", "late_slot"):
            old, new = pair("A_LOCAL_C1", "A2_LOCAL_C1_J2")
            key = next(iter(new.metadata))
            if change == "target":
                new.metadata[key]["target"] = "[4,3,2,1]"
            elif change == "token":
                new.encoded[key] = EncodedExample(key, (9,), (4, 5, 16))
            elif change == "token_types":
                value = new.encoded[key]
                new.encoded[key] = EncodedExample(
                    key, value.prompt_ids, value.target_ids, (0,) * len(value.prompt_ids)
                )
            elif change == "loss":
                new.identity["loss"] = "different loss"
            else:
                new.sampler.late_error = True
            with self.assertRaises(ValueError):
                preregister_semantics(old, new)

    def test_complete_student_restore_including_rng(self):
        runtime = make_runtime()
        runtime.update()
        saved = runtime.capture()
        expected = runtime.update()
        endpoint = runtime.capture()
        expected_rng = random.random(), float(np.random.rand()), float(torch.rand(()))
        runtime.restore(saved)
        self.assertEqual(runtime.update(), expected)
        self.assertEqual(state_hash(runtime.capture()), state_hash(endpoint))
        self.assertEqual(
            (random.random(), float(np.random.rand()), float(torch.rand(()))), expected_rng
        )
        bad = copy.deepcopy(saved)
        bad["optimizer"]["state"] = {}
        with self.assertRaises(ValueError):
            runtime.restore(bad)

    def test_parent_loader_reads_only_control_files(self):
        from src.exposure_position.runtime import load_parent_backend
        from src.exposure_substitution.schema import file_digest
        from src.modeling_v3.io import canonical_hash

        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            runtime_path, protocol_path = run / "runtime.json", run / "protocol.json"
            runtime_path.write_text("{}")
            protocol = {"historical": True}
            protocol_path.write_text(json.dumps(protocol))
            machine = {
                "runtime_path": str(runtime_path),
                "frozen_protocol_path": str(protocol_path),
                "frozen_protocol_hash": canonical_hash(protocol),
            }
            frozen = {
                "later_exclusion_status": "VERIFIED_EXPLICIT_SCAN",
                "external_inputs": {
                    key: {"status": "VERIFIED_BYTES", "sha256": file_digest(machine[key])}
                    for key in ("runtime_path", "frozen_protocol_path")
                },
            }
            parent = {"checkpoint_id": "S96", "status": "AVAILABLE"}
            (run / "FROZEN_PLAN.json").write_text(json.dumps(frozen))
            registered = {"legacy_frozen_plan_sha256": file_digest(run / "FROZEN_PLAN.json")}
            reads = []

            def read_control(root, relative):
                reads.append(relative)
                if relative == "machine.json":
                    return machine
                if relative == "parent_availability.json":
                    return {"checkpoints": [parent]}
                raise AssertionError("Worker opened a target/confirmation file")

            with (
                patch("src.exposure_position.runtime.source_binding", return_value={}),
                patch("src.exposure_position.runtime.legacy_run_directory", return_value=run),
                patch("src.exposure_position.schema.read_training_bound", return_value=registered),
                patch("src.exposure_substitution.schema._frozen_header", return_value=frozen),
                patch("src.exposure_substitution.schema.read_bound", side_effect=read_control),
                patch(
                    "src.exposure_substitution.schema.verify_frozen",
                    side_effect=AssertionError("full archive read"),
                ),
                patch(
                    "src.protocol_state_probes.inference.FrozenBackend", return_value="backend"
                ) as backend,
            ):
                self.assertEqual(load_parent_backend(run, "S96", allow_gpu=True), "backend")
                backend.assert_called_once_with(str(runtime_path), parent, protocol, allow_gpu=True)
            self.assertEqual(reads, ["machine.json", "parent_availability.json"])


@contextmanager
def g0_fixture():
    from src.exposure_position.schema import EXPERIMENT_ID, file_digest

    with tempfile.TemporaryDirectory() as temporary:
        run = Path(temporary) / "new"
        legacy = Path(temporary) / "legacy"
        run.mkdir()
        legacy.mkdir()
        source = {"source_hash": "source", "status": "SOURCE_BOUND"}
        code = {"fixture.py": "code"}
        (legacy / "FROZEN_PLAN.json").write_text('{"historical":true}')
        (run / "SOURCE_BOUND.json").write_text(json.dumps(source))
        (run / "SER_J23_DESIGN.json").write_text('{"test_design":true}')
        (run / "DESIGN_SOURCE.md").write_text("exact test prose")
        design_sha = file_digest(run / "SER_J23_DESIGN.json")
        prose_sha = file_digest(run / "DESIGN_SOURCE.md")
        frozen_sha = file_digest(legacy / "FROZEN_PLAN.json")
        fixed = (
            "reports/METRICS_BY_CELL.csv",
            "manifests/new_root_registry_AUDIT_ONLY.jsonl",
            "protocol.json",
            "machine.json",
        )
        records = []
        checks = {}
        registered = {}
        for relative in fixed:
            path = legacy / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixed historical " + relative)
            sha = file_digest(path)
            records.append(dict(relative_path=relative, path=str(path), sha256=sha))
            checks["run_v1/" + relative] = dict(status="PASS", path=str(path), sha256=sha)
            registered["run_v1/" + relative] = sha
        parents = {
            name: dict(
                checkpoint_full_cpu_state_verified=True,
                checkpoint=dict(
                    path=str(legacy / (name + ".pt")), sha256="a" * 64, state_hash="b" * 64
                ),
            )
            for name in ("S96", "REP96")
        }
        lookup = {
            "PARENT." + name: dict(
                status="VERIFIED_FULL_CPU_STATE",
                path=parent["checkpoint"]["path"],
                checkpoint_sha256=parent["checkpoint"]["sha256"],
            )
            for name, parent in parents.items()
        }
        files = {
            "machine.json": dict(legacy_run=str(legacy), legacy_frozen_plan_sha256=frozen_sha),
            "PARENT_AND_ENDPOINT_BINDINGS.json": dict(
                phase_id=EXPERIMENT_ID,
                status="REUSE_CANDIDATE",
                legacy_run=str(legacy),
                parents=parents,
                checkpoint_lookup=lookup,
                frozen_plan=dict(sha256=frozen_sha),
                full_A2_B2_checkpoints_verified=60,
            ),
            "REUSE_AUDIT.json": dict(
                phase_id=EXPERIMENT_ID,
                status="REUSE_CANDIDATE",
                scientific_invariants_verified=True,
                verify_weights=True,
                model_calls=0,
                optimizer_updates=0,
                legacy_run=str(legacy),
                design=dict(sha256=design_sha),
                checks=checks,
                full_A2_B2_checkpoints_verified=60,
                expected_A2_B2_checkpoints=60,
            ),
            "SOURCE_PROVENANCE.json": dict(
                phase_id=EXPERIMENT_ID,
                legacy_run=str(legacy),
                input_design_sha256=design_sha,
                input_prose_sha256=prose_sha,
                source_bound_hash=source["source_hash"],
                source_files=records,
            ),
            "CPU_CHECK.json": dict(
                phase_id=EXPERIMENT_ID,
                status="PASS",
                source_bound_hash="source",
                code_sha256=code,
                model_calls=0,
            ),
            "VALIDATION_RECEIPT.json": dict(
                schema="ser-j23-integrated-cpu-validation-v1",
                phase_id=EXPERIMENT_ID,
                status="PASS",
                code_sha256=code,
                source_bound_hash="source",
                failures=0,
                errors=0,
                tests=14,
                passed=14,
                real_model_calls=0,
                new_confirmation_generated=False,
            ),
        }

        def save():
            binding_path = run / "PARENT_AND_ENDPOINT_BINDINGS.json"
            binding_path.write_text(json.dumps(files[binding_path.name]))
            files["REUSE_AUDIT.json"]["bindings_file"] = dict(sha256=file_digest(binding_path))
            for name, content in files.items():
                (run / name).write_text(json.dumps(content))

        save()
        with (
            patch("src.exposure_position.schema.verify_source_bound", return_value=source),
            patch("src.exposure_position.control.code_bindings", return_value=code),
            patch(
                "src.exposure_position.control.validate_design",
                return_value={"legacy_input_sha256": registered},
            ),
            patch("src.exposure_position.control.DESIGN_SHA256", design_sha),
            patch("src.exposure_position.control.PLAN_SHA256", prose_sha),
        ):
            yield run, files, save


class G0Tests(unittest.TestCase):
    def test_complete_g0_accepts_verified_reuse_and_rerun_candidates(self):
        with g0_fixture() as (run, files, save):
            self.assertEqual(validate_g0(run)["candidate_status"], "REUSE_CANDIDATE")
            for name in ("REUSE_AUDIT.json", "PARENT_AND_ENDPOINT_BINDINGS.json"):
                files[name]["status"] = "RERUN_CANDIDATE"
                files[name]["full_A2_B2_checkpoints_verified"] = 0
            save()
            self.assertEqual(validate_g0(run)["candidate_status"], "RERUN_CANDIDATE")

    def test_standalone_bridge_rejects_incomplete_or_cross_run_g0_before_model_load(self):
        cases = (
            ("REUSE_AUDIT.json", lambda v: v.update(verify_weights=False)),
            ("REUSE_AUDIT.json", lambda v: v.update(scientific_invariants_verified=False)),
            ("REUSE_AUDIT.json", lambda v: v.update(status="WEIGHTS_NOT_VERIFIED")),
            ("REUSE_AUDIT.json", lambda v: v.update(legacy_run="/another/legacy")),
            ("REUSE_AUDIT.json", lambda v: v["design"].update(sha256="wrong")),
            ("PARENT_AND_ENDPOINT_BINDINGS.json", lambda v: v.update(legacy_run="/another/legacy")),
            (
                "PARENT_AND_ENDPOINT_BINDINGS.json",
                lambda v: v["frozen_plan"].update(sha256="wrong"),
            ),
            (
                "PARENT_AND_ENDPOINT_BINDINGS.json",
                lambda v: v["parents"]["S96"].update(checkpoint_full_cpu_state_verified=False),
            ),
            ("SOURCE_PROVENANCE.json", lambda v: v["source_files"].pop()),
            ("SOURCE_PROVENANCE.json", lambda v: v["source_files"][0].update(sha256="wrong")),
            ("CPU_CHECK.json", lambda v: v.update(status="FAIL")),
            ("CPU_CHECK.json", lambda v: v.update(source_bound_hash="wrong")),
            ("CPU_CHECK.json", lambda v: v.update(code_sha256={})),
            ("VALIDATION_RECEIPT.json", lambda v: v.update(code_sha256={})),
            ("VALIDATION_RECEIPT.json", lambda v: v.update(failures=1)),
            ("VALIDATION_RECEIPT.json", lambda v: v.update(source_bound_hash="another-source")),
            ("VALIDATION_RECEIPT.json", lambda v: v.update(tests=0)),
            ("VALIDATION_RECEIPT.json", lambda v: v.update(schema="unregistered")),
        )
        for filename, mutate in cases:
            with (
                self.subTest(filename=filename),
                g0_fixture() as (run, files, save),
                patch("src.exposure_position.bridge.load_parent_backend") as load,
            ):
                mutate(files[filename])
                save()
                with self.assertRaises(ValueError):
                    run_bridge(run, allow_gpu=True)
                load.assert_not_called()
                self.assertFalse((run / "bridge").exists())

    def test_missing_integrated_validation_receipt_cannot_load_model(self):
        with (
            g0_fixture() as (run, _, _),
            patch("src.exposure_position.bridge.load_parent_backend") as load,
        ):
            (run / "VALIDATION_RECEIPT.json").unlink()
            with self.assertRaises(FileNotFoundError):
                run_bridge(run, allow_gpu=True)
            load.assert_not_called()

    def test_preflight_model_load_and_encoding_failures_leave_unique_zero_update_attempt(self):
        from types import SimpleNamespace

        for failure in ("parent_load", "encoding"):
            with (
                self.subTest(failure=failure),
                g0_fixture() as (run, files, _),
                patch(
                    "src.exposure_position.runtime.legacy_run_directory",
                    return_value=Path(files["machine.json"]["legacy_run"]),
                ),
                patch("src.exposure_position.bridge.load_parent_backend") as load,
                patch("src.exposure_position.schema.training_rows", return_value={}),
                patch("src.exposure_position.training.encode_rows") as encode,
            ):
                if failure == "parent_load":
                    load.side_effect = RuntimeError("parent load failed")
                else:
                    load.return_value = SimpleNamespace(receipt={"fixture": True})
                    encode.side_effect = RuntimeError("encoding failed")
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    run_bridge(run, allow_gpu=True)
                output = run / "bridge"
                result = json.loads((output / "BRIDGE_RESULT.json").read_text())
                preflight = json.loads((output / "preflight/PREFLIGHT_RESULT.json").read_text())
                self.assertEqual(result["status"], "BLOCKED_TECHNICAL")
                self.assertEqual(result["failed_phase"], "preflight")
                self.assertEqual(result["physical_updates_started"], 0)
                self.assertEqual(result["physical_updates_completed"], 0)
                self.assertEqual(preflight["status"], "PREFLIGHT_FAILED")
                self.assertTrue((output / "PREREGISTRATION.json").is_file())
                if failure == "encoding":
                    self.assertTrue((output / "preflight/PARENT_LOAD.json").is_file())
                with self.assertRaisesRegex(ValueError, "already started"):
                    run_bridge(run, allow_gpu=True)
                load.assert_called_once()


class BridgeTests(unittest.TestCase):
    def test_exact_32_attempts_and_complete_checkpoint_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "bridge"
            result = run_bridge_suite(
                output,
                pair,
                lambda p, a: make_runtime(a, parent=p),
                source_bound_hash="source",
                matched_targets_equal=True,
                inference_checker=inference_fixture,
            )
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["execution_kind"], "CPU_TEST_FIXTURE")
            self.assertTrue(result["legacy_compatibility_pass"])
            self.assertEqual(result["physical_updates_started"], 32)
            self.assertEqual(result["physical_updates_completed"], 32)
            self.assertEqual(result["processed_target_sequences"], 512)
            self.assertEqual(len(list(output.glob("attempts/*.json"))), 32)
            self.assertEqual(len(list(output.glob("*/*.pt"))), 24)
            with self.assertRaises(FileExistsError):
                run_bridge_suite(
                    output, pair, None, source_bound_hash="source", matched_targets_equal=True
                )

    def test_failed_attempt_consumes_budget_without_retry(self):
        def fail(old, new):
            legacy, current = pair(old, new)
            legacy.update = lambda: (_ for _ in ()).throw(RuntimeError("interrupted"))
            return legacy, current

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "bridge"
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                run_bridge_suite(
                    output, fail, None, source_bound_hash="source", matched_targets_equal=True
                )
            result = json.loads((output / "BRIDGE_RESULT.json").read_text())
            self.assertEqual(result["physical_updates_started"], 1)
            self.assertEqual(result["physical_updates_completed"], 0)
            self.assertEqual(result["status"], "BLOCKED_TECHNICAL")
            self.assertEqual(
                json.loads((output / "attempts/physical01.json").read_text())["status"],
                "UPDATE_STARTED",
            )

    def test_exhausted_budget_fails_before_update(self):
        with tempfile.TemporaryDirectory() as temporary:
            budget = PhysicalBudget(temporary)
            budget.started = 32
            with self.assertRaisesRegex(ValueError, "exhausted"):
                budget.update(None, "case", "branch")

    def test_legacy_mismatch_permits_only_rerun_status(self):
        def altered(old, new):
            previous, current = pair(old, new)
            with torch.no_grad():
                current.adapter.model.lora_A.weight.add_(0.01)
            return previous, current

        with tempfile.TemporaryDirectory() as temporary:
            result = run_bridge_suite(
                Path(temporary) / "bridge",
                altered,
                lambda p, a: make_runtime(a, parent=p),
                source_bound_hash="source",
                matched_targets_equal=True,
                inference_checker=inference_fixture,
            )
            self.assertEqual(result["status"], "PASS_NEW_RUNTIME_REUSE_INCOMPATIBLE")
            self.assertFalse(result["legacy_compatibility_pass"])
            self.assertEqual(result["physical_updates_started"], 32)

    def test_new_runtime_resume_failure_cannot_pass_for_rerun(self):
        from src.optimizer_fork import restore_state

        def missing_weights(model, optimizer, state, **kwargs):
            before = {name: p.detach().clone() for name, p in model.named_parameters()}
            restore_state(model, optimizer, state, **kwargs)
            if state["step"] == 1:
                with torch.no_grad():
                    for name, parameter in model.named_parameters():
                        parameter.copy_(before[name])

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("src.optimizer_fork.restore_state", side_effect=missing_weights),
        ):
            result = run_bridge_suite(
                Path(temporary) / "bridge",
                pair,
                lambda p, a: make_runtime(a, parent=p),
                source_bound_hash="source",
                matched_targets_equal=True,
                inference_checker=inference_fixture,
            )
            self.assertEqual(result["status"], "BLOCKED_TECHNICAL")
            self.assertFalse(result["checks"]["new_arm_resume_complete_state_equal"])
            self.assertEqual(result["physical_updates_started"], 32)

    def test_guard_rejects_cpu_incomplete_and_source_drift(self):
        from src.modeling_v3.io import canonical_hash

        source = {"fixture": True}
        good = dict(
            status="PASS",
            execution_kind="REAL_CUDA_BRIDGE",
            technical_only=True,
            source_bound_hash=canonical_hash(source),
            maximum_physical_updates=32,
            physical_updates_started=32,
            physical_updates_completed=32,
            generations=0,
            E_CONFIRM2_accessed=False,
            code_sha256={"fixture.py": "hash"},
            checks={k: True for k in REQUIRED_BRIDGE_CHECKS},
        )
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch(
                "src.exposure_position.schema.verify_frozen",
                return_value={"code_sha256": {"fixture.py": "hash"}},
            ),
            patch("src.exposure_position.control.assert_code_bindings"),
            patch("src.exposure_position.schema.verify_source_bound", return_value=source),
            patch("src.exposure_position.schema.read_bound") as read,
        ):
            for change in (
                {"execution_kind": "CPU_TEST_FIXTURE"},
                {"physical_updates_started": 33},
                {"physical_updates_completed": 31},
                {"source_bound_hash": "wrong"},
                {"checks": {}},
                {"E_CONFIRM2_accessed": True},
            ):
                read.side_effect = lambda run, path, value={**good, **change}: (
                    source if path == "SOURCE_BOUND.json" else value
                )
                with self.assertRaises(ValueError):
                    require_bridge(temporary)
            read.side_effect = lambda run, path: source if path == "SOURCE_BOUND.json" else good
            self.assertEqual(require_bridge(temporary), good)
            (Path(temporary) / "STOP").touch()
            with self.assertRaisesRegex(RuntimeError, "STOP"):
                require_bridge(temporary)

    def test_mode_requires_full_matrix_and_excludes_reused_arms(self):
        jobs = [
            dict(parent=p, block=b, block_seed=108701 + b, updates=256, logical_arm_id=a)
            for p in ("S96", "REP96")
            for b in range(3)
            for a in ("A3_LOCAL_C1_J3", "B3_FORWARD_C4_J3")
        ]
        with (
            patch(
                "src.exposure_position.training.require_bridge",
                return_value={"legacy_compatibility_pass": True},
            ),
            patch(
                "src.exposure_position.schema.read_bound",
                side_effect=lambda run, path: (
                    {"mode": "REUSE_12"} if path == "EXECUTION_MODE.json" else jobs
                ),
            ),
        ):
            self.assertEqual(require_training_job("/tmp", "S96", 0, "A3_LOCAL_C1_J3"), jobs[0])
            with self.assertRaises(PermissionError):
                require_training_job("/tmp", "S96", 0, "A2_LOCAL_C1_J2")
            jobs.pop()
            with self.assertRaisesRegex(ValueError, "complete selected mode"):
                require_training_job("/tmp", "S96", 0, "A3_LOCAL_C1_J3")


if __name__ == "__main__":
    unittest.main()
