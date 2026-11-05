"""CPU-only rejection tests for explicit GPU execution sealing, not model proof."""
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from adapter.backend import object_sha, file_sha
from adapter.seal_smoke import Evidence
from gpu_production.seal_path import (committed_preupdate, verify_checkpoint_bridge,
                                     verify_execution_bridge, verify_gpu_update_evidence)


class GpuProductionSealTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.package = Path(self.temp.name).resolve()
        self.stage = self.package / 'run/paths/b0-11031-R'
        self.execution = self.stage / 'executions/test'
        self.execution.mkdir(parents=True)
        self.original = {'science': 'unchanged'}
        self.identity = dict(arm='R', train_seed=11031, run_id='SR-ORD-TRAIN-1/b0-11031-R')
        self.worker = dict(self.original, path_identity=self.identity)
        self.evidence = Evidence()

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def bridge(self, worker, actual):
        return verify_execution_bridge(self.package, self.stage, self.execution, worker, actual,
                                       self.original, self.identity, self.evidence)

    def gpu_fixture(self):
        worker = dict(self.worker, gpu_production={'explicit': True})
        self.write(self.execution / 'GPU_EXECUTION_BINDING.json', worker)
        actual = dict(binding_sha256=object_sha(worker), hardware=dict(scope='REGISTERED_GPU_PRODUCTION_EXECUTION', cuda_visible_device_count=2,
            peer_access={'0_to_1': True, '1_to_0': True}, devices=[
                dict(cuda_uuid='a', name='NVIDIA RTX PRO 6000'),
                dict(cuda_uuid='b', name='NVIDIA RTX PRO 6000')]))
        module = types.ModuleType('gpu_production.binding')
        module.verify_production_binding = Mock(return_value=self.original)
        return worker, actual, module

    def test_legacy_identity_preserved_and_hidden_bridge_rejected(self):
        actual = dict(binding_sha256=object_sha(self.worker))
        self.assertEqual(self.bridge(self.worker, actual), self.package / 'adapter/backend.py')
        self.write(self.execution / 'GPU_RECOVERY_LINEAGE.json', {'forged': True})
        with self.assertRaisesRegex(PermissionError, 'hidden GPU bridge'):
            self.bridge(self.worker, actual)

    def test_original_worker_binding_cannot_change(self):
        wrong = dict(self.worker, science='different')
        with self.assertRaisesRegex(PermissionError, 'Worker binding changed'):
            self.bridge(wrong, dict(binding_sha256=object_sha(wrong)))

    def test_gpu_binding_hardware_and_source_are_required(self):
        worker, actual, module = self.gpu_fixture()
        with patch.dict(sys.modules, {'gpu_production.binding': module}):
            self.assertEqual(self.bridge(worker, actual), self.package / 'gpu_production/backend.py')
            module.verify_production_binding.assert_called_with(worker, self.execution)
            actual['hardware']['peer_access']['1_to_0'] = False
            with self.assertRaisesRegex(PermissionError, 'peer identity'):
                self.bridge(worker, actual)
            actual['hardware']['peer_access']['1_to_0'] = True
            module.verify_production_binding.return_value = {'science': 'changed'}
            with self.assertRaisesRegex(PermissionError, 'scientific binding'):
                self.bridge(worker, actual)

    def test_gpu_immutable_execution_binding_cannot_differ(self):
        worker, actual, module = self.gpu_fixture()
        self.write(self.execution / 'GPU_EXECUTION_BINDING.json', dict(worker, forged=True))
        with patch.dict(sys.modules, {'gpu_production.binding': module}):
            with self.assertRaisesRegex(PermissionError, 'execution binding differs'):
                self.bridge(worker, actual)

    def test_rng_count_requires_matching_provenance_at_step_zero(self):
        marker = dict(step=0, sampler_state={})
        self.assertIsNone(verify_checkpoint_bridge(self.stage, marker, {'cuda': ['gpu0']}))
        with self.assertRaisesRegex(PermissionError, 'lacks explicit provenance'):
            verify_checkpoint_bridge(self.stage, marker, {'cuda': ['gpu0', 'gpu1']})
        marker['sampler_state']['gpu_execution_provenance'] = {'forged': True}
        with self.assertRaisesRegex(PermissionError, 'Legacy one-GPU'):
            verify_checkpoint_bridge(self.stage, marker, {'cuda': ['gpu0']})
        with self.assertRaisesRegex(PermissionError, 'cardinality'):
            verify_checkpoint_bridge(self.stage, marker, {'cuda': []})

    def test_checkpoint_and_update_must_bind_identical_execution(self):
        provenance = {'step': 5, 'execution_binding': {'path': str(self.execution / 'GPU_EXECUTION_BINDING.json')}}
        marker = dict(step=5, sampler_state={'gpu_execution_provenance': provenance})
        update_path = self.stage / 'checkpoints/step-005/UPDATE.json'
        self.write(update_path, {'gpu_execution_provenance': provenance})
        module = types.ModuleType('gpu_production.state')
        module.verify_checkpoint_provenance = Mock()
        with patch.dict(sys.modules, {'gpu_production.state': module}):
            self.assertEqual(verify_checkpoint_bridge(self.stage, marker, {'cuda': [1, 2]}), provenance)
            module.verify_checkpoint_provenance.assert_called_with(self.package, marker)
            self.write(update_path, {'gpu_execution_provenance': dict(provenance, step=6)})
            with self.assertRaisesRegex(PermissionError, 'provenance differ'):
                verify_checkpoint_bridge(self.stage, marker, {'cuda': [1, 2]})
            module.verify_checkpoint_provenance.side_effect = PermissionError('forged binding digest')
            with self.assertRaisesRegex(PermissionError, 'forged binding digest'):
                verify_checkpoint_bridge(self.stage, marker, {'cuda': [1, 2]})

    def test_legacy_update_cannot_hide_gpu_metadata(self):
        self.write(self.stage / 'checkpoints/step-001/UPDATE.json', {'gpu_execution_provenance': {}})
        with self.assertRaisesRegex(PermissionError, 'Legacy update'):
            verify_checkpoint_bridge(self.stage, dict(step=1, sampler_state={}), {'cuda': [1]})

    def test_preupdate_uses_committed_execution_and_keeps_failed_others(self):
        other = self.execution.with_name('earlier-failed')
        passes = {5: [({'audit': 'earlier'}, other), ({'audit': 'committed'}, self.execution)]}
        loaded = {self.execution: {'worker_binding': {'gpu_production': {}}}}
        update = {'gpu_execution_provenance': {'execution_binding': {'path': str(self.execution / 'GPU_EXECUTION_BINDING.json')}}}
        before = copy.deepcopy(passes)
        self.assertEqual(committed_preupdate(5, update, passes, loaded), {'audit': 'committed'})
        self.assertEqual(passes, before)
        passes[5].append(({'audit': 'second-in-same-execution'}, self.execution))
        with self.assertRaisesRegex(PermissionError, 'ambiguous committed GPU'):
            committed_preupdate(5, update, passes, loaded)

    def test_legacy_ambiguity_or_gpu_update_without_provenance_blocks(self):
        loaded = {self.execution: {'worker_binding': {}}}
        passes = {5: [({'audit': 1}, self.execution), ({'audit': 2}, self.execution)]}
        with self.assertRaisesRegex(PermissionError, 'ambiguous pre-Adam'):
            committed_preupdate(5, {}, passes, loaded)
        passes[5] = passes[5][:1]
        loaded[self.execution]['worker_binding']['gpu_production'] = {}
        with self.assertRaisesRegex(PermissionError, 'missing version provenance'):
            committed_preupdate(5, {}, passes, loaded)

    def update_fixture(self, step=3):
        rows = [dict(row_id=f'r{i}') for i in range(128)]
        stats = dict(copied_bytes=1024, unpacked_copies=1, cpu_offload=False,
                     version_failures=0, copy_failures=0, unpack_failures=0)
        check = dict(status='GPU_PRE_ADAM_PASS', step=step, gradient_forwards=128,
            gpu1_rng_unchanged=True, cpu_activation_offload_bytes=0, copied_bytes=128*1024,
            first_recovery_comparison='NOT_THIS_REGISTERED_B0R_STEP5',
            original_long_gradient_equivalence='UNVERIFIED_ORIGINAL_BASELINE_OOM')
        counts = dict(status='UPDATED_UNCOMMITTED', step=step, new_generations=0,
            pre_adam_calls=1, adam_permitted=True, optimizer_returned_calls=1,
            scheduler_entered_calls=1, scheduler_returned_calls=1,
            forwards=[dict(row, forward_returned=True, stats=copy.deepcopy(stats)) for row in rows])
        update = dict(gpu_execution_provenance={'execution_binding':
                         {'path': str(self.execution / 'GPU_EXECUTION_BINDING.json')}})
        for key, prefix, value in [('gpu_pre_adam_check', 'GPU_PRE_ADAM_CHECK', check),
                                   ('gpu_update_accounting', 'GPU_UPDATE_ACCOUNTING', counts)]:
            path = self.execution / f'{prefix}_step-{step:03d}.json'
            self.write(path, value)
            update[key] = dict(path=str(path), sha256=file_sha(path))
        return rows, check, counts, update

    def test_committed_gpu_evidence_checks_hashes_order_and_actual_counts(self):
        rows, check, counts, update = self.update_fixture()
        verify_gpu_update_evidence(self.stage, self.identity, 3, update, rows, self.evidence)
        path = Path(update['gpu_update_accounting']['path'])
        counts['optimizer_returned_calls'] = 0
        self.write(path, counts)
        with self.assertRaisesRegex(PermissionError, 'bytes differ'):
            verify_gpu_update_evidence(self.stage, self.identity, 3, update, rows, self.evidence)
        update['gpu_update_accounting']['sha256'] = file_sha(path)
        with self.assertRaisesRegex(PermissionError, 'accounting differs'):
            verify_gpu_update_evidence(self.stage, self.identity, 3, update, rows, self.evidence)
        counts['optimizer_returned_calls'] = 1
        counts['forwards'].reverse()
        self.write(path, counts); update['gpu_update_accounting']['sha256'] = file_sha(path)
        with self.assertRaisesRegex(PermissionError, 'response order differs'):
            verify_gpu_update_evidence(self.stage, self.identity, 3, update, rows, self.evidence)

    def test_gpu_evidence_rejects_cpu_storage_and_escaped_refs(self):
        rows, check, counts, update = self.update_fixture()
        path = Path(update['gpu_update_accounting']['path'])
        counts['forwards'][0]['stats']['cpu_offload'] = True
        self.write(path, counts); update['gpu_update_accounting']['sha256'] = file_sha(path)
        with self.assertRaisesRegex(PermissionError, 'backward evidence invalid'):
            verify_gpu_update_evidence(self.stage, self.identity, 3, update, rows, self.evidence)
        update['gpu_update_accounting']['path'] = str(path.parent.parent / path.name)
        with self.assertRaisesRegex(PermissionError, 'evidence path differs'):
            verify_gpu_update_evidence(self.stage, self.identity, 3, update, rows, self.evidence)

    def test_b0r_first_update_requires_exact_retained_gradient_oracle(self):
        rows, check, counts, update = self.update_fixture(step=5)
        summary_path = self.package / 'run/diagnostic/SUMMARY.json'
        self.write(summary_path, {'capture': {'gradient_names': ['a', 'b'], 'gradient_file_sha256': 'a'*64}})
        summary_ref = dict(path=str(summary_path), sha256=file_sha(summary_path))
        acceptance_path = self.package / 'run/acceptance.json'
        self.write(acceptance_path, dict(diagnostic_summary=summary_ref))
        self.write(self.execution / 'GPU_EXECUTION_BINDING.json', {'gpu_production': {'acceptance':
            dict(path=str(acceptance_path), sha256=file_sha(acceptance_path))}})
        with self.assertRaisesRegex(PermissionError, 'exact retained diagnostic comparison'):
            verify_gpu_update_evidence(self.stage, self.identity, 5, update, rows, self.evidence)
        check.update(first_recovery_comparison='EXACT_DIAGNOSTIC_128_AND_CLIPPED_GRADIENTS',
                     compared_gradients=2, gradient_sha256='a'*64, diagnostic_summary=summary_ref)
        path = Path(update['gpu_pre_adam_check']['path'])
        self.write(path, check); update['gpu_pre_adam_check']['sha256'] = file_sha(path)
        verify_gpu_update_evidence(self.stage, self.identity, 5, update, rows, self.evidence)
        check['compared_gradients'] = 1
        self.write(path, check); update['gpu_pre_adam_check']['sha256'] = file_sha(path)
        with self.assertRaisesRegex(PermissionError, 'exact retained diagnostic comparison'):
            verify_gpu_update_evidence(self.stage, self.identity, 5, update, rows, self.evidence)

    def test_original_scientific_and_atomic_helpers_unchanged(self):
        package = Path(__file__).resolve().parents[1]
        def functions(path):
            tree = ast.parse(path.read_text())
            return {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef)}
        original = functions(package / 'adapter/seal_path.py')
        production = functions(package / 'gpu_production/seal_path.py')
        for name in ('path_parts', 'put_bytes', 'put_json', 'publish_receipt', 'reference',
                     'verify_token_vectors', 'verify_groups', 'main'):
            self.assertEqual(original[name], production[name], name)


if __name__ == '__main__':
    unittest.main()
