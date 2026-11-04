"""CPU evidence/control fixtures only: no actual CUDA execution is claimed."""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapter.transactions import file_hash
from diagnostics.gpu_gradient_overlap import (
    EXACT_FIELDS, EXPECTED_GRADIENTS, MANIFEST_FIELDS, _cleanup_attempt,
    _recheck_refs, compare_fields, compare_gradients, load_prior_evidence,
)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class ScalarTensor:
    """Scalar arithmetic fixture; the device assertion is a mock, not CUDA."""
    dtype, shape = 'float32', (1,)

    def __init__(self, value):
        self.value = value

    def to(self, *, device, non_blocking=False):
        assert device == 'cuda:0' and non_blocking is False
        return self

    def double(self):
        return self

    def __sub__(self, other):
        return ScalarTensor(self.value - other.value)

    def abs(self):
        return ScalarTensor(abs(self.value))

    def max(self):
        return self.value

    def numel(self):
        return 1


FAKE_TORCH = SimpleNamespace(
    is_tensor=lambda value: isinstance(value, ScalarTensor),
    isfinite=lambda value: SimpleNamespace(all=lambda: math.isfinite(value.value)),
    equal=lambda left, right: left.value == right.value,
)


class HistoricalEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.package = Path(self.temp.name).resolve()
        self.manifest = {key: key + '-fixture' for key in MANIFEST_FIELDS}
        self.manifest.update(path='b0-11031-R', checkpoint_step=4, next_step=5,
                             probes=[{'request_id': 'first'}, {'request_id': 'longest'}])
        self.root = self.package / 'run/memory_diag_264303_b0-11031-R'
        save(self.root / 'MANIFEST.json', self.manifest)
        self.results = []
        for index, mode in ((0, 'baseline'), (1, 'bounded_cpu_saved_tensors')):
            folder = self.root / f'probe-{index}-{mode}'
            folder.mkdir()
            gradient = folder / 'adapter_gradients.pt'
            gradient.write_bytes(b'immutable gradient evidence fixture, not a torch tensor')
            value = {key: key + '-fixture' for key in EXACT_FIELDS}
            value.update(status='COMPLETE', mode=mode, probe=self.manifest['probes'][index],
                         request_id=self.manifest['probes'][index]['request_id'],
                         state_unchanged_after_attempt=True, rng_restored_after_attempt=True,
                         restored_rng_sha256=self.manifest['rng_sha256'],
                         immutable_state_hashes={'policy': self.manifest['policy_sha256'],
                                                 'reference': self.manifest['reference_sha256'],
                                                 'optimizer': 'adam', 'scheduler': 'lr'},
                         new_generations=0, optimizer_steps=0, scheduler_steps=0,
                         gradient_file=str(gradient), gradient_file_sha256=file_hash(gradient),
                         gradient_names=[f'adapter-{i}' for i in range(EXPECTED_GRADIENTS)],
                         gradient_none_names=[], train_logprobs=[-.5], reference_logprobs=[-.6],
                         scalar_terms={'loss': .1}, tensor_terms={'ratio': {'values': [1.1]}})
            result_path = folder / 'RESULT.json'
            save(result_path, value)
            self.results.append((result_path, value))
        save(self.root / 'probe-1-baseline/RESULT.json',
             {'status': 'OOM', 'mode': 'baseline', 'probe': self.manifest['probes'][1]})

    def test_fixed_short_long_evidence_pinned(self):
        result = load_prior_evidence(self.package, self.manifest)
        self.assertEqual(result['status'], 'HISTORICAL_EVIDENCE_PINNED')
        self.assertEqual(result['original_long_baseline_status'], 'OOM')
        self.assertEqual([p['mode'] for p in result['probes']], ['baseline', 'bounded_cpu_saved_tensors'])
        self.assertEqual(len(result['refs']), 6)
        _recheck_refs(result['refs'])

    def test_changed_gradient_bytes_rejected(self):
        Path(self.results[0][1]['gradient_file']).write_bytes(b'changed')
        with self.assertRaises(PermissionError):
            load_prior_evidence(self.package, self.manifest)

    def test_each_manifest_identity_field_bound(self):
        for key in MANIFEST_FIELDS:
            with self.subTest(key=key):
                changed = copy.deepcopy(self.manifest)
                changed[key] = 'other'
                with self.assertRaises((PermissionError, KeyError)):
                    load_prior_evidence(self.package, changed)

    def test_long_original_oom_cannot_be_relabelled(self):
        save(self.root / 'probe-1-baseline/RESULT.json',
             {'status': 'COMPLETE', 'mode': 'baseline', 'probe': self.manifest['probes'][1]})
        with self.assertRaises(PermissionError):
            load_prior_evidence(self.package, self.manifest)

    def test_old_result_request_identity_and_audit_required(self):
        path, original = self.results[0]
        for key, replacement in [('request_id', 'wrong'), ('optimizer_steps', 1),
                                 ('state_unchanged_after_attempt', False),
                                 ('rng_restored_after_attempt', False),
                                 ('restored_rng_sha256', 'wrong')]:
            with self.subTest(key=key):
                save(path, dict(original, **{key: replacement}))
                with self.assertRaises(PermissionError):
                    load_prior_evidence(self.package, self.manifest)
        save(path, original)

    def test_historical_mutation_after_pin_rejected(self):
        pinned = load_prior_evidence(self.package, self.manifest)
        self.results[0][0].write_text('{}')
        with self.assertRaises(PermissionError):
            _recheck_refs(pinned['refs'])

    def test_b0g_absence_is_not_available(self):
        manifest = dict(self.manifest, path='b0-11031-G')
        result = load_prior_evidence(self.package, manifest)
        self.assertEqual(result['status'], 'NOT_AVAILABLE')
        self.assertEqual(result['refs'], [])
        (self.package / 'run/memory_diag_264303_b0-11031-G').mkdir()
        with self.assertRaises(PermissionError):
            load_prior_evidence(self.package, manifest)

    def test_every_objective_probability_identity_inventory_field_compared(self):
        previous = self.results[0][1]
        self.assertTrue(all(compare_fields(copy.deepcopy(previous), previous).values()))
        for key in EXACT_FIELDS:
            with self.subTest(key=key):
                actual = copy.deepcopy(previous)
                actual[key] = 'different'
                self.assertFalse(compare_fields(actual, previous)[key])
                del actual[key]
                with self.assertRaises(PermissionError):
                    compare_fields(actual, previous)

    def test_rng_cleanup_attempted_despite_earlier_cleanup_failures(self):
        calls = []
        def failed():
            calls.append('zero_grad')
            raise RuntimeError('fixture zero_grad failure')
        errors = _cleanup_attempt((('zero_grad', failed),
                                   ('restore_rng', lambda: calls.append('restored_full_two_gpu_rng'))))
        self.assertEqual(calls, ['zero_grad', 'restored_full_two_gpu_rng'])
        self.assertEqual(errors[0][0], 'zero_grad')

    def test_all_496_gradients_compared_with_exact_values_and_none_status(self):
        actual = {f'adapter-{i}': ScalarTensor(i) for i in range(EXPECTED_GRADIENTS)}
        previous = copy.deepcopy(actual)
        comparison = compare_gradients(actual, previous, torch=FAKE_TORCH)
        self.assertEqual(len(comparison), EXPECTED_GRADIENTS)
        self.assertTrue(all(value['exact_equal'] for value in comparison.values()))
        previous['adapter-495'] = ScalarTensor(495.25)
        previous['adapter-1'] = None
        comparison = compare_gradients(actual, previous, torch=FAKE_TORCH)
        self.assertFalse(comparison['adapter-495']['exact_equal'])
        self.assertEqual(comparison['adapter-495']['max_absolute_difference'], .25)
        self.assertFalse(comparison['adapter-1']['exact_equal'])

    def test_gradient_nonfinite_dtype_shape_inventory_fail_closed(self):
        actual = {f'adapter-{i}': ScalarTensor(i) for i in range(EXPECTED_GRADIENTS)}
        for field, replacement in [('value', float('nan')), ('dtype', 'float16'), ('shape', (2,))]:
            with self.subTest(field=field):
                previous = copy.deepcopy(actual)
                setattr(previous['adapter-495'], field, replacement)
                with self.assertRaises(PermissionError):
                    compare_gradients(actual, previous, torch=FAKE_TORCH)
        previous = dict(actual)
        previous.pop('adapter-495')
        with self.assertRaises(PermissionError):
            compare_gradients(actual, previous, torch=FAKE_TORCH)


if __name__ == '__main__':
    unittest.main()
