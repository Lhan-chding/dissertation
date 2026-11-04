"""Mocked-device CPU control tests. These are NOT actual CUDA equivalence tests."""
from contextlib import contextmanager
import gc
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnostics.gpu_saved_tensors import CAP_BYTES, GPUSavedTensors, gpu_peak_stats, validate_gpu_pair


class FakeTensor:
    """Bytes and shared-version fixture; never allocates claimed virtual GPU bytes."""
    next_pointer = 100
    copies = []
    fail_destination = None
    def __init__(self, payload=b'\x00\x00\x80?\x00\x00\x00@', *, device='cuda:0',
                 count=2, shape=None, contiguous=True, state=None, dtype='float32'):
        self.device = device
        self.dtype = dtype
        self.layout = 'strided'
        self.shape = tuple(shape if shape is not None else (count,))
        self.count = count
        self.contiguous = contiguous
        self.requires_grad = False
        if state is None:
            FakeTensor.next_pointer += 1
            state = SimpleNamespace(payload=payload, version=0, pointer=FakeTensor.next_pointer)
        self.state = state

    @property
    def _version(self):
        return self.state.version

    def numel(self):
        return self.count

    def element_size(self):
        return 4

    def untyped_storage(self):
        return SimpleNamespace(data_ptr=lambda: self.state.pointer)

    def is_contiguous(self):
        return self.contiguous

    def is_conj(self):
        return False

    def is_neg(self):
        return False

    def detach(self):
        return FakeTensor(device=self.device, count=self.count, shape=self.shape,
                          contiguous=self.contiguous, state=self.state, dtype=self.dtype)

    def to(self, *, device, copy, non_blocking):
        FakeTensor.copies.append((self.device, device, copy, non_blocking))
        if device == FakeTensor.fail_destination:
            raise RuntimeError('fixture transfer OOM')
        if device not in ('cuda:0', 'cuda:1') or not copy or non_blocking:
            raise AssertionError('CPU/async/noncopy transfer is forbidden')
        return FakeTensor(self.state.payload, device=device, count=self.count, shape=self.shape, dtype=self.dtype)

    def mutate(self, payload):
        self.state.payload = payload
        self.state.version += 1


class FakeParameter(FakeTensor):
    pass


class FakeTorch:
    strided = 'strided'
    nn = SimpleNamespace(Parameter=FakeParameter)
    def __init__(self):
        self.current = 0
        self.devices = 2
        self.names = ['NVIDIA RTX PRO 6000 Blackwell Server Edition'] * 2
        self.peer = {(0, 1): True, (1, 0): True}
        self.sync_calls = []
        self.reset_calls = []
        self.hook_entries = 0
        self.cuda = SimpleNamespace(
            is_available=lambda: True, device_count=lambda: self.devices,
            current_device=lambda: self.current,
            get_device_name=lambda i: self.names[i],
            get_device_properties=lambda i: SimpleNamespace(major=12, minor=0, total_memory=96*1024**3, uuid=str(i)),
            can_device_access_peer=lambda a, b: self.peer[(a, b)],
            synchronize=self.sync_calls.append, reset_peak_memory_stats=self.reset_calls.append,
            memory_allocated=lambda i: 100+i, memory_reserved=lambda i: 200+i,
            max_memory_allocated=lambda i: 300+i, max_memory_reserved=lambda i: 400+i)
        self.autograd = SimpleNamespace(graph=SimpleNamespace(saved_tensors_hooks=self.hooks))

    @staticmethod
    def is_tensor(value):
        return isinstance(value, FakeTensor)

    @contextmanager
    def hooks(self, pack, unpack):
        self.hook_entries += 1
        yield


class MockedGPUControlTests(unittest.TestCase):
    def setUp(self):
        FakeTensor.copies = []
        FakeTensor.fail_destination = None
        self.torch = FakeTorch()
        self.patch = patch('diagnostics.gpu_saved_tensors._torch', return_value=self.torch)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def hook(self, persistent=()):
        return GPUSavedTensors(persistent, row_id='original-row-17')

    def test_peer_must_work_both_directions_without_fallback(self):
        self.torch.peer[(1, 0)] = False
        with self.assertRaisesRegex(RuntimeError, 'Bidirectional'):
            self.hook()
        self.assertEqual(FakeTensor.copies, [])

    def test_two_same_pro6000_and_cuda0_are_mandatory(self):
        for attribute, value in [('devices', 1), ('current', 1), ('names', ['A100', 'A100']),
                                 ('names', ['RTX PRO 6000', 'RTX PRO 6000 other'])]:
            old = getattr(self.torch, attribute)
            setattr(self.torch, attribute, value)
            with self.subTest(attribute=attribute, value=value), self.assertRaises(RuntimeError):
                validate_gpu_pair()
            setattr(self.torch, attribute, old)

    def test_lossless_snapshot_restores_metadata_and_uses_blocking_gpu_copies(self):
        hook = self.hook()
        source = FakeTensor(b'\x00\x00\xc0\x7f\x00\x00\x00\x80')  # Preserve arbitrary bits, including NaN/-0.
        packed = hook.pack(source)
        restored = hook.unpack(packed)
        self.assertEqual(restored.state.payload, source.state.payload)
        self.assertEqual((restored.device, restored.dtype, restored.shape), ('cuda:0', source.dtype, source.shape))
        self.assertEqual(FakeTensor.copies, [('cuda:0', 'cuda:1', True, False), ('cuda:1', 'cuda:0', True, False)])
        self.assertEqual(hook.stats['copied_bytes'], 8)
        self.assertEqual(hook.stats['copied_tensors'], 1)
        self.assertEqual(hook.stats['unpacked_copies'], 1)
        self.assertEqual(hook.stats['unpacked_bytes'], 8)
        self.assertFalse(hook.stats['cpu_offload'])
        self.assertGreaterEqual(hook.stats['transfer_wall_seconds'], 0.)

    def test_original_tensor_has_no_retained_strong_reference(self):
        hook = self.hook()
        source = FakeTensor()
        reference = weakref.ref(source)
        packed = hook.pack(source)
        del source
        gc.collect()
        self.assertIsNone(reference())
        hook.unpack(packed)
        self.assertEqual(hook.stats['source_version_unobservable'], 1)

    def test_live_source_inplace_is_detected_before_copy_back(self):
        hook = self.hook()
        source = FakeTensor()
        packed = hook.pack(source)
        source.mutate(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'Original saved tensor'):
            hook.unpack(packed)
        self.assertEqual(hook.stats['version_failures'], 1)
        self.assertEqual(len(FakeTensor.copies), 1)

    def test_snapshot_mutation_is_detected(self):
        hook = self.hook()
        packed = hook.pack(FakeTensor())
        packed.tensor.mutate(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'snapshot/kept'):
            hook.unpack(packed)

    def test_dead_original_alias_limit_is_counted_and_snapshot_remains_exact(self):
        hook = self.hook()
        source = FakeTensor()
        before = source.state.payload
        alias = source.detach()
        packed = hook.pack(source)
        del source
        gc.collect()
        alias.mutate(b'changed alias')
        restored = hook.unpack(packed)
        self.assertEqual(restored.state.payload, before)
        self.assertEqual(hook.stats['source_version_unobservable'], 1)
        self.assertFalse(hook.stats['equivalence_verified'])

    def test_parameters_buffers_and_their_aliases_stay_on_cuda0(self):
        parameter, buffer = FakeParameter(), FakeTensor()
        hook = self.hook((parameter, buffer))
        for tensor in (parameter, parameter.detach(), buffer, buffer.detach(), FakeParameter()):
            packet = hook.pack(tensor)
            self.assertFalse(packet.copied)
            self.assertEqual(hook.unpack(packet).device, 'cuda:0')
        self.assertEqual(FakeTensor.copies, [])
        self.assertEqual(hook.stats['persistent_storage_skips'], 4)

    def test_noncontiguous_and_other_device_values_are_not_copied(self):
        hook = self.hook()
        for tensor in (FakeTensor(contiguous=False), FakeTensor(device='cuda:1'), FakeTensor(device='cpu')):
            self.assertFalse(hook.pack(tensor).copied)
        self.assertEqual(FakeTensor.copies, [])

    def test_nonstrided_saved_tensor_does_not_require_storage_access(self):
        hook = self.hook()
        tensor = FakeTensor()
        tensor.layout = 'sparse_coo'
        def unavailable():
            raise RuntimeError('sparse tensors have no directly accessible storage')
        tensor.untyped_storage = unavailable
        self.assertFalse(hook.pack(tensor).copied)
        self.assertEqual(FakeTensor.copies, [])

    def test_persistent_cpu_tensor_rejects_model_placement(self):
        with self.assertRaisesRegex(RuntimeError, 'already be on cuda:0'):
            self.hook([FakeTensor(device='cpu')])

    def test_fixed_16gib_cumulative_cap_never_resets_on_unpack(self):
        hook = self.hook()
        source = FakeTensor(count=CAP_BYTES // 4)
        packet = hook.pack(source)
        self.assertTrue(packet.copied)
        hook.unpack(packet)
        self.assertFalse(hook.pack(FakeTensor()).copied)
        self.assertEqual(hook.stats['copied_bytes'], CAP_BYTES)
        self.assertEqual(hook.stats['cap_skips'], 1)
        self.assertTrue(self.hook().pack(FakeTensor()).copied)

    def test_transfer_failure_propagates_without_cpu_or_retry(self):
        hook = self.hook()
        FakeTensor.fail_destination = 'cuda:1'
        with self.assertRaisesRegex(RuntimeError, 'transfer OOM'):
            hook.pack(FakeTensor())
        self.assertEqual(len(FakeTensor.copies), 1)
        self.assertEqual(hook.stats['copy_failures'], 1)
        self.assertEqual(hook.stats['copied_bytes'], 0)

    def test_one_context_per_original_row_and_backward_outside_context(self):
        hook = self.hook()
        with hook.context():
            packet = hook.pack(FakeTensor())
        hook.unpack(packet)
        with self.assertRaisesRegex(RuntimeError, 'new hook'):
            with hook.context():
                pass
        self.assertEqual(self.torch.hook_entries, 1)

    def test_peak_counters_report_both_gpus_without_implicit_reset(self):
        stats = gpu_peak_stats()
        self.assertEqual(set(stats), {'cuda:0', 'cuda:1'})
        self.assertEqual(stats['cuda:1']['peak_allocated_bytes'], 301)
        self.assertEqual(self.torch.reset_calls, [])
        gpu_peak_stats(reset=True)
        self.assertEqual(self.torch.reset_calls, [0, 1])


if __name__ == '__main__':
    unittest.main()
