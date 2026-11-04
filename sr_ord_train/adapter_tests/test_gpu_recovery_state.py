"""CPU-only synthetic checks. These are not CUDA/model equivalence evidence."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from adapter.transactions import PathTransactions, SCHEMA, file_hash, write_once
from adapter.backend import object_sha
from diagnostics import gpu_recovery_state as recovery

PACKAGE = Path(__file__).resolve().parents[1]


class Tensor:
    def __init__(self, values, dtype='float32', requires_grad=False):
        self.values, self.dtype, self.requires_grad = list(values), dtype, requires_grad
        self.shape = (len(self.values),)
    def detach(self): return self
    def clone(self): return copy.deepcopy(self)
    def cpu(self): return self
    def tolist(self): return list(self.values)
    def copy_(self, value): self.values = list(value.values)


def normalize(value):
    if isinstance(value, Tensor): return {'__tensor__': value.values, 'dtype': value.dtype}
    if isinstance(value, dict): return {k: normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [normalize(v) for v in value]
    return value


def state_hash(value):
    return hashlib.sha256(json.dumps(normalize(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def decode(value):
    return Tensor(value['__tensor__'], value['dtype']) if '__tensor__' in value else value


class StateHolder:
    def __init__(self, state): self.state = copy.deepcopy(state)
    def state_dict(self): return copy.deepcopy(self.state)
    def load_state_dict(self, state): self.state = copy.deepcopy(state)


class GPURecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.package = Path(self.temp.name).resolve()
        for relative in ('diagnostics', 'adapter', 'run/diag/new', 'frozen/receipt_plans/paths'):
            (self.package / relative).mkdir(parents=True)
        self.output = self.package / 'run/diag/new'
        for relative in ('diagnostics/gpu_backend.py', 'diagnostics/gpu_recovery_state.py', 'adapter/backend.py'):
            (self.package / relative).write_text('synthetic ' + relative)
        source = dict(experiment_id='SR-ORD-TRAIN-1', package_root=str(self.package),
                      manifest_sha256='a'*64, source_sha256='b'*64, code_commit='synthetic-commit',
                      implementation_files=[dict(path=str(self.package / 'adapter/backend.py'),
                                                 sha256=file_hash(self.package / 'adapter/backend.py'))])
        self.source_path = self.package / 'run/RUN_BINDING.json'
        write_once(self.source_path, source)
        self.patch_file = patch.object(recovery, '__file__', str(self.package / 'diagnostics/gpu_recovery_state.py'))
        self.patch_file.start()
        self.path_identity = dict(arm='R', train_seed=11031, run_id='SR-ORD-TRAIN-1/b0-11031-R')
        self.binding = recovery.make_diagnostic_binding(self.source_path, self.output, path_identity=self.path_identity)
        plan = self.package / 'frozen/receipt_plans/paths/b0-11031-R.json'
        plan.write_text('[{"step":1}]\n')
        write_once(self.package / 'frozen/receipt_plans/INDEX.json', {'files': {'paths/b0-11031-R.json': file_hash(plan)}})
        self.identity = dict(self.path_identity, experiment_id='SR-ORD-TRAIN-1', parent='b0',
                             manifest_sha256='a'*64, source_sha256='b'*64, code_commit='synthetic-commit',
                             plan_sha256=file_hash(plan), implementation_sha256=object_sha(source['implementation_files']))
        self.parameter = Tensor([9., 9.], requires_grad=True)
        self.model = SimpleNamespace(named_parameters=lambda: [('adapter.weight', self.parameter)])
        self.runtime = SimpleNamespace(model=self.model, identity=dict(self.identity, parent_id='b0',
                          execution_mode='TRAINING', step=-1))
        self.optimizer, self.scheduler = StateHolder({'state': {}, 'param_groups': [1]}), StateHolder({'last_epoch': 0})
        self.reference = {'adapter.weight': Tensor([1., 2.])}
        self.source_rng = dict(python=['python-state'], numpy=['MT19937', [1, 2], 0, 0, 0.],
                               cpu=Tensor([3, 4], 'uint8'), cuda=[Tensor([5, 6], 'uint8')])
        self.current_rng = dict(python=['different-python'], numpy=['MT19937', [9, 8], 0, 0, 0.],
                                cpu=Tensor([7, 8], 'uint8'), cuda=[Tensor([8, 9], 'uint8'), Tensor([21, 22], 'uint8')])
        self.source_state = dict(schema=SCHEMA, identity=self.identity, step=0, sampler_state={'next_step': 1},
                                 reference_sha256=state_hash(self.reference), policy=self.reference,
                                 optimizer=self.optimizer.state_dict(), scheduler=self.scheduler.state_dict(), rng=self.source_rng)
        self.tx = PathTransactions(self.package / 'run/paths/b0-11031-R', self.identity).__enter__()
        self.checkpoint = self.tx.directory / 'checkpoints/step-000'
        self.checkpoint.mkdir()
        self.publish_state()
        torch = ModuleType('torch')
        torch.float32 = 'float32'
        torch.cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 2)
        from contextlib import nullcontext
        torch.no_grad = nullcontext
        torch.load = lambda path, **kwargs: json.loads(Path(path).read_text(), object_hook=decode)
        torch.isfinite = lambda tensor: SimpleNamespace(all=lambda: all(math.isfinite(v) for v in tensor.values))
        core = ModuleType('mm_core.training')
        core.capture_rng = lambda: copy.deepcopy(self.current_rng)
        def restore(state): self.current_rng = copy.deepcopy(state)
        core.restore_rng = restore
        core.state_hash = state_hash
        core.trainable_state = lambda model: {n: p.clone() for n, p in model.named_parameters() if p.requires_grad}
        training = ModuleType('sr_f3_ce.training')
        self.optimizer_checks = []
        training.validate_optimizer = lambda opt, sched, model: self.optimizer_checks.append((opt, sched, model))
        numpy = ModuleType('numpy')
        numpy.uint32 = 'uint32'
        numpy.asarray = lambda values, dtype: list(values)
        self.modules = patch.dict('sys.modules', {'torch': torch, 'numpy': numpy,
            'mm_core': ModuleType('mm_core'), 'mm_core.training': core,
            'sr_f3_ce': ModuleType('sr_f3_ce'), 'sr_f3_ce.training': training})
        self.modules.start()
        self.core, self.torch = core, torch

    def publish_state(self):
        p = self.checkpoint / 'state.pt'
        p.write_text(json.dumps(normalize(self.source_state)))
        self.marker = dict(schema=SCHEMA, identity=self.identity, step=0, previous_commit_sha256=None,
                           sampler_state=self.source_state['sampler_state'], state_sha256=file_hash(p),
                           reference_sha256=state_hash(self.reference), policy_sha256=state_hash(self.source_state['policy']),
                           optimizer_sha256=state_hash(self.source_state['optimizer']),
                           scheduler_sha256=state_hash(self.source_state['scheduler']), rng_sha256=state_hash(self.source_state['rng']))
        (self.checkpoint / 'COMMIT.json').write_text(json.dumps(self.marker))

    def tearDown(self):
        self.modules.stop()
        self.tx.__exit__(None, None, None)
        self.patch_file.stop()
        self.temp.cleanup()

    def restore(self):
        return recovery.restore_latest_gpu(self.tx, self.runtime, self.optimizer, self.scheduler,
                                           self.reference, self.binding, self.output)

    def test_exact_original_state_and_explicit_gpu1_lineage_no_source_writes(self):
        before = {str(p): file_hash(p) for p in self.tx.directory.rglob('*') if p.is_file()}
        gpu1 = state_hash(self.current_rng['cuda'][1])
        marker = self.restore()
        after = {str(p): file_hash(p) for p in self.tx.directory.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(state_hash(recovery.original_rng_projection(self.current_rng)), self.marker['rng_sha256'])
        self.assertEqual(state_hash(self.current_rng['cuda'][1]), gpu1)
        self.assertEqual(state_hash({'adapter.weight': self.parameter}), self.marker['policy_sha256'])
        self.assertEqual(marker['step'], self.runtime.identity['step'])
        self.assertEqual(len(self.optimizer_checks), 1)
        receipt = json.loads((self.output / 'GPU_RECOVERY_LINEAGE.json').read_text())
        self.assertFalse(receipt['production_commit_authorized'])
        self.assertEqual(receipt['gpu1_rng']['state_bytes'], [21, 22])
        self.assertNotEqual(receipt['source_rng_sha256'], receipt['restored_full_two_gpu_rng_sha256'])
        with self.assertRaisesRegex(PermissionError, 'already exists'): self.restore()

    def test_binding_preserves_original_and_rejects_source_or_new_code_changes(self):
        original = json.loads(self.source_path.read_text())
        self.assertNotIn('gpu_diagnostic', original)
        self.assertEqual(self.binding['implementation_files'][:1], original['implementation_files'])
        bad = copy.deepcopy(self.binding)
        bad['manifest_sha256'] = 'c'*64
        with self.assertRaisesRegex(PermissionError, 'RUN_BINDING fields'): recovery.verify_diagnostic_binding(bad, self.output)
        (self.package / 'diagnostics/gpu_backend.py').write_text('tamper')
        with self.assertRaisesRegex(PermissionError, 'bound file'): self.restore()

    def test_original_implementation_changes_and_inventory_replacement_rejected(self):
        bad = copy.deepcopy(self.binding)
        bad['implementation_files'] = bad['implementation_files'][1:]
        with self.assertRaisesRegex(PermissionError, 'exact prefix'): recovery.verify_diagnostic_binding(bad, self.output)
        (self.package / 'adapter/backend.py').write_text('tamper old')
        with self.assertRaisesRegex(PermissionError, 'bound file'): self.restore()

    def test_production_output_and_symlink_rejected(self):
        with self.assertRaisesRegex(PermissionError, 'outside production'): recovery.verify_diagnostic_binding(self.binding, self.tx.directory)
        link = self.package / 'run/link'
        link.symlink_to(self.output, target_is_directory=True)
        with self.assertRaisesRegex(PermissionError, 'Symlink'): recovery.verify_diagnostic_binding(self.binding, link)

    def test_runtime_identity_and_manifest_checked(self):
        self.runtime.identity['arm'] = 'G'
        with self.assertRaisesRegex(PermissionError, 'Runtime path'): self.restore()
        self.runtime.identity['arm'] = 'R'
        self.runtime.identity['manifest_sha256'] = 'c'*64
        with self.assertRaisesRegex(PermissionError, 'Runtime manifest'): self.restore()

    def test_unknown_attempt_and_changed_checkpoint_rejected(self):
        self.tx._attempts = {'unfinished': {'begin': {}}}
        with self.assertRaisesRegex(PermissionError, 'UNKNOWN'): self.restore()
        self.tx._attempts = {}
        self.tx.published()  # Cache is deliberately populated before corruption.
        with (self.checkpoint / 'state.pt').open('a') as stream: stream.write(' ')
        with self.assertRaisesRegex(PermissionError, 'Checkpoint bytes'): self.restore()

    def test_serialized_metadata_content_and_rng_validated(self):
        for field, value in [('schema', 'wrong'), ('identity', {}), ('step', 9), ('sampler_state', {})]:
            with self.subTest(field=field):
                state = copy.deepcopy(self.source_state)
                state[field] = value
                with patch.object(self.torch, 'load', return_value=state):
                    with self.assertRaisesRegex(PermissionError, 'metadata'): self.restore()
        for field, value in [('policy', {'adapter.weight': Tensor([0., 0.])}),
                             ('optimizer', {}), ('scheduler', {}), ('reference_sha256', '0'*64),
                             ('rng', dict(self.source_rng, cpu=Tensor([0], 'uint8')))]:
            with self.subTest(field=field):
                state = copy.deepcopy(self.source_state)
                state[field] = value
                with patch.object(self.torch, 'load', return_value=state):
                    with self.assertRaisesRegex(PermissionError, 'content|RNG'): self.restore()

    def test_dtype_inventory_and_actual_loaded_optimizer_validated(self):
        self.parameter.dtype = 'bfloat16'
        with self.assertRaisesRegex(PermissionError, 'dtype/shape'): self.restore()
        self.parameter.dtype = 'float32'
        self.optimizer.load_state_dict = lambda state: None
        self.optimizer.state = {'corrupt': True}
        with self.assertRaisesRegex(PermissionError, 'Loaded optimizer'): self.restore()

    def test_gpu0_projection_and_gpu1_preservation_are_independent_checks(self):
        original_restore = self.core.restore_rng
        def corrupt_zero(state):
            original_restore(state)
            self.current_rng['cuda'][0].values[0] += 1
        self.core.restore_rng = corrupt_zero
        with self.assertRaisesRegex(PermissionError, 'projection'): self.restore()
        def corrupt_one(state):
            original_restore(state)
            self.current_rng['cuda'][1].values[0] += 1
        self.core.restore_rng = corrupt_one
        with self.assertRaisesRegex(PermissionError, 'GPU1 RNG'): self.restore()
        self.assertFalse((self.output / 'GPU_RECOVERY_LINEAGE.json').exists())

    def test_source_must_have_one_gpu_and_destination_two(self):
        self.source_state['rng']['cuda'].append(Tensor([1], 'uint8'))
        self.publish_state()
        with self.assertRaisesRegex(PermissionError, 'exactly one CUDA'): self.restore()
        self.torch.cuda.device_count = lambda: 1
        with self.assertRaisesRegex(PermissionError, 'Two visible GPUs'): self.restore()

    def test_hardware_identity_matches_both_uuids_and_requires_peer(self):
        self.torch.version = SimpleNamespace(cuda='synthetic')
        self.torch.cuda.get_device_properties = lambda i: SimpleNamespace(name='RTX PRO 6000 SYNTHETIC',
            uuid='GPU-' + str(i), total_memory=100, major=12, minor=0)
        self.torch.cuda.can_device_access_peer = lambda a, b: True
        rows = 'RTX PRO 6000 SYNTHETIC,GPU-0,0000:01,580,100\nRTX PRO 6000 SYNTHETIC,GPU-1,0000:02,580,100\n'
        with patch.object(recovery.subprocess, 'run', return_value=SimpleNamespace(stdout=rows)):
            hardware = recovery.actual_cuda_identity()
            self.assertEqual(hardware['cuda_visible_device_count'], 2)
            self.assertEqual([d['cuda_uuid'] for d in hardware['devices']], ['GPU-0', 'GPU-1'])
            self.torch.cuda.can_device_access_peer = lambda a, b: False
            with self.assertRaisesRegex(PermissionError, 'peer access'): recovery.actual_cuda_identity()


class BackendCopyTests(unittest.TestCase):
    def test_only_explicit_diagnostic_binding_and_hardware_differences(self):
        source = (PACKAGE / 'adapter/backend.py').read_text()
        changes = [
            ('    """Fail before importing native model code or reading model tensors."""\n',
             '    """Fail before importing native model code or reading model tensors."""\n'
             '    from diagnostics.gpu_recovery_state import verify_diagnostic_binding\n'
             '    verify_diagnostic_binding(binding, run_dir)\n'),
            ('        from mm_dev.runtime import actual_cuda_identity,configure_audited_backend\n',
             '        from mm_dev.runtime import configure_audited_backend\n'
             '        from diagnostics.gpu_recovery_state import actual_cuda_identity\n'),
            ('torch.cuda.device_count() != 1:', 'torch.cuda.device_count() != 2:'),
            ("raise RuntimeError('Exactly one visible CUDA GPU required')",
             "raise RuntimeError('Exactly two visible CUDA GPUs required for isolated diagnostic')")]
        for before, after in changes:
            self.assertEqual(source.count(before), 1)
            source = source.replace(before, after)
        self.assertEqual(source, (PACKAGE / 'diagnostics/gpu_backend.py').read_text())


if __name__ == '__main__': unittest.main()
