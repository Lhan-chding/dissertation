"""496-scalar CPU loader/real transaction tests; GPU construction is explicitly mocked."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from execution import evaluation_loader as e
spec = importlib.util.spec_from_file_location('_eval_transaction_fixture',
    Path(__file__).with_name('test_training_transaction.py'))
fixtures = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixtures)


@unittest.skipUnless((fixtures.fixtures.NATIVE/'sr_f3_ce/training.py').exists(), 'Native CPU sources required')
class EvaluationLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.TrainingTransactionTests.setUpClass()
        cls.addClassCleanup(fixtures.TrainingTransactionTests.doClassCleanups)

    def setUp(self):
        self.fx = fixtures.TrainingTransactionTests('test_real_adam_two_steps_resume_and_commit_idempotency')
        self.fx.setUp(); self.addCleanup(self.fx.doCleanups)
        self.ops = self.fx.fixture.ops
        self.initial = self.ops.trainable_state(self.fx.fixture.runtime.model)
        self.reference = copy.deepcopy(self.fx.fixture.reference)
        self.first = self.fx.run_step(1); self.second = self.fx.run_step(2)
        self.root = self.fx.root
        self.registry = SimpleNamespace(root=self.root, freeze_path=self.fx.freeze, freeze_sha256=self.fx.freeze_sha,
            config=dict(experiment_id=e.state.EXPERIMENT, parent='b0',
                        start_bindings={'11031':copy.deepcopy(self.fx.fixture.source)}))
        self.registry.requests = {name:dict(request_id=name, scope='MAIN',start_seed=11031,arm=arm,step=step)
                                  for name,arm,step in [('start','START',0),('two','T0',2)]}
        self.registry.request = lambda rid:copy.deepcopy(self.registry.requests[rid])
        model = torch.nn.Module(); model.lora_fixture = torch.nn.ParameterList(
            [torch.nn.Parameter(value.clone()) for value in self.initial.values()])
        runtime = SimpleNamespace(model=model, identity=dict(experiment_id=e.state.EXPERIMENT,
            arm='START',step=0,train_seed=11031,reference_parent='b0'), _satdose_active=None,
            _satdose_closed=False, _vision_hook=SimpleNamespace(remove=Mock()))
        self.handle = SimpleNamespace(runtime=runtime,native_ops=self.ops,
            fork=SimpleNamespace(reference=copy.deepcopy(self.reference),optimizer=None,scheduler=None),
            registry=self.registry,start_seed=11031)
        self.source_hashes = {str(path):e.file_sha(path) for path in
            (Path(self.first['checkpoint'])/'state.pt',Path(self.second['checkpoint'])/'state.pt')}

    def load(self, **kwargs):
        args = dict(step=2,expected_commit_sha256=self.second['commit_sha256'],expected_identity=self.fx.identity)
        args.update(kwargs)
        return e.load_committed_policy(self.handle,self.fx.store,**args)

    def fingerprint(self):
        return self.ops.state_hash(dict(policy=self.ops.trainable_state(self.handle.runtime.model),
            reference=self.handle.fork.reference,identity=self.handle.runtime.identity,rng=self.ops.capture_rng()))

    def test_real_full_chain_policy_only_no_optimizer_or_training_rng_restore(self):
        rng = self.ops.state_hash(self.ops.capture_rng())
        raw_load = torch.load
        with patch.object(self.ops,'fresh_optimizer',side_effect=AssertionError('optimizer factory forbidden')), \
             patch.object(self.ops,'restore_rng',side_effect=AssertionError('RNG restore forbidden')), \
             patch.object(torch.optim.AdamW,'__init__',side_effect=AssertionError('Adam construction forbidden')), \
             patch.object(torch,'load',wraps=raw_load) as loads:
            receipt = self.load()
        self.assertTrue(loads.call_args_list)
        self.assertTrue(all(call.kwargs.get('weights_only') is True for call in loads.call_args_list))
        marker = e.read_json(Path(self.second['checkpoint'])/'COMMIT.json')
        self.assertEqual(receipt['policy_sha256'],marker['policy_sha256'])
        self.assertEqual(self.ops.state_hash(self.ops.trainable_state(self.handle.runtime.model)),marker['policy_sha256'])
        self.assertEqual(self.ops.state_hash(self.handle.fork.reference),self.fx.fixture.source['reference_sha256'])
        self.assertEqual(self.ops.state_hash(self.ops.capture_rng()),rng)
        self.assertIsNone(self.handle.fork.optimizer); self.assertIsNone(self.handle.fork.scheduler)
        self.assertFalse(receipt['training_rng_restored'])
        self.assertEqual(self.handle.runtime.identity['step'],2)
        for path,sha in self.source_hashes.items():self.assertEqual(e.file_sha(path),sha)

    def test_wrong_trusted_commit_no_mutation(self):
        before = self.fingerprint()
        with self.assertRaises(PermissionError):self.load(expected_commit_sha256='0'*64)
        self.assertEqual(self.fingerprint(),before)

    def test_unknown_other_path_blocks_full_chain(self):
        other = self.fx.store.store/'start-11032/T1/.staging-001'; other.mkdir(parents=True)
        (other/'UPDATE_INTENT.json').write_text('{}')
        before = self.fingerprint()
        with self.assertRaisesRegex(PermissionError,'occupied staging'):
            self.load()
        self.assertEqual(self.fingerprint(),before)

    def test_dirty_runtime_and_live_optimizer_rejected(self):
        self.handle.runtime.model.lora_fixture[0].grad = torch.zeros(1)
        before = self.fingerprint()
        with self.assertRaisesRegex(PermissionError,'Dirty/active'):self.load()
        self.assertEqual(self.fingerprint(),before)
        self.handle.runtime.model.zero_grad(set_to_none=True)
        self.handle.fork.optimizer = object()
        with self.assertRaisesRegex(PermissionError,'optimizer'):self.load()

    def test_unregistered_or_cross_arm_endpoint_rejected(self):
        before = self.fingerprint()
        with self.assertRaisesRegex(PermissionError,'registered checkpoint'):self.load(step=1)
        wrong = copy.deepcopy(self.fx.identity); wrong['arm']='T1'
        with self.assertRaisesRegex(PermissionError,'Foreign transaction'):self.load(expected_identity=wrong)
        self.assertEqual(before,self.fingerprint())

    def test_state_tamper_fails_before_copy(self):
        with (Path(self.second['checkpoint'])/'state.pt').open('ab') as stream:stream.write(b'changed')
        before = self.fingerprint()
        with self.assertRaises(PermissionError):self.load()
        self.assertEqual(before,self.fingerprint())

    def test_independent_handle_checks_trained_policy_and_closes_without_frozen_R24_close(self):
        loaded = self.load()
        endpoint = e.registered_endpoint(self.registry,11031,'T0',2)
        self.handle.close = Mock(side_effect=AssertionError('frozen close assumes R24'))
        result = e.EvaluationLoadHandle(self.handle,endpoint,loaded['policy_sha256'],self.source_hashes,loaded)
        self.assertTrue(self.handle.runtime._satdose_closed)
        self.assertEqual(result.audit()['step'],2)
        self.assertFalse(hasattr(result,'generate_one'))
        self.assertFalse(hasattr(result,'runtime'))
        receipt = result.close()
        self.assertTrue(receipt['closed'])
        self.handle.runtime._vision_hook.remove.assert_called_once()
        self.handle.close.assert_not_called()
        with self.assertRaisesRegex(PermissionError,'already closed'):result.audit()

    def test_load_only_handle_reopening_is_rejected(self):
        loaded = self.load()
        result = e.EvaluationLoadHandle(self.handle,e.registered_endpoint(self.registry,11031,'T0',2),
            loaded['policy_sha256'],{},loaded)
        self.handle.runtime._satdose_closed = False
        with self.assertRaisesRegex(PermissionError,'reopened'):
            result.close()
        self.handle.runtime._vision_hook.remove.assert_called_once()

    def test_dirty_handle_close_still_removes_hook(self):
        loaded = self.load()
        result = e.EvaluationLoadHandle(self.handle,e.registered_endpoint(self.registry,11031,'T0',2),
            loaded['policy_sha256'],{},loaded)
        with torch.no_grad():self.handle.runtime.model.lora_fixture[0].add_(1)
        with self.assertRaisesRegex(PermissionError,'Dirty evaluation'):result.close()
        self.handle.runtime._vision_hook.remove.assert_called_once()

    def test_actual_start_factory_boundary_mocked_no_GPU_claim(self):
        # START branch deliberately calls the same actual frozen builder symbol;
        # it is mocked here, so this test is not GPU model-loading evidence.
        registry = object.__new__(e.FrozenRegistry)
        registry.__dict__.update(self.registry.__dict__)
        statefile = self.root/'original_R24.pt';statefile.write_bytes(b'CPU source identity fixture')
        source = dict(registry.config['start_bindings']['11031'],original_state_path=str(statefile),state_sha256=e.file_sha(statefile))
        registry.config = dict(registry.config,start_bindings={'11031':source})
        self.handle.registry = registry
        parent = self.root/'b0';parent.mkdir()
        for name in ('adapter_config.json','adapter_model.safetensors'):(parent/name).write_bytes(b'CPU parent fixture')
        lockpath = self.root/'load_lock.json';lockpath.write_text('{}')
        lock = dict(native_files={},parent_files={p.name:e.file_sha(p) for p in parent.iterdir()})
        output = self.root/'load_audit';output.mkdir()
        rng = self.ops.state_hash(self.ops.capture_rng())
        def model_factory(*args,**kwargs):
            torch.rand(5)  # The original GPU constructor seeds/initializes RNG too.
            return self.handle
        with patch.object(e,'verify_sources',return_value=(lock,Path('/mock/native'),None,None,parent)), \
             patch.object(e.fork_loader,'load_native_ops',return_value=self.ops), \
             patch.object(e.frozen,'build_runtime',side_effect=model_factory) as factory:
            result=e.build_evaluation_loader(registry,lockpath,start_seed=11031,arm='START',step=0,
                output_dir=output,execution_sources={},expected_execution_sources_sha256='f'*64)
        factory.assert_called_once()
        self.assertEqual(self.ops.state_hash(self.ops.capture_rng()),rng)
        self.assertEqual(result.audit()['policy_sha256'],source['policy_sha256'])
        self.assertFalse(result.receipt['gpu_acceptance'])
        self.assertFalse(result.receipt['loaded']['optimizer_constructed'])
        self.assertTrue((output/'EVALUATION_LOAD.json').is_file())
        result.close()

    def test_copy_failure_rolls_back_policy_and_preserves_caller_rng(self):
        before = self.fingerprint()
        original_copy = e.fork_loader._copy
        copies = []
        def interrupted_copy(parameters, policy):
            original_copy(parameters, policy)
            copies.append(True)
            if len(copies) == 1:
                torch.rand(5)
                raise RuntimeError('synthetic interrupted policy copy')
        caller_rng = self.ops.capture_rng()
        training_rng = e.state.unpack_rng(torch.load(Path(self.second['checkpoint'])/'state.pt',
                                                     weights_only=True)['rng'])
        # Make caller RNG observably different from the checkpoint's training RNG.
        torch.rand(1)
        caller_rng = self.ops.capture_rng(); before = self.fingerprint()
        raw_restore = self.ops.restore_rng
        with patch.object(e.fork_loader, '_copy', side_effect=interrupted_copy), \
             patch.object(self.ops, 'restore_rng', wraps=raw_restore) as restore:
            with self.assertRaisesRegex(RuntimeError, 'interrupted policy copy'):
                self.load()
        self.assertEqual(len(copies),2)
        self.assertEqual(before,self.fingerprint())
        restore.assert_called_once()
        self.assertEqual(self.ops.state_hash(restore.call_args.args[0]), self.ops.state_hash(caller_rng))
        self.assertNotEqual(self.ops.state_hash(restore.call_args.args[0]), self.ops.state_hash(training_rng))

    def test_evaluation_source_inventory_trusted_sha_required(self):
        with self.assertRaisesRegex(PermissionError,'Untrusted evaluation source'):
            e.verify_sources(self.registry,self.root/'source_lock.json',{},'a'*64)


if __name__ == '__main__':unittest.main()
