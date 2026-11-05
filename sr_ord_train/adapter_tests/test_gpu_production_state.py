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
from gpu_production import state as recovery
from gpu_production import binding as bridge

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


class GPUProductionStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.package = Path(self.temp.name).resolve()
        for relative in ('diagnostics', 'gpu_production', 'adapter', 'execution', 'run/paths/b0-11031-R/executions/test-first', 'frozen/receipt_plans/paths'):
            (self.package / relative).mkdir(parents=True)
        self.output = self.package / 'run/paths/b0-11031-R/executions/test-first'
        for relative in (*bridge.REQUIRED_FILES, 'adapter/backend.py', 'diagnostics/gpu_full_batch_probe.py'):
            (self.package / relative).write_text('synthetic ' + relative)
        source = dict(experiment_id='SR-ORD-TRAIN-1', package_root=str(self.package),
                      manifest_sha256='a'*64, source_sha256='b'*64, code_commit='synthetic-commit',
                      implementation_files=[dict(path=str(self.package / 'adapter/backend.py'),
                                                 sha256=file_hash(self.package / 'adapter/backend.py'))])
        self.source_path = self.package / 'execution/RUN_BINDING.json'
        write_once(self.source_path, source)
        self.patch_file = patch.object(bridge, '__file__', str(self.package / 'gpu_production/binding.py'))
        self.patch_file.start()
        self.patch_state_file = patch.object(recovery, "__file__", str(self.package / "gpu_production/state.py"))
        self.patch_state_file.start()
        self.path_identity = dict(arm='R', train_seed=11031, run_id='SR-ORD-TRAIN-1/b0-11031-R')
        self.added = [dict(path=str(self.package / p), sha256=file_hash(self.package / p)) for p in bridge.REQUIRED_FILES]
        self.summary = dict(status='FULL_BATCH_PRE_ADAM_PASS_NO_UPDATE', baseline_responses=128, gradient_forwards=128,
            diagnostic_backward_accounting={'backward_calls_completed': 128}, probability_comparison='EXACT_ALL_128',
            historical_gradient_overlap={'status':'EXACT_HISTORICAL_OVERLAP'}, production_files_unchanged=True,
            rng_restored=True,rng_restored_after_attempt=True,state_unchanged=True,state_unchanged_after_attempt=True,
            new_generations=0,optimizer_body_calls=0,optimizer_steps=0,scheduler_steps=0,original_long_gradient_equivalence='UNVERIFIED',
            activation_storage='cuda:1; arithmetic cuda:0; CPU fallback forbidden',
            saved_tensor_max_bytes_per_forward=16*1024**3,saved_tensor_copied_bytes_total=1024,
            gpu_storage_source_sha256=file_hash(self.package/'diagnostics/gpu_saved_tensors.py'),
            diagnostic_source_sha256=file_hash(self.package/'diagnostics/gpu_full_batch_probe.py'),
            native_audit=dict(status='PREUPDATE_PASS',precommit_gate='PASS_BEFORE_ADAM',numerics={'status':'FINITE'}))
        self.summary_path = self.package / 'run/diagnostic_summary.json'
        write_once(self.summary_path, self.summary)
        self.acceptance_path = self.package / 'run/acceptance.json'
        self.acceptance = dict(schema=bridge.ACCEPTANCE_SCHEMA,status='GPU_ONLY_PRODUCTION_BRIDGE_ACCEPTED',
            source_run_binding=dict(path=str(self.source_path),sha256=file_hash(self.source_path)),
            diagnostic_summary=dict(path=str(self.summary_path),sha256=file_hash(self.summary_path)),
            allowed_mode=bridge.MODE,original_long_gradient_equivalence='UNVERIFIED_ORIGINAL_BASELINE_OOM',
            execution_code_commit='c'*40,implementation_sha256=object_sha(self.added),allowed_paths=['b0-11031-R'])
        write_once(self.acceptance_path,self.acceptance)
        self.binding = self.make_binding(self.output)
        write_once(self.output / 'GPU_EXECUTION_BINDING.json',self.binding)
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

    def make_binding(self, output):
        return bridge.make_production_binding(self.source_path,output,path_identity=self.path_identity,
            execution_code_commit='c'*40,acceptance_path=self.acceptance_path)

    def publish_state(self):
        p = self.checkpoint / 'state.pt'
        p.write_text(json.dumps(normalize(self.source_state)))
        self.marker = dict(schema=SCHEMA, identity=self.identity, step=0, previous_commit_sha256=None,
                           sampler_state=self.source_state['sampler_state'], state_sha256=file_hash(p),
                           reference_sha256=state_hash(self.reference), policy_sha256=state_hash(self.source_state['policy']),
                           optimizer_sha256=state_hash(self.source_state['optimizer']),
                           scheduler_sha256=state_hash(self.source_state['scheduler']), rng_sha256=state_hash(self.source_state['rng']))
        (self.checkpoint / 'COMMIT.json').write_text(json.dumps(self.marker))
        self.tx._markers = None

    def tearDown(self):
        self.modules.stop()
        self.tx.__exit__(None, None, None)
        self.patch_file.stop()
        self.patch_state_file.stop()
        self.temp.cleanup()

    def restore(self):
        return recovery.restore_latest_gpu(self.tx, self.runtime, self.optimizer, self.scheduler,
                                           self.reference, self.binding, self.output)

    def test_exact_single_gpu_restore_and_immutable_lineage(self):
        old = {str(p): file_hash(p) for p in self.checkpoint.iterdir()}
        gpu1 = state_hash(self.current_rng['cuda'][1])
        self.restore()
        self.assertEqual(old, {str(p): file_hash(p) for p in self.checkpoint.iterdir()})
        self.assertEqual(state_hash(recovery.original_rng_projection(self.current_rng)), self.marker['rng_sha256'])
        self.assertEqual(state_hash(self.current_rng['cuda'][1]), gpu1)
        lineage = json.loads((self.output/'GPU_RECOVERY_LINEAGE.json').read_text())
        self.assertEqual(lineage['status'],'EXACT_ORIGINAL_STATE_RESTORED')
        self.assertEqual(lineage['execution_binding_sha256'],object_sha(self.binding))
        self.assertEqual(recovery.execution_provenance(self.binding,self.output,1)['source_step'],0)
        with self.assertRaisesRegex(PermissionError,'already exists'): self.restore()

    def test_binding_preserves_science_inventory_and_new_actual_version(self):
        original = json.loads(self.source_path.read_text())
        self.assertEqual(self.binding['implementation_files'][:1],original['implementation_files'])
        self.assertEqual(self.binding['code_commit'],'synthetic-commit')
        self.assertEqual(self.binding['gpu_production']['execution_code_commit'],'c'*40)
        for field,value in [('manifest_sha256','d'*64),('source_sha256','e'*64),('code_commit','different')]:
            bad=copy.deepcopy(self.binding);bad[field]=value
            with self.subTest(field=field),self.assertRaisesRegex(PermissionError,'RUN_BINDING fields'):
                bridge.verify_production_binding(bad,self.output)
        (self.package/'gpu_production/backend.py').write_text('changed')
        with self.assertRaisesRegex(PermissionError,'bound file'): self.restore()

    def test_exact_directory_and_path_identity(self):
        bad=copy.deepcopy(self.binding);bad['path_identity']['run_id']='SR-ORD-TRAIN-1/b1-11031-R'
        with self.assertRaisesRegex(PermissionError,'execution directory'):
            bridge.verify_production_binding(bad,self.output)
        bad=copy.deepcopy(self.binding);bad['path_identity']['arm']='D'
        with self.assertRaisesRegex(PermissionError,'fields disagree'):
            bridge.verify_production_binding(bad,self.output)
        link=self.package/'run/linked';link.symlink_to(self.output,target_is_directory=True)
        with self.assertRaisesRegex(PermissionError,'Symlink'):
            bridge.verify_production_binding(self.binding,link)

    def test_missing_dependency_or_actual_execution_commit_rejected(self):
        bad=copy.deepcopy(self.binding);bad['implementation_files']=bad['implementation_files'][:-1]
        with self.assertRaisesRegex(PermissionError,'digest'):
            bridge.verify_production_binding(bad,self.output)
        bad=copy.deepcopy(self.binding);bad['gpu_production']['execution_code_commit']='uncommitted'
        with self.assertRaisesRegex(PermissionError,'committed'):
            bridge.verify_production_binding(bad,self.output)

    def test_acceptance_tampering_and_nonaccepted_path(self):
        a=copy.deepcopy(self.acceptance);a['allowed_paths']=['b1-11031-R']
        self.acceptance_path.write_text(json.dumps(a))
        with self.assertRaisesRegex(PermissionError,'not accepted'):
            self.make_binding(self.output)
        a=copy.deepcopy(self.acceptance);a['original_long_gradient_equivalence']='PASS'
        self.acceptance_path.write_text(json.dumps(a))
        with self.assertRaisesRegex(PermissionError,'limitation'):
            self.make_binding(self.output)
        self.acceptance_path.write_text(json.dumps(self.acceptance))
        self.summary['optimizer_steps']=1;self.summary_path.write_text(json.dumps(self.summary))
        self.acceptance['diagnostic_summary']['sha256']=file_hash(self.summary_path)
        self.acceptance_path.write_text(json.dumps(self.acceptance))
        with self.assertRaisesRegex(PermissionError,'diagnostic evidence'):
            self.make_binding(self.output)

    def test_unknown_staging_and_unbound_two_gpu_checkpoint_rejected(self):
        self.tx._attempts={'unfinished':{'begin':{}}}
        with self.assertRaisesRegex(PermissionError,'UNKNOWN'): self.restore()
        self.tx._attempts={}
        staging=self.tx.directory/'checkpoints/.staging-unresolved';staging.mkdir()
        with self.assertRaisesRegex(PermissionError,'staging'): self.restore()
        staging.rmdir()
        self.source_state['rng']['cuda'].append(Tensor([22,23],'uint8'));self.publish_state()
        with self.assertRaisesRegex(PermissionError,'checkpoint provenance'): self.restore()

    def test_changed_state_and_loaded_optimizer_fail_closed(self):
        self.tx.published()
        with (self.checkpoint/'state.pt').open('a') as f:f.write(' ')
        with self.assertRaisesRegex(PermissionError,'Checkpoint bytes'): self.restore()
        self.publish_state()
        self.optimizer.load_state_dict=lambda state:None
        self.optimizer.state={'corrupt':True}
        with self.assertRaisesRegex(PermissionError,'Loaded optimizer'): self.restore()

    def test_fresh_parent_two_gpu_checkpoint_provenance_and_exact_restore(self):
        # Build a new synthetic path: this fixture-only deletion is not recovery policy.
        for p in self.checkpoint.iterdir():p.unlink()
        self.checkpoint.rmdir();self.tx._markers=None
        self.parameter.copy_(self.reference['adapter.weight']);self.runtime.identity['step']=0
        self.assertIsNone(self.restore())
        lineage=json.loads((self.output/'GPU_RECOVERY_LINEAGE.json').read_text())
        self.assertEqual(lineage['status'],'INITIAL_PARENT_TWO_GPU')
        prov=recovery.execution_provenance(self.binding,self.output,0)
        self.source_state['rng']=copy.deepcopy(self.current_rng)
        self.source_state['sampler_state']['gpu_execution_provenance']=prov
        self.checkpoint.mkdir();self.publish_state()
        marker=self.tx.published()[0]
        bound,origin=recovery.verify_checkpoint_provenance(self.package,marker)
        self.assertEqual(bound,self.binding);self.assertEqual(origin,lineage)
        output2=self.output.parent/'test-second';output2.mkdir()
        binding2=self.make_binding(output2);write_once(output2/'GPU_EXECUTION_BINDING.json',binding2)
        self.current_rng['cuda'][1]=Tensor([99,99],'uint8')
        recovery.restore_latest_gpu(self.tx,self.runtime,self.optimizer,self.scheduler,self.reference,binding2,output2)
        self.assertEqual(state_hash(self.current_rng),self.marker['rng_sha256'])
        restored=json.loads((output2/'GPU_RECOVERY_LINEAGE.json').read_text())
        self.assertEqual(restored['status'],'EXACT_TWO_GPU_STATE_RESTORED')

    def test_forged_lineage_rng_and_missing_execution_binding_rejected(self):
        self.restore();p=self.output/'GPU_RECOVERY_LINEAGE.json';lineage=json.loads(p.read_text())
        lineage['restored_original_rng_projection_sha256']='0'*64;p.write_text(json.dumps(lineage))
        with self.assertRaisesRegex(PermissionError,'RNG preservation'):
            recovery.execution_provenance(self.binding,self.output,1)
        (self.output/'GPU_EXECUTION_BINDING.json').write_text('{}')
        with self.assertRaisesRegex(PermissionError,'execution binding'):
            recovery.execution_provenance(self.binding,self.output,1)


class BackendCopyTests(unittest.TestCase):
    def test_only_explicit_binding_hardware_and_mode_changes(self):
        source=(PACKAGE/'adapter/backend.py').read_text()
        source=source.replace('    """Fail before importing native model code or reading model tensors."""\n',
            '    """Fail before importing native model code or reading model tensors."""\n'
            '    from gpu_production.binding import verify_production_binding\n'
            '    verify_production_binding(binding, run_dir)\n'
            '    if mode != "TRAINING":\n'
            '        raise PermissionError("GPU production bridge only permits registered training")\n',1)
        source=source.replace('        from mm_dev.runtime import actual_cuda_identity,configure_audited_backend\n',
            '        from mm_dev.runtime import configure_audited_backend\n'
            '        from gpu_production.state import actual_cuda_identity\n',1)
        source=source.replace('if not torch.cuda.is_available() or torch.cuda.device_count() != 1:',
            'if not torch.cuda.is_available() or torch.cuda.device_count() != 2:',1)
        source=source.replace("raise RuntimeError('Exactly one visible CUDA GPU required')",
            "raise RuntimeError('Exactly two visible CUDA GPUs required')",1)
        self.assertEqual(source,(PACKAGE/'gpu_production/backend.py').read_text())


if __name__=='__main__': unittest.main()
