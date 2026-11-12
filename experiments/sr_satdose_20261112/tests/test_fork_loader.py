"""CPU fixtures only: real native hash/Adam; no production model or admission."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
import fork_loader as f

NATIVE = os.environ.get('SATDOSE_NATIVE_SRC')


class TinyAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Parameter(torch.tensor([3.]), requires_grad=False)
        self.lora_fixture = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor([i / 1000.])) for i in range(496)])


@unittest.skipUnless(NATIVE, 'Set SATDOSE_NATIVE_SRC to the independently verified native source')
class ForkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(NATIVE)
        cls.ops = f.load_native_ops(root, {
            'mm_core/training.py': '43b027fb35710411d28a0309bd926c2b23172de4ae5fe7022dc2c66fb8377af2',
            'sr_f3_ce/training.py': 'fcca2f453f97814ee74e2da15339b016cd899310d9211aff6e247209755b0d94'})

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'legacy.pt'
        self.runtime = SimpleNamespace(model=TinyAdapter(), identity={'step': 0, 'experiment_id': 'OLD'})
        self.reference = self.ops.trainable_state(self.runtime.model)
        self.policy = {n: p + 0.25 for n, p in self.reference.items()}
        self.state = dict(schema=f.LEGACY_SCHEMA, identity=dict(parent='b0', train_seed=11031, arm='R'),
            step=24, reference_sha256=self.ops.state_hash(self.reference), policy=self.policy,
            optimizer={'POISON_OLD_ADAM': True}, scheduler={'last_epoch': 24})
        torch.save(self.state, self.path)
        self.config = dict(experiment_id=f.EXPERIMENT, parent='b0',
            precision={'expected_trainable_tensors':496}, start_bindings={'11031':dict(
                original_state_path=str(self.path), state_sha256=f.file_sha(self.path),
                policy_sha256=self.ops.state_hash(self.policy), reference_sha256=self.ops.state_hash(self.reference))})

    def fork(self, arm='T0'):
        return f.fork_runtime(self.runtime, self.config, 11031, arm, self.ops)

    def test_reference_precedes_policy_and_old_adam_ignored(self):
        base = self.runtime.model.base.detach().clone()
        out = self.fork()
        self.assertEqual(self.ops.state_hash(out.reference), self.ops.state_hash(self.reference))
        self.assertEqual(self.ops.state_hash(self.ops.trainable_state(self.runtime.model)), self.ops.state_hash(self.policy))
        self.assertFalse(out.optimizer.state)
        self.assertEqual(out.scheduler.last_epoch, 0)
        self.assertTrue(torch.equal(base, self.runtime.model.base))
        self.assertEqual(f.file_sha(self.path), self.config['start_bindings']['11031']['state_sha256'])
        self.assertTrue(all(not x.requires_grad for x in out.reference.values()))
        self.assertEqual(self.runtime.identity['arm'], 'T0')

    def test_readonly_fork_never_creates_optimizer(self):
        native = copy.copy(self.ops)
        def forbidden(_):
            raise AssertionError('readonly fork called optimizer factory')
        native.fresh_optimizer = forbidden
        out = f.fork_readonly(self.runtime, self.config, 11031, native)
        self.assertIsNone(out.optimizer)
        self.assertIsNone(out.scheduler)
        self.assertFalse(out.receipt['optimizer_created'])
        self.assertIsNone(out.receipt['optimizer_sha256'])
        self.assertEqual(self.runtime.identity['arm'], 'START')
        self.assertEqual(self.ops.state_hash(out.reference), self.ops.state_hash(self.reference))
        self.assertEqual(self.ops.state_hash(self.ops.trainable_state(self.runtime.model)), self.ops.state_hash(self.policy))

    def test_two_arms_identical_policy_distinct_identity(self):
        first = self.fork()
        self.runtime = SimpleNamespace(model=TinyAdapter(), identity={'step':0})
        second = self.fork('T1')
        self.assertEqual(first.receipt['policy_sha256'], second.receipt['policy_sha256'])
        self.assertNotEqual(first.receipt['arm'], second.receipt['arm'])

    def test_wrong_parent_fails_without_mutation(self):
        with torch.no_grad(): self.runtime.model.lora_fixture[0].add_(1)
        before = self.ops.state_hash(self.ops.trainable_state(self.runtime.model))
        with self.assertRaisesRegex(PermissionError, 'reference not loaded first'): self.fork()
        self.assertEqual(before, self.ops.state_hash(self.ops.trainable_state(self.runtime.model)))

    def test_changed_file_rejected(self):
        with self.path.open('ab') as stream: stream.write(b'changed')
        with self.assertRaisesRegex(PermissionError, 'checkpoint bytes differ'): self.fork()

    def test_wrong_source_identity_rejected(self):
        self.state['identity']['arm']='D'; torch.save(self.state,self.path)
        self.config['start_bindings']['11031']['state_sha256']=f.file_sha(self.path)
        with self.assertRaisesRegex(PermissionError,'path identity'): self.fork()

    def test_shape_error_precedes_mutation(self):
        key = next(iter(self.policy)); self.state['policy'][key]=torch.ones(2)
        torch.save(self.state,self.path);self.config['start_bindings']['11031']['state_sha256']=f.file_sha(self.path)
        with self.assertRaisesRegex(PermissionError,'dtype/shape'): self.fork()
        self.assertEqual(self.ops.state_hash(self.reference),self.ops.state_hash(self.ops.trainable_state(self.runtime.model)))

    def test_non_lora_trainable_rejected(self):
        self.runtime.model.base.requires_grad_(True)
        with self.assertRaisesRegex(PermissionError,'496-parameter'): self.fork()

    def test_failed_optimizer_rolls_back(self):
        bad = copy.copy(self.ops)
        def fail(_): raise RuntimeError('injected optimizer failure')
        bad.fresh_optimizer=fail
        with self.assertRaisesRegex(RuntimeError,'injected'):
            f.fork_runtime(self.runtime,self.config,11031,'T0',bad)
        self.assertEqual(self.ops.state_hash(self.reference),self.ops.state_hash(self.ops.trainable_state(self.runtime.model)))
        self.assertEqual(self.runtime.identity,{'step':0,'experiment_id':'OLD'})

    def checkpoint(self, out):
        # One tiny synthetic step exercises nonempty real Adam restoration only.
        for p in self.runtime.model.parameters():
            if p.requires_grad: p.grad = torch.ones_like(p)
        out.optimizer.step();out.scheduler.step();out.optimizer.zero_grad()
        identity=dict(experiment_id=f.EXPERIMENT,parent='b0',arm='T0',train_seed=11031,config_sha256='unit-only')
        rng=self.ops.capture_rng(); packed=copy.deepcopy(rng); n=packed['numpy']; packed['numpy']=(n[0],n[1].tolist(),*n[2:])
        state=dict(schema=f.SCHEMA,identity=identity,step=1,policy=self.ops.trainable_state(self.runtime.model),
            reference_sha256=self.ops.state_hash(out.reference),optimizer=out.optimizer.state_dict(),
            scheduler=out.scheduler.state_dict(),rng=packed,
            sampler_state=dict(committed_step=1,next_step=2,generated_responses=128))
        directory=Path(self.temp.name)/'step-001';directory.mkdir();torch.save(state,directory/'state.pt')
        marker=dict(schema=f.SCHEMA,identity=identity,step=1,state_sha256=f.file_sha(directory/'state.pt'),
            reference_sha256=state['reference_sha256'],sampler_state=state['sampler_state'],rng_sha256=self.ops.state_hash(rng))
        marker.update({key+'_sha256':self.ops.state_hash(state[key]) for key in ('policy','optimizer','scheduler')})
        (directory/'COMMIT.json').write_text(json.dumps(marker))
        return directory,identity,marker

    def test_exact_resume_restores_adam_scheduler_and_policy(self):
        out=self.fork(); directory,identity,marker=self.checkpoint(out)
        with torch.no_grad():
            for p in self.runtime.model.lora_fixture:p.add_(1)
        out.optimizer.state.clear()
        result=f.restore_committed(self.runtime,out.optimizer,out.scheduler,out.reference,self.ops,
            checkpoint_dir=directory,expected_commit_sha256=f.file_sha(directory/'COMMIT.json'),expected_identity=identity)
        self.assertEqual(result,marker)
        self.assertEqual(self.ops.state_hash(out.optimizer.state_dict()),marker['optimizer_sha256'])
        self.assertEqual(self.ops.state_hash(self.ops.trainable_state(self.runtime.model)),marker['policy_sha256'])
        self.assertEqual(self.runtime.identity['step'],1)

    def test_resume_cannot_cross_arm(self):
        out=self.fork(); directory,identity,_=self.checkpoint(out); identity['arm']='T1'
        with self.assertRaisesRegex(PermissionError,'identity mismatch'):
            f.restore_committed(self.runtime,out.optimizer,out.scheduler,out.reference,self.ops,
                checkpoint_dir=directory,expected_commit_sha256=f.file_sha(directory/'COMMIT.json'),expected_identity=identity)

    def test_resume_rejects_untrusted_commit_sha(self):
        out=self.fork(); directory,identity,_=self.checkpoint(out)
        with self.assertRaisesRegex(PermissionError,'COMMIT bytes differ'):
            f.restore_committed(self.runtime,out.optimizer,out.scheduler,out.reference,self.ops,
                checkpoint_dir=directory,expected_commit_sha256='0'*64,expected_identity=identity)


if __name__ == '__main__': unittest.main()
