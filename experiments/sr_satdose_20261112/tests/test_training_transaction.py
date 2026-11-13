"""Synthetic captures + real native tiny Adam/codec; no GPU/model forward calls."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import torch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from execution import training_transaction as t
from execution import training_state as state
sys.path.pop(0)
spec = importlib.util.spec_from_file_location('_transaction_state_fixture', Path(__file__).with_name('test_training_state.py'))
fixtures = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixtures)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(t.canonical(value)+b'\n')


def lines(path, values):
    path.write_bytes(b''.join(t.canonical(value)+b'\n' for value in values))


@unittest.skipUnless((fixtures.NATIVE/'sr_f3_ce/training.py').exists(), 'Pinned native source required')
class TrainingTransactionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(fixtures.TrainingStateTests.doClassCleanups)
        fixtures.TrainingStateTests.setUpClass()

    def setUp(self):
        self.fixture = fixtures.TrainingStateTests('test_real_two_step_full_restore_via_frozen_loader')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root; self.identity = self.fixture.identity
        self.calls = dict(collect=0, prepare=0, optimizer=0, scheduler=0, writer=0, storage=0)
        config = dict(experiment_id=t.EXPERIMENT, steps=24, groups_per_step=16, draws_per_group=8,
                      start_bindings={str(s): self.fixture.source for s in t.STARTS})
        dump(self.root/'config.json', config); dump(self.root/'source_lock.json', dict(synthetic_cpu_fixture=True))
        groups, manifest = [], []
        for start in t.STARTS:
            for arm in t.ARMS:
                for step in range(1,25):
                    for index in range(16):
                        group = f'{start}/{arm}/{step}/{index}'; root = f'{start}/{step}/{index}'
                        ids = [group+f'/{draw}' for draw in range(8)]
                        groups.append(dict(start_seed=start, arm=arm, step=step, group_id=group, root_id=root, request_ids=ids))
                        for draw, rid in enumerate(ids):
                            manifest.append(dict(request_id=rid, split='TRAIN', start_seed=start, arm=arm, step=step,
                                group_id=group, root_id=root, draw=draw, seed=step*1000+index*8+draw))
        lines(self.root/'train_manifest.jsonl', manifest); lines(self.root/'training_groups.jsonl', groups)
        self.freeze = self.root/'DESIGN_FREEZE.json'
        dump(self.freeze, dict(schema='sr-satdose-design-freeze-v1', experiment_id=t.EXPERIMENT,
            config_sha256=t.file_sha(self.root/'config.json'), artifacts={key:t.reference(self.root,self.root/name)
            for key,name in [('config','config.json'),('source_lock','source_lock.json'),
                             ('train_manifest','train_manifest.jsonl'),('training_groups','training_groups.jsonl')]}))
        self.freeze_sha = t.file_sha(self.freeze)
        self.genesis_reference = self.genesis(self.identity)
        self.store = t.TransactionStore(self.root, self.freeze, self.freeze_sha, self.identity,
                                       native=self.fixture.ops,genesis_reference=self.genesis_reference)

    def genesis(self, identity):
        source=identity['source_binding'];start=identity['train_seed'];arm=identity['arm']
        hashes=dict(optimizer_sha256=self.fixture.ops.state_hash(self.fixture.optimizer.state_dict()),
            scheduler_sha256=self.fixture.ops.state_hash(self.fixture.scheduler.state_dict()),
            rng_sha256=self.fixture.ops.state_hash(self.fixture.ops.capture_rng()))
        fork=dict(schema='sr-satdose-fork-v1',experiment_id=t.EXPERIMENT,start_seed=start,arm=arm,
            source_state_path=source['original_state_path'],source_state_sha256=source['state_sha256'],
            policy_sha256=source['policy_sha256'],reference_sha256=source['reference_sha256'],optimizer_created=True,
            source_optimizer_restored=False,trainable_tensors=496,model_calls=0,optimizer_steps=0,**hashes)
        value=dict(schema='sr-satdose-training-genesis-v1',status='FRESH_R24_FORK_NOT_TRAINING_ADMISSION',
            permission='NO_TRAINING_ADMISSION',identity=identity,design_freeze_sha256=self.freeze_sha,source_binding=source,
            endpoint=dict(experiment_id=t.EXPERIMENT,start_seed=start,arm=arm,step=0,
                policy_sha256=source['policy_sha256'],reference_sha256=source['reference_sha256']),fork_receipt=fork,
            optimizer_steps=0,scheduler_steps=0,parameter_updates=0,**hashes)
        path=self.root/f'genesis/{start}-{arm}.json';dump(path,value);return t.reference(self.root,path)

    def collect(self, stage, requests):
        self.calls['collect'] += 1
        step = requests[0]['step']; directory = self.root/f'captures/{step:03d}'; directory.mkdir(parents=True)
        raw = []; refs = []
        policy = self.fixture.ops.state_hash(self.fixture.ops.trainable_state(self.fixture.runtime.model))
        for request in requests:
            origin = dict(request, scope='TRAIN', generation_status='COMPLETE', technical_validation_errors=[],
                design_freeze_sha256=self.freeze_sha, policy_sha256=policy,
                reference_sha256=self.fixture.source['reference_sha256'], tokens=[3,4], raw_tokens=[3,4],
                completion_token_count=2, sampler_logprobs=[-.1,-.2],
                sampler_logprob_source='actual_generate_raw_logits_selected_token', finish_reason='balanced')
            path = directory/(hashlib.sha256(request['request_id'].encode()).hexdigest()+'.COMPLETE.json')
            dump(path, origin); refs.append(t.reference(self.root,path))
            raw.append(dict(origin,native_score={'J':1,'known_mask':{'S':1}},J=1,exact_authorised=1))
        lines(directory/'raw.jsonl', raw); raw_ref = t.reference(self.root,directory/'raw.jsonl')
        dump(directory/'scores.json', dict(schema='sr-satdose-training-scores-v1',identity=self.identity,step=step,
            raw=raw_ref,native_rescoring_complete=True,rows=[{k:row[k] for k in
            ('request_id','native_score','J','exact_authorised')} for row in raw]))
        dump(directory/'ROLLOUT_COMPLETE.json', dict(schema='sr-satdose-training-rollout-v1',
            status='COMPLETE_NOT_TRAINING_ADMISSION',identity=self.identity,step=step,responses=128,
            design_freeze_sha256=self.freeze_sha,source_lock_sha256=t.file_sha(self.root/'source_lock.json'),
            raw=raw_ref,complete_files=refs,scores=t.reference(self.root,directory/'scores.json')))
        return t.reference(self.root,directory/'ROLLOUT_COMPLETE.json')

    def prepare(self, stage, receipt):
        self.calls['prepare'] += 1
        step = t.read_json(self.root/receipt['path'])['step']; directory = self.root/f'captures/{step:03d}'
        for index, parameter in enumerate(self.fixture.runtime.model.parameters()):
            parameter.grad = torch.full_like(parameter, (index+1)/1000*(1 if step==1 else -.3))
        torch.nn.utils.clip_grad_norm_(self.fixture.runtime.model.parameters(),1.0,foreach=False)
        torch.save([p.grad for p in self.fixture.runtime.model.parameters()], directory/'postclip.pt')
        dump(directory/'precommit.json',dict(schema='sr-satdose-training-precommit-v1',identity=self.identity,
            step=step,rollout_receipt=receipt,gradient_stage='postclip',numerical_checks_passed=True,
            artifacts=[t.reference(self.root,directory/'postclip.pt')]))
        return t.reference(self.root,directory/'precommit.json')

    def storage(self, stage, step):
        self.calls['storage'] += 1
        path = self.root/f'captures/{step:03d}/quota.json'; dump(path,dict(synthetic_cpu_fixture=True))
        return dict(schema='sr-satdose-storage-readiness-v1',ready=True,quota_verified=True,writable_verified=True,
            required_bytes=10_000_000,effective_available_bytes=20_000_000,evidence=[t.reference(self.root,path)])

    def optimizer_step(self):
        self.calls['optimizer'] += 1
        intents = list(self.store.path.glob('.staging-*/UPDATE_INTENT.json'))
        self.assertEqual(len(intents),1); self.assertTrue((intents[0].parent/'STORAGE_READY.json').is_file())
        self.fixture.optimizer.step()

    def scheduler_step(self):
        self.calls['scheduler'] += 1
        self.fixture.scheduler.step(); self.fixture.optimizer.zero_grad(set_to_none=True)

    def writer(self, stage, fields):
        self.calls['writer'] += 1; step = t.read_json(stage/'UPDATE.json')['step']
        return state.write_checkpoint_files(stage,self.fixture.runtime,self.fixture.optimizer,self.fixture.scheduler,
            self.fixture.reference,self.fixture.ops,identity=self.identity,step=step,
            sampler_state=dict(committed_step=step,next_step=step+1,generated_responses=128*step,
                original_manifest_sha256=t.file_sha(self.root/'train_manifest.jsonl')),commit_fields=fields)

    def run_step(self, step=1, **overrides):
        callbacks = dict(collect_evidence=self.collect,prepare_update=self.prepare,optimizer_step=self.optimizer_step,
            scheduler_step=self.scheduler_step,write_checkpoint=self.writer,storage_ready=self.storage)
        callbacks.update(overrides)
        return self.store.commit_step(step,**callbacks)

    def test_real_adam_two_steps_resume_and_commit_idempotency(self):
        first = self.run_step(); calls = self.calls.copy()
        self.assertEqual(first['status'],'COMMITTED'); self.assertEqual(first['next_step'],2)
        self.assertFalse(any(self.store.path.glob('.staging*')))
        duplicate = self.run_step()
        self.assertEqual(duplicate['status'],'ALREADY_COMMITTED'); self.assertEqual(self.calls,calls)
        other,opt,sched = self.fixture.make_runtime()
        state.restore_checkpoint(other,opt,sched,self.fixture.reference,self.fixture.ops,
            checkpoint_dir=Path(first['checkpoint']),expected_commit_sha256=first['commit_sha256'],expected_identity=self.identity)
        self.fixture.runtime,self.fixture.optimizer,self.fixture.scheduler = other,opt,sched
        second = self.run_step(2)
        self.assertEqual(second['next_step'],3)
        self.assertTrue(all(float(v['step'])==2 for v in opt.state.values()))
        commit = t.read_json(Path(second['checkpoint'])/'COMMIT.json')
        self.assertEqual(commit['transaction']['previous_commit_sha256'],first['commit_sha256'])
        self.assertEqual(commit['sampler_state']['generated_responses'],256)
        self.assertEqual(second['permission'],'NO_TRAINING_ADMISSION')
        calls = self.calls.copy(); again = self.run_step(1)
        self.assertEqual(again['next_step'],3); self.assertEqual(self.calls,calls)

    def test_all_fault_boundaries_never_retry_unknown_update(self):
        boundaries = ('after_step_intent','after_rollout_validation','after_evidence','after_precommit',
            'after_storage_check','after_update_intent','after_optimizer_step','after_scheduler_step',
            'after_update','after_checkpoint_write','after_state_validation','after_stage_fsync','after_rename','after_parent_fsync')
        # Isolated full native state and filesystem per failure; no production data.
        for boundary in boundaries:
            with self.subTest(boundary=boundary):
                case = type(self)('test_real_adam_two_steps_resume_and_commit_idempotency'); case.setUp()
                try:
                    def fault(name):
                        if name==boundary: raise RuntimeError('injected '+name)
                    with self.assertRaisesRegex(RuntimeError,'injected'):
                        case.run_step(fault=fault)
                    before = case.calls.copy()
                    if boundary in ('after_rename','after_parent_fsync'):
                        result=case.run_step(); self.assertEqual(result['status'],'ALREADY_COMMITTED')
                    else:
                        with self.assertRaisesRegex(PermissionError,'occupied staging'):
                            case.run_step()
                        self.assertTrue((case.store.path/'.staging-step-001').is_dir())
                    self.assertEqual(case.calls,before)
                finally:
                    case.doCleanups()

    def test_storage_not_ready_blocks_before_update_intent_and_adam(self):
        def insufficient(stage,step):
            result=self.storage(stage,step);result['effective_available_bytes']=result['required_bytes']-1;return result
        with self.assertRaisesRegex(PermissionError,'actual quota'):
            self.run_step(storage_ready=insufficient)
        self.assertEqual(self.calls['optimizer'],0)
        self.assertFalse((self.store.path/'.staging-step-001/UPDATE_INTENT.json').exists())
        with self.assertRaisesRegex(PermissionError,'occupied staging'):
            self.run_step()

    def test_lock_and_step_budget_prevent_callbacks(self):
        with t.exclusive_lock(self.store.store):
            with self.assertRaisesRegex(PermissionError,'holds the lock'):
                self.run_step()
        for step in (0,25,True,2):
            with self.assertRaises(PermissionError): self.run_step(step)
        self.assertEqual(sum(self.calls.values()),0)

    def test_unknown_other_arm_begin_blocks_all_updates(self):
        directory=self.store.store/'start-11032/T1';directory.mkdir(parents=True)
        dump(directory/'BEGIN.json',dict(unknown=True))
        with self.assertRaisesRegex(PermissionError,'occupied staging'):
            self.run_step()
        self.assertEqual(sum(self.calls.values()),0)

    def test_foreign_identity_and_incomplete_g8_or_substituted_request_rejected(self):
        for field,value in [('train_seed',11033),('arm','R'),('experiment_id','OLD')]:
            identity=copy.deepcopy(self.identity);identity[field]=value
            with self.assertRaisesRegex(PermissionError,'foreign training identity'):
                t.TransactionStore(self.root,self.freeze,self.freeze_sha,identity,native=self.fixture.ops,
                                   genesis_reference=self.genesis_reference)
        for change in ('foreign_start','foreign_arm','missing','request','tokens'):
            with self.subTest(change=change):
                case=type(self)('test_real_adam_two_steps_resume_and_commit_idempotency');case.setUp()
                try:
                    def corrupt(stage,requests):
                        ref=case.collect(stage,requests);receipt=t.read_json(case.root/ref['path'])
                        raw_path=case.root/receipt['raw']['path'];raw=t.rows(raw_path)
                        if change=='foreign_start':raw[0]['start_seed']=11032
                        elif change=='foreign_arm':raw[0]['arm']='T1'
                        elif change=='missing':raw.pop()
                        elif change=='request':raw[0]['request_id']='not-original'
                        else:raw[0]['tokens']=[99,100]
                        lines(raw_path,raw);receipt['raw']=t.reference(case.root,raw_path)
                        dump(case.root/ref['path'],receipt);return t.reference(case.root,case.root/ref['path'])
                    with self.assertRaises(PermissionError):case.run_step(collect_evidence=corrupt)
                    self.assertEqual(case.calls['optimizer'],0)
                finally:case.doCleanups()

    def test_committed_state_or_evidence_tamper_prevents_idempotent_success(self):
        first=self.run_step();before=self.calls.copy()
        path=Path(first['checkpoint'])/'UPDATE.json';value=t.read_json(path);value['optimizer_callback_invocations']=2;dump(path,value)
        with self.assertRaisesRegex(PermissionError,'artifact SHA'):
            self.run_step()
        self.assertEqual(self.calls,before)

    def test_callback_failure_and_lease_reuse_never_steps_twice(self):
        def fail():
            self.optimizer_step();raise RuntimeError('native Adam callback interrupted')
        with self.assertRaisesRegex(RuntimeError,'interrupted'):
            self.run_step(optimizer_step=fail)
        self.assertEqual(self.calls['optimizer'],1);self.assertEqual(self.calls['scheduler'],0)
        with self.assertRaisesRegex(PermissionError,'occupied staging'):
            self.run_step()
        stage=self.store.path/'.staging-step-001';lease=t._UpdateLease(stage,self.identity,1)
        # Independently exercising the per-process lease with harmless callbacks;
        # durable recovery is blocked by the store, not this internal constructor.
        calls=[];lease.consume(lambda:calls.append('adam'),lambda:calls.append('schedule'),lambda _:None)
        with self.assertRaisesRegex(PermissionError,'already consumed'):
            lease.consume(lambda:calls.append('adam'),lambda:calls.append('schedule'),lambda _:None)
        self.assertEqual(calls,['adam','schedule'])

    def test_invalid_native_update_count_cannot_publish(self):
        def double_native_step():
            self.optimizer_step();self.fixture.optimizer.step()
        with self.assertRaises(PermissionError):
            self.run_step(optimizer_step=double_native_step)
        self.assertFalse((self.store.path/'step-001').exists())
        self.assertTrue((self.store.path/'.staging-step-001/UPDATE_INTENT.json').exists())
        with self.assertRaisesRegex(PermissionError,'occupied staging'):
            self.run_step()

    def test_unknown_raw_begin_is_rejected_before_update(self):
        def with_unknown(stage, requests):
            ref=self.collect(stage,requests)
            dump((self.root/ref['path']).parent/'unknown.BEGIN.json',dict(request_id='foreign'))
            return ref
        with self.assertRaisesRegex(PermissionError,'unknown rollout BEGIN'):
            self.run_step(collect_evidence=with_unknown)
        self.assertEqual(self.calls['optimizer'],0)

    def test_other_published_path_bad_state_or_broken_chain_blocks_callbacks(self):
        for damage in ('state','chain'):
            with self.subTest(damage=damage):
                case=type(self)('test_real_adam_two_steps_resume_and_commit_idempotency');case.setUp()
                try:
                    original_store=case.store;original_identity=case.identity
                    other=copy.deepcopy(case.identity);other['arm']='T1';other['run_id']=t.EXPERIMENT+'/b0-11031-T1'
                    case.identity=other;case.fixture.runtime.identity.update(arm='T1',run_id=other['run_id'])
                    case.store=t.TransactionStore(case.root,case.freeze,case.freeze_sha,other,native=case.fixture.ops,
                        genesis_reference=case.genesis(other))
                    published=case.run_step();directory=Path(published['checkpoint'])
                    case.store=original_store;case.identity=original_identity
                    if damage=='state':
                        with (directory/'state.pt').open('ab') as stream:stream.write(b'corrupt')
                    else:
                        marker=t.read_json(directory/'COMMIT.json');marker['transaction']['previous_commit_sha256']='f'*64
                        dump(directory/'COMMIT.json',marker)
                    before=case.calls.copy()
                    with self.assertRaisesRegex(PermissionError,'state changed|broken COMMIT'):
                        case.run_step()
                    self.assertEqual(case.calls,before)
                finally:case.doCleanups()

    def test_inspect_committed_binds_sha_and_runs_no_update_callbacks(self):
        result=self.run_step();before=self.calls.copy()
        marker=self.store.inspect_committed(1,result['commit_sha256'])
        self.assertEqual(marker['step'],1);self.assertEqual(self.calls,before)
        marker['identity']['arm']='T1'
        self.assertEqual(self.store.inspect_committed(1,result['commit_sha256'])['identity'],self.identity)
        with self.assertRaisesRegex(PermissionError,'SHA differs'):
            self.store.inspect_committed(1,'0'*64)


if __name__=='__main__':unittest.main()
