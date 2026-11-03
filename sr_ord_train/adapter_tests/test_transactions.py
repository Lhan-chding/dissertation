"""Synthetic persistence fixtures; never real model evidence."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from adapter.transactions import PathTransactions, write_once

IDENTITY = dict(experiment_id='SR-ORD-TRAIN-1', parent='b0', arm='R', train_seed=11031,
                run_id='SYNTHETIC/b0/R/11031', manifest_sha256='a'*64,
                plan_sha256='b'*64, source_sha256='c'*64)

def request(i=0):
    return dict(request_id=f'synthetic-{i}', seed=i+7, parent='b0', arm='R', train_seed=11031,
                step=1, prompt='synthetic public prompt', image_sha256='f'*64)

def raw(i=0):
    return dict(request_id=f'synthetic-{i}', seed=i+7, policy_hash='d'*64,
                tokens=[17, 18], sampler_logprobs=[-1.0, -2.0], full_text='{}')

class FileTransactionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name).resolve() / 'path'
    def tearDown(self):
        self.tmp.cleanup()
    def synthetic_zero(self, tx):
        tx._markers = [dict(step=0, policy_sha256='d'*64)]
    def test_lock_identity_and_write_once(self):
        with PathTransactions(self.path, IDENTITY):
            with self.assertRaises(BlockingIOError):
                with PathTransactions(self.path, IDENTITY): pass
            with self.assertRaises(FileExistsError):
                write_once(self.path / 'IDENTITY.json', IDENTITY)
        with self.assertRaises(PermissionError):
            with PathTransactions(self.path, dict(IDENTITY, arm='G')): pass
    def test_complete_reuse_exact_identity_and_raw_bytes(self):
        with PathTransactions(self.path, IDENTITY) as tx:
            self.synthetic_zero(tx)
            begin = tx.begin_attempt(request(), 'd'*64, 1)
            self.assertEqual(begin['status'], 'NEW')
            tx.complete_attempt(begin['attempt_id'], raw())
            reused = tx.begin_attempt(request(), 'd'*64, 1)
            self.assertEqual(reused['status'], 'COMPLETE')
            self.assertEqual(reused['raw'], raw())
            with self.assertRaises(PermissionError): tx.complete_attempt(begin['attempt_id'], raw())
            with self.assertRaises(PermissionError): tx.begin_attempt(request(), 'e'*64, 1)
            with self.assertRaises(PermissionError): tx.begin_attempt(dict(request(), prompt='changed'), 'd'*64, 1)
        with PathTransactions(self.path, IDENTITY) as tx:
            self.assertEqual(tx.begin_attempt(request(), 'd'*64, 1)['raw'], raw())
        p = self.path / 'attempts' / (begin['attempt_id'] + '.json')
        p.write_text('{}\n')
        with self.assertRaises(PermissionError):
            with PathTransactions(self.path, IDENTITY): pass
    def test_unknown_blocks_retry_and_other_requests(self):
        with PathTransactions(self.path, IDENTITY) as tx:
            self.synthetic_zero(tx)
            tx.begin_attempt(request(), 'd'*64, 1)
        with PathTransactions(self.path, IDENTITY) as tx:
            with self.assertRaisesRegex(PermissionError, 'UNKNOWN'): tx.begin_attempt(request(), 'd'*64, 1)
            with self.assertRaisesRegex(PermissionError, 'UNKNOWN'): tx.begin_attempt(request(1), 'd'*64, 1)
    def test_raw_identity_and_sampler_must_match(self):
        with PathTransactions(self.path, IDENTITY) as tx:
            self.synthetic_zero(tx)
            aid = tx.begin_attempt(request(), 'd'*64, 1)['attempt_id']
            for bad in [dict(raw(), seed=99), dict(raw(), policy_hash='e'*64), dict(raw(), tokens=[]),
                        dict(raw(), sampler_logprobs=[float('nan'), 0])]:
                with self.assertRaises(PermissionError): tx.complete_attempt(aid, bad)
            tx.complete_attempt(aid, raw())
    def test_staging_and_partial_journal_fail_closed(self):
        with PathTransactions(self.path, IDENTITY) as tx:
            (self.path / 'checkpoints/.staging-crash').mkdir()
            with self.assertRaisesRegex(PermissionError, 'staging'): tx.published()
        (self.path / 'ATTEMPTS.jsonl').write_text('{"unfinished":')
        with self.assertRaisesRegex(PermissionError, 'Partial attempt'):
            with PathTransactions(self.path, IDENTITY): pass
    def test_symlink_and_unlocked_access_rejected(self):
        actual = Path(self.tmp.name).resolve() / 'actual'
        actual.mkdir()
        self.path.symlink_to(actual, target_is_directory=True)
        with self.assertRaises(PermissionError): PathTransactions(self.path, IDENTITY)
        tx = PathTransactions(actual, IDENTITY)
        with self.assertRaises(PermissionError): tx.published()
    def test_journal_chain_rejects_removed_event(self):
        with PathTransactions(self.path, IDENTITY) as tx:
            self.synthetic_zero(tx)
            aid = tx.begin_attempt(request(), 'd'*64, 1)['attempt_id']
            tx.complete_attempt(aid, raw())
        p = self.path / 'ATTEMPTS.jsonl'
        p.write_bytes(p.read_bytes().splitlines(keepends=True)[1])
        with self.assertRaises(PermissionError):
            with PathTransactions(self.path, IDENTITY): pass

@unittest.skipUnless(importlib.util.find_spec('torch'), 'torch unavailable; requires bound server environment')
class TorchTransactionTests(unittest.TestCase):
    def test_checkpoint_roundtrip_and_tamper(self):
        import torch
        from types import SimpleNamespace
        from mm_core.training import trainable_state, state_hash
        from sr_f3_ce.training import fresh_optimizer
        model = torch.nn.Linear(2, 1).float()
        optimizer, scheduler = fresh_optimizer(model)
        reference = trainable_state(model)
        runtime = SimpleNamespace(model=model, identity=dict(IDENTITY, parent_id='b0',
                                  reference_hash=state_hash(reference), step=0))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp).resolve() / 'training'
            with PathTransactions(path, IDENTITY) as tx:
                zero = tx.commit(runtime, optimizer, scheduler, reference, step=0,
                                 sampler_state=dict(next_step=1, committed_step=0, generated_responses=0))
                self.assertEqual(zero['policy_sha256'], state_hash(reference))
            with torch.no_grad():
                for p in model.parameters(): p.add_(1)
            with PathTransactions(path, IDENTITY) as tx:
                restored = tx.restore_latest(runtime, optimizer, scheduler, reference)
                self.assertEqual(restored['step'], 0)
                self.assertEqual(state_hash(trainable_state(model)), state_hash(reference))
            # One synthetic complete 128-response transaction and Adam roundtrip.
            with PathTransactions(path, IDENTITY) as tx:
                tx.restore_latest(runtime, optimizer, scheduler, reference)
                policy_before = state_hash(trainable_state(model))
                for i in range(128):
                    begin = tx.begin_attempt(request(i), policy_before, 1)
                    tx.complete_attempt(begin['attempt_id'], dict(raw(i), policy_hash=policy_before))
                model(torch.ones(1, 2)).sum().backward()
                optimizer.step()
                scheduler.step()
                update = dict(status='UPDATED_UNCOMMITTED', precommit_gate='PASS_BEFORE_ADAM',
                              optimizer_updates=1, sequence_count=128, step=1,
                              policy_before_sha256=policy_before,
                              policy_after_sha256=state_hash(trainable_state(model)),
                              reference_sha256=state_hash(reference),
                              numerics=dict(sampler_train_mean_absolute_difference=0.,
                                            sampler_train_max_absolute_difference=0.,
                                            tis_truncated_fraction=0., kl_unweighted=0.,
                                            gradient_norm_before_clip=1., gradient_norm_after_clip=1.))
                one = tx.commit(runtime, optimizer, scheduler, reference, step=1,
                                sampler_state=dict(next_step=2, committed_step=1, generated_responses=128),
                                update=update)
                self.assertEqual(len(one['attempts']), 128)
                self.assertEqual(runtime.identity['step'], 0)
            with PathTransactions(path, IDENTITY) as tx:
                restored = tx.restore_latest(runtime, optimizer, scheduler, reference)
                self.assertEqual(restored['step'], 1)
                self.assertEqual(runtime.identity['step'], 1)
                with self.assertRaises(PermissionError):
                    tx.commit(runtime, optimizer, scheduler, reference, step=1,
                              sampler_state=dict(next_step=2, committed_step=1, generated_responses=128), update=update)
            state_path = path / 'checkpoints/step-000/state.pt'
            with state_path.open('ab') as f: f.write(b'tamper')
            with PathTransactions(path, IDENTITY) as tx:
                with self.assertRaises(PermissionError): tx.published()

if __name__ == '__main__': unittest.main()
