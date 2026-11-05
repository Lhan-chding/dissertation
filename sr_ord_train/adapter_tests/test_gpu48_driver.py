"""CPU orchestration regressions; no model, generation or optimizer execution."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from adapter.backend import file_sha, object_sha
from adapter.training import canonical_hash
from gpu_production_v2 import driver


class StopFixture(BaseException):
    """Stop after the first committed recovery update, before later sampling."""


class GPU48ProductionDriverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.package = Path(self.temp.name).resolve()
        self.name = 'b0-11031-G'
        self.stage = self.package / 'run/paths' / self.name
        self.execution = self.stage / 'executions/test-recovery'
        self.execution.mkdir(parents=True)
        plan_root = self.package / 'frozen/receipt_plans'
        (plan_root / 'paths').mkdir(parents=True)
        self.plan = [dict(step=step, groups=[dict(request_ids=[f's{step}/g{group}/d{draw}'
                     for draw in range(8)]) for group in range(16)]) for step in range(1, 25)]
        plan_path = plan_root / 'paths' / (self.name + '.json')
        plan_path.write_text(json.dumps(self.plan))
        (plan_root / 'INDEX.json').write_text(json.dumps({'files': {'paths/' + self.name + '.json': file_sha(plan_path)}}))
        self.policy, self.reference = 'a' * 64, 'b' * 64
        self.marker = dict(step=8, policy_sha256=self.policy, reference_sha256=self.reference)
        self.rows, self.requests, self.attempts = {}, {}, {}
        for step in range(1, 25):
            for group in self.plan[step - 1]['groups']:
                for rid in group['request_ids']:
                    self.requests[rid] = dict(request_id=rid, step=step, seed=len(self.requests))
                    if step == 9:
                        self.rows[rid] = dict(row_id=rid, tokens=[1, 2], sampler_logprobs=[-1., -2.], token_path_sha256='c' * 64)
                        self.attempts[rid] = dict(begin=dict(request=self.requests[rid], step=9,
                                                           policy_sha256=self.policy), complete={'saved': True}, raw=self.rows[rid])
        summary = dict(native_audit=dict(step=9, policy_before_sha256=self.policy,
            reference_sha256=self.reference, plan_step_sha256=canonical_hash(self.plan[8]),
            probability_vectors=list(copy.deepcopy(self.rows).values())))
        summary_path = self.package / 'diagnostic_summary.json'
        summary_path.write_text(json.dumps(summary))
        acceptance_path = self.package / 'acceptance.json'
        acceptance_path.write_text(json.dumps({'diagnostic_summary': {'path': str(summary_path), 'sha256': file_sha(summary_path)}}))
        self.original_binding = dict(implementation_files=[{'path': 'original.py', 'sha256': 'd' * 64}], code_commit='e' * 40)
        self.binding = dict(manifest_sha256='f' * 64, source_sha256='1' * 64,
            gpu_production={'acceptance': {'path': str(acceptance_path), 'sha256': file_sha(acceptance_path)}})
        runtime = SimpleNamespace(identity=dict(arm='G', train_seed=11031, run_id='SR-ORD-TRAIN-1/' + self.name, step=8))
        self.backend = SimpleNamespace(runtime=runtime, parent='b0', binding=self.binding, run_dir=self.execution,
            optimizer=object(), scheduler=object(), reference=object(), requests=self.requests,
            policy_hash=lambda: self.policy, generate_one=Mock(side_effect=AssertionError('Unexpected new generation')),
            close=Mock(side_effect=AssertionError('Fixture stopped before path completion')))
        self.begin_calls, self.commit_calls, self.provenance_calls = [], [], []
        owner = self

        class FakeTransactions:
            def __init__(self, directory, identity):
                self.directory, self.identity, self._attempts = directory, identity, owner.attempts
                owner.tx = self
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def _verify_raw(self, attempt): return copy.deepcopy(attempt['raw'])
            def begin_attempt(self, request, policy_hash, step):
                owner.begin_calls.append((request, policy_hash, step))
                if step != 9:
                    raise StopFixture()
                return dict(status='COMPLETE', raw=copy.deepcopy(owner.rows[request['request_id']]))
            def complete_attempt(self, *args):
                raise AssertionError('Reused original rows must not be completed again')
            def commit(self, runtime, optimizer, scheduler, reference, **kwargs):
                owner.commit_calls.append(copy.deepcopy(kwargs))
                return dict(step=kwargs['step'], policy_sha256='2' * 64, reference_sha256=owner.reference)
        self.fake_transactions = FakeTransactions

        def provenance(binding, directory, step):
            self.assertIs(binding, self.binding)
            self.assertEqual(directory, self.execution)
            self.assertIsInstance(directory, Path)
            self.provenance_calls.append((directory, step))
            return dict(step=step, execution_binding={'path': str(directory / 'GPU_EXECUTION_BINDING.json'), 'sha256': '3' * 64})
        self.provenance = provenance
        self.update = Mock(return_value={'status': 'UPDATED_UNCOMMITTED'})

    def run_driver(self, marker='default'):
        restored = self.marker if marker == 'default' else marker
        with ExitStack() as stack:
            stack.enter_context(patch.object(driver, 'PACKAGE', self.package))
            stack.enter_context(patch('adapter.transactions.PathTransactions', self.fake_transactions))
            stack.enter_context(patch.object(driver, 'verify_production_binding', return_value=self.original_binding))
            stack.enter_context(patch.object(driver, 'restore_latest_gpu', return_value=restored))
            stack.enter_context(patch.object(driver, 'execution_provenance', side_effect=self.provenance))
            stack.enter_context(patch.object(driver, 'production_update', self.update))
            stack.enter_context(patch.object(driver, 'score', return_value={'J': 0}))
            stack.enter_context(patch('builtins.print'))
            driver.train_path(self.backend, self.stage, self.name, {})

    def test_original128_reused_once_and_commit_binds_execution_directory(self):
        with self.assertRaises(StopFixture):
            self.run_driver()
        self.backend.generate_one.assert_not_called()
        self.update.assert_called_once()
        records = self.update.call_args.args[3]
        expected = [rid for group in self.plan[8]['groups'] for rid in group['request_ids']]
        self.assertEqual([row['row_id'] for row in records], expected)
        self.assertEqual(len(self.begin_calls), 129)  # 128 reused, then blocked at first step10 attempt.
        self.assertEqual(len(self.commit_calls), 1)
        commit = self.commit_calls[0]
        self.assertEqual(commit['step'], 9)
        self.assertEqual(commit['sampler_state']['generated_responses'], 1152)
        self.assertEqual(commit['sampler_state']['gpu_execution_provenance'], commit['update']['gpu_execution_provenance'])
        self.assertEqual(self.provenance_calls, [(self.execution, 9)])
        self.assertEqual(self.backend.runtime.identity['step'], 9)
        self.assertEqual(self.tx.identity['implementation_sha256'], object_sha(self.original_binding['implementation_files']))
        self.assertEqual(self.tx.identity['code_commit'], self.original_binding['code_commit'])

    def assert_recovery_denied_before_sampling(self):
        with self.assertRaises(PermissionError):
            self.run_driver()
        self.assertEqual(self.begin_calls, [])
        self.assertEqual(self.commit_calls, [])
        self.backend.generate_one.assert_not_called()
        self.update.assert_not_called()

    def test_missing_original_blocks_before_new_begin_or_generation(self):
        self.attempts.pop(next(iter(self.attempts)))
        self.assert_recovery_denied_before_sampling()

    def test_unknown_original_blocks_before_new_begin_or_generation(self):
        next(iter(self.attempts.values())).pop('complete')
        self.assert_recovery_denied_before_sampling()

    def test_changed_original_tokens_block_before_sampling(self):
        next(iter(self.attempts.values()))['raw']['tokens'] = [9, 9]
        self.assert_recovery_denied_before_sampling()

    def test_changed_source_policy_blocks_before_sampling(self):
        self.marker['policy_sha256'] = '4' * 64
        self.assert_recovery_denied_before_sampling()

    def test_changed_acceptance_bytes_block_before_sampling(self):
        path = Path(self.binding['gpu_production']['acceptance']['path'])
        path.write_text(path.read_text() + '\n')
        self.assert_recovery_denied_before_sampling()

    def test_new_path_step_zero_binds_execution_directory_before_generation(self):
        self.backend.runtime.identity['step'] = 0
        self.attempts.clear()
        with self.assertRaises(StopFixture):
            self.run_driver(marker=None)
        self.assertEqual(self.provenance_calls, [(self.execution, 0)])
        self.assertEqual(len(self.commit_calls), 1)
        self.assertEqual(self.commit_calls[0]['step'], 0)
        self.assertEqual(self.commit_calls[0]['sampler_state']['generated_responses'], 0)
        self.backend.generate_one.assert_not_called()
        self.update.assert_not_called()


if __name__ == '__main__':
    unittest.main()
