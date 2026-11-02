"""CPU journal fixtures are not model evidence and cannot produce smoke seals."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from order_probe_server import driver as d


def identity(config, parent):
    return dict(parent=parent, snapshot=config['model']['snapshot'],
                model_path=config['model']['model_path'], adapter_path=config['model']['parents'][parent],
                parameter_dtypes=['float32'], buffer_dtypes=['float32'], trainable_parameters=[],
                optimizer=None, optimizer_updates=0, parent_hashes_before={'adapter': 'frozen'},
                parent_hashes_after={'adapter': 'frozen'}, policy_tensor_hash='tensor-frozen')


def response(req, config):
    return {key: req[key] for key in ('request_id', 'seed', 'parent', 'assistant_prefill',
            'prompt_sha256', 'prefill_sha256', 'image_sha256', 'split')} | dict(
        suffix_text='"answer":"nonsense"}', full_text=req['assistant_prefill'] + '"answer":"nonsense"}',
        finish_reason='balanced', prompt_input_ids=[1, 2], suffix_token_ids=[3], generated_tokens=1,
        prefill_boundary={'same_assistant_message': True, 'prefill_token_ids': [2]},
        sampling=config['generation'], technical_validation_errors=[], optimizer_updates=0,
        model_identity=identity(config, req['parent']))


class FakeBackend:
    def __init__(self, config, fail=False):
        self.config, self.fail, self.calls = config, fail, []

    def generate_one(self, request):
        self.calls.append(copy.deepcopy(request))
        if self.fail:
            raise OSError('fixture interrupted transport')
        return response(request, self.config)


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = d.read(ROOT / 'config/plan.json')
        self.req = next(r for r in d.requests_for(ROOT / 'data', 'SMOKE') if r['condition'] == 'C0')

    def tearDown(self):
        self.temporary.cleanup()

    def test_freeze_and_reopen(self):
        d.prepare(ROOT / 'config/plan.json', ROOT / 'data', self.root)
        root, inputs, config = d.load_context(self.root)
        self.assertEqual(config, self.config)
        self.assertEqual(inputs, ROOT / 'data')
        self.assertEqual(root, self.root.resolve())

    def test_shards_are_complete_disjoint_ordered(self):
        for split in ('SMOKE', 'MAIN'):
            whole = d.requests_for(ROOT / 'data', split, 'b0')
            a = d.requests_for(ROOT / 'data', split, 'b0', 0, 2)
            b = d.requests_for(ROOT / 'data', split, 'b0', 1, 2)
            self.assertEqual(a, whole[::2])
            self.assertEqual(b, whole[1::2])
            self.assertEqual({r['request_id'] for r in a} & {r['request_id'] for r in b}, set())
            self.assertEqual(len(a) + len(b), len(whole))

    def test_saved_invalid_model_answer_is_never_resampled(self):
        backend = FakeBackend(self.config)
        self.assertEqual(d.execute_request(self.root, self.req, self.config, backend, 'test'), 'GENERATED')
        self.assertEqual(d.execute_request(self.root, self.req, self.config, backend, 'test'), 'REUSED')
        self.assertEqual(len(backend.calls), 1)
        self.assertIn('nonsense', d.read(d.result_path(self.root, self.req))['full_text'])

    def test_one_same_seed_infrastructure_retry(self):
        failed = FakeBackend(self.config, fail=True)
        with self.assertRaises(OSError):
            d.execute_request(self.root, self.req, self.config, failed, 'first')
        backend = FakeBackend(self.config)
        d.execute_request(self.root, self.req, self.config, backend, 'second')
        self.assertEqual(failed.calls, backend.calls)
        self.assertEqual(d.read(d.result_path(self.root, self.req))['attempt'], 2)

    def test_second_failure_exhausts_retry(self):
        failed = FakeBackend(self.config, fail=True)
        for _ in range(2):
            with self.assertRaises(OSError):
                d.execute_request(self.root, self.req, self.config, failed, 'test')
        with self.assertRaisesRegex(RuntimeError, 'retry exhausted'):
            d.execute_request(self.root, self.req, self.config, failed, 'third')
        self.assertEqual(len(failed.calls), 2)

    def test_interrupted_started_attempt_counts_against_budget(self):
        journal = self.root / 'ATTEMPTS_BY_ID' / (d.text_sha(self.req['request_id']) + '.jsonl')
        d.append(journal, {'event': 'START', 'seed': self.req['seed'], 'attempt': 1})
        backend = FakeBackend(self.config)
        d.execute_request(self.root, self.req, self.config, backend, 'test')
        self.assertEqual(d.read(d.result_path(self.root, self.req))['attempt'], 2)

    def test_technical_failure_preserved_and_not_resampled(self):
        backend = FakeBackend(self.config)
        bad = response(self.req, self.config)
        bad['prefill_boundary']['same_assistant_message'] = False
        with patch.object(backend, 'generate_one', return_value=bad) as generate:
            for _ in range(2):
                with self.assertRaises(RuntimeError):
                    d.execute_request(self.root, self.req, self.config, backend, 'test')
            self.assertEqual(generate.call_count, 1)
        self.assertTrue(d.result_path(self.root, self.req).is_file())

    def test_duplicate_publication_does_not_replace(self):
        path = self.root / 'one.json'
        d.immutable(path, {'value': 1})
        d.immutable(path, {'value': 1})
        with self.assertRaises(FileExistsError):
            d.immutable(path, {'value': 2})
        self.assertEqual(d.read(path), {'value': 1})

    def test_premature_prefix_stop_rejected(self):
        req = next(r for r in d.requests_for(ROOT / 'data', 'SMOKE') if r['branch'] == 'PREFIX')
        row = response(req, self.config)
        row['suffix_text'] = '"bindings":[{"category":"Book"}'
        row['full_text'] = req['assistant_prefill'] + row['suffix_text']
        self.assertIn('premature_balanced_stop', d.validate_record(row, req, self.config))

    def test_wrong_parent_and_seed_rejected(self):
        row = response(self.req, self.config)
        row['seed'] += 1
        row['model_identity']['adapter_path'] = 'other'
        errors = d.validate_record(row, self.req, self.config)
        self.assertIn('seed', errors)
        self.assertIn('identity:adapter_path', errors)

    def test_main_is_not_sealed_by_boolean_receipt(self):
        d.immutable(self.root / 'EXECUTION/SMOKE_RECEIPT.json', {'status': 'SMOKE_TECHNICAL_PASS'})
        with self.assertRaises((FileNotFoundError, RuntimeError)):
            d.verify_smoke(self.root)


    def test_complete_collection_requires_all_worker_closures(self):
        d.prepare(ROOT / 'config/plan.json', ROOT / 'data', self.root)
        requests = d.requests_for(ROOT / 'data', 'SMOKE')
        for req in requests:
            row = response(req, self.config)
            # Give prefix requests a closed top-level object while retaining their forced facts.
            row.update(worker_id=req['parent'], request_sha256=d.digest(req))
            d.immutable(d.result_path(self.root, req), row)
        for parent in self.config['model']['parents']:
            ident = identity(self.config, parent)
            d.immutable(self.root / 'WORKERS' / parent / 'START.json', {'parent': parent, 'identity': ident})
            closed = dict(ident, policy_tensor_hash_after='tensor-frozen', end_model_unchanged=True)
            if parent != 'b2':
                d.immutable(self.root / 'WORKERS' / parent / 'END.json', {'parent': parent, 'identity': closed, 'status': 'CLOSED_READONLY'})
        with self.assertRaisesRegex(RuntimeError, 'resume reconciliation'):
            d.collect(self.root, 'SMOKE')
        ident = identity(self.config, 'b2')
        closed = dict(ident, policy_tensor_hash_after='tensor-frozen', end_model_unchanged=True)
        d.immutable(self.root / 'WORKERS/b2/END.json', {'parent': 'b2', 'identity': closed, 'status': 'CLOSED_READONLY'})
        receipt = d.collect(self.root, 'SMOKE')
        self.assertEqual(receipt['requests'], 216)
        self.assertEqual(len(d.read_jsonl(self.root / 'RAW/SMOKE.jsonl')), 216)
        self.assertEqual(d.collect(self.root, 'SMOKE'), receipt)

    def test_incomplete_smoke_cannot_collect(self):
        d.prepare(ROOT / 'config/plan.json', ROOT / 'data', self.root)
        with self.assertRaisesRegex(RuntimeError, 'incomplete/extra'):
            d.collect(self.root, 'SMOKE')


    def test_interrupted_worker_recovery_preserves_unknown_terminal_state(self):
        backend = FakeBackend(self.config)
        old_identity = identity(self.config, self.req['parent'])
        d.immutable(self.root / 'WORKERS/old/START.json', {'identity': old_identity, 'parent': self.req['parent']})
        d.execute_request(self.root, self.req, self.config, backend, 'old')
        row = d.read(d.result_path(self.root, self.req))
        closed = dict(old_identity, policy_tensor_hash_after='tensor-frozen', end_model_unchanged=True)
        d.immutable(self.root / 'WORKERS/new/END.json', {'identity': closed, 'parent': self.req['parent'], 'status': 'CLOSED_READONLY'})
        d.reconcile_reused(self.root, [row], 'new', closed)
        d.check_worker_closure(self.root, 'old', [row], self.config)
        self.assertFalse((self.root / 'WORKERS/old/END.json').exists())
        proof = d.read(self.root / 'RECOVERY/old--new.json')
        self.assertEqual(proof['original_terminal_model_state'], 'UNKNOWN_PROCESS_INTERRUPTED')
        self.assertEqual(len(backend.calls), 1)

    def test_failed_model_closure_cannot_be_reconciled(self):
        ident = identity(self.config, self.req['parent'])
        d.immutable(self.root / 'WORKERS/old/START.json', {'identity': ident, 'parent': self.req['parent']})
        d.immutable(self.root / 'WORKERS/old/CLOSE_FAILED.json', {'error': 'model changed'})
        row = dict(response(self.req, self.config), worker_id='old')
        closed = dict(ident, policy_tensor_hash_after='tensor-frozen', end_model_unchanged=True)
        with self.assertRaisesRegex(RuntimeError, 'failed model closure'):
            d.reconcile_reused(self.root, [row], 'new', closed)


    def test_invalid_array_is_model_failure_not_technical_failure(self):
        row = response(self.req, self.config)
        row['suffix_text'] = '"x":[}'
        row['full_text'] = self.req['assistant_prefill'] + row['suffix_text']
        self.assertEqual(d.native_object_end(row['full_text']), len(row['full_text']))
        self.assertEqual(d.validate_record(row, self.req, self.config), [])

    def test_hashed_not_run_precheck_is_not_pass(self):
        checks = {}
        for name in d.PRECHECK_NAMES:
            path = self.root / (name + '.json')
            d.immutable(path, {'status': 'NOT_RUN'})
            checks[name] = {'path': path.name, 'sha256': d.file_sha(path)}
        d.immutable(self.root / 'PRECHECKS.json', {'checks': checks})
        with self.assertRaisesRegex(RuntimeError, 'has not passed'):
            d.check_prechecks(self.root)

    def test_precheck_hash_is_verified(self):
        checks = {}
        for name in d.PRECHECK_NAMES:
            path = self.root / (name + '.json')
            d.immutable(path, {'status': sorted(d.PRECHECK_STATUSES[name])[0]})
            checks[name] = {'path': path.name, 'sha256': d.file_sha(path)}
        d.immutable(self.root / 'PRECHECKS.json', {'checks': checks})
        d.check_prechecks(self.root)
        (self.root / 'history.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'history'):
            d.check_prechecks(self.root)


if __name__ == '__main__':
    unittest.main()
