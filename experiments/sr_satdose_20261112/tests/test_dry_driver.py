"""Synthetic CPU orchestration only; fixtures never leave TemporaryDirectory."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))
d = importlib.import_module('runtime.dry_driver')


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


class DryDriverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.addCleanup(self.temp.cleanup)
        self.source = self.root/'source.json'
        self.source.write_text('{}')
        self.calls = []
        self.built = []
        self.closed = []
        self.score_finishes = []
        self.patches = [patch.object(d, 'verify_implementation', return_value={'code_commit':'c'*40}),
                        patch.object(d, 'verify_source_lock', return_value=({}, None, None, None, None))]
        for p in self.patches:
            p.start(); self.addCleanup(p.stop)

    def registry(self, scope):
        owner = self
        requests = {}
        for i in range(d.SCOPES[scope]):
            rid = f'{scope}/{i:04d}'
            requests[rid] = dict(request_id=rid, scope=scope, start_seed=11031, arm='START', step=0,
                case_id=f'case{i//8}', group_id=f'group{i//8}', seed=i, assistant_prefill='{',
                prompt='synthetic', image_file='unused.png', image_sha256='e'*64)
        cases = [dict(case_id=f'case{i}', task={'world':{'categories':list('ABCDEFGH')}, 'query':{}})
                 for i in range(80)]
        casefile = self.root/'private_cases.jsonl'
        casefile.write_text(''.join(json.dumps(x)+'\n' for x in cases))
        freeze = self.root/'freeze.json'
        freeze.write_text(json.dumps({'artifacts':{'private_cases':{'path':casefile.name, 'sha256':d.file_sha(casefile)}}}))
        class Registry:
            root = owner.root
            freeze_path = freeze
            freeze_sha256 = 'f'*64
            config = dict(experiment_id=d.EXPERIMENT if hasattr(d,'EXPERIMENT') else 'SR-SATDOSE-1-20261112',
                start_bindings={'11031':{'policy_sha256':'a'*64, 'reference_sha256':'b'*64}})
            def request(self, rid):
                return copy.deepcopy(self.requests[rid])
        reg = Registry(); reg.requests = requests
        return reg

    def builder(self, registry, source, **kwargs):
        self.built.append(kwargs)
        owner = self
        endpoint = dict(experiment_id='SR-SATDOSE-1-20261112', start_seed=11031, arm='START', step=0,
                        policy_sha256='a'*64, reference_sha256='b'*64)
        class Handle:
            native_ops = SimpleNamespace(state_hash=digest)
            fork = SimpleNamespace(optimizer=None, scheduler=None)
            runtime = SimpleNamespace(_satdose_accounting=[])
            def endpoint(self):
                return copy.deepcopy(endpoint)
            def close(self):
                owner.closed.append(True)
                return dict(endpoint, optimizer_steps=0, parameter_updates=0, closed=True)
        return Handle()

    def runner_factory(self, handle, lock):
        def runner(request, root, capture):
            self.calls.append(request['request_id'])
            capture(dict(seed=request['seed'], model_input={k:request[k] for k in ('prompt','image_file','image_sha256')}, tokens=[1], raw_tokens=[1], completion_token_count=1, sampler_logprobs=[-.1],
                sampler_logprob_source='actual_generate_raw_logits_selected_token', assistant_prefill='{',
                generated_text='}', full_text='{}', finish_reason='eos', generation_status='COMPLETE',
                technical_validation_errors=[]))
        return runner

    def scorer_factory(self, lock):
        def scorer(text, world, query, **kwargs):
            self.score_finishes.append(kwargs)
            return {'J':0, 'test_fixture_only':True}
        return scorer

    def run_fixture(self, reg, scope, name='out', **kwargs):
        return d.run_calibration(reg, self.source, start_seed=11031, scope=scope, output=self.root/name,
            runtime_builder=self.builder, runner_factory=kwargs.get('runner_factory', self.runner_factory),
            scorer_factory=self.scorer_factory)

    def test_dry_full_640_original_rows_balanced_only_scoring(self):
        reg = self.registry('DRY')
        result = self.run_fixture(reg, 'DRY')
        self.assertEqual(result['responses'],640)
        self.assertFalse(result['optimizer_created'])
        self.assertFalse(result['scheduler_created'])
        self.assertEqual(len(self.calls),640)
        self.assertTrue(all(x == {'finished':False,'finish_reason':'eos'} for x in self.score_finishes))
        rows = [json.loads(x) for x in (self.root/'out/DRY_RAW.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),640)
        self.assertEqual(rows[0]['native_score']['J'],0)
        self.assertIn('group_id', rows[0]); self.assertIn('case_id', rows[0])
        self.assertEqual(rows[0]['suffix_text'],'}')
        self.assertEqual(rows[0]['raw_tokens'],[1])
        self.assertEqual(self.closed,[True])
        self.assertFalse((self.root/'out/HOLD.json').exists())

    def test_prefix_no_native_J(self):
        result = self.run_fixture(self.registry('PREFIX_CALIB'),'PREFIX_CALIB')
        self.assertEqual(result['responses'],48)
        self.assertFalse(self.score_finishes)
        first = json.loads((self.root/'out/PREFIX_RAW.jsonl').read_text().splitlines()[0])
        self.assertIsNone(first['native_J'])
        self.assertFalse(first['native_J_applicable'])

    def test_failure_holds_and_different_output_cannot_resample(self):
        reg = self.registry('PREFIX_CALIB')
        def fail_factory(handle, lock):
            def fail(*args):
                raise RuntimeError('synthetic OOM')
            return fail
        with self.assertRaisesRegex(RuntimeError,'OOM'):
            self.run_fixture(reg,'PREFIX_CALIB',runner_factory=fail_factory)
        self.assertTrue((self.root/'out/HOLD.json').is_file())
        self.assertEqual(self.closed,[True])
        with self.assertRaises(FileExistsError):
            self.run_fixture(reg,'PREFIX_CALIB',name='different')
        self.assertEqual(len(self.built),1)

    def test_incomplete_allocation_no_model_load(self):
        reg = self.registry('DRY'); reg.requests.pop(next(iter(reg.requests)))
        with self.assertRaisesRegex(ValueError,'population differs'):
            self.run_fixture(reg,'DRY')
        self.assertFalse(self.built)

    def test_calibration_rejects_constructed_optimizer(self):
        registry = self.registry('PREFIX_CALIB')
        original = self.builder
        def incorrect_builder(*args, **kwargs):
            handle = original(*args, **kwargs)
            handle.fork.optimizer = object()
            return handle
        with self.assertRaisesRegex(ValueError, 'must not construct'):
            d.run_calibration(registry, self.source, start_seed=11031, scope='PREFIX_CALIB',
                output=self.root/'out', runtime_builder=incorrect_builder,
                runner_factory=self.runner_factory, scorer_factory=self.scorer_factory)
        self.assertFalse(self.calls)
        self.assertTrue((self.root/'out/HOLD.json').is_file())
        self.assertEqual(self.closed,[True])

    def test_source_inventory_cannot_omit_extra_python_file(self):
        # Use actual verifier on an isolated pretend deployment, not model code.
        self.patches[0].stop()
        code = self.root/'code'; code.mkdir()
        driver = code/'dry_driver.py'; driver.write_text('# synthetic source fixture\n')
        expected = {driver.name:d.file_sha(driver)}
        freeze = self.root/'freeze_code.json'
        freeze.write_text(json.dumps(dict(code_commit='a'*40, runtime_sources=expected,
            artifacts={'driver':{'path':'code/dry_driver.py','sha256':d.file_sha(driver)}})))
        with patch.object(d,'__file__',str(driver)):
            self.assertEqual(d.verify_implementation(freeze)['runtime_sources'],expected)
            (code/'unregistered.py').write_text('# not in freeze\n')
            with self.assertRaisesRegex(ValueError,'inventory/SHA'):
                d.verify_implementation(freeze)

    def test_changed_implementation_no_model_load(self):
        reg = self.registry('DRY')
        with patch.object(d,'verify_implementation',side_effect=ValueError('SHA differs')):
            with self.assertRaisesRegex(ValueError,'SHA differs'):
                self.run_fixture(reg,'DRY')
        self.assertFalse(self.built)


if __name__ == '__main__':
    unittest.main()
