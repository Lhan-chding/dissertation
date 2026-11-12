"""CPU-only tests against the real pinned legacy constructor, never model outputs."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

BASE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('satdose_data_adapter', BASE / 'runtime/data_adapter.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
LEGACY = Path(os.environ.get('SATDOSE_LEGACY_ROOT', BASE.parents[1] / '.worktrees/sr-ord-train-20261103/sr_ord_train'))
HISTORY = Path(os.environ.get('SR_HISTORICAL_BUNDLE', '/private/tmp/satdose_historical_verified_20261112'))


@unittest.skipUnless(LEGACY.is_dir() and HISTORY.is_dir(), 'real legacy source and historical bundle required')
class DataAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.lock = cls.root / 'source_lock.json'
        cls.lock.write_text(json.dumps(m.source_inventory(LEGACY)))
        cls.legacy = m.load_legacy(LEGACY, json.loads(cls.lock.read_text()))
        cls.out = cls.root / 'prepared'
        cls.report = m.construct(BASE / 'supplied_package', LEGACY, cls.lock, HISTORY, cls.out)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_complete_budget_and_no_freeze(self):
        self.assertEqual(self.report['generation_counts'], dict(TRAIN=12288, DRY=1280, DEV=1512, PREFIX_CALIB=96, MAIN=13760))
        self.assertFalse((self.out / 'DESIGN_FREEZE.json').exists())
        self.assertEqual(self.report['model_calls'], 0)
        ids = [r['request_id'] for p in self.out.glob('*_manifest.jsonl') if p.name != 'visual_review_manifest.jsonl' for r in m.rows(p)]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 28936)

    def test_schedule_pairing_and_only_list_difference(self):
        cases = {r['case_id']: r for r in m.rows(self.out / 'private_cases.jsonl')}
        inputs = {r['input_id']: r for r in m.rows(self.out / 'inputs.jsonl')}
        groups = m.rows(self.out / 'training_groups.jsonl')
        for x, y in zip(groups[::2], groups[1::2]):
            self.assertEqual((x['arm'], y['arm']), ('T0', 'T1'))
            self.assertEqual(x['draw_seeds'], y['draw_seeds'])
            a, b = cases[x['case_id']], cases[y['case_id']]
            pa, pb = inputs[a['input_id']], inputs[b['input_id']]
            self.assertEqual(pa['image_file'], pb['image_file'])
            self.assertEqual(a['task'], b['task'])
            self.assertEqual(m.strip_list(pa['prompt'], self.legacy['data_design'].RANK_PREFIX), m.strip_list(pb['prompt'], self.legacy['data_design'].RANK_PREFIX))
            if not y['planned_conflict']:
                self.assertEqual(pa, pb)
        self.assertEqual(sum(g['planned_conflict'] for g in groups), 576)

    def test_public_projection_and_probe_disjoint(self):
        for row in m.rows(self.out / 'inputs.jsonl'):
            self.assertEqual(set(row), {'input_id', 'prompt', 'assistant_prefill', 'image_file'})
        probe = json.loads((self.out / 'PROBE_SPLITS.json').read_text())
        self.assertEqual(len(probe['probe_root_ids']), 16)
        self.assertEqual(len(probe['witness_root_ids']), 16)
        self.assertFalse(set(probe['probe_root_ids']) & set(probe['witness_root_ids']))
        for key in ('probe', 'witness'):
            self.assertEqual(list(map(len, probe[key + '_halves'])), [8, 8])

    def test_frozen_eval_seed_pairs(self):
        seen = {}
        for phase in ('main', 'dev', 'prefix_calib'):
            for r in m.rows(self.out / (phase + '_manifest.jsonl')):
                key = (r['panel'], r['root_id'], r['draw'])
                self.assertEqual(seen.setdefault(key, r['seed']), r['seed'])
                expected = self.legacy['srf6_ref.core'].seed_for(2026110301, 'eval', 'b0', *key)
                self.assertEqual(r['seed'], expected)

    def test_source_tamper_and_overwrite_rejected(self):
        lock = json.loads(self.lock.read_text()); lock['data_design.py'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'source inventory'):
            m.load_legacy(LEGACY, lock)
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            m.construct(BASE / 'supplied_package', LEGACY, self.lock, HISTORY, self.out)

    def test_historical_duplicate_rejected(self):
        bank = m.Bank(self.legacy, HISTORY, self.root, None, False, None)
        task = next(r['task'] for r in m.rows(HISTORY / 'frozen/private_cases.jsonl') if 'boundary_indices' in r['task'])
        with self.assertRaisesRegex(ValueError, 'historical world collision'):
            bank.add(self.legacy['data_design'].build_context(task, 'BASE', 'R', 0), 'TRAIN')

    def test_cross_split_duplicate_rejected(self):
        bank = m.Bank(self.legacy, HISTORY, self.root, None, False, None)
        task = self.legacy['data_design'].make_world('SATDOSE_UNIT', 0)
        ctx = self.legacy['data_design'].build_context(task, 'BASE', 'R', 0)
        bank.add(ctx, 'TRAIN')
        with self.assertRaisesRegex(ValueError, 'cross-split'):
            bank.add(ctx, 'MAIN')

    def test_render_structure_detects_wrong_world(self):
        font = Path(os.environ.get('SATDOSE_TEST_FONT', '/System/Library/Fonts/Helvetica.ttc'))
        if not font.is_file():
            self.skipTest('set SATDOSE_TEST_FONT to an installed font')
        task = self.legacy['data_design'].make_world('SATDOSE_UNIT', 1)
        path = self.root / 'chart.png'
        self.legacy['order_probe.render'].render(task, path, str(font))
        self.assertTrue(m.inspect_image(path, task)['image_structure_pass'])
        wrong = copy.deepcopy(task); wrong['world']['series']['Alpha'][0] += 5
        with self.assertRaisesRegex(ValueError, 'bar'):
            m.inspect_image(path, wrong)


if __name__ == '__main__':
    unittest.main()
