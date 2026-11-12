"""Machine-spec budget, template, and identity checks. No model calls."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from satdose_core import read_json
from check_admission import REQUIRED_TECH, verify_gpu


class ConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = read_json(ROOT / 'config/experiment.json')

    def test_formal_budget(self):
        c = self.cfg
        b = c['budget']
        paths = len(c['starts']) * len(c['arms'])
        self.assertEqual(paths, 4)
        self.assertEqual(paths, b['formal_training_paths'])
        self.assertEqual(paths * c['steps'], b['formal_training_updates'])
        self.assertEqual(paths * c['steps'] * c['groups_per_step'] * c['draws_per_group'], b['formal_training_responses'])
        self.assertEqual(b['formal_training_responses'], 12288)

    def test_dry_budget(self):
        c = self.cfg
        n = len(c['starts']) * (c['dry']['conflict_groups_per_start'] + c['dry']['aligned_groups_per_start']) * c['draws_per_group']
        self.assertEqual(n, c['dry']['generated_responses'])
        self.assertEqual(n, c['budget']['dry_responses'])
        self.assertEqual(n, 1280)

    def test_panel_arithmetic(self):
        c = self.cfg
        for name, panel in c['final_panels'].items():
            v = len(panel['views']) if 'views' in panel else panel['views_per_root']
            self.assertEqual(panel['roots'] * v * panel['draws'], panel['rows_per_endpoint'], name)
        n = sum(p['rows_per_endpoint'] for p in c['final_panels'].values())
        self.assertEqual(n, c['budget']['final_rows_per_full_endpoint'])
        self.assertEqual(n * c['budget']['final_full_endpoints'], c['budget']['final_full_responses'])
        self.assertEqual(n, 1952)

    def test_generation_total(self):
        b = self.cfg['budget']
        keys = ('formal_training_responses', 'dry_responses', 'prefix_calibration_responses', 'dev_monitor_responses', 'final_full_responses', 'intermediate_primary_responses')
        self.assertEqual(sum(b[k] for k in keys), b['all_planned_generations'])
        self.assertEqual(b['all_planned_generations'], 28936)
        self.assertEqual(b['technical_replay_generations'], 0)

    def test_main_and_dev_budget(self):
        c = self.cfg
        b = c['budget']
        self.assertEqual(8 * c['final_primary']['roots'] * c['final_primary']['draws'], b['intermediate_primary_responses'])
        self.assertEqual((2 + 4 * 3) * (32 * 3 + 12), b['dev_monitor_responses'])
        self.assertEqual(2 * 16 * 3, b['prefix_calibration_responses'])
        self.assertTrue(c['confirmation']['main_evaluation_sealed_until_training_terminal'])
        self.assertTrue(c['confirmation']['intermediate_no_efficacy_stop'])

    def test_bindings_and_restart_identity(self):
        c = self.cfg
        binding = read_json(ROOT / 'config/source_bindings.json')
        self.assertEqual(set(c['start_bindings']), {'11031', '11032'})
        self.assertTrue(c['optimizer']['fresh_at_fork'])
        self.assertTrue(c['optimizer']['restore_on_resume'])
        self.assertEqual(c['reference'], 'original_b0_parent_not_R24')
        refs = {v['reference_sha256'] for v in c['start_bindings'].values()}
        self.assertEqual(len(refs), 1)
        self.assertFalse(c['train_source_reward'])
        self.assertEqual(c['train_authority'], 'chart')
        for v in c['start_bindings'].values():
            for k in ('policy_sha256', 'state_sha256', 'reference_sha256'):
                self.assertEqual(len(v[k]), 64)
                int(v[k], 16)
        self.assertIn('native_scorer_pins', binding)

    def test_gpu_template_cannot_authorize(self):
        template = read_json(ROOT / 'templates/GPU_VERIFICATION.template.json')
        self.assertNotEqual(template['mode'], 'production')
        self.assertEqual(set(template['checks']), set(REQUIRED_TECH))
        self.assertTrue(all(v['passed'] is False for v in template['checks'].values()))
        with self.assertRaises(ValueError):
            verify_gpu(template, self.cfg, ROOT / 'config/experiment.json', ROOT / 'DOES_NOT_EXIST.json', ROOT)

    def test_restrictions_and_effect_threshold(self):
        c = self.cfg
        self.assertEqual(c['final_primary']['SESOI'], .10)
        self.assertEqual(c['final_primary']['run_SD_sensitivity'], [0., .03, .05])
        self.assertEqual(c['steps'], 24)
        self.assertEqual(c['T1_conflict_per_step'] / c['groups_per_step'], .75)
        self.assertEqual(c['T0_conflict_per_step'], 0)
        self.assertIn('automatic_stage2', c['prohibited'])
        self.assertIn('gpu_hour_scientific_stop', c['prohibited'])


if __name__ == '__main__':
    unittest.main()
