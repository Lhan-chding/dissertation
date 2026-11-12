"""CPU boundary tests: synthetic captures are never GPU/production evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('satdose_backend', Path(__file__).parents[1]/'runtime/backend.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        (self.root/'image.png').write_bytes(b'fixture image')
        self.row = dict(request_id='dry/1', split='DRY', start_seed=11031, arm='START', step=0,
                        input_id='input1', seed=37)
        self.item = dict(input_id='input1', prompt='public question', image_file='image.png', assistant_prefill='{')
        self.image = dict(image_file='image.png', png_sha256=b.file_sha(self.root/'image.png'))
        self.request = b.join_public_request(self.row, self.item, self.image, self.root)
        self.identity = dict(experiment_id=b.EXPERIMENT, start_seed=11031, arm='START', step=0,
                             policy_sha256='a'*64, reference_sha256='b'*64)
        self.addCleanup(self.temp.cleanup)

    def test_no_private_field_projection_bypass(self):
        for target in ('request', 'input'):
            row, item = copy.deepcopy(self.row), copy.deepcopy(self.item)
            (row if target == 'request' else item)['gold'] = {'answer': 4}
            with self.assertRaisesRegex(ValueError, 'private'):
                b.join_public_request(row, item, self.image, self.root)

    def test_original_image_identity(self):
        (self.root/'image.png').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Image bytes'):
            b.join_public_request(self.row, self.item, self.image, self.root)

    def test_symlink_rejected(self):
        (self.root/'alias.png').symlink_to(self.root/'image.png')
        self.item['image_file'] = self.image['image_file'] = 'alias.png'
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            b.join_public_request(self.row, self.item, self.image, self.root)

    def test_prefix_not_training(self):
        self.item['assistant_prefill'] = '{"evidence":['
        with self.assertRaisesRegex(ValueError, 'forced prefix'):
            b.join_public_request(self.row, self.item, self.image, self.root)

    def test_main_sealed(self):
        request = dict(self.request, scope='MAIN')
        with self.assertRaisesRegex(ValueError, 'sealed'):
            b.authorize_request(request, self.identity)
        b.authorize_request(request, self.identity, terminal={'all_paths_terminal': True})

    def test_training_freshness_and_canary(self):
        request = dict(self.request, scope='TRAIN', arm='T0', step=3)
        identity = dict(self.identity, arm='T0', step=2)
        admission = {'status': 'ADMITTED_FIRST_TWO_STEPS_ONLY'}
        with self.assertRaisesRegex(ValueError, 'Canary'):
            b.authorize_request(request, identity, admission=admission)
        with self.assertRaisesRegex(ValueError, 'Stale'):
            b.authorize_request(request, dict(identity, step=1), admission=admission)
        b.authorize_request(request, identity, admission=admission, canary={'status':'CANARY_RESUME_PASS'})

    def bridge(self, runner):
        owner = self
        class Registry:
            root = owner.root
            freeze_sha256 = 'f'*64
            def request(self, rid):
                assert rid == owner.request['request_id']
                return copy.deepcopy(owner.request)
        return b.GenerationBridge(Registry(), self.root, runner, lambda: copy.deepcopy(self.identity))

    def capture(self):
        return dict(tokens=[17, 18], raw_tokens=[17, 18], completion_token_count=2,
                    sampler_logprob_source='actual_generate_raw_logits_selected_token', sampler_logprobs=[-0.1, -0.4], assistant_prefill='{',
                    generated_text='}', full_text='{}', generation_status='COMPLETE',
                    technical_validation_errors=[], finish_reason='balanced')

    def test_exact_original_capture_and_no_second_attempt(self):
        calls = []
        def runner(request, root, callback):
            calls.append(request['request_id'])
            callback(self.capture())
        bridge = self.bridge(runner)
        result = bridge.generate_one('dry/1')
        self.assertEqual(result['tokens'], [17, 18])
        self.assertEqual(result['sampler_logprobs'], [-0.1, -0.4])
        saved = json.loads(Path(result['native_capture_path']).read_text())
        self.assertEqual(saved, self.capture())
        with self.assertRaisesRegex(ValueError, 'never resample'):
            bridge.generate_one('dry/1')
        self.assertEqual(len(calls), 1)

    def test_exception_keeps_unknown_and_no_retry(self):
        def runner(*args):
            raise RuntimeError('fixture OOM')
        bridge = self.bridge(runner)
        with self.assertRaisesRegex(RuntimeError, 'OOM'):
            bridge.generate_one('dry/1')
        self.assertEqual(len(list(self.root.glob('*.BEGIN.json'))), 1)
        self.assertEqual(len(list(self.root.glob('*.ERROR.json'))), 1)
        self.assertEqual(len(list(self.root.glob('*.COMPLETE.json'))), 0)
        with self.assertRaisesRegex(ValueError, 'never resample'):
            bridge.generate_one('dry/1')

    def test_policy_changed_is_technical_hold(self):
        def runner(request, root, callback):
            callback(self.capture())
            self.identity['policy_sha256'] = 'c'*64
        with self.assertRaisesRegex(ValueError, 'Endpoint changed'):
            self.bridge(runner).generate_one('dry/1')
        self.assertEqual(len(list(self.root.glob('*.COMPLETE.json'))), 0)

    def test_truncated_real_capture_is_retained(self):
        def runner(request, root, callback):
            row = self.capture()
            row['finish_reason'] = 'length'
            callback(row)
        self.assertEqual(self.bridge(runner).generate_one('dry/1')['finish_reason'], 'length')

    def registry_files(self):
        cfg = self.root/'config.json'
        cfg.write_text(json.dumps({'experiment_id': b.EXPERIMENT}))
        refs = {}
        for name, value in [('requests', self.row), ('inputs', self.item), ('images', self.image)]:
            path = self.root/(name+'.jsonl')
            path.write_text(json.dumps(value)+'\n')
            refs[name] = {'path': path.name, 'sha256': b.file_sha(path)}
        freeze = self.root/'DESIGN_FREEZE.json'
        freeze.write_text(json.dumps(dict(schema='sr-satdose-design-freeze-v1',
            experiment_id=b.EXPERIMENT, config_sha256=b.file_sha(cfg), artifacts=refs)))
        return cfg, freeze

    def test_partial_manifest_cannot_admit(self):
        cfg, freeze = self.registry_files()
        with self.assertRaisesRegex(ValueError, 'Incomplete or expanded'):
            b.FrozenRegistry(cfg, self.root, freeze, [self.root/'requests.jsonl'],
                             self.root/'inputs.jsonl', self.root/'images.jsonl')

    def test_frozen_input_changed_before_registry(self):
        cfg, freeze = self.registry_files()
        (self.root/'inputs.jsonl').write_text('{}\n')
        with self.assertRaisesRegex(ValueError, 'Frozen artifact changed'):
            b.FrozenRegistry(cfg, self.root, freeze, [self.root/'requests.jsonl'],
                             self.root/'inputs.jsonl', self.root/'images.jsonl')

    def test_duplicate_json_rejected(self):
        p = self.root/'duplicates.json'
        p.write_text('{"key":1,"key":2}')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            b.read_json(p)


if __name__ == '__main__':
    unittest.main()
