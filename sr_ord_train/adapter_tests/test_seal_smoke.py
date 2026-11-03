"""Synthetic files exercise the seal only; no fixture is a real GPU receipt."""
from collections import Counter
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from adapter.backend import object_sha, text_sha
from adapter.registration import Registration, canonical, digest
from adapter.seal_smoke import collect, finalize, independent_scores, numerical_check, prepare
from adapter.training import trajectory_hash
from runtime_contract import PINS, SNAPSHOT

PACKAGE = Path(__file__).resolve().parents[1]


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value) + b'\n')


def put_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b''.join(canonical(row) + b'\n' for row in rows))


def fixture(run):
    registry = Registration(PACKAGE)
    requests = list(itertools.islice(registry.requests(), 192))
    manifest = run / 'REQUESTS.jsonl'
    put_rows(manifest, requests)
    raw_all = []
    for parent in ('b0', 'b1'):
        stage = run / 'dev_smoke' / parent
        execution = stage / 'executions' / 'SYNTHETIC_NO_GPU'
        execution.mkdir(parents=True)
        policy = ('a' if parent == 'b0' else 'b') * 64
        source_files = [{'path': '/synthetic/source.py', 'sha256': 'c' * 64}]
        binding = {'experiment_id': 'SR-ORD-TRAIN-1', 'model': {'snapshot': SNAPSHOT},
                   'manifest_sha256': registry.freeze_sha256, 'source_files': source_files,
                   'source_sha256': object_sha(source_files),
                   'requests_manifest': {'path': str(manifest), 'sha256': digest(manifest)}}
        loaded = {'experiment_id': 'SR-ORD-TRAIN-1', 'execution_mode': 'DEV_SMOKE', 'parent_id': parent,
                  'snapshot': SNAPSHOT, 'manifest_sha256': registry.freeze_sha256,
                  'binding_sha256': object_sha(binding), 'source_sha256': binding['source_sha256'],
                  'native_runtime_sha256': PINS['sr_f6/runtime.py'], 'dtype': 'float32',
                  'input_float_dtype': 'float32', 'probability_dtype': 'float32',
                  'forward_parameter_dtypes': ['float32'], 'forward_float_buffer_dtypes': ['float32'],
                  'attention_backend': 'eager', 'optimizer_created': False, 'step': 0,
                  'hardware': {'cuda_visible_device_count': 1, 'cuda_name': 'SYNTHETIC_NOT_REAL_GPU',
                               'cuda_uuid': 'SYNTHETIC_NOT_REAL_GPU'},
                  'parent_tensor_sha256': policy, 'reference_hash': policy}
        put(execution / 'WORKER_BINDING.json', binding)
        put(execution / 'LOADED_IDENTITY.json', loaded)
        raw, replay = [], []
        for request in [r for r in requests if r['parent'] == parent]:
            rid = request['request_id']
            route = {'same_assistant_message': True, 'image_token_count': 4,
                     'input_float_dtype': 'float32', 'probability_dtype': 'float32',
                     'forward_input_dtypes': {'pixel_values': 'float32', 'input_ids': 'int64'},
                     'input_tensor_hash': 'd' * 64}
            # Invalid JSON is a legitimate zero-reward observation, not a seal failure.
            row = {**request, 'row_id': rid, 'policy_hash': policy, 'reference_hash': policy,
                   'run_id': 'SR-ORD-TRAIN-1/DEV_SMOKE/' + parent, 'protocol': 'structured_v4',
                   'protocol_revision': 'v2', 'generation_status': 'COMPLETE',
                   'technical_validation_errors': [], 'native_exception': None,
                   'source_sha256': binding['source_sha256'],
                   'model_input': {k: request[k] for k in ('prompt', 'image_file', 'image_sha256')},
                   'tokens': [42, 43], 'raw_tokens': [42, 43], 'completion_token_count': 2,
                   'sampler_logprobs': [-2., -2.],
                   'sampler_logprob_source': 'actual_generate_raw_logits_selected_token',
                   'prompt_tensor_hash': 'd' * 64, 'native_prompt_inputs': {'input_ids': [[1, 2]]},
                   'full_text': '{invalid', 'suffix_text': 'invalid', 'generated_text': 'invalid',
                   'finish_reason': 'eos', 'finished': False, 'input_routing': route,
                   'sampling_parameters': dict(max_new_tokens=1536, temperature=1., top_p=1., top_k=0, do_sample=True)}
            native_attempt = execution / 'generation_attempts' / text_sha(rid)
            capture_path = native_attempt.with_suffix('.CAPTURE.json')
            capture = dict(row, protocol='structured', adapter_identity={'step': 0, 'run_id': row['run_id']})
            put(capture_path, capture)
            row.update(native_capture_path=str(capture_path), native_capture_sha256=digest(capture_path))
            row['token_path_sha256'] = trajectory_hash(row)
            put(native_attempt.with_suffix('.BEGIN.json'), {'request_id': rid, 'seed': request['seed'],
                'policy_sha256': policy, 'reference_sha256': policy, 'status': 'ATTEMPTED_UNKNOWN'})
            put(native_attempt.with_suffix('.COMPLETE.json'), row)
            put(stage / (object_sha(rid) + '.BEGIN.json'),
                {'request': request, 'policy_sha256': policy, 'status': 'ATTEMPTED_UNKNOWN'})
            raw.append(row)
            replay.append({'request_id': rid, 'tokens': row['tokens'], 'sampler_logprobs': row['sampler_logprobs'],
                           'train_logprobs': [-2., -2.], 'mean_absolute_difference': 0.,
                           'max_absolute_difference': 0., 'tis_truncated_fraction': 0.,
                           'prompt_tensor_hash': row['prompt_tensor_hash']})
        put_rows(stage / 'RAW.jsonl', raw)
        put_rows(stage / 'REPLAY.jsonl', replay)
        put(execution / 'WORKER_COMPLETE.json', {'status': 'WORKER_COMPLETE_REVIEW_PENDING', 'scope': 'DEV_SMOKE',
            'parent': parent, 'generated_responses': 96, 'optimizer_steps': 0, 'parameter_updates': 0,
            'policy_before_sha256': policy, 'policy_after_sha256': policy, 'raw_sha256': digest(stage / 'RAW.jsonl')})
        processor_dir = execution / 'processor'
        processor_dir.mkdir()
        image = Image.new('RGB', (2, 2), (1, 2, 3))
        image.save(processor_dir / 'processed.png')
        request = next(r for r in requests if r['parent'] == parent)
        put(processor_dir / 'PROCESSOR.json', {'request_id': request['request_id'],
            'source_file': request['image_file'], 'source_sha256': request['image_sha256'],
            'processed_file': str(processor_dir / 'processed.png'),
            'processed_file_sha256': digest(processor_dir / 'processed.png'),
            'processed_rgb_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
            'routing': {'input_tensor_hash': 'd' * 64}, 'prompt_input_ids': [1, 2],
            'visual_review': 'PENDING', 'model_calls': 0, 'optimizer_updates': 0})
        raw_all.extend(raw)
    cases = {r['case_id']: r for r in map(json.loads, (PACKAGE / 'frozen/private_cases.jsonl').read_text().splitlines())}
    scores = independent_scores(PACKAGE, raw_all, cases)
    for scored in scores['scores']:
        parent = next(r['parent'] for r in raw_all if r['request_id'] == scored['request_id'])
        put(run / 'dev_smoke' / parent / 'scores' / (object_sha(scored['request_id']) + '.json'), scored['score'])
    return raw_all


class SmokeSealTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.run = Path(self.temp.name).resolve()
        self.raw = fixture(self.run)

    def tearDown(self):
        self.temp.cleanup()

    def sign(self):
        path = self.run / 'SIGNED_VISUAL_FIXTURE.json'
        value = json.loads((self.run / 'smoke_review/VISUAL_REVIEW_TEMPLATE.json').read_text())
        value.update(status='VISUAL_REVIEW_PASS', reviewed_by='SYNTHETIC_TEST_ONLY', reviewed_at='TEST_ONLY')
        for row in value['processor_reviews']:
            row.update(labels_readable=True, values_readable=True, source_to_processed_consistent=True)
        put(path, value)
        return path

    def test_prepare_requires_separate_signature_and_keeps_zero_quality(self):
        summary = prepare(PACKAGE, self.run)
        self.assertEqual(summary['status'], 'SMOKE_TECHNICAL_PENDING_VISUAL_REVIEW')
        self.assertEqual(summary['quality_counts'], {'b0': {'0': 96}, 'b1': {'0': 96}})
        self.assertFalse((self.run / 'SMOKE_TECHNICAL_REVIEW.json').exists())
        with self.assertRaisesRegex(ValueError, 'explicit visual review signature'):
            finalize(PACKAGE, self.run, self.run / 'smoke_review/VISUAL_REVIEW_TEMPLATE.json')
        result = finalize(PACKAGE, self.run, self.sign())
        self.assertEqual(result['status'], 'SMOKE_TECHNICAL_PASS')
        self.assertFalse(result['scientific_result'])
        self.assertEqual(result['runtime_contract']['responses'], 192)
        with self.assertRaisesRegex(ValueError, 'existing seal'):
            finalize(PACKAGE, self.run, self.sign())

    def test_missing_replay_or_unknown_attempt_blocks(self):
        replay = self.run / 'dev_smoke/b0/REPLAY.jsonl'
        lines = replay.read_text().splitlines()
        replay.write_text('\n'.join(lines[:-1]) + '\n')
        with self.assertRaisesRegex(ValueError, 'replay coverage'):
            collect(PACKAGE, self.run)
        replay.write_text('\n'.join(lines) + '\n')
        native = self.run / 'dev_smoke/b0/executions/SYNTHETIC_NO_GPU/generation_attempts'
        next(native.glob('*.COMPLETE.json')).unlink()
        with self.assertRaisesRegex(ValueError, 'unresolved native'):
            collect(PACKAGE, self.run)

    def test_independent_native_scorer_detects_false_reward(self):
        score_path = next((self.run / 'dev_smoke/b0/scores').glob('*.json'))
        value = json.loads(score_path.read_text())
        value['J'] = 1
        put(score_path, value)
        with self.assertRaisesRegex(ValueError, 'independent native scores differ'):
            collect(PACKAGE, self.run)

    def test_finalize_rejects_live_evidence_change(self):
        prepare(PACKAGE, self.run)
        signed = self.sign()
        processor = self.run / 'dev_smoke/b0/executions/SYNTHETIC_NO_GPU/processor/processed.png'
        processor.write_bytes(b'changed pixels')
        with self.assertRaisesRegex(ValueError, 'processed PNG differs'):
            finalize(PACKAGE, self.run, signed)
        self.assertFalse((self.run / 'SMOKE_TECHNICAL_REVIEW.json').exists())

    def test_visual_template_cannot_change_reviewed_sources(self):
        prepare(PACKAGE, self.run)
        path = self.sign()
        value = json.loads(path.read_text())
        value['processor_reviews'][0]['source_sha256'] = 'e' * 64
        put(path, value)
        with self.assertRaisesRegex(ValueError, 'visual source/processor review incomplete'):
            finalize(PACKAGE, self.run, path)


class NumericalSmokeTests(unittest.TestCase):
    def test_historical_gate_is_token_weighted_and_rejects_misaligned_tokens(self):
        raw, replay = [], []
        for rid, size, difference in [('short', 1, .02), ('long', 100, 0.)]:
            raw.append({'request_id': rid, 'tokens': [1] * size, 'raw_tokens': [1] * size,
                        'completion_token_count': size, 'sampler_logprobs': [-2.] * size, 'prompt_tensor_hash': 'a' * 64})
            replay.append({'request_id': rid, 'tokens': [1] * size, 'sampler_logprobs': [-2.] * size,
                           'train_logprobs': [-2. + difference] * size, 'prompt_tensor_hash': 'a' * 64,
                           'mean_absolute_difference': difference, 'max_absolute_difference': difference,
                           'tis_truncated_fraction': 0.})
        self.assertAlmostEqual(numerical_check(raw, replay)['sampler_train_mean_absolute_difference'], .02 / 101)
        replay[0]['tokens'][0] = 99
        with self.assertRaisesRegex(ValueError, 'replay original trajectory differs'):
            numerical_check(raw, replay)

    def test_mean_and_tail_gates_fail_without_quality_threshold(self):
        raw = [{'request_id': 'row', 'tokens': [1], 'raw_tokens': [1], 'completion_token_count': 1,
                'sampler_logprobs': [-2.], 'prompt_tensor_hash': 'a' * 64}]
        replay = [{'request_id': 'row', 'tokens': [1], 'sampler_logprobs': [-2.], 'train_logprobs': [-1.98],
                   'prompt_tensor_hash': 'a' * 64, 'mean_absolute_difference': .02,
                   'max_absolute_difference': .02, 'tis_truncated_fraction': 0.}]
        with self.assertRaisesRegex(ValueError, 'numerical gate'):
            numerical_check(raw, replay)


if __name__ == '__main__':
    unittest.main()
