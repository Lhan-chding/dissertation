"""CPU-only serialization/routing regression tests; no model claims."""
import copy
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.native_adapter import (bind_native_kernel, prefix_helpers, validate_request,
                                    prepare_public_input, NativeBackend)
from order_probe.common import text_sha
from order_probe.measurement import first_object_end


class FakeStop:
    def __init__(self, tokenizer, prompt_len, rows, protocol, *, revision):
        self.prefill = '{'
        self.counts = [None] * rows


def decode(_, tokens):
    return ''.join(tokens)


class NativeAdapterTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / 'config/plan.json').read_text())
        self.request = json.loads((ROOT / 'data/SMOKE_REQUESTS_PUBLIC.jsonl').read_text().splitlines()[0])
        self.prefill = '{"evidence":[{"value":1}],'
        self.helpers = prefix_helpers(self.prefill, decode, FakeStop)

    def test_nested_binding_does_not_stop(self):
        boundary = self.helpers['first_boundary_token']
        suffix = list('"bindings":[{"operand":1}],"answer":1}')
        n, cut = boundary(None, suffix, 'structured')
        self.assertEqual(n, len(suffix))
        self.assertEqual(cut, len(self.prefill) + len(suffix))

    def test_quoted_braces_and_escapes(self):
        suffix = list('"work":"a \\\" } [","answer":1}')
        n, _ = self.helpers['first_boundary_token'](None, suffix, 'structured')
        self.assertEqual(n, len(suffix))

    def test_token_attached_trailing_output_preserved(self):
        text = '"answer":1}EXTRA'
        out = self.helpers['decoded_protocol'](text, 'structured')
        self.assertEqual(out['full_text'], self.prefill + text)
        self.assertEqual(out['format_protocol_error'], 'trailing_after_close')

    def test_prefix_stop_starts_with_complete_prefill(self):
        stop = self.helpers['BatchedProtocolStop'](None, 20, 1, 'structured')
        self.assertEqual(stop.prefill, self.prefill)
        with self.assertRaises(PermissionError):
            self.helpers['BatchedProtocolStop'](None, 20, 2, 'structured')

    def test_native_function_code_and_globals_remain_unchanged(self):
        namespace = {'_prefill': lambda *args: 'old'}
        exec("def generate(*, protocol='structured'):\n return _prefill(protocol, 'v2')", namespace)
        original = namespace['generate']
        bound = bind_native_kernel(original, self.helpers)
        self.assertIs(original.__code__, bound.__code__)
        self.assertEqual(original(), 'old')
        self.assertEqual(bound(), self.prefill)
        self.assertEqual(bound.__kwdefaults__, original.__kwdefaults__)

    def test_extra_sampler_rebinding_rejected(self):
        with self.assertRaises(PermissionError):
            bind_native_kernel(lambda: None, {**self.helpers, 'seed_all': lambda: None})

    def test_public_projection_excludes_metadata(self):
        safe = validate_request(self.request, self.config)
        self.assertEqual(set(safe), {'prompt', 'image_file', 'image_sha256'})
        for key in ('world', 'gold', 'query', 'rename', 'boundary_indices'):
            with self.assertRaises(PermissionError):
                validate_request({**self.request, key: {}}, self.config)

    def test_tampered_text_sampling_and_seed_rejected(self):
        for change in [{'prompt': 'wrong'}, {'assistant_prefill': '{}'}, {'seed': True},
                       {'sampling': {**self.request['sampling'], 'temperature': 0.5}}]:
            with self.assertRaises(PermissionError):
                validate_request({**self.request, **change}, self.config)

    def test_finished_prefill_rejected_even_with_valid_hash(self):
        req = {**self.request, 'assistant_prefill': '{}', 'prefill_sha256': text_sha('{}')}
        with self.assertRaises(PermissionError):
            validate_request(req, self.config)

    def captured_backend(self, *, mismatch=False):
        class Values:
            def __init__(self, value): self.value = value
            def __getitem__(self, key): return Values(self.value[key])
            def detach(self): return self
            def cpu(self): return self
            def tolist(self): return self.value
        backend = NativeBackend.__new__(NativeBackend)
        backend._readonly = lambda: None
        backend._extra_identity, backend._accounting = {}, []
        backend.runtime = types.SimpleNamespace(identity={}, eos_ids=[], _srf5_prepared={
            'inputs': {'input_ids': Values([[90, 100]])},
            'routing': {'base_prompt_token_count': 1,
                        'assistant_prefill_token_ids': [999 if mismatch else 100],
                        'assistant_prefill_native_char_span': [1, 2],
                        'assistant_prefill_native_offsets': [[1, 2]]}})
        row = {'technical_validation_errors': [], 'generated_text': '"answer":1}',
               'tokens': [1], 'sampler_logprobs': [1000.0],
               'full_text': self.request['assistant_prefill'] + '"answer":1}',
               'balanced_token_count': 1, 'balanced_cut_char': 11}
        return backend, row

    def test_captured_boundary_failure_preserves_exact_raw_output(self):
        backend, row = self.captured_backend(mismatch=True)
        before = copy.deepcopy(row)
        result = backend._preserve_captured(self.request, [row], None)
        self.assertEqual(result['raw_native_captures'], [before])
        self.assertEqual(row, before)
        self.assertEqual(result['generation_status'], 'TECHNICAL_INVALID')
        self.assertIn('boundary mismatch', result['technical_validation_errors'][-1])
        self.assertEqual(json.loads(json.dumps(result))['raw_native_captures'], [before])

    def test_captured_probability_overflow_preserves_exact_raw_output(self):
        backend, row = self.captured_backend()
        result = backend._preserve_captured(self.request, [row], None)
        self.assertEqual(result['raw_native_captures'], [row])
        self.assertIn('OverflowError', result['technical_validation_errors'][-1])
        self.assertEqual(result['suffix_token_ids'], row['tokens'])
        self.assertIn('NO_RESAMPLING', result['recovery'])

    def test_multiple_captures_are_preserved_without_resampling(self):
        backend, row = self.captured_backend()
        result = backend._preserve_captured(self.request, [row, row], 'native failure')
        self.assertEqual(result['captured_response_count'], 2)
        self.assertEqual(result['raw_native_captures'], [row, row])
        self.assertEqual(result['native_exception'], 'native failure')

    def test_all_registered_prefixes_unfinished(self):
        for line in (ROOT / 'data/SMOKE_REQUESTS_PUBLIC.jsonl').read_text().splitlines():
            req = json.loads(line)
            validate_request(req, self.config)
            self.assertIsNone(first_object_end(req['assistant_prefill']))


if __name__ == '__main__':
    unittest.main()
