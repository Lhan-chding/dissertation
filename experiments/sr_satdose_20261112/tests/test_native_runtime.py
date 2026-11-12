"""CPU checks only: no model load, generation, GPU equivalence or admission."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.native_runtime import RegisteredRuntimeMixin, RuntimeHandle, verify_source_lock, build_runtime
import torch


class FakeRegistry:
    def __init__(self, root, requests):
        self.root, self.requests = root, requests
    def request(self, rid):
        return copy.deepcopy(self.requests[rid])


class RequestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name).resolve()
        (self.root / 'x.png').write_bytes(b'IMAGE')
        self.request = dict(request_id='r1', start_seed=11031, seed=1, scope='PREFIX_CALIB', arm='START', step=0,
            prompt='question', image_file='x.png', image_sha256=hashlib.sha256(b'IMAGE').hexdigest(),
            assistant_prefill='{"evidence":[')
        class Runtime(RegisteredRuntimeMixin):
            pass
        self.r = Runtime(); self.r._satdose_registry = FakeRegistry(self.root, {'r1': self.request})
        self.r._satdose_start = 11031; self.r._satdose_active = None
        self.calls = []
        def prepare(runtime, req, root):
            self.calls.append(req['assistant_prefill'])
            return dict(prefix=req['assistant_prefill'])
        self.r._satdose_legacy = SimpleNamespace(prepare_public_input=prepare)
    def tearDown(self):
        self.temp.cleanup()
    def test_requires_bound_request(self):
        with self.assertRaises(ValueError):
            self.r.prepare_input({}, self.root)
    def test_private_or_changed_request_rejected(self):
        with self.assertRaises(ValueError):
            self.r.bind_request(dict(self.request, gold='answer'))
    def test_wrong_start_rejected(self):
        self.r._satdose_start = 11032
        with self.assertRaises(ValueError): self.r.bind_request(self.request)
    def test_prefix_in_cache_key(self):
        self.r.bind_request(self.request)
        safe = {k: self.request[k] for k in ('prompt', 'image_file', 'image_sha256')}
        self.r.prepare_input(safe, self.root)
        self.r.unbind_request()
        other = dict(self.request, request_id='r2', assistant_prefill='{"evidence":[{}')
        self.r._satdose_registry.requests['r2'] = other
        self.r.bind_request(other); self.r.prepare_input(safe, self.root)
        self.assertEqual(self.calls, [self.request['assistant_prefill'], other['assistant_prefill']])
    def test_nested_request_rejected(self):
        self.r.bind_request(self.request)
        with self.assertRaises(ValueError): self.r.bind_request(self.request)
    def test_image_tamper_checked_at_prepare(self):
        self.r.bind_request(self.request); (self.root / 'x.png').write_bytes(b'TAMPER')
        with self.assertRaises(ValueError):
            self.r.prepare_input({k: self.request[k] for k in ('prompt','image_file','image_sha256')}, self.root)
    def test_public_input_injection_rejected(self):
        self.r.bind_request(self.request)
        with self.assertRaises(ValueError): self.r.prepare_input(dict(self.request), self.root)
    def test_runtime_construction_rejects_fake_registry(self):
        with self.assertRaises(ValueError):
            build_runtime(self.r._satdose_registry, self.root/'lock', start_seed=11031, output_dir=self.root)


class ForwardTests(unittest.TestCase):
    def test_original_token_concatenation_and_mask(self):
        class Runtime(RegisteredRuntimeMixin): pass
        r = Runtime(); r.torch = torch; r._satdose_active = {'assistant_prefill':'{','scope':'DRY'}
        r.processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0)); r.eos_ids = [9]
        r._satdose_repeat_prompt = lambda inputs, n: dict(inputs)
        received = []
        r._native_forward = lambda inputs, **kwargs: received.append((inputs, kwargs))
        prepared = dict(inputs=dict(input_ids=torch.tensor([[1,2]]), attention_mask=torch.ones(1,2,dtype=torch.long),
            mm_token_type_ids=torch.ones(1,2,dtype=torch.long)), routing=dict(protocol='structured',protocol_revision='v2',assistant_prefill='{'))
        r._batch_forward(prepared, [[3,4,9]], purpose='training_policy', grad=True)
        inputs, meta = received[0]
        self.assertEqual(inputs['input_ids'].tolist(), [[1,2,3,4,9]])
        self.assertEqual(inputs['mm_token_type_ids'].tolist(), [[1,1,0,0,0]])
        self.assertEqual(meta, dict(completion_start=2,lengths=[3],purpose='training_policy',grad=True))
        self.assertEqual(prepared['inputs']['input_ids'].tolist(), [[1,2]])
        with self.assertRaises(ValueError): r._batch_forward(prepared, [[]],purpose='training_policy',grad=True)


class SourceLockTests(unittest.TestCase):
    def test_source_lock_must_be_frozen_and_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve(); native = root/'native'; legacy = root/'legacy'
            model = root/'snapshot'; parent = root/'parent'
            for d in (native,legacy,model,parent): d.mkdir()
            (native/'x.py').write_text('x=1')
            required = ['gpu_production_v2/backend.py','diagnostics/gpu_capacity_v2.py','runtime_contract.py']
            for name in required:
                p = legacy/name; p.parent.mkdir(parents=True,exist_ok=True);p.write_text('x=1')
            (model/'weights').write_bytes(b'weights')
            for name in ('adapter_config.json','adapter_model.safetensors'): (parent/name).write_bytes(b'parent')
            sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            lock = dict(native_src=str(native),legacy_root=str(legacy),native_files={'x.py':sha(native/'x.py')},
                legacy_files={n:sha(legacy/n) for n in required},snapshot='snapshot',model_path=str(model),
                parent_path=str(parent),model_files={'weights':sha(model/'weights')},
                parent_files={n:sha(parent/n) for n in ('adapter_config.json','adapter_model.safetensors')})
            path=root/'lock.json';path.write_text(json.dumps(lock))
            freeze=root/'freeze.json';freeze.write_text(json.dumps({'artifacts':{'source':{'path':'lock.json','sha256':sha(path)}}}))
            registry=SimpleNamespace(root=root,freeze_path=freeze,config={'model_snapshot':'snapshot'})
            self.assertEqual(verify_source_lock(registry,path)[0],lock)
            (native/'extra.py').write_text('x=2')
            with self.assertRaisesRegex(ValueError,'inventory incomplete'): verify_source_lock(registry,path)
            (native/'extra.py').unlink();(model/'weights').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'Model bytes changed'): verify_source_lock(registry,path)


if __name__ == '__main__': unittest.main()
