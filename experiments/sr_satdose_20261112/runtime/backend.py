"""Frozen request admission and native generation bridge for SR-SATDOSE.

This module does NOT construct a GPU runtime, publish a checkpoint or grant GPU
admission. The caller must supply the separately audited FP32 runtime/fork and
live endpoint identity. Original native sampling and prefix functions are used
without calling the old experiment's identity verifier.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path

EXPERIMENT = 'SR-SATDOSE-1-20261112'
COUNTS = {'TRAIN': 12288, 'DRY': 1280, 'PREFIX_CALIB': 96, 'DEV': 1512, 'MAIN': 13760}
META = {'request_id', 'split', 'scope', 'start_seed', 'train_seed', 'arm', 'step',
        'root_id', 'case_id', 'input_id', 'group_id', 'draw', 'seed', 'panel', 'view',
        'structure', 'parent', 'slot', 'kind', 'endpoint_id'}
PUBLIC = {'prompt', 'image_file', 'image_sha256'}


def require(value, message):
    if not value:
        raise ValueError(message)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def object_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path):
    def pairs(rows):
        result = dict(rows)
        require(len(result) == len(rows), 'Duplicate JSON key')
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid)


def safe_path(root, relative):
    p = Path(relative)
    require(not p.is_absolute() and '..' not in p.parts, 'Path escapes evidence root')
    result = root / p
    require(not result.is_symlink() and result.resolve(strict=True).is_relative_to(root),
            'Symlink or escaped evidence')
    require(result.is_file(), 'Expected evidence file')
    return result


def write_once(path, value):
    require(not path.is_symlink(), 'Evidence path is a symlink')
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def join_public_request(row, item, image, root):
    """Reject private fields before projecting three native input fields."""
    require(set(row) <= META, 'Unknown/private request metadata')
    require(set(item) == {'input_id', 'image_file', 'prompt', 'assistant_prefill'},
            'Unknown/private public input fields')
    require(row['input_id'] == item['input_id'], 'Input identity mismatch')
    require(item['image_file'] == image['image_file'], 'Image identity mismatch')
    require(type(row['seed']) is int and 0 <= row['seed'] < 2**63, 'Invalid seed')
    require(row.get('start_seed', row.get('train_seed')) in (11031, 11032), 'Invalid start')
    scope = row.get('scope', row.get('split'))
    require(scope in COUNTS, 'Invalid scope')
    require(not ('scope' in row and 'split' in row) or row['scope'] == row['split'], 'Scope mismatch')
    require(type(item['prompt']) is str and item['prompt'], 'Missing public prompt')
    require(type(item['assistant_prefill']) is str and item['assistant_prefill'].startswith('{'),
            'Missing original same-message prefill')
    if scope in ('TRAIN', 'DRY'):
        require(item['assistant_prefill'] == '{', 'Training/DRY forced prefix prohibited')
    if scope == 'TRAIN':
        require(row.get('arm') in ('T0', 'T1') and type(row.get('step')) is int and
                1 <= row['step'] <= 24, 'Invalid training endpoint')
    else:
        require(row.get('arm') in ('START', 'T0', 'T1'), 'Invalid evaluation endpoint')
        require(type(row.get('step')) is int and row['step'] in (0, 2, 8, 16, 24), 'Invalid endpoint step')
        require((row['arm'] == 'START') == (row['step'] == 0), 'Start/step mismatch')
    path = safe_path(root, item['image_file'])
    require(file_sha(path) == image['png_sha256'], 'Image bytes changed')
    result = copy.deepcopy(row)
    result.update(scope=scope, prompt=item['prompt'], assistant_prefill=item['assistant_prefill'],
                  image_file=item['image_file'], image_sha256=image['png_sha256'])
    return result


class FrozenRegistry:
    """Validate a complete finite manifest against the pre-model design freeze."""
    def __init__(self, config_path, evidence_root, freeze_path, manifest_paths, inputs_path, images_path):
        self.root = Path(evidence_root).resolve(strict=True)
        self.config = read_json(config_path)
        require(self.config['experiment_id'] == EXPERIMENT, 'Wrong experiment')
        self.freeze_path = Path(freeze_path).resolve(strict=True)
        require(self.freeze_path.is_relative_to(self.root), 'Freeze outside evidence root')
        freeze = read_json(self.freeze_path)
        require(freeze['schema'] == 'sr-satdose-design-freeze-v1' and
                freeze['experiment_id'] == EXPERIMENT and
                freeze['config_sha256'] == file_sha(config_path), 'Wrong design freeze/config')
        self.freeze_sha256 = file_sha(self.freeze_path)
        bound = {}
        for ref in freeze['artifacts'].values():
            p = safe_path(self.root, ref['path'])
            require(file_sha(p) == ref['sha256'], 'Frozen artifact changed')
            bound[p.resolve()] = ref['sha256']
        def rows(path):
            p = Path(path).resolve(strict=True)
            require(p in bound, 'Manifest/index absent from design freeze')
            for line in p.read_text().splitlines():
                if line.strip():
                    # Same duplicate-key checks as JSON evidence.
                    def pairs(items):
                        obj = dict(items)
                        require(len(obj) == len(items), 'Duplicate JSONL key')
                        return obj
                    yield json.loads(line, object_pairs_hook=pairs)
        inputs = {}
        for row in rows(inputs_path):
            require(row['input_id'] not in inputs, 'Duplicate input')
            inputs[row['input_id']] = row
        images = {}
        for row in rows(images_path):
            require(row['image_file'] not in images, 'Duplicate image')
            images[row['image_file']] = row
        self.requests = {}
        counts = {key: 0 for key in COUNTS}
        for path in manifest_paths:
            for row in rows(path):
                rid = row['request_id']
                require(type(rid) is str and rid and rid not in self.requests, 'Duplicate/missing request ID')
                item = inputs[row['input_id']]
                joined = join_public_request(row, item, images[item['image_file']], self.root)
                counts[joined['scope']] += 1
                self.requests[rid] = joined
        require(counts == COUNTS, 'Incomplete or expanded scientific request budget')

    def request(self, request_id):
        require(file_sha(self.freeze_path) == self.freeze_sha256, 'Design freeze changed')
        row = self.requests[request_id]
        require(file_sha(safe_path(self.root, row['image_file'])) == row['image_sha256'],
                'Image changed after registration')
        return copy.deepcopy(row)


def authorize_request(request, endpoint, *, admission=None, canary=None, terminal=None):
    """Stage gate, called with independently verified evidence objects by driver.

    Receipt authenticity/SHA verification remains the driver's responsibility;
    this function is not sufficient authorization for production on its own.
    """
    require(endpoint['experiment_id'] == EXPERIMENT, 'Wrong endpoint experiment')
    require(endpoint['start_seed'] == request.get('start_seed', request.get('train_seed')),
            'Wrong starting model')
    require(endpoint['arm'] == request['arm'], 'Wrong path/arm')
    if request['scope'] == 'TRAIN':
        require(endpoint['step'] + 1 == request['step'], 'Stale/future training policy')
        require(admission and admission.get('status') == 'ADMITTED_FIRST_TWO_STEPS_ONLY',
                'Training has no verified admission')
        if request['step'] > 2:
            require(canary and canary.get('status') == 'CANARY_RESUME_PASS', 'Canary not passed')
    else:
        require(endpoint['step'] == request['step'], 'Wrong evaluation checkpoint')
        if request['scope'] == 'MAIN':
            require(terminal and terminal.get('all_paths_terminal') is True, 'MAIN remains sealed')


class GenerationLedger:
    """Exclusive BEGIN survives failure. No automatic retry or resampling."""
    def __init__(self, run_dir):
        self.root = Path(run_dir).resolve(strict=True)
        require(self.root.is_dir(), 'Missing isolated output directory')

    def begin(self, request, identity, freeze_sha256):
        stem = hashlib.sha256(request['request_id'].encode()).hexdigest()
        prefix = self.root / stem
        require(not any(self.root.glob(stem + '.*')), 'Existing attempt: audit, never resample')
        write_once(prefix.with_suffix('.BEGIN.json'), dict(request_id=request['request_id'],
            seed=request['seed'], endpoint=identity, design_freeze_sha256=freeze_sha256,
            status='ATTEMPTED_UNKNOWN'))
        return prefix


class NativeKernelRunner:
    """Invoke pinned native sampler; the supplied runtime must be constructed separately."""
    def __init__(self, runtime, native, legacy_backend, balanced_cut, *, source_hashes):
        for module in (native, legacy_backend):
            path = str(Path(module.__file__).resolve())
            require(path in source_hashes and file_sha(path) == source_hashes[path], 'Unbound native source')
        require(runtime.identity['dtype'] == 'float32', 'FP32 runtime required')
        recipe = dict(max_new_tokens=1536, temperature=1., top_p=1., top_k=0,
                      do_sample=True, num_beams=1)
        require(all(getattr(runtime.generation_config, key, None) == value
                    for key, value in recipe.items()), 'Native generation recipe changed')
        self.runtime, self.native, self.legacy = runtime, native, legacy_backend
        self.balanced_cut = balanced_cut

    def __call__(self, request, inputs_root, capture):
        safe = {k: request[k] for k in PUBLIC}
        helpers = self.legacy.prefix_helpers(request['assistant_prefill'], self.native.decode_generated,
                                            self.native.BatchedProtocolStop, self.balanced_cut)
        kernel = self.legacy.bind_native_kernel(self.native.Runtime._generate_native_group, helpers)
        self.runtime.bind_request(request)
        try:
            with self.native.independent_draw_rng(self.runtime.torch, [request['seed']]):
                kernel(self.runtime, safe, inputs_root, [request['seed']],
                       protocol='structured', on_completion=capture)
        finally:
            self.runtime.unbind_request()


class GenerationBridge:
    """Original token capture plus no-retry persistence, without optimizer access.

    Runner and endpoint factory are injected for integration/testing. They are
    not proof of a production runtime or GPU acceptance; caller must bind their
    sources and verify all original checkpoint/reference identities.
    """
    def __init__(self, registry, run_dir, runner, endpoint):
        self.registry, self.runner, self.endpoint = registry, runner, endpoint
        self.ledger = GenerationLedger(run_dir)

    def generate_one(self, request_id, **gates):
        request = self.registry.request(request_id)
        before = copy.deepcopy(self.endpoint())
        authorize_request(request, before, **gates)
        prefix = self.ledger.begin(request, before, self.registry.freeze_sha256)
        captured = []
        def capture(row):
            require(not captured, 'More than one native completion')
            captured.append(copy.deepcopy(row))
            write_once(prefix.with_suffix('.CAPTURE.json'), row)
        try:
            self.runner(request, self.registry.root, capture)
            require(len(captured) == 1, 'Missing native completion')
            row = captured[0]
            require(row['finish_reason'] in ('balanced', 'eos', 'length'), 'Unknown finish')
            tokens, probs = row['tokens'], row['sampler_logprobs']
            require(type(tokens) is list and 1 <= len(tokens) <= 1536 and
                    all(type(t) is int and t >= 0 for t in tokens), 'Invalid original tokens')
            require(row.get('raw_tokens') == tokens and row.get('completion_token_count') == len(tokens),
                    'Native original token aliases differ')
            require(row.get('sampler_logprob_source') == 'actual_generate_raw_logits_selected_token',
                    'Sampler probabilities are not original raw logits')
            require(len(tokens) == len(probs) and all(type(p) in (int, float) and math.isfinite(p)
                    for p in probs), 'Invalid original sampler logprobs')
            require(row['assistant_prefill'] == request['assistant_prefill'] and
                    row['full_text'] == request['assistant_prefill'] + row['generated_text'], 'Serialization mismatch')
            require(row.get('generation_status') == 'COMPLETE' and
                    row.get('technical_validation_errors') == [], 'Native technical failure')
            require(self.endpoint() == before, 'Endpoint changed during generation')
            row.update(request_id=request_id, seed=request['seed'],
                       native_capture_path=str(prefix.with_suffix('.CAPTURE.json')),
                       native_capture_sha256=file_sha(prefix.with_suffix('.CAPTURE.json')),
                       policy_sha256=before['policy_sha256'], reference_sha256=before['reference_sha256'],
                       design_freeze_sha256=self.registry.freeze_sha256)
            write_once(prefix.with_suffix('.COMPLETE.json'), row)
            return row
        except Exception as exc:
            write_once(prefix.with_suffix('.ERROR.json'), dict(error_type=type(exc).__name__,
                message=str(exc), status='HOLD_TECHNICAL', recovery='AUDIT_EXISTING_ATTEMPT_NO_RESAMPLING'))
            raise
