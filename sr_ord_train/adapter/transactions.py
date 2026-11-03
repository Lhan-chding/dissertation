"""Independent SR-ORD-TRAIN path transactions; never authorizes an old phase.

Hold the exclusive lock for generation, update, and commit. A durable BEGIN
without COMPLETE is UNKNOWN and is never retried automatically. Completed raw
bytes can only be reused under the exact request, path, step and policy identity.
A staging directory is an unresolved post-update transaction, not a checkpoint.

API: PathTransactions(directory, identity), begin_attempt(request, policy_sha256,
step), complete_attempt(attempt_id, raw), published(), commit(runtime, optimizer,
scheduler, reference, step=..., sampler_state=..., update=None), restore_latest(...).
Sampler state requires next_step, committed_step, generated_responses; all extra
cursor fields are retained. Only a fully published COMMIT may advance the driver.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import uuid

SCHEMA = 'sr-ord-train-transaction-v1'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def require(value, reason):
    if not value:
        raise PermissionError(reason)


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[a-f0-9]{64}', value) is not None


def _plain(path):
    path = Path(os.path.abspath(path))
    for item in (*reversed(path.parents), path):
        require(not item.is_symlink(), 'Symlink forbidden: ' + str(item))
    return path


def _read(path):
    fd = os.open(_plain(path), os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as f:
        require(stat.S_ISREG(os.fstat(f.fileno()).st_mode), 'Regular file required')
        return f.read()


def read_json(path):
    def pairs(items):
        require(len(dict(items)) == len(items), 'Duplicate JSON key')
        return dict(items)
    return json.loads(_read(path), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def file_hash(path):
    h = hashlib.sha256()
    fd = os.open(_plain(path), os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as f:
        require(stat.S_ISREG(os.fstat(f.fileno()).st_mode), 'Regular file required')
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _sync_dir(path):
    fd = os.open(_plain(path), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_once(path, value):
    path = _plain(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as f:
        f.write(canonical(value) + b'\n')
        f.flush()
        os.fsync(f.fileno())
    _sync_dir(path.parent)


def _finite(value):
    if hasattr(value, 'detach'):
        import torch
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def _rng_pack(rng):
    result = copy.deepcopy(rng)
    n = rng['numpy']
    result['numpy'] = (n[0], n[1].tolist(), n[2], n[3], n[4])
    return result


def _rng_unpack(rng):
    import numpy as np
    result = copy.deepcopy(rng)
    n = rng['numpy']
    result['numpy'] = (n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4])
    return result


class PathTransactions:
    def __init__(self, directory, identity):
        self.directory = _plain(directory)
        require(isinstance(identity, dict), 'Path identity required')
        require(identity.get('experiment_id') == 'SR-ORD-TRAIN-1', 'Wrong experiment')
        require(identity.get('parent') in ('b0', 'b1'), 'Wrong parent')
        require(identity.get('arm') in ('R', 'G', 'D'), 'Wrong training arm')
        require(type(identity.get('train_seed')) is int and identity['train_seed'] in (11031, 11032),
                'Wrong training seed')
        require(isinstance(identity.get('run_id'), str) and identity['run_id'], 'run_id required')
        for key in ('manifest_sha256', 'plan_sha256', 'source_sha256'):
            require(_sha(identity.get(key)), key + ' required')
        canonical(identity)
        self.identity = copy.deepcopy(identity)
        self._fd = None
        self._attempts = None
        self._markers = None

    def __enter__(self):
        require(self._fd is None, 'Already locked')
        self.directory.mkdir(parents=True, exist_ok=True)
        _plain(self.directory)
        fd = os.open(self.directory / 'LOCK', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        try:
            for name in ('attempts', 'checkpoints'):
                p = _plain(self.directory / name)
                p.mkdir(exist_ok=True)
            p = self.directory / 'IDENTITY.json'
            if p.exists():
                require(read_json(p) == self.identity, 'Existing path identity differs')
            else:
                write_once(p, self.identity)
            self._load_attempts()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *unused):
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None
        self._attempts = self._markers = None

    def _locked(self):
        require(self._fd is not None, 'Exclusive path lock required')
        require(read_json(self.directory / 'IDENTITY.json') == self.identity, 'Path identity changed')

    def _append(self, event):
        p = _plain(self.directory / 'ATTEMPTS.jsonl')
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'ab') as f:
            require(stat.S_ISREG(os.fstat(f.fileno()).st_mode), 'Regular journal required')
            f.write(canonical(event) + b'\n')
            f.flush()
            os.fsync(f.fileno())
        _sync_dir(self.directory)

    def _load_attempts(self):
        self._attempts = {}
        self._journal_tail = '0' * 64
        p = self.directory / 'ATTEMPTS.jsonl'
        if not p.exists():
            return
        data = _read(p)
        require(not data or data.endswith(b'\n'), 'Partial attempt journal: audit UNKNOWN first')
        previous = '0' * 64
        for line in data.splitlines():
            event = json.loads(line)
            require(event.get('previous_sha256') == previous, 'Attempt journal chain differs')
            previous = digest(event)
            aid = event['attempt_id']
            require(_sha(aid), 'Unsafe attempt identifier')
            if event['event'] == 'BEGIN':
                context = {key: event[key] for key in ('identity_sha256', 'request', 'policy_sha256', 'step')}
                require(digest(context) == aid, 'Attempt registration content differs')
                require(aid not in self._attempts, 'Duplicate attempt BEGIN')
                require(event['identity_sha256'] == digest(self.identity), 'Attempt path differs')
                self._attempts[aid] = {'begin': event}
            elif event['event'] == 'COMPLETE':
                require(aid in self._attempts and 'complete' not in self._attempts[aid],
                        'Invalid attempt COMPLETE')
                self._attempts[aid]['complete'] = event
                self._verify_raw(self._attempts[aid])
            else:
                raise PermissionError('Unknown attempt event')
        self._journal_tail = previous

    def _event(self, value):
        event = dict(value, previous_sha256=getattr(self, '_journal_tail', '0' * 64))
        self._append(event)
        self._journal_tail = digest(event)
        return event

    def _verify_raw(self, attempt):
        event = attempt['complete']
        p = self.directory / 'attempts' / (event['attempt_id'] + '.json')
        require(file_hash(p) == event['raw_sha256'], 'Completed raw bytes differ')
        raw = read_json(p)
        require(digest(raw) == event['raw_content_sha256'], 'Completed raw content differs')
        return raw

    def begin_attempt(self, request, policy_sha256, step):
        self._locked()
        require(_sha(policy_sha256), 'Actual policy hash required')
        require(type(step) is int and 1 <= step <= 24, 'Invalid target training step')
        require(isinstance(request, dict) and isinstance(request.get('request_id'), str)
                and request['request_id'] and type(request.get('seed')) is int, 'Request and seed required')
        for key in ('parent', 'arm', 'train_seed'):
            require(request.get(key) == self.identity[key], 'Request path differs: ' + key)
        require(request.get('step') == step, 'Request target step differs')
        context = dict(identity_sha256=digest(self.identity), request=copy.deepcopy(request),
                       policy_sha256=policy_sha256, step=step)
        aid = digest(context)
        if aid in self._attempts:
            attempt = self._attempts[aid]
            require('complete' in attempt, 'UNKNOWN attempt: audit original call; automatic retry forbidden')
            return dict(status='COMPLETE', attempt_id=aid, raw=self._verify_raw(attempt))
        require(not any(a['begin']['request']['request_id'] == request['request_id']
                        for a in self._attempts.values()), 'Request already attempted under another identity')
        require(not any('complete' not in a for a in self._attempts.values()),
                'UNKNOWN attempt blocks new path calls')
        markers = self.published()
        require(markers and markers[-1]['step'] + 1 == step, 'Commit step zero before generation; cursor differs')
        require(markers[-1]['policy_sha256'] == policy_sha256, 'Sampling policy differs from latest commit')
        event = self._event(dict(event='BEGIN', attempt_id=aid, **context))
        self._attempts[aid] = {'begin': event}
        return dict(status='NEW', attempt_id=aid)

    def complete_attempt(self, attempt_id, raw):
        self._locked()
        require(attempt_id in self._attempts, 'Missing durable BEGIN')
        attempt = self._attempts[attempt_id]
        require('complete' not in attempt, 'Already completed; raw is immutable')
        require(isinstance(raw, dict), 'Original raw object required')
        begin = attempt['begin']
        request = begin['request']
        require(raw.get('request_id', raw.get('row_id')) == request['request_id'], 'Raw request differs')
        require(raw.get('seed') == request['seed'], 'Raw actual seed differs')
        require(raw.get('policy_hash') == begin['policy_sha256'], 'Raw policy differs')
        require(isinstance(raw.get('tokens'), list) and 1 <= len(raw['tokens']) <= 1536,
                'Original token path required')
        require(all(type(v) is int and v >= 0 for v in raw['tokens']), 'Invalid sampled token')
        require(isinstance(raw.get('sampler_logprobs'), list)
                and len(raw['sampler_logprobs']) == len(raw['tokens'])
                and all(type(v) in (int, float) and math.isfinite(v) for v in raw['sampler_logprobs']),
                'Original finite sampler probabilities required')
        require(isinstance(raw.get('full_text'), str), 'Full original response required')
        p = self.directory / 'attempts' / (attempt_id + '.json')
        write_once(p, raw)
        event = self._event(dict(event='COMPLETE', attempt_id=attempt_id,
                                 raw_sha256=file_hash(p), raw_content_sha256=digest(raw)))
        attempt['complete'] = event
        return copy.deepcopy(event)

    def published(self):
        self._locked()
        checkpoints = _plain(self.directory / 'checkpoints')
        require(not list(checkpoints.glob('.staging-*')), 'Unresolved checkpoint staging: audit before resume')
        if self._markers is not None:
            return copy.deepcopy(self._markers)
        markers = []
        for p in sorted(checkpoints.iterdir()):
            _plain(p)
            require(p.is_dir() and re.fullmatch(r'step-\d{3}', p.name), 'Unknown checkpoint entry')
            marker = read_json(p / 'COMMIT.json')
            require(marker.get('schema') == SCHEMA and marker.get('identity') == self.identity,
                    'Checkpoint path identity differs')
            require(type(marker.get('step')) is int and marker['step'] == len(markers) and p.name == f"step-{len(markers):03d}",
                    'Noncontiguous committed checkpoints')
            require(file_hash(p / 'state.pt') == marker['state_sha256'], 'Checkpoint bytes differ')
            previous = markers[-1] if markers else None
            require(marker.get('previous_commit_sha256') == (previous['commit_sha256'] if previous else None),
                    'Checkpoint chain differs')
            if marker['step']:
                require(file_hash(p / 'UPDATE.json') == marker['update_sha256'], 'Update bytes differ')
                require(marker['policy_before_sha256'] == previous['policy_sha256'], 'Policy chain differs')
                require(marker['reference_sha256'] == previous['reference_sha256'], 'Reference chain differs')
                for ref in marker['attempts']:
                    attempt = self._attempts.get(ref['attempt_id'])
                    require(attempt is not None and 'complete' in attempt, 'Committed attempt missing')
                    require(attempt['complete']['raw_sha256'] == ref['raw_sha256'], 'Committed raw differs')
                    self._verify_raw(attempt)
                require(len(marker['attempts']) == 128, 'Commit must retain 128 raw responses')
            marker['commit_sha256'] = file_hash(p / 'COMMIT.json')
            markers.append(marker)
        self._markers = markers
        return copy.deepcopy(markers)

    def _check_runtime(self, runtime):
        for key, expected in [('experiment_id', 'SR-ORD-TRAIN-1'), ('run_id', self.identity['run_id']),
                              ('parent_id', self.identity['parent']), ('arm', self.identity['arm']),
                              ('train_seed', self.identity['train_seed']), ('source_sha256', self.identity['source_sha256'])]:
            require(runtime.identity.get(key) == expected, 'Runtime path differs: ' + key)

    def commit(self, runtime, optimizer, scheduler, reference, *, step, sampler_state, update=None):
        self._locked()
        import torch
        from mm_core.training import capture_rng, state_hash, trainable_state
        from sr_f3_ce.training import validate_optimizer
        markers = self.published()
        require(type(step) is int and step == len(markers) and 0 <= step <= 24, 'Commit cursor differs')
        self._check_runtime(runtime)
        require(runtime.identity.get('step') == max(0, step - 1), 'Runtime cursor advanced before COMMIT')
        require(sampler_state.get('next_step') == step + 1 and sampler_state.get('committed_step') == step
                and sampler_state.get('generated_responses') == step * 128, 'Sampler cursor differs')
        canonical(sampler_state)
        validate_optimizer(optimizer, scheduler, model=runtime.model)
        require(scheduler.state_dict().get('last_epoch') == step, 'Scheduler cursor differs')
        policy = trainable_state(runtime.model)
        require(policy and set(policy) == set(reference), 'Policy/reference adapter inventory differs')
        require(all(p.dtype == torch.float32 and reference[n].dtype == torch.float32
                    and p.shape == reference[n].shape and not reference[n].requires_grad for n, p in policy.items()), 'Adapter/reference dtype or shape differs')
        require(_finite(policy) and _finite(reference) and _finite(optimizer.state_dict())
                and _finite(scheduler.state_dict()), 'Nonfinite post-update checkpoint')
        policy_hash, reference_hash = state_hash(policy), state_hash(reference)
        require(runtime.identity.get('reference_hash') == reference_hash, 'Frozen reference differs')
        refs = []
        if step == 0:
            require(update is None and not optimizer.state and not self._attempts, 'Step zero must precede all attempts/updates')
            require(policy_hash == reference_hash, 'Step zero differs from parent')
        else:
            require(isinstance(update, dict) and update.get('status') == 'UPDATED_UNCOMMITTED'
                    and update.get('precommit_gate') == 'PASS_BEFORE_ADAM'
                    and update.get('optimizer_updates') == 1 and update.get('sequence_count') == 128
                    and update.get('step') == step,
                    'Missing completed gated update')
            from runtime_contract import check_numerics
            check_numerics(update['numerics'])
            require(update['policy_before_sha256'] == markers[-1]['policy_sha256']
                    and update['policy_after_sha256'] == policy_hash and update['reference_sha256'] == reference_hash,
                    'Update content identity differs')
            require(reference_hash == markers[-1]['reference_sha256'], 'Reference moved')
            for aid, attempt in self._attempts.items():
                if attempt['begin']['step'] != step:
                    continue
                require('complete' in attempt, 'UNKNOWN attempt blocks commit')
                require(attempt['begin']['policy_sha256'] == markers[-1]['policy_sha256'], 'Attempt policy differs')
                self._verify_raw(attempt)
                refs.append(dict(attempt_id=aid, raw_sha256=attempt['complete']['raw_sha256']))
            require(len(refs) == 128, 'Exactly 128 complete original responses required')
            require(optimizer.state and all(float(v.get('step', -1)) == step for v in optimizer.state.values()),
                    'Adam cursor differs')
        rng = capture_rng()
        state = dict(schema=SCHEMA, identity=self.identity, step=step, policy=policy,
                     reference_sha256=reference_hash, optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(), rng=_rng_pack(rng), sampler_state=copy.deepcopy(sampler_state))
        folder = self.directory / 'checkpoints' / ('.staging-' + uuid.uuid4().hex)
        folder.mkdir()
        _sync_dir(folder.parent)
        with (folder / 'state.pt').open('xb') as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
        marker = dict(schema=SCHEMA, identity=self.identity, step=step, policy_sha256=policy_hash,
                      reference_sha256=reference_hash, rng_sha256=state_hash(rng),
                      optimizer_sha256=state_hash(state['optimizer']), scheduler_sha256=state_hash(state['scheduler']),
                      sampler_state=copy.deepcopy(sampler_state), state_sha256=file_hash(folder / 'state.pt'),
                      previous_commit_sha256=markers[-1]['commit_sha256'] if markers else None,
                      attempts=sorted(refs, key=lambda x: x['attempt_id']))
        if step:
            write_once(folder / 'UPDATE.json', update)
            marker.update(update_sha256=file_hash(folder / 'UPDATE.json'), policy_before_sha256=markers[-1]['policy_sha256'])
        write_once(folder / 'COMMIT.json', marker)
        _sync_dir(folder)
        destination = folder.parent / f'step-{step:03d}'
        require(not destination.exists(), 'Committed checkpoint is immutable')
        os.rename(folder, destination)
        _sync_dir(destination.parent)
        marker['commit_sha256'] = file_hash(destination / 'COMMIT.json')
        self._markers = markers + [marker]
        return copy.deepcopy(marker)

    def restore_latest(self, runtime, optimizer, scheduler, reference):
        self._locked()
        import torch
        from mm_core.training import restore_rng, capture_rng, state_hash, trainable_state
        from sr_f3_ce.training import validate_optimizer
        self._check_runtime(runtime)
        markers = self.published()
        if not markers:
            return None
        require(not any('complete' not in a for a in self._attempts.values()),
                'UNKNOWN attempt requires audit before restore/resampling')
        marker = markers[-1]
        require(state_hash(reference) == marker['reference_sha256'], 'Restore parent reference differs')
        path = self.directory / 'checkpoints' / f"step-{marker['step']:03d}" / 'state.pt'
        require(file_hash(path) == marker['state_sha256'], 'State changed before restore')
        state = torch.load(path, map_location='cpu', weights_only=True)
        require(state['schema'] == SCHEMA and state['identity'] == self.identity
                and state['step'] == marker['step'] and state['sampler_state'] == marker['sampler_state'],
                'Serialized checkpoint metadata differs')
        require(state['reference_sha256'] == marker['reference_sha256']
                and state_hash(state['policy']) == marker['policy_sha256']
                and state_hash(state['optimizer']) == marker['optimizer_sha256']
                and state_hash(state['scheduler']) == marker['scheduler_sha256'], 'Serialized content differs')
        rng = _rng_unpack(state['rng'])
        require(state_hash(rng) == marker['rng_sha256'], 'Serialized RNG differs')
        parameters = {n: p for n, p in runtime.model.named_parameters() if p.requires_grad}
        require(set(parameters) == set(state['policy']) and _finite(state['policy'])
                and _finite(state['optimizer']), 'Restore parameter inventory/nonfinite state')
        with torch.no_grad():
            for name, value in state['policy'].items():
                require(value.dtype == torch.float32 and parameters[name].dtype == torch.float32
                        and value.shape == parameters[name].shape, 'Restore parameter dtype/shape differs')
                parameters[name].copy_(value)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        validate_optimizer(optimizer, scheduler, model=runtime.model)
        require(state_hash(optimizer.state_dict()) == marker['optimizer_sha256']
                and state_hash(scheduler.state_dict()) == marker['scheduler_sha256'], 'Loaded optimizer/scheduler differs')
        require(state_hash(trainable_state(runtime.model)) == marker['policy_sha256'], 'Loaded policy differs')
        restore_rng(rng)
        require(state_hash(capture_rng()) == marker['rng_sha256'], 'Loaded RNG differs')
        runtime.identity.update(step=marker['step'], policy_hash=marker['policy_sha256'])
        return copy.deepcopy(marker)
