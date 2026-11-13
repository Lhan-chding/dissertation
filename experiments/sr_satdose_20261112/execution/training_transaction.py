"""Durable, fail-closed storage for the fixed SATDOSE training transactions.

This is NOT a training driver or an admission check. An admitted future driver
must supply native optimizer/scheduler callbacks and the training_state codec.
No generation, optimizer construction, GPU claim, recovery, or automatic retry
is implemented here. A callback may have executed after UPDATE_INTENT even if
publication failed: every such unpublished staging directory remains UNKNOWN.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re

from . import training_state

EXPERIMENT = 'SR-SATDOSE-1-20261112'
STARTS = (11031, 11032)
ARMS = ('T0', 'T1')
SCHEMA = 'sr-satdose-training-transaction-v1'


def require(ok, message):
    if not ok:
        raise PermissionError('HOLD: '+message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def loads(text):
    def pairs(items):
        result = dict(items)
        require(len(result) == len(items), 'duplicate JSON key')
        return result
    return json.loads(text, object_pairs_hook=pairs,
        parse_constant=lambda value: require(False, 'nonfinite JSON '+value))


def read_json(path):
    return loads(Path(path).read_text())


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_once(path, value):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(canonical(value)+b'\n'); stream.flush(); os.fsync(stream.fileno())
    fsync_directory(path.parent)


def bound(root, reference):
    relative = Path(reference['path'])
    require(not relative.is_absolute() and '..' not in relative.parts, 'artifact escapes root')
    path = root/relative
    require(path.resolve(strict=True) == path and path.is_file() and not path.is_symlink(), 'noncanonical artifact')
    require(re.fullmatch('[0-9a-f]{64}', reference['sha256']) and file_sha(path) == reference['sha256'],
            'artifact SHA mismatch: '+str(relative))
    return path


def reference(root, path):
    require(path.resolve(strict=True).is_relative_to(root), 'reference escapes root')
    return dict(path=str(path.relative_to(root)), sha256=file_sha(path))


def rows(path):
    return [loads(line) for line in path.read_text().splitlines() if line.strip()]


@contextmanager
def exclusive_lock(root):
    require(root.resolve(strict=True) == root, 'transaction root must be canonical')
    descriptor = os.open(root/'TRANSACTION.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PermissionError('HOLD: another training transaction holds the lock') from exc
        yield
    finally:
        os.close(descriptor)


class _UpdateLease:
    """One in-process invocation lease, issued only after durable UPDATE_INTENT.

    This limits callbacks; it is not proof of native Adam math or GPU admission.
    The separate state validator must authenticate the resulting native state.
    """
    def __init__(self, stage, identity, step):
        self.stage, self.identity, self.step = stage, copy.deepcopy(identity), step
        self.used = False

    def consume(self, optimizer_step, scheduler_step, fault):
        require(not self.used, 'update lease already consumed')
        intent = read_json(self.stage/'UPDATE_INTENT.json')
        require(intent['identity'] == self.identity and intent['step'] == self.step,
                'durable update lease identity differs')
        self.used = True  # consume BEFORE calling user/native code, including exceptions
        optimizer_step(); fault('after_optimizer_step')
        scheduler_step(); fault('after_scheduler_step')
        return dict(optimizer_callback_invocations=1, scheduler_callback_invocations=1)


class TransactionStore:
    """Fixed 2 starts x 2 arms x 24 steps, with a global exclusive publish lock."""
    def __init__(self, evidence_root, freeze, expected_freeze_sha256, identity, *, native, genesis_reference):
        self.root = Path(evidence_root).resolve(strict=True)
        self.freeze_path = Path(freeze).resolve(strict=True)
        require(self.freeze_path == self.root/'DESIGN_FREEZE.json' and
                file_sha(self.freeze_path) == expected_freeze_sha256, 'wrong frozen design')
        self.freeze_sha = expected_freeze_sha256
        self.identity = copy.deepcopy(identity)
        self.genesis_reference = copy.deepcopy(genesis_reference)
        self.native = native
        require(identity.get('experiment_id') == EXPERIMENT and identity.get('parent') == 'b0' and
                type(identity.get('train_seed')) is int and identity['train_seed'] in STARTS and
                identity.get('arm') in ARMS, 'foreign training identity')
        training_state._validate_identity(self.identity)
        self.start, self.arm = identity['train_seed'], identity['arm']
        self.store = self.root/'training_transactions'
        require(not self.store.is_symlink(), 'transaction root symlink')
        if not self.store.exists():
            self.store.mkdir(); fsync_directory(self.root)
        self.path = self.store/f'start-{self.start}'/self.arm
        self._design()

    def _design(self):
        require(file_sha(self.freeze_path) == self.freeze_sha, 'freeze changed')
        freeze = read_json(self.freeze_path)
        require(freeze['schema'] == 'sr-satdose-design-freeze-v1' and freeze['experiment_id'] == EXPERIMENT,
                'foreign design')
        config_path = bound(self.root, freeze['artifacts']['config'])
        config = read_json(config_path)
        require(file_sha(config_path) == freeze['config_sha256'] and config['experiment_id'] == EXPERIMENT and
                config['steps'] == 24 and config['groups_per_step'] == 16 and config['draws_per_group'] == 8,
                'changed finite training schedule')
        require(self.identity.get('source_binding') == config['start_bindings'][str(self.start)],
                'identity does not bind original R24/source reference')
        manifest = rows(bound(self.root, freeze['artifacts']['train_manifest']))
        groups = rows(bound(self.root, freeze['artifacts']['training_groups']))
        require(len(manifest) == 12288 and len({r['request_id'] for r in manifest}) == 12288 and
                len(groups) == 1536 and len({g['group_id'] for g in groups}) == 1536, 'incomplete/duplicate frozen schedule')
        populations = {(start, arm, step): [] for start in STARTS for arm in ARMS for step in range(1,25)}
        for row in manifest:
            key = (row.get('start_seed'), row.get('arm'), row.get('step'))
            require(key in populations and row.get('scope', row.get('split')) == 'TRAIN', 'foreign frozen request')
            populations[key].append(row)
        require(all(len(population) == 128 for population in populations.values()), 'non-128 frozen step')
        selected = [g for g in groups if g['start_seed'] == self.start and g['arm'] == self.arm]
        self.plans = {}
        for step in range(1,25):
            own = [g for g in selected if g['step'] == step]
            originals = populations[(self.start, self.arm, step)]
            lookup = {r['request_id']: r for r in originals}
            require(len(own) == 16 and len({g['root_id'] for g in own}) == 16, 'incomplete/duplicate G8 roots')
            ids = [rid for g in own for rid in g['request_ids']]
            require(len(ids) == len(set(ids)) == 128 and set(ids) == set(lookup) and
                    all(len(g['request_ids']) == 8 for g in own), 'incomplete/foreign original G8')
            for group in own:
                for rid in group['request_ids']:
                    require(lookup[rid]['group_id'] == group['group_id'] and lookup[rid]['root_id'] == group['root_id'],
                            'frozen request/group disagreement')
            self.plans[step] = [lookup[rid] for rid in ids]
        # Real TrainingHandle identities always bind their execution inventory.
        # Such identities MUST supply native capture/update evidence. Identities
        # without that field are the lower-level transaction/CPU fixture contract,
        # never sufficient to enter training_step.run_step.
        if 'execution_sources_sha256' in self.identity:
            require(re.fullmatch('[0-9a-f]{64}',self.identity['execution_sources_sha256'] or '') is not None,
                    'invalid native execution source identity')
        self.source_lock = freeze['artifacts']['source_lock']
        bound(self.root, self.source_lock)
        genesis = read_json(self._external(self.genesis_reference))
        source = self.identity['source_binding']
        require(genesis.get('schema') == 'sr-satdose-training-genesis-v1' and
                genesis.get('status') == 'FRESH_R24_FORK_NOT_TRAINING_ADMISSION' and
                genesis.get('permission') == 'NO_TRAINING_ADMISSION' and genesis.get('identity') == self.identity and
                genesis.get('design_freeze_sha256') == self.freeze_sha and genesis.get('source_binding') == source and
                all(type(genesis.get(k)) is int and genesis[k] == 0 for k in
                    ('optimizer_steps','scheduler_steps','parameter_updates')), 'invalid original training GENESIS')
        expected_endpoint = dict(experiment_id=EXPERIMENT,start_seed=self.start,arm=self.arm,step=0,
            policy_sha256=source['policy_sha256'],reference_sha256=source['reference_sha256'])
        require(all(genesis.get('endpoint', {}).get(k) == v for k,v in expected_endpoint.items()), 'GENESIS endpoint differs')
        fork = genesis.get('fork_receipt', {})
        expected_fork = dict(schema='sr-satdose-fork-v1',experiment_id=EXPERIMENT,start_seed=self.start,arm=self.arm,
            source_state_path=source['original_state_path'],source_state_sha256=source['state_sha256'],
            policy_sha256=source['policy_sha256'],reference_sha256=source['reference_sha256'],
            optimizer_created=True,source_optimizer_restored=False,trainable_tensors=496,model_calls=0,optimizer_steps=0)
        require(all(fork.get(k) == v for k,v in expected_fork.items()) and
                all(re.fullmatch('[0-9a-f]{64}',genesis.get(k,'') or '') and genesis[k] == fork.get(k)
                    for k in ('optimizer_sha256','scheduler_sha256','rng_sha256')), 'GENESIS fresh fork receipt differs')

    def _rollout(self, receipt_ref, step, policy_sha):
        receipt_path = self._external(receipt_ref)
        receipt = read_json(receipt_path)
        require(not (receipt_path.parent/'HOLD.json').exists() and not (receipt_path.parent/'ERROR.json').exists() and
                not any(receipt_path.parent.glob('*.ERROR.json')), 'failed rollout cannot commit')
        require(receipt.get('schema') == 'sr-satdose-training-rollout-v1' and
                receipt.get('status') == 'COMPLETE_NOT_TRAINING_ADMISSION' and receipt.get('identity') == self.identity and
                receipt.get('step') == step and receipt.get('responses') == 128 and
                receipt.get('design_freeze_sha256') == self.freeze_sha and
                receipt.get('source_lock_sha256') == self.source_lock['sha256'], 'incomplete/foreign rollout receipt')
        raw = rows(self._external(receipt['raw']))
        originals = self.plans[step]
        require(len(raw) == 128 and [r['request_id'] for r in raw] == [r['request_id'] for r in originals],
                'raw must preserve all 16 G8 original request order')
        refs = receipt['complete_files']
        require(len(refs) == 128 and len({r['path'] for r in refs}) == 128, 'missing/duplicate COMPLETE inventory')
        complete = {}
        for ref in refs:
            path = self._external(ref); value = read_json(path); rid = value['request_id']
            require(rid not in complete and path.name == hashlib.sha256(rid.encode()).hexdigest()+'.COMPLETE.json',
                    'foreign/duplicate COMPLETE identity')
            complete[rid] = value
        require(set(complete) == {r['request_id'] for r in raw}, 'COMPLETE/request coverage differs')
        for path in receipt_path.parent.glob('*.BEGIN.json'):
            value = read_json(path)
            require(value.get('request_id') in complete and path.name ==
                    hashlib.sha256(value['request_id'].encode()).hexdigest()+'.BEGIN.json', 'unknown rollout BEGIN cannot retry')
        scores = read_json(self._external(receipt['scores']))
        require(scores.get('schema') == 'sr-satdose-training-scores-v1' and scores.get('identity') == self.identity and
                scores.get('step') == step and scores.get('raw') == receipt['raw'] and
                scores.get('native_rescoring_complete') is True, 'unbound/incomplete native scoring evidence')
        scored = scores['rows']
        require(len(scored) == 128 and [r['request_id'] for r in scored] == [r['request_id'] for r in raw],
                'native scoring coverage differs')
        for original, row, score in zip(originals, raw, scored, strict=True):
            require(all(row.get(k) == v for k, v in original.items()), 'raw substitutes frozen request')
            if 'execution_sources_sha256' in self.identity:
                self._native_capture(row,step,policy_sha)
            origin = complete[row['request_id']]
            require(all(k in row and canonical(row[k]) == canonical(v) for k, v in origin.items()),
                    'raw changed original COMPLETE fields')
            require(row.get('scope') == 'TRAIN' and row.get('finish_reason') in ('balanced','eos','length') and
                    row.get('generation_status') == 'COMPLETE' and row.get('technical_validation_errors') == [] and
                    row.get('design_freeze_sha256') == self.freeze_sha and row.get('policy_sha256') == policy_sha and
                    row.get('reference_sha256') == self.identity['source_binding']['reference_sha256'],
                    'foreign/failed original capture')
            tokens = row.get('tokens'); logs = row.get('sampler_logprobs')
            require(isinstance(tokens, list) and 1 <= len(tokens) <= 1536 and
                    all(type(t) is int and t >= 0 for t in tokens) and tokens == row.get('raw_tokens') and
                    row.get('completion_token_count') == len(tokens) and isinstance(logs, list) and len(logs) == len(tokens) and
                    all(type(v) in (int,float) and math.isfinite(v) for v in logs) and
                    row.get('sampler_logprob_source') == 'actual_generate_raw_logits_selected_token',
                    'missing original token/logprob capture')
            require(score.get('native_score') == row.get('native_score') and score.get('J') == row.get('J') and
                    score.get('exact_authorised') == row.get('exact_authorised'), 'raw/scoring projection differs')
            require(type(score['J']) is int and score['J'] in (0,1) and
                    type(score['exact_authorised']) is int and score['exact_authorised'] in (0,1) and
                    score['J'] == score['native_score']['J'] and
                    (score['native_score']['known_mask']['S'] or score['exact_authorised'] == 0),
                    'invalid native-known scoring projection')
        return receipt

    def _native_capture(self, row, step, policy_sha):
        """Recheck original native bytes on every real-driver historical audit."""
        rid=row['request_id'];stem=hashlib.sha256(rid.encode()).hexdigest()
        capture=Path(row['native_capture_path'])
        require(capture.is_absolute() and capture.is_relative_to(self.root) and
                capture.name==stem+'.CAPTURE.json','foreign native CAPTURE path')
        capture=self._external(dict(path=str(capture.relative_to(self.root)),sha256=row['native_capture_sha256']))
        original=read_json(capture)
        require(all(k in row and canonical(row[k])==canonical(v) for k,v in original.items()),
                'raw changed original native CAPTURE fields')
        require(all(k in original for k in ('tokens','raw_tokens','completion_token_count','sampler_logprobs',
                'sampler_logprob_source','prompt_tensor_hash','prompt','model_input','seed','assistant_prefill',
                'full_text','generated_text','finish_reason','generation_status','technical_validation_errors')),
                'native CAPTURE lacks original token/input identity')
        require(original['prompt']==original['model_input'].get('prompt')==row['prompt'],
                'native CAPTURE prompt differs')
        begin=capture.with_name(stem+'.BEGIN.json')
        begin=self._external(dict(path=str(begin.relative_to(self.root)),sha256=row['native_begin_sha256']))
        value=read_json(begin)
        endpoint=dict(experiment_id=EXPERIMENT,start_seed=self.start,arm=self.arm,step=step-1,
            policy_sha256=policy_sha,reference_sha256=self.identity['source_binding']['reference_sha256'])
        require(value.get('request_id')==rid and value.get('seed')==row['seed'] and
                value.get('endpoint')==endpoint and value.get('design_freeze_sha256')==self.freeze_sha and
                value.get('status')=='ATTEMPTED_UNKNOWN','native BEGIN identity differs')

    def _native_update(self, commit, precommit, step, policy_before):
        """Native-driver evidence is mandatory, including on peer/idempotent reads."""
        if 'execution_sources_sha256' not in self.identity and 'native_update' not in commit:
            return  # Explicit low-level transaction-only interface, not a TRAIN driver.
        require(isinstance(commit.get('native_update'),dict),'native driver COMMIT missing NATIVE_UPDATE reference')
        value=read_json(self._external(commit['native_update']))
        require(value.get('schema')=='sr-satdose-training-native-update-v1' and
                value.get('status')=='UPDATED_UNCOMMITTED_NOT_ADMISSION' and
                value.get('identity')==self.identity and type(value.get('step')) is int and value['step']==step and
                value.get('policy_before_sha256')==policy_before and
                value.get('policy_after_sha256')==commit['policy_sha256'] and
                value.get('reference_sha256')==commit['reference_sha256'] and
                all(type(value.get(k)) is int and value[k]==1 for k in ('optimizer_calls','scheduler_calls')),
                'native update identity/policy/counter differs')
        norm=value.get('parameter_update_norm')
        require(type(norm) in (int,float) and math.isfinite(norm) and norm>=0,'invalid native update norm')
        capture=value.get('production_capture')
        require(isinstance(capture,dict) and capture==precommit.get('production_capture') and
                capture in precommit['artifacts'],'native update production capture differs from PRECOMMIT')
        self._external(capture)

    def _external(self, ref):
        path = bound(self.root, ref)
        require(not path.is_relative_to(self.store),
                'capture/precommit source must live outside renamed transaction storage')
        return path

    def _precommit(self, ref, receipt_ref, step):
        value = read_json(self._external(ref))
        require(value.get('schema') == 'sr-satdose-training-precommit-v1' and value.get('identity') == self.identity and
                value.get('step') == step and value.get('rollout_receipt') == receipt_ref and
                value.get('gradient_stage') == 'postclip' and value.get('numerical_checks_passed') is True and
                isinstance(value.get('artifacts'), list) and value['artifacts'], 'invalid postclip PRECOMMIT evidence')
        for artifact in value['artifacts']:
            self._external(artifact)
        return value

    def _storage(self, value):
        require(value.get('schema') == 'sr-satdose-storage-readiness-v1' and value.get('ready') is True and
                value.get('quota_verified') is True and value.get('writable_verified') is True and
                type(value.get('required_bytes')) is int and value['required_bytes'] > 0 and
                type(value.get('effective_available_bytes')) is int and
                value['effective_available_bytes'] >= value['required_bytes'] and
                isinstance(value.get('evidence'), list) and value['evidence'],
                'driver must verify actual quota/writable space and explicit required bytes before update')
        for ref in value['evidence']:
            self._external(ref)

    def _validate_commit(self, directory, step, previous, *, decode_state=True):
        require(directory.resolve(strict=True) == directory and directory.is_dir(), 'noncanonical committed directory')
        commit_path = directory/'COMMIT.json'; commit_sha = file_sha(commit_path); commit = read_json(commit_path)
        require(commit.get('schema') == training_state.SCHEMA and commit.get('identity') == self.identity and
                commit.get('step') == step, 'foreign committed identity/step')
        tx = commit['transaction']
        require(tx.get('schema') == SCHEMA and tx.get('previous_commit_sha256') == previous and
                tx.get('design_freeze_sha256') == self.freeze_sha and tx.get('genesis_reference') == self.genesis_reference,
                'broken COMMIT/GENESIS chain')
        expected_names = {'step_intent': 'STEP_INTENT.json', 'rollout': 'ROLLOUT.json', 'evidence': 'EVIDENCE.json',
                          'precommit': 'PRECOMMIT.json', 'storage': 'STORAGE_READY.json',
                          'update_intent': 'UPDATE_INTENT.json', 'update': 'UPDATE.json'}
        require({p.name for p in directory.iterdir()} == set(expected_names.values()) | {'state.pt', 'COMMIT.json'},
                'unknown BEGIN/update/file in checkpoint')
        data = {}
        for key, name in expected_names.items():
            require(tx[key]['path'] == name, 'relocated transaction evidence')
            data[key] = read_json(bound(directory, tx[key]))
            require(data[key].get('identity') == self.identity and data[key].get('step') == step,
                    'foreign transaction evidence')
        require(data['step_intent']['previous_commit_sha256'] == previous and
                data['step_intent']['genesis_reference'] == self.genesis_reference and
                data['update_intent']['precommit_sha256'] == tx['precommit']['sha256'] and
                data['update_intent']['storage_ready_sha256'] == tx['storage']['sha256'] and
                data['update']['update_intent_sha256'] == tx['update_intent']['sha256'] and
                data['update']['optimizer_callback_invocations'] == data['update']['scheduler_callback_invocations'] == 1,
                'broken update intent/once chain')
        evidence = data['evidence']; policy = data['step_intent']['policy_before_sha256']
        require(data['rollout']['rollout_receipt'] == evidence['rollout_receipt'] and
                data['step_intent']['request_ids'] == [r['request_id'] for r in self.plans[step]], 'original request/evidence chain differs')
        self._storage(data['storage']['value'])
        self._rollout(evidence['rollout_receipt'], step, policy)
        precommit=self._precommit(evidence['precommit_reference'], evidence['rollout_receipt'], step)
        require(data['precommit']['value'] == read_json(bound(self.root, evidence['precommit_reference'])),
                'PRECOMMIT copy differs')
        require(file_sha(directory/'state.pt') == commit['state_sha256'], 'committed state changed')
        require(commit.get('sampler_state', {}).get('committed_step') == step and
                commit['sampler_state'].get('next_step') == step+1 and
                commit['sampler_state'].get('generated_responses') == 128*step and
                commit.get('reference_sha256') == self.identity['source_binding']['reference_sha256'],
                'committed cursor/reference differs')
        self._native_update(commit,precommit,step,policy)
        if decode_state:
            training_state.validate_checkpoint(directory, commit_sha, copy.deepcopy(self.identity), self.native)
        return commit_sha, commit

    def _chain(self):
        # Never ignore leftover BEGIN/UPDATE_INTENT/staging, even on a different
        # path: an unresolved optimizer attempt needs a separate read-only audit.
        for start_dir in self.store.iterdir():
            if start_dir.name == 'TRANSACTION.lock':
                continue
            require(start_dir.name in {f'start-{s}' for s in STARTS} and start_dir.is_dir() and not start_dir.is_symlink(),
                    'unknown occupied transaction entry')
            for arm_dir in start_dir.iterdir():
                require(arm_dir.name in ARMS and arm_dir.is_dir() and not arm_dir.is_symlink(), 'unknown arm entry')
                for directory in arm_dir.iterdir():
                    require(re.fullmatch(r'step-(00[1-9]|01[0-9]|02[0-4])', directory.name) and directory.is_dir() and
                            not directory.is_symlink() and (directory/'COMMIT.json').is_file(),
                            'occupied staging/BEGIN/UPDATE without published COMMIT; no retry')
                if arm_dir != self.path and any(arm_dir.iterdir()):
                    first = sorted(arm_dir.iterdir())[0]
                    other = read_json(first/'COMMIT.json')
                    peer = TransactionStore(self.root, self.freeze_path, self.freeze_sha, other['identity'], native=self.native,
                                            genesis_reference=other['transaction']['genesis_reference'])
                    require(peer.path == arm_dir, 'foreign path COMMIT identity differs from directory')
                    # All referenced bytes/chain fields are checked globally;
                    # only this active path needs repeated state deserialization.
                    peer._published_chain(decode_state=False)
        return self._published_chain()

    def _published_chain(self, *, decode_state=True):
        if not self.path.exists():
            return []
        found = sorted(self.path.iterdir())
        require([p.name for p in found] == [f'step-{n:03d}' for n in range(1,len(found)+1)], 'gap in published COMMIT chain')
        chain = []; previous = None
        policy = self.identity['source_binding']['policy_sha256']
        for step, directory in enumerate(found, 1):
            digest, commit = self._validate_commit(directory, step, previous, decode_state=decode_state)
            intent = read_json(directory/'STEP_INTENT.json')
            require(intent['policy_before_sha256'] == policy, 'stale policy in COMMIT chain')
            chain.append((digest, commit)); previous = digest; policy = commit['policy_sha256']
        fsync_directory(self.path)
        return chain

    def inspect_committed(self, step, expected_commit_sha256):
        """Read-only full-chain inspection for runtime advance after publication.

        Call after commit_step returns, never inside its callbacks/held lock.
        No update callback or runtime construction occurs here.
        """
        require(type(step) is int and 1 <= step <= 24, 'invalid inspected step')
        with exclusive_lock(self.store):
            self._design(); chain = self._chain()
            require(step <= len(chain) and chain[step-1][0] == expected_commit_sha256,
                    'requested published COMMIT missing or SHA differs')
            return copy.deepcopy(chain[step-1][1])

    def commit_step(self, step, *, collect_evidence, prepare_update, optimizer_step,
                    scheduler_step, write_checkpoint, storage_ready, fault=None):
        """Publish one fixed update or return a fully verified idempotent COMMIT.

        collect_evidence(stage, original_requests) -> root-relative rollout ref;
        prepare_update(stage, rollout_ref) -> root-relative postclip evidence ref;
        write_checkpoint(stage, commit_fields) -> training_state codec result.
        storage_ready(stage, step) -> explicit live quota/writable-space receipt.
        Capture/precommit/storage source references must live outside the renamed
        transaction tree; their immutable hashes are copied into its COMMIT chain.
        Admission, native arithmetic and state codec are supplied by the driver.
        No callback runs on a repeated COMMIT or an occupied/unknown transaction.
        """
        require(type(step) is int and 1 <= step <= 24, 'step outside fixed 24-update budget')
        fault = fault or (lambda boundary: None)
        with exclusive_lock(self.store):
            self._design(); chain = self._chain()
            if step <= len(chain):
                return dict(status='ALREADY_COMMITTED', permission='NO_TRAINING_ADMISSION', step=step,
                    next_step=len(chain)+1, commit_sha256=chain[step-1][0], checkpoint=str(self.path/f'step-{step:03d}'))
            require(step == len(chain)+1, 'cannot skip an uncommitted step')
            for path in (self.path.parent, self.path):
                if not path.exists():
                    path.mkdir(); fsync_directory(path.parent)
            stage = self.path/f'.staging-step-{step:03d}'
            stage.mkdir(); fsync_directory(self.path)
            previous = chain[-1][0] if chain else None
            policy = chain[-1][1]['policy_sha256'] if chain else self.identity['source_binding']['policy_sha256']
            common = dict(identity=self.identity, step=step, permission='NO_TRAINING_ADMISSION')
            write_once(stage/'STEP_INTENT.json', dict(common, previous_commit_sha256=previous,
                policy_before_sha256=policy, design_freeze_sha256=self.freeze_sha,
                genesis_reference=self.genesis_reference, request_ids=[r['request_id'] for r in self.plans[step]]))
            fault('after_step_intent')
            rollout_ref = collect_evidence(stage, copy.deepcopy(self.plans[step]))
            self._rollout(rollout_ref, step, policy)
            write_once(stage/'ROLLOUT.json', dict(common, rollout_receipt=rollout_ref)); fault('after_rollout_validation')
            precommit_ref = prepare_update(stage, copy.deepcopy(rollout_ref))
            precommit = self._precommit(precommit_ref, rollout_ref, step)
            write_once(stage/'EVIDENCE.json', dict(common, rollout_receipt=rollout_ref, precommit_reference=precommit_ref))
            fault('after_evidence')
            write_once(stage/'PRECOMMIT.json', dict(common, value=precommit)); fault('after_precommit')
            storage = storage_ready(stage, step); self._storage(storage)
            write_once(stage/'STORAGE_READY.json', dict(common, value=storage)); fault('after_storage_check')
            write_once(stage/'UPDATE_INTENT.json', dict(common, precommit_sha256=file_sha(stage/'PRECOMMIT.json'),
                storage_ready_sha256=file_sha(stage/'STORAGE_READY.json'),
                status='UNKNOWN_UNTIL_ATOMIC_COMMIT_NO_RETRY'))
            fault('after_update_intent')
            counts = _UpdateLease(stage, self.identity, step).consume(optimizer_step, scheduler_step, fault)
            write_once(stage/'UPDATE.json', dict(common, **counts, update_intent_sha256=file_sha(stage/'UPDATE_INTENT.json')))
            fault('after_update')
            fields = dict(transaction=dict(schema=SCHEMA, previous_commit_sha256=previous, design_freeze_sha256=self.freeze_sha,
                genesis_reference=self.genesis_reference,
                **{key: reference(stage, stage/name) for key, name in {
                    'step_intent': 'STEP_INTENT.json', 'rollout': 'ROLLOUT.json', 'evidence': 'EVIDENCE.json',
                    'precommit': 'PRECOMMIT.json', 'storage': 'STORAGE_READY.json',
                    'update_intent': 'UPDATE_INTENT.json', 'update': 'UPDATE.json'}.items()}))
            saved = write_checkpoint(stage, copy.deepcopy(fields)); fault('after_checkpoint_write')
            require(saved['commit_sha256'] == file_sha(stage/'COMMIT.json') and
                    saved['commit'] == read_json(stage/'COMMIT.json') and saved['commit']['transaction'] == fields['transaction'],
                    'state codec returned a different COMMIT')
            self._validate_commit(stage, step, previous); fault('after_state_validation')
            for path in stage.iterdir():
                require(path.is_file() and not path.is_symlink(), 'unexpected staging content')
                with path.open('rb') as stream:
                    os.fsync(stream.fileno())
            fsync_directory(stage); fault('after_stage_fsync')
            destination = self.path/f'step-{step:03d}'
            require(not destination.exists() and stage.stat().st_dev == self.path.stat().st_dev,
                    'publish destination occupied or filesystem differs')
            os.rename(stage, destination); fault('after_rename')
            fsync_directory(self.path); fault('after_parent_fsync')
            return dict(status='COMMITTED', permission='NO_TRAINING_ADMISSION', step=step, next_step=step+1,
                        commit_sha256=saved['commit_sha256'], checkpoint=str(destination))
