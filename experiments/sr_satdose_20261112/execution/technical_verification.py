"""Read-only report assembly from independently trusted, hash-bound GPU evidence.

This module does not execute CUDA, import the producer, run an optimizer, mint
admission, or write files. The control plane must supply independently trusted
references, including this source's SHA. The completed independent numerical
audit is verified against its actual input bytes; supplementary CPU checks have
separate producer/source and input hashes. A PREFIX exception never becomes a
measurement pass. The amended admission authority must separately revalidate
DRY and the complete, source-bound PREFIX waiver chain before any generation.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re

EXPERIMENT = 'SR-SATDOSE-1-20261112'
STARTS = ('11031', '11032')
AUDITOR_SHA = 'cb3015f96a6d0149cd48cbbb0bd9283cf1f8eb83530af4dc59395ad90960a56e'
COUNTS = dict(TRAIN=12288, DRY=1280, DEV=1512, PREFIX_CALIB=96, MAIN=13760)
SUPPLEMENTARY = ('start_and_reference_hash_verified', 'native_scorer_golden_pass',
                 'disjoint_splits_checked', 'gold_never_in_model_inputs', 'dry_unchanged_policy')
GPU_CHECKS = ('native_loss_replay_equal', 'production_gradient_hook_verified',
              'sampler_teacher_gate', 'raw_token_identity', 'same_runtime_backend',
              'long_sequence_fullbatch_pass', 'allzero_pg_zero_kl_retained',
              'shadow_adam_update_replay_pass', 'evaluation_loader_policy_verified')
COSINES = ('J_vs_source_pg_cosine', 'J_split_half_cosine', 'source_split_half_cosine')
EFFECTS = ('heldout_effect_half_1', 'heldout_effect_half_2', 'effect_repeat_error_floor')


def require(ok, message):
    if not ok:
        raise ValueError('HOLD_TECHNICAL_VERIFICATION: ' + message)


def file_sha(path):
    path = Path(path)
    require(path.is_file() and path.resolve(strict=True) == path and not path.is_symlink(),
            'Noncanonical ordinary evidence: ' + str(path))
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    def pairs(items):
        require(len(dict(items)) == len(items), 'Duplicate JSON key')
        return dict(items)
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda _: require(False, 'Nonfinite JSON'))


class Evidence:
    def __init__(self, root):
        self.root = Path(root)
        require(self.root.is_absolute() and self.root.resolve(strict=True) == self.root,
                'Canonical absolute root required')
        self.observed = {}

    def path(self, name, digest):
        path = Path(name)
        require('..' not in path.parts and isinstance(digest, str) and
                re.fullmatch('[a-f0-9]{64}', digest), 'Unsafe path/digest')
        if not path.is_absolute():
            path = self.root/path
        actual = file_sha(path)
        require(actual == digest and self.observed.get(path, actual) == actual,
                'Evidence SHA changed: ' + str(path))
        self.observed[path] = actual
        return path

    def ref(self, reference, *, external=False):
        require(isinstance(reference, dict) and set(reference) == {'path', 'sha256'}, 'Malformed reference')
        require(external or not Path(reference['path']).is_absolute(), 'Expected root-relative reference')
        return self.path(reference['path'], reference['sha256'])

    def json(self, reference):
        return read_json(self.ref(reference))

    def audited(self, name, audit):
        require(name in audit['evidence'], 'Independent audit did not observe: ' + name)
        return read_json(self.path(name, audit['evidence'][name]))

    def stable(self):
        require(all(file_sha(path) == digest for path, digest in self.observed.items()),
                'Evidence changed while constructing report')


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _direction(value):
    require(value.get('direction_checks_satisfied') is True and value.get('hold_reasons') == [],
            'Direction gate failed')
    require(all(_finite(value.get(k)) and 0 < value[k] <= 1 for k in COSINES), 'Nonpositive/invalid cosine')
    require(all(_finite(value.get(k)) for k in EFFECTS) and value['effect_repeat_error_floor'] >= 0 and
            all(value[k] > value['effect_repeat_error_floor'] for k in EFFECTS[:2]), 'Heldout effect gate failed')
    require(value.get('production_replay') == dict(bit_exact=True, different_tensor_names=[]),
            'Production replay differs')
    require(_finite(value.get('recomposition_relative_l2')) and 0 <= value['recomposition_relative_l2'] <= 1e-5,
            'Original recomposition threshold failed')


def _supplementary(e, name, ref, cfg, config_ref, binding):
    proof = e.json(ref)
    require(proof.get('schema') == 'sr-satdose-supplementary-technical-proof-v1' and
            proof.get('check') == name and proof.get('passed') is True and
            proof.get('experiment_id') == EXPERIMENT and proof.get('config_sha256') == config_ref['sha256'] and
            proof.get('design_freeze_sha256') == binding['design_freeze']['sha256'], 'Foreign supplementary proof')
    verifier = e.ref(proof['verifier'], external=True)
    evidence = proof.get('evidence')
    require(isinstance(evidence, dict) and evidence and
            any(e.root/Path(p) != verifier if not Path(p).is_absolute() else Path(p) != verifier for p in evidence),
            'Missing actual supplementary inputs')
    for path, digest in evidence.items():
        e.path(path, digest)
    require(all(evidence.get(r['path']) == r['sha256'] for r in
                (config_ref, binding['design_freeze'], binding['source_lock'])),
            'Supplementary proof lacks frozen config/source inputs')
    d = proof['details']
    if name in ('start_and_reference_hash_verified', 'dry_unchanged_policy'):
        require(set(d['starts']) == set(STARTS), 'Supplementary start coverage differs')
        for start in STARTS:
            source, found = cfg['start_bindings'][start], d['starts'][start]
            require(found['policy_before_sha256'] == found['policy_after_sha256'] == source['policy_sha256'] and
                    found['reference_sha256'] == source['reference_sha256'], 'Policy/reference identity differs')
            if name == 'start_and_reference_hash_verified':
                require(found['state_sha256'] == source['state_sha256'] and
                        evidence.get(source['original_state_path']) == source['state_sha256'], 'R24 source not audited')
            else:
                require(type(found['responses']) is int and found['responses'] == 640, 'Incomplete DRY start')
        if name == 'dry_unchanged_policy':
            require(type(d['total_responses']) is int and d['total_responses'] == 1280 and
                    all(type(d[k]) is int and d[k] == 0 for k in ('optimizer_steps', 'parameter_updates')),
                    'DRY counters differ')
    elif name == 'native_scorer_golden_pass':
        require(type(d['fixture_count']) is int and d['fixture_count'] == 9 and
                type(d['failures']) is int and d['failures'] == 0 and
                d['native_source_lock_sha256'] == binding['source_lock']['sha256'], 'Golden fixtures not verified')
    elif name == 'disjoint_splits_checked':
        require(d['request_counts'] == COUNTS and
                all(d.get(k) is True for k in ('new_splits_disjoint', 'historical_exclusion_verified', 'retention_exception_only')),
                'Split/historical exclusion proof incomplete')
    else:
        require(type(d['total_requests']) is int and d['total_requests'] == 28936 and
                d['model_input_fields'] == ['image_file', 'image_sha256', 'prompt'] and
                d.get('prefix_prefill_is_registered') is True and d.get('private_metadata_absent') is True,
                'Public input boundary proof incomplete')
    return dict(passed=True, evidence=copy.deepcopy(ref), verified_scope=copy.deepcopy(d))


def _gpu_start(e, start, cfg, binding, binding_ref, audit):
    summary = audit['starts'][start]
    require(summary.get('production_rows') == 128 and type(summary.get('production_rows')) is int and
            summary.get('max_original_tokens') == (709 if start == '11031' else 1536) and
            summary.get('long_original_row') == (None if start == '11031' else 'SAT1/DRY/11032/g004/d07') and
            type(summary.get('shadow_optimizer_steps')) is int and summary['shadow_optimizer_steps'] == 2,
            'Original fullbatch/long-row/shadow scope differs')
    require(summary.get('evaluation_scope') == 'START_R24_RECEIPTS_AND_ORIGINAL_CPU_TENSOR_INVENTORY' and
            summary.get('memory_peak') == 'NOT_RECORDED_FOR_PRODUCTION' and summary.get('permission') == 'NO_TRAINING',
            'Audited scope expanded')
    _direction(summary['direction'])
    source = cfg['start_bindings'][start]
    for field, prefix in [('evaluation_loader_slots','evaluation_loader_checks'),
                          ('backend_equivalence_slots','backend_equivalence'),('allzero_slots','allzero_check')]:
        directory = f'{prefix}/start-{start}'
        require(binding[field][start] == directory and not (e.root/directory/'HOLD.json').exists(), 'Stage slot/HOLD differs')
        marker = e.audited(directory+'/COMPLETE.json', audit)
        require(marker['technical_binding_sha256'] == binding_ref['sha256'] and marker['start_seed'] == int(start) and
                marker.get('permission') == 'NO_TRAINING' and type(marker.get('optimizer_steps')) is int and
                marker['optimizer_steps'] == 0, 'Phase identity/counters differ')
        if prefix == 'evaluation_loader_checks':
            require(marker['status'] == 'EVALUATION_START_LOAD_CAPTURE_COMPLETE_NOT_GPU_ADMISSION' and
                    marker['tested_scope'] == 'START_R24_ONLY' and marker['trained_checkpoint_gpu_verified'] is False and
                    marker['trainable_tensors'] == 496 and marker['caller_rng_preserved'] is True and
                    marker['policy_sha256'] == source['policy_sha256'] and
                    marker['reference_sha256'] == source['reference_sha256'], 'START evaluation loader differs')
        elif prefix == 'backend_equivalence':
            require(marker['status'] == 'BACKEND_EQUIVALENCE_CAPTURED_NOT_GPU_ADMISSION' and
                    marker['bit_exact'] is True and marker['differences'] == [], 'Backend comparison failed')
        else:
            c = marker['comparison']
            require(marker['status'] == 'ALLZERO_CAPTURE_COMPLETE_NOT_GPU_ADMISSION' and
                    c['pg_exact_zero'] is True and c['kl_retained'] is True and c['original_scores_unchanged'] is True and
                    c['explicit_diagnostic_advantage'] == 0 and c['kl_coefficient'] == .02 and
                    _finite(c['recomposition_relative_l2']) and 0 <= c['recomposition_relative_l2'] <= 1e-5,
                    'Allzero PG/KL gate failed')
    direction_path = f'shadow_inputs/start-{start}/trial-0/DIRECTION.json'
    require(e.audited(direction_path, audit) == summary['direction'], 'Direction receipt differs from independent audit')
    for trial in (0, 1):
        directory = f'shadow_inputs/start-{start}/trial-{trial}'
        claim = f'shadow_claims/start-{start}/trial-{trial}'
        require(not any((e.root/d/'HOLD.json').exists() for d in (directory,claim)), 'Existing shadow HOLD')
        for name in ('PRODUCTION_CAPTURE.pt','RESTORE.json','ACCOUNTING.json','WORKER_COMPLETE.json'):
            require(directory+'/'+name in audit['evidence'], 'Missing actual shadow evidence: '+name)
        require(claim+'/RESULT.pt' in audit['evidence'], 'Missing actual shadow result')
        marker = e.audited(claim+'/COMPLETE.json',audit)
        require(marker['status'] == 'SHADOW_TRIAL_COMPLETE_NOT_GPU_ADMISSION' and marker['mode'] == 'production' and
                marker['technical_binding_sha256'] == binding_ref['sha256'] and marker['start_seed'] == int(start) and
                type(marker['trial']) is int and marker['trial'] == trial and
                all(type(marker[k]) is int and marker[k] == 1 for k in ('optimizer_body_calls','optimizer_steps')) and
                all(type(marker[k]) is int and marker[k] == 0 for k in ('formal_optimizer_steps','scheduler_steps')),
                'Shadow update budget/identity differs')
    return dict(policy_before_sha256=source['policy_sha256'],policy_after_sha256=source['policy_sha256'],
                **{k:summary['direction'][k] for k in (*COSINES,*EFFECTS)},
                evidence=dict(path=direction_path,sha256=audit['evidence'][direction_path]))


def build_report(root, *, config_ref, registration_ref, binding_ref, audit_ref,
                 auditor_ref, extra_proofs, source_sha256):
    """Construct a report only; callers publish bytes and authorize separately.

    extra_proofs maps exactly SUPPLEMENTARY to root-relative path/SHA refs.
    All supplied refs/source SHA are independent control-plane trust inputs.
    No file or scientific result is changed by this function.
    """
    require(file_sha(Path(__file__).resolve()) == source_sha256, 'Report constructor source SHA differs')
    e = Evidence(root)
    cfg, reg, binding, audit = (e.json(r) for r in (config_ref,registration_ref,binding_ref,audit_ref))
    require(cfg.get('experiment_id') == EXPERIMENT and reg.get('schema') == 'sr-satdose-registration-v1' and
            reg.get('mode') == 'production' and reg.get('experiment_id') == EXPERIMENT and
            reg.get('config_sha256') == config_ref['sha256'], 'Registration/config identity differs')
    require(binding.get('schema') == 'sr-satdose-technical-execution-v1' and binding.get('mode') == 'production' and
            binding.get('experiment_id') == EXPERIMENT and binding.get('experiment_root') == str(e.root) and
            binding['config'] == config_ref and reg['artifacts']['design_freeze'] == binding['design_freeze'],
            'Binding/freeze identity differs')
    freeze = e.json(binding['design_freeze'])
    require(freeze['config_sha256'] == config_ref['sha256'] and freeze['experiment_id'] == EXPERIMENT,
            'Changed design freeze')
    require(auditor_ref['sha256'] == AUDITOR_SHA, 'Unapproved independent auditor')
    e.ref(auditor_ref, external=True)
    require(audit.get('schema') == 'sr-satdose-independent-gpu-evidence-audit-v1' and
            audit.get('status') == 'FINITE_GPU_EVIDENCE_AUDITED_NOT_ADMISSION' and audit.get('permission') == 'NO_TRAINING' and
            audit.get('technical_binding_sha256') == binding_ref['sha256'] and audit.get('auditor_sha256') == AUDITOR_SHA and
            all(type(audit.get(k)) is int and audit[k] == v for k,v in
                [('completed_fixed_stages',10),('shadow_optimizer_steps',4),('formal_optimizer_steps',0),
                 ('audit_model_calls',0),('audit_optimizer_calls',0)]) and set(audit['starts']) == set(STARTS),
            'Independent audit identity/counters differ')
    require(isinstance(audit.get('evidence'),dict) and audit['evidence'].get(binding_ref['path']) == binding_ref['sha256'],
            'Audit lacks exact execution binding')
    for path,digest in audit['evidence'].items():
        e.path(path,digest)
    for prefix in ('evaluation_loader_checks','backend_equivalence','allzero_check','shadow_inputs','shadow_claims'):
        parent = e.root/prefix
        require(parent.is_dir() and parent.resolve(strict=True) == parent and
                {p.name for p in parent.iterdir()} == {'start-11031','start-11032'},
                'Unexpected or missing finite stage directory')
        for start in STARTS:
            folder = parent/f'start-{start}'
            require(folder.resolve(strict=True) == folder and folder.is_dir(), 'Noncanonical stage folder')
            if prefix in ('shadow_inputs','shadow_claims'):
                require({p.name for p in folder.iterdir()} == {'trial-0','trial-1'}, 'Unexpected shadow trial')
    gradient = {start:_gpu_start(e,start,cfg,binding,binding_ref,audit) for start in STARTS}
    require(set(extra_proofs) == set(SUPPLEMENTARY), 'Supplementary check coverage differs')
    checks = {name:_supplementary(e,name,extra_proofs[name],cfg,config_ref,binding) for name in SUPPLEMENTARY}
    for name in GPU_CHECKS:
        checks[name] = dict(passed=True,evidence=copy.deepcopy(audit_ref),
                            independently_audited=True)
    waiver, result = e.json(binding['prefix_waiver']), e.json(binding['prefix_continuation'])
    original = e.json(binding['prefix_result'])
    require(original.get('status') == 'HOLD_PREFIX_MEASUREMENT' and
            original.get('stopped_start') == 11031 and
            original.get('prefix_analyses', {}).get('11031', {}).get('candidate_gate_passed') is False,
            'Original PREFIX failure was changed')
    require(waiver.get('schema') == 'sr-satdose-prefix-waiver-v1' and
            waiver.get('status') == 'USER_AUTHORIZED_MEASUREMENT_GATE_WAIVER' and waiver.get('experiment_id') == EXPERIMENT and
            waiver.get('scope') == 'PREFIX_CALIB_MEASUREMENT_GATE_ONLY' and waiver.get('evidence_root') == str(e.root) and
            waiver.get('design_freeze_sha256') == binding['design_freeze']['sha256'] and
            waiver.get('config_sha256') == config_ref['sha256'] and waiver.get('original_result') == binding['prefix_result'] and
            waiver.get('preserve_original_results') is True and waiver.get('rewrite_measurement_pass') is False and
            waiver.get('other_gates_unchanged') is True and waiver.get('technical_failure_or_unknown_waived') is False,
            'PREFIX waiver scope/identity differs')
    require(result.get('status') == 'PREFIX_CAPTURE_COMPLETE_WITH_AUTHORIZED_MEASUREMENT_WAIVER' and
            result.get('gate_decision') == 'PASS_USER_OVERRIDE' and result.get('responses') == 96 and
            result.get('resampled_responses') == 0 and result.get('waiver') == binding['prefix_waiver'] and
            result.get('original_result') == binding['prefix_result'] and set(result['prefix_analyses']) == set(STARTS) and
            result['prefix_analyses']['11031']['candidate_gate_passed'] is False,
            'PREFIX original failure/complete coverage differs')
    checks['prefix_calibration_pass'] = dict(passed=False,evidence=copy.deepcopy(binding['prefix_result']),
        execution_decision='PASS_USER_OVERRIDE',waiver=copy.deepcopy(binding['prefix_waiver']),
        complete_capture=copy.deepcopy(binding['prefix_continuation']))
    e.stable()
    return dict(schema='sr-satdose-gpu-verification-v1',mode='production',experiment_id=EXPERIMENT,
        status='TECHNICAL_EVIDENCE_COMPLETE_PREFIX_MEASUREMENT_WAIVED_NOT_ADMISSION',permission='NO_TRAINING',
        config_sha256=config_ref['sha256'],registration_sha256=registration_ref['sha256'],
        design_freeze_sha256=binding['design_freeze']['sha256'],technical_binding=copy.deepcopy(binding_ref),
        independent_audit=copy.deepcopy(audit_ref),constructor_sha256=source_sha256,
        dry_optimizer_steps=0,dry_parameter_updates=0,shadow_optimizer_steps=4,checks=checks,gradient=gradient,
        prefix_gate_decision='PASS_USER_OVERRIDE',prefix_waiver=copy.deepcopy(binding['prefix_waiver']),
        limitations=copy.deepcopy(audit['limitations']),
        observed_evidence={str(p.relative_to(e.root)) if p.is_relative_to(e.root) else str(p):h
                           for p,h in e.observed.items()})
