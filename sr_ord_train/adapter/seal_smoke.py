"""Independently verify the fixed 192-draw smoke; no model or scheduler calls.

First run prepares immutable evidence and a PENDING visual-review template.
Finalize re-verifies original evidence and requires a separately signed review.
Answer accuracy, invalid JSON and constant-reward groups never gate this seal.
"""
from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from adapter.backend import object_sha, text_sha
from adapter.registration import Registration, canonical, digest, strict_json
from adapter.training import trajectory_hash
from runtime_contract import PINS, SNAPSHOT, check_smoke


def need(condition, why):
    if not condition:
        raise ValueError(why)


def hash_string(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


class Evidence:
    def __init__(self):
        self.refs = {}

    def file(self, path, beneath=None):
        path = Path(path)
        need(path.is_absolute() and not path.is_symlink() and path.is_file(), 'missing/unsafe evidence: ' + str(path))
        need(path.resolve() == path, 'noncanonical evidence: ' + str(path))
        if beneath is not None:
            need(path.is_relative_to(beneath), 'evidence escapes worker stage: ' + str(path))
        self.refs[str(path)] = digest(path)
        return path

    def json(self, path, beneath=None):
        return strict_json(self.file(path, beneath).read_bytes())

    def rows(self, path, beneath=None):
        with self.file(path, beneath).open('rb') as handle:
            return [strict_json(line) for line in handle]


def unique(records, key='request_id'):
    result = {row[key]: row for row in records}
    need(len(result) == len(records), 'duplicate evidence identity: ' + key)
    return result


def independent_scores(package, raw, cases):
    """Fresh interpreter prevents shadowing by training imports or their caches."""
    payload = []
    for row in raw:
        task = cases[row['case_id']]['task']
        payload.append([row['request_id'], row['full_text'], task['world'], task['query'],
                        row['finished'], row['finish_reason']])
    script = """
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);sys.path.insert(0,str(p))
from runtime_contract import native_fixtures
receipt=native_fixtures(p/'vendor/native_src')
sys.path.insert(0,str(p/'vendor/native_src'))
from sr_f6.measurement import score_response
rows=[]
for rid,text,world,query,finished,reason in json.load(sys.stdin):
 rows.append({'request_id':rid,'score':score_response(text,world,query,finished=finished,finish_reason=reason)})
print(json.dumps({'native_fixtures':receipt,'scores':rows},sort_keys=True,allow_nan=False))
"""
    result = subprocess.run([sys.executable, '-B', '-c', script, str(package)],
                            input=json.dumps(payload), text=True, capture_output=True, timeout=60)
    need(result.returncode == 0, 'independent native scorer failed: ' + result.stderr)
    return strict_json(result.stdout)


def numerical_check(raw_rows, replay_rows):
    raw = unique(raw_rows)
    replay = unique(replay_rows)
    need(raw.keys() == replay.keys(), 'replay coverage differs')
    absolute, clipped, total = [], 0, 0
    for rid, row in raw.items():
        replayed = replay[rid]
        tokens = row['tokens']
        need(isinstance(tokens, list) and 1 <= len(tokens) <= 1536 and
             all(type(t) is int and t >= 0 for t in tokens), 'invalid original tokens: ' + rid)
        need(row['raw_tokens'] == tokens and row['completion_token_count'] == len(tokens), 'token alias/count differs')
        need(replayed['tokens'] == tokens and replayed['sampler_logprobs'] == row['sampler_logprobs'] and
             replayed['prompt_tensor_hash'] == row['prompt_tensor_hash'], 'replay original trajectory differs')
        sampler, current = row['sampler_logprobs'], replayed['train_logprobs']
        need(len(sampler) == len(current) == len(tokens), 'probability length differs')
        need(all(type(v) in (float, int) and math.isfinite(v) for v in sampler + current), 'nonfinite probability')
        delta = [c - s for c, s in zip(current, sampler)]
        local = [abs(v) for v in delta]
        checks = {'mean_absolute_difference': sum(local) / len(local),
                  'max_absolute_difference': max(local),
                  'tis_truncated_fraction': sum(v > math.log(2) for v in delta) / len(delta)}
        for name, value in checks.items():
            need(math.isclose(replayed[name], value, rel_tol=2e-5, abs_tol=2e-7),
                 'reported replay statistic differs: ' + name)
        absolute.extend(local)
        clipped += sum(v > math.log(2) for v in delta)
        total += len(delta)
    result = {'token_count': total, 'responses': len(raw),
              'sampler_train_mean_absolute_difference': sum(absolute) / total,
              'sampler_train_max_absolute_difference': max(absolute),
              'tis_truncated_fraction': clipped / total,
              'aggregation': 'all original completion tokens, historical token-weighted gate'}
    need(result['sampler_train_mean_absolute_difference'] <= .01 and
         result['sampler_train_max_absolute_difference'] <= 5 and
         result['tis_truncated_fraction'] <= .01, 'smoke sampler/replay numerical gate failed')
    return result


def collect(package, run):
    package, run = Path(package).resolve(strict=True), Path(run).resolve(strict=True)
    registry = Registration(package)
    expected_rows = list(itertools.islice(registry.requests(), 192))
    need(all(r['scope'] == 'DEV_SMOKE' for r in expected_rows), 'registration smoke ordering differs')
    expected = unique(expected_rows)
    evidence = Evidence()
    slots = evidence.json(package / 'frozen/receipt_plans/smoke_slots.json')
    index = evidence.json(package / 'frozen/receipt_plans/INDEX.json')
    need(digest(package / 'frozen/receipt_plans/smoke_slots.json') == index['files']['smoke_slots.json'],
         'smoke slot digest differs')
    cases = unique(evidence.rows(package / 'frozen/private_cases.jsonl'), 'case_id')
    raw_all, replay_all, bindings, processors, policies, saved_scores = [], [], [], [], {}, {}
    seen_requests = set()
    for parent in ('b0', 'b1'):
        stage = run / 'dev_smoke' / parent
        raw_path = stage / 'RAW.jsonl'
        raw = evidence.rows(raw_path, stage)
        replay = evidence.rows(stage / 'REPLAY.jsonl', stage)
        by_id = unique(raw)
        wanted = {k for k, r in expected.items() if r['parent'] == parent}
        need(by_id.keys() == wanted and len(by_id) == 96, 'raw smoke denominator differs: ' + parent)
        need(not seen_requests.intersection(by_id), 'cross-parent duplicate request')
        seen_requests.update(by_id)
        executions = sorted((stage / 'executions').iterdir())
        completed = []
        loaded_by_execution = {}
        for execution in executions:
            need(execution.is_dir() and not execution.is_symlink(), 'unsafe execution directory')
            # Early zero-output setup failures need no synthetic completion receipt.
            if not (execution / 'LOADED_IDENTITY.json').exists():
                need(not list(execution.glob('generation_attempts/*.BEGIN.json')), 'unbound generation attempt')
                continue
            loaded = evidence.json(execution / 'LOADED_IDENTITY.json', stage)
            binding = evidence.json(execution / 'WORKER_BINDING.json', stage)
            need(loaded['experiment_id'] == binding['experiment_id'] == 'SR-ORD-TRAIN-1' and
                 loaded['execution_mode'] == 'DEV_SMOKE' and loaded['parent_id'] == parent and
                 loaded['snapshot'] == binding['model']['snapshot'] == SNAPSHOT and
                 loaded['manifest_sha256'] == binding['manifest_sha256'] == registry.freeze_sha256,
                 'loaded experiment/parent/manifest differs')
            need(loaded['binding_sha256'] == object_sha(binding) and
                 loaded['source_sha256'] == binding['source_sha256'] == object_sha(binding['source_files']),
                 'loaded source or binding identity differs')
            need(loaded['native_runtime_sha256'] == PINS['sr_f6/runtime.py'], 'native runtime differs')
            need(loaded['dtype'] == loaded['input_float_dtype'] == loaded['probability_dtype'] == 'float32' and
                 loaded['forward_parameter_dtypes'] == ['float32'] and
                 set(loaded['forward_float_buffer_dtypes']) <= {'float32'} and
                 loaded['attention_backend'] == 'eager', 'actual FP32/eager identity differs')
            need(loaded['optimizer_created'] is False and loaded['step'] == 0,
                 'smoke optimizer/cursor differs')
            need(loaded['hardware']['cuda_visible_device_count'] == 1 and
                 loaded['hardware']['cuda_name'] and loaded['hardware']['cuda_uuid'], 'missing actual GPU identity')
            policy = loaded['parent_tensor_sha256']
            need(hash_string(policy) and loaded['reference_hash'] == policy, 'unbound parent tensor')
            need(parent not in policies or policies[parent] == policy, 'parent changed across executions')
            policies[parent] = policy
            loaded_by_execution[execution] = loaded
            bindings.append({'execution': str(execution), 'loaded_identity': loaded, 'worker_binding': binding})
            request_ref = binding['requests_manifest']
            path = evidence.file(Path(request_ref['path']))
            need(evidence.refs[str(path)] == request_ref['sha256'], 'worker request manifest changed')
            # Read the manifest once per distinct path; full equality is checked for every smoke request.
            if not hasattr(evidence, 'request_cache'):
                evidence.request_cache = {}
            if str(path) not in evidence.request_cache:
                selected = [r for r in evidence.rows(path) if r.get('scope') == 'DEV_SMOKE']
                evidence.request_cache[str(path)] = unique(selected)
            need(evidence.request_cache[str(path)] == expected, 'registered smoke requests differ from frozen plan')
            if (execution / 'WORKER_COMPLETE.json').exists():
                complete = evidence.json(execution / 'WORKER_COMPLETE.json', stage)
                need(complete['status'] == 'WORKER_COMPLETE_REVIEW_PENDING' and complete['scope'] == 'DEV_SMOKE' and
                     complete['parent'] == parent and complete['generated_responses'] == 96 and
                     complete['optimizer_steps'] == complete['parameter_updates'] == 0 and
                     complete['policy_before_sha256'] == complete['policy_after_sha256'] == policy and
                     complete['raw_sha256'] == evidence.refs[str(raw_path)], 'worker completion differs')
                completed.append(complete)
            processor_path = execution / 'processor/PROCESSOR.json'
            if processor_path.exists():
                processor = evidence.json(processor_path, stage)
                rid = processor['request_id']
                need(rid in wanted and processor['source_file'] == expected[rid]['image_file'] and
                     processor['source_sha256'] == expected[rid]['image_sha256'], 'processor source mismatch')
                processed = evidence.file(Path(processor['processed_file']), stage)
                need(evidence.refs[str(processed)] == processor['processed_file_sha256'], 'processed PNG differs')
                from PIL import Image
                with Image.open(processed) as im:
                    im.load()
                    import hashlib
                    need(hashlib.sha256(im.tobytes()).hexdigest() == processor['processed_rgb_sha256'],
                         'processor reconstructed pixels differ')
                need(processor['model_calls'] == processor['optimizer_updates'] == 0 and
                     processor['routing']['input_tensor_hash'] == by_id[rid]['prompt_tensor_hash'] and
                     processor['prompt_input_ids'] == by_id[rid]['native_prompt_inputs']['input_ids'][0],
                     'processor identity differs from actual smoke')
                processors.append({'parent': parent, 'receipt_path': str(processor_path), 'receipt': processor})
        need(completed and parent in policies, 'worker lacks final completion: ' + parent)
        need(any(p['parent'] == parent for p in processors), 'missing real processor evidence: ' + parent)
        native_attempt_ids = []
        for begin_path in sorted(stage.glob('executions/*/generation_attempts/*.BEGIN.json')):
            begin = evidence.json(begin_path, stage)
            rid = begin['request_id']
            need(rid in wanted and begin['seed'] == expected[rid]['seed'] and
                 begin['policy_sha256'] == begin['reference_sha256'] == policies[parent], 'native attempt identity differs')
            complete_path = begin_path.with_name(begin_path.name.replace('.BEGIN.json', '.COMPLETE.json'))
            need(complete_path.exists(), 'unresolved native generation attempt: ' + rid)
            need(evidence.json(complete_path, stage) == by_id[rid], 'native completion differs from raw')
            native_attempt_ids.append(rid)
        need(len(native_attempt_ids) == 96 and set(native_attempt_ids) == wanted, 'duplicate/missing native attempts')
        stage_attempts = list(stage.glob('*.BEGIN.json'))
        need(len(stage_attempts) == 96, 'stage attempt count differs')
        for begin_path in stage_attempts:
            begin = evidence.json(begin_path, stage)
            rid = begin['request']['request_id']
            need(rid in wanted and begin['request'] == expected[rid] and
                 begin_path.name == object_sha(rid) + '.BEGIN.json' and
                 begin['policy_sha256'] == policies[parent], 'stage attempt identity differs')
        for rid, row in by_id.items():
            request = expected[rid]
            need(all(row.get(k) == request[k] for k in request if k not in ('prompt', 'image_file')),
                 'raw request fields differ: ' + rid)
            need(row['row_id'] == rid and row['policy_hash'] == row['reference_hash'] == policies[parent] and
                 row['run_id'] == 'SR-ORD-TRAIN-1/DEV_SMOKE/' + parent and
                 row['protocol'] == 'structured_v4' and row['protocol_revision'] == 'v2', 'raw policy/protocol differs')
            need(row['generation_status'] == 'COMPLETE' and row['technical_validation_errors'] == [] and
                 row.get('native_exception') is None, 'technical failure in raw')
            need(row['model_input'] == {k: request[k] for k in ('prompt', 'image_file', 'image_sha256')},
                 'raw model input differs')
            need(row['sampler_logprob_source'] == 'actual_generate_raw_logits_selected_token' and
                 row['token_path_sha256'] == trajectory_hash(row), 'raw original trajectory identity differs')
            need(row['full_text'] == row['assistant_prefill'] + row['suffix_text'] and
                 row['suffix_text'] == row['generated_text'] and row['finish_reason'] in ('balanced', 'eos', 'length') and
                 row['finished'] is (row['finish_reason'] == 'balanced'), 'raw full text or stopping identity differs')
            route = row['input_routing']
            need(route['same_assistant_message'] is True and route['image_token_count'] > 0 and
                 route['input_float_dtype'] == route['probability_dtype'] == 'float32' and
                 all(v not in ('float16', 'bfloat16', 'float64') for v in route['forward_input_dtypes'].values()),
                 'raw processor FP32/image/prefix routing differs')
            need(all(row['sampling_parameters'].get(k) == v for k, v in
                     dict(max_new_tokens=1536, temperature=1., top_p=1., top_k=0, do_sample=True).items()),
                 'actual sampling recipe differs')
            capture_path = evidence.file(Path(row['native_capture_path']), stage)
            need(digest(capture_path) == row['native_capture_sha256'], 'native captured bytes changed')
            capture = evidence.json(capture_path, stage)
            execution = capture_path.parent.parent
            need(execution in loaded_by_execution and
                 row['source_sha256'] == loaded_by_execution[execution]['source_sha256'], 'capture source differs')
            need(all(capture.get(k) == row.get(k) for k in ('tokens', 'raw_tokens', 'sampler_logprobs',
                 'prompt_tensor_hash', 'model_input', 'seed', 'full_text', 'assistant_prefill', 'generated_text',
                 'finish_reason', 'sampling_parameters')),
                 'raw differs from original capture')
            need(capture['adapter_identity']['step'] == 0 and capture['adapter_identity']['run_id'] == row['run_id'],
                 'capture policy cursor differs')
            saved_scores[rid] = evidence.json(stage / 'scores' / (object_sha(rid) + '.json'), stage)
        raw_all.extend(raw)
        replay_all.extend(replay)
    numerics = numerical_check(raw_all, replay_all)
    scorer = independent_scores(package, raw_all, cases)
    scores = unique(scorer['scores'])
    need(set(scores) == set(expected) and all(scores[rid]['score'] == saved_scores[rid] for rid in expected),
         'independent native scores differ')
    groups = [dict(group, policy_before_sha256=policies[group['parent']],
                   policy_after_sha256=policies[group['parent']]) for group in slots]
    summary = {'status': 'SMOKE_TECHNICAL_PENDING_VISUAL_REVIEW', 'manifest_sha256': registry.freeze_sha256,
               'plan_sha256': index['files']['smoke_slots.json'], 'generated_responses': 192,
               'optimizer_steps': 0, 'parameter_updates': 0, 'numerics': numerics,
               'parents': policies, 'native_scores_independently_replayed': 192,
               'scientific_result': False,
               'quality_counts': {parent: dict(Counter(str(scores[r['request_id']]['score']['J'])
                                  for r in raw_all if r['parent'] == parent)) for parent in policies},
               'source_evidence_files': len(evidence.refs)}
    # Every byte used to prepare the review is bound again at finalize.
    for relative, expected_hash in registry.source_hashes.items():
        evidence.refs[str(package / relative)] = expected_hash
    summary['source_evidence_files'] = len(evidence.refs)
    return dict(summary=summary, raw=raw_all, replay=replay_all, scorer=scorer,
                native_binding={'bindings': bindings, 'source_evidence': evidence.refs},
                processors={'processors': processors}, groups=groups, slots=slots)


def write_json(path, value):
    with Path(path).open('xb') as handle:
        handle.write(canonical(value) + b'\n')
        handle.flush()
        os.fsync(handle.fileno())


def prepare(package, run, out=None):
    package, run = Path(package).resolve(strict=True), Path(run).resolve(strict=True)
    out = Path(out) if out else run / 'smoke_review'
    need(out.is_absolute() and not out.exists() and not out.is_symlink() and out.resolve().is_relative_to(run),
         'review needs a new directory inside run root')
    data = collect(package, run)
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.smoke-review-', dir=out.parent))
    try:
        for name in ('raw', 'replay'):
            with (temp / (name + '.jsonl')).open('xb') as handle:
                for row in data[name]:
                    handle.write(canonical(row) + b'\n')
        for name in ('scorer', 'native_binding', 'processors', 'groups'):
            write_json(temp / (name + '.json'), data[name])
        summary = data['summary']
        summary['review_files'] = {p.name: digest(p) for p in sorted(temp.iterdir())}
        write_json(temp / 'SUMMARY.json', summary)
        template = {'status': 'PENDING', 'reviewed_by': None, 'reviewed_at': None,
                    'summary_sha256': digest(temp / 'SUMMARY.json'), 'processor_reviews': [
                    {'parent': p['parent'], 'receipt_path': p['receipt_path'],
                     'source_sha256': p['receipt']['source_sha256'],
                     'processed_file_sha256': p['receipt']['processed_file_sha256'],
                     'labels_readable': None, 'values_readable': None, 'source_to_processed_consistent': None}
                    for p in data['processors']['processors']]}
        write_json(temp / 'VISUAL_REVIEW_TEMPLATE.json', template)
        os.rename(temp, out)
        return summary
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def finalize(package, run, visual_review, out=None):
    package, run = Path(package).resolve(strict=True), Path(run).resolve(strict=True)
    out = Path(out) if out else run / 'smoke_review'
    need(out.resolve(strict=True).is_relative_to(run) and not out.is_symlink(), 'review escapes run root')
    summary = strict_json((out / 'SUMMARY.json').read_bytes())
    need(summary['status'] == 'SMOKE_TECHNICAL_PENDING_VISUAL_REVIEW', 'review is not pending')
    for name, expected_hash in summary['review_files'].items():
        need(Path(name).name == name and digest(out / name) == expected_hash, 'prepared evidence changed')
    fresh = collect(package, run)
    for key, value in fresh['summary'].items():
        need(summary[key] == value, 'live smoke evidence changed: ' + key)
    for name in ('scorer', 'native_binding', 'processors', 'groups'):
        need(strict_json((out / (name + '.json')).read_bytes()) == fresh[name], 'live source changed: ' + name)
    visual_path = Path(visual_review).resolve(strict=True)
    visual = strict_json(visual_path.read_bytes())
    need(visual.get('status') == 'VISUAL_REVIEW_PASS' and
         isinstance(visual.get('reviewed_by'), str) and visual['reviewed_by'].strip() and
         isinstance(visual.get('reviewed_at'), str) and visual['reviewed_at'].strip() and
         visual.get('summary_sha256') == digest(out / 'SUMMARY.json'), 'explicit visual review signature required')
    assigned = [{'parent': p['parent'], 'receipt_path': p['receipt_path'],
                 'source_sha256': p['receipt']['source_sha256'],
                 'processed_file_sha256': p['receipt']['processed_file_sha256']}
                for p in fresh['processors']['processors']]
    reviews = visual.get('processor_reviews', [])
    need(len(reviews) == len(assigned), 'visual review coverage differs')
    for review, expected in zip(reviews, assigned, strict=True):
        need(all(review.get(k) == expected[k] for k in
                 ('parent', 'receipt_path', 'source_sha256', 'processed_file_sha256')) and
             all(review.get(k) is True for k in ('labels_readable', 'values_readable', 'source_to_processed_consistent')),
             'visual source/processor review incomplete')
    target = run / 'SMOKE_TECHNICAL_REVIEW.json'
    need(not target.exists() and not (out / 'SMOKE_RECEIPT.json').exists(), 'existing seal cannot be replaced')
    write_json(out / 'VISUAL_REVIEW.json', visual)
    processor_checks = {**fresh['processors'], 'visual_review': visual,
                        'visual_review_source': {'path': str(visual_path), 'sha256': digest(visual_path)}}
    write_json(out / 'PROCESSOR_CHECKS.json', processor_checks)
    names = {'raw': 'raw.jsonl', 'scorer_receipt': 'scorer.json', 'token_replay': 'replay.jsonl',
             'native_binding': 'native_binding.json', 'processor_checks': 'PROCESSOR_CHECKS.json'}
    receipt = dict(schema='order-runtime-v1', scope='DEV_SMOKE', manifest_sha256=summary['manifest_sha256'],
                   snapshot=SNAPSHOT, dtype='float32', reward='native_srf6_J_only', protocol='structured_v4',
                   aux_enabled=False, prefix_training=False, has_gpu_hour_stopping=False,
                   optimizer_steps=0, parameter_updates=0, generated_responses=192, groups=fresh['groups'],
                   evidence={k: {'path': name, 'sha256': digest(out / name)} for k, name in names.items()})
    checked = check_smoke(receipt, summary['manifest_sha256'], out, fresh['slots'])
    write_json(out / 'SMOKE_RECEIPT.json', receipt)
    sealed = {'status': 'SMOKE_TECHNICAL_PASS', 'manifest_sha256': summary['manifest_sha256'],
              'plan_sha256': summary['plan_sha256'], 'generated_responses': 192,
              'optimizer_steps': 0, 'parameter_updates': 0, 'numerics': summary['numerics'],
              'native_scores_independently_replayed': 192, 'runtime_contract': checked,
              'summary': {'path': str(out / 'SUMMARY.json'), 'sha256': digest(out / 'SUMMARY.json')},
              'receipt': {'path': str(out / 'SMOKE_RECEIPT.json'), 'sha256': digest(out / 'SMOKE_RECEIPT.json')},
              'visual_review': {'path': str(out / 'VISUAL_REVIEW.json'), 'sha256': digest(out / 'VISUAL_REVIEW.json')},
              'scientific_result': False}
    write_json(target, sealed)
    return sealed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--finalize', action='store_true')
    parser.add_argument('--visual-review', type=Path)
    args = parser.parse_args()
    run = args.run_root or args.package / 'run'
    if args.finalize:
        if args.visual_review is None:
            parser.error('--finalize requires --visual-review')
        result = finalize(args.package, run, args.visual_review, args.out)
    else:
        result = prepare(args.package, run, args.out)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
