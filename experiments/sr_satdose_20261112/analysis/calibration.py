"""Read-only CPU analysis of completed, frozen SATDOSE calibration captures.

This module cannot launch, retry or admit training. It refuses to open raw files
before a matching CALIBRATION_COMPLETE receipt exists. Never use partial results
for configuration selection. The optional registration command only binds real
completed DRY files; a separate genuine GPU report is still mandatory.
"""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import importlib
import json
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parents[1] / 'supplied_package'
TOOLS = PACKAGE / 'tools'
sys.path.insert(0, str(TOOLS))
import satdose_core as core
sys.path.pop(0)
core.require(Path(core.__file__).resolve() == (TOOLS/'satdose_core.py').resolve(), 'Shadowed SATDOSE helper')

VIEWS = ('p_stable', 'p_revtie', 'p_placebo')


def implementation_identity():
    return dict(analysis_sha256=core.file_sha(__file__), helper_sha256=core.file_sha(core.__file__))


def _index(rows, key):
    result = {}
    for row in rows:
        value = row[key]
        core.require(isinstance(value, str) and value and value not in result, 'Duplicate/invalid '+key)
        result[value] = row
    return result


def _file_ref(root, path):
    path = Path(path).resolve(strict=True)
    core.require(path.is_relative_to(root), 'Evidence path outside experiment root')
    return dict(path=str(path.relative_to(root)), sha256=core.file_sha(path))


def _freeze(config, freeze, root):
    root = Path(root).resolve(strict=True)
    config, freeze = Path(config).resolve(strict=True), Path(freeze).resolve(strict=True)
    core.require(freeze.is_relative_to(root), 'Freeze outside experiment root')
    cfg, frozen = core.read_json(config), core.read_json(freeze)
    core.require(frozen['schema'] == 'sr-satdose-design-freeze-v1' and
                 frozen['experiment_id'] == cfg['experiment_id'] and
                 frozen['config_sha256'] == core.file_sha(config), 'Freeze/config identity mismatch')
    return cfg, frozen, root


def _artifact(root, frozen, key):
    return core.check_artifact(root, frozen['artifacts'][key])


def load_scorer(native_src):
    native_src = Path(native_src).resolve(strict=True)
    binding = core.read_json(PACKAGE/'config/source_bindings.json')
    pins = binding['native_scorer_pins']
    for relative, expected in pins.items():
        core.require(core.file_sha(native_src/relative) == expected, 'Native scorer SHA mismatch: '+relative)
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(native_src))
        for relative in pins:
            module = importlib.import_module(relative[:-3].replace('/', '.'))
            core.require(Path(module.__file__).resolve() == (native_src/relative).resolve(), 'Shadowed native scorer')
        scorer = importlib.import_module('sr_f6.measurement').score_response
    finally:
        sys.path[:] = old_path
    return scorer, pins


def _completed(config, freeze, root, receipt_path, start, scope):
    """Receipt and immutable input checks precede the first raw open/hash."""
    core.require(type(start) is int and start in core.STARTS, 'Unknown calibration start')
    cfg, frozen, root = _freeze(config, freeze, root)
    receipt_path = Path(receipt_path).resolve(strict=True)
    core.require(receipt_path.is_relative_to(root), 'Completion receipt outside experiment root')
    receipt = core.read_json(receipt_path)
    count = 640 if scope == 'DRY' else 48
    freeze_sha = core.file_sha(freeze)
    core.require(receipt['status'] == 'CALIBRATION_CAPTURE_COMPLETE_NOT_GPU_ADMISSION' and
                 receipt['scope'] == scope and receipt['start_seed'] == start and
                 receipt['responses'] == count, 'Incomplete/wrong calibration receipt')
    core.require(receipt['design_freeze_sha256'] == freeze_sha, 'Completion freeze mismatch')
    source_lock = _artifact(root, frozen, 'source_lock')
    core.require(receipt['source_lock_sha256'] == frozen['artifacts']['source_lock']['sha256'] ==
                 core.file_sha(source_lock), 'Completion source lock mismatch')
    core.require(not (receipt_path.parent/'HOLD.json').exists() and
                 not (receipt_path.parent/'ERROR.json').exists() and
                 not any(receipt_path.parent.glob('*.ERROR.json')), 'Calibration contains HOLD/ERROR evidence')
    core.require(type(receipt['optimizer_steps']) is int and receipt['optimizer_steps'] == 0 and
                 type(receipt['parameter_updates']) is int and receipt['parameter_updates'] == 0 and
                 receipt['optimizer_created'] is False and receipt['scheduler_created'] is False,
                 'Calibration mutated parameters or constructed optimizer')
    bound = cfg['start_bindings'][str(start)]
    for endpoint in ('policy_before', 'policy_after'):
        core.require(receipt[endpoint]['policy_sha256'] == bound['policy_sha256'] and
                     receipt[endpoint]['reference_sha256'] == bound['reference_sha256'],
                     'Completion policy/reference mismatch')
    paths = {key:_artifact(root, frozen, key) for key in
             ('private_cases', 'inputs', 'images', 'dry_manifest' if scope == 'DRY' else 'prefix_calib_manifest')}
    cases = _index(core.read_jsonl(paths['private_cases']), 'case_id')
    inputs = _index(core.read_jsonl(paths['inputs']), 'input_id')
    images = _index(core.read_jsonl(paths['images']), 'image_file')
    manifests = list(core.read_jsonl(paths['dry_manifest' if scope == 'DRY' else 'prefix_calib_manifest']))
    core.require(len(manifests) == 2*count, 'Frozen calibration population differs')
    all_requests = _index(manifests, 'request_id')
    core.require({m['start_seed'] for m in manifests} == set(core.STARTS), 'Manifest starts differ')
    selected = {rid:m for rid,m in all_requests.items() if m['start_seed'] == start}
    core.require(len(selected) == count, 'Single-start manifest allocation differs')
    complete_refs=receipt['complete_files']
    core.require(isinstance(complete_refs,list) and len(complete_refs)==count, 'Incomplete COMPLETE file inventory')
    complete={};complete_paths=set()
    for ref in complete_refs:
        relative=Path(ref['path'])
        core.require(not relative.is_absolute() and len(relative.parts)==1 and relative.name.endswith('.COMPLETE.json'),
                     'COMPLETE file must be directly inside receipt directory')
        core.require(ref['path'] not in complete_paths, 'Duplicate COMPLETE file path')
        complete_paths.add(ref['path'])
        path=core.check_artifact(receipt_path.parent,ref)
        core.require(path.parent==receipt_path.parent, 'COMPLETE resolves outside receipt directory')
        item=core.read_json(path);rid=item['request_id']
        core.require(rid in selected and rid not in complete, 'Extra/duplicate COMPLETE request')
        expected_name=hashlib.sha256(rid.encode()).hexdigest()+'.COMPLETE.json'
        core.require(relative.name==expected_name, 'COMPLETE filename/request identity differs')
        core.require(item['design_freeze_sha256']==freeze_sha, 'COMPLETE freeze mismatch')
        complete[rid]=item
    core.require(set(complete)==set(selected), 'Missing COMPLETE requests')
    core.require({path.name for path in receipt_path.parent.glob('*.COMPLETE.json')}==complete_paths,
                 'Unlisted/missing COMPLETE file on disk')
    # Opening raw is permitted only after a completed, immutable start is verified.
    raw_path = core.check_artifact(receipt_path.parent, receipt['raw'])
    raw_rows = list(core.read_jsonl(raw_path))
    raw = _index(raw_rows, 'request_id')
    core.require(set(raw) == set(selected) and len(raw) == count, 'Missing/extra calibration responses')
    for rid, manifest in selected.items():
        row = raw[rid]
        original=complete[rid]
        core.require(all(key in row for key in original) and
                     core.object_sha({key:row[key] for key in original})==core.object_sha(original),
                     'Raw changed original COMPLETE fields')
        core.require(manifest.get('split') == scope and manifest['arm'] == 'START' and manifest['step'] == 0,
                     'Wrong frozen calibration endpoint')
        for key in ('case_id','input_id','root_id','seed','start_seed','arm','step','draw'):
            core.require(type(row.get(key)) is type(manifest[key]) and row[key] == manifest[key],
                         'Raw/manifest identity mismatch: '+key)
        core.require(row['scope'] == scope and row['design_freeze_sha256'] == freeze_sha, 'Raw freeze/scope mismatch')
        core.require(row['policy_sha256'] == bound['policy_sha256'] and
                     row['reference_sha256'] == bound['reference_sha256'], 'Raw checkpoint/reference mismatch')
        core.require(row['generation_status'] == 'COMPLETE' and row['technical_validation_errors'] == [],
                     'Technical failure cannot be scored as a behavioral zero')
        core.require(row['finish_reason'] in ('balanced','eos','length'), 'Unknown finish reason')
        case, inp = cases[manifest['case_id']], inputs[manifest['input_id']]
        core.require(case['task']['root_id'] == manifest['root_id'], 'Private root differs')
        core.require(case['input_id'] == manifest['input_id'], 'Private/public input differs')
        expected_input = dict(prompt=inp['prompt'],image_file=inp['image_file'],
                              image_sha256=images[inp['image_file']]['png_sha256'])
        core.require(row['model_input'] == expected_input, 'Actual model input differs from frozen public input')
        core.require(row['assistant_prefill'] == inp['assistant_prefill'] and
                     row['full_text'] == row['assistant_prefill']+row['suffix_text'], 'Prefill/raw text differs')
        core.require(row['dry_optimizer_steps'] == row['dry_parameter_updates'] == 0, 'Raw update count differs')
    return cfg, frozen, root, selected, raw, cases, inputs, raw_path, receipt_path


def _strict_selection(text, finish, gold):
    """Same field-local selected semantics as legacy score_eval.fields.

    Unrelated schema fields do not invalidate a readable selected field. Duplicate
    selected labels are known but incorrect; duplicate JSON keys/nonfinite values
    or trailing text make the whole JSON unreadable. Length never becomes known.
    """
    obj = None
    try:
        core.require(isinstance(text,str) and len(text) <= 100000, 'TEXT_BOUND')
        obj = core.loads(text, decimal=True)
        core.require(isinstance(obj,dict), 'TOP_LEVEL_NOT_OBJECT')
        def finite(value):
            if isinstance(value,Decimal):return value.is_finite()
            if isinstance(value,dict):return all(finite(v) for v in value.values())
            if isinstance(value,list):return all(finite(v) for v in value)
            return True
        core.require(finite(obj), 'NONFINITE_JSON')
    except (ValueError, TypeError, ArithmeticError, RecursionError):
        obj = None
    selected = obj.get('selected') if obj is not None else None
    known = int(finish in ('balanced','eos') and isinstance(selected,list) and all(type(v) is str for v in selected))
    return dict(json_valid=int(obj is not None), S_known=known,
                S_hit=core.exact_hit(selected,gold) if known else 0,
                selected=selected if known else None)


def prefix_selection(text, finish, gold):
    strict = _strict_selection(text,finish,gold)
    repaired = _strict_selection(text+'}',finish,gold) if finish == 'eos' and not strict['json_valid'] else None
    eligible = repaired is not None and repaired['json_valid'] == 1
    return dict(strict=strict, sensitivity=repaired if eligible else dict(strict),
                brace_eligible=int(eligible), appended_char='}' if eligible else None)


def analyze_dry(config, freeze, root, receipt, start, native_src):
    cfg,frozen,root,requests,raw,cases,inputs,raw_path,receipt_path = _completed(
        config,freeze,root,receipt,start,'DRY')
    scorer,pins = load_scorer(native_src)
    manifest = list(core.read_jsonl(_artifact(root,frozen,'calibration_groups')))
    core.require(len(manifest) == 160, 'Expected 160 frozen calibration groups')
    all_groups = _index(manifest,'group_id')
    groups = [g for g in manifest if g['start_seed'] == start]
    core.require(len(groups) == 80, 'Expected 80 complete groups per start')
    count, strata, root_ids, used_requests = Counter(), Counter(), set(), set()
    scored, conflict_groups, group_rows = [], [], []
    for group in groups:
        task = cases[group['case_id']]['task'];audit=cases[group['case_id']]['audit']
        kind = group['kind'];geo=core.geometry(task,audit)
        core.require(kind in ('CONFLICT','ALIGNED') and len(task['world']['categories']) == 8,
                     'Wrong DRY kind/category count')
        core.require(task.get('authority','chart') == 'chart' and group['root_id'] == task['root_id'] and
                     group['structure'] == task['structure'], 'DRY task identity differs')
        core.require(geo['alt'] is not None and (geo['chart'] != geo['alt']) == (kind == 'CONFLICT'),
                     'False conflict assignment')
        core.require(group['root_id'] not in root_ids, 'Duplicate root within start')
        root_ids.add(group['root_id']);count[kind]+=1
        strata[kind,group['structure'],task['query']['direction'],task['query']['aggregate']]+=1
        core.require(len(group['request_ids']) == len(group['draw_seeds']) == 8 and
                     len(set(group['draw_seeds'])) == 8, 'Incomplete G8 allocation')
        group_scores=[]
        for rid,seed in zip(group['request_ids'],group['draw_seeds'],strict=True):
            core.require(rid in requests and rid not in used_requests, 'Extra/duplicate group request')
            used_requests.add(rid);row=raw[rid];req=requests[rid]
            core.require(row['group_id'] == req['group_id'] == group['group_id'] and row['seed'] == seed and
                         row['case_id'] == group['case_id'] and row['assistant_prefill'] == '{', 'DRY group identity differs')
            native = scorer(row['full_text'],task['world'],task['query'],
                            finished=row['finish_reason']=='balanced',finish_reason=row['finish_reason'])
            core.require(native == row['native_score'], 'Native score replay mismatch')
            known=int(native['known_mask']['S']);sel=native['witness'].get('selected') if known else None
            correct=core.exact_hit(sel,geo['authorised'])
            core.require(correct == (native['S'] if known else 0), 'Native/oracle selected mismatch')
            field=_strict_selection(row['full_text'],row['finish_reason'],geo['authorised'])
            item=dict(request_id=rid,group_id=group['group_id'],root_id=group['root_id'],case_id=group['case_id'],
                      kind=kind,finish_reason=row['finish_reason'],J=native['J'],c=correct,known=known,
                      source_class=core.source_class(sel,geo),field_local_c=field['S_hit'],
                      field_local_known=field['S_known'],native_score=native)
            if native['J'] == 1 and not correct:
                item.update(full_text=row['full_text'],authorised=sorted(geo['authorised']),selected=sel)
            group_scores.append(item);scored.append(item)
        group_values=dict(J=[r['J'] for r in group_scores],c=[r['c'] for r in group_scores],known=[r['known'] for r in group_scores])
        if kind == 'CONFLICT':conflict_groups.append(group_values)
        group_rows.append(dict(group_id=group['group_id'],kind=kind,**core.group_metrics(**group_values)))
    core.require(count == {'CONFLICT':64,'ALIGNED':16} and used_requests == set(requests), 'DRY denominator/allocation differs')
    expected={(kind,st,dr,agg):(4 if kind=='CONFLICT' else 1) for kind in ('CONFLICT','ALIGNED')
              for st in core.STRUCTURES for dr in ('largest','smallest') for agg in ('sum','mean')}
    core.require(dict(strata) == expected, 'DRY strata allocation differs')
    metrics=core.dry_metrics(conflict_groups,B=cfg['dry']['bootstrap_B'])
    lengths=sum(r['finish_reason']=='length' for r in scored)
    metrics.update(length_count_all_dry=lengths,length_denominator_all_dry=640,truncation_rate_all_dry=lengths/640,
                   class_counts=dict(Counter(r['source_class'] for r in scored if r['kind']=='CONFLICT')))
    failures=core.dry_gate(metrics,cfg['dry'])
    if lengths/640 > cfg['dry']['max_truncation_rate']:failures.append('TRUNCATION_RATE')
    thresholds=[('signal_groups','>=',cfg['dry']['min_signal_groups']),('dose_mean','>=',cfg['dry']['min_dose_mean']),
                ('dose_lower95','>=',cfg['dry']['min_dose_lower95']),('direction_fraction','>=',cfg['dry']['min_direction_fraction']),
                ('unknown_rate','<=',cfg['dry']['max_unknown_rate']),('J1_source0_rate','<=',cfg['dry']['max_J1_source0_rate']),
                ('truncation_rate_all_dry','<=',cfg['dry']['max_truncation_rate'])]
    checks={name:dict(value=metrics[name],operator=op,threshold=limit,
                     passed=metrics[name] is not None and (metrics[name]>=limit if op=='>=' else metrics[name]<=limit))
            for name,op,limit in thresholds}
    checks['conflict_coverage']=dict(groups=64,responses=512,passed=True)
    return dict(schema='sr-satdose-dry-analysis-v1',start_seed=start,
                status='HOLD_DRY_SIGNAL' if failures else 'DRY_METRICS_PASS_GPU_CHECK_STILL_REQUIRED',
                permission='NO_TRAINING',metrics=metrics,checks=checks,failures=failures,
                J1_c0_details=[r for r in scored if r['J']==1 and r['c']==0],groups=group_rows,rescored_rows=scored,
                config_sha256=core.file_sha(config),design_freeze_sha256=core.file_sha(freeze),native_scorer_pins=pins,
                implementation=implementation_identity(),
                raw=_file_ref(root,raw_path),completion_receipt=_file_ref(root,receipt_path))


def analyze_prefix(config, freeze, root, receipt, start):
    cfg,frozen,root,requests,raw,cases,inputs,raw_path,receipt_path = _completed(
        config,freeze,root,receipt,start,'PREFIX_CALIB')
    rows=[];root_views={};structures={}
    for rid,req in requests.items():
        row=raw[rid];case=cases[req['case_id']];task=case['task'];view=req['view'];rt=req['root_id']
        core.require(req['panel']=='prefix10' and row['panel']=='prefix10' and row['view']==view and view in VIEWS and req['draw']==0,
                     'Wrong prefix panel/view/draw')
        core.require(len(task['world']['categories'])==10 and case['audit']['native_J_applicable'] is False,
                     'Native J is not applicable to prefix10')
        core.require(row.get('native_score') is None and row.get('native_J') is None and row.get('native_J_applicable') is False,
                     'Prefix cannot contain native J scores')
        core.require(len(row['assistant_prefill'])>1, 'Forced prefix absent')
        core.require(view not in root_views.setdefault(rt,set()), 'Duplicate root/view')
        root_views[rt].add(view)
        core.require(structures.setdefault(rt,task['structure'])==task['structure'], 'Root changes structure')
        value=prefix_selection(row['full_text'],row['finish_reason'],task['gold']['selected'])
        rows.append(dict(request_id=rid,root_id=rt,view=view,finish_reason=row['finish_reason'],**value))
    core.require(len(root_views)==16 and all(v==set(VIEWS) for v in root_views.values()), 'Incomplete 16x3 prefix panel')
    core.require(Counter(structures.values())==dict.fromkeys(core.STRUCTURES,4), 'Prefix structure imbalance')
    summaries={}
    for variant in ('strict','sensitivity'):
        by_view={}
        for view in VIEWS:
            cell=[r for r in rows if r['view']==view];known=sum(r[variant]['S_known'] for r in cell)
            hit=sum(r[variant]['S_hit'] for r in cell);unknown=16-known
            by_view[view]=dict(denominator=16,known=known,correct=hit,unknown=unknown,
                               correct_rate=hit/16,unknown_rate=unknown/16,known_gate=known>=14,
                               bounds=[hit/16,(hit+unknown)/16],
                               length_count=sum(r['finish_reason']=='length' for r in cell),
                               brace_eligible=sum(r['brace_eligible'] for r in cell))
        stable,revtie=by_view['p_stable'],by_view['p_revtie']
        difference=stable['correct']-revtie['correct']
        checks=dict(all_views_known_at_least_14=all(v['known_gate'] for v in by_view.values()),
                    stable_correct_at_least_12=stable['correct']>=12,
                    stable_minus_revtie_at_least_4=difference>=4)
        summaries[variant]=dict(views=by_view,stable_minus_revtie_count=difference,
                               stable_minus_revtie_rate=difference/16,
                               contrast_missing_bounds=[stable['bounds'][0]-revtie['bounds'][1],stable['bounds'][1]-revtie['bounds'][0]],
                               checks=checks,passed=all(checks.values()))
    # Literal OR applies to two COMPLETE readouts, never cherry-picked per view
    # or per metric. The phrase needs a recorded protocol clarification before
    # this diagnostic can be turned into a prefix admission claim.
    candidate_passed=any(summaries[variant]['passed'] for variant in ('strict','sensitivity'))
    return dict(schema='sr-satdose-prefix-calibration-analysis-v1',start_seed=start,
                status='PENDING_PROTOCOL_CLARIFICATION',
                permission='NO_TRAINING',candidate_decision_rule='strict_complete_gate_OR_sensitivity_complete_gate',
                candidate_gate_passed=candidate_passed,
                protocol_question='Does the prefix gate accept either complete readout, or require the terminal-brace sensitivity readout?',
                readouts=summaries,rows=rows,config_sha256=core.file_sha(config),design_freeze_sha256=core.file_sha(freeze),
                implementation=implementation_identity(),
                raw=_file_ref(root,raw_path),completion_receipt=_file_ref(root,receipt_path))


def write_once(path, value):
    path=Path(path)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,allow_nan=False,indent=2);stream.write('\n')


def build_registration(config, freeze, root, receipts, native_src, output):
    """Bind both completed DRY populations without inventing GPU evidence."""
    cfg,frozen,root=_freeze(config,freeze,root)
    core.require(set(receipts)==set(core.STARTS), 'Both registered starts required')
    analyses={start:analyze_dry(config,freeze,root,receipts[start],start,native_src) for start in core.STARTS}
    destination=Path(output).absolute()
    core.require(destination.parent.resolve(strict=True).is_relative_to(root) and not destination.exists(),
                 'Registration output must be a new directory inside evidence root')
    destination.mkdir(exist_ok=False)
    raw_path=destination/'DRY_RAW.jsonl'
    with raw_path.open('x',encoding='utf-8') as stream:
        for start in core.STARTS:
            source=core.check_artifact(root,analyses[start]['raw'])
            for row in core.read_jsonl(source):stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    artifacts={key:dict(frozen['artifacts'][key]) for key in ('calibration_groups','private_cases')}
    artifacts.update(design_freeze=_file_ref(root,freeze),dry_raw=_file_ref(root,raw_path))
    registration=dict(schema='sr-satdose-registration-v1',mode='production',experiment_id=cfg['experiment_id'],
                      config_sha256=core.file_sha(config),artifacts=artifacts)
    for start,report in analyses.items():write_once(destination/f'DRY_METRICS_{start}.json',report)
    write_once(destination/'ADMISSION_INPUTS.json',registration)
    return dict(status='ADMISSION_INPUTS_BOUND_GPU_REPORT_STILL_REQUIRED',permission='NO_TRAINING',
                registration=_file_ref(root,destination/'ADMISSION_INPUTS.json'),
                dry_status={str(start):report['status'] for start,report in analyses.items()})


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('dry','prefix','registration'))
    for name in ('config','freeze','evidence-root','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--native-src',type=Path)
    parser.add_argument('--receipt',type=Path)
    parser.add_argument('--start',type=int,choices=core.STARTS)
    parser.add_argument('--receipt-11031',type=Path);parser.add_argument('--receipt-11032',type=Path)
    args=parser.parse_args(argv)
    if args.command=='registration':
        core.require(args.native_src is not None and args.receipt_11031 is not None and args.receipt_11032 is not None,
                     'Registration needs native source and both completion receipts')
        result=build_registration(args.config,args.freeze,args.evidence_root,
                                  {11031:args.receipt_11031,11032:args.receipt_11032},args.native_src,args.output)
    else:
        core.require(args.receipt is not None and args.start is not None, 'Receipt and start required')
        if args.command=='dry':
            core.require(args.native_src is not None, 'Pinned native source required')
            result=analyze_dry(args.config,args.freeze,args.evidence_root,args.receipt,args.start,args.native_src)
        else:result=analyze_prefix(args.config,args.freeze,args.evidence_root,args.receipt,args.start)
        write_once(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k in ('status','permission','start_seed','registration')},indent=2))
    return 2 if result['status'].startswith(('HOLD','PENDING')) else 0


if __name__=='__main__':
    raise SystemExit(main())
