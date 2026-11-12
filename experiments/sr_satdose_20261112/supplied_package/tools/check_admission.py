#!/usr/bin/env python3
"""Fail-closed prospective DRY admission, not a cluster launcher.

Requires REAL raw outputs, independently pinned native rescoring, a frozen
calibration manifest/cases, and hashed GPU verification evidence. No receipts
or model outputs are manufactured by this tool. A PASS does not guarantee efficacy.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import importlib
import json
import math
from pathlib import Path
import sys
from satdose_core import (STARTS, STRUCTURES, check_artifact, dry_gate, dry_metrics,
                          exact_hit, file_sha, geometry, read_json, read_jsonl,
                          require, source_class, write_json)

REQUIRED_TECH = (
    'start_and_reference_hash_verified','native_scorer_golden_pass',
    'disjoint_splits_checked','gold_never_in_model_inputs',
    'native_loss_replay_equal','production_gradient_hook_verified',
    'sampler_teacher_gate','raw_token_identity','same_runtime_backend',
    'long_sequence_fullbatch_pass','allzero_pg_zero_kl_retained',
    'shadow_adam_update_replay_pass','evaluation_loader_policy_verified',
    'dry_unchanged_policy','prefix_calibration_pass',
)


def verify_gpu(gpu: dict, cfg: dict, config_path: Path, registration: Path, root: Path) -> None:
    require(gpu['schema']=='sr-satdose-gpu-verification-v1' and gpu['mode']=='production',
            'GPU evidence is absent, simulated, or has the wrong schema')
    require(gpu['experiment_id']==cfg['experiment_id'], 'GPU evidence experiment mismatch')
    require(gpu['config_sha256']==file_sha(config_path) and gpu['registration_sha256']==file_sha(registration),
            'GPU verification not bound to exact configuration/registration')
    for key in REQUIRED_TECH:
        item=gpu['checks'][key]
        require(item['passed'] is True, 'GPU/technical gate not passed: '+key)
        check_artifact(root, item['evidence'])
    require(type(gpu['dry_optimizer_steps']) is int and type(gpu['dry_parameter_updates']) is int
            and gpu['dry_optimizer_steps']==0 and gpu['dry_parameter_updates']==0,
            'dry run changed parameters')
    require(type(gpu['shadow_optimizer_steps']) is int and 0 < gpu['shadow_optimizer_steps'] <= 4,
            'shadow replay budget mismatch')
    for start in STARTS:
        s=str(start);g=gpu['gradient'][s]
        require(g['policy_before_sha256']==g['policy_after_sha256']==cfg['start_bindings'][s]['policy_sha256'],
                'gradient probe changed starting policy')
        check_artifact(root,g['evidence'])
        # These signs are engineering admission checks, not efficacy tests or universal constants.
        names=('J_vs_source_pg_cosine','J_split_half_cosine','source_split_half_cosine',
               'heldout_effect_half_1','heldout_effect_half_2','effect_repeat_error_floor')
        require(all(type(g[k]) in (int,float) and math.isfinite(g[k]) for k in names), 'invalid gradient diagnostics')
        require(all(-1 <= g[k] <= 1 for k in ('J_vs_source_pg_cosine','J_split_half_cosine','source_split_half_cosine')), 'cosine outside [-1,1]')
        require(g['J_vs_source_pg_cosine']>0 and g['J_split_half_cosine']>0 and g['source_split_half_cosine']>0,
                f'HOLD_DIRECTION_UNCERTAIN for {s}: gradient direction not stable')
        floor=g['effect_repeat_error_floor']
        require(floor>=0 and g['heldout_effect_half_1']>floor and g['heldout_effect_half_2']>floor,
                f'HOLD_DIRECTION_UNCERTAIN for {s}: held-out first-order effects not consistently positive')


def run(config: Path, registration: Path, evidence_root: Path, native_src: Path) -> dict:
    cfg=read_json(config);reg=read_json(registration)
    require(reg['schema']=='sr-satdose-registration-v1' and reg['mode']=='production', 'wrong/fake registration')
    require(reg['experiment_id']==cfg['experiment_id'] and reg['config_sha256']==file_sha(config), 'registration/config mismatch')
    refs=reg['artifacts']
    paths={k:check_artifact(evidence_root,refs[k]) for k in ('design_freeze','calibration_groups','private_cases','dry_raw')}
    freeze=read_json(paths['design_freeze'])
    require(freeze['schema']=='sr-satdose-design-freeze-v1' and freeze['config_sha256']==file_sha(config), 'missing pre-generation design freeze')
    for k in ('calibration_groups','private_cases'):
        require(freeze['artifacts'][k]['sha256']==refs[k]['sha256'], 'post-generation design substitution: '+k)
    # GPU report is bound separately to avoid a circular hash dependency.
    gpu_path=check_artifact(evidence_root,reg['gpu_verification_file']) if 'gpu_verification_file' in reg else None
    require(gpu_path is None, 'registration must not embed the report that hashes the registration; pass --gpu-report separately')
    binding=read_json(Path(__file__).resolve().parents[1]/'config/source_bindings.json')
    for mod,hx in binding['native_scorer_pins'].items():
        require(file_sha(native_src/mod)==hx,'unverified native scorer source: '+mod)
    sys.path.insert(0,str(native_src.resolve()))
    try:
        for mod in binding['native_scorer_pins']:
            imp=importlib.import_module(mod[:-3].replace('/','.'))
            require(Path(imp.__file__).resolve()==(native_src/mod).resolve(),'shadowed native scorer module')
        scorer=importlib.import_module('sr_f6.measurement').score_response
        case_rows=list(read_jsonl(paths['private_cases']))
        cases={x['case_id']:x for x in case_rows}
        require(len(cases)==len(case_rows), 'duplicate private case_id')
        manifest=list(read_jsonl(paths['calibration_groups']))
        require(len(manifest)==160,'need 80 calibration groups per starting checkpoint')
        groups={}; requests={};counts=Counter();strata=Counter();root_roles={};roots_by_start=defaultdict(set)
        for g in manifest:
            s=g['start_seed'];kind=g['kind'];gid=g['group_id']
            require(s in STARTS and kind in ('CONFLICT','ALIGNED') and gid not in groups,'bad/duplicate dry group')
            require(g['structure'] in STRUCTURES and g['case_id'] in cases,'unregistered dry case')
            task=cases[g['case_id']]['task'];audit=cases[g['case_id']]['audit']
            require(task['root_id']==g['root_id'] and task['structure']==g['structure'],'dry task identity mismatch')
            require(task.get('authority','chart')=='chart','dry authority must be chart')
            geo=geometry(task,audit)
            require(geo['alt'] is not None and (geo['chart']!=geo['alt'])==(kind=='CONFLICT'),'false conflict label')
            require(g['root_id'] not in roots_by_start[s], 'dry roots are not independent within start')
            roots_by_start[s].add(g['root_id'])
            require(root_roles.setdefault(g['root_id'],kind)==kind, 'same dry root used for both roles')
            require(len(g['request_ids'])==8 and len(g['draw_seeds'])==8 and len(set(g['draw_seeds']))==8,'incomplete dry G8')
            groups[gid]=(g,geo);counts[s,kind]+=1
            strata[s,kind,g['structure'],task['query']['direction'],task['query']['aggregate']]+=1
            for rid,seed in zip(g['request_ids'],g['draw_seeds']):
                require(rid not in requests, 'duplicate calibration request')
                requests[rid]=(g,geo,seed)
        for s in STARTS:
            require(counts[s,'CONFLICT']==64 and counts[s,'ALIGNED']==16,'calibration allocation differs')
            for st in STRUCTURES:
                for d in ('largest','smallest'):
                    for a in ('sum','mean'):
                        require(strata[s,'CONFLICT',st,d,a]==4 and strata[s,'ALIGNED',st,d,a]==1,'unbalanced dry strata')
        actual={};joined=defaultdict(list);trunc=Counter()
        for row in read_jsonl(paths['dry_raw']):
            rid=row['request_id'];require(rid in requests and rid not in actual,'extra/duplicate dry result')
            g,geo,seed=requests[rid];s=g['start_seed'];task=cases[g['case_id']]['task']
            require(row['group_id']==g['group_id'] and row['case_id']==g['case_id'] and row['seed']==seed,'dry identity mismatch')
            require(row['policy_sha256']==cfg['start_bindings'][str(s)]['policy_sha256'], 'wrong dry checkpoint')
            require(row['reference_sha256']==cfg['start_bindings'][str(s)]['reference_sha256'], 'wrong dry reference')
            require(row['generation_status']=='COMPLETE' and row['technical_validation_errors']==[], 'failed/unknown generation cannot be scored as zero')
            require(row['finish_reason'] in ('balanced','eos','length'), 'unknown termination')
            require(row['full_text']==row['assistant_prefill']+row['suffix_text'], 'dry full_text mismatch')
            require(row['assistant_prefill']=='{', 'training dry must not use forced evidence prefixes')
            # Native TRAINING finish semantics, not the broader field-local evaluation convention.
            ns=scorer(row['full_text'],task['world'],task['query'],
                      finished=row['finish_reason']=='balanced',finish_reason=row['finish_reason'])
            require(ns['J']==row['native_score']['J'],'dry native reward replay mismatch')
            known=int(ns['known_mask']['S']);sel=ns['witness'].get('selected') if known else None
            c=exact_hit(sel,geo['authorised'])
            require((ns['S'] if known else 0)==c,'native selection disagrees with independent oracle')
            joined[g['group_id']].append((rid,ns['J'],c,known,source_class(sel,geo)))
            actual[rid]=True;trunc[s]+=int(row['finish_reason']=='length')
        require(set(actual)==set(requests) and len(actual)==1280, 'missing dry responses')
        result={};fail=[]
        for s in STARTS:
            ys=[];class_counts=Counter()
            for gid,(g,geo) in groups.items():
                if g['start_seed']==s and g['kind']=='CONFLICT':
                    rows=sorted(joined[gid]);require(len(rows)==8,'incomplete observed G8')
                    ys.append(dict(J=[x[1] for x in rows],c=[x[2] for x in rows],known=[x[3] for x in rows]))
                    class_counts.update(x[4] for x in rows)
            m=dry_metrics(ys,B=cfg['dry']['bootstrap_B']);m['class_counts']=dict(class_counts)
            m['truncation_rate_all_dry']=trunc[s]/640
            bad=dry_gate(m,cfg['dry'])
            if m['truncation_rate_all_dry']>cfg['dry']['max_truncation_rate']:bad.append('TRUNCATION_RATE')
            result[str(s)]={'metrics':m,'failures':bad}
            fail.extend(f'{s}:{x}' for x in bad)
        return dict(schema='sr-satdose-admission-v1',experiment_id=cfg['experiment_id'],
                    config_sha256=file_sha(config),registration_sha256=file_sha(registration),
                    status='HOLD_DRY_SIGNAL' if fail else 'DRY_METRICS_PASS_GPU_CHECK_STILL_REQUIRED',
                    starts=result,failures=fail,permission='NO_TRAINING_YET')
    finally:
        sys.path.remove(str(native_src.resolve()))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('config','registration','evidence-root','native-src','gpu-report','out'):
        ap.add_argument('--'+name,required=True,type=Path)
    a=ap.parse_args()
    require(not a.out.exists(),'refuse to overwrite an admission receipt')
    require(a.out.resolve() not in {x.resolve() for x in (a.config,a.registration,a.gpu_report)},'output overwrites input')
    try:
        r=run(a.config,a.registration,a.evidence_root,a.native_src)
        if not r['failures']:
            verify_gpu(read_json(a.gpu_report),read_json(a.config),a.config,a.registration,a.evidence_root)
            r.update(status='ADMITTED_FIRST_TWO_STEPS_ONLY',permission='FOUR_PATHS_MAX_STEP_2',gpu_report_sha256=file_sha(a.gpu_report))
        write_json(a.out,r)
        print(json.dumps(r,indent=2,ensure_ascii=False))
        return 0 if r['status']=='ADMITTED_FIRST_TWO_STEPS_ONLY' else 2
    except (ValueError,KeyError,FileNotFoundError,TypeError) as ex:
        write_json(a.out,{'status':'HOLD_TECHNICAL_OR_INCOMPLETE_EVIDENCE','permission':'NO_TRAINING','reason':str(ex)})
        print('HOLD:',ex,file=sys.stderr);return 2

if __name__=='__main__':sys.exit(main())
