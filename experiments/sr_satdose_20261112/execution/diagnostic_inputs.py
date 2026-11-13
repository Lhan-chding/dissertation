"""Prepare fixed GPU-diagnostic inputs from two completed, passing DRY starts.

CPU only: this adapter never loads a model, invokes a GPU, samples, or permits
training. It reuses full calibration receipt validation and pinned native
rescoring, then projects the pre-model probe/witness root mapping onto each
start's original groups and request order. No original raw dictionary is edited.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from analysis import calibration
from runtime import diagnostics
sys.path.pop(0)

core = calibration.core


def _read_bound_rows(root, reference):
    path=core.check_artifact(root,reference)
    data=path.read_bytes()
    core.require(hashlib.sha256(data).hexdigest()==reference['sha256'], 'Bound rows changed during read')
    rows=[core.loads(line) for line in data.decode('utf-8').splitlines() if line.strip()]
    core.require(all(isinstance(row,dict) for row in rows), 'Bound JSONL contains non-object rows')
    return rows


def _validate_roots(mapping):
    for name in ('probe','witness'):
        roots=mapping[name+'_root_ids'];halves=mapping[name+'_halves']
        core.require(isinstance(roots,list) and len(roots)==16 and
                     all(isinstance(root,str) and root for root in roots) and len(set(roots))==16,
                     'Expected sixteen unique frozen '+name+' roots')
        core.require(isinstance(halves,list) and len(halves)==2 and
                     all(isinstance(half,list) and len(half)==8 for half in halves),
                     'Expected two frozen eight-root '+name+' halves')
        flat=[root for half in halves for root in half]
        core.require(all(isinstance(root,str) for root in flat) and len(set(flat))==16 and set(flat)==set(roots),
                     'Frozen '+name+' halves are not a disjoint complete partition')
    core.require(not set(mapping['probe_root_ids']) & set(mapping['witness_root_ids']),
                 'Probe and independent witness roots overlap')


def _adapt_start(start, config, mapping, groups, raw_rows, report):
    own=[group for group in groups if group['start_seed']==start]
    core.require(len(own)==80, 'Incomplete starting calibration group inventory')
    lookup={group['root_id']:group for group in own}
    core.require(len(lookup)==80, 'Duplicate calibration root within start')
    core.require(len({group['group_id'] for group in own})==80, 'Duplicate calibration group within start')
    raw=calibration._index(raw_rows,'request_id')
    scored=calibration._index(report['rescored_rows'],'request_id')
    core.require(set(raw)==set(scored) and len(raw)==640, 'Completed raw/rescored coverage differs')
    batches={}
    for name in ('probe','witness'):
        ordered_roots=mapping[name+'_root_ids']
        core.require(all(root in lookup for root in ordered_roots), 'Frozen '+name+' root absent from calibration')
        selected=[lookup[root] for root in ordered_roots]
        kinds=[group['kind'] for group in selected]
        core.require((kinds.count('CONFLICT')==12 and kinds.count('ALIGNED')==4) if name=='probe'
                     else kinds==['CONFLICT']*16, 'Frozen '+name+' conflict/aligned allocation differs')
        rows=[]
        for group in selected:
            ids=group['request_ids']
            core.require(len(ids)==8 and len(set(ids))==8 and all(rid in raw for rid in ids), 'Incomplete/duplicate frozen G8')
            for rid in ids:
                original=raw[rid];score=scored[rid]
                core.require(original['group_id']==score['group_id']==group['group_id'] and
                             original['root_id']==score['root_id']==group['root_id'] and
                             original['case_id']==score['case_id']==group['case_id'] and
                             original['start_seed']==start, 'Wrong calibration grouping')
                native=score['native_score']
                core.require(native==original['native_score'] and native['J']==score['J'] and
                             int(native['known_mask']['S'])==score['known'] and
                             type(score['c']) is int and score['c'] in (0,1) and
                             (score['known'] or score['c']==0), 'Native-known scoring projection differs')
                row=copy.deepcopy(original)
                row.update(J=int(native['J']),exact_authorised=score['c'])
                rows.append(row)
        core.require(len(rows)==128 and len({row['request_id'] for row in rows})==128, 'Diagnostic batch must contain 128 unique rows')
        half_groups=[[lookup[root]['group_id'] for root in half] for half in mapping[name+'_halves']]
        # validate_partition takes GROUP IDs. Root IDs must not be passed here.
        halves=diagnostics.validate_partition(rows,half_groups)
        core.require([len(half) for half in halves]==[64,64], 'Diagnostic halves must contain 64 rows each')
        batches[name]=dict(root_ids=copy.deepcopy(ordered_roots),group_ids=[g['group_id'] for g in selected],
                           rows=rows,half_group_ids=half_groups,half_rows=copy.deepcopy(halves),
                           groups=copy.deepcopy(selected))
    core.require(not {row['request_id'] for row in batches['probe']['rows']} &
                 {row['request_id'] for row in batches['witness']['rows']}, 'Probe and witness request overlap')
    bindings=config['start_bindings'][str(start)]
    expected_identity=dict(experiment_id=config['experiment_id'],parent='b0',start_seed=start,arm='START',step=0,
                           policy_sha256=bindings['policy_sha256'],reference_sha256=bindings['reference_sha256'])
    return dict(expected_identity=expected_identity,
                production_plan=dict(scope='DRY',step=0,groups=copy.deepcopy(batches['probe']['groups'])),
                probe=batches['probe'],witness=batches['witness'],
                counts=dict(probe_groups=16,probe_responses=128,probe_halves=[64,64],
                            witness_groups=16,witness_responses=128,witness_halves=[64,64]),
                scoring_semantics='native_training_balanced_only_known_S_and_independent_authorised_set',
                raw=copy.deepcopy(report['raw']),completion_receipt=copy.deepcopy(report['completion_receipt']),
                dry_analysis_sha256=core.object_sha(report),dry_metrics=copy.deepcopy(report['metrics']))


def build_diagnostic_inputs(config, freeze, root, receipts, native_src):
    """Return both start-specific batches, only after both full DRY gates pass."""
    cfg,frozen,root=calibration._freeze(config,freeze,root)
    core.require(set(receipts)==set(core.STARTS), 'Both registered DRY completion receipts required')
    # Do not even begin single-start raw analysis if the other start is unfinished.
    for start in core.STARTS:
        receipt=core.read_json(receipts[start])
        core.require(receipt.get('status')=='CALIBRATION_CAPTURE_COMPLETE_NOT_GPU_ADMISSION' and
                     receipt.get('scope')=='DRY' and receipt.get('start_seed')==start and
                     receipt.get('responses')==640, 'Both DRY starts must be complete before diagnostic preparation')
    splits_path=core.check_artifact(root,frozen['artifacts']['probe_splits'])
    mapping=core.read_json(splits_path)
    _validate_roots(mapping)
    groups=_read_bound_rows(root,frozen['artifacts']['calibration_groups'])
    core.require(len(groups)==160 and len({g['group_id'] for g in groups})==160, 'Wrong/duplicate calibration group inventory')
    reports={start:calibration.analyze_dry(config,freeze,root,receipts[start],start,native_src) for start in core.STARTS}
    failures={str(start):report['failures'] for start,report in reports.items()
              if report['status']!='DRY_METRICS_PASS_GPU_CHECK_STILL_REQUIRED' or report['failures']}
    core.require(not failures, 'Both DRY signal gates must pass before diagnostic preparation: '+json.dumps(failures))
    starts={}
    for start in core.STARTS:
        raw=_read_bound_rows(root,reports[start]['raw'])
        starts[str(start)]=_adapt_start(start,cfg,mapping,groups,raw,reports[start])
    return dict(schema='sr-satdose-diagnostic-inputs-v1',experiment_id=cfg['experiment_id'],
                status='FIXED_DIAGNOSTIC_INPUTS_PREPARED_NOT_GPU_ADMISSION',permission='NO_TRAINING',
                model_calls=0,optimizer_steps=0,parameter_updates=0,
                counts=dict(completed_starts=2,source_dry_responses=1280,
                            selected_probe_responses=256,selected_witness_responses=256,
                            new_generations=0),
                config_sha256=core.file_sha(config),design_freeze_sha256=core.file_sha(freeze),
                frozen_artifacts={name:copy.deepcopy(frozen['artifacts'][name]) for name in
                                  ('probe_splits','calibration_groups','dry_manifest','private_cases','inputs','images','source_lock')},
                native_scorer_pins=copy.deepcopy(reports[11031]['native_scorer_pins']),
                implementation=dict(adapter_sha256=core.file_sha(__file__),analysis_sha256=core.file_sha(calibration.__file__),
                                    partition_helper_sha256=core.file_sha(diagnostics.__file__),core_sha256=core.file_sha(core.__file__)),
                starts=starts)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','freeze','evidence-root','receipt-11031','receipt-11032','native-src','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(argv)
    result=build_diagnostic_inputs(args.config,args.freeze,args.evidence_root,
                                  {11031:args.receipt_11031,11032:args.receipt_11032},args.native_src)
    calibration.write_once(args.output,result)
    print(json.dumps({key:result[key] for key in ('status','permission','model_calls','optimizer_steps')},indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
