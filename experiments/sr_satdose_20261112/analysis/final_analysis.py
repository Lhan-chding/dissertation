"""External, read-only aggregation of all fourteen completed MAIN endpoints.

Load under a unique external module name after authenticating original packages.
No import-path changes, scoring replacements, model calls, writes, partial report,
MAIN release, checkpoint selection, bootstrap tuning or follow-up authorization.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import types

EXPERIMENT='SR-SATDOSE-1-20261112'
PERMISSION='FINAL_ANALYSIS_ONLY_NO_TRAINING_MAIN_OR_FOLLOWUP_AUTHORITY'
STARTS=(11031,11032)
ALLOCATION={f'{s}/START/0':1952 for s in STARTS}
ALLOCATION.update({f'{s}/{a}/{step}':1952 if step==24 else 256
                   for s in STARTS for a in ('T0','T1') for step in (8,16,24)})
PLANNED_GENERATIONS=dict(DRY=1280,PREFIX_CALIB=96,DEV=1512,TRAIN=12288,MAIN_FULL=11712,MAIN_INTERMEDIATE=2048)


def require(value,message):
    if not value:raise PermissionError('HOLD: '+message)


def sha(path):
    path=Path(path)
    require(path.is_absolute() and path.resolve(strict=True)==path and path.is_file() and not path.is_symlink(),
            'Noncanonical source/evidence file')
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(1<<20),b''):h.update(data)
    return h.hexdigest()


def read(path):
    def pairs(values):
        out=dict(values);require(len(out)==len(values),'Duplicate JSON key');return out
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,
                      parse_constant=lambda value:require(False,'Nonfinite JSON'))


def _reference(reference):
    require(isinstance(reference,dict) and set(reference)=={'path','sha256'} and isinstance(reference['path'],str) and
            isinstance(reference['sha256'],str) and re.fullmatch('[0-9a-f]{64}',reference['sha256']), 'Invalid trusted reference')


class Evidence:
    def __init__(self,root):self.root=root;self.observed={}
    def path(self,path,expected):
        path=Path(path);require(sha(path)==expected,'Source/evidence SHA differs: '+str(path))
        require(self.observed.setdefault(str(path),expected)==expected,'Evidence changed within analysis')
        return path
    def ref(self,reference):
        _reference(reference);relative=Path(reference['path'])
        require(not relative.is_absolute() and '..' not in relative.parts,'Evidence escapes root')
        return self.path(self.root/relative,reference['sha256'])
    def absolute(self,reference):
        _reference(reference);return self.path(Path(reference['path']),reference['sha256'])
    def json(self,reference):return read(self.ref(reference))
    def stable(self):
        for path,digest in self.observed.items():require(sha(path)==digest,'Observed evidence changed during aggregation')


def _inventory(package):
    require(package.is_absolute() and package.resolve(strict=True)==package and package.is_dir(),'Noncanonical original package')
    result={}
    for parent,dirs,files in os.walk(package,followlinks=False):
        for name in dirs:
            path=Path(parent)/name;require(path.resolve(strict=True)==path and not path.is_symlink(),'Original source directory symlink')
        for name in files:
            path=Path(parent)/name;result[str(path.relative_to(package))]=sha(path)
    return result


def _sources(e,spec):
    require(set(spec)=={'analyzer','original_train_release','primary_helper'},'Exact external analysis source specification required')
    require(e.absolute(spec['analyzer'])==Path(__file__).resolve(),'Unpinned external final analyzer')
    release=e.json(spec['original_train_release']);package=Path(release['release_root'])
    require(release.get('schema')=='sr-satdose-first-two-release-v1' and release.get('experiment_id')==EXPERIMENT and
            release.get('status')=='AUDITED_FIRST_TWO_AND_READONLY_RELOAD_ONLY' and release.get('evidence_root')==str(e.root) and
            _inventory(package)==release['sources'],'Original TRAIN release inventory/identity differs')
    require(not Path(__file__).resolve().is_relative_to(package),'Final analyzer must be external to immutable TRAIN package')
    for name,module in tuple(sys.modules.items()):
        if name in ('analysis','execution','runtime'):
            paths=list(getattr(module,'__path__',[]))
            require(paths and all(Path(p).resolve()==package/name for p in paths),'Shadowed original package: '+name)
        elif name.startswith(('analysis.','execution.','runtime.')):
            require(getattr(module,'__file__',None) and Path(module.__file__).resolve()==package/(name.replace('.','/')+'.py'),
                    'Shadowed original module: '+name)
    evaluation=sys.modules.get('analysis.evaluation')
    require(evaluation is not None,'Preauthenticated original analysis.evaluation must already be imported')
    core=evaluation.core
    require(sys.modules.get('satdose_core') is core and Path(core.__file__).resolve()==package/'supplied_package/tools/satdose_core.py',
            'Original core namespace identity differs')
    for name in ('analysis/evaluation.py','analysis/calibration.py','supplied_package/tools/satdose_core.py',
                 'supplied_package/tools/analyze_primary.py','supplied_package/config/source_bindings.json'):
        e.path(package/name,release['sources'][name])
    primary_path=e.absolute(spec['primary_helper'])
    require(primary_path==package/'supplied_package/tools/analyze_primary.py' and
            spec['primary_helper']['sha256']==release['sources']['supplied_package/tools/analyze_primary.py'],
            'Primary helper is not the frozen supplied implementation')
    data=primary_path.read_bytes();require(hashlib.sha256(data).hexdigest()==spec['primary_helper']['sha256'],'Primary source changed')
    # Execute authenticated current bytes without replacing an original namespace
    # or consulting a stale bytecode cache. Its satdose_core import is already bound.
    primary=types.ModuleType('_satdose_external_primary_'+spec['primary_helper']['sha256'][:16])
    primary.__file__=str(primary_path);exec(compile(data,str(primary_path),'exec'),primary.__dict__)
    require(primary.analyze.__defaults__==(10000,2026111219),'Frozen primary bootstrap defaults changed')
    return evaluation,primary,release,package


def _complete_population(e,evaluation,freeze,refs):
    require(isinstance(refs,dict) and set(refs)==set(ALLOCATION),'All exact fourteen MAIN endpoints required before any scoring')
    manifest=list(evaluation.core.read_jsonl(e.ref(freeze['artifacts']['main_manifest'])))
    grouped=evaluation.validate_manifest(manifest,'MAIN');seen=set();receipts={}
    for key,count in ALLOCATION.items():
        reference=refs[key];require(set(reference)=={'receipt','binding'},'Explicit receipt and binding required')
        receipt=e.json(reference['receipt']);binding=e.json(reference['binding'])
        ep=binding['endpoint'];actual=f"{ep['start_seed']}/{ep['arm']}/{ep['step']}"
        require(actual==key and binding.get('scope')==receipt.get('scope')=='MAIN' and
                receipt.get('schema')=='sr-satdose-evaluation-completion-v1' and
                receipt.get('status')=='EVALUATION_ENDPOINT_CAPTURE_COMPLETE_NO_TRAINING_PERMISSION' and
                receipt.get('responses')==binding.get('requests')==count and
                receipt.get('endpoint_binding_sha256')==reference['binding']['sha256'],
                'MAIN endpoint completion/identity/allocation differs')
        require(reference['receipt']['path'] not in seen,'Duplicate MAIN receipt');seen.add(reference['receipt']['path'])
        selected=grouped[ep['start_seed'],ep['arm'],ep['step']]
        require(binding['request_ids_sha256']==evaluation.core.object_sha([r['request_id'] for r in selected]),
                'Frozen endpoint request order differs')
        receipts[key]=receipt
    require(sum(ALLOCATION.values())==13760,'Internal registered denominator changed')
    return manifest,grouped,receipts


def _scored_rows(result,selected,endpoint):
    rows=result['rows'];summary=result['summary']
    require(summary.get('status')=='COMPLETE_ENDPOINT_CPU_SCORED' and summary.get('scope')=='MAIN' and
            summary.get('endpoint')==endpoint and summary['metrics']['assigned']==len(selected),'Original endpoint scoring scope differs')
    require([r.get('request_id') for r in rows]==[m['request_id'] for m in selected], 'Scored population is missing, duplicated or reordered')
    for row,manifest in zip(rows,selected,strict=True):
        require(all(type(row.get(k)) is type(v) and row[k]==v for k,v in manifest.items()),'Scored pairing/case/CRN differs')
        for key in ('S_hit','S_known','A','B_hit','B_known','S_and_A','json_valid','completed_generation'):
            require(type(row[key]) is int and row[key] in (0,1),'Invalid original binary metric')
        require(row['S_hit']<=row['S_known'] and row['B_hit']<=row['B_known'] and row['S_and_A']==row['S_hit']*row['A'],
                'Unknown outcome cannot become confirmed correct')
    return rows


def _primary_rows(by_endpoint,step):
    baseline=[r for r in by_endpoint['11031/START/0'] if (r['panel'],r['view'])==('list8','conflict')]
    coordinates=[(r['root_id'],r['draw'],r['structure'],r['case_id'],r['input_id'],r['seed']) for r in baseline]
    require(len(coordinates)==256 and len(set(coordinates))==256,'Primary complete paired cells required')
    root_order=list(dict.fromkeys(r['root_id'] for r in baseline));require(len(root_order)==128,'Exactly128 primary roots required')
    out=[]
    for start in STARTS:
        selected={label:[r for r in by_endpoint[f'{start}/{arm}/{at}'] if (r['panel'],r['view'])==('list8','conflict')]
                  for label,arm,at in (('start','START',0),('T0','T0',step),('T1','T1',step))}
        for rows in selected.values():
            require([(r['root_id'],r['draw'],r['structure'],r['case_id'],r['input_id'],r['seed']) for r in rows]==coordinates,
                    'Paired primary case/draw order or CRN differs')
        for root in root_order:
            cells={label:[r for r in rows if r['root_id']==root] for label,rows in selected.items()}
            require(all(len(rows)==2 and {r['draw'] for r in rows}=={0,1} for rows in cells.values()),'Missing/duplicate primary draw')
            row=dict(root_id=root,structure=cells['start'][0]['structure'],start_seed=start,draws=2)
            for label,rows in cells.items():row[label+'_correct']=sum(r['S_hit'] for r in rows)
            row['request_ids']={label:[r['request_id'] for r in rows] for label,rows in cells.items()}
            out.append(row)
    return out


def _nullable(values):
    return dict(assigned=len(values),known=sum(v is not None for v in values),unknown=sum(v is None for v in values),
                confirmed_true=sum(v==1 for v in values),confirmed_false=sum(v==0 for v in values))


def _panel_tables(rows):
    groups=defaultdict(list)
    for row in rows:groups[row['panel'],row['view'],row['structure']].append(row)
    out=[]
    for (panel,view,structure),values in sorted(groups.items()):
        n=len(values);native=[r['native_score'] for r in values if r['native_score'] is not None]
        out.append(dict(panel=panel,view=view,structure=structure,assigned=n,
            counts={k:sum(r[k] for r in values) for k in ('A','S_hit','S_known','S_and_A','B_hit','B_known','json_valid','completed_generation')},
            S_unknown=sum(1-r['S_known'] for r in values),B_unknown=sum(1-r['B_known'] for r in values),
            source_counts=dict(Counter(r['source_class'] for r in values)),finish_counts=dict(Counter(r['finish_reason'] for r in values)),
            J_native=dict(applicable=len(native),NA=n-len(native),**_nullable([r['J_native'] for r in values if r['native_score'] is not None])),
            native_evidence={k:dict(NA=n-len(native),**_nullable([r.get(k) for r in native]))
                             for k in ('certificate','certificate_truth','certificate_sufficient','P','E_raw','E_anchored')},
            registered_report_order=dict(NA=sum(not r['U_applicable'] for r in values),
                **_nullable([r['U'] for r in values if r['U_applicable']]))))
    return out


def _prefix(rows):
    values=[r for r in rows if r['panel']=='prefix10']
    if not values:return None
    out={}
    for label in ('strict','eos_single_brace_sensitivity'):
        levels={}
        for view in ('p_stable','p_revtie','p_placebo'):
            cell=[r for r in values if r['view']==view];n=len(cell);require(n==192,'Prefix denominator differs')
            metrics=[r if label=='strict' else r['eos_single_brace_sensitivity']['metrics'] for r in cell]
            hit=sum(r['S_hit'] for r in metrics);unknown=sum(1-r['S_known'] for r in metrics)
            levels[view]=dict(assigned=n,confirmed_S_correct=hit,S_unknown=unknown,S_hit_all_assigned=hit/n,
                S_known_coverage=(n-unknown)/n,S_finite_identification_bounds=[hit/n,(hit+unknown)/n],
                A_confirmed=sum(r['A'] for r in metrics),json_valid=sum(r['json_valid'] for r in metrics),
                prefix_induced_observable_hits=sum(r['prefix_choice_hit']==1 for r in metrics),
                eligible_brace=sum(r['eos_single_brace_sensitivity']['eligible'] for r in cell))
        contrasts={}
        for a,b in (('p_stable','p_revtie'),('p_placebo','p_revtie'),('p_stable','p_placebo')):
            x,y=levels[a],levels[b];lo,hi=x['S_finite_identification_bounds'];lo2,hi2=y['S_finite_identification_bounds']
            contrasts[a+'-'+b]=dict(S_hit_difference=x['S_hit_all_assigned']-y['S_hit_all_assigned'],
                unknown_rate_difference=x['S_unknown']/x['assigned']-y['S_unknown']/y['assigned'],
                finite_identification_bounds_not_CI=[lo-hi2,hi-lo2])
        out[label]=dict(levels=levels,contrasts=contrasts,
                       role='PRIMARY_STRICT_OUTCOMES' if label=='strict' else 'SENSITIVITY_ONLY_NEVER_REPLACE_STRICT')
    return out


def _enumeration(raw,case,core):
    """Literal declared Alpha order, not a new correctness or mechanism scorer."""
    obj,_=core.parse_selected(raw['full_text'],raw['finish_reason']);order=None;status='UNPARSEABLE'
    if raw['finish_reason'] not in ('balanced','eos'):status='INCOMPLETE_GENERATION'
    elif obj is not None:
        facts=obj.get('evidence');status='MISSING_OR_MALFORMED_EVIDENCE'
        if isinstance(facts,list) and all(isinstance(x,dict) for x in facts):
            order=[x.get('category') for x in facts if x.get('series')=='Alpha']
            if not all(type(x) is str for x in order):order=None
            names=case['task']['world']['categories']
            status=('COMPLETE_UNIQUE_ALPHA_SEQUENCE' if order is not None and len(order)==len(names) and
                    len(set(order))==len(order) and set(order)==set(names) else 'PARTIAL_OR_DUPLICATE_ALPHA_SEQUENCE')
    known=status=='COMPLETE_UNIQUE_ALPHA_SEQUENCE'
    names=case['task']['world']['categories'];listed=case['audit'].get('list_order')
    return dict(status=status,alpha_categories=order,
        equals_chart_category_order=(order==names if known else None),
        equals_explicit_list_order=(order==listed if known and listed is not None else None),
        explicit_list_order_applicable=listed is not None)


def _joint(rows,raw,cases,core):
    counts=Counter();marginal=Counter();selection=Counter()
    for row,original in zip(rows,raw,strict=True):
        enum=_enumeration(original,cases[row['case_id']],core)
        category=json.dumps(enum,sort_keys=True,separators=(',',':'))
        counts[row['panel'],row['view'],category,row['source_class'],row['S_hit'],row['S_known'],row['A']]+=1
        marginal[row['panel'],row['view'],category]+=1;selection[row['panel'],row['view'],row['source_class']]+=1
    return dict(role='DESCRIPTIVE_OBSERVABLE_DECLARATIONS_NOT_INTERNAL_SOURCE_OR_CAUSAL_MEDIATION',
        assigned=len(rows),joint=[dict(panel=p,view=v,enumeration=json.loads(en),source_class=src,S_hit=s,S_known=k,A=a,count=n)
            for (p,v,en,src,s,k,a),n in sorted(counts.items())],
        enumeration_marginal=[dict(panel=p,view=v,enumeration=json.loads(en),count=n) for (p,v,en),n in sorted(marginal.items())],
        selection_marginal=[dict(panel=p,view=v,source_class=src,count=n) for (p,v,src),n in sorted(selection.items())])


COST_STAGES=('DRY','PREFIX_CALIB','SHADOW','DEV','TRAIN','MAIN','INCIDENTS')
COST_METRICS=('completion_attempts','generated_tokens','extra_forward_sequences','diagnostic_gradient_calls','backward_calls',
              'optimizer_steps','scheduler_steps','wall_seconds','allocation_gpu_seconds','measured_gpu_compute_seconds',
              'peak_allocated_bytes','peak_reserved_bytes')


def _costs(e,core,cost_refs,main_receipts):
    require(isinstance(cost_refs,dict) and set(cost_refs)==set(COST_STAGES),'Explicit complete stage cost map required')
    used_files=set();used_fields=set();resource_levels=defaultdict(set);resource_observations=set();out={}
    for stage in COST_STAGES:
        spec=cost_refs[stage]
        require(set(spec)=={'accounting','measurements','resource_evidence'} and
                all(isinstance(v,list) for v in spec.values()),'Malformed stage cost specification')
        sums=defaultdict(list);accounting=copy.deepcopy(spec['accounting'])
        if stage=='MAIN':accounting += [r['accounting'] for r in main_receipts.values()]
        for reference in accounting:
            path=e.ref(reference);require(path not in used_files,'Duplicate accounting cost file');used_files.add(path)
            events=list(core.read_jsonl(path));require(events,'Empty supplied accounting does not establish zero cost')
            for event in events:
                require(event.get('kind') in COST_METRICS and event['kind'] not in
                        ('wall_seconds','allocation_gpu_seconds','measured_gpu_compute_seconds','peak_allocated_bytes','peak_reserved_bytes') and
                        type(event.get('count')) is int and event['count']>=0,'Invalid native accounting cost event')
                sums[event['kind']].append(event['count'])
        measurements=[]
        for measure in spec['measurements']:
            timed=measure.get('metric') in ('wall_seconds','allocation_gpu_seconds','measured_gpu_compute_seconds')
            require(set(measure)=={'reference','field_path','metric'}|({'resource_scope'} if timed else set()) and measure['metric'] in COST_METRICS and
                    isinstance(measure['field_path'],list) and measure['field_path'],'Malformed explicit cost field pointer')
            path=e.ref(measure['reference']);value=read(path)
            require(path not in used_files,'JSON measurement cannot double count an accounting file')
            for key in measure['field_path']:
                require((isinstance(value,dict) and type(key) is str and key in value) or
                        (isinstance(value,list) and type(key) is int and 0<=key<len(value)),'Missing cost measurement field')
                value=value[key]
            require(type(value) in (int,float) and math.isfinite(value) and value>=0,'Invalid/nonfinite actual cost measurement')
            if not timed:require(type(value) is int,'Actual count/byte cost must be integer')
            else:
                scope=measure['resource_scope']
                require(set(scope)=={'job_id','level','observation_id'} and scope['level'] in ('job','step','measurement') and
                        all(isinstance(scope[k],str) and scope[k] for k in ('job_id','observation_id')),'Missing resource observation scope')
                if measure['metric']=='measured_gpu_compute_seconds':
                    require(scope['level']=='measurement','Measured GPU computation cannot be inferred from Slurm allocation')
                else:
                    require(scope['level'] in ('job','step'),'Allocation/elapsed requires actual job or step observation')
                    resource_levels[scope['job_id']].add(scope['level'])
                    require(len(resource_levels[scope['job_id']])==1,'Overlapping Slurm parent job and child step cost')
                observation=(measure['metric'],scope['job_id'],scope['level'],
                             'whole_job' if scope['level']=='job' else scope['observation_id'])
                require(observation not in resource_observations,'Duplicate resource observation');resource_observations.add(observation)
            identity=(str(path),tuple(measure['field_path']))
            require(identity not in used_fields,'Duplicate observed cost field');used_fields.add(identity)
            sums[measure['metric']].append(value);measurements.append(dict(measure,value=value))
        resources=[dict(reference=reference,document=e.json(reference)) for reference in spec['resource_evidence']]
        values={metric:((max(sums[metric]) if metric.startswith('peak_') else sum(sums[metric])) if sums[metric] else None)
                for metric in COST_METRICS}
        out[stage]=dict(observed=values,metric_status={k:'OBSERVED_FROM_BOUND_EVIDENCE' if v is not None else 'UNKNOWN'
                                                    for k,v in values.items()},
            accounting=accounting,measurements=measurements,resource_evidence=resources,
            coverage='Supplied immutable observation inventory; omitted technical incidents are not inferred absent.')
    require(out['MAIN']['observed']['completion_attempts']>=13760,'MAIN observed attempts omit registered completed population')
    return out


def analyze_complete(root,*,config_ref,freeze_ref,endpoints,source_spec,cost_refs):
    """Return all rows/tables only after fourteen complete endpoints pass.

    endpoints maps exact start/arm/step to root-relative receipt/binding refs.
    source_spec binds this external analyzer, original_train_release (relative)
    and frozen primary_helper (absolute). Existing original modules must already
    be authenticated/imported; this function never edits sys.path or namespaces.
    cost_refs has exact COST_STAGES, each accounting JSONL refs, measurements
    ({reference, field_path, metric}), and resource_evidence JSON refs. Time fields
    additionally bind resource_scope {job_id, level, observation_id}. Missing
    measurements remain UNKNOWN; resource units follow metric names, not guesses.
    """
    root=Path(root);require(root.is_absolute() and root.resolve(strict=True)==root,'Noncanonical evidence root')
    e=Evidence(root);evaluation,primary,release,package=_sources(e,source_spec)
    cfg=e.json(config_ref);freeze=e.json(freeze_ref)
    require(cfg['experiment_id']==freeze['experiment_id']==EXPERIMENT and cfg['parent']=='b0' and
            freeze_ref['path']=='DESIGN_FREEZE.json' and freeze['config_sha256']==config_ref['sha256'] and
            release['design_freeze_sha256']==freeze_ref['sha256'] and release['config_sha256']==config_ref['sha256'],
            'Original scientific config/freeze differs')
    manifest,groups,receipts=_complete_population(e,evaluation,freeze,endpoints)
    for reference in freeze['artifacts'].values():e.ref(reference)
    cases=evaluation._index(evaluation.core.read_jsonl(e.ref(freeze['artifacts']['private_cases'])),'case_id')
    inputs=evaluation._index(evaluation.core.read_jsonl(e.ref(freeze['artifacts']['inputs'])),'input_id')
    images=evaluation._index(evaluation.core.read_jsonl(e.ref(freeze['artifacts']['images'])),'image_file')
    by_endpoint={};summaries={};panels={};prefix={};joints={};total_tokens=0;allrows=[]
    for key in ALLOCATION:
        refs=endpoints[key];binding=e.json(refs['binding']);receipt=receipts[key];ep=binding['endpoint']
        result=evaluation.analyze_completed_endpoint(e.ref(config_ref),e.ref(freeze_ref),root,e.ref(refs['receipt']),
            expected_receipt_sha256=refs['receipt']['sha256'],endpoint_binding_path=e.ref(refs['binding']),
            expected_endpoint_binding_sha256=refs['binding']['sha256'])
        selected=groups[ep['start_seed'],ep['arm'],ep['step']]
        scores=_scored_rows(result,selected,ep);by_endpoint[key]=scores;summaries[key]=result['summary'];allrows.extend(scores)
        for field in ('scorer_sources','analysis_sources'):
            for path,digest in result['summary'][field].items():e.path(Path(path),digest)
        raw=list(evaluation.core.read_jsonl(e.ref(receipt['raw'])));e.ref(receipt['accounting'])
        require([r['request_id'] for r in raw]==[r['request_id'] for r in scores],'Raw changed after endpoint scoring')
        for row,complete in zip(raw,receipt['complete_files'],strict=True):
            e.ref(complete);capture=e.path(Path(row['native_capture_path']),row['native_capture_sha256'])
            begin=capture.with_name(capture.name.replace('.CAPTURE.json','.BEGIN.json'));e.path(begin,sha(begin))
            require(read(begin)==dict(request_id=row['request_id'],seed=row['seed'],endpoint=ep,
                    design_freeze_sha256=freeze_ref['sha256'],status='ATTEMPTED_UNKNOWN'),
                    'Original BEGIN changed after endpoint scoring')
            total_tokens+=len(row['tokens'])
        for m in selected:
            image=images[inputs[m['input_id']]['image_file']];e.ref(dict(path=image['image_file'],sha256=image['png_sha256']))
        if binding['checkpoint']:
            for reference in binding['checkpoint'].values():e.ref(reference)
        panels[key]=_panel_tables(scores);prefix[key]=_prefix(scores);joints[key]=_joint(scores,raw,cases,evaluation.core)
    require(len(allrows)==13760 and len({r['request_id'] for r in allrows})==13760,'Full fixed MAIN denominator differs')
    primary_rows=_primary_rows(by_endpoint,24)
    # Exactly the supplied implementation/defaults, not a reimplementation or tunable wrapper.
    primary_result=primary.analyze(primary_rows)
    trajectories=[]
    for step in (8,16,24):
        table=primary_rows if step==24 else _primary_rows(by_endpoint,step)
        trajectories.append(dict(step=step,role='DESCRIPTIVE_ONLY_FIXED_TRAINED_PATHS',per_start={str(s):dict(
            delta_T1_minus_T0=sum((r['T1_correct']-r['T0_correct'])/2 for r in table if r['start_seed']==s)/128,
            absolute_gain_T1_minus_START=sum((r['T1_correct']-r['start_correct'])/2 for r in table if r['start_seed']==s)/128)
            for s in STARTS}))
    costs=_costs(e,evaluation.core,cost_refs,receipts)
    require(costs['MAIN']['observed']['generated_tokens']>=total_tokens,'MAIN token cost omits immutable completed captures')
    require(_inventory(package)==release['sources'],'Original source inventory changed during final analysis');e.stable()
    return dict(schema='sr-satdose-final-analysis-v1',status='COMPLETE_REGISTERED_MAIN_ANALYSIS_NO_AUTOMATIC_FOLLOWUP',
        permission=PERMISSION,experiment_id=EXPERIMENT,parent='b0',start_seeds=list(STARTS),assigned=13760,
        endpoint_allocations=copy.deepcopy(ALLOCATION),endpoint_references=copy.deepcopy(endpoints),
        endpoint_summaries=summaries,rows=allrows,primary_root_rows=primary_rows,primary=primary_result,
        descriptive_trajectory=trajectories,panel_structure_tables=panels,prefix_bounds=prefix,mechanism_joint_tables=joints,
        costs=dict(planned_registered_generations=copy.deepcopy(PLANNED_GENERATIONS),planned_total=28936,
            registered_MAIN_completed_responses=13760,registered_MAIN_generated_tokens=total_tokens,
            observed_MAIN_completion_attempts=costs['MAIN']['observed']['completion_attempts'],
            observed_MAIN_generated_tokens=costs['MAIN']['observed']['generated_tokens'],
            additional_observed_MAIN_attempts=costs['MAIN']['observed']['completion_attempts']-13760,
            actual_stage_costs=costs,
            whole_experiment_actual_cost_status='NOT_INFERRED_FROM_PLANNED_BUDGET_OR_MAIN_ONLY',
            this_analysis_new_model_calls=0,this_analysis_new_optimizer_calls=0),
        source_spec=copy.deepcopy(source_spec),observed_files=dict(sorted(e.observed.items())),
        main_terminal_authority_verified=False,training_admission=False,followup_authorized=False,
        limitations=['Primary CI conditions on four fixed trained paths, two fixed starts and CRN draws; not a training-seed population CI.',
            'Only step24 list8/conflict S_hit T1-T0 is confirmatory; step8/16 trajectories and all mechanism tables are descriptive.',
            'UNKNOWN selected stays in assigned S_hit denominator with zero confirmed success; missing records cause HOLD.',
            'Strict results and limited single-brace sensitivity remain separate; sensitivity never replaces strict results.',
            'Native J and certificate fields are NA where the original scorer is inapplicable; absent applicable certificate fields remain UNKNOWN.',
            'Literal enumeration and selected/source declarations do not reveal an internal source or establish causal mediation.',
            'No alpha or mutual-information model is fitted; report-order U remains NA when absent from the frozen case contract.',
            'Run-noise SD0/3/5pp bands are frozen hypothetical sensitivity, not estimated run variance or calibrated population CIs.',
            'Cost documents retain incidents and resource evidence as supplied; MAIN completion does not establish total experiment cost.',
            'Additional or unresolved attempts are retained as cost observations, never scored as extra scientific samples; unrecorded tokens remain unknown.',
            'Allocation GPU time is separate from measured GPU computation; absent peaks/utilization remain UNKNOWN, and overlapping Slurm parent/step time is rejected.',
            'This CPU analyzer validates completed evidence; it neither authenticates terminal MAIN release nor authorizes generation or Stage2.'])
