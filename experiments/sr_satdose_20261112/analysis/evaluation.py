"""CPU-only scoring of complete, immutable DEV/MAIN endpoints; no generation CLI.

Completion contract (all file references are {path,sha256}, relative to the
frozen evidence root; hashes use satdose_core.object_sha):

* Binding schema sr-satdose-evaluation-endpoint-binding-v1: scope; endpoint
  {experiment_id,start_seed,arm,step,policy_sha256,reference_sha256}; requests;
  request_ids_sha256 (ORIGINAL manifest order, filtered to this endpoint);
  design_freeze_sha256, config_sha256, source_lock_sha256, manifest_sha256,
  execution_sources_sha256; checkpoint=None for START, otherwise {commit,state}.
* Receipt schema sr-satdose-evaluation-completion-v1, status
  EVALUATION_ENDPOINT_CAPTURE_COMPLETE_NO_TRAINING_PERMISSION; scope;
  endpoint_binding_sha256; endpoint_before/endpoint_after; responses;
  optimizer_created=scheduler_created=False; optimizer_steps=scheduler_steps=
  parameter_updates=extra_forward_sequences=0; raw/accounting references;
  complete_files references in ORIGINAL filtered manifest order. Raw preserves
  EVERY frozen GenerationBridge COMPLETE field, adds manifest metadata (scope
  replaces or accompanies split) and suffix_text. BEGIN/CAPTURE/COMPLETE remain
  unmodified original bridge records in the receipt directory.

The caller supplies trusted binding and receipt SHA values. This analyzer
checks captured endpoint identity and checkpoint bytes; it does NOT authenticate
training admission, MAIN terminal-release authority, or live GPU construction.
No missing/failed/UNKNOWN request is converted to a behavioral zero. Prefix
single-brace sensitivity is separate from strict outcomes, never OR-selected.
Final primary bootstrap and global DEV stopping decisions belong elsewhere.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import hashlib
import importlib
import importlib.util
import math
from pathlib import Path
import re
import sys

from . import calibration

core = calibration.core
require = core.require
EXPERIMENT = 'SR-SATDOSE-1-20261112'
PERMISSION = 'ANALYSIS_ONLY_NO_TRAINING_OR_MAIN_RELEASE'
ENDPOINT_KEYS = {'experiment_id','start_seed','arm','step','policy_sha256','reference_sha256'}
MANIFEST_KEYS = {'request_id','split','start_seed','arm','step','panel','view','root_id',
                 'structure','draw','seed','case_id','input_id'}


def _sha(value):return isinstance(value,str) and re.fullmatch('[a-f0-9]{64}',value) is not None


def _bound(root,ref):
    require(isinstance(ref,dict) and set(ref)=={'path','sha256'} and _sha(ref['sha256']), 'Invalid file reference')
    relative=Path(ref['path'])
    require(not relative.is_absolute() and '..' not in relative.parts,'Escaped artifact')
    path=root/relative
    require(path.resolve(strict=True)==path and path.is_file() and not path.is_symlink(),'Noncanonical artifact')
    require(core.file_sha(path)==ref['sha256'],'Artifact SHA differs: '+str(relative))
    return path


def _index(rows,key):
    result={}
    for row in rows:
        value=row[key]
        require(isinstance(value,str) and value and value not in result,'Duplicate/invalid '+key)
        result[value]=row
    return result


def endpoint_allocations(scope):
    """The exact pre-registered endpoint denominator; no efficacy-based selection."""
    require(scope in ('DEV','MAIN'),'Only DEV/MAIN analysis exists')
    result={(start,'START',0):(108 if scope=='DEV' else 1952) for start in core.STARTS}
    for start in core.STARTS:
        for arm in core.ARMS:
            for step in ((2,8,16) if scope=='DEV' else (8,16,24)):
                result[start,arm,step]=108 if scope=='DEV' else (1952 if step==24 else 256)
    return result


def _layout(scope,arm,step):
    if scope=='DEV':return {'list8':(32,('aligned','conflict','no_list'),1),'retention8':(12,('v0',),1)}
    if arm!='START' and step in (8,16):return {'list8':(128,('conflict',),2)}
    return {'list8':(128,('aligned','conflict','no_list'),2),
            'swap8':(32,('p0l0','p0l1','p1l0','p1l1'),1),
            'authority8':(96,('chart','list'),2),'prefix10':(96,('p_stable','p_revtie','p_placebo'),2),
            'retention8':(24,('v0','v1','v2','v3'),1)}


def validate_manifest(rows,scope):
    """Verify every endpoint, cell, root and paired input of a complete manifest."""
    allocation=endpoint_allocations(scope)
    require(len(rows)==sum(allocation.values()),'Incomplete/expanded frozen evaluation manifest')
    _index(rows,'request_id');groups=defaultdict(list)
    for row in rows:
        require(set(row)==MANIFEST_KEYS and row['split']==scope,'Evaluation manifest schema/split differs')
        require(all(type(row[key]) is int for key in ('start_seed','step','draw','seed')) and
                row['draw']>=0 and 0<=row['seed']<2**63,'Invalid manifest integer identity')
        endpoint=row['start_seed'],row['arm'],row['step']
        require(endpoint in allocation,'Unregistered evaluation endpoint')
        expected=f"SAT1/{scope}/{endpoint[0]}/{endpoint[1]}/s{endpoint[2]:02d}/{row['panel']}/{row['root_id']}/{row['view']}/d{row['draw']:02d}"
        require(row['request_id']==expected,'Request ID does not name its frozen cell')
        groups[endpoint].append(row)
    require(set(groups)==set(allocation),'Missing evaluation endpoint')
    paired={(r['panel'],r['root_id'],r['view'],r['draw']):{k:r[k] for k in ('structure','case_id','input_id','seed')}
            for r in groups[11031,'START',0]}
    owners={}
    for endpoint,selected in groups.items():
        require(len(selected)==allocation[endpoint],'Endpoint allocation differs')
        panels=_layout(scope,*endpoint[1:])
        require({r['panel'] for r in selected}==set(panels),'Endpoint panels differ')
        for panel,(count,views,draws) in panels.items():
            subset=[r for r in selected if r['panel']==panel];roots=defaultdict(list)
            for row in subset:roots[row['root_id']].append(row)
            require(len(roots)==count,'Panel root denominator differs')
            strata=Counter()
            for root,cells in roots.items():
                require(owners.setdefault(root,panel)==panel,'Root reused between evaluation panels')
                require(len({r['structure'] for r in cells})==1 and len(cells)==len(views)*draws and
                        {(r['view'],r['draw']) for r in cells}=={(v,d) for v in views for d in range(draws)},
                        'Incomplete/duplicate panel root cells')
                strata[cells[0]['structure']]+=1
                for row in cells:
                    key=panel,root,row['view'],row['draw']
                    projection={k:row[k] for k in ('structure','case_id','input_id','seed')}
                    require(key in paired and paired[key]==projection,'Paired endpoint input/CRN differs')
            if panel!='retention8':
                require(strata==dict.fromkeys(core.STRUCTURES,count//4),'Panel structural balance differs')
            else:
                expected=(dict(BINDING=3,IMPACT=3,SIGNED=2,INTERVAL=2,K_DIRECTION=2) if scope=='DEV'
                          else dict.fromkeys(('BINDING','IMPACT','TIES','SIGNED','INTERVAL','K_DIRECTION'),4))
                require(strata==expected,'Historical public DEV retention allocation differs')
    return dict(groups)


def load_scorers(source_lock_path,expected_sha256):
    """Load original CPU functions, verifying pins and actual imported locations."""
    require(_sha(expected_sha256) and core.file_sha(source_lock_path)==expected_sha256,'Unbound scorer source lock')
    lock=core.read_json(source_lock_path)
    legacy=Path(lock['legacy_root']).resolve(strict=True);native=Path(lock['native_src']).resolve(strict=True)
    required={'score_eval.py','analyze.py',
              'vendor/probe_ref/order_probe/__init__.py','vendor/probe_ref/order_probe/measurement.py',
              'vendor/probe_ref/order_probe/common.py','vendor/probe_ref/order_probe/inputs.py'}
    require(required<=set(lock['legacy_files']),'Missing legacy field-scorer source pins')
    checked={}
    for relative in required:
        ref=dict(path=relative,sha256=lock['legacy_files'][relative]);path=_bound(legacy,ref)
        checked[str(path)]=ref['sha256']
    pins=core.read_json(calibration.PACKAGE/'config/source_bindings.json')['native_scorer_pins']
    for relative,value in pins.items():
        require(lock['native_files'].get(relative)==value,'Native scorer differs from scientific pins')
        checked[str(_bound(native,dict(path=relative,sha256=value)))]=value
    old_path=list(sys.path);old_bytecode=sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode=True
        native_score,_=calibration.load_scorer(native)
        sys.path[:0]=[str(legacy),str(legacy/'vendor/probe_ref')]
        for name,relative in (('analyze','analyze.py'),('order_probe','vendor/probe_ref/order_probe/__init__.py'),('order_probe.common','vendor/probe_ref/order_probe/common.py'),
                ('order_probe.inputs','vendor/probe_ref/order_probe/inputs.py'),
                ('order_probe.measurement','vendor/probe_ref/order_probe/measurement.py')):
            module=importlib.import_module(name)
            require(Path(module.__file__).resolve()==legacy/relative,'Shadowed legacy scoring dependency: '+name)
        spec=importlib.util.spec_from_file_location('_satdose_pinned_score_eval',legacy/'score_eval.py')
        scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)
        require(scorer.native_score is native_score,'Legacy scorer uses another native module')
        # CRN must use the original native helper, not SATDOSE schedule seed_for.
        native_core=importlib.import_module('srf6_ref.core')
        require(Path(native_core.__file__).resolve()==native/'srf6_ref/core.py','Shadowed original eval seed helper')
    finally:
        sys.path[:]=old_path;sys.dont_write_bytecode=old_bytecode
    return scorer,native_core.seed_for,checked


def _endpoint_binding(cfg,frozen,freeze_sha,binding,scope,selected,root):
    require(binding['schema']=='sr-satdose-evaluation-endpoint-binding-v1' and binding['scope']==scope,
            'Wrong endpoint binding schema/scope')
    endpoint=binding['endpoint']
    require(set(endpoint)==ENDPOINT_KEYS and endpoint['experiment_id']==EXPERIMENT and
            _sha(endpoint['policy_sha256']) and _sha(endpoint['reference_sha256']), 'Invalid bound endpoint identity')
    require(type(endpoint['start_seed']) is int and type(endpoint['step']) is int,'Invalid endpoint integer identity')
    require(binding['design_freeze_sha256']==freeze_sha and binding['config_sha256']==frozen['config_sha256'] and
            binding['source_lock_sha256']==frozen['artifacts']['source_lock']['sha256'] and
            binding['manifest_sha256']==frozen['artifacts'][scope.lower()+'_manifest']['sha256'] and
            _sha(binding['execution_sources_sha256']), 'Endpoint scientific/source binding differs')
    require(binding['requests']==len(selected) and type(binding['requests']) is int and
            binding['request_ids_sha256']==core.object_sha([r['request_id'] for r in selected]),
            'Endpoint request allocation/order differs')
    source=cfg['start_bindings'][str(endpoint['start_seed'])]
    require(endpoint['reference_sha256']==source['reference_sha256'],'Original b0 reference differs')
    if endpoint['arm']=='START':
        require(endpoint['step']==0 and endpoint['policy_sha256']==source['policy_sha256'] and binding['checkpoint'] is None,
                'START must be original R24 policy')
    else:
        checkpoint=binding['checkpoint']
        require(set(checkpoint)=={'commit','state'},'Explicit committed checkpoint required')
        marker=core.read_json(_bound(root,checkpoint['commit']));_bound(root,checkpoint['state'])
        identity=marker['identity']
        require(marker['step']==endpoint['step'] and marker['policy_sha256']==endpoint['policy_sha256'] and
                marker['reference_sha256']==endpoint['reference_sha256'] and marker['state_sha256']==checkpoint['state']['sha256'],
                'Checkpoint marker/bytes differ from endpoint')
        expected=dict(experiment_id=EXPERIMENT,parent='b0',train_seed=endpoint['start_seed'],arm=endpoint['arm'],
            run_id=f"{EXPERIMENT}/b0-{endpoint['start_seed']}-{endpoint['arm']}",source_binding=source,
            config_sha256=frozen['config_sha256'],design_freeze_sha256=freeze_sha,
            manifest_sha256=frozen['artifacts']['train_manifest']['sha256'],plan_sha256=frozen['artifacts']['training_groups']['sha256'])
        require(all(identity.get(k)==v for k,v in expected.items()),'Checkpoint scientific identity differs')
        lock=core.read_json(_bound(root,frozen['artifacts']['source_lock']))
        # Runtime backend object_sha uses ensure_ascii=True (native paths are ASCII).
        import json
        native_sha=hashlib.sha256(json.dumps(lock['native_files'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        require(identity.get('source_sha256')==native_sha and _sha(identity.get('execution_sources_sha256')),
                'Checkpoint native/training implementation identity missing')
    return endpoint


def _capture(root,directory,ref,manifest,endpoint,freeze_sha,inp,image):
    path=_bound(root,ref);rid=manifest['request_id'];stem=hashlib.sha256(rid.encode()).hexdigest()
    require(path==directory/(stem+'.COMPLETE.json'),'COMPLETE location/request differs')
    original=core.read_json(path)
    require(original['request_id']==rid and original['seed']==manifest['seed'] and
            original['policy_sha256']==endpoint['policy_sha256'] and original['reference_sha256']==endpoint['reference_sha256'] and
            original['design_freeze_sha256']==freeze_sha,'Original completion identity differs')
    capture=directory/(stem+'.CAPTURE.json');begin=directory/(stem+'.BEGIN.json')
    require(original['native_capture_path']==str(capture),'Original capture relocated')
    captured=core.read_json(_bound(root,dict(path=str(capture.relative_to(root)),sha256=original['native_capture_sha256'])))
    require(begin.resolve(strict=True)==begin and begin.is_file(),'Invalid original BEGIN')
    started=core.read_json(begin)
    require(core.object_sha(started)==core.object_sha(dict(request_id=rid,seed=manifest['seed'],endpoint=endpoint,
            design_freeze_sha256=freeze_sha,status='ATTEMPTED_UNKNOWN')),'Original BEGIN identity differs')
    require(all(key in original and core.object_sha(original[key])==core.object_sha(value) for key,value in captured.items()),
            'COMPLETE changed native CAPTURE fields')
    require(original['generation_status']=='COMPLETE' and original['technical_validation_errors']==[] and
            original['finish_reason'] in ('balanced','eos','length'),'Technical failure cannot be scored as zero')
    tokens=original['tokens'];probs=original['sampler_logprobs']
    require(type(tokens) is list and 1<=len(tokens)<=1536 and all(type(t) is int and t>=0 for t in tokens) and
            core.object_sha(original['raw_tokens'])==core.object_sha(tokens) and
            type(original['completion_token_count']) is int and original['completion_token_count']==len(tokens) and
            type(probs) is list and len(probs)==len(tokens) and all(type(v) in (int,float) and math.isfinite(v) for v in probs) and
            original['sampler_logprob_source']=='actual_generate_raw_logits_selected_token','Invalid original token/logit capture')
    require(isinstance(original.get('prompt_tensor_hash'),str) and original['prompt_tensor_hash'],'Missing original prompt hash')
    public=dict(prompt=inp['prompt'],image_file=inp['image_file'],image_sha256=image['png_sha256'])
    require(original['model_input']==public and original['assistant_prefill']==inp['assistant_prefill'] and
            original['full_text']==inp['assistant_prefill']+original['generated_text'],'Original input/prefix differs')
    require(all(key in captured for key in ('tokens','raw_tokens','sampler_logprobs','prompt_tensor_hash','model_input',
            'seed','assistant_prefill','full_text','generated_text','finish_reason','generation_status','technical_validation_errors')),
            'Incomplete native capture proof')
    return original


def _summary(rows,scope,endpoint):
    def metrics(values):
        n=len(values);require(n>0,'Empty metric denominator')
        keys=('A','S_hit','S_known','B_hit','B_known','json_valid','completed_generation','S_and_A')
        totals={key:sum(row[key] for row in values) for key in keys}
        native=[row['J_native'] for row in values if row['J_native'] is not None]
        return dict(assigned=n,counts=totals,rates={k:v/n for k,v in totals.items()},
            native_J=dict(applicable=len(native),NA=n-len(native),sum=sum(native),mean=sum(native)/len(native) if native else None),
            finish_counts=dict(Counter(row['finish_reason'] for row in values)),
            source_counts=dict(Counter(row['source_class'] for row in values)))
    by=defaultdict(list)
    for row in rows:by[row['panel'],row['view']].append(row)
    out=dict(schema='sr-satdose-evaluation-analysis-v1',status='COMPLETE_ENDPOINT_CPU_SCORED',permission=PERMISSION,
        scope=scope,endpoint=copy.deepcopy(endpoint),metrics=metrics(rows),
        panel_views=[dict(panel=panel,view=view,**metrics(values)) for (panel,view),values in sorted(by.items())],
        main_terminal_authority_verified=False,gpu_acceptance=False,training_admission=False)
    prefix=[row for row in rows if row['panel']=='prefix10']
    if prefix:
        out['prefix_sensitivity']=[]
        for view in ('p_stable','p_revtie','p_placebo'):
            values=[row for row in prefix if row['view']==view]
            items=[row['eos_single_brace_sensitivity'] for row in values]
            out['prefix_sensitivity'].append(dict(view=view,assigned=len(values),eligible=sum(x['eligible'] for x in items),
                counts={key:sum(x['metrics'][key] for x in items) for key in ('A','S_hit','S_known','json_valid')},
                B_known=sum(x['metrics']['B'] is not None for x in items),B_hit=sum(x['metrics']['B']==1 for x in items),
                role='SENSITIVITY_ONLY_NEVER_REPLACE_STRICT'))
    if scope=='DEV':
        out['dev_risk_inputs']=dict(endpoint=copy.deepcopy(endpoint),
            conflict=metrics(by['list8','conflict']),
            aligned_no_list=metrics(by['list8','aligned']+by['list8','no_list']),
            rule='INPUTS_ONLY_COMPARE_TWO_SCHEDULED_CHECKPOINTS_AND_TRAINING_WINDOWS_ELSEWHERE',
            hold_decision=None)
    return out


def analyze_completed_endpoint(config,freeze,evidence_root,receipt_path,*,expected_receipt_sha256,
                               endpoint_binding_path,expected_endpoint_binding_sha256):
    """Return {rows,summary}; validate completed evidence BEFORE opening raw."""
    root=Path(evidence_root).resolve(strict=True)
    def trusted(path,sha):return _bound(root,dict(path=str(Path(path).relative_to(root)),sha256=sha))
    config,freeze=Path(config),Path(freeze)
    cfg=core.read_json(config);frozen=core.read_json(freeze);freeze_sha=core.file_sha(freeze)
    require(cfg['experiment_id']==frozen['experiment_id']==EXPERIMENT and cfg['parent']=='b0' and
            frozen['schema']=='sr-satdose-design-freeze-v1' and frozen['config_sha256']==core.file_sha(config),
            'Wrong scientific config/freeze')
    require(freeze.resolve(strict=True)==freeze and freeze.is_relative_to(root),'Noncanonical freeze')
    receipt_path=trusted(receipt_path,expected_receipt_sha256)
    binding_path=trusted(endpoint_binding_path,expected_endpoint_binding_sha256)
    receipt=core.read_json(receipt_path);binding=core.read_json(binding_path);scope=binding['scope']
    require(receipt['schema']=='sr-satdose-evaluation-completion-v1' and
            receipt['status']=='EVALUATION_ENDPOINT_CAPTURE_COMPLETE_NO_TRAINING_PERMISSION' and
            receipt['endpoint_binding_sha256']==expected_endpoint_binding_sha256 and receipt['scope']==scope,
            'Incomplete/foreign evaluation receipt')
    directory=receipt_path.parent
    require(not any(p.name in ('HOLD.json','ERROR.json') or p.name.endswith('.ERROR.json') for p in directory.rglob('*')),
            'HOLD/ERROR evaluation cannot be scored')
    require(receipt['optimizer_created'] is False and receipt['scheduler_created'] is False and
            all(type(receipt.get(k)) is int and receipt[k]==0 for k in
                ('optimizer_steps','scheduler_steps','parameter_updates','extra_forward_sequences')), 'Evaluation performed training/replay work')
    artifacts={name:_bound(root,frozen['artifacts'][name]) for name in
               ('private_cases','inputs','images','source_lock',scope.lower()+'_manifest')}
    manifests=list(core.read_jsonl(artifacts[scope.lower()+'_manifest']))
    groups=validate_manifest(manifests,scope)
    ep=binding['endpoint'];key=ep['start_seed'],ep['arm'],ep['step']
    require(key in groups,'Endpoint absent from frozen manifest')
    selected=groups[key];endpoint=_endpoint_binding(cfg,frozen,freeze_sha,binding,scope,selected,root)
    require(core.object_sha(receipt['endpoint_before'])==core.object_sha(receipt['endpoint_after'])==core.object_sha(endpoint) and
            type(receipt['responses']) is int and receipt['responses']==len(selected),'Wrong completed endpoint/count')
    cases=_index(core.read_jsonl(artifacts['private_cases']),'case_id')
    inputs=_index(core.read_jsonl(artifacts['inputs']),'input_id');images=_index(core.read_jsonl(artifacts['images']),'image_file')
    refs=receipt['complete_files']
    require(type(refs) is list and len(refs)==len(selected),'Incomplete COMPLETE inventory')
    originals=[];seen_files=set()
    for manifest,ref in zip(selected,refs,strict=True):
        inp=inputs[manifest['input_id']];case=cases[manifest['case_id']];im=images[inp['image_file']]
        require(set(inp)=={'input_id','prompt','image_file','assistant_prefill'} and
                inp['input_id']==core.object_sha({k:v for k,v in inp.items() if k!='input_id'}) and
                case['case_id']==core.object_sha({k:case[k] for k in ('input_id','task','audit')}), 'Private/public content identity differs')
        require(case['input_id']==inp['input_id'] and case['task']['root_id']==manifest['root_id'] and
                case['task']['structure']==manifest['structure'],'Case/root/structure identity differs')
        _bound(root,dict(path=inp['image_file'],sha256=im['png_sha256']))
        originals.append(_capture(root,directory,ref,manifest,endpoint,freeze_sha,inp,im))
        stem=hashlib.sha256(manifest['request_id'].encode()).hexdigest()
        seen_files.update(stem+suffix for suffix in ('.BEGIN.json','.CAPTURE.json','.COMPLETE.json'))
    actual={p.name for p in directory.iterdir() if any(p.name.endswith(suffix) for suffix in ('.BEGIN.json','.CAPTURE.json','.COMPLETE.json'))}
    require(actual==seen_files,'Extra/unresolved evaluation attempt')
    # Completed population and immutable captures are now established; open raw.
    raw_path=_bound(root,receipt['raw']);raw=list(core.read_jsonl(raw_path))
    require([row.get('request_id') for row in raw]==[m['request_id'] for m in selected], 'Raw coverage/order differs')
    for manifest,row,original in zip(selected,raw,originals,strict=True):
        require(all(key in row and core.object_sha(row[key])==core.object_sha(value) for key,value in original.items()),
                'Raw changed original COMPLETE fields')
        require(all(type(row.get(key)) is type(value) and row[key]==value for key,value in manifest.items() if key!='split') and
                row['scope']==scope and row.get('split',scope)==scope,'Raw manifest identity differs')
        require(row['suffix_text']==row['generated_text'],'Raw suffix differs from native generation')
    events=list(core.read_jsonl(_bound(root,receipt['accounting'])))
    require(len(events)==2*len(selected),'Evaluation accounting population differs')
    for index,(manifest,row) in enumerate(zip(selected,raw,strict=True)):
        for offset,kind,count in ((0,'completion_attempts',1),(1,'generated_tokens',len(row['tokens']))):
            event=events[2*index+offset]
            require(event['kind']==kind and type(event['count']) is int and event['count']==count and
                    event['metadata']['group_seed']==manifest['seed'],'Native evaluation accounting differs')
    scorer,seed_for,source_pins=load_scorers(artifacts['source_lock'],frozen['artifacts']['source_lock']['sha256'])
    scores=[]
    for manifest,row,complete_ref in zip(selected,raw,refs,strict=True):
        require(manifest['seed']==seed_for(2026110301,'eval','b0',manifest['panel'],manifest['root_id'],manifest['draw']),
                'Frozen request differs from original native evaluation CRN')
        case=cases[manifest['case_id']];task,audit=case['task'],case['audit']
        require(type(audit.get('native_J_applicable')) is bool,'Missing native J applicability')
        expected_app=not(manifest['panel']=='prefix10' or (manifest['panel']=='authority8' and manifest['view']=='list'))
        require(audit['native_J_applicable'] is expected_app,'Panel/native J applicability differs')
        value=scorer.score_one(manifest,row,case,inputs[manifest['input_id']])
        value['native_score']=(scorer.native_score(row['full_text'],task['world'],task['query'],
            finished=row['finish_reason'] in ('balanced','eos'),finish_reason=row['finish_reason']) if expected_app else None)
        value.update(native_J_applicable=expected_app,policy_sha256=endpoint['policy_sha256'],
            reference_sha256=endpoint['reference_sha256'],design_freeze_sha256=freeze_sha,
            original_complete=copy.deepcopy(complete_ref),native_capture_path=row['native_capture_path'],
            native_capture_sha256=row['native_capture_sha256'])
        value['S_and_A']=value['S_hit']*value['A']
        _,selection=core.parse_selected(row['full_text'],row['finish_reason'])
        value['source_class']=(core.source_class(selection,core.geometry(task,audit,manifest['panel']))
                              if manifest['panel']!='retention8' else 'not_applicable')
        value['strict_metrics']={k:value[k] for k in ('A','S_hit','S_known','B_hit','B_known','json_valid')}
        value['eos_single_brace_sensitivity']=None
        if manifest['panel']=='prefix10':
            value['eos_single_brace_sensitivity']=dict(eligible=bool(value['prefix_brace_eligible']),
                metrics={k:value['brace_'+k] for k in scorer.PREFIX_KEYS},
                role='SENSITIVITY_ONLY_NEVER_REPLACE_STRICT')
        scores.append(value)
    for path,sha in source_pins.items():require(core.file_sha(path)==sha,'Scorer source changed during analysis')
    trusted(receipt_path,expected_receipt_sha256);trusted(binding_path,expected_endpoint_binding_sha256)
    for name,path in artifacts.items():require(core.file_sha(path)==frozen['artifacts'][name]['sha256'],'Frozen scoring artifact changed')
    require(core.file_sha(freeze)==freeze_sha and core.file_sha(config)==frozen['config_sha256'],'Config/freeze changed during scoring')
    _bound(root,receipt['raw']);_bound(root,receipt['accounting'])
    for manifest,ref,original in zip(selected,refs,originals,strict=True):
        inp=inputs[manifest['input_id']];im=images[inp['image_file']]
        require(_capture(root,directory,ref,manifest,endpoint,freeze_sha,inp,im)==original,'Capture changed during scoring')
    for filename in {inputs[m['input_id']]['image_file'] for m in selected}:
        _bound(root,dict(path=filename,sha256=images[filename]['png_sha256']))
    _endpoint_binding(cfg,frozen,freeze_sha,binding,scope,selected,root)
    summary=_summary(scores,scope,endpoint)
    summary.update(evidence=dict(receipt_sha256=expected_receipt_sha256,endpoint_binding_sha256=expected_endpoint_binding_sha256,
        design_freeze_sha256=freeze_sha,source_lock_sha256=frozen['artifacts']['source_lock']['sha256'],
        manifest_sha256=frozen['artifacts'][scope.lower()+'_manifest']['sha256'],request_ids_sha256=binding['request_ids_sha256'],
        raw_sha256=receipt['raw']['sha256']),scorer_sources=source_pins,
        analysis_sources={str(Path(__file__).resolve()):core.file_sha(__file__),str(Path(core.__file__).resolve()):core.file_sha(core.__file__),
            str(Path(calibration.__file__).resolve()):core.file_sha(calibration.__file__),
            str(calibration.PACKAGE/'config/source_bindings.json'):core.file_sha(calibration.PACKAGE/'config/source_bindings.json')})
    return dict(rows=scores,summary=summary)
