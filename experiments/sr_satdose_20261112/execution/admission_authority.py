"""Source-pinned amended admission for the explicit PREFIX exception only.

No CLI, writes, generation, CUDA or optimizer calls. ``prepare`` replays native
DRY scoring and the full 96-row PREFIX verifier, checks the separately trusted
GPU audit/report, and returns an admission *candidate* for a control plane to
publish once. ``FirstTwoStepAdmission`` requires that exact published receipt.
It rehashes protected artifacts at every step boundary; there is no disk-cache
or mtime shortcut. Training's transaction layer still validates the live COMMIT
chain. No step3, canary, DEV, MAIN or terminal authority is supplied here.

The caller must independently pin this source and the trusted spec; a spec
read from an arbitrary path or hashes computed from unreviewed current files
are not authority. All source refs are absolute; all document refs root-relative.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import math
from pathlib import Path
import re
import sys

from . import training_step

tx = training_step.tx
require = tx.require
EXPERIMENT = tx.EXPERIMENT
PREFIX = 'prefix_calibration_pass'
TECHNICAL = (
    'start_and_reference_hash_verified','native_scorer_golden_pass',
    'disjoint_splits_checked','gold_never_in_model_inputs',
    'native_loss_replay_equal','production_gradient_hook_verified',
    'sampler_teacher_gate','raw_token_identity','same_runtime_backend',
    'long_sequence_fullbatch_pass','allzero_pg_zero_kl_retained',
    'shadow_adam_update_replay_pass','evaluation_loader_policy_verified',
    'dry_unchanged_policy',
)
DOCUMENTS = {'config','registration','gpu_report','independent_audit','binding','waiver'}
SOURCES = {'authority','check_admission','satdose_core','scorer_bindings',
           'prefix_waiver','calibration','gpu_auditor'}
GRADIENT_FIELDS = ('J_vs_source_pg_cosine','J_split_half_cosine','source_split_half_cosine',
                   'heldout_effect_half_1','heldout_effect_half_2','effect_repeat_error_floor')


def _reference(reference):
    require(isinstance(reference,dict) and set(reference)=={'path','sha256'} and
            isinstance(reference['path'],str) and isinstance(reference['sha256'],str) and
            re.fullmatch('[0-9a-f]{64}',reference['sha256']), 'Malformed trusted reference')


def _absolute(reference):
    _reference(reference);path=Path(reference['path'])
    require(path.is_absolute() and path.resolve(strict=True)==path and path.is_file() and
            not path.is_symlink() and tx.file_sha(path)==reference['sha256'],
            'Trusted source/artifact changed: '+str(path))
    return path


class _Observed:
    def __init__(self,root):self.root=root;self.files={};self.directories={};self.model_links={};self.model_directories={}
    def absolute(self,path,expected=None):
        path=Path(path)
        require(path.is_absolute() and path.resolve(strict=True)==path and path.is_file() and
                not path.is_symlink(),'Noncanonical observed artifact')
        digest=tx.file_sha(path)
        require(expected is None or digest==expected,'Observed artifact SHA differs: '+str(path))
        require(path not in self.files or self.files[path]==digest,'Artifact changed during verification')
        self.files[path]=digest;return path
    def ref(self,reference):
        _reference(reference)
        return self.absolute(tx.bound(self.root,reference),reference['sha256'])
    def json(self,reference):return tx.read_json(self.ref(reference))
    def tree(self,path):
        path=Path(path)
        require(path.resolve(strict=True)==path and path.is_dir() and not path.is_symlink(),
                'Noncanonical protected directory')
        entries={str(p.relative_to(path)) for p in path.rglob('*')}
        for name in entries:
            p=path/name
            require(not p.is_symlink(),'Symlink in protected evidence')
            if p.is_file():self.absolute(p)
            else:require(p.is_dir(),'Nonordinary protected entry')
        self.directories[path]=entries
    def model_source_lock(self,lock):
        """Accept only the frozen HF snapshot's direct links into its own blobs.

        General evidence/source paths remain strictly canonical. Both original
        link text/resolution and actual target bytes are checked at boundaries.
        """
        root=Path(lock['model_path'])
        require(root.is_absolute() and root.resolve(strict=True)==root and root.is_dir() and
                root.name==lock['snapshot'] and root.parent.name=='snapshots','Noncanonical model snapshot')
        expected=set(lock['model_files']);entries={str(p.relative_to(root)) for p in root.rglob('*')}
        require(expected and {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}==expected,
                'Model snapshot inventory differs')
        for relative in entries-expected:
            directory=root/relative
            require(directory.is_dir() and directory.resolve(strict=True)==directory and not directory.is_symlink(),
                    'Nonordinary model snapshot directory')
        self.model_directories[root]=entries
        blobs=root.parent.parent/'blobs'
        for relative,digest in lock['model_files'].items():
            rel=Path(relative)
            require(not rel.is_absolute() and '..' not in rel.parts,'Model source escapes snapshot')
            path=root/rel
            require(path.parent.resolve(strict=True)==path.parent,'Model ancestor symlink')
            if path.is_symlink():
                link=path.readlink();target=path.resolve(strict=True)
                require(blobs.resolve(strict=True)==blobs and target.parent==blobs and
                        re.fullmatch(r'(?:[0-9a-f]{40}|[0-9a-f]{64})',target.name),'Model link escapes pinned repository blobs')
                self.absolute(target,digest)
                require(path.readlink()==link and path.resolve(strict=True)==target,'Model link changed while hashing')
                self.model_links[path]=(link,target)
            else:self.absolute(path,digest)
    def stable(self):
        for path,digest in self.files.items():self.absolute(path,digest)
        for path,entries in self.directories.items():
            require(path.resolve(strict=True)==path and
                    {str(p.relative_to(path)) for p in path.rglob('*')}==entries,
                    'Protected evidence directory changed: '+str(path))
        for path,(link,target) in self.model_links.items():
            require(path.is_symlink() and path.parent.resolve(strict=True)==path.parent and
                    path.readlink()==link and path.resolve(strict=True)==target,'Frozen model link changed: '+str(path))
        for path,entries in self.model_directories.items():
            require(path.resolve(strict=True)==path and
                    {str(p.relative_to(path)) for p in path.rglob('*')}==entries,'Model snapshot inventory changed')


def _load(reference,name):
    path=_absolute(reference)
    spec=importlib.util.spec_from_file_location(name,path)
    require(spec is not None and spec.loader is not None,'Cannot load pinned verifier')
    module=importlib.util.module_from_spec(spec)
    # Execute the already authenticated bytes, never a stale bytecode cache.
    source=path.read_bytes()
    require(hashlib.sha256(source).hexdigest()==reference['sha256'] and
            tx.file_sha(path)==reference['sha256'],'Verifier changed during read')
    exec(compile(source,str(path),'exec'),module.__dict__)
    return module


def _sources(spec,observed):
    sources=spec['sources'];require(set(sources)==SOURCES,'Incomplete verifier source pins')
    for reference in sources.values():observed.absolute(_absolute(reference),reference['sha256'])
    require(Path(sources['authority']['path'])==Path(__file__).resolve(), 'Foreign authority source')
    actual={'satdose_core':training_step.calibration.core.__file__,
            'calibration':training_step.calibration.__file__,
            'scorer_bindings':training_step.calibration.PACKAGE/'config/source_bindings.json'}
    for name,path in actual.items():
        require(Path(sources[name]['path'])==Path(path).resolve(),'Shadowed admission dependency: '+name)
    original=_load(sources['check_admission'],'_satdose_original_admission')
    require(tuple(original.REQUIRED_TECH)==TECHNICAL+(PREFIX,), 'Frozen gate inventory changed')
    waiver=_load(sources['prefix_waiver'],'_satdose_prefix_admission_verifier')
    require(waiver.calibration is training_step.calibration,'Shadowed PREFIX calibration import')
    return original,waiver


def verify_gpu_report(gpu,cfg,config_ref,registration_ref,observed,audit,audit_ref,waiver_ref):
    """Original fourteen checks/gradient bounds, without manufacturing PREFIX pass.

    Evidence references are externally trusted through the GPU report SHA.
    This validator does not turn arbitrary evidence into a report or infer
    missing boolean results. The independent audit must be separately pinned.
    """
    require(gpu.get('schema')=='sr-satdose-gpu-verification-v1' and gpu.get('mode')=='production' and
            gpu.get('experiment_id')==cfg['experiment_id']==EXPERIMENT,'Wrong GPU report identity')
    require(gpu.get('config_sha256')==config_ref['sha256'] and
            gpu.get('registration_sha256')==registration_ref['sha256'],'GPU config/registration differs')
    require(set(gpu.get('checks',{}))==set(TECHNICAL)|{PREFIX},'GPU gate inventory differs')
    require(isinstance(gpu.get('observed_evidence'),dict) and gpu['observed_evidence'],
            'GPU report missing supplementary observed evidence')
    for name,digest in gpu['observed_evidence'].items():
        path=Path(name)
        observed.absolute(path if path.is_absolute() else observed.root/path,digest)
    for key in TECHNICAL:
        item=gpu['checks'][key]
        require(item.get('passed') is True,'GPU/technical gate not passed: '+key)
        path=observed.ref(item['evidence'])
        require(gpu['observed_evidence'].get(str(path),
                gpu['observed_evidence'].get(item['evidence']['path']))==item['evidence']['sha256'],
                'Technical proof absent from report observed evidence')
    prefix=gpu['checks'][PREFIX]
    require(prefix.get('passed') is False,'Original PREFIX measurement false must remain false')
    observed.ref(prefix['evidence'])
    require(gpu.get('prefix_gate_decision')=='PASS_USER_OVERRIDE' and
            gpu.get('prefix_waiver')==waiver_ref and gpu.get('independent_audit')==audit_ref,
            'Missing exact scoped PREFIX authority/independent audit')
    require(type(gpu.get('dry_optimizer_steps')) is int and gpu['dry_optimizer_steps']==0 and
            type(gpu.get('dry_parameter_updates')) is int and gpu['dry_parameter_updates']==0 and
            type(gpu.get('shadow_optimizer_steps')) is int and 0<gpu['shadow_optimizer_steps']<=4,
            'DRY update or shadow budget differs')
    require(gpu['shadow_optimizer_steps']==audit.get('shadow_optimizer_steps')==4,
            'GPU report differs from four actual audited shadow updates')
    require(set(gpu.get('gradient',{}))=={'11031','11032'},'Gradient starts differ')
    for start in ('11031','11032'):
        g=gpu['gradient'][start];direction=audit['starts'][start]['direction']
        require(g.get('policy_before_sha256')==g.get('policy_after_sha256')==
                cfg['start_bindings'][start]['policy_sha256'],'Probe changed starting policy')
        observed.ref(g['evidence'])
        require(all(type(g.get(k)) in (int,float) and math.isfinite(g[k]) for k in GRADIENT_FIELDS),
                'Invalid gradient diagnostics')
        require(all(-1<=g[k]<=1 and g[k]>0 for k in GRADIENT_FIELDS[:3]),
                'HOLD_DIRECTION_UNCERTAIN: cosine not positive')
        floor=g['effect_repeat_error_floor']
        require(floor>=0 and g['heldout_effect_half_1']>floor and g['heldout_effect_half_2']>floor,
                'HOLD_DIRECTION_UNCERTAIN: heldout effects do not exceed repeat floor')
        require(all(g[k]==direction[k] for k in GRADIENT_FIELDS),
                'GPU gradient scalars differ from independently audited actual tensors')


def _verified_prefix(root,binding,waiver_ref,waiver_module):
    """Invoke the pinned full96 verifier, never accept a gate-label callback."""
    waiver=waiver_module.verify_completed(root,binding)
    require(waiver.get('gate_decision')=='PASS_USER_OVERRIDE' and waiver.get('responses')==96 and
            waiver.get('resampled_responses')==0 and waiver.get('original_measurements_preserved') is True and
            waiver.get('other_gates_unchanged') is True and waiver.get('waiver')==waiver_ref and
            waiver.get('candidate_gate_passed',{}).get('11031') is False,
            'PREFIX waiver does not preserve exact authorized failed measurement')
    return waiver


def _gpu_slots(root):
    """Check live occupancy/HOLD state, not only the earlier audit's snapshot."""
    require(not any((root/name).exists() for name in
            ('HOLD.json','ERROR.json','training_control/HOLD.json','training_control/ERROR.json')),
            'Global HOLD/error exists before admission')
    for name in ('evaluation_loader_checks','backend_equivalence','allzero_check','shadow_inputs','shadow_claims'):
        base=root/name
        require(base.resolve(strict=True)==base and base.is_dir() and not base.is_symlink() and
                {p.name for p in base.iterdir()}=={'start-11031','start-11032'},
                'Missing/extra GPU start slot: '+name)
        for start in (11031,11032):
            path=base/f'start-{start}'
            require(path.is_dir() and path.resolve(strict=True)==path,'Noncanonical GPU start slot')
            if name.startswith('shadow_'):
                require({p.name for p in path.iterdir()}=={'trial-0','trial-1'},'Unknown shadow attempt/budget')
                for trial in (0,1):
                    slot=path/f'trial-{trial}'
                    require(slot.is_dir() and slot.resolve(strict=True)==slot,'Noncanonical shadow slot')
        for path in base.rglob('*'):
            require(not path.is_symlink(),'Symlink in GPU evidence')
            require(path.name not in ('HOLD.json','ERROR.json') and
                    not path.name.endswith(('.HOLD.json','.ERROR.json')),
                    'Existing GPU technical HOLD/error')


def _validate(spec):
    """Full CPU validation. Input spec is externally pinned, never auto-discovered."""
    spec=copy.deepcopy(spec)
    require(set(spec)=={'root','documents','sources','training_execution_sources_sha256'},
            'Unexpected/missing trusted-spec fields')
    root=Path(spec['root'])
    require(root.is_absolute() and root.resolve(strict=True)==root and root.is_dir(),'Noncanonical trusted root')
    require(re.fullmatch('[0-9a-f]{64}',spec['training_execution_sources_sha256'] or ''),
            'Missing trusted training execution inventory')
    refs=spec['documents'];require(set(refs)==DOCUMENTS,'Unexpected/missing trusted documents')
    e=_Observed(root);original,waiver_module=_sources(spec,e)
    docs={key:e.json(ref) for key,ref in refs.items()}
    cfg,binding,audit=docs['config'],docs['binding'],docs['independent_audit']
    require(binding['config']==refs['config'] and binding['prefix_waiver']==refs['waiver'],
            'Binding/config/PREFIX authority differs')
    waiver_document=docs['waiver']
    require(waiver_document.get('scope')=='PREFIX_CALIB_MEASUREMENT_GATE_ONLY' and
            waiver_document.get('other_gates_unchanged') is True and
            waiver_document.get('technical_failure_or_unknown_waived') is False and
            waiver_document.get('rewrite_measurement_pass') is False and
            waiver_document.get('preserve_original_results') is True,
            'PREFIX exception cannot waive technical failures or rewrite measurement')
    freeze=e.json(binding['design_freeze'])
    require(binding['design_freeze']['path']=='DESIGN_FREEZE.json' and
            freeze['config_sha256']==refs['config']['sha256'],'Frozen design differs')
    require(audit.get('schema')=='sr-satdose-independent-gpu-evidence-audit-v1' and
            audit.get('status')=='FINITE_GPU_EVIDENCE_AUDITED_NOT_ADMISSION' and
            audit.get('permission')=='NO_TRAINING' and
            audit.get('technical_binding_sha256')==refs['binding']['sha256'] and
            audit.get('auditor_sha256')==spec['sources']['gpu_auditor']['sha256'] and
            audit.get('completed_fixed_stages')==10 and audit.get('shadow_optimizer_steps')==4 and
            audit.get('formal_optimizer_steps')==audit.get('audit_model_calls')==audit.get('audit_optimizer_calls')==0 and
            set(audit.get('starts',{}))=={'11031','11032'},'Incomplete/untrusted independent GPU audit')
    require(audit.get('evidence'),'GPU audit missing observed evidence')
    _gpu_slots(root)
    for name,digest in audit['evidence'].items():
        path=Path(name);e.absolute(path if path.is_absolute() else root/path,digest)
    for start in ('11031','11032'):
        d=audit['starts'][start]
        require(d['production_rows']==128 and d['shadow_optimizer_steps']==2 and
                d['direction']['direction_checks_satisfied'] is True and not d['direction']['hold_reasons'],
                'Incomplete actual start evidence/direction HOLD')
    # Capture immutable inputs before expensive verification; rehash afterwards.
    for ref in freeze['artifacts'].values():e.ref(ref)
    for ref in docs['registration']['artifacts'].values():e.ref(ref)
    for name in ('prefix_control','prefix_waiver_control','calibration_allocations','protocol_clarifications',
                 'evaluation_loader_checks','backend_equivalence','allzero_check','shadow_inputs','shadow_claims'):
        e.tree(root/name)
    # Also freeze the run-directory inventory: an added retry/unknown PREFIX
    # directory must not become invisible to the cached step-boundary guard.
    e.tree(root/'runs')
    for scope in ('DRY','PREFIX_CALIB'):
        for start in (11031,11032):
            require((root/f'runs/{scope}_{start}').is_dir(),'Missing fixed calibration run')
    continuation=e.json(binding['prefix_continuation'])
    for reference in (binding['gpu_release'],waiver_document['original_release'],continuation['release']):
        release=e.json(reference);e.tree(Path(release['release_root']))
    lock=e.json(binding['source_lock'])
    for directory,files in [('native_src','native_files'),('legacy_root','legacy_files'),('parent_path','parent_files')]:
        for relative,digest in lock[files].items():e.absolute(Path(lock[directory])/relative,digest)
    e.model_source_lock(lock)
    # The helper imports its frozen native scorer and recomputes all 1,280 rows.
    dry=original.run(e.ref(refs['config']),e.ref(refs['registration']),root,Path(lock['native_src']))
    require(dry.get('status')=='DRY_METRICS_PASS_GPU_CHECK_STILL_REQUIRED' and dry.get('failures')==[],
            'Actual DRY signal failed')
    _verified_prefix(root,binding,refs['waiver'],waiver_module)
    require(docs['gpu_report']['checks'][PREFIX]['evidence']==binding['prefix_result'],
            'PREFIX report must reference original failed result')
    verify_gpu_report(docs['gpu_report'],cfg,refs['config'],refs['registration'],e,audit,
                      refs['independent_audit'],refs['waiver'])
    candidate=copy.deepcopy(dry)
    candidate.update(status='ADMITTED_FIRST_TWO_STEPS_ONLY',permission='FOUR_PATHS_MAX_STEP_2',
        gpu_report_sha256=refs['gpu_report']['sha256'],prefix_gate_decision='PASS_USER_OVERRIDE',
        prefix_measurement_passed=False,prefix_waiver=copy.deepcopy(refs['waiver']),
        independent_audit=copy.deepcopy(refs['independent_audit']),
        technical_binding_sha256=refs['binding']['sha256'],authority_source=copy.deepcopy(spec['sources']['authority']))
    e.stable()
    return candidate,e,cfg,binding,freeze


def prepare(trusted_spec):
    """Return a verified candidate; caller owns publication, no file is written."""
    candidate,_,_,_,_=_validate(trusted_spec)
    return candidate


class FirstTwoStepAdmission(training_step.AuthenticatedAdmission):
    def __init__(self,trusted_spec,*,admission):
        self._spec=copy.deepcopy(trusted_spec)
        super().__init__(copy.deepcopy(trusted_spec['sources']['authority']))
        candidate,self._observed,self._config,self._binding,self._freeze=_validate(self._spec)
        actual=self._observed.json(admission)
        require(actual==candidate,'Published amended admission differs from independently recomputed candidate')
        self._admission=copy.deepcopy(admission)
        self._observed.stable()

    def verify(self,*,root,identity,design_freeze_sha256,target_step,sampled_endpoint):
        training_step.AuthenticatedAdmission.check_source(self)
        require(type(target_step) is int and target_step in (1,2),
                'Only first two TRAIN steps authorized; no step3/DEV/MAIN authority')
        require(Path(root)==self._observed.root and design_freeze_sha256==self._binding['design_freeze']['sha256'],
                'Foreign root/frozen design')
        require(not any((self._observed.root/name).exists() for name in
                ('HOLD.json','ERROR.json','training_control/HOLD.json','training_control/ERROR.json')),
                'Global training HOLD/error exists')
        self._observed.stable()
        start=identity.get('train_seed');arm=identity.get('arm')
        require(type(start) is int and start in (11031,11032) and arm in ('T0','T1') and
                identity.get('experiment_id')==EXPERIMENT and identity.get('parent')=='b0' and
                identity.get('run_id')==f'{EXPERIMENT}/b0-{start}-{arm}' and
                identity.get('config_sha256')==self._spec['documents']['config']['sha256'] and
                identity.get('design_freeze_sha256')==design_freeze_sha256 and
                identity.get('execution_sources_sha256')==self._spec['training_execution_sources_sha256'] and
                identity.get('source_binding')==self._config['start_bindings'][str(start)] and
                identity.get('manifest_sha256')==self._freeze['artifacts']['train_manifest']['sha256'] and
                identity.get('plan_sha256')==self._freeze['artifacts']['training_groups']['sha256'],
                'Unregistered training path/source identity')
        endpoint=dict(experiment_id=EXPERIMENT,start_seed=start,arm=arm,step=target_step-1,
                      reference_sha256=identity['source_binding']['reference_sha256'])
        require(set(sampled_endpoint)==set(endpoint)|{'policy_sha256'} and
                all(sampled_endpoint[k]==v for k,v in endpoint.items()) and
                re.fullmatch('[0-9a-f]{64}',sampled_endpoint['policy_sha256'] or ''), 'Invalid sampled TRAIN endpoint')
        if target_step==1:
            require(sampled_endpoint['policy_sha256']==identity['source_binding']['policy_sha256'],
                    'Step1 must sample original R24')
        else:
            directory=self._observed.root/f'training_transactions/start-{start}/{arm}/step-001'
            require(directory.resolve(strict=True)==directory,'Noncanonical prior committed step')
            marker=tx.read_json(directory/'COMMIT.json')
            require(marker.get('identity')==identity and marker.get('step')==1 and
                    marker.get('policy_sha256')==sampled_endpoint['policy_sha256'] and
                    marker.get('reference_sha256')==endpoint['reference_sha256'] and
                    tx.file_sha(directory/'state.pt')==marker.get('state_sha256'),
                    'Step2 must sample actual committed step1 endpoint')
            # Full raw/update/optimizer chain is additionally checked by run_step's
            # TransactionStore under its global lock, not replaced by this guard.
        refs=self._spec['documents']
        return dict(identity_sha256=training_step.object_sha(identity),design_freeze_sha256=design_freeze_sha256,
                    target_step=target_step,sampled_endpoint=copy.deepcopy(sampled_endpoint),
                    admission=copy.deepcopy(self._admission),registration=copy.deepcopy(refs['registration']),
                    gpu_report=copy.deepcopy(refs['gpu_report']))
