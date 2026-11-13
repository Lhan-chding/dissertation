"""Synthetic four-path captures/controller/GPU witnesses; real CPU496 Adam/state.

No GPU, model forward, process launch or actual training admission. Tests
manufacture clearly synthetic producer records only to exercise audit rejection.
"""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
from execution import canary_audit as a
sys.path.pop(0)
spec=importlib.util.spec_from_file_location('_canary_audit_transaction_fixture',Path(__file__).with_name('test_training_transaction.py'))
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
t=a.tx


def dump(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(t.canonical(value)+b'\n')
def lines(path,values):path.write_bytes(b''.join(t.canonical(v)+b'\n' for v in values))
def process(pid):return dict(host='CPU_SYNTHETIC',boot_id='CPU_SYNTHETIC_BOOT',pid=pid,start_ticks=pid*10)


@unittest.skipUnless((f.fixtures.NATIVE/'sr_f3_ce/training.py').exists(),'Pinned CPU native optimizer required')
class CanaryAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(f.TrainingTransactionTests.doClassCleanups);f.TrainingTransactionTests.setUpClass()

    def setUp(self):
        case=f.TrainingTransactionTests('test_real_adam_two_steps_resume_and_commit_idempotency')
        case.setUp();self.addCleanup(case.doCleanups);self.case=case;self.tiny=case.fixture
        self.root=case.root;self.native=self.tiny.ops;self.auditor_sha=t.file_sha(a.__file__)
        self.sources={name:dict(path=str(Path(module).resolve()),sha256=t.file_sha(module)) for name,module in
            [('training_runtime',a.runtime.__file__),('training_state',a.state.__file__),('training_transaction',t.__file__),
             ('canary_replay',a.replay.__file__),('frozen_fingerprint',a.__file__),('canary_controller',__file__)]}
        config=t.read_json(self.root/'config.json')
        for start in (11031,11032):
            source=copy.deepcopy(self.tiny.source);path=self.root/f'R24-{start}.pt'
            torch.save(dict(schema=a.fork_loader.LEGACY_SCHEMA,step=24,identity=dict(parent='b0',train_seed=start,arm='R'),
                reference_sha256=source['reference_sha256'],policy=self.native.trainable_state(self.tiny.runtime.model)),path)
            source.update(original_state_path=str(path),state_sha256=t.file_sha(path));config['start_bindings'][str(start)]=source
        dump(self.root/'config.json',config)
        native=f.fixtures.NATIVE.resolve()
        dump(self.root/'source_lock.json',dict(native_src=str(native),native_files={str(p.relative_to(native)):t.file_sha(p)
            for p in native.rglob('*.py') if p.is_file()}))
        freeze=t.read_json(case.freeze);freeze['config_sha256']=t.file_sha(self.root/'config.json')
        for key in ('config','source_lock'):freeze['artifacts'][key]=t.reference(self.root,self.root/(key+'.json'))
        runtime_dir=Path(a.fork_loader.__file__).parent
        freeze['runtime_sources']={p.name:t.file_sha(p) for p in runtime_dir.glob('*.py')}
        dump(case.freeze,freeze);case.freeze_sha=t.file_sha(case.freeze)
        self.freeze_sha=case.freeze_sha;self.trusted={};self.frozen={};self.exits={};self.contexts={}
        for index,key in enumerate(a.PATHS):self.make_path(index,key,config)

    def ref(self,path):return t.reference(self.root,path)

    def make_path(self,index,key,config):
        start,arm=key.split('/');start=int(start)
        runtime,opt,scheduler=self.tiny.make_runtime();reference=copy.deepcopy(self.tiny.reference)
        source=config['start_bindings'][str(start)]
        base=dict(self.tiny.base,train_seed=start,arm=arm,run_id=f'{a.state.EXPERIMENT}/b0-{start}-{arm}',
            execution_sources_sha256=a.object_sha(self.sources))
        identity=a.state.bind_identity(base,runtime.model,opt,source_binding=source)
        runtime.identity.update(train_seed=start,arm=arm,run_id=identity['run_id'])
        self.tiny.runtime,self.tiny.optimizer,self.tiny.scheduler=runtime,opt,scheduler
        genesis_ref=self.case.genesis(identity)
        store=t.TransactionStore(self.root,self.case.freeze,self.freeze_sha,identity,native=self.native,genesis_reference=genesis_ref)
        audit=self.root/f'audit-{start}-{arm}';audit.mkdir();genesis=t.read_json(t.bound(self.root,genesis_ref))
        dump(audit/'FORK_LOAD.json',dict(identity=identity,fork_receipt=genesis['fork_receipt'],resume=False))
        saved=[];context={}
        def collect(stage,requests):
            step=requests[0]['step'];folder=audit/f'step-{step}';folder.mkdir();context['folder']=folder;raw=[];refs=[]
            policy=self.native.state_hash(self.native.trainable_state(runtime.model))
            endpoint=dict(experiment_id=a.state.EXPERIMENT,start_seed=start,arm=arm,step=step-1,
                policy_sha256=policy,reference_sha256=source['reference_sha256'])
            for req in requests:
                stem=hashlib.sha256(req['request_id'].encode()).hexdigest();capture=folder/(stem+'.CAPTURE.json')
                body=dict(seed=req['seed'],tokens=[3,4],raw_tokens=[3,4],completion_token_count=2,sampler_logprobs=[-.2,-.3],
                    sampler_logprob_source='actual_generate_raw_logits_selected_token',prompt='CPU_SYNTHETIC',prompt_tensor_hash='d'*64,
                    model_input={'prompt':'CPU_SYNTHETIC','image_file':'cpu.png','image_sha256':'e'*64},assistant_prefill='{',
                    full_text='{}',generated_text='}',finish_reason='balanced',generation_status='COMPLETE',technical_validation_errors=[])
                dump(capture,body);begin=folder/(stem+'.BEGIN.json')
                dump(begin,dict(request_id=req['request_id'],seed=req['seed'],endpoint=endpoint,
                    design_freeze_sha256=self.freeze_sha,status='ATTEMPTED_UNKNOWN'))
                complete=dict(body,request_id=req['request_id'],native_capture_path=str(capture),native_capture_sha256=t.file_sha(capture),
                    policy_sha256=policy,reference_sha256=source['reference_sha256'],design_freeze_sha256=self.freeze_sha)
                path=folder/(stem+'.COMPLETE.json');dump(path,complete);refs.append(self.ref(path))
                raw.append(dict(complete,**{k:v for k,v in req.items() if k not in complete},scope='TRAIN',
                    native_begin_sha256=t.file_sha(begin),J=1,exact_authorised=1,native_score={'J':1,'known_mask':{'S':1}}))
            lines(folder/'raw.jsonl',raw);raw_ref=self.ref(folder/'raw.jsonl')
            dump(folder/'scores.json',dict(schema='sr-satdose-training-scores-v1',identity=identity,step=step,raw=raw_ref,
                native_rescoring_complete=True,rows=[{k:r[k] for k in ('request_id','native_score','J','exact_authorised')} for r in raw]))
            dump(folder/'ROLLOUT_COMPLETE.json',dict(schema='sr-satdose-training-rollout-v1',status='COMPLETE_NOT_TRAINING_ADMISSION',
                identity=identity,step=step,responses=128,design_freeze_sha256=self.freeze_sha,
                source_lock_sha256=t.file_sha(self.root/'source_lock.json'),raw=raw_ref,complete_files=refs,scores=self.ref(folder/'scores.json')))
            return self.ref(folder/'ROLLOUT_COMPLETE.json')
        def prepare(stage,rollout):
            step=t.read_json(t.bound(self.root,rollout))['step'];folder=context['folder']
            context['before']=self.native.trainable_state(runtime.model)
            for p in runtime.model.parameters():p.grad=torch.full_like(p,.01)
            torch.save({'CPU_SYNTHETIC':True},folder/'postclip.pt');capture=self.ref(folder/'postclip.pt')
            context['capture']=capture
            dump(folder/'PRECOMMIT.json',dict(schema='sr-satdose-training-precommit-v1',identity=identity,step=step,
                rollout_receipt=rollout,gradient_stage='postclip',numerical_checks_passed=True,production_capture=capture,artifacts=[capture]))
            return self.ref(folder/'PRECOMMIT.json')
        def storage(stage,step):
            path=context['folder']/'CPU_QUOTA.json';dump(path,{'CPU_SYNTHETIC':True})
            return dict(schema='sr-satdose-storage-readiness-v1',ready=True,quota_verified=True,writable_verified=True,
                required_bytes=1,effective_available_bytes=2,evidence=[self.ref(path)])
        def schedule():scheduler.step();opt.zero_grad(set_to_none=True)
        def writer(stage,fields):
            step=t.read_json(stage/'UPDATE.json')['step'];after=self.native.trainable_state(runtime.model);folder=context['folder']
            delta=a._delta(context['before'],after,identity['optimizer_parameter_order'])
            dump(folder/'NATIVE_UPDATE.json',dict(schema='sr-satdose-training-native-update-v1',status='UPDATED_UNCOMMITTED_NOT_ADMISSION',
                identity=identity,step=step,policy_before_sha256=self.native.state_hash(context['before']),
                policy_after_sha256=self.native.state_hash(after),reference_sha256=source['reference_sha256'],
                optimizer_calls=1,scheduler_calls=1,parameter_update_norm=delta['parameter_update_norm'],production_capture=context['capture']))
            return a.state.write_checkpoint_files(stage,runtime,opt,scheduler,reference,self.native,identity=identity,step=step,
                sampler_state=dict(committed_step=step,next_step=step+1,generated_responses=step*128),
                commit_fields=dict(fields,native_update=self.ref(folder/'NATIVE_UPDATE.json')))
        for step in (1,2):
            saved.append(store.commit_step(step,collect_evidence=collect,prepare_update=prepare,optimizer_step=opt.step,
                scheduler_step=schedule,write_checkpoint=writer,storage_ready=storage));runtime.identity['step']=step
        basepath=self.root/f'canary_replay/start-{start}/{arm}';basepath.mkdir(parents=True)
        spec=dict(schema=a.replay.SCHEMA,permission='READONLY_STEP2_CANARY_ONLY',identity=identity,
            design_freeze_sha256=self.freeze_sha,execution_sources=self.sources,baseline_audit=str(audit.relative_to(self.root)),
            restored_audit=str((basepath/'restored/audit').relative_to(self.root)),
            slots={p:str((basepath/p).relative_to(self.root)) for p in ('baseline','restored')},
            commits={str(step):self.ref(store.path/f'step-{step:03d}/COMMIT.json') for step in (1,2)})
        dump(basepath/'TECHNICAL_EXECUTION.json',spec);spec_ref=self.ref(basepath/'TECHNICAL_EXECUTION.json');self.trusted[key]=spec_ref
        chain=[(entry['commit_sha256'],t.read_json(Path(entry['checkpoint'])/'COMMIT.json')) for entry in saved]
        rows,raw_ref=a.replay._originals(store,chain[1][1]);phases={}
        for phase,pid in (('baseline',100+index),('restored',200+index)):
            slot=basepath/phase;slot.mkdir();output=audit if phase=='baseline' else slot/'audit'
            if phase=='restored':
                output.mkdir();(output/'original_tokens').mkdir()
                dump(output/'FORK_LOAD.json',dict(identity=identity,fork_receipt=genesis['fork_receipt'],resume=True))
            claim=dict(schema='sr-satdose-canary-replay-claim-v1',phase=phase,identity=identity,spec=spec_ref,
                process=process(pid),status='UNKNOWN_UNTIL_COMPLETE_NO_RETRY');dump(slot/'CLAIM.json',claim)
            movement=[];vectors=[];events=[]
            sampled=dict(experiment_id=a.state.EXPERIMENT,start_seed=start,arm=arm,step=1,policy_sha256=chain[0][1]['policy_sha256'],
                reference_sha256=source['reference_sha256']);teacher=dict(sampled,step=2,policy_sha256=chain[1][1]['policy_sha256'])
            for row in rows:
                src=Path(row['native_capture_path'])
                for original in (src,src.with_name(src.name.replace('.CAPTURE.json','.BEGIN.json'))):
                    dst=original if phase=='baseline' else output/'original_tokens'/original.name
                    if phase=='restored':shutil.copyfile(original,dst)
                    movement.append(dict(source=self.ref(original),destination=self.ref(dst)))
                two=[dict(kind='extra_forward_sequences',count=1,metadata=dict(execution_phase='CANARY_RESTORE_TEACHER',purpose=p))
                     for p in ('training_reference','training_policy')];events.extend(two)
                vectors.append(dict(status='CANARY_TEACHER_CAPTURE_NOT_RESUME_ADMISSION',request_id=row['request_id'],tokens=row['tokens'],
                    sampler_logprobs=row['sampler_logprobs'],sampled_endpoint=sampled,teacher_endpoint=teacher,
                    sampled_commit_sha256=chain[0][0],teacher_commit_sha256=chain[1][0],backward_calls=0,optimizer_steps=0,new_generations=0,
                    accounting=two,teacher_logprobs=torch.tensor([-.4,-.5]),reference_logprobs=torch.tensor([-.6,-.7])))
            torch.save(vectors,slot/'TEACHER.pt');lines(output/'accounting.jsonl',events)
            marker=chain[1][1];snapshot=dict(identity=identity,trainable_tensors=496,optimizer_parameter_order=identity['optimizer_parameter_order'],
                model_training=True,**{k+'_sha256':marker[k+'_sha256'] for k in ('policy','optimizer','scheduler','rng','reference')})
            dump(slot/'STATE_IDENTITY.json',snapshot);dump(slot/'ACCOUNTING.json',dict(events=events,measured_counts={'training_policy':128,'training_reference':128}))
            dump(slot/'RELOCATION.json',dict(rows=movement,raw=raw_ref))
            result=dict(schema='sr-satdose-canary-replay-result-v1',phase=phase,status=a.replay.BASELINE if phase=='baseline' else a.replay.RESTORED,
                permission='NO_STEP3_OR_AGGREGATE_ADMISSION',identity=identity,spec=spec_ref,process=process(pid),request_count=128,
                sampled_commit_sha256=chain[0][0],teacher_commit_sha256=chain[1][0],claim=self.ref(slot/'CLAIM.json'),tensors=self.ref(slot/'TEACHER.pt'),
                state_identity=self.ref(slot/'STATE_IDENTITY.json'),accounting=self.ref(slot/'ACCOUNTING.json'),relocation=self.ref(slot/'RELOCATION.json'),
                fork_load=self.ref(output/'FORK_LOAD.json'),raw=raw_ref,measured_counts={'training_policy':128,'training_reference':128})
            if phase=='restored':result['baseline']=self.ref(basepath/'baseline/COMPLETE.json')
            dump(slot/'COMPLETE.json',result);phases[phase]=dict(result=result,reference=self.ref(slot/'COMPLETE.json'))
        context=dict(store=store,chain=chain,spec=spec,spec_ref=spec_ref,phases=phases);self.contexts[key]=context
        self.make_proofs(key,context,genesis)

    def make_proofs(self,key,ctx,genesis):
        identity=ctx['store'].identity;base=self.root/'proofs'/key;beforeafter={}
        # Manufactured CPU test records, NEVER actual GPU observations/admission.
        row=dict(name='model.visual.CPU_SYNTHETIC',shape=[1],dtype='torch.float32',device='cuda:0',requires_grad=False,tensor_sha256='b'*64)
        for phase,label,step in (('BEFORE_FIRST_STEP','before',0),('AFTER_STEP2','after',2)):
            marker=ctx['chain'][1][1];source=identity['source_binding'];expected=genesis if step==0 else marker
            hashes={k+'_sha256':expected[k+'_sha256'] for k in ('optimizer','scheduler','rng')}
            hashes.update(policy_sha256=source['policy_sha256'] if step==0 else marker['policy_sha256'],reference_sha256=source['reference_sha256'])
            value=dict(schema=a.WITNESS_SCHEMA,mode='CUDA_OBSERVATION',permission='NO_TRAINING',phase=phase,
                producer=self.sources['frozen_fingerprint'],identity=identity,process=ctx['phases']['baseline']['result']['process'],
                endpoint=dict(experiment_id=a.state.EXPERIMENT,start_seed=identity['train_seed'],arm=identity['arm'],step=step,
                    policy_sha256=hashes['policy_sha256'],reference_sha256=source['reference_sha256']),protected_state=hashes,
                scope='ALL_NAMED_NON_LORA_PARAMETERS_ALIASES_INCLUDED',parameters=[row],inventory_sha256=a.object_sha([{k:v for k,v in row.items() if k!='tensor_sha256'}]),
                values_sha256=a.object_sha([row]),buffers_included=False,rng_unchanged=True,model_calls=0,optimizer_steps=0,backward_calls=0,
                gpu_inventory=[dict(index=i,name='CPU_SYNTHETIC_PRO6000',major=12,minor=0,total_memory=1,uuid=str(i)) for i in (0,1)])
            path=base/(label+'.json');dump(path,value);beforeafter[label]=self.ref(path)
        self.frozen[key]=beforeafter;controller=process(999)
        for label,event,tick,phase in (('wait','BASELINE_WAIT_RETURNED',10,'baseline'),('launch','RESTORED_LAUNCH_REQUESTED',20,'restored')):
            value=dict(schema='sr-satdose-controller-process-event-v1',event=event,controller_source=self.sources['canary_controller'],
                controller_process=controller,identity=identity,spec=ctx['spec_ref'],monotonic_ns=tick,
                worker_process=ctx['phases'][phase]['result']['process'],worker_claim=ctx['phases'][phase]['result']['claim'])
            if label=='wait':value.update(kind='DIRECT_CHILD_WAIT',waited_pid=value['worker_process']['pid'],returncode=0)
            dump(base/(label+'.json'),value)
        proof=dict(schema=a.EXIT_SCHEMA,identity=identity,spec=ctx['spec_ref'],controller_source=self.sources['canary_controller'],
            controller_process=controller,baseline_process=ctx['phases']['baseline']['result']['process'],
            restored_process=ctx['phases']['restored']['result']['process'],baseline_complete=ctx['phases']['baseline']['reference'],
            restored_claim=ctx['phases']['restored']['result']['claim'],wait_event=self.ref(base/'wait.json'),restored_launch_event=self.ref(base/'launch.json'))
        dump(base/'EXIT.json',proof);self.exits[key]=self.ref(base/'EXIT.json')

    def audit(self,**changes):
        kwargs=dict(trusted_specs=self.trusted,execution_sources=self.sources,expected_auditor_sha256=self.auditor_sha,
            expected_freeze_sha256=self.freeze_sha,frozen_witnesses=self.frozen,process_exit_proofs=self.exits)
        kwargs.update(changes);return a.audit_four_paths(self.root,**kwargs)

    def test_complete_synthetic_contract_actual_cpu_states_and_no_side_effects(self):
        before={str(p):t.file_sha(p) for p in self.root.rglob('*') if p.is_file()};rng=self.native.state_hash(self.native.capture_rng())
        with patch.object(torch.cuda,'get_rng_state_all',side_effect=AssertionError('audit touched GPU')):
            result=self.audit()
        self.assertEqual(result['status'],'FOUR_PATH_CANARY_EVIDENCE_AUDITED_NOT_ADMISSION');self.assertEqual(result['permission'],'NO_TRAINING')
        self.assertEqual(set(result['paths']),set(a.PATHS));self.assertFalse(result['pending'])
        for path in result['paths'].values():
            self.assertTrue(all(d['parameter_update_norm']>0 and d['compared_tensors']==496 for d in path['actual_nonzero_policy_deltas']))
        self.assertEqual(before,{str(p):t.file_sha(p) for p in self.root.rglob('*') if p.is_file()})
        self.assertEqual(self.native.state_hash(self.native.capture_rng()),rng)

    def test_missing_exit_or_frozen_proofs_is_pending_never_positive(self):
        for kwargs in ({'frozen_witnesses':None},{'process_exit_proofs':None}):
            with self.subTest(kwargs=kwargs):
                result=self.audit(**kwargs);self.assertEqual(result['status'],'PENDING_CANARY_EVIDENCE');self.assertTrue(result['pending'])
                self.assertEqual(result['permission'],'NO_TRAINING')
        missing=dict(self.trusted);missing.pop('11032/T1')
        with self.assertRaisesRegex(PermissionError,'Exact four'):self.audit(trusted_specs=missing)

    def test_changed_arrays_or_original_order_are_recomputed_not_status(self):
        ctx=self.contexts['11031/T0'];complete=self.root/ctx['spec']['slots']['restored']/'COMPLETE.json'
        result=t.read_json(complete);path=t.bound(self.root,result['tensors']);original=path.read_bytes()
        for damage in ('array','order'):
            vectors=torch.load(path,weights_only=True)
            if damage=='array':vectors[0]['teacher_logprobs'][0]+=.01
            else:vectors[0],vectors[1]=vectors[1],vectors[0]
            torch.save(vectors,path);result['tensors']=self.ref(path);dump(complete,result)
            with self.assertRaisesRegex(PermissionError,'bit-exact|token/order'):self.audit()
            path.write_bytes(original)

    def test_source_or_trusted_spec_tampering_refuses(self):
        with self.assertRaisesRegex(PermissionError,'auditor source'):self.audit(expected_auditor_sha256='0'*64)
        sources=copy.deepcopy(self.sources);sources['training_state']['sha256']='0'*64
        with self.assertRaisesRegex(PermissionError,'SHA differs'):self.audit(execution_sources=sources)
        path=t.bound(self.root,self.trusted['11031/T0']);path.write_bytes(path.read_bytes()+b'\n')
        with self.assertRaises(PermissionError):self.audit()

    def test_zero_delta_cannot_be_hidden_by_nonzero_metadata(self):
        ctx=self.contexts['11031/T0'];policy=self.native.trainable_state(self.tiny.runtime.model)
        with self.assertRaisesRegex(PermissionError,'zero/nonfinite'):
            a._delta(policy,copy.deepcopy(policy),ctx['store'].identity['optimizer_parameter_order'])
        # Full audit delegates to actual arrays, even though every UPDATE says nonzero.
        original=a._delta
        def zero_first(before,after,order):return original(before,before,order)
        with patch.object(a,'_delta',side_effect=zero_first):
            with self.assertRaisesRegex(PermissionError,'zero/nonfinite'):self.audit()

    def test_wrong_waited_pid_or_event_order_and_cpu_witness_refuse(self):
        key='11031/T0';proof_path=t.bound(self.root,self.exits[key]);proof=t.read_json(proof_path)
        wait_path=t.bound(self.root,proof['wait_event']);wait=t.read_json(wait_path);wait['waited_pid']=999
        dump(wait_path,wait);proof['wait_event']=self.ref(wait_path);dump(proof_path,proof);self.exits[key]=self.ref(proof_path)
        with self.assertRaisesRegex(PermissionError,'wrapper/foreign PID'):self.audit()
        wait['waited_pid']=proof['baseline_process']['pid'];wait['monotonic_ns']=30;dump(wait_path,wait)
        proof['wait_event']=self.ref(wait_path);dump(proof_path,proof);self.exits[key]=self.ref(proof_path)
        with self.assertRaisesRegex(PermissionError,'exit not observed'):self.audit()
        witness=t.bound(self.root,self.frozen[key]['before']);value=t.read_json(witness);value['mode']='SYNTHETIC_CPU_TEST';dump(witness,value)
        self.frozen[key]['before']=self.ref(witness)
        with self.assertRaisesRegex(PermissionError,'actual frozen-parameter'):self.audit()

    def test_single_task_srun_binds_actual_worker_not_wrapper(self):
        key='11031/T0';proof_path=t.bound(self.root,self.exits[key]);proof=t.read_json(proof_path)
        wait_path=t.bound(self.root,proof['wait_event']);wait=t.read_json(wait_path)
        wait.update(kind='SINGLE_TASK_SRUN_WAIT',wrapper_pid=777,slurm_state='COMPLETED',slurm_exit_code='0:0',
                    task_count=1,slurm_job_id='synthetic-job',slurm_step_id='synthetic-step')
        observation=dict(schema='sr-satdose-controller-worker-slurm-v1',identity=proof['identity'],spec=proof['spec'],
            worker_process=proof['baseline_process'],worker_claim=wait['worker_claim'],controller_source=proof['controller_source'],
            slurm_job_id=wait['slurm_job_id'],slurm_step_id=wait['slurm_step_id'],task_count=1)
        observation_path=wait_path.parent/'WORKER_SLURM.json';dump(observation_path,observation)
        wait['worker_slurm_observation']=self.ref(observation_path);dump(wait_path,wait)
        proof['wait_event']=self.ref(wait_path);dump(proof_path,proof);self.exits[key]=self.ref(proof_path)
        self.assertEqual(self.audit()['status'],'FOUR_PATH_CANARY_EVIDENCE_AUDITED_NOT_ADMISSION')
        observation['worker_process']=dict(proof['baseline_process'],pid=777);dump(observation_path,observation)
        wait['worker_slurm_observation']=self.ref(observation_path);dump(wait_path,wait)
        proof['wait_event']=self.ref(wait_path);dump(proof_path,proof);self.exits[key]=self.ref(proof_path)
        with self.assertRaisesRegex(PermissionError,'actual model worker'):self.audit()

    def test_frozen_tensor_tamper_and_missing_parameter_refuse(self):
        key='11031/T0';path=t.bound(self.root,self.frozen[key]['after']);value=t.read_json(path)
        value['parameters'][0]['tensor_sha256']='c'*64;value['values_sha256']=a.object_sha(value['parameters'])
        dump(path,value);self.frozen[key]['after']=self.ref(path)
        with self.assertRaisesRegex(PermissionError,'bytes changed'):self.audit()

    def test_same_process_and_active_transaction_reject(self):
        with t.exclusive_lock(self.root/'training_transactions'):
            with self.assertRaisesRegex(PermissionError,'Active training'):self.audit()
        ctx=self.contexts['11031/T0'];slot=self.root/ctx['spec']['slots']['restored']
        claim=t.read_json(slot/'CLAIM.json');claim['process']=ctx['phases']['baseline']['result']['process'];dump(slot/'CLAIM.json',claim)
        value=t.read_json(slot/'COMPLETE.json');value['claim']=self.ref(slot/'CLAIM.json');value['process']=claim['process'];dump(slot/'COMPLETE.json',value)
        with self.assertRaisesRegex(PermissionError,'different process'):self.audit()


class FrozenFingerprintCPUHelpers(unittest.TestCase):
    def model(self):
        model=torch.nn.Module();model.visual=torch.nn.Linear(2,2,bias=False);model.visual.weight.requires_grad_(False)
        model.base=torch.nn.Parameter(torch.tensor([1.,2.]),requires_grad=False)
        model.lora_fixture=torch.nn.Parameter(torch.tensor([.1]));model.forward=lambda *args:(_ for _ in ()).throw(AssertionError('no forward'))
        return model
    def test_cpu_fingerprint_all_nonlora_only_and_no_rng_or_model_changes(self):
        model=self.model();rng=torch.random.get_rng_state().clone();before={n:p.clone() for n,p in model.named_parameters()}
        value=a.fingerprint_cpu_for_tests(model)
        self.assertEqual(value['mode'],'SYNTHETIC_CPU_TEST');self.assertEqual({r['name'] for r in value['parameters']},{'base','visual.weight'})
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        self.assertTrue(all(torch.equal(p,before[n]) for n,p in model.named_parameters()))
        with torch.no_grad():model.visual.weight[0,0]+=.1
        changed=a.fingerprint_cpu_for_tests(model);self.assertEqual(value['inventory_sha256'],changed['inventory_sha256'])
        self.assertNotEqual(value['values_sha256'],changed['values_sha256'])
    def test_nonlora_trainable_rejected_and_production_requires_actual_handle(self):
        model=self.model();model.base.requires_grad_(True)
        with self.assertRaisesRegex(PermissionError,'all be frozen'):a.fingerprint_cpu_for_tests(model)
        with self.assertRaisesRegex(PermissionError,'producer identity'):
            a.fingerprint_first_two(SimpleNamespace(),phase='BEFORE_FIRST_STEP',execution_sources={},expected_auditor_sha256=t.file_sha(a.__file__))


if __name__=='__main__':unittest.main()
