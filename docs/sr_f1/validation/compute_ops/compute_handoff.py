"""Observe one registered validation allocation, then execute gated activation once."""
import datetime,json,os,subprocess,sys,time
from pathlib import Path
r=Path('/projects/_ssd/varunssd/louis-ssvc/sr_f11_20261010');i=r/'technical_incidents/compute_parallel_20261010';py='/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12'
def read(p):return json.loads(p.read_text())
def emit(event,**kw):print(json.dumps(dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),event=event,**kw)),flush=True)
def save(p,value):
 with p.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def run(args):
 x=subprocess.run(args,capture_output=True,text=True,timeout=60);return dict(args=args,returncode=x.returncode,stdout=x.stdout,stderr=x.stderr)
def wait_complete(job):
 previous=None
 while True:
  q=run(['squeue','-j',job,'--noheader','--format=%i|%T']);a=run(['sacct','-j',job,'-n','-P','--format=JobIDRaw,State,ExitCode'])
  if q['returncode'] or a['returncode']:
   emit('SCHEDULER_UNKNOWN_NO_MUTATION',job_id=job,queue=q,accounting=a);time.sleep(60);continue
  rows={ln.split('|')[0]:ln.split('|')[1:3] for ln in a['stdout'].splitlines()};state=rows.get(job)
  signature=(q['stdout'].strip(),str(state))
  if signature!=previous:emit('WAITING_FOR_VALIDATION',job_id=job,queue=q,accounting=a);previous=signature
  if not q['stdout'].strip() and state:
   terminal=state[0].split()[0].rstrip('+')
   if terminal=='COMPLETED' and state[1]=='0:0':return dict(queue=q,accounting=a)
   if terminal in {'FAILED','CANCELLED','TIMEOUT','PREEMPTED','OUT_OF_MEMORY','NODE_FAIL'}:raise RuntimeError('VALIDATION_NOT_PASS:'+job+':'+str(state))
  time.sleep(30)
def action(script,log):
 with (i/log).open('x') as f:
  x=subprocess.run([py,str(i/script)],stdout=f,stderr=subprocess.STDOUT)
 if x.returncode:raise RuntimeError('GATED_ACTION_FAILED:'+script+':'+str(x.returncode))
job=read(i/'PROBE_GPU_SUBMISSION.json')['stdout'].strip().split(';')[0]
try:
 evidence=wait_complete(job);save(i/'HANDOFF_PROBE_TERMINAL.json',evidence)
 if (i/'ACTIVATION_INTENT.json').exists():raise RuntimeError('ACTIVATION_INTENT_EXISTS_RECONCILE_NO_REPEAT')
 emit('GPU_VALIDATION_COMPLETED_VERIFY_AND_ACTIVATE',job_id=job)
 action('compute_activate.py','HANDOFF_ACTIVATION.log')
 verify=read(i/'VERIFY_CPU_SUBMISSION.json')['stdout'].strip().split(';')[0];evidence=wait_complete(verify);save(i/'HANDOFF_VERIFY_TERMINAL.json',evidence)
 emit('ACTIVATION_PREFLIGHT_COMPLETED_RESUME_CONTROLLER',job_id=verify)
 action('compute_resume.py','HANDOFF_RESUME.log')
 save(i/'HANDOFF_COMPLETE.json',dict(status='VALIDATED_COMPUTE_CONTROLLER_RESUMED',controller=read(i/'CONTROLLER_SUBMISSION.json'),engine_acceptance=False,scientific_training_started=False))
 emit('HANDOFF_COMPLETE',controller=read(i/'CONTROLLER_SUBMISSION.json'))
except BaseException as exc:
 emit('HANDOFF_STOPPED_REQUIRES_TECHNICAL_DIAGNOSIS',error_type=type(exc).__name__,error=str(exc));raise
