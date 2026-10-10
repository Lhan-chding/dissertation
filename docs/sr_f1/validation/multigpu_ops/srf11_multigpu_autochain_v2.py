import argparse,datetime,hashlib,json,os,pathlib,re,subprocess
r=pathlib.Path('/projects/_ssd/varunssd/louis-ssvc/sr_f11_20261010');inc=r/'technical_incidents/multigpu_20261010'
py='/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python'
def read(p):return json.loads(p.read_text())
def save(n,x):
 with (inc/n).open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def run(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=45)
 return dict(command=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,time=datetime.datetime.now(datetime.timezone.utc).isoformat())
def submit(phase,dependencies):
 assert phase in {'maintenance','deploy','resume'}
 assert dependencies and all(re.fullmatch('[0-9]+',j) for j in dependencies)
 accounting=run(['sacct','-j',','.join(dependencies),'--format=JobID,State,ExitCode','-n','-P'])
 assert accounting['returncode']==0,accounting
 rows={z[0]:z[1:3] for z in (line.split('|') for line in accounting['stdout'].splitlines())}
 queue=run(['squeue','--noheader','--user','varun024','--format=%i|%j|%T'])
 assert queue['returncode']==0,queue
 active={line.split('|')[0] for line in queue['stdout'].splitlines()}
 completed=[j for j in dependencies if rows.get(j)==['COMPLETED','0:0']]
 pending=[j for j in dependencies if j not in completed]
 assert all(j in active and rows.get(j,['UNKNOWN'])[0] in ['RUNNING','PENDING','COMPLETING'] for j in pending),(pending,rows,active)
 save('AUTO_V2_'+phase.upper()+'_DEPENDENCY_CHECK.json',dict(accounting=accounting,queue=queue,completed_verified=completed,live_dependencies=pending))
 script=inc/('autochain_v2_'+phase+'.sbatch')
 with script.open('x') as f:f.write(f'''#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
{py} {inc}/srf11_multigpu_autochain_v2.py --phase {phase}
''')
 args=['sbatch','--hold','--parsable','--account=rose','--qos=override-limits-but-killable','--partition=cluster02','--cpus-per-task=2','--mem=12G','--time=30',*( ['--dependency=afterok:'+':'.join(pending)] if pending else [] ),'--job-name=srf11-mgpu-v2-'+phase,'--output='+str(inc/('autochain_v2_'+phase+'_%j.log')),str(script)]
 save('AUTO_V2_'+phase.upper()+'_SUBMISSION_INTENT.json',dict(command=args))
 x=run(args);save('AUTO_V2_'+phase.upper()+'_SUBMISSION.json',x);assert x['returncode']==0,x
 job=x['stdout'].strip().split(';')[0];assert re.fullmatch('[0-9]+',job)
 release=run(['scontrol','release',job]);save('AUTO_V2_'+phase.upper()+'_RELEASE.json',release);assert release['returncode']==0,release
 return job
def execute(name):
 script=inc/('srf11_multigpu_'+name+'.py')
 result=subprocess.run([py,str(script)],check=False)
 if result.returncode:raise SystemExit(result.returncode)
parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['submit','maintenance','deploy','resume'],required=True);args=parser.parse_args()
assert r.resolve()==r and not (r/'STOP').exists()
for n,h in read(inc/'OPERATION_SCRIPT_HASHES_V2.json').items():assert hashlib.sha256((inc/n).read_bytes()).hexdigest()==h,n
if args.phase=='submit':
 failed=read(inc/'AUTO_MAINTENANCE_SUBMISSION.json')
 assert failed['returncode']!=0 and not failed['stdout'].strip() and 'Job dependency problem' in failed['stderr']
 assert not (inc/'MAINTENANCE_CANCEL_INTENT.json').exists()
 q=run(['squeue','--noheader','--user','varun024','--format=%i|%j|%T'])
 assert q['returncode']==0 and 'srf11-mgpu-maintenance' not in q['stdout'] and 'srf11-mgpu-v2-' not in q['stdout'],q
 probe=read(inc/'GPU_PROBE_SUBMISSION.json')['stdout'].strip().split(';')[0]
 cpu=read(inc/'CANDIDATE_CPU_V2_SUBMISSION.json')['stdout'].strip().split(';')[0]
 print(json.dumps(dict(status='MAINTENANCE_CHAIN_REGISTERED',job_id=submit('maintenance',[probe,cpu]))),flush=True)
else:
 own=os.environ['SLURM_JOB_ID'];assert own.isdigit()
 bound=read(inc/('AUTO_V2_'+args.phase.upper()+'_SUBMISSION.json'))['stdout'].strip().split(';')[0];assert own==bound
 if args.phase=='maintenance':
  execute('maintain');execute('wait_terminal');execute('preserve');execute('state_review');execute('recovery_review');execute('state_review_submit_marker')
  print(json.dumps(dict(status='PRESERVED_AND_AUDITED_DEPLOY_QUEUED',job_id=submit('deploy',[own]))),flush=True)
 elif args.phase=='deploy':
  execute('deploy')
  verify=read(inc/'VERIFY_SUBMISSION.json')['stdout'].strip().split(';')[0]
  print(json.dumps(dict(status='DEPLOYED_RESUME_WAITS_VERIFICATION',job_id=submit('resume',[own,verify]))),flush=True)
 else:execute('resume')
