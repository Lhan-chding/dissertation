import pathlib,json,subprocess,shlex,sys
tag=sys.argv[1];art=pathlib.Path('artifacts/sr_satdose_20261112')
release=json.loads((art/f'intake/FIRST_TWO_RELEASE_PUBLICATION_{tag}_20261116.json').read_text())
pre=json.loads((art/f'intake/FIRST_TWO_PREFLIGHT_{tag}_20261116.json').read_text())
assert pre['returncode']==0 and json.loads(pre['stdout'])==dict(status='FIRST_TWO_PREFLIGHT_COMPLETE_NO_MODEL_CALLS',start_seed=11031,arm='T0',phase='baseline')
assert pre['release']=={k:release[k] for k in ('path','sha256')}
remote=r'''
import sys,json,pathlib,hashlib,subprocess,datetime,os
d=json.load(sys.stdin);r=d['release'];base=pathlib.Path(r['evidence_root']);code=pathlib.Path(r['release_root']);receipt=pathlib.Path(d['path'])
assert receipt.resolve(strict=True)==receipt and hashlib.sha256(receipt.read_bytes()).hexdigest()==d['sha256']
assert json.loads(receipt.read_text())==r and {str(p.relative_to(code)):hashlib.sha256(p.read_bytes()).hexdigest() for p in code.rglob('*') if p.is_file()}==r['sources']
assert not (base/'first_two_control').exists() and not (base/'first_two_launcher').exists()
def run(args):
 p=subprocess.run(args,capture_output=True,text=True);return dict(args=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
prior=run(['sacct','-j','1740676','-n','-P','--format=JobID,State,ExitCode'])
assert prior['returncode']==0 and '1740676|COMPLETED|0:0' in prior['stdout']
queue=run(['squeue','-u','varun024','-h','-o','%i|%j|%T|%q|%b|%R']);assert queue['returncode']==0 and 'LH-SATDOSE-FIRST-TWO' not in queue['stdout']
directory=base.parent/'scheduling'/('first_two_'+code.name.rsplit('_',1)[-1]);directory.mkdir()
def write(name,value):
 with (directory/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
args=['sbatch','--parsable','--job-name=LH-SATDOSE-FIRST-TWO','--account=rose','--qos=soujanya-poria-startfund-2026-03','--partition=cluster02','--nodes=1','--ntasks=1','--cpus-per-task=8','--gpus=pro6000:2','--mem=66G','--time=24:00:00','--no-requeue','--chdir='+str(code),'--output='+str(directory/'slurm-%j.out'),'--error='+str(directory/'slurm-%j.err'),str(code/'scheduling/run_first_two.sh'),str(receipt),d['sha256']]
intent=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='SUBMISSION_INTENT_UNKNOWN_UNTIL_RECEIPT',scope='FOUR_PATHS_MAX_STEP_2_NO_STEP3',train_responses=1024,formal_optimizer_steps=8,canary_new_generations=0,release={k:d[k] for k in ('path','sha256')},args=args,prerequisite=prior,queue_before=queue)
write('SUBMISSION_INTENT.json',intent)
response=run(args);write('SUBMISSION_RESPONSE.json',response)
assert response['returncode']==0 and response['stdout'].strip().split(';')[0].isdigit(),response
job=int(response['stdout'].strip().split(';')[0]);result=dict(status='FOUR_FIRST_TWO_PATHS_AND_READONLY_RELOAD_SUBMITTED',job_id=job,intent=intent,response=response)
write('SUBMISSION.json',result);print(json.dumps(result))
'''
p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','eee-cluster','python3 -c '+shlex.quote(remote)],input=json.dumps(release).encode(),capture_output=True)
if p.returncode:print(p.stdout.decode());print(p.stderr.decode());raise SystemExit(p.returncode)
d=json.loads(p.stdout)
with (art/f'intake/FIRST_TWO_SUBMISSION_{tag}_20261116.json').open('x') as f:json.dump(d,f,indent=2);f.write('\n')
print(d['status'],d['job_id']);print(d['intent']['queue_before']['stdout'])
