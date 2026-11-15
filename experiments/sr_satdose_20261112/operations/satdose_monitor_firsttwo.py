import pathlib,subprocess,shlex,json,sys
job=int(sys.argv[1])
remote=r'''
import pathlib,json,subprocess,datetime,sys
job=str(int(sys.argv[1]));base=pathlib.Path('/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/prepared_data_v1')
def run(args):
 p=subprocess.run(args,capture_output=True,text=True);return dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
def summary(p):
 out=dict(exists=p.exists())
 if not p.exists():return out
 out['generation_counts']={suffix:len(list(p.rglob('*.'+suffix+'.json'))) for suffix in ('BEGIN','CAPTURE','COMPLETE')}
 out['incidents']={str(f.relative_to(base)):json.loads(f.read_text()) for f in p.rglob('*.json') if f.name in ('HOLD.json','ERROR.json') or f.name.endswith('.ERROR.json')}
 out['results']={str(f.relative_to(base)):json.loads(f.read_text()) for f in p.rglob('RESULT.json')}
 return out
out=dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),job_id=int(job),sacct=run(['sacct','-j',job,'-n','-P','--format=JobID,State,Elapsed,ExitCode,NodeList']),squeue=run(['squeue','-j',job,'-h','-o','%i|%T|%M|%l|%q|%R']),paths={})
for start in (11031,11032):
 for arm in ('T0','T1'):
  key=f'{start}/{arm}';p=base/'first_two_control'/f'start-{start}'/arm
  tx=base/'training_transactions'/f'start-{start}'/arm
  out['paths'][key]=dict(controller=summary(p),commits=sorted(str(f.relative_to(base)) for f in tx.glob('step-*/COMMIT.json')),staging=sorted(str(f.relative_to(base)) for f in tx.glob('.staging*')))
out['launcher']=summary(base/'first_two_launcher')
logs=base.parent/'scheduling/first_two_71018e5'
out['log_tail']={str(p.relative_to(base.parent)):p.read_text(errors='replace')[-2500:] for p in logs.glob('slurm-'+job+'.*')}
print(json.dumps(out))
'''
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','eee-cluster','python3 -c '+shlex.quote(remote)+' '+str(job)],capture_output=True)
if r.returncode:print(r.stderr.decode());raise SystemExit(r.returncode)
d=json.loads(r.stdout);tag=d['checked_at_utc'][:19].replace('-','').replace(':','')+'Z'
p=pathlib.Path('artifacts/sr_satdose_20261112/intake')/('MONITOR_'+str(job)+'_'+tag+'.json')
with p.open('x') as f:json.dump(d,f,indent=2);f.write('\n')
print(json.dumps(d,indent=2))
