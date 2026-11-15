import pathlib,subprocess,shlex,json
remote=r'''
import pathlib,json,subprocess,datetime
base=pathlib.Path('/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/prepared_data_v1')
def run(args):
 p=subprocess.run(args,capture_output=True,text=True);return dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
out=dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),job_id=1740676,sacct=run(['sacct','-j','1740676','-n','-P','--format=JobID,State,Elapsed,ExitCode,NodeList']),squeue=run(['squeue','-j','1740676','-h','-o','%i|%T|%M|%l|%q|%R']),starts={})
for start in (11031,11032):
 p=base/'start_dev'/f'start-{start}';endpoint=p/'endpoint'
 info=dict(exists=p.exists(),begun=len(list(endpoint.glob('*.BEGIN.json'))),captured=len(list(endpoint.glob('*.CAPTURE.json'))),completed=len(list(endpoint.glob('*.COMPLETE.json'))))
 for name in ('RESULT.json','HOLD.json'):
  if (p/name).exists():info[name]=json.loads((p/name).read_text())
 out['starts'][str(start)]=info
logs=base.parent/'scheduling/start_dev_9b925a3'
out['log_tail']={name:(logs/name).read_text(errors='replace')[-2500:] for name in ('slurm-1740676.out','slurm-1740676.err') if (logs/name).exists()}
print(json.dumps(out))
'''
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','eee-cluster','python3 -c '+shlex.quote(remote)],capture_output=True)
if r.returncode:print(r.stderr.decode());raise SystemExit(r.returncode)
d=json.loads(r.stdout);tag=d['checked_at_utc'][:19].replace('-','').replace(':','').replace('T','T')+'Z'
p=pathlib.Path('artifacts/sr_satdose_20261112/intake')/('MONITOR_1740676_'+tag+'.json')
with p.open('x') as f:json.dump(d,f,indent=2);f.write('\n')
print(json.dumps(d,indent=2))
