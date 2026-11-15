import sys,json,subprocess,pathlib,shlex,datetime
tag=sys.argv[1];art=pathlib.Path('artifacts/sr_satdose_20261112');d=json.loads((art/f'intake/FIRST_TWO_RELEASE_PUBLICATION_{tag}_20261116.json').read_text())
args=['/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python','-B','-m','execution.first_two_control','--release-receipt',d['path'],'--release-sha256',d['sha256'],'--start-seed','11031','--arm','T0','--phase','baseline']
cmd='cd '+shlex.quote(d['release']['release_root'])+' && env PYTHONDONTWRITEBYTECODE=1 '+shlex.join(args)
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','eee-cluster',cmd],capture_output=True,text=True)
out=dict(started_at_utc=started,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),args=args,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,release={k:d[k] for k in ('path','sha256')})
path=art/f'intake/FIRST_TWO_PREFLIGHT_{tag}_20261116.json'
with path.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(r.stdout);print(r.stderr);raise SystemExit(r.returncode)
