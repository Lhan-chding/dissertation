from pathlib import Path
import json,hashlib,subprocess,datetime
r=Path('/projects/_ssd/varunssd/louis-ssvc/sr_f12_20261010')
def j(p):return json.loads((r/p).read_text())
m=j('source_r0001b.json');c=r/'code_repair_r0001b'
assert all(hashlib.sha256((c/n).read_bytes()).hexdigest()==h for n,h in m['files'].items())
baseline=[]
for s in sorted((r/'raw/evaluation').glob('*/shard*')):
 rows=[json.loads(p.read_text()) for p in (s/'groups').glob('*.json')]
 baseline.append(dict(shard=s.name,groups=len(rows),records=sum(len(v['records']) for v in rows)))
a=j('source_revisions/r0001/AUTHORIZATION.json')
for n,h in a['reuse_preflight']['files'].items():assert hashlib.sha256((r/'technical/preflight/rollouts'/n).read_bytes()).hexdigest()==h
out=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=m['git_commit'],source_files_verified=len(m['files']),reused_raw_files_verified=4,reused_raw_records=32,applied=j('source_revisions/r0001/APPLIED.json'),cpu_validation=j('deployment/r0001b/VALIDATION_COMPLETE.json'),orchestration=j('orchestration/STATE.json'),endpoint=j('endpoint_orchestration/STATE.json'),baseline=baseline,baseline_total=sum(v['records'] for v in baseline),accounting=subprocess.check_output(['sacct','-X','-n','-P','-j','196823,196847,196869,196875,196876,196877,196826,196827,196828','-o','JobID,State,ExitCode,Submit,Start,End'],text=True),queue=subprocess.check_output(['squeue','-u','varun024','-h','-o','%i|%j|%T|%q|%b|%R'],text=True))
print(json.dumps(out,ensure_ascii=False))
