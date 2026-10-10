import datetime,hashlib,json,os,pathlib,re,subprocess,tarfile
r=pathlib.Path('/projects/_ssd/varunssd/louis-ssvc/sr_f11_20261010');inc=r/'technical_incidents/multigpu_20261010';code=r/'code_engine_multigpu_candidate_v2'
py='/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def run(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=45)
 return dict(command=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,time=now())
assert r.resolve()==r and sha(r/'EXECUTION_FREEZE.json')=='c7cf051d3aa6fb45889324e64b08d31a008137668aff2382283e4e19f99ff2b8'
assert not (r/'ENGINE_MULTIGPU_REPAIR.json').exists() and not code.exists()
inv=read(inc/'inventory_v2.json');assert sha(inc/'source_v2.tar.gz')==inv['sha256']
code.mkdir()
with tarfile.open(inc/'source_v2.tar.gz') as t:
 m=t.getmembers();assert len(m)==len(inv['files'])==len({x.name for x in m})
 for x in m:
  n=pathlib.PurePosixPath(x.name);assert x.isfile() and not n.is_absolute() and '..' not in n.parts
  b=t.extractfile(x).read();assert hashlib.sha256(b).hexdigest()==inv['files'][x.name]
  p=code/x.name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
identity=read(code/'SOURCE_DEPLOYMENT.json')
assert identity['source_commit']==inv['source_commit']
for n,h in identity['source_file_hashes'].items():assert sha(code/n)==h,n
save(inc/'STAGING_V2_VERIFIED.json',dict(time=now(),source_commit=identity['source_commit'],source_tree_sha256=identity['source_tree_sha256'],files=len(inv['files']),status='CANDIDATE_ONLY_PRODUCTION_UNCHANGED'))
script=inc/'candidate_cpu_v2.sbatch'
script.write_text(f'''#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH={code}/src
cd {code}
{py} -m pytest -q --import-mode=importlib tests/sr_f1 tests/mm_dev/test_cached_teacher_forcing.py tests/mm_core/test_training.py docs/sr_f1/package/tests > {inc}/CANDIDATE_SERVER_TESTS_V2.log 2>&1
{py} -c 'import json; from sr_f1.prepare import verify_package,source_identity; print(json.dumps(dict(package=verify_package(),source=source_identity())))' > {inc}/CANDIDATE_SOURCE_CHECK_V2.json
''')
args=['sbatch','--parsable','--account=rose','--qos=override-limits-but-killable','--partition=cluster02','--cpus-per-task=2','--time=30','--job-name=srf11-mgpu-cpu-v2','--output='+str(inc/'candidate_cpu_v2_%j.log'),str(script)]
save(inc/'CANDIDATE_CPU_V2_SUBMISSION_INTENT.json',dict(command=args,time=now()))
x=run(args);save(inc/'CANDIDATE_CPU_V2_SUBMISSION.json',x);assert x['returncode']==0,x
job=x['stdout'].strip().split(';')[0];assert re.fullmatch('[0-9]+',job)
print(json.dumps(dict(status='CANDIDATE_STAGED_CPU_TESTS_SUBMITTED',cpu_job=job,source_commit=identity['source_commit'],production_unchanged=True)))
