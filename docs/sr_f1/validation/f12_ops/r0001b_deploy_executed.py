import pathlib,json,hashlib,tarfile,subprocess,sys
r=pathlib.Path('/projects/_ssd/varunssd/louis-ssvc/sr_f12_20261010');c=r/'code_repair_r0001b';d=r/'deployment/r0001b'
d.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 with (d/n).open('x') as f:json.dump(x,f,indent=2)
p=r/'srf12_r0001b.tar.gz';assert sha(p)==sys.argv[1]
assert not c.exists();c.mkdir()
with tarfile.open(p) as t:
 members=t.getmembers();assert len({x.name for x in members})==len(members)
 assert all(x.isfile() and not pathlib.PurePosixPath(x.name).is_absolute() and '..' not in pathlib.PurePosixPath(x.name).parts for x in members)
 t.extractall(c)
m=json.loads((r/'source_r0001b.json').read_text())
for f,h in m['files'].items():assert sha(c/f)==h,f
sys.path.insert(0,str(c/'src'))
from sr_f12.evaluation import verify_source_manifest
from sr_f12.source_revision import generation_identity,training_identity
source_hash=verify_source_manifest(r/'source_r0001b.json',c)
assert generation_identity(r/'code')==generation_identity(c)
assert training_identity(r/'code')==training_identity(c)
save('SOURCE_VERIFIED.json',dict(status='PASS',source_sha256=source_hash,source_commit=m['git_commit'],files=len(m['files']),generation_unchanged=True,training_core_unchanged=True))
py='/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12'
verify=d/'validate_complete.py';verify.write_text('from pathlib import Path\nfrom sr_f12.orchestration import save_json\nfrom sr_f12.evaluation import verify_source_manifest\nr=Path('+repr(str(r))+')\ns=verify_source_manifest(r/"source_r0001b.json",r/"code_repair_r0001b")\nsave_json(r/"deployment/r0001b/VALIDATION_COMPLETE.json",dict(status="PASS",source_sha256=s,cpu_tests_passed=True),exclusive=True)\n')
script=d/'validate.sbatch';script.write_text('#!/bin/bash\nset -euo pipefail\nexport CUDA_VISIBLE_DEVICES=""\nexport PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\nexport PYTHONPATH='+str(c/'src')+'\ncd '+str(c)+'\n'+py+' -m pytest tests/sr_f12 tests/sr_f1/test_contract_boundary.py tests/sr_f1/test_json_protocol.py -q\n'+py+' '+str(verify)+'\n'+py+' '+str(d/'activate.py')+'\n')
a=['sbatch','--parsable','--account=rose','--partition=cluster02','--qos=override-limits-but-killable','--cpus-per-task=4','--mem=12G','--time=30','--job-name=srf12-r0001b-validation','--output='+str(d/'validate-%j.log'),str(script)]
save('CPU_INTENT.json',dict(args=a,source_commit=m['git_commit']))
p=subprocess.run(a,text=True,capture_output=True);save('CPU_SUBMISSION.json',dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr));assert p.returncode==0,p.stderr
print('R0001_VALIDATION',p.stdout.strip())
