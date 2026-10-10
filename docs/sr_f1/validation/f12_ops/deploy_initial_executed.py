import pathlib,json,hashlib,tarfile,subprocess,os
r=pathlib.Path('/projects/_ssd/varunssd/louis-ssvc/sr_f12_20261010');c=r/'code';d=r/'deployment';d.mkdir(exist_ok=True)
def save(n,v):
 with (d/n).open('x') as f:json.dump(v,f,indent=2)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8<<20),b''):h.update(x)
 return h.hexdigest()
p=r/'srf12_source_3542869.tar.gz'
assert sha(p)=='5b795a368061a09383c0d1628fda1b33047f84ada372fdbe57fe2f828efaf276'
assert not c.exists();c.mkdir()
with tarfile.open(p) as t:
 members=t.getmembers();assert len({x.name for x in members})==len(members)
 assert all(x.isfile() and not pathlib.PurePosixPath(x.name).is_absolute() and '..' not in pathlib.PurePosixPath(x.name).parts for x in members)
 t.extractall(c)
m=json.loads((r/'srf12_source_manifest.json').read_text())
for name,h in m['files'].items():assert sha(c/name)==h,name
save('SOURCE_VERIFIED.json',dict(status='PASS',git_commit=m['git_commit'],files=len(m['files']),manifest_sha256=sha(r/'srf12_source_manifest.json')))
py='/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12'
verify=d/'verify_inputs.py'
verify.write_text('''import pathlib,json,hashlib\nfrom sr_f12.protocol import build_config,register_amendment,immutable_json\nr=pathlib.Path('/projects/_ssd/varunssd/louis-ssvc/sr_f12_20261010')\ndef sha(p):\n h=hashlib.sha256()\n with p.open('rb') as f:\n  for b in iter(lambda:f.read(8<<20),b''):h.update(b)\n return h.hexdigest()\ninputs=json.loads((r/'cleanup/INPUTS_RETAINED.json').read_text())\nfor x in inputs['files']:\n p=r/x['path'];assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],x['path']\nmodel=json.loads((r/'MODEL_ENVIRONMENT_IDENTITY.json').read_text())\nfor x in model['model_files']:\n p=pathlib.Path(model['model_path'])/x['name'];assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],x['name']\nm=json.loads((r/'srf12_source_manifest.json').read_text())\nregister_amendment(r,build_config(),source_commit=m['git_commit'])\nimmutable_json(r/'INPUT_IDENTITY.json',dict(status='PASS',files=inputs['files'],model_files=model['model_files'],model_path=model['model_path'],model_weights_hash=model['model_weights_hash'],source_commit=m['git_commit']))\nimmutable_json(r/'deployment/CPU_VALIDATION_COMPLETE.json',dict(status='PASS',input_files=len(inputs['files']),model_files=len(model['model_files']),source_commit=m['git_commit']))\nprint('INPUT_MODEL_REGISTRATION_PASS')\n''')
s=d/'validate.sbatch'
s.write_text('#!/bin/bash\nset -euo pipefail\nexport CUDA_VISIBLE_DEVICES=""\nexport PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\nexport PYTHONPATH='+str(c/'src')+'\ncd '+str(c)+'\n'+py+' -m pytest tests/sr_f12 tests/sr_f1/test_contract_boundary.py tests/sr_f1/test_json_protocol.py -q\n'+py+' '+str(verify)+'\n')
a=['sbatch','--parsable','--account=rose','--partition=cluster02','--qos=override-limits-but-killable','--cpus-per-task=4','--mem=12G','--time=30','--job-name=srf12-cpu-validation','--comment=srf12-cpu-validation-3542869','--output='+str(d/'cpu-%j.log'),str(s)]
save('CPU_SUBMIT_INTENT.json',dict(args=a))
p=subprocess.run(a,capture_output=True,text=True)
save('CPU_SUBMISSION.json',dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr));print(p.stdout,p.stderr)
assert p.returncode==0
