import datetime, hashlib, json, os, pathlib, re, subprocess
r = pathlib.Path('/projects/_ssd/varunssd/louis-ssvc/sr_f11_20261010')
inc = r / 'technical_incidents/multigpu_20261010'
py = '/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python'
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n, x):
    with (inc / n).open('x') as f:
        json.dump(x, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
def run(args):
    p = subprocess.run(args, capture_output=True, text=True, timeout=45)
    return dict(command=args, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
assert r.resolve() == r and not (r / 'STOP').exists()
assert not (inc / 'AUTO_V2_RESUME_SUBMISSION_INTENT.json').exists()
assert not (inc / 'CONTROLLER_SUBMISSION_INTENT.json').exists()
assert sha(r / 'ENGINE_MULTIGPU_REPAIR.json') == 'fdf036fc9836aa9de7b48d8978b03bb39926ac43cef113ed2b254329b61a21fd'
assert read(r / 'code/SOURCE_DEPLOYMENT.json')['source_commit'] == '22735a2def1ee1ed79d69f603e580c67e301c5dd'
verify = read(inc / 'VERIFY_SUBMISSION.json')['stdout'].strip().split(';')[0]
deploy = read(inc / 'AUTO_V2_DEPLOY_SUBMISSION.json')['stdout'].strip().split(';')[0]
assert verify == '196475' and deploy == '196474'
a = run(['sacct', '-j', deploy + ',' + verify, '--format=JobID,State,ExitCode', '-n', '-P'])
rows = {z[0]: z[1:3] for z in (line.split('|') for line in a['stdout'].splitlines())}
assert a['returncode'] == 0 and rows[verify] == ['COMPLETED', '0:0'] and rows[deploy] == ['FAILED', '1:0'], a
log = inc / 'autochain_v2_deploy_196474.log'
assert 'MULTIGPU_SOURCE_ACTIVATED_CPU_VERIFICATION_SUBMITTED' in log.read_text()
assert 'AssertionError:' in log.read_text() and "['196474', '196475']" in log.read_text()
assert read(inc / 'PREFLIGHT.log')['status'] == 'PASS'
state = read(r / 'orchestration/STATE.json')
assert state['test_sealed'] and state['tasks']['ENGINE']['status'] == 'RETRYABLE'
assert state['engine_multigpu_repair_resume']['consumed_by_attempt_id'] is None
assert len(state['tasks']['ENGINE']['attempts']) == 4
assert all(not v['attempts'] for k,v in state['tasks'].items() if k not in {'ENGINE','COMMON_START'})
q = run(['squeue', '--noheader', '--user', 'varun024', '--format=%i|%j|%T'])
assert q['returncode'] == 0 and not any(name in q['stdout'] for name in ['srf1-9ef274c9fc6a', 'srf11-controller', 'srf11-mgpu-resume-recovered']), q
for n, h in read(inc / 'OPERATION_SCRIPT_HASHES_V2.json').items(): assert sha(inc / n) == h, n
save('RESUME_CHAIN_REPAIR.json', dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(), status='VERIFIED_DEPLOYED_SOURCE_AND_COMPLETED_POSTCHECK_RESUME_ONLY', diagnosis='New job was present in squeue but not yet in sacct; the conservative dependency guard stopped before submitting the resume phase.', accounting=a, queue=q, prior_wrapper_log_sha256=sha(log), repair_sha256=sha(r/'ENGINE_MULTIGPU_REPAIR.json'), verification_receipt_sha256=sha(inc/'RESUME_AUTHORIZATION.log'), operation_script_sha256=sha(inc/'srf11_multigpu_resume.py'), production_code_changed=False, old_deployment_repeated=False))
script = inc / 'resume_recovered.sbatch'
with script.open('x') as f:
    f.write(f'#!/bin/bash\nset -euo pipefail\nexport PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\n{py} {inc}/srf11_multigpu_resume.py\n')
args = ['sbatch','--hold','--parsable','--account=rose','--qos=override-limits-but-killable','--partition=cluster02','--cpus-per-task=2','--time=30','--job-name=srf11-mgpu-resume-recovered','--output='+str(inc/'resume_recovered_%j.log'),str(script)]
save('AUTO_V3_RESUME_SUBMISSION_INTENT.json',dict(command=args))
x = run(args); save('AUTO_V3_RESUME_SUBMISSION.json',x); assert x['returncode']==0,x
job = x['stdout'].strip().split(';')[0]; assert re.fullmatch('[0-9]+',job)
y = run(['scontrol','release',job]); save('AUTO_V3_RESUME_RELEASE.json',y); assert y['returncode']==0,y
print(json.dumps(dict(status='POSTCHECK_VERIFIED_RESUME_SUBMITTED', job_id=job, prior_evidence_unchanged=True)))
