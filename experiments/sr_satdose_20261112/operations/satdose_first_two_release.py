import pathlib,json,subprocess,shlex
art=pathlib.Path('artifacts/sr_satdose_20261112')
dep=json.loads((art/'intake/FIRST_TWO_CODE_DEPLOYMENT_71018e5_20261116.json').read_text())
adm=json.loads((art/'intake/FIRST_TWO_ADMISSION_PUBLICATION_71018e5_20261116.json').read_text())
budget=json.loads((art/'intake/FIRST_TWO_STORAGE_BUDGET_PUBLICATION_20261116.json').read_text())
risk=json.loads((art/'intake/TRAIN_RISK_CLARIFICATION_PUBLICATION_20261116.json').read_text())
execution=json.loads((art/'intake/ORIGINAL_TRAIN_EXECUTION_SOURCES_71018e5.json').read_text())
starts={str(s):json.loads((art/f'start_dev_20261116/start-{s}/INDEPENDENT_ANALYSIS_PUBLICATION.json').read_text()) for s in (11031,11032)}
release=dict(schema='sr-satdose-first-two-release-v1',status='AUDITED_FIRST_TWO_AND_READONLY_RELOAD_ONLY',permission='FOUR_PATHS_MAX_STEP_2_NO_STEP3',experiment_id='SR-SATDOSE-1-20261112',evidence_root=dep['release_root'].rsplit('/',1)[0],release_root=dep['release_root'],design_freeze_sha256='6c4b948c533fdd736e089bf3836378f4dbcb76ac66f1d480fa979eaca56c84e0',config_sha256='58841639b0f36a69ac77824bc8c683ef7fbeb5c50a41e9975b89b54880f4ecfe',sources=dep['sources'],execution_sources=execution,admission_spec=adm['spec'],admission=adm['admission'],storage_budget={k:budget[k] for k in ('path','sha256')},risk_clarification={k:risk[k] for k in ('path','sha256')},start_dev_results={s:x['result'] for s,x in starts.items()},start_dev_analyses={s:x['analysis'] for s,x in starts.items()},git_commit=dep['git_commit'])
remote=r'''
import sys,json,pathlib,hashlib,os
d=json.load(sys.stdin);root=pathlib.Path(d['evidence_root']);code=pathlib.Path(d['release_root'])
assert root.resolve(strict=True)==root and code.resolve(strict=True)==code
assert {str(p.relative_to(code)):hashlib.sha256(p.read_bytes()).hexdigest() for p in code.rglob('*') if p.is_file()}==d['sources']
refs=[d[k] for k in ('admission_spec','admission','storage_budget','risk_clarification')]+list(d['start_dev_results'].values())+list(d['start_dev_analyses'].values())
for ref in refs:
 p=root/ref['path'];assert p.resolve(strict=True)==p and p.is_relative_to(root) and hashlib.sha256(p.read_bytes()).hexdigest()==ref['sha256']
assert not (root/'first_two_control').exists() and not (root/'first_two_launcher').exists()
spec=json.loads((root/d['admission_spec']['path']).read_text());assert spec['training_execution_sources_sha256']==hashlib.sha256(json.dumps(d['execution_sources'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
path=root/'execution_releases/FIRST_TWO_RELEASE_71018e5.json';b=(json.dumps(d,sort_keys=True,indent=2)+'\n').encode()
with path.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
print(json.dumps(dict(path=str(path),sha256=hashlib.sha256(b).hexdigest(),release=d)))
'''
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','eee-cluster','python3 -c '+shlex.quote(remote)],input=json.dumps(release).encode(),capture_output=True)
if r.returncode:print(r.stderr.decode());raise SystemExit(r.returncode)
d=json.loads(r.stdout)
with (art/'intake/FIRST_TWO_RELEASE_PUBLICATION_71018e5_20261116.json').open('x') as f:json.dump(d,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in d.items() if k!='release'}))
