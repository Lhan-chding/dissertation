import pathlib,json,subprocess,shlex,sys,base64,hashlib
start=int(sys.argv[1]);assert start in (11031,11032)
art=pathlib.Path('artifacts/sr_satdose_20261112');dep=json.loads((art/'intake/FIRST_TWO_CODE_DEPLOYMENT_71018e5_20261116.json').read_text())
remote=r'''
import pathlib,json,sys,hashlib,base64,os,datetime
d=json.load(sys.stdin);start=d['start'];dep=d['deployment'];code=pathlib.Path(dep['release_root']);root=code.parent
assert start in (11031,11032) and code.resolve(strict=True)==code and root.resolve(strict=True)==root
assert {str(p.relative_to(code)):hashlib.sha256(p.read_bytes()).hexdigest() for p in code.rglob('*') if p.is_file()}==dep['sources']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bound(ref):
 p=root/ref['path'];assert p.resolve(strict=True)==p and p.is_relative_to(root) and sha(p)==ref['sha256'];return p
path=root/f'start_dev/start-{start}/RESULT.json';b=path.read_bytes();result=json.loads(b)
assert result['status']=='START_DEV_ENDPOINT_COMPLETE' and result['responses']==108 and result['start_seed']==start
assert not any(p.name in ('HOLD.json','ERROR.json') or p.is_symlink() for p in path.parent.rglob('*'))
assert sha(root/'DESIGN_FREEZE.json')=='6c4b948c533fdd736e089bf3836378f4dbcb76ac66f1d480fa979eaca56c84e0'
sys.path.insert(0,str(code));from analysis.evaluation import analyze_completed_endpoint
freeze=json.loads((root/'DESIGN_FREEZE.json').read_text())
analysis=analyze_completed_endpoint(bound(freeze['artifacts']['config']),root/'DESIGN_FREEZE.json',root,bound(result['receipt']),expected_receipt_sha256=result['receipt']['sha256'],endpoint_binding_path=bound(result['endpoint_binding']),expected_endpoint_binding_sha256=result['endpoint_binding']['sha256'])
assert analysis['summary']['metrics']['assigned']==108 and sha(path)==hashlib.sha256(b).hexdigest()
out=root/f'execution_releases/START_DEV_ANALYSIS_71018e5_{start}.json';data=(json.dumps(analysis,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
with out.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
print(json.dumps(dict(status='START_DEV_INDEPENDENT_ANALYSIS_COMPLETE_NO_TRAINING_RELEASE',start_seed=start,completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),analysis=dict(path=str(out.relative_to(root)),sha256=hashlib.sha256(data).hexdigest()),result=dict(path=str(path.relative_to(root)),sha256=hashlib.sha256(b).hexdigest()),analysis_code_root=str(code),analysis_bytes_b64=base64.b64encode(data).decode(),result_bytes_b64=base64.b64encode(b).decode())))
'''
cmd='env PYTHONDONTWRITEBYTECODE=1 /projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python -B -c '+shlex.quote(remote)
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','eee-cluster',cmd],input=json.dumps(dict(start=start,deployment=dep)).encode(),capture_output=True)
if r.returncode:print(r.stderr.decode());print(r.stdout.decode());raise SystemExit(r.returncode)
d=json.loads(r.stdout);folder=art/f'start_dev_20261116/start-{start}';folder.mkdir(parents=True,exist_ok=True)
for key in ('analysis','result'):
 b=base64.b64decode(d.pop(key+'_bytes_b64'));assert hashlib.sha256(b).hexdigest()==d[key]['sha256']
 with (folder/(key.upper()+'.json')).open('xb') as f:f.write(b)
with (folder/'INDEPENDENT_ANALYSIS_PUBLICATION.json').open('x') as f:json.dump(d,f,indent=2);f.write('\n')
summary=json.loads((folder/'ANALYSIS.json').read_text())['summary']
print(json.dumps(d));print(json.dumps({k:v for k,v in summary.items() if k not in ('analysis_sources','scorer_sources','evidence')}))
