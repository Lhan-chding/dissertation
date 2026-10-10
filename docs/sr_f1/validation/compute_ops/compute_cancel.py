import importlib.util,json,subprocess,sys
from pathlib import Path
r=Path('/projects/_ssd/varunssd/louis-ssvc/sr_f11_20261010'); inc=r/'technical_incidents/compute_parallel_20261010'; c=r/'code_engine_compute_candidate_20261010'
spec=importlib.util.spec_from_file_location('maintenance',c/'scripts/sr_f1/compute_parallel_maintenance.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
receipt=m.read(inc/'CANDIDATE_CPU_SUBMISSION.json'); cpu=receipt['stdout'].strip().split(';')[0]
a=m.command(['sacct','-j',cpu,'-n','-P','--format=JobIDRaw,State,ExitCode'])
m.require(a['returncode']==0 and [cpu,'COMPLETED','0:0'] in [line.split('|')[:3] for line in a['stdout'].splitlines()], 'CANDIDATE_CPU_NOT_COMPLETED')
p=m.read(inc/'PRE_MAINTENANCE_SNAPSHOT.json');m.verify_preservation(r,p,destination='pre_maintenance')
identity=m.identities(r)
for role in ['controller','gpu']:
 m.require(identity[role]['job_id']==p['identities'][role]['job_id'],'TARGET_CHANGED')
 m.require(identity[role]['fields']['JobState']=='RUNNING','TARGET_NOT_RUNNING')
ids=[identity[k]['job_id'] for k in ['controller','gpu']]
m.require(ids==['196654','196508'],'EXPECTED_TARGETS_CHANGED')
intent=inc/'CANCEL_INTENT.json';m.require(not intent.exists(),'CANCEL_INTENT_ALREADY_EXISTS_RECONCILE')
m.save(intent,dict(authorized_user_request='不改变实验设置的前提下，让多张卡真正并行计算提速',identities=identity,job_ids=ids,unrelated_jobs_modified=0))
result=m.command(['scancel',*ids]);m.save(inc/'CANCEL_RESULT.json',result);print(json.dumps(result));m.require(result['returncode']==0,'CANCEL_RESULT_UNKNOWN')
