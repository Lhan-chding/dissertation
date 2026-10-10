import datetime,json,os,pathlib,re,subprocess
r=pathlib.Path('/projects/_ssd/varunssd/louis-ssvc/sr_f11_20261010');inc=r/'technical_incidents/multigpu_20261010';qos='soujanya-poria-startfund-2026-03'
def read(p):return json.loads(p.read_text())
def save(n,x):
 with (inc/n).open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def run(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=30)
 return dict(command=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,time=datetime.datetime.now(datetime.timezone.utc).isoformat())
def fields(x):return dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S*)',x['stdout']))
def tres(s):return dict(i.split('=',1) for i in s.split(',') if '=' in i)
job=read(inc/'GPU_PROBE_SUBMISSION.json')['stdout'].strip().split(';')[0];assert job=='196459'
assert not (inc/'GPU_PROBE_RELEASE.json').exists()
x=run(['scontrol','show','job',job,'--oneliner']);f=fields(x);t=tres(f['ReqTRES'])
assert x['returncode']==0 and f['JobId']==job and f['UserId'].split('(')[0]=='varun024' and f['Account']=='rose' and f['QOS']==qos and f['NumNodes'] in {'1','1-1'} and f['Features']=='highmem' and f['JobState']=='PENDING' and f['Reason']=='JobHeldUser'
assert t['node']=='1' and t['cpu']=='12' and t['mem']=='270G' and t['gres/gpu']=='3' and t['gres/gpu:pro6000']=='3' and f['Command']==str(inc/'gpu_probe.sbatch')
queue=run(['squeue','--noheader','--user','varun024','--format=%i|%q']);assert queue['returncode']==0
jobs={}
for row in (line.split('|') for line in queue['stdout'].splitlines()):
 if row[1]!=qos:continue
 obs=run(['scontrol','show','job',row[0],'--oneliner']);z=fields(obs);assert obs['returncode']==0 and z['QOS']==qos and z['UserId'].split('(')[0]=='varun024'
 values=tres(z['ReqTRES']);assert 'gres/gpu' in values;jobs[row[0]]=int(values['gres/gpu'])
assert sum(jobs.values())<=5 and jobs[job]==3,jobs
save('GPU_PROBE_ALLOCATION_CHECK_SINGLE_NODE_RANGE.json',dict(observation=x,single_node_range='1-1_ACCEPTED_AS_EXACTLY_ONE',req_node_verified=1,teacher_qos_reservations=jobs,queue=queue,prior_guard_rejection_preserved=True))
x=run(['scontrol','release',job]);save('GPU_PROBE_RELEASE.json',x);assert x['returncode']==0,x
print(json.dumps(dict(status='VERIFIED_PROBE_RELEASED',job_id=job,teacher_qos_total_reserved=sum(jobs.values()))))
