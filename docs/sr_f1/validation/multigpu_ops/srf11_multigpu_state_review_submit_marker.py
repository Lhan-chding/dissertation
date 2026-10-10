import datetime,json,os,pathlib
p=pathlib.Path('/projects/_ssd/varunssd/louis-ssvc/sr_f11_20261010/technical_incidents/multigpu_20261010/STATE_REVIEW_SUBMISSION.json')
job=os.environ['SLURM_JOB_ID'];assert job.isdigit()
with p.open('x') as f:json.dump(dict(returncode=0,stdout=job+'\n',stderr='',command=['CURRENT_CPU_MAINTENANCE_JOB_STATE_REVIEW'],time=datetime.datetime.now(datetime.timezone.utc).isoformat()),f,indent=2);f.write('\n')
