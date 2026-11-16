#!/usr/bin/env python3
"""Read-only observation of one explicitly selected trained DEV Slurm job."""
import argparse
import datetime
import json
from pathlib import Path
import shlex
import subprocess

REMOTE = r'''
import datetime,json,pathlib,subprocess,sys
job,start,arm,step=json.load(sys.stdin)
base=pathlib.Path('/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/prepared_data_v1')
assert base.resolve(strict=True)==base
slot=base/'trained_dev'/f'start-{start}'/arm/f'step-{step:03d}'
def run(args):
 p=subprocess.run(args,capture_output=True,text=True)
 return dict(args=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
def read(path):
 assert path.resolve(strict=True)==path and not path.is_symlink()
 try:return json.loads(path.read_bytes())
 except (ValueError,OSError) as error:return dict(observation_error=type(error).__name__,message=str(error))
endpoint=slot/'endpoint'
out=dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),job_id=job,
 selection=dict(start_seed=start,arm=arm,step=step),exists=slot.exists(),
 sacct=run(['sacct','-j',str(job),'-n','-P','--format=JobID,State,Elapsed,ExitCode,NodeList']),
 squeue=run(['squeue','-j',str(job),'-h','-o','%i|%T|%M|%l|%q|%R']),
 generation_counts={suffix:len(list(endpoint.glob('*.'+suffix+'.json'))) for suffix in ('BEGIN','CAPTURE','COMPLETE')},
 records={},global_incidents={})
for name in ('CLAIM.json','RESULT.json','HOLD.json','ERROR.json','POST_RISK.json'):
 p=slot/name
 if p.exists():out['records'][name]=read(p)
for name in ('HOLD.json','ERROR.json','training_control/HOLD.json','training_control/ERROR.json'):
 p=base/name
 if p.exists():out['global_incidents'][name]=read(p)
logs=base.parent/'scheduling/trained_dev_71018e5'/f'start-{start}'/arm/f'step-{step:03d}'
out['log_tail']={p.name:p.read_text(errors='replace')[-3000:] for p in logs.glob(f'slurm-{job}.*') if p.is_file() and not p.is_symlink()}
print(json.dumps(out,allow_nan=False))
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id',type=int,required=True)
    parser.add_argument('--start-seed',type=int,choices=(11031,11032),required=True)
    parser.add_argument('--arm',choices=('T0','T1'),required=True)
    parser.add_argument('--step',type=int,choices=(2,8,16),required=True)
    args=parser.parse_args()
    if args.job_id<=0:parser.error('job-id must be positive')
    response=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','eee-cluster',
        'python3 -c '+shlex.quote(REMOTE)],input=json.dumps([args.job_id,args.start_seed,args.arm,args.step]),
        capture_output=True,text=True)
    if response.returncode:
        print(response.stderr);return response.returncode
    value=json.loads(response.stdout)
    tag=value['checked_at_utc'][:19].replace('-','').replace(':','')+'Z'
    path=Path('artifacts/sr_satdose_20261112/intake')/f'MONITOR_DEV_{args.job_id}_{tag}.json'
    with path.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
    display={key:item for key,item in value.items() if key!='records'}
    display['receipt']=str(path)
    display['records']={name:{key:item[key] for key in ('status','permission','responses','global_scientific_hold',
        'global_hold','hold_reasons','pending_required_dev','error_type','error','observation_error','message') if key in item}
        for name,item in value['records'].items()}
    print(json.dumps(display,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
