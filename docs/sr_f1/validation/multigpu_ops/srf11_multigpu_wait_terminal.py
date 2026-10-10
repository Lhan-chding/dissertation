import json,pathlib,subprocess,time
inc=pathlib.Path('/projects/_ssd/varunssd/louis-ssvc/sr_f11_20261010/technical_incidents/multigpu_20261010')
assert json.loads((inc/'MAINTENANCE_CANCEL_RESULT.json').read_text())['returncode']==0
for index in range(300):
 q=subprocess.run(['squeue','--noheader','--user','varun024','--format=%i|%T'],capture_output=True,text=True,timeout=30)
 assert q.returncode==0,q.stderr
 if not any(row.split('|')[0] in {'196225','196227'} for row in q.stdout.splitlines()):
  a=subprocess.run(['sacct','-j','196225,196227','--format=JobID,State,ExitCode','-n','-P'],capture_output=True,text=True,timeout=30)
  assert a.returncode==0,a.stderr
  rows={row.split('|')[0]:row.split('|')[1:] for row in a.stdout.splitlines()}
  if all(rows.get(job,[''])[0].startswith('CANCELLED') for job in ['196225','196227']):
   print(json.dumps(dict(status='TERMINAL_CONFIRMED',accounting=a.stdout,queue=q.stdout)),flush=True);break
 if index%15==0:print(json.dumps(dict(status='WAITING_FOR_OLD_TERMINAL',check=index,queue=q.stdout)),flush=True)
 time.sleep(2)
else:raise TimeoutError('Old allocations still active; no deployment or duplicate submission')
