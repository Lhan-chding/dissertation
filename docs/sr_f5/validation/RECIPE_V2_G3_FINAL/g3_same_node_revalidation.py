import fcntl,json,socket,datetime
from pathlib import Path
from sr_f5.pilot import compare_recovery,aggregate,file_metadata
from sr_f5.calibration import digest
r=Path("/projects/_ssd/varunssd/louis-ssvc/sr_f5_semantic_j_recipe_v2_20261026")
d=r/"CONTINUATION/G3_SAME_NODE_REVALIDATION"
d.mkdir(exist_ok=False)
with (r/"CONTINUATION/G3_AGGREGATION_RUN.lock").open("a") as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (r/"G3/METRICS.json").exists()
 assert json.loads((r/"GATES/G3.json").read_text())["status"]=="NOT_RUN"
 evidence={"host":socket.gethostname(),"at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"reason":"host-local st_dev binding; exact frozen recovery recomputation on aggregation node", "source_revision":"47d5e85fb0c9b39d8f996465045d21956825f59d", "arms":{}}
 for a in ("J","SHUF","DENSE","LOCAL","HARD"):
  p=r/"G3/RUNS/b0"/a/"RECOVERY.json"
  raw=p.read_bytes();old=json.loads(raw)
  (d/(a+"_RECOVERY_ORIGINAL.json")).write_bytes(raw)
  before={}
  for lane in ("primary","recovery"):
   base=p.parent/lane
   current=file_metadata(base/"checkpoints/step-012/state.pt")
   assert all(current[k]==old["checkpoint_metadata"][lane][k] for k in ("size","mtime_ns","inode"))
   assert digest(json.loads((base/"COMPLETE.json").read_text()))==old["completion_digests"][lane]
   before[lane]=current
  new=compare_recovery(r,"b0",a)
  assert new["status"]=="PASS" and all(new["checks"].values())
  assert {k:v for k,v in new.items() if k!="checkpoint_metadata"}=={k:v for k,v in old.items() if k!="checkpoint_metadata"}
  evidence["arms"][a]={"original_metadata":old["checkpoint_metadata"],"current_metadata":before,"status":new["status"],"checks":new["checks"],"completion_digests":new["completion_digests"]}
  (d/(a+"_RECOVERY_REVALIDATED.json")).write_text(json.dumps(new,indent=2))
  (d/"DIAGNOSIS.json").write_text(json.dumps(evidence,indent=2))
 result=aggregate(r)
 print(json.dumps(result,indent=2),flush=True)
