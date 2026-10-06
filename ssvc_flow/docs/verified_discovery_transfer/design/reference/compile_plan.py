"""Compile metadata only. Does not generate model tasks or invoke any model."""
from pathlib import Path
import json
from contracts import workload, planned_jobs

ROOT=Path(__file__).resolve().parents[1]
config=json.loads((ROOT/'protocol.json').read_text())
(ROOT/'workload.json').write_text(json.dumps(workload(config),ensure_ascii=False,indent=2)+'\n')
(ROOT/'templates'/'run_matrix.json').write_text(json.dumps({
    'status':'PLANNED_ONLY',
    'not_a_server_queue_yet':True,
    'data_views_and_checkpoint_availability_must_be_resolved':True,
    'jobs':planned_jobs(config)
},ensure_ascii=False,indent=2)+'\n')
print('Compiled workload.json and planned run_matrix.json; no model calls.')
