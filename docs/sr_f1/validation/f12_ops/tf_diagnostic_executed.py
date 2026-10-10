#!/usr/bin/env python3
"""Read-only numerical replay of retained preflight tokens; no generate or Adam."""
import argparse
import gc
import json
import math
import time
from pathlib import Path

import torch
from sr_f12.runner import load_real_runtime, read_json, write_once
from sr_f12.protocol import validate_config
from sr_f1.contract import digest

p=argparse.ArgumentParser()
p.add_argument('--run-root',required=True,type=Path)
p.add_argument('--output',required=True,type=Path)
a=p.parse_args()
plan=validate_config(a.run_root/'technical/PREFLIGHT_PLAN.json')
paths=sorted((a.run_root/'technical/preflight/rollouts').glob('*.json'))
groups=[read_json(p) for p in paths]
if len(groups)!=4 or any(g['sha256']!=digest(g['records']) for g in groups):
    raise PermissionError('Retained 32-record preflight evidence is incomplete or changed')
records=[r for g in groups for r in g['records']][:16]
runtime,optimizer,scheduler,modules=load_real_runtime(plan,a.run_root,task_id='technical_tf_readonly_diagnostic')
runtime.model.eval()
optimizer.zero_grad(set_to_none=True)
result={'status':'RUNNING','operation':'NO_GENERATION_NO_OPTIMIZER_UPDATE',
        'records_hash':digest(records),'records_count':len(records),'modules':modules,
        'gpu':torch.cuda.get_device_name(0),'single_rows':[],'microbatches':[]}
start=time.perf_counter()
singles=[]
for i,r in enumerate(records):
    v=runtime.batch_sequence_forward([r],purpose='diagnostic_single',grad=False)[0].detach().float().cpu()
    singles.append(v)
    d=(v-torch.tensor(r['sampler_logprobs'])).abs()
    result['single_rows'].append({'index':i,'qid':r['qid'],'group_row_index':r['group_row_index'],
        'tokens':len(r['tokens']),'sampler_difference_mean':float(d.mean()),
        'sampler_difference_max':float(d.max()),'single_logprobs':v.tolist()})
    print(json.dumps({'single_done':i+1,'seconds':time.perf_counter()-start}),flush=True)
for size in (1,2,4):
    row={'microbatch_size':size,'status':'RUNNING','per_row':[]}
    diffs=[]
    sampler=[]
    clipping=[]
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        began=time.perf_counter()
        for offset in range(0,16,size):
            batch=records[offset:offset+size]
            values=runtime.batch_sequence_forward(batch,purpose=f'diagnostic_batch{size}',grad=False)
            for j,(r,v) in enumerate(zip(batch,values,strict=True)):
                idx=offset+j
                v=v.detach().float().cpu()
                delta=(v-singles[idx]).abs()
                sd=v-torch.tensor(r['sampler_logprobs'])
                diffs.append(delta)
                sampler.append(sd.abs())
                clipping.append(sd>math.log(2))
                positions=torch.topk(delta,min(5,len(delta))).indices.tolist()
                row['per_row'].append({'index':idx,'qid':r['qid'],'group_row_index':r['group_row_index'],
                    'length':len(r['tokens']),'batch_max_length':max(len(x['tokens']) for x in batch),
                    'mean':float(delta.mean()),'max':float(delta.max()),
                    'largest':[{'position':k,'token':r['tokens'][k], 'difference':float(delta[k]),
                        'single':float(singles[idx][k]),'batch':float(v[k])} for k in positions]})
            del values
        d=torch.cat(diffs)
        s=torch.cat(sampler)
        row.update(status='COMPLETE',mean=float(d.mean()),p99=float(torch.quantile(d,.99)),
            maximum=float(d.max()),token_count=d.numel(),sampler_mean=float(s.mean()),
            sampler_p99=float(torch.quantile(s,.99)),sampler_maximum=float(s.max()),
            tis_clipped_fraction=float(torch.cat(clipping).float().mean()),
            seconds=time.perf_counter()-began,peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except torch.cuda.OutOfMemoryError as exc:
        row.update(status='CUDA_OOM',error=str(exc))
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
    result['microbatches'].append(row)
    print(json.dumps({k:v for k,v in row.items() if k!='per_row'}),flush=True)
    write_once(a.output.with_name(a.output.stem+f'-mb{size}.json'),row)
result['status']='COMPLETE'
result['seconds']=time.perf_counter()-start
result['optimizer_state_count']=len(optimizer.state)
if optimizer.state:
    raise RuntimeError('Read-only diagnostic unexpectedly consumed an optimizer update')
write_once(a.output,result)
print(json.dumps({'output':str(a.output),'status':'COMPLETE'}),flush=True)
