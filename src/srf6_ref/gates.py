"""Artifact-bound launch checks. Templates and CPU-only receipts cannot unlock GPUs."""
from __future__ import annotations
import hashlib,json
from pathlib import Path

STAGES=('SOURCE_BINDING','DATA_AND_MEASUREMENT','PARENT_CENSUS','RUNTIME_PILOT','FORMAL_FREEZE')

def file_hash(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()

def launch_check(directory:Path,config_path:Path)->dict:
    errors=[];digest=file_hash(config_path)
    for stage in STAGES:
        p=directory/f'{stage}.json'
        if not p.exists():errors.append(f'missing {stage}');continue
        rec=json.loads(p.read_text())
        if rec.get('status')!='PASS':errors.append(f'{stage}: not PASS')
        if rec.get('config_sha256')!=digest:errors.append(f'{stage}: wrong config identity')
        if rec.get('execution_origin')!='SERVER_ACTUAL':errors.append(f'{stage}: not actual server evidence')
        ev=rec.get('evidence',[])
        if not ev:errors.append(f'{stage}: no evidence')
        for e in ev:
            ep=directory/e['path']
            if not ep.is_file() or file_hash(ep)!=e.get('sha256'):errors.append(f'{stage}: evidence missing or hash mismatch')
    return {'allowed':not errors,'errors':errors,'purpose':'local consistency guard, not a cryptographic proof of experiment truth'}
