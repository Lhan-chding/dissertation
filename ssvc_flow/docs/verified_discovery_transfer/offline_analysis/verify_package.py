#!/usr/bin/env python3
"""Verify the extracted analysis package without running experiment code."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
manifest=json.loads((root/'PAYLOAD_SHA256.json').read_text())
for name, expected in manifest['files'].items():
    rel=Path(name)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Unsafe manifest path: '+name)
    p=root/rel
    if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(root):
        raise ValueError('Invalid payload file: '+name)
    if p.stat().st_size!=expected['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=expected['sha256']:
        raise ValueError('Payload mismatch: '+name)
print('PASS: '+str(len(manifest['files']))+' payload files; technical integrity only, not scientific superiority')
