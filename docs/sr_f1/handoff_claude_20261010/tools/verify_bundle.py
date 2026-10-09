#!/usr/bin/env python3
"""Verify every payload and checksum offline with only Python's standard library."""
import hashlib,json,sys
from pathlib import Path,PurePosixPath

def verify(root):
    root=Path(root).resolve();manifest=json.loads((root/'MANIFEST.json').read_text());expected={}
    for n,row in manifest['files'].items():
        part=PurePosixPath(n);assert not part.is_absolute() and '..' not in part.parts,n
        p=root/n;assert p.is_file() and not p.is_symlink(),n
        b=p.read_bytes();h=hashlib.sha256(b).hexdigest();assert len(b)==row['bytes'] and h==row['sha256'],n
        expected[n]=h
    expected['MANIFEST.json']=hashlib.sha256((root/'MANIFEST.json').read_bytes()).hexdigest()
    sums={}
    for line in (root/'SHA256SUMS.txt').read_text().splitlines():
        h,n=line.split('  ',1);assert n not in sums,n;sums[n]=h
    assert sums==expected,'Checksum list differs from manifest'
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    assert actual==set(expected)|{'SHA256SUMS.txt'},'Missing or extra payload'
    return {'status':'PASS','payload_files':len(manifest['files']),'manifest_and_checksum_verified':True}

if __name__=='__main__':print(json.dumps(verify(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[1]),indent=2))
