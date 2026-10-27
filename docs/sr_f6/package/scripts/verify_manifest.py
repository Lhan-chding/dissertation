"""Verify all distributable file contents. No model or server calls."""
import hashlib,json,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
m=json.loads((root/'MANIFEST_SHA256.json').read_text());errors=[]
for row in m['files']:
 p=root/row['path']
 if not p.is_file():errors.append('MISSING '+row['path'])
 elif hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:errors.append('HASH '+row['path'])
print(json.dumps({'status':'PASS' if not errors else 'FAIL','checked_files':len(m['files']),'errors':errors},indent=2))
sys.exit(1 if errors else 0)
