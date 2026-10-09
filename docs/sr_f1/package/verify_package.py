"""Check declared SHA-256 hashes without network or model access."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
manifest=json.loads((root/'PACKAGE_SHA256.json').read_text())
bad=[]
for name,digest in manifest.items():
    path=root/name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:bad.append(name)
print(json.dumps({'checked_files':len(manifest),'mismatches':bad,'model_training_verified':False},ensure_ascii=False,indent=2))
raise SystemExit(bool(bad))
