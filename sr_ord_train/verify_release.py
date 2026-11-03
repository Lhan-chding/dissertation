"""Check the release byte inventory; operating outputs belong outside the release."""
import hashlib,json,sys
from pathlib import Path,PurePosixPath
BASE=Path(__file__).resolve().parent
def main():
    m=json.loads((BASE/'RELEASE.json').read_text())
    for f,s in m['members'].items():
        q=PurePosixPath(f)
        if q.is_absolute() or '..' in q.parts:raise ValueError('unsafe release path')
        p=BASE/q
        if not p.is_file() or p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest()!=s:raise ValueError('changed/missing: '+f)
    print('PASS registered package bytes:',len(m['members']),'files; no model action')
if __name__=='__main__':main()
