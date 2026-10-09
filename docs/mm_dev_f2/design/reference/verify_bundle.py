"""Verify a delivered MM-DEV plan package. No network or model execution."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(root: Path) -> dict:
    root = root.resolve()
    manifest = json.loads((root / 'BUNDLE_MANIFEST.json').read_text(encoding='utf-8'))
    expected = set()
    errors = []
    for item in manifest['files']:
        rel = PurePosixPath(item['path'])
        if rel.is_absolute() or '..' in rel.parts:
            errors.append({'path': item['path'], 'reason': 'unsafe path'})
            continue
        name = str(rel)
        if name in expected:
            errors.append({'path': name, 'reason': 'duplicate manifest entry'})
            continue
        expected.add(name)
        path = root / name
        if not path.is_file() or path.is_symlink():
            errors.append({'path': name, 'reason': 'missing or symlink'})
        elif path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']:
            errors.append({'path': name, 'reason': 'length or SHA mismatch'})
    actual = {
        p.relative_to(root).as_posix()
        for p in root.rglob('*')
        if p.is_file() and p.name != 'BUNDLE_MANIFEST.json'
        and '__pycache__' not in p.parts and p.suffix != '.pyc'
    }
    for name in sorted(actual - expected):
        errors.append({'path': name, 'reason': 'unexpected file'})
    return {'status': 'PASS' if not errors else 'FAIL', 'verified_files': len(expected),
            'errors': errors, 'model_calls': 0, 'server_checkpoint_verification': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle_root', type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.bundle_root)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f'Bundle verification could not finish: {exc}\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)
