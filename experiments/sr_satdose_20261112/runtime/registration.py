"""CPU source binding and immutable pre-model freeze; no model imports or calls."""
from __future__ import annotations
import argparse
import datetime
import json
from pathlib import Path
from .backend import file_sha, read_json, require, write_once


def build_source_lock(source_bindings, output):
    pins = read_json(source_bindings)
    legacy = Path(pins['remote_package_root']).resolve(strict=True)
    binding_path = legacy / 'execution/RUN_BINDING.json'
    require(file_sha(binding_path) == pins['legacy_binding_sha256'], 'Historical binding differs')
    binding = read_json(binding_path)
    native = Path(pins['remote_native_code_root']) / 'src'
    native_files = {}
    for item in binding['source_files']:
        path = Path(item['path'])
        require(path.resolve().is_relative_to(native), 'Historical source escaped native root')
        require(file_sha(path) == item['sha256'], 'Native file changed: ' + str(path))
        native_files[str(path.relative_to(native))] = item['sha256']
    require(set(native_files) == {str(p.relative_to(native)) for p in native.rglob('*.py')}, 'Incomplete native inventory')
    old_path = Path(binding['historical_source_binding'])
    require(file_sha(old_path) == binding['historical_source_binding_sha256'], 'Historical model binding differs')
    old = read_json(old_path)
    model = Path(binding['model']['model_path'])
    model_files = {}
    for item in old['model_files']:
        path = Path(item['path'])
        rel = path.relative_to(model)
        require(file_sha(path) == item['sha256'], 'Model bytes changed: ' + str(rel))
        model_files[str(rel)] = item['sha256']
    require(set(model_files) == {str(p.relative_to(model)) for p in model.rglob('*') if p.is_file()}, 'Incomplete model inventory')
    parent = Path(binding['model']['parents']['b0'])
    parent_files = {name: file_sha(parent / name) for name in ('adapter_config.json', 'adapter_model.safetensors')}
    # These pins are independently checked by the source/tensor audit as well.
    require(parent_files == {'adapter_config.json':'b4cd04ac2482b3615ccdbba959aa129201ea321b9e59e66245bdd80f41535568',
        'adapter_model.safetensors':'501e9ff25dde77ab5c2042dba82236f74e8a39c5df89b5fb3ed1d2901314e5a0'}, 'Original b0 files differ')
    result = dict(native_src=str(native), legacy_root=str(legacy), native_files=native_files,
        legacy_files={str(p.relative_to(legacy)): file_sha(p) for p in sorted(legacy.rglob('*.py'))},
        snapshot=binding['model']['snapshot'], model_path=str(model), model_files=model_files,
        parent_path=str(parent), parent_files=parent_files,
        historical_binding_sha256=file_sha(binding_path), model_calls=0)
    write_once(Path(output), result)
    return result


def freeze_design(root, config_path, visual_receipt, source_lock, code_dir, code_commit):
    root, code_dir = Path(root).resolve(strict=True), Path(code_dir).resolve(strict=True)
    config_path, visual_receipt, source_lock = map(lambda p: Path(p).resolve(strict=True), (config_path, visual_receipt, source_lock))
    require(len(code_commit) == 40 and all(c in '0123456789abcdef' for c in code_commit), 'Exact science commit required')
    require(all(p.is_relative_to(root) for p in (config_path, visual_receipt, source_lock, code_dir)), 'Freeze inputs must be local')
    preparation = read_json(root / 'PREPARATION.json')
    require(preparation['images_rendered'] is True and preparation['model_calls'] == 0, 'Rendered zero-call preparation required')
    require(preparation['config_sha256'] == file_sha(config_path), 'Preparation config changed')
    for name, expected in preparation['files'].items():
        require(file_sha(root / name) == expected, 'Prepared artifact changed: ' + name)
    images = [json.loads(s) for s in (root / 'images.jsonl').read_text().splitlines()]
    for row in images:
        p = root / row['image_file']
        require(p.resolve().is_relative_to(root) and not p.is_symlink(), 'Image outside freeze')
        require(row['image_structure_pass'] is True and row['metadata_empty'] is True and file_sha(p) == row['png_sha256'], 'Image verification failed')
    panel = [json.loads(s) for s in (root / 'visual_review_manifest.jsonl').read_text().splitlines()]
    require(len(panel) == 32 and len({r['image_file'] for r in panel}) == 32, 'Visual panel must contain 32 distinct images')
    review = read_json(visual_receipt)
    require(review['status'] == 'PASS' and review['reviewed_images'] == 32 and
            review['manifest_sha256'] == file_sha(root / 'visual_review_manifest.jsonl'), 'Missing fixed-panel visual review')
    require(review['reviewer_kind'] == 'human' or
            (review['reviewer_kind'] == 'AI' and review.get('user_authorized_substitution') is True and review.get('authorization_quote')),
            'Manual review or explicit user-authorized substitution required')
    require(not any(root.rglob('*.BEGIN.json')) and not (root / 'runs').exists(), 'Model attempt already exists')
    required = ['calibration_groups', 'private_cases', 'training_groups', 'inputs', 'images', 'visual_review_manifest', 'structural_rejections']
    required += [phase + '_manifest' for phase in ('train', 'dry', 'dev', 'prefix_calib', 'main')]
    artifacts = {name: {'path': name + '.jsonl', 'sha256': file_sha(root / (name + '.jsonl'))} for name in required}
    for name, path in [('preparation',root/'PREPARATION.json'), ('probe_splits',root/'PROBE_SPLITS.json'),
                       ('visual_review',visual_receipt),('source_lock',source_lock),('config',config_path)]:
        artifacts[name] = dict(path=str(path.relative_to(root)), sha256=file_sha(path))
    for p in sorted(code_dir.rglob('*.py')):
        artifacts['code:' + str(p.relative_to(code_dir))] = dict(path=str(p.relative_to(root)),sha256=file_sha(p))
    require(any(k.startswith('code:') for k in artifacts), 'No implementation files')
    result = dict(schema='sr-satdose-design-freeze-v1', experiment_id=read_json(config_path)['experiment_id'],
        config_sha256=file_sha(config_path), code_commit=code_commit,
        frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), artifacts=artifacts,
        model_calls_before_freeze=0, generation_budget=28936,
        runtime_sources={str(p.relative_to(code_dir)): file_sha(p) for p in sorted(code_dir.rglob("*.py"))})
    write_once(root/'DESIGN_FREEZE.json', result)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    a=sub.add_parser('source-lock');a.add_argument('--bindings',required=True);a.add_argument('--out',required=True)
    a=sub.add_parser('freeze')
    for n in ('root','config','visual-receipt','source-lock','code-dir','code-commit'):a.add_argument('--'+n,required=True)
    args=ap.parse_args()
    if args.command=='source-lock': build_source_lock(args.bindings,args.out)
    else: freeze_design(args.root,args.config,args.visual_receipt,args.source_lock,args.code_dir,args.code_commit)

if __name__=='__main__':main()
