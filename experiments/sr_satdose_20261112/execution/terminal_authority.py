"""Standalone, externally pinned MAIN terminal authority; no model or writes.

Deploy this file outside the immutable TRAIN package. Original execution/runtime/
analysis modules are imported from their original paths after the complete two
source inventories are authenticated. No training source is copied or amended.
The independent terminal audit runs only in ``prepare`` before a MAIN lease;
``VerifiedTerminal`` binds its full observed evidence for subsequent evaluation.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

EXPERIMENT='SR-SATDOSE-1-20261112'
PATHS=tuple(f'{s}/{a}' for s in (11031,11032) for a in ('T0','T1'))
KEYS={'schema','root','authority','auditor','external_root','external_sources','original_train_release','audit_spec','report'}


def require(value,message):
    if not value:raise PermissionError('HOLD: '+message)


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def object_sha(value):return hashlib.sha256(canonical(value)).hexdigest()


def read(path):
    def pairs(values):
        out=dict(values);require(len(out)==len(values),'Duplicate JSON key');return out
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,
                      parse_constant=lambda value:require(False,'Nonfinite JSON'))


def sha(path):
    path=Path(path)
    require(path.is_absolute() and path.resolve(strict=True)==path and path.is_file() and not path.is_symlink(),
            'Noncanonical source/evidence file')
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def reference(value):
    require(isinstance(value,dict) and set(value)=={'path','sha256'} and isinstance(value['path'],str) and
            isinstance(value['sha256'],str) and re.fullmatch('[0-9a-f]{64}',value['sha256']), 'Malformed trusted reference')


def bound(root,value):
    reference(value);relative=Path(value['path'])
    require(not relative.is_absolute() and '..' not in relative.parts,'Evidence escapes root')
    path=Path(root)/relative;require(sha(path)==value['sha256'],'Evidence SHA mismatch');return path


def absolute(value):
    reference(value);path=Path(value['path']);require(sha(path)==value['sha256'],'Source SHA mismatch');return path


def inventory(root):
    root=Path(root);require(root.is_absolute() and root.resolve(strict=True)==root and root.is_dir(),'Noncanonical source root')
    out={}
    for parent,dirs,files in os.walk(root,followlinks=False):
        for name in dirs:
            p=Path(parent)/name;require(p.resolve(strict=True)==p and not p.is_symlink(),'Source directory symlink')
        for name in files:
            p=Path(parent)/name;out[str(p.relative_to(root))]=sha(p)
    return out


def load_source(value,name):
    """Execute authenticated current bytes, with no bytecode-cache shortcut."""
    path=absolute(value);data=path.read_bytes()
    require(hashlib.sha256(data).hexdigest()==value['sha256'] and sha(path)==value['sha256'],'Source changed while reading')
    require(name not in sys.modules,'External verifier module already loaded')
    spec=importlib.util.spec_from_file_location(name,path);require(spec and spec.loader,'Cannot load external source')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    try:exec(compile(data,str(path),'exec'),module.__dict__)
    except BaseException:sys.modules.pop(name,None);raise
    return module


def _original(root,reference):
    value=read(bound(root,reference));package=Path(value['release_root'])
    require(value.get('schema')=='sr-satdose-first-two-release-v1' and
            value.get('status')=='AUDITED_FIRST_TWO_AND_READONLY_RELOAD_ONLY' and
            value.get('experiment_id')==EXPERIMENT and value.get('evidence_root')==str(root) and
            package.is_relative_to(root) and inventory(package)==value['sources'],'Original TRAIN release identity/inventory differs')
    return value,package


def import_original(package):
    """Keep original namespace/path identity; reject any pre-imported shadow."""
    sys.dont_write_bytecode=True
    package=Path(package)
    for name,module in tuple(sys.modules.items()):
        if name in ('execution','analysis','runtime'):
            paths=list(getattr(module,'__path__',[]))
            require(paths and all(Path(p).resolve()==package/name for p in paths),'Shadowed original package '+name)
        elif name.startswith(('execution.','analysis.','runtime.')):
            actual=getattr(module,'__file__',None)
            require(actual and Path(actual).resolve()==package/(name.replace('.','/')+'.py'),'Shadowed original module '+name)
    text=str(package)
    if text not in sys.path:sys.path.insert(0,text)
    names=('execution.evaluation_runtime','execution.training_transaction','execution.training_state',
           'execution.admission_authority','execution.continuation_authority','execution.first_two_control',
           'execution.storage_ready','analysis.risk','analysis.evaluation','runtime.backend','runtime.fork_loader')
    modules={name:importlib.import_module(name) for name in names}
    for name,module in modules.items():
        require(Path(module.__file__).resolve()==package/(name.replace('.','/')+'.py'),'Imported shadow module '+name)
    return modules


class Observed:
    def __init__(self,root):
        self.root=root;self.files={};self.directories={};self.model_links={};self.model_directories={}
    def path(self,path,digest=None):
        path=Path(path);actual=sha(path)
        require(digest is None or actual==digest,'Observed evidence SHA differs')
        require(path not in self.files or self.files[path]==actual,'Evidence changed during verification')
        self.files[path]=actual;return path
    def ref(self,value):return self.path(bound(self.root,value),value['sha256'])
    def json(self,value):return read(self.ref(value))
    def tree(self,path,expected=None):
        path=Path(path);require(path.resolve(strict=True)==path and path.is_dir(),'Noncanonical protected directory')
        entries=sorted(str(p.relative_to(path)) for p in path.rglob('*'))
        require(expected is None or entries==expected,'Protected directory inventory differs')
        for name in entries:
            p=path/name;require(not p.is_symlink() and p.resolve(strict=True)==p,'Protected directory symlink')
            require(p.is_dir() or p.is_file(),'Nonordinary protected entry')
        self.directories[path]=entries
    def stable(self):
        for path,digest in self.files.items():require(sha(path)==digest,'Terminal evidence changed')
        for path,entries in self.directories.items():self.tree(path,entries)
        for path,value in self.model_links.items():
            require(path.is_symlink() and path.parent.resolve(strict=True)==path.parent and
                    str(path.readlink())==value['link'] and str(path.resolve(strict=True))==value['target'],
                    'Frozen terminal model link changed')
        for path,entries in self.model_directories.items():
            require(path.resolve(strict=True)==path and path.is_dir() and
                    sorted(str(p.relative_to(path)) for p in path.rglob('*'))==entries,'Terminal model inventory changed')


def _inputs(spec,published):
    require(set(spec)==KEYS|({'published'} if published else set()) and
            spec.get('schema')=='sr-satdose-terminal-authority-spec-v1','Unexpected terminal authority spec')
    root=Path(spec['root']);require(root.is_absolute() and root.resolve(strict=True)==root and root.is_dir(),'Noncanonical root')
    external=Path(spec['external_root'])
    require(external.is_relative_to(root) and inventory(external)==spec['external_sources'],'External source inventory differs')
    require(absolute(spec['authority'])==Path(__file__).resolve() and
            Path(spec['authority']['path']).is_relative_to(external),'Untrusted terminal authority source')
    auditor=absolute(spec['auditor']);require(auditor.is_relative_to(external),'Auditor outside external release')
    require(spec['external_sources'][str(auditor.relative_to(external))]==spec['auditor']['sha256'],'Unbound external auditor')
    original,package=_original(root,spec['original_train_release'])
    require(not external.is_relative_to(package) and not package.is_relative_to(external),'External code overlaps frozen TRAIN package')
    observed=Observed(root)
    for base,files in ((external,spec['external_sources']),(package,original['sources'])):
        observed.tree(base)
        for name,digest in files.items():observed.path(base/name,digest)
    observed.ref(spec['original_train_release'])
    audit_spec=observed.json(spec['audit_spec'])
    require(audit_spec.get('root')==str(root) and audit_spec.get('auditor')==spec['auditor'] and
            audit_spec.get('original_train_release')==spec['original_train_release'],'Terminal audit specification differs')
    modules=import_original(package)
    return root,observed,original,package,audit_spec,modules


def _candidate(spec,observed,original,audit_spec,report):
    require(report.get('schema')=='sr-satdose-terminal-independent-audit-v1' and
            report.get('status')=='TERMINAL_TRAINING_EVIDENCE_AUDITED_NOT_MAIN_ADMISSION' and
            report.get('permission')=='NO_TRAINING_OR_MAIN' and report.get('auditor')==spec['auditor'] and
            report.get('original_train_release')==spec['original_train_release'] and
            report.get('spec_sha256')==object_sha(audit_spec) and report.get('source_and_input_bytes_reverified') is True and
            type(report.get('audit_model_calls')) is int and report['audit_model_calls']==0 and
            type(report.get('audit_optimizer_calls')) is int and report['audit_optimizer_calls']==0,
            'Untrusted/incomplete independent terminal audit')
    require(all(type(report.get(k)) is int and report[k]==v for k,v in
                [('formal_optimizer_steps',96),('train_responses',12288),('dev_endpoints',14),('dev_responses',1512)]) and
            report.get('terminal_steps')=={key:24 for key in PATHS} and report.get('global_hold') is False and
            report.get('pending_required_dev')==[],'Terminal budgets or risk outcome differ')
    files=report.get('observed_files');directories=report.get('observed_directories')
    require(isinstance(files,dict) and files and isinstance(directories,dict) and directories,'Missing observed terminal evidence')
    for name,digest in files.items():observed.path(Path(name),digest)
    for name,entries in directories.items():
        require(isinstance(entries,list) and entries==sorted(set(entries)),'Invalid terminal directory observation')
        observed.tree(Path(name),entries)
    model_links=report.get('observed_model_links');model_directories=report.get('observed_model_directories')
    require(isinstance(model_links,dict) and isinstance(model_directories,dict) and model_directories,
            'Missing authenticated model link observations')
    for name,value in model_links.items():
        path=Path(name)
        require(path.is_absolute() and isinstance(value,dict) and set(value)=={'link','target'} and
                value['target'] in files,'Unbound terminal model link target')
        observed.model_links[path]=copy.deepcopy(value)
    for name,entries in model_directories.items():
        path=Path(name)
        require(path.is_absolute() and isinstance(entries,list) and entries==sorted(set(entries)),'Invalid model directory observation')
        observed.model_directories[path]=copy.deepcopy(entries)
    for name in ('admission_spec','admission','canary_spec','final_risk'):
        path=observed.ref(audit_spec[name]);require(files.get(str(path))==audit_spec[name]['sha256'],'Missing audited terminal prerequisite')
    require(set(audit_spec['trusted_commits'])==set(PATHS),'Terminal needs exactly the original four paths')
    for refs in audit_spec['trusted_commits'].values():
        require(len(refs)==24,'Terminal requires all24 committed steps')
        for value in refs:
            path=observed.ref(value);require(files.get(str(path))==value['sha256'],'COMMIT not independently audited')
    freeze=observed.root/'DESIGN_FREEZE.json';observed.path(freeze,original['design_freeze_sha256'])
    return dict(schema='sr-satdose-main-terminal-admission-v1',status='TRAINING_TERMINAL_VERIFIED_MAIN_ALLOWED',
        permission='FIXED_MAIN_MANIFEST_ONLY_NO_TRAINING',experiment_id=EXPERIMENT,
        design_freeze_sha256=original['design_freeze_sha256'],config_sha256=original['config_sha256'],
        original_train_release=copy.deepcopy(spec['original_train_release']),independent_audit=copy.deepcopy(spec['report']),
        audit_spec=copy.deepcopy(spec['audit_spec']),auditor=copy.deepcopy(spec['auditor']),authority=copy.deepcopy(spec['authority']),
        terminal_steps={key:24 for key in PATHS},formal_optimizer_steps=96,train_responses=12288,dev_responses=1512,
        all_paths_terminal=True,main_endpoint_count=14,main_response_count=13760,new_generation_calls=0,new_optimizer_calls=0)


def prepare(spec):
    """Independently repeat actual CPU audit; return candidate, never publish."""
    root,observed,original,package,audit_spec,modules=_inputs(copy.deepcopy(spec),False)
    report=observed.json(spec['report'])
    name='satdose_terminal_audit_'+spec['auditor']['sha256'][:16]
    auditor=load_source(spec['auditor'],name)
    try:actual=auditor.audit(audit_spec)
    finally:sys.modules.pop(name,None)
    require(actual==report,'Fresh independent terminal audit differs from trusted report')
    result=_candidate(spec,observed,original,audit_spec,report);observed.stable();return result


class VerifiedTerminal:
    def __init__(self,spec):
        self.spec=copy.deepcopy(spec)
        self.root,self.observed,self.original,self.package,self.audit_spec,self.modules=_inputs(self.spec,True)
        self.report=self.observed.json(self.spec['report'])
        candidate=_candidate(self.spec,self.observed,self.original,self.audit_spec,self.report)
        require(self.observed.json(self.spec['published'])==candidate,'Published terminal admission differs from actual audited evidence')
        self.receipt=copy.deepcopy(self.spec['published']);self.stable()
    def stable(self):
        for name in ('HOLD.json','ERROR.json','training_control/HOLD.json','training_control/ERROR.json',
                     'first_two_launcher/HOLD.json','first_two_launcher/ERROR.json'):
            require(not os.path.lexists(self.root/name),'Global terminal HOLD/error')
        self.observed.stable()


def verify(spec):return VerifiedTerminal(spec)
