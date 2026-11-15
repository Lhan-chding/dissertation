"""Independent terminal evidence-boundary tests; no production audit fixture.

Filesystem/hash/lock checks are real. Chunk tests isolate the already-tested
original release verifier; synthetic reports here never certify GPU/training.
"""
import copy
import fcntl
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
loader=importlib.util.spec_from_file_location('satdose_terminal_audit_test',PACKAGE/'execution/terminal_audit.py')
a=importlib.util.module_from_spec(loader);loader.loader.exec_module(a)
sys.path.pop(0)


def reference(name='p.json'):return dict(path=name,sha256='a'*64)


def spec():
    commits={key:[reference(f'{key}/{n}.json') for n in range(1,25)] for key in a.PATHS}
    dev={f'{s}/START/0':dict(receipt=reference(),binding=reference()) for s in (11031,11032)}
    dev.update({f'{key}/{n}':dict(receipt=reference(),binding=reference()) for key in a.PATHS for n in (2,8,16)})
    chunks={f'{key}/chunk-{lo:03d}-{hi:03d}':reference() for key in a.PATHS for lo,hi in a.CHUNKS.items()}
    return dict(schema='sr-satdose-terminal-audit-spec-v1',root='/untrusted',auditor=reference(),
        original_train_release=reference('train.json'),admission_spec=reference(),admission=reference(),canary_spec=reference('canary.json'),
        trusted_commits=commits,dev_endpoints=dev,continuation_results=chunks,final_risk=reference())


class PopulationTests(unittest.TestCase):
    def test_exact_96_commits_14_dev_12_chunks(self):a._populations(spec())

    def test_step23_not_terminal(self):
        value=spec();value['trusted_commits']['11031/T0'].pop()
        with self.assertRaises(PermissionError):a._populations(value)

    def test_extra_step25_rejected(self):
        value=spec();value['trusted_commits']['11031/T0'].append(reference())
        with self.assertRaises(PermissionError):a._populations(value)

    def test_missing_or_foreign_path_rejected(self):
        for name in ('11031/T0','11032/T1'):
            value=spec();value['trusted_commits']['99999/T1']=value['trusted_commits'].pop(name)
            with self.assertRaises(PermissionError):a._populations(value)

    def test_missing_dev_cannot_be_imputed(self):
        value=spec();del value['dev_endpoints']['11031/T1/16']
        with self.assertRaises(PermissionError):a._populations(value)

    def test_main_or_unscheduled_dev_cannot_replace_dev(self):
        value=spec();value['dev_endpoints']['11031/T1/24']=value['dev_endpoints'].pop('11031/T1/16')
        with self.assertRaises(PermissionError):a._populations(value)

    def test_missing_final_chunk_rejected(self):
        value=spec();del value['continuation_results']['11032/T1/chunk-017-024']
        with self.assertRaises(PermissionError):a._populations(value)

    def test_schema_extra_field_and_malformed_dev_rejected(self):
        for mutate in [lambda x:x.update(pass_override=True),lambda x:x.update(schema='fake'),
                       lambda x:x['dev_endpoints']['11031/T0/2'].update(score=1)]:
            value=spec();mutate(value)
            with self.assertRaises(PermissionError):a._populations(value)

    def test_audit_rejects_foreign_root_before_original_verifier(self):
        with patch.object(a.first_two,'verify_release') as verify:
            with self.assertRaises(PermissionError):a.audit(spec())
            verify.assert_not_called()


class RealLockTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name).resolve();self.path=self.root/'DIAGNOSTIC_WORKER.lock';self.path.touch()

    def test_idle_existing_lock_does_not_write(self):
        before=self.path.stat()
        with a._idle_lock(self.root,self.path.name):pass
        after=self.path.stat()
        self.assertEqual((before.st_size,before.st_mtime_ns),(after.st_size,after.st_mtime_ns))

    def test_active_worker_is_rejected_without_termination(self):
        descriptor=os.open(self.path,os.O_RDWR)
        try:
            fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(PermissionError):
                with a._idle_lock(self.root,self.path.name):pass
            self.assertEqual(os.fstat(descriptor).st_ino,self.path.stat().st_ino)
        finally:os.close(descriptor)

    def test_missing_lock_not_created(self):
        self.path.unlink()
        with self.assertRaises((PermissionError,FileNotFoundError)):
            with a._idle_lock(self.root,self.path.name):pass
        self.assertFalse(self.path.exists())

    def test_symlink_lock_rejected(self):
        other=self.root/'actual';other.touch();self.path.unlink();self.path.symlink_to(other)
        with self.assertRaises(PermissionError):
            with a._idle_lock(self.root,self.path.name):pass

    def test_replaced_lease_rejected_on_exit(self):
        with self.assertRaises(PermissionError):
            with a._idle_lock(self.root,self.path.name):
                self.path.rename(self.root/'old');self.path.touch()


class ChunkEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name).resolve();self.spec=spec();self.spec['root']=str(self.root)
        self.canary=SimpleNamespace(identities={key:dict(path=key) for key in a.PATHS})
        self.results={};self.risks={};self.releases={}
        for key in a.PATHS:
            start,arm=key.split('/')
            for lo,hi in a.CHUNKS.items():
                name=f'{key}/chunk-{lo:03d}-{hi:03d}';folder=self.root/f'continuation_control/start-{start}/{arm}/chunk-{lo:03d}-{hi:03d}'
                risks=[]
                for step in range(lo,hi+1):
                    path=folder/f'RISK_AFTER_STEP_{step:03d}.json'
                    value=dict(schema='sr-satdose-scientific-risk-assessment-v1',global_hold=False,hold_reasons=[],
                        trusted_commits={key:self.spec['trusted_commits'][key][:step]})
                    risks.append(self.write(path,value));self.risks[name,step]=value
                release=dict(original_train_release=self.spec['original_train_release'],canary_spec=self.spec['canary_spec'],
                    selection=dict(start_seed=int(start),arm=arm,first_step=lo,last_step=hi))
                release_ref=self.write(self.root/f'releases/{start}-{arm}-{lo}.json',release)
                self.releases[name]=release
                value=dict(identity=self.canary.identities[key],original_train_release=self.spec['original_train_release'],
                    process_exit_observed=False,main_authorized=False,train_responses=(hi-lo+1)*128,
                    formal_optimizer_steps=hi-lo+1,scheduler_steps=hi-lo+1,release=release_ref,
                    risk_assessments=risks,post_risk=risks[-1])
                self.spec['continuation_results'][name]=self.write(folder/'RESULT.json',value)
                self.results[name]=value
        self.verify=patch.object(a.continuation,'verify_release',side_effect=lambda path,digest:(a.tx.read_json(path),{}))
        self.verify.start();self.addCleanup(self.verify.stop)

    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(a.tx.canonical(value)+b'\n')
        return dict(path=str(path.relative_to(self.root)),sha256=a.tx.file_sha(path))

    def rewrite_result(self,name):
        path=self.root/self.spec['continuation_results'][name]['path']
        self.spec['continuation_results'][name]=self.write(path,self.results[name])

    def check(self):a._chunks(a.base._Observed(self.root),self.spec,{},self.canary)

    def test_exact_twelve_chunks_close_88_updates(self):self.check()

    def test_wrong_response_or_update_count_rejected(self):
        name='11031/T0/chunk-003-008';self.results[name]['train_responses']=767;self.rewrite_result(name)
        with self.assertRaises(PermissionError):self.check()

    def test_manufactured_exit_or_main_authority_rejected(self):
        name='11031/T0/chunk-003-008';self.results[name]['process_exit_observed']=True;self.rewrite_result(name)
        with self.assertRaises(PermissionError):self.check()

    def test_substituted_original_identity_rejected(self):
        name='11031/T0/chunk-003-008';self.results[name]['identity']={'path':'11032/T1'};self.rewrite_result(name)
        with self.assertRaises(PermissionError):self.check()

    def test_missing_step_risk_rejected(self):
        name='11031/T0/chunk-003-008';self.results[name]['risk_assessments'].pop(0);self.rewrite_result(name)
        with self.assertRaises(PermissionError):self.check()

    def test_historical_hold_cannot_disappear_at_terminal(self):
        name='11031/T0/chunk-003-008';ref=self.results[name]['risk_assessments'][0]
        value=self.risks[name,3];value['global_hold']=True;value['hold_reasons']=['HOLD_SEVERE_REGRESSION']
        self.results[name]['risk_assessments'][0]=self.write(self.root/ref['path'],value);self.rewrite_result(name)
        with self.assertRaises(PermissionError):self.check()

    def test_untrusted_mutation_rejected_by_hash(self):
        name='11031/T0/chunk-003-008';ref=self.spec['continuation_results'][name]
        (self.root/ref['path']).write_text('{}')
        with self.assertRaises(PermissionError):self.check()

    def test_release_selection_changed_rejected(self):
        name='11031/T0/chunk-003-008';value=self.releases[name];value['selection']['last_step']=24
        ref=self.results[name]['release'];self.results[name]['release']=self.write(self.root/ref['path'],value);self.rewrite_result(name)
        with self.assertRaises(PermissionError):self.check()


class ObservationTests(unittest.TestCase):
    def test_copying_original_modules_into_external_root_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve()
            with self.assertRaises(PermissionError):a._modules(dict(release_root=str(root),sources={}),a.base._Observed(root))

    def test_model_link_observations_preserved_and_repoint_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();one=root/'one';two=root/'two';one.write_text('a');two.write_text('a')
            link=root/'link';link.symlink_to(one.name)
            source=a.base._Observed(root);source.absolute(one);source.model_links[link]=(Path(one.name),one)
            source.model_directories[root]={p.name for p in root.iterdir()}
            target=a.base._Observed(root);a._merge(target,source);target.stable()
            link.unlink();link.symlink_to(two.name)
            with self.assertRaises(PermissionError):target.stable()


if __name__=='__main__':unittest.main()
