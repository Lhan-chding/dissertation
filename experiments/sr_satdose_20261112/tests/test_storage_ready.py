"""Local CPU filesystem tests with mocked Ceph xattrs; no server observations."""
import copy
from datetime import datetime, timedelta, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
from execution import storage_ready as s
sys.path.pop(0)


class StorageReadyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.quota=Path(self.temp.name).resolve()/'varunssd'
        self.root=self.quota/'louis-ssvc/sr_satdose_20261112/prepared_data_v1'
        self.stage=self.root/'training_transactions/start-11031/T0/.staging-step-001'
        self.stage.mkdir(parents=True)
        freeze=self.root/'DESIGN_FREEZE.json';freeze.write_text('{"synthetic_cpu_fixture":true}\n')
        self.freeze_sha=hashlib.sha256(freeze.read_bytes()).hexdigest()
        self.now=datetime(2026,11,13,12,0,0,tzinfo=timezone.utc)
        self.identity=dict(experiment_id=s.EXPERIMENT,train_seed=11031,arm='T0')
        self.dump(self.stage/'STEP_INTENT.json',dict(identity=self.identity,step=1,design_freeze_sha256=self.freeze_sha))
        self.budget_path=self.root/'storage_budget.json'
        self.budget=dict(schema='sr-satdose-storage-budget-v1',experiment_id=s.EXPERIMENT,evidence_root=str(self.root),
            design_freeze_sha256=self.freeze_sha,required_bytes=500,max_observation_age_seconds=30,
            created_at_utc=(self.now-timedelta(minutes=1)).isoformat(),expires_at_utc=(self.now+timedelta(minutes=10)).isoformat(),
            accounting_note='Synthetic CPU test only; not a production budget')
        self.dump(self.budget_path,self.budget)
        self.budget_ref=dict(path=self.budget_path.name,sha256=s._sha(self.budget_path))
        self.quotas={self.quota:(10_000,9_000)}
        self._patch(s,'ROOT',self.root);self._patch(s,'QUOTA_ANCESTOR',self.quota);self._patch(s,'FREEZE_SHA',self.freeze_sha)
        self._patch(s,'_now',lambda:self.now)
        self._patch(s.os,'getxattr',self.getxattr)
        self._patch(s.os,'statvfs',lambda path:SimpleNamespace(f_bavail=6000,f_frsize=1))

    def _patch(self,obj,name,value):
        item=patch.object(obj,name,value,create=True);item.start();self.addCleanup(item.stop)

    @staticmethod
    def dump(path,value):
        path.write_text(json.dumps(value,sort_keys=True,allow_nan=False)+'\n')

    def getxattr(self,path,key,*,follow_symlinks):
        self.assertFalse(follow_symlinks)
        pair=self.quotas.get(Path(path))
        if pair is None:raise OSError(errno.ENODATA,'synthetic absent quota')
        value=pair[0 if key==s.QUOTA_KEY else 1]
        if isinstance(value,Exception):raise value
        return value if isinstance(value,bytes) else str(value).encode()

    def call(self):
        return s.make_storage_ready(self.budget_ref)(self.stage,1)

    def assert_clean(self):
        self.assertEqual(list(self.stage.glob('.storage_probe-*')),[])

    def test_live_positive_quota_write_fsync_and_receipt_are_bound(self):
        sentinel=self.stage/'keep-me.txt';sentinel.write_text('unrelated data')
        result=self.call()
        self.assertEqual(result['required_bytes'],500);self.assertEqual(result['effective_available_bytes'],1000)
        self.assertEqual(result['budget_reference'],self.budget_ref)
        self.assertEqual(result['evidence'][0],self.budget_ref)
        self.assertTrue(result['ready']);self.assertEqual(result['permission'],'NO_TRAINING_ADMISSION')
        self.assertTrue(result['probe']['file_fsync']);self.assertTrue(result['probe']['directory_fsync'])
        self.assertTrue(result['probe']['probe_removed']);self.assert_clean()
        self.assertEqual(sentinel.read_text(),'unrelated data')
        self.assertEqual(s.validate_readiness(result,budget_reference=self.budget_ref,stage=self.stage,step=1),result)
        # Compatibility with the transaction's exact storage-ready boundary,
        # without importing a model or claiming actual training readiness.
        self.assertEqual(result['schema'],'sr-satdose-storage-readiness-v1')
        self.assertTrue(result['quota_verified'] and result['writable_verified'])
        for ref in result['evidence']:
            self.assertEqual(s._sha(self.root/ref['path']),ref['sha256'])
        with patch.object(sys,'path',[str(PACKAGE),*sys.path]):
            from execution import training_transaction
        validator=object.__new__(training_transaction.TransactionStore)
        validator.root=self.root;validator.store=self.root/'training_transactions'
        validator._storage(result)

    def test_nearer_positive_quota_and_statvfs_each_limit_capacity(self):
        self.quotas[self.root]=(700,100)
        result=self.call();self.assertEqual(result['effective_available_bytes'],600)
        first=result['observations'][0]
        self.assertEqual(len([q for q in first['quota_observations'] if q['status']=='POSITIVE_QUOTA']),2)
        # Separate current step, retaining the first step's write-once evidence.
        (self.root/result['evidence'][1]['path']).unlink()
        with patch.object(s.os,'statvfs',return_value=SimpleNamespace(f_bavail=550,f_frsize=1)):
            result=self.call();self.assertEqual(result['effective_available_bytes'],550)

    def test_insufficient_quota_rejects_even_when_df_is_large(self):
        self.quotas[self.quota]=(10_000,9_700)
        with patch.object(s.os,'statvfs',return_value=SimpleNamespace(f_bavail=10**15,f_frsize=4096)):
            with self.assertRaisesRegex(PermissionError,'insufficient live quota'):
                self.call()
        self.assert_clean();self.assertFalse((self.root/'storage_readiness').exists())

    def test_enodata_only_is_no_local_limit_but_parent_limit_is_required(self):
        result=self.call()
        self.assertTrue(any(q['status']=='NO_LOCAL_LIMIT_ENODATA' for q in result['observations'][0]['quota_observations']))
        (self.root/result['evidence'][1]['path']).unlink()
        del self.quotas[self.quota]
        with self.assertRaisesRegex(PermissionError,'no verified positive quota'):
            self.call()

    def test_zero_negative_nonfinite_and_malformed_quota_fail_closed(self):
        for value in (b'0',b'-1',b'nan',b'10.5',b' 1000',b'1000\n'):
            with self.subTest(value=value):
                self.quotas[self.quota]=(value,0)
                with self.assertRaises(PermissionError):self.call()
        self.assert_clean()

    def test_xattr_permissions_unsupported_or_missing_usage_fail_closed(self):
        for error in (errno.EACCES,errno.ENOTSUP,errno.EIO):
            with self.subTest(error=error):
                self.quotas[self.quota]=(OSError(error,'synthetic'),0)
                with self.assertRaisesRegex(PermissionError,'cannot verify'):self.call()
        self.quotas[self.quota]=(1000,OSError(errno.ENODATA,'missing usage'))
        with self.assertRaisesRegex(PermissionError,'cannot verify'):self.call()
        self.assert_clean()

    def test_probe_permission_or_fsync_failure_cleans_only_own_probe(self):
        real_open=os.open
        def denied(path,*args,**kwargs):
            if Path(path).name.startswith('.storage_probe-'):raise PermissionError(errno.EACCES,'synthetic write denied')
            return real_open(path,*args,**kwargs)
        with patch.object(s.os,'open',side_effect=denied):
            with self.assertRaises(PermissionError):self.call()
        self.assert_clean()
        sentinel=self.stage/'sentinel';sentinel.write_text('preserve')
        with patch.object(s.os,'fsync',side_effect=OSError(errno.EIO,'synthetic fsync failure')):
            with self.assertRaises(OSError):self.call()
        self.assert_clean();self.assertEqual(sentinel.read_text(),'preserve')
        self.assertFalse((self.root/'storage_readiness').exists())

    def test_capacity_drop_after_probe_is_rejected_without_probe_leftover(self):
        real=self.getxattr;uses=0
        def changing(path,key,**kwargs):
            nonlocal uses
            if Path(path)==self.quota and key==s.USAGE_KEY:
                uses+=1;return b'9000' if uses==1 else b'9900'
            return real(path,key,**kwargs)
        with patch.object(s.os,'getxattr',side_effect=changing):
            with self.assertRaisesRegex(PermissionError,'capacity decreased'):self.call()
        self.assert_clean();self.assertFalse((self.root/'storage_readiness').exists())

    def test_changed_quota_during_usage_read_is_rejected(self):
        real=self.getxattr;limits=0
        def changing(path,key,**kwargs):
            nonlocal limits
            if Path(path)==self.quota and key==s.QUOTA_KEY:
                limits+=1;return b'10000' if limits==1 else b'20000'
            return real(path,key,**kwargs)
        with patch.object(s.os,'getxattr',side_effect=changing):
            with self.assertRaisesRegex(PermissionError,'quota limit changed'):self.call()

    def test_budget_requires_independent_sha_positive_bytes_and_fresh_utc(self):
        with patch.object(s,'_probe',side_effect=AssertionError('must not probe')):
            for field,value in [('required_bytes',0),('required_bytes',True),('max_observation_age_seconds',0),
                                ('expires_at_utc',(self.now-timedelta(seconds=1)).isoformat()),
                                ('created_at_utc',(self.now+timedelta(seconds=1)).isoformat()),
                                ('created_at_utc','2026-11-13T00:00:00')]:
                with self.subTest(field=field,value=value):
                    budget=copy.deepcopy(self.budget);budget[field]=value;self.dump(self.budget_path,budget)
                    ref=dict(path=self.budget_path.name,sha256=s._sha(self.budget_path))
                    with self.assertRaises(PermissionError):s.make_storage_ready(ref)(self.stage,1)
            self.dump(self.budget_path,self.budget)
            with self.assertRaisesRegex(PermissionError,'SHA differs'):
                s.make_storage_ready(dict(self.budget_ref,sha256='0'*64))(self.stage,1)

    def test_stale_measurement_or_tampered_receipt_is_rejected(self):
        result=self.call()
        with patch.object(s,'_now',return_value=self.now+timedelta(seconds=31)):
            with self.assertRaisesRegex(PermissionError,'stale storage'):
                s.validate_readiness(result,budget_reference=self.budget_ref,stage=self.stage,step=1)
        changed=copy.deepcopy(result);changed['effective_available_bytes']+=1
        with self.assertRaisesRegex(PermissionError,'content mismatch'):
            s.validate_readiness(changed,budget_reference=self.budget_ref,stage=self.stage,step=1)
        self.assert_clean()

    def test_slow_observation_and_stage_symlink_or_wrong_step_rejected(self):
        with patch.object(s.time,'monotonic',side_effect=[100.0,131.0]):
            with self.assertRaisesRegex(PermissionError,'stale storage'):self.call()
        self.assert_clean()
        alias=self.root/'stage-alias';alias.symlink_to(self.stage,target_is_directory=True)
        with self.assertRaises(PermissionError):s.make_storage_ready(self.budget_ref)(alias,1)
        with self.assertRaises(PermissionError):s.make_storage_ready(self.budget_ref)(self.stage,2)
        (self.stage/'UPDATE_INTENT.json').write_text('{}')
        with self.assertRaisesRegex(PermissionError,'before UPDATE_INTENT'):self.call()

    def test_write_once_measurement_is_never_overwritten(self):
        result=self.call();path=self.root/result['evidence'][1]['path'];before=path.read_bytes()
        with patch.object(s,'_probe',side_effect=AssertionError('must not re-probe an occupied slot')):
            with self.assertRaisesRegex(PermissionError,'already exists'):self.call()
        self.assertEqual(path.read_bytes(),before);self.assert_clean()

    def test_no_xattr_api_or_malformed_statvfs_is_not_a_ready_result(self):
        with patch.object(s.os,'getxattr',None):
            with self.assertRaisesRegex(PermissionError,'cannot read Ceph xattrs'):self.call()
        for free,size in ((-1,1),(100,0),(True,1),(100,1.5)):
            with self.subTest(free=free,size=size):
                with patch.object(s.os,'statvfs',return_value=SimpleNamespace(f_bavail=free,f_frsize=size)):
                    with self.assertRaisesRegex(PermissionError,'malformed statvfs'):self.call()
        self.assert_clean()


if __name__=='__main__':unittest.main()
