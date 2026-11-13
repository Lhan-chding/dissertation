"""Synthetic completed-DRY fixtures only; no live raw or GPU/model access."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
from execution import diagnostic_inputs as d
from execution import production
sys.path.pop(0)

# Reuse the isolated synthetic completion factory, not its TestCase collection.
spec=importlib.util.spec_from_file_location('_diagnostic_fixture_factory',Path(__file__).with_name('test_calibration_analysis.py'))
fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)
NATIVE=fixtures.NATIVE
core=d.core


@unittest.skipUnless((NATIVE/'sr_f6/measurement.py').is_file(),'Pinned native scorer sources required')
class DiagnosticInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(fixtures.CalibrationTests.doClassCleanups)
        fixtures.CalibrationTests.setUpClass()
        cls.factory=fixtures.CalibrationTests
        cls.tmp=tempfile.TemporaryDirectory();cls.addClassCleanup(cls.tmp.cleanup)
        cls.base=Path(cls.tmp.name).resolve()/'fixture';shutil.copytree(cls.factory.base,cls.base)
        groups=list(core.read_jsonl(cls.base/'calibration_groups.jsonl'))
        # Deliberately unsorted frozen orders make accidental sorting observable.
        within=[7,3,0,6,1,5,2,4]
        for group in groups:
            group['request_ids']=[group['request_ids'][i] for i in within]
            group['draw_seeds']=[group['draw_seeds'][i] for i in within]
        fixtures.lines(cls.base/'calibration_groups.jsonl',groups)
        own=[g for g in groups if g['start_seed']==11031]
        conflicts=[g['root_id'] for g in own if g['kind']=='CONFLICT']
        aligned=[g['root_id'] for g in own if g['kind']=='ALIGNED']
        probe=[conflicts[i] for i in (11,2,7,0,9,4,1,10,5,8,3,6)]+[aligned[i] for i in (3,1,2,0)]
        witness=[conflicts[i] for i in range(55,39,-1)]
        splits=dict(probe_root_ids=probe,witness_root_ids=witness,
                    probe_halves=[probe[1::2],probe[::2]],witness_halves=[witness[1::2],witness[::2]])
        fixtures.dump(cls.base/'PROBE_SPLITS.json',splits)
        cls.rebind(cls.base)

    @staticmethod
    def rebind(root):
        """Change ONLY a temporary synthetic design before its analysis tests."""
        path=root/'DESIGN_FREEZE.json';freeze=core.read_json(path)
        for key,name in [('probe_splits','PROBE_SPLITS.json'),('calibration_groups','calibration_groups.jsonl')]:
            freeze['artifacts'][key]=dict(path=name,sha256=core.file_sha(root/name))
        fixtures.dump(path,freeze);digest=core.file_sha(path)
        for scope in ('DRY','PREFIX_CALIB'):
            for start in core.STARTS:
                output=root/f'{scope}-{start}';raw_path=output/'raw.jsonl'
                rows=list(core.read_jsonl(raw_path))
                for row in rows:row['design_freeze_sha256']=digest
                fixtures.lines(raw_path,rows)
                refs=fixtures.write_fixture_complete(output,rows)
                receipt=core.read_json(output/'CALIBRATION_COMPLETE.json')
                receipt.update(design_freeze_sha256=digest,complete_files=refs)
                receipt['raw']['sha256']=core.file_sha(raw_path)
                fixtures.dump(output/'CALIBRATION_COMPLETE.json',receipt)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()/'fixture';shutil.copytree(self.base,self.root)
    def receipts(self):return {s:self.root/f'DRY-{s}/CALIBRATION_COMPLETE.json' for s in core.STARTS}
    def build(self):
        return d.build_diagnostic_inputs(self.root/'config.json',self.root/'DESIGN_FREEZE.json',self.root,self.receipts(),NATIVE)
    def test_frozen_root_request_and_half_orders_are_preserved(self):
        before={s:(self.root/f'DRY-{s}/raw.jsonl').read_bytes() for s in core.STARTS}
        result=self.build();mapping=core.read_json(self.root/'PROBE_SPLITS.json')
        cfg=core.read_json(self.root/'config.json')
        groups=list(core.read_jsonl(self.root/'calibration_groups.jsonl'))
        for start in core.STARTS:
            value=result['starts'][str(start)];byroot={g['root_id']:g for g in groups if g['start_seed']==start}
            for name in ('probe','witness'):
                batch=value[name]
                self.assertEqual(batch['root_ids'],mapping[name+'_root_ids'])
                self.assertEqual(batch['group_ids'],[byroot[root]['group_id'] for root in batch['root_ids']])
                self.assertEqual([r['request_id'] for r in batch['rows']],
                                 [rid for root in batch['root_ids'] for rid in byroot[root]['request_ids']])
                self.assertEqual(batch['half_group_ids'],[[byroot[root]['group_id'] for root in half] for half in mapping[name+'_halves']])
                self.assertEqual([len(half) for half in batch['half_rows']],[64,64])
                for half,ids in zip(batch['half_rows'],batch['half_group_ids']):
                    self.assertEqual([row['group_id'] for row in half[::8]],ids)
            production.validate_batch(value['probe']['rows'],value['production_plan'],value['expected_identity'],cfg,
                                      value['expected_identity']['policy_sha256'],value['expected_identity']['reference_sha256'],
                                      SimpleNamespace(advantages=core.native_advantages))
            self.assertEqual(before[start],(self.root/f'DRY-{start}/raw.jsonl').read_bytes())
            self.assertFalse(set(value['probe']['root_ids']) & set(value['witness']['root_ids']))
        self.assertEqual(result['permission'],'NO_TRAINING');self.assertEqual(result['model_calls'],0)
        self.assertEqual(result['optimizer_steps'],0)
        self.assertEqual(result['counts']['source_dry_responses'],1280)
        self.assertEqual(result['counts']['selected_probe_responses'],256)
        self.assertEqual(result['counts']['selected_witness_responses'],256)
        self.assertEqual(result['frozen_artifacts']['probe_splits']['sha256'],core.file_sha(self.root/'PROBE_SPLITS.json'))
    def test_second_start_must_be_complete_before_any_raw_analysis(self):
        path=self.receipts()[11032];receipt=core.read_json(path);receipt['responses']=639;fixtures.dump(path,receipt)
        with patch.object(d.calibration,'analyze_dry',side_effect=AssertionError('partial calibration read')):
            with self.assertRaisesRegex(ValueError,'Both DRY starts must be complete'):self.build()
    def test_probe_splits_changed_after_freeze_rejected(self):
        path=self.root/'PROBE_SPLITS.json';mapping=core.read_json(path);mapping['probe_root_ids'].reverse();fixtures.dump(path,mapping)
        with self.assertRaisesRegex(ValueError,'artifact hash mismatch'):self.build()
    def test_duplicate_roots_or_halves_rejected(self):
        original=core.read_json(self.root/'PROBE_SPLITS.json')
        for which in ('probe_root_ids','probe_halves'):
            with self.subTest(which=which):
                mapping=copy.deepcopy(original)
                if which=='probe_root_ids':mapping[which][-1]=mapping[which][0]
                else:mapping[which][1][-1]=mapping[which][0][0]
                with self.assertRaises(ValueError):d._validate_roots(mapping)
    def test_probe_witness_overlap_rejected(self):
        mapping=core.read_json(self.root/'PROBE_SPLITS.json')
        old=mapping['witness_root_ids'][0];new=mapping['probe_root_ids'][0]
        mapping['witness_root_ids'][0]=new
        mapping['witness_halves']=[[new if root==old else root for root in half] for half in mapping['witness_halves']]
        with self.assertRaisesRegex(ValueError,'overlap'):d._validate_roots(mapping)
    def test_wrong_probe_conflict_allocation_rejected(self):
        path=self.root/'PROBE_SPLITS.json';mapping=core.read_json(path)
        groups=list(core.read_jsonl(self.root/'calibration_groups.jsonl'))
        old=mapping['probe_root_ids'][0]
        new=next(g['root_id'] for g in groups if g['start_seed']==11031 and g['kind']=='ALIGNED'
                 and g['root_id'] not in mapping['probe_root_ids'])
        mapping['probe_root_ids'][0]=new
        mapping['probe_halves']=[[new if root==old else root for root in half] for half in mapping['probe_halves']]
        fixtures.dump(path,mapping);self.rebind(self.root)
        with self.assertRaisesRegex(ValueError,'probe conflict/aligned allocation'):self.build()
    def test_duplicate_group_inventory_rejected(self):
        path=self.root/'calibration_groups.jsonl';groups=list(core.read_jsonl(path));groups[-1]=groups[0]
        fixtures.lines(path,groups);self.rebind(self.root)
        with self.assertRaisesRegex(ValueError,'duplicate calibration group'):self.build()
    def test_raw_replaced_original_tokens_rejected(self):
        path=self.root/'DRY-11031/raw.jsonl';rows=list(core.read_jsonl(path));rows[0]['tokens']=[99,100]
        fixtures.lines(path,rows);rp=self.receipts()[11031];receipt=core.read_json(rp)
        receipt['raw']['sha256']=core.file_sha(path);fixtures.dump(rp,receipt)
        with self.assertRaisesRegex(ValueError,'original COMPLETE fields'):self.build()
    def test_native_known_source_used_instead_of_field_local(self):
        mapping=core.read_json(self.root/'PROBE_SPLITS.json')
        path=self.root/'DRY-11031/raw.jsonl';rows=list(core.read_jsonl(path))
        target=next(row for row in rows if row['root_id']==mapping['probe_root_ids'][0] and row['native_score']['J']==1)
        cases={r['case_id']:r for r in core.read_jsonl(self.root/'private_cases.jsonl')};task=cases[target['case_id']]['task']
        target['finish_reason']='eos'
        target['native_score']=self.factory.scorer(target['full_text'],task['world'],task['query'],finished=False,finish_reason='eos')
        self.assertFalse(target['native_score']['known_mask']['S'])
        self.assertEqual(d.calibration._strict_selection(target['full_text'],'eos',task['gold']['selected'])['S_hit'],1)
        fixtures.lines(path,rows);rp=self.receipts()[11031];receipt=core.read_json(rp)
        receipt['raw']['sha256']=core.file_sha(path);receipt['complete_files']=fixtures.write_fixture_complete(path.parent,rows);fixtures.dump(rp,receipt)
        original=path.read_bytes();result=self.build()
        row=next(row for row in result['starts']['11031']['probe']['rows'] if row['request_id']==target['request_id'])
        self.assertEqual(row['exact_authorised'],0);self.assertEqual(row['J'],0)
        self.assertNotIn('exact_authorised',target);self.assertNotIn('J',target)
        self.assertEqual(original,path.read_bytes())
        # Returned half rows are independent objects; changing one cannot alter
        # the full batch or the immutable original source on disk.
        half=result['starts']['11031']['probe']['half_rows'][0][0]
        same=next(r for r in result['starts']['11031']['probe']['rows'] if r['request_id']==half['request_id'])
        half['tokens'][0]=999
        self.assertNotEqual(same['tokens'][0],999)
    def test_failing_signal_blocks_both_start_preparation(self):
        path=self.root/'DRY-11032/raw.jsonl';rows=list(core.read_jsonl(path));cases={r['case_id']:r for r in core.read_jsonl(self.root/'private_cases.jsonl')}
        for row in rows:
            row.update(full_text='{}',suffix_text='}',generated_text='}')
            task=cases[row['case_id']]['task']
            row['native_score']=self.factory.scorer('{}',task['world'],task['query'],finished=True,finish_reason='balanced')
        fixtures.lines(path,rows);rp=self.receipts()[11032];receipt=core.read_json(rp)
        receipt['raw']['sha256']=core.file_sha(path);receipt['complete_files']=fixtures.write_fixture_complete(path.parent,rows);fixtures.dump(rp,receipt)
        with self.assertRaisesRegex(ValueError,'Both DRY signal gates must pass'):self.build()


if __name__=='__main__':unittest.main()
