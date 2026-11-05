"""CPU guards for pinned failure selection, without any model/GPU execution."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from diagnostics.gpu_capacity_overlap_v2 import FAILURE_ROW, failure_evidence


class FixedFailureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.file = self.root/'run/paths/b0-11031-G/executions/274335-fixed/GPU_UPDATE_ACCOUNTING_step-009.json'
        self.file.parent.mkdir(parents=True)
        self.records = [dict(row_id=str(i)) for i in range(128)]
        self.records[58]['row_id'] = FAILURE_ROW
        self.value = dict(status='FAILED', step=9,pre_adam_calls=0,optimizer_returned_calls=0,
                          scheduler_entered_calls=0,scheduler_returned_calls=0,new_generations=0,
                          adam_permitted=False,forwards=[dict(row_id=r['row_id'],forward_returned=True)
                          for r in self.records[:59]])
        self.value['forwards'][-1] = dict(row_id=FAILURE_ROW, stats=dict(copied_bytes=16*1024**3,cap_skips=1))

    def check(self,value):
        self.file.write_text(json.dumps(value))
        return failure_evidence(self.root,self.records)

    def test_exact_failure_is_pinned_by_hash(self):
        ref=self.check(self.value)
        self.assertEqual(ref['path'],str(self.file))
        self.assertEqual(len(ref['sha256']),64)

    def test_post_adam_or_generation_or_wrong_failure_rejected(self):
        for key,value in [('pre_adam_calls',1),('optimizer_returned_calls',1),('scheduler_entered_calls',1),
                          ('scheduler_returned_calls',1),('new_generations',1),('adam_permitted',True),
                          ('step',10),('status','COMPLETE')]:
            bad=copy.deepcopy(self.value);bad[key]=value
            with self.subTest(key=key),self.assertRaises(PermissionError):self.check(bad)

    def test_missing_cap_or_wrong_fixed_row_rejected(self):
        for mutate in [lambda v:v['forwards'][-1]['stats'].update(copied_bytes=0),
                       lambda v:v['forwards'][-1]['stats'].update(cap_skips=0),
                       lambda v:v['forwards'][-1].update(row_id='different'),
                       lambda v:v['forwards'][0].update(forward_returned=False),
                       lambda v:v['forwards'].pop(0)]:
            bad=copy.deepcopy(self.value);mutate(bad)
            with self.assertRaises(PermissionError):self.check(bad)


if __name__=='__main__':unittest.main()
