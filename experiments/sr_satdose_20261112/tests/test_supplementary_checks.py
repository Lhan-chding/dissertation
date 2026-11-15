"""Synthetic CPU boundaries only; these fixtures are never admission evidence."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from execution.supplementary_checks import Files, check_disjoint
sys.path.pop(0)


class SupplementaryTests(unittest.TestCase):
    def fixture(self):
        row = dict(split='TRAIN',case_id='c',input_id='i',root_id='root',request_id='SYNTHETIC/1')
        return ({'TRAIN':[row]}, {'c':dict(input_id='i', task=dict(root_id='root',key='new',fact='newfact'))},
                {'p':dict(png_sha256='x',original_sha256='x')}, {'i':dict(image_file='p')})

    def run_check(self, data):
        return check_disjoint(*data, {'old'}, {'oldfact'}, lambda x:x['key'], lambda x:x['fact'])

    def test_repeated_draws_of_same_root_allowed(self):
        data=self.fixture();data[0]['TRAIN'].append(dict(data[0]['TRAIN'][0],request_id='SYNTHETIC/2'))
        self.assertEqual(self.run_check(data)['new_cases_checked'],1)

    def test_cross_split_collision_rejected(self):
        data=self.fixture();data[0]['DEV']=[dict(data[0]['TRAIN'][0],split='DEV')]
        with self.assertRaisesRegex(ValueError,'Cross-split'):self.run_check(data)

    def test_historical_collision_without_retention_rejected(self):
        data=self.fixture();data[1]['c']['task']['fact']='oldfact'
        with self.assertRaisesRegex(ValueError,'Historical collision'):self.run_check(data)

    def test_retention_requires_old_task_image_and_eval_scope(self):
        data=self.fixture();row=data[0].pop('TRAIN')[0];row.update(split='DEV',panel='retention8')
        data[0]['DEV']=[row];data[1]['c']['task'].update(key='old',fact='oldfact')
        self.assertEqual(self.run_check(data)['retention_cases_checked'],1)
        changed=copy.deepcopy(data);changed[2]['p']['original_sha256']='different'
        with self.assertRaisesRegex(ValueError,'Retention'):self.run_check(changed)
        row['split']='TRAIN';data[0]['TRAIN']=data[0].pop('DEV')
        with self.assertRaisesRegex(ValueError,'Retention outside'):self.run_check(data)

    def test_case_identity_mismatch(self):
        data=self.fixture();data[1]['c']['input_id']='another'
        with self.assertRaisesRegex(ValueError,'identity'):self.run_check(data)

    def test_file_mutation_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();p=root/'f';p.write_bytes(b'first');files=Files(root)
            files.sha('f',hashlib.sha256(b'first').hexdigest());p.write_bytes(b'second')
            with self.assertRaisesRegex(ValueError,'Changed file'):files.sha('f')
            (root/'link').symlink_to(p)
            with self.assertRaisesRegex(ValueError,'Noncanonical'):files.sha('link')


if __name__ == '__main__':
    unittest.main()
