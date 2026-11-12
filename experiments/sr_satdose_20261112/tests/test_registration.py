"""Synthetic CPU gate fixtures; never experimental evidence."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).parents[1]))
r=importlib.import_module('runtime.registration')

class FreezeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.code=self.root/'code';self.code.mkdir()
        (self.code/'test_only.py').write_text('# fixture\n')
        self.cfg=self.root/'config.json';self.dump(self.cfg,{'experiment_id':'SR-SATDOSE-1-20261112'})
        self.lock=self.root/'source_lock.json';self.dump(self.lock,{})
        names=['calibration_groups','private_cases','training_groups','inputs','structural_rejections']+[p+'_manifest' for p in ('train','dry','dev','prefix_calib','main')]
        for n in names:(self.root/(n+'.jsonl')).write_text('{}\n')
        images=[]
        for i in range(32):
            p=self.root/f'image{i}.png';p.write_bytes(b'synthetic CPU fixture')
            images.append({'image_file':p.name,'png_sha256':r.file_sha(p),'image_structure_pass':True,'metadata_empty':True})
        for n in ('images','visual_review_manifest'):(self.root/(n+'.jsonl')).write_text(''.join(json.dumps(x)+'\n' for x in images))
        self.dump(self.root/'PROBE_SPLITS.json',{})
        self.dump(self.root/'PREPARATION.json',{'images_rendered':True,'model_calls':0,'config_sha256':r.file_sha(self.cfg),'files':{p.name:r.file_sha(p) for p in self.root.glob('*.jsonl')}})
        self.visual=self.root/'VISUAL.json';self.review={'status':'PASS','reviewed_images':32,'manifest_sha256':r.file_sha(self.root/'visual_review_manifest.jsonl'),'reviewer_kind':'AI'}
        self.dump(self.visual,self.review)
    def dump(self,p,obj):p.write_text(json.dumps(obj))
    def freeze(self):return r.freeze_design(self.root,self.cfg,self.visual,self.lock,self.code,'a'*40)
    def authorize_fixture(self):
        self.review.update(user_authorized_substitution=True,authorization_quote='UNIT TEST ONLY');self.dump(self.visual,self.review)
    def test_ai_does_not_impersonate_manual_review(self):
        with self.assertRaisesRegex(ValueError,'Manual review'):self.freeze()
    def test_explicit_substitution_and_write_once(self):
        self.authorize_fixture();out=self.freeze();self.assertEqual(out['runtime_sources'],{'test_only.py':r.file_sha(self.code/'test_only.py')})
        with self.assertRaises(FileExistsError):self.freeze()
    def test_changed_preparation_fails(self):
        self.authorize_fixture();(self.root/'private_cases.jsonl').write_text('tamper')
        with self.assertRaisesRegex(ValueError,'Prepared artifact'):self.freeze()
    def test_existing_attempt_prevents_retroactive_freeze(self):
        self.authorize_fixture();(self.root/'x.BEGIN.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'already exists'):self.freeze()
    def test_changed_image_fails(self):
        self.authorize_fixture();(self.root/'image0.png').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Image verification'):self.freeze()

if __name__=='__main__':unittest.main()
