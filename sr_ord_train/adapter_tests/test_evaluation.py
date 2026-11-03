"""Synthetic endpoint tests; never reads the released formal manifest."""
import ast
import copy
from types import SimpleNamespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from adapter.evaluation import (completed_paths,endpoint_descriptors,endpoint_id,
    expected_paths,register_evaluation,request_row,resolve_public,training_identity,validate_path_receipt)
from adapter.backend import file_sha,object_sha,write_once

class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()
    def tearDown(self):self.temp.cleanup()
    def synthetic_paths(self):
        paths=[]
        for parent,seed,arm,name in expected_paths():
            reference=('a' if parent=='b0' else 'b')*64
            identity=dict(parent=parent,arm=arm,train_seed=seed)
            markers=[dict(step=s,policy_sha256=reference if s==0 else str(s%10)*64,
                reference_sha256=reference,state_sha256='c'*64,commit_sha256='d'*64,update_sha256='e'*64)
                for s in range(25)]
            paths.append((self.root/name,identity,markers))
        return paths
    def test_endpoint_matrix_preserves_both_checkpoints_and_single_parents(self):
        endpoints=endpoint_descriptors(self.synthetic_paths())
        self.assertEqual(len(endpoints),26)
        self.assertEqual(sum(e['arm']=='PARENT' for e in endpoints.values()),2)
        self.assertEqual(endpoints['b0-11031-D-step-012']['step'],12)
        self.assertEqual(endpoints['b0-11031-D-step-024']['step'],24)
        self.assertTrue(endpoints['b0-11031-D-step-012']['checkpoint']['path'].endswith('step-012/state.pt'))
    def test_endpoint_gate_matches_actual_driver_implementation_identity(self):
        # Evaluate only the real driver's identity construction, never its model path.
        driver=Path(__file__).resolve().parents[1]/'adapter/driver.py'
        module=ast.parse(driver.read_text())
        function=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='train_path')
        assignment=next(n for n in function.body if isinstance(n,ast.Assign)
                        and any(isinstance(t,ast.Name) and t.id=='identity' for t in n.targets))
        expression=compile(ast.Expression(assignment.value),str(driver),'eval')
        binding=dict(manifest_sha256='a'*64,source_sha256='b'*64,
            implementation_files=[dict(path='/new/adapter/backend.py',sha256='c'*64)],code_commit='d'*40)
        for parent,seed,arm,name in expected_paths():
            runtime=SimpleNamespace(identity=dict(arm=arm,train_seed=seed,run_id='SR-ORD-TRAIN-1/'+name))
            backend=SimpleNamespace(parent=parent,binding=binding)
            actual=eval(expression,{'dict':dict,'object_sha':object_sha},
                        {'backend':backend,'r':runtime,'plan_sha':'e'*64})
            expected=training_identity(binding,parent,seed,arm,'e'*64)
            self.assertEqual(expected,actual)
            self.assertEqual(expected['code_commit'],binding['code_commit'])
            self.assertEqual(expected['implementation_sha256'],object_sha(binding['implementation_files']))

    def receipt_fixture(self):
        directory=self.root/'path';directory.mkdir()
        binding=self.root/'RUN_BINDING.json';write_once(binding,{'original':'binding'})
        replay=directory/'replay.json'
        write_once(replay,dict(status='INDEPENDENT_NATIVE_REPLAY_PASS',responses=3072))
        identity=dict(manifest_sha256='a'*64)
        markers=[dict(policy_sha256='b'*64,reference_sha256='b'*64,state_sha256=str(i%10)*64)
                 for i in range(25)]
        receipt=dict(seal_status='PATH_NATIVE_REPLAY_AND_RECEIPT_VERIFIED',identity=identity,
            parent_tensor_sha256='b'*64,reference_tensor_sha256='b'*64,
            evaluation_policy_sha256={'12':'b'*64,'24':'b'*64},
            evidence=dict(native_binding={'sha256':file_sha(binding)},
                zero_checkpoint={'sha256':markers[0]['state_sha256']},
                checkpoint_12={'sha256':markers[12]['state_sha256']},
                checkpoint_24={'sha256':markers[24]['state_sha256']}),
            native_replay={'path':'replay.json','sha256':file_sha(replay)})
        write_once(directory/'PATH_RECEIPT.json',receipt)
        return directory,identity,markers,binding,receipt

    def test_path_seal_gate_passes_independent_expected_plan_to_checker(self):
        directory,identity,markers,binding,receipt=self.receipt_fixture()
        expected='e'*64
        result=dict(status='PATH_RECEIPT_STRUCTURE_AND_ARTIFACT_DIGESTS_VERIFIED',steps=24,responses=3072)
        with patch('runtime_contract.check_path',return_value=result) as checker:
            got=validate_path_receipt(directory,identity,markers,binding,expected)
        checker.assert_called_once_with(receipt,identity['manifest_sha256'],directory,expected)
        self.assertEqual(got['sha256'],file_sha(directory/'PATH_RECEIPT.json'))

    def test_unaccepted_path_seal_blocks_before_contract_or_manifest(self):
        directory,identity,markers,binding,receipt=self.receipt_fixture()
        receipt['seal_status']='EXPORT_IN_PROGRESS'
        (directory/'PATH_RECEIPT.json').write_text(json.dumps(receipt))
        with patch('runtime_contract.check_path') as checker:
            with self.assertRaisesRegex(PermissionError,'not passed'):
                validate_path_receipt(directory,identity,markers,binding,'e'*64)
            checker.assert_not_called()

    def test_sealed_endpoint_cannot_substitute_other_checkpoint(self):
        directory,identity,markers,binding,receipt=self.receipt_fixture()
        receipt['evidence']['checkpoint_12']['sha256']=markers[24]['state_sha256']
        (directory/'PATH_RECEIPT.json').write_text(json.dumps(receipt))
        with self.assertRaisesRegex(PermissionError,'checkpoint bytes differ'):
            validate_path_receipt(directory,identity,markers,binding,'e'*64)

    def test_parent_drift_between_paths_rejected(self):
        paths=self.synthetic_paths();paths[1][2][0]['policy_sha256']='f'*64
        paths[1][2][0]['reference_sha256']='f'*64
        with self.assertRaises(PermissionError):endpoint_descriptors(paths)
    def test_parent_cannot_be_duplicated_by_training_seed(self):
        for seed in (11031,11032):
            with self.assertRaises(PermissionError):endpoint_id('b0','PARENT',seed,0)
    def test_missing_training_path_blocks_before_formal_manifest_is_opened(self):
        (self.root/'frozen/receipt_plans').mkdir(parents=True)
        write_once(self.root/'frozen/receipt_plans/INDEX.json',{'files':{}})
        with self.assertRaisesRegex(PermissionError,'Training incomplete'):
            with completed_paths(self.root,{}):self.fail('Incomplete training was admitted')
    def test_registration_does_not_open_formal_manifest_on_failed_gate(self):
        (self.root/'execution').mkdir();(self.root/'frozen').mkdir();(self.root/'run').mkdir()
        freeze=self.root/'frozen/FREEZE.json';write_once(freeze,{'files':{}})
        write_once(self.root/'execution/RUN_BINDING.json',dict(experiment_id='SR-ORD-TRAIN-1',
            package_root=str(self.root),manifest_sha256=file_sha(freeze)))
        # A deliberately malformed manifest would cause JSON error if opened early.
        (self.root/'frozen/evaluation_manifest.jsonl').write_text('MUST NOT OPEN')
        with patch('adapter.evaluation.completed_paths',side_effect=PermissionError('incomplete gate')):
            with self.assertRaisesRegex(PermissionError,'incomplete gate'):
                register_evaluation(self.root,self.root/'run/evaluation_registration','a'*40)
        self.assertFalse((self.root/'run/evaluation_registration').exists())
    def test_public_resolution_keeps_long_same_assistant_prefix(self):
        (self.root/'frozen/images').mkdir(parents=True)
        image=self.root/'frozen/images/new.png';image.write_bytes(b'new original PNG fixture')
        prefix='{"evidence":[{"category":"Tray","value":60}],"selected":'
        public=dict(image_file='images/new.png',prompt='Frozen prompt',assistant_prefill=prefix)
        images={'images/new.png':dict(legacy_original_required=False,png_sha256=file_sha(image))}
        got=resolve_public(self.root,public,images,{})
        self.assertEqual(got['assistant_prefill'],prefix)
        self.assertEqual(got['image_file'],'frozen/images/new.png')
    def test_legacy_resolution_uses_original_binding_only(self):
        (self.root/'run/bound_assets/legacy').mkdir(parents=True)
        image=self.root/'run/bound_assets/legacy/OLD.png';image.write_bytes(b'historical bytes')
        public=dict(image_file='legacy/OLD.png',prompt='v3 prompt',assistant_prefill='{')
        legacy={'legacy/OLD.png':dict(sha256=file_sha(image))}
        got=resolve_public(self.root,public,{},legacy)
        self.assertEqual(got['image_file'],'run/bound_assets/legacy/OLD.png')
        image.write_bytes(b'redrawn replacement')
        with self.assertRaises(PermissionError):resolve_public(self.root,public,{},legacy)
    def test_request_identity_preserves_all_nine_keys_without_private_data(self):
        row=dict(panel='prefix10',root_id='root',structure='2_choose_1',parent='b0',
            train_seed=11031,arm='D',step=24,view='p_revtie',draw=0,case_id='case',input_id='input',
            seed=123,U_applicable=False,world='must not leak')
        endpoint=dict(endpoint_id='b0-11031-D-step-024',policy_sha256='a'*64)
        public=dict(prompt='prompt',assistant_prefill='{"evidence":',image_file='frozen/images/a.png',
                    image_sha256='b'*64,prompt_sha256='c'*64,prefill_sha256='d'*64)
        before=copy.deepcopy(row);request=request_row(row,public,endpoint,'qid')
        self.assertNotIn('world',request)
        self.assertEqual(request['seed'],123);self.assertEqual(request['train_seed'],11031)
        self.assertEqual(request['assistant_prefill'],'{"evidence":')
        self.assertEqual(row,before)
        swapped=dict(row,step=12)
        with self.assertRaises(PermissionError):request_row(swapped,public,endpoint,'qid')

if __name__=='__main__':unittest.main()
