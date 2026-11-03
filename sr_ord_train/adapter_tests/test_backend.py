"""Boundary and native helper equivalence tests; no model loading."""
import copy
import ast
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from adapter.backend import (NativeBackend, bind_native_kernel, file_sha,
    prefix_helpers, public_request, text_sha, verify_binding, write_once)

class BackendBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.image = self.root / 'image.png'
        self.image.write_bytes(b'fixed image fixture')
        self.request = dict(request_id='smoke-b0-0', scope='DEV_SMOKE', parent='b0',
            prompt='Read this chart.', prompt_sha256=text_sha('Read this chart.'),
            assistant_prefill='{',prefill_sha256=text_sha('{'),image_file='image.png',
            image_sha256=file_sha(self.image),seed=123)
    def tearDown(self): self.temp.cleanup()
    def test_public_projection_is_exact_and_immutable(self):
        original = copy.deepcopy(self.request)
        self.assertEqual(set(public_request(self.request,self.root)),{'prompt','image_file','image_sha256'})
        self.assertEqual(self.request,original)
    def test_private_field_rejected(self):
        for name in ['gold','world','query','model_payload','surprise']:
            with self.subTest(name=name),self.assertRaises(PermissionError):
                public_request(dict(self.request,**{name:{}}),self.root)
    def test_changed_input_and_seed_rejected(self):
        changes = [{'prompt':'changed'},{'seed':True},{'seed':2**63},
                   {'assistant_prefill':'{}','prefill_sha256':text_sha('{}')},
                   {'scope':'EVAL'},{'parent':'b2'}]
        for change in changes:
            with self.subTest(change=change),self.assertRaises(PermissionError):
                public_request(dict(self.request,**change),self.root)
    def test_image_escape_or_replacement_rejected(self):
        for image in ['/etc/passwd','../image.png']:
            with self.assertRaises(PermissionError):
                public_request(dict(self.request,image_file=image),self.root)
        (self.root/'link.png').symlink_to(self.image)
        with self.assertRaises(PermissionError):
            public_request(dict(self.request,image_file='link.png'),self.root)
        self.image.write_bytes(b'changed')
        with self.assertRaises(PermissionError): public_request(self.request,self.root)
    def test_registration_denies_historical_identity_before_import(self):
        for experiment in ['SR-F6','SR-ORDER-PROBE',None]:
            with self.assertRaises(PermissionError):
                verify_binding({'experiment_id':experiment},'b0',self.root,'TRAINING')
    def test_attempt_persistence_refuses_duplicate(self):
        path = self.root/'attempts'/'one.json'
        write_once(path,{'state':'ATTEMPTED_UNKNOWN'})
        with self.assertRaises(FileExistsError): write_once(path,{'state':'RETRY'})
        self.assertEqual(json.loads(path.read_text()),{'state':'ATTEMPTED_UNKNOWN'})
    def test_native_rebinding_preserves_code_and_original_globals(self):
        def original(): return _prefill('structured','v2')
        before = dict(original.__globals__)
        helpers = dict(_prefill=lambda p,r:'{',first_boundary_token=None,
                       decoded_protocol=None,BatchedProtocolStop=None)
        bound = bind_native_kernel(original,helpers)
        self.assertIs(original.__code__,bound.__code__)
        self.assertEqual(bound(),'{')
        self.assertEqual(original.__globals__,before)
        with self.assertRaises(PermissionError): bind_native_kernel(original,dict(helpers,torch=None))
    def test_prefix_serialization_preserves_full_text(self):
        class Stop:
            def __init__(self,*args,**kwargs): pass
        def scan(s): return len(s) if s.endswith('}') else None
        helpers=prefix_helpers('{',lambda tokenizer,tokens:''.join(tokens),Stop,scan)
        self.assertEqual(helpers['first_boundary_token'](None,['"a":1','}'],'structured'),(2,7))
        row=helpers['decoded_protocol']('"a":1}','structured')
        self.assertEqual(row['full_text'],'{"a":1}')
        self.assertEqual(row['assistant_prefill'],'{')
        self.assertEqual(row['generated_text'],'"a":1}')
        self.assertEqual(helpers['BatchedProtocolStop'](None,3,1,'structured').prefill,'{')
        with self.assertRaises(PermissionError): helpers['BatchedProtocolStop'](None,3,2,'structured')
    def test_path_scope_filter(self):
        backend=NativeBackend.__new__(NativeBackend)
        backend.parent='b0';backend.mode='TRAINING'
        backend.binding={'path_identity':{'arm':'D','train_seed':11031}}
        row={'scope':'TRAIN','parent':'b0','arm':'D','train_seed':11031}
        self.assertTrue(backend._matches_path(row))
        for change in [{'parent':'b1'},{'arm':'G'},{'train_seed':11032},{'scope':'CLARIFICATION'}]:
            self.assertFalse(backend._matches_path(dict(row,**change)))
    def test_loaded_recipe_receipt_matches_actual_generation_limit(self):
        # Execute the real nested method against a CPU-only legacy parent fixture.
        tree=ast.parse((Path(__file__).parents[1]/'adapter/backend.py').read_text())
        method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_set_recipe')
        class Config(SimpleNamespace):
            def to_dict(self): return vars(self).copy()
        class Legacy:
            def _set_recipe(self):
                self.generation_config=Config(max_new_tokens=1024)
                self.identity={'runtime_recipe':'legacy-1024','generation_config_expanded':{'max_new_tokens':1024}}
        cls=ast.ClassDef(name='RecipeFixture',bases=[ast.Name(id='Legacy',ctx=ast.Load())],
                         keywords=[],body=[copy.deepcopy(method)],decorator_list=[])
        module=ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[]))
        namespace={'Legacy':Legacy,'native':SimpleNamespace(generation_recipe=lambda:{'max_new_tokens':1536})}
        exec(compile(module,'actual_recipe_fixture','exec'),namespace)
        runtime=namespace['RecipeFixture']()
        runtime._set_recipe()
        self.assertEqual(runtime.generation_config.max_new_tokens,1536)
        self.assertEqual(runtime.identity['generation_config_expanded']['max_new_tokens'],1536)
        self.assertEqual(runtime.identity['runtime_recipe'],'SR-ORD-TRAIN-1-native-full-sequence-1536')
if __name__=='__main__': unittest.main()
