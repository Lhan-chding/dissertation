"""Run independent reference tests, writing an explicit scope-tagged report."""
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import unittest

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
suite=unittest.defaultTestLoader.discover(str(HERE),pattern='test_modeling_reference.py')
stream=io.StringIO()
result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
log=stream.getvalue()
print(log,end='')
(HERE/'reference_test_log.txt').write_text(log,encoding='utf-8')
report={
 'scope':'INDEPENDENT_ALGEBRA_AND_SPECIFICATION_TESTS_ONLY',
 'python':platform.python_version(), 'numpy':np.__version__,
 'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
 'skipped':len(result.skipped),'passed':result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
 'successful':result.wasSuccessful(),
 'full_cpu_toy_experiment_run':False,'repository_tests_run':False,
 'new_qwen_calls':False,'new_gpu_started':False,'online_ssvc_started':False,
 'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},
 'log_sha256':hashlib.sha256(log.encode()).hexdigest(),
}
(HERE/'reference_test_result.json').write_text(json.dumps(report,indent=2)+'\n')
sys.exit(0 if result.wasSuccessful() else 1)
