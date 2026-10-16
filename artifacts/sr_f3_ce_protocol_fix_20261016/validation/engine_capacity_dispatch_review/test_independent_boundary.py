"""CPU-only independent boundary reproduction, no live Slurm."""

import hashlib
import importlib.util
import json
from pathlib import Path

WT = Path('/Users/louis/.codex/worktrees/sr-f3-ce-20261016/dissertation-ntu')
OUT = Path('/private/tmp/srf3-engine-capacity-review')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fixtures = load('engine_dispatch_review_fixtures', WT / 'tests/sr_f3_ce/test_dispatch_registered.py')
receipts = load('engine_dispatch_review_receipts', WT / 'tests/sr_f3_ce/test_orchestration.py')
helper = fixtures.helper
registered = fixtures.registered
engine_api = fixtures.engine_api


def test_local_p0_only_boundary(helper, registered, engine_api):
    api = engine_api
    receipts.complete_local_p0(api.controller)
    before = api.controller.status()
    assert before['completed'] == ['p0'] and not before['attempts']
    assert set(before['ready']) == helper.ENGINES - {'p0'}
    driver = fixtures.make_driver(helper, registered, api)
    error = None
    result = None
    try:
        result = driver.poll()
    except (PermissionError, RuntimeError) as exc:
        error = repr(exc)
    intents = fixtures.engine_intents(helper, registered)
    evidence = dict(
        review='LOCAL_P0_ONLY_REAL_CONTROLLER_FAKE_SLURM_BOUNDARY',
        source_sha256=hashlib.sha256((WT / 'scripts/sr_f3_ce/dispatch_registered.py').read_bytes()).hexdigest(),
        p0_before=before['completed'],
        real_p0_allocations_before=len(before['attempts']),
        returned=result,
        error=error,
        engine_intents=[row['task_ids'] for row in intents],
        sbatch_calls=sum(call[0] == 'sbatch' for call in api.slurm.calls),
        science_freezes=len(api.freezes),
        server_commands=0,
        scientific_source_modified=False,
    )
    (OUT / ('LOCAL_P0_' + evidence['source_sha256'][:12] + '.json')).write_text(json.dumps(evidence, indent=2) + '\n')
    assert error is not None and not intents, evidence
