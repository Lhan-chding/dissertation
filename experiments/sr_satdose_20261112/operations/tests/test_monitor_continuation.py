"""Read-only monitor offline fixtures; never SSH or scheduler access."""
import base64
from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

OPERATIONS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OPERATIONS))
import satdose_audit_continuation as a
import satdose_monitor_continuation as m


class Monitor(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='satdose-monitor-OFFLINE-'); self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name).resolve(); self.root = self.base / 'prepared'; self.root.mkdir()
        self.patcher = patch.object(a, 'ROOT', self.root); self.patcher.start(); self.addCleanup(self.patcher.stop)
        self.s = a.selection(11031, 'T0', 3); self.job = 9999999
        schedule = a.schedule_for(self.s); schedule.mkdir(parents=True)
        self.release = dict(path='execution_releases/FAKE.json', sha256='a'*64)
        accepted = dict(job_id=self.job, selection=self.s, release=self.release, job_name=a.job_name(self.s))
        a.write_once(schedule / 'SUBMISSION.json', accepted)
        submitted = dict(job_id=self.job, selection=self.s, release=self.release, submission=a.reference(schedule / 'SUBMISSION.json'))
        a.write_once(schedule / 'RELEASE_RESULT.json', submitted)
        self.payload = dict(selection=self.s, job_id=self.job, submitted=submitted,
                            submission_ref=a.reference(schedule / 'RELEASE_RESULT.json'))
        self.scheduler = dict(accounting=dict(returncode=0, stdout_b64=''), queue=dict(returncode=0, stdout_b64=''))

    def observe(self):
        with patch.object(a, 'scheduler', return_value=self.scheduler): return m.observe(a, self.payload)

    def make_draw(self, step=3, complete=True):
        slot = a.slot_for(self.s); directory = slot / 'audit/steps' / f'step-{step:03d}'; directory.mkdir(parents=True, exist_ok=True)
        if not (slot / 'CLAIM.json').exists():
            a.write_once(slot / 'CLAIM.json', dict(selection=self.s, release=self.release, allocation=dict(job_id=str(self.job))))
        request = f'OFFLINE-{step}'; stem = a.digest(request.encode())
        a.write_once(directory / (stem + '.BEGIN.json'), dict(request_id=request,
            endpoint=dict(start_seed=11031, arm='T0', step=step-1)))
        a.write_once(directory / (stem + '.CAPTURE.json'), {})
        if complete: a.write_once(directory / (stem + '.COMPLETE.json'), {})
        return directory, stem

    def test_pending_has_no_worker_or_log_requirement(self):
        value = self.observe(); self.assertFalse(value['slot_exists']); self.assertEqual(value['log_tails'], {})

    def test_counts_map_sampled_endpoint_to_actual_training_step(self):
        self.make_draw(3); self.make_draw(4, False)
        value = self.observe()
        self.assertEqual([value['per_step']['3'][x] for x in ('BEGIN','CAPTURE','COMPLETE')], [1,1,1])
        self.assertEqual([value['per_step']['4'][x] for x in ('BEGIN','CAPTURE','COMPLETE')], [1,1,0])
        self.assertEqual(len(value['unresolved_capture_paths']), 1)

    def test_foreign_claim_job_rejected(self):
        self.make_draw(); path = a.slot_for(self.s) / 'CLAIM.json'
        value = a.loads(path.read_bytes()); value['allocation']['job_id'] = '123'; path.write_bytes(a.canonical(value))
        with self.assertRaises(ValueError): self.observe()

    def test_foreign_draw_endpoint_rejected(self):
        directory, stem = self.make_draw(); path = directory / (stem + '.BEGIN.json')
        value = a.loads(path.read_bytes()); value['endpoint']['arm'] = 'T1'; path.write_bytes(a.canonical(value))
        with self.assertRaises(ValueError): self.observe()

    def test_orphan_capture_rejected(self):
        directory, _ = self.make_draw(); a.write_once(directory / ('b'*64 + '.CAPTURE.json'), {})
        with self.assertRaises(ValueError): self.observe()

    def test_symlink_log_not_read(self):
        outside = self.base / 'unrelated.log'; outside.write_bytes(b'private data')
        (a.schedule_for(self.s) / f'slurm-{self.job}.out').symlink_to(outside)
        with self.assertRaises(ValueError): self.observe()

    def test_log_tail_preserves_raw_bytes_and_only_explicit_job(self):
        path = a.schedule_for(self.s) / f'slurm-{self.job}.out'; path.write_bytes(b'x'*10000 + b'\xff')
        (path.parent / 'slurm-123.out').write_bytes(b'FOREIGN LOG MUST NEVER READ')
        value = self.observe(); self.assertEqual(set(value['log_tails']), {path.name})
        self.assertEqual(base64.b64decode(value['log_tails'][path.name]['tail_b64']), path.read_bytes()[-8192:])

    def test_scientific_hold_preserved_as_incident(self):
        directory = self.root / 'training_control'; directory.mkdir()
        a.write_once(directory / 'HOLD.json', dict(status='GLOBAL_SCIENTIFIC_HOLD', reason='OFFLINE FIXTURE'))
        value = self.observe(); self.assertTrue(value['incidents']); self.assertFalse(value['next_stage_authorized'])

    def test_foreign_scheduler_row_rejected(self):
        self.scheduler['accounting']['stdout_b64'] = base64.b64encode(b'123|x|x|x|x|x|x|x|x|x').decode()
        with self.assertRaises(ValueError): self.observe()

    def test_foreign_queue_row_rejected(self):
        self.scheduler['queue']['stdout_b64'] = base64.b64encode(b'123|foreign|RUNNING|node').decode()
        with self.assertRaises(ValueError): self.observe()

    def test_dependency_source_pin_matches(self):
        helper, raw = m.load_auditor(); self.assertEqual(a.digest(raw), m.AUDITOR_SHA)
        self.assertEqual(helper.SUBMITTER_SHA, a.SUBMITTER_SHA)

    def argv(self):
        return ['--submission-receipt', str(self.base / 'FAKE.json'), '--submission-sha256', 'a'*64,
                '--job-id', str(self.job), '--start-seed', '11031', '--arm', 'T0', '--first-step', '3',
                '--output-dir', str(self.base)]

    def test_default_plan_zero_subprocess_zero_local_attempt(self):
        before = set(self.base.iterdir())
        with patch.object(m, 'load_auditor', return_value=(a, Path(a.__file__).read_bytes())), \
                patch.object(a, 'local_payload', return_value=self.payload), \
                patch.object(a.subprocess, 'run', side_effect=AssertionError('No subprocess')), redirect_stdout(io.StringIO()):
            self.assertEqual(m.main(self.argv()), 0)
        self.assertEqual(set(self.base.iterdir()), before)

    def test_network_failure_timestamped_raw_retention(self):
        def fake(source, mode, data, out, err):
            self.assertEqual(mode, '--remote-monitor'); out.write_bytes(b'partial\xff'); err.write_bytes(b'SSH failed'); return 255
        with patch.object(m, 'load_auditor', return_value=(a, Path(a.__file__).read_bytes())), \
                patch.object(a, 'local_payload', return_value=self.payload), patch.object(a, 'ssh_call', side_effect=fake), \
                redirect_stdout(io.StringIO()): self.assertEqual(m.main(self.argv()+['--execute']), 2)
        directories = list(self.base.glob('MONITOR_CONTINUATION_*')); self.assertEqual(len(directories), 1)
        self.assertEqual((directories[0] / 'STDOUT.log').read_bytes(), b'partial\xff')
        self.assertTrue((directories[0] / 'MONITOR_SOURCE.py').is_file())
        self.assertTrue((directories[0] / 'RECEIPT.json').is_file())


if __name__ == '__main__': unittest.main()
