"""Offline regressions: no SSH, scheduler, model, or original evidence mutations."""
import contextlib
import fcntl
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'satdose_inspect_interrupted_submission.py'
spec = importlib.util.spec_from_file_location('interrupted_inspector_test', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class InspectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve() / 'root'
        self.root.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def response(self, stdout='', stderr='', code=0):
        return subprocess.CompletedProcess([], code, stdout, stderr)

    def test_absence_checks_symlink_ancestors_even_when_leaf_missing(self):
        (self.root / 'real').mkdir()
        (self.root / 'alias').symlink_to(self.root / 'real', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Noncanonical'):
            m.absent(self.root / 'alias/missing/leaf')
        self.assertEqual(m.absent(self.root / 'real/missing/leaf')['state'], 'ABSENT_LEXISTS_FALSE')

    def test_dangling_leaf_is_occupied(self):
        (self.root / 'slot').symlink_to('missing')
        with self.assertRaisesRegex(ValueError, 'Occupied'):
            m.absent(self.root / 'slot')

    def test_existing_file_slot_is_occupied(self):
        (self.root / 'slot').write_text('')
        with self.assertRaisesRegex(ValueError, 'Occupied'):
            m.absent(self.root / 'slot')

    def test_scheduler_valid_empty_history_and_other_queue(self):
        calls = []
        def run(argv, **kwargs):
            calls.append(argv)
            return self.response('1741000|unrelated|RUNNING\n' if argv[0] == 'squeue' else '')
        result = m.scheduler(run)
        self.assertEqual(len(result), 2)
        self.assertIn('--format=%i|%200j|%T', calls[0])
        self.assertIn('--name=' + m.JOB_NAME, calls[1])
        self.assertIn('--starttime=2026-11-12T00:00:00', calls[1])
        self.assertEqual([x[0] for x in calls], ['squeue', 'sacct'])

    def test_scheduler_rejects_exact_job_any_state(self):
        for state in ('PENDING', 'COMPLETED', 'CANCELLED'):
            with self.subTest(state=state), self.assertRaisesRegex(ValueError, 'Matching'):
                m.scheduler(lambda *a, **k: self.response('1741000|' + m.JOB_NAME + '|' + state))

    def test_scheduler_rejects_malformed_failure_warning_truncated(self):
        for result in (self.response('invalid'), self.response(code=1), self.response(stderr='warning'),
                       self.response('1741000|LH-SATDOSE-TRAIN+|PENDING')):
            with self.subTest(result=result), self.assertRaises(ValueError):
                m.scheduler(lambda *a, **k: result)

    def test_exact_accounting_filter_unexpected_row_is_unknown(self):
        def run(argv, **kwargs):
            return self.response('1741000|other|COMPLETED' if argv[0] == 'sacct' else '')
        with self.assertRaisesRegex(ValueError, 'unexpected'):
            m.scheduler(run)

    def make_locks(self):
        for name in ('FIRST_TWO_LAUNCHER.lock', 'DIAGNOSTIC_WORKER.lock'):
            (self.root / name).write_bytes(b'')

    def test_existing_locks_order_modes_and_no_content_write(self):
        self.make_locks()
        with m.leases(self.root) as locks:
            result = m.stable_locks(locks)
            self.assertEqual([x['mode'] for x in result], ['SHARED', 'EXCLUSIVE'])
            self.assertTrue(all(x['inode'] > 0 for x in result))
        self.assertEqual([p.read_bytes() for p in self.root.iterdir()], [b'', b''])

    def test_busy_lock_fails_without_creating_anything(self):
        self.make_locks()
        fd = os.open(self.root / 'DIAGNOSTIC_WORKER.lock', os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError), m.leases(self.root):
                self.fail('Busy lock accepted')
        finally:
            os.close(fd)
        self.assertEqual(len(list(self.root.iterdir())), 2)

    def test_replaced_lock_fails(self):
        self.make_locks()
        with m.leases(self.root) as locks:
            path = self.root / 'DIAGNOSTIC_WORKER.lock'
            path.rename(self.root / 'old')
            path.write_bytes(b'')
            with self.assertRaisesRegex(ValueError, 'replaced'):
                m.stable_locks(locks)

    def test_missing_lock_not_created(self):
        with self.assertRaises(FileNotFoundError), m.leases(self.root):
            self.fail('Missing lock accepted')
        self.assertEqual(list(self.root.iterdir()), [])

    def make_process(self, pid, cmdline, comm='python3', state='S'):
        process = self.root / str(pid)
        process.mkdir()
        uid = os.getuid()
        (process / 'status').write_text('Uid:\t' + '\t'.join([str(uid)] * 4) + '\n')
        (process / 'stat').write_text(f'{pid} ({comm}) {state} 1 ' + ' '.join(['0'] * 17) + ' 123 0\n')
        (process / 'cmdline').write_bytes(cmdline)
        (process / 'comm').write_text(comm + '\n')
        return process

    def test_own_process_census_redacts_secret_args_and_self(self):
        self.make_process(1, b'python3\0unrelated\0SECRET_TOKEN=abc\0')
        self.make_process(2, str(m.ROOT).encode())
        result = m.processes(dict(original_remote_sha256=m.digest(b'original')), self.root, self_pid=2)
        encoded = json.dumps(result)
        self.assertNotIn('SECRET_TOKEN', encoded)
        self.assertNotIn(str(m.ROOT), encoded)
        self.assertEqual(len(result['processes']), 2)
        self.assertIn('NOT_GLOBAL', result['scope'])

    def test_exact_original_source_candidate_is_rejected(self):
        self.make_process(1, b'python3\0-c\0original source\0')
        self.make_process(2, b'inspector\0')
        with self.assertRaisesRegex(ValueError, 'Candidate'):
            m.processes(dict(original_remote_sha256=m.digest(b'original source')), self.root, self_pid=2)

    def test_project_substring_candidate_is_rejected(self):
        self.make_process(1, b'python3\0' + str(m.ROOT).encode() + b'/other.py\0')
        self.make_process(2, b'inspector\0')
        with self.assertRaisesRegex(ValueError, 'Candidate'):
            m.processes(dict(original_remote_sha256=m.digest(b'original')), self.root, self_pid=2)

    def test_root_owned_proc_directory_does_not_hide_own_uid_candidate(self):
        process = self.make_process(1, b'python3\0original\0')
        self.make_process(2, b'inspector\0')
        real_stat = Path.stat
        from types import SimpleNamespace
        def modified_stat(path, *args, **kwargs):
            if path == process:
                return SimpleNamespace(st_uid=os.getuid() + 1)
            return real_stat(path, *args, **kwargs)
        with patch.object(Path, 'stat', modified_stat), self.assertRaisesRegex(ValueError, 'Candidate'):
            m.processes(dict(original_remote_sha256=m.digest(b'original')), self.root, self_pid=2)

    def test_other_uid_cmdline_never_read_and_unknown_uid_fails(self):
        process = self.make_process(1, b'private other-user args')
        (process / 'status').write_text('Uid:\t' + '\t'.join([str(os.getuid() + 1)] * 4) + '\n')
        (process / 'cmdline').unlink()
        self.make_process(2, b'inspector\0')
        result = m.processes(dict(original_remote_sha256=m.digest(b'original')), self.root, self_pid=2)
        self.assertEqual([p['pid'] for p in result['processes']], [2])
        (process / 'status').unlink()
        with self.assertRaisesRegex(ValueError, 'Unreadable'):
            m.processes(dict(original_remote_sha256=m.digest(b'original')), self.root, self_pid=2)

    def test_unreadable_or_empty_live_process_is_unknown(self):
        process = self.make_process(1, b'')
        with self.assertRaisesRegex(ValueError, 'Empty live'):
            m.processes(dict(original_remote_sha256=m.digest(b'original')), self.root, self_pid=1 + 1)
        (process / 'cmdline').unlink()
        with self.assertRaisesRegex(ValueError, 'Unreadable'):
            m.processes(dict(original_remote_sha256=m.digest(b'original')), self.root, self_pid=1)

    def test_inventory_exact84_and_extra_or_changed_rejected(self):
        code = self.root / 'train_code_71018e5'
        code.mkdir()
        expected = {}
        for i in range(84):
            path = code / f'{i}.py'
            path.write_bytes(str(i).encode())
            expected[path.name] = m.digest(path.read_bytes())
        self.assertEqual(m.inventory(self.root, expected)['file_count'], 84)
        (code / '0.py').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'source84'):
            m.inventory(self.root, expected)

    def test_holds_only_active_control_families_and_target_slots(self):
        historical = self.root / 'unrelated_historical_audit'
        historical.mkdir(); (historical / 'HOLD.json').write_text('{}')
        self.assertEqual(len(m.slots_and_holds(self.root)), 4)
        target = self.root / 'training_control'
        target.mkdir(); (target / 'HOLD.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'HOLD'):
            m.slots_and_holds(self.root)

    def test_default_local_plan_cannot_call_ssh(self):
        with patch.object(m, 'local_payload', return_value={}), patch.object(m.subprocess, 'run') as run:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(m.main([]), 0)
            run.assert_not_called()
            result = json.loads(output.getvalue())
            self.assertEqual(result['permission'], m.PERMISSION)
            self.assertTrue(result['original_unknown_preserved'])

    def test_original_crlf_stderr_projection_keeps_raw_digest(self):
        data = b'banner\r\nconnection closed\r\n'
        before = m.digest(data)
        self.assertEqual(m.text_read(data), 'banner\nconnection closed\n')
        self.assertEqual(m.digest(data), before)
        self.assertNotEqual(before, m.digest(m.text_read(data).encode()))

    def test_partial_or_forged_success_report_rejected(self):
        with self.assertRaises(ValueError):
            m.validate_report({'status': 'PRE_SUBMISSION_ABSENCE_VERIFIED'}, 'source')

    def test_timeout_preserves_rawlogs_and_unknown_without_retry(self):
        output = self.root / 'inspection'
        def run(*args, **kwargs):
            kwargs['stdout'].write(b'partial observation')
            kwargs['stderr'].write(b'closed')
            raise subprocess.TimeoutExpired('redacted-test-command', 300)
        with patch.object(m, 'local_payload', return_value={}), patch.object(m.subprocess, 'run', side_effect=run) as runner:
            with self.assertRaises(subprocess.TimeoutExpired):
                m.main(['--execute', '--local-dir', str(output)])
            self.assertEqual(runner.call_count, 1)
            error = json.loads((output / 'ERROR.json').read_bytes())
            self.assertEqual(error['status'], 'INSPECTION_FAILED_UNKNOWN')
            self.assertEqual(error['raw_logs']['SSH_STDOUT.raw']['sha256'], m.digest(b'partial observation'))
            self.assertTrue(error['original_unknown_preserved'])
            self.assertFalse((output / 'RECEIPT.json').exists())

    def test_wrong_or_short_hostname_rejected_and_observed_before_locks(self):
        payload = dict(selection=m.SELECTION, inspector_sha256='source', failed_receipt_sha256=m.RECEIPT_SHA,
                       submitter_sha256=m.SUBMITTER_SHA, original_intent_sha256=m.INTENT_SHA,
                       expected_hostname=m.EXPECTED_HOSTNAME)
        for hostname in ('login-3.cluster02.eee.ntu.edu.sg', 'login-2', 'login-2.cluster02.eee.ntu.edu.sg.attacker'):
            with self.subTest(hostname=hostname), patch.object(m, 'ROOT', self.root), \
                 patch.object(m, 'INSPECTOR_SOURCE_SHA', 'source'), \
                 patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}), \
                 patch.object(m.socket, 'gethostname', return_value=hostname), patch.object(m, 'leases') as locks:
                with self.assertRaisesRegex(ValueError, 'exact login FQDN'):
                    m.remote_inspect(payload)
                locks.assert_not_called()
                self.assertEqual(m.REMOTE_PROGRESS['observed_hostname'], hostname)

    def test_remote_inspection_emits_success_only_under_both_locks(self):
        self.make_locks()
        p = dict(selection=m.SELECTION, inspector_sha256='source', failed_receipt_sha256=m.RECEIPT_SHA,
                 original_intent_sha256=m.INTENT_SHA, submitter_sha256=m.SUBMITTER_SHA,
                 expected_hostname=m.EXPECTED_HOSTNAME, expected_username='varun024', original_payload={
                     'sources': {}, 'preparation_ref': {'path': 'pre', 'sha256': 'x'},
                     'original': {'path': 'original', 'sha256': 'x'},
                     'canary_spec': {'path': 'canary', 'sha256': 'x'},
                     'preparation': {key: {'path': key, 'sha256': 'x'}
                                     for key in ('intent', 'release', 'storage_budget', 'risk')}})
        import pwd
        from types import SimpleNamespace
        def emit(value, *, flush):
            self.assertTrue(flush)
            record = json.loads(value)
            self.assertEqual(record['status'], 'PRE_SUBMISSION_ABSENCE_VERIFIED')
            fd = os.open(self.root / 'DIAGNOSTIC_WORKER.lock', os.O_RDONLY)
            try:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(fd)
        real_read = Path.read_text
        def read(path, *args, **kwargs):
            if str(path) == '/proc/sys/kernel/random/boot_id':
                return 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'
            return real_read(path, *args, **kwargs)
        with patch.object(m, 'ROOT', self.root), patch.object(m, 'INSPECTOR_SOURCE_SHA', 'source'), \
             patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}), \
             patch.object(m.socket, 'gethostname', return_value=m.EXPECTED_HOSTNAME), \
             patch.object(pwd, 'getpwuid', return_value=SimpleNamespace(pw_name='varun024')), \
             patch.object(Path, 'read_text', read), patch.object(m, 'inventory', return_value={'file_count': 84}), \
             patch.object(m, 'checked'), patch.object(m, 'processes', return_value={}), \
             patch.object(m, 'scheduler', return_value=[]), patch('builtins.print', side_effect=emit):
            result = m.remote_inspect(p)
        self.assertTrue(result['original_unknown_preserved'])
        self.assertEqual(result['permission'], m.PERMISSION)
        self.assertIn('no remote execution-host/boot-ID observation was recorded', ' '.join(result['limits']))


if __name__ == '__main__':
    unittest.main()
