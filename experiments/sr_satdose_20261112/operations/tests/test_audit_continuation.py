"""Offline boundary tests; fixtures are synthetic, never production audit evidence.

No SSH, scheduler, model, training or original project preflight is run. The
native full-chain/risk composition is source-reviewed, not claimed executed here.
"""
import base64
from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import shlex
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

OPERATIONS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OPERATIONS))
import satdose_audit_continuation as a


def fake_ref(name): return dict(path='execution_releases/' + name, sha256='a' * 64)


def fake_payload():
    return dict(selection=a.selection(11031, 'T0', 3), job_id=9999999,
                operation_source_sha256=a.digest(b'OFFLINE SOURCE'))


def accounting(payload, state='COMPLETED', exit_code='0:0'):
    job = payload['job_id']; s = payload['selection']
    rows = [[str(job), state, exit_code, a.job_name(s), 'rose', a.QOS, 'cluster02',
             '2026-11-17T00:00:00', '2026-11-17T01:00:00', 'gpu02']]
    for suffix in ('batch', 'extern', '0'):
        rows.append([f'{job}.{suffix}', state, exit_code, suffix, 'rose', '', '',
                     '2026-11-17T00:00:01', '2026-11-17T00:59:59', 'gpu02'])
    encode = lambda value: base64.b64encode(value.encode()).decode()
    return dict(accounting=dict(returncode=0, stdout_b64=encode('\n'.join('|'.join(r) for r in rows))),
                queue=dict(returncode=0, stdout_b64=encode('')))


class Boundaries(unittest.TestCase):
    def test_compile_both_scripts(self):
        for name in ('satdose_audit_continuation.py', 'satdose_monitor_continuation.py'):
            path = OPERATIONS / name; compile(path.read_bytes(), str(path), 'exec')

    def test_strict_json_duplicate_and_nan(self):
        for value in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            with self.assertRaises(ValueError): a.loads(value)

    def test_only_frozen_selections(self):
        for first, last in a.CHUNKS.items(): self.assertEqual(a.selection(11032, 'T1', first)['last_step'], last)
        for args in [(True, 'T0', 3), (11031, 'T2', 3), (11031, 'T0', 4)]:
            with self.assertRaises(ValueError): a.selection(*args)

    def test_all_exact_scheduler_steps(self):
        p = fake_payload(); value = accounting(p)
        self.assertEqual(len(a.scheduler_complete(value, p['job_id'], p['selection'], 'gpu02')), 4)

    def test_scheduler_failure_and_active_rejected(self):
        p = fake_payload()
        for state, code in [('COMPLETED', '1:0'), ('RUNNING', '0:0'), ('FAILED', '0:0')]:
            with self.assertRaises(ValueError): a.scheduler_complete(accounting(p, state, code), p['job_id'], p['selection'])
        value = accounting(p); value['queue']['stdout_b64'] = base64.b64encode(b'9999999|active').decode()
        with self.assertRaises(ValueError): a.scheduler_complete(value, p['job_id'], p['selection'])

    def test_scheduler_missing_duplicate_foreign_and_node_rejected(self):
        p = fake_payload(); raw = base64.b64decode(accounting(p)['accounting']['stdout_b64']).decode(); lines = raw.splitlines()
        for bad in ['\n'.join(lines[:-1]), raw + '\n' + lines[0], raw.replace('9999999.0', '9999998.0'),
                    raw.replace('gpu02', 'gpu99', 1)]:
            value = accounting(p); value['accounting']['stdout_b64'] = base64.b64encode(bad.encode()).decode()
            with self.assertRaises(ValueError): a.scheduler_complete(value, p['job_id'], p['selection'], 'gpu02')

    def test_actual_cluster_purged_job_requires_exact_error_and_complete_sacct(self):
        p = fake_payload(); value = accounting(p)
        value['queue'].update(returncode=1, stderr_b64=base64.b64encode(
            b'slurm_load_jobs error: Invalid job id specified\n').decode())
        self.assertEqual(len(a.scheduler_complete(value, p['job_id'], p['selection'])), 4)
        for message in (b'Connection refused\n', b'slurm_load_jobs error: Invalid job id specified', b''):
            bad = copy.deepcopy(value); bad['queue']['stderr_b64'] = base64.b64encode(message).decode()
            with self.assertRaises(ValueError): a.scheduler_complete(bad, p['job_id'], p['selection'])
        bad = copy.deepcopy(value); bad['accounting']['returncode'] = 1
        with self.assertRaises(ValueError): a.scheduler_complete(bad, p['job_id'], p['selection'])
        bad = accounting(p, 'FAILED', '2:0'); bad['queue'] = value['queue']
        with self.assertRaises(ValueError): a.scheduler_complete(bad, p['job_id'], p['selection'])

    def test_scheduler_wrong_name_account_qos(self):
        p = fake_payload()
        for old, new in [(a.job_name(p['selection']), 'OTHER-JOB'), ('rose', 'other'), (a.QOS, 'normal')]:
            value = accounting(p); raw = base64.b64decode(value['accounting']['stdout_b64']).replace(old.encode(), new.encode())
            value['accounting']['stdout_b64'] = base64.b64encode(raw).decode()
            with self.assertRaises(ValueError): a.scheduler_complete(value, p['job_id'], p['selection'])

    def test_worker_stdout_exact_one_result(self):
        result = dict(status='CONTINUATION_CHUNK_COMPLETE', value=3)
        raw = b'library diagnostics\n' + a.canonical(result)
        a.worker_stdout(raw, result)
        for bad in [b'', raw + raw, a.canonical(dict(status='HOLD_CONTINUATION_CONTROL'))]:
            with self.assertRaises(ValueError): a.worker_stdout(bad, result)

    def test_process_incarnation_not_pid_only(self):
        value = dict(host='gpu02', boot_id='aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee', pid=22, start_ticks=100)
        a.validate_process(value)
        for key in value:
            bad = value.copy(); bad.pop(key)
            with self.assertRaises(ValueError): a.validate_process(bad)

    def test_marker_duplicate_rejected(self):
        raw = a.MARKER + b'{}\n'
        self.assertEqual(a.marker(raw, a.MARKER), {})
        with self.assertRaises(ValueError): a.marker(raw + raw, a.MARKER)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='satdose-audit-OFFLINE-'); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()

    def test_symlink_and_changed_bytes_rejected(self):
        path = self.root / 'file'; path.write_bytes(b'original'); ref = a.reference(path)
        link = self.root / 'link'; link.symlink_to(path)
        with self.assertRaises(ValueError): a.ordinary(link)
        path.write_bytes(b'changed')
        with self.assertRaises(ValueError): a.pinned(ref)

    def test_write_once_and_lease_replacement(self):
        path = self.root / 'lock'; path.write_bytes(b'')
        with a.idle_lock(path) as stable:
            path.rename(self.root / 'old'); path.write_bytes(b'')
            with self.assertRaises(ValueError): stable()
            # Restore inode so context manager boundary also succeeds.
            path.unlink(); (self.root / 'old').rename(path)
        with self.assertRaises(FileExistsError): a.write_once(path, b'overwrite')

    def test_busy_lease_rejected(self):
        path = self.root / 'lock'; path.write_bytes(b'')
        fd = os.open(path, os.O_RDONLY)
        try:
            a.fcntl.flock(fd, a.fcntl.LOCK_EX | a.fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                with a.idle_lock(path): pass
        finally: os.close(fd)

    def make_bundle(self, success=False):
        self.payload = fake_payload(); self.fake_root = self.root / 'prepared'; self.fake_root.mkdir(exist_ok=True)
        with patch.object(a, 'ROOT', self.fake_root):
            directory = a.audit_for(self.payload['selection']); a.mkdir_new(directory)
            a.write_once(directory / 'OPERATION_SOURCE.py', b'OFFLINE SOURCE')
            a.write_once(directory / 'INTENT.json', dict(payload=self.payload))
            if success:
                for name in a.SUCCESS_FILES - {'PUBLICATION.json', 'OPERATION_SOURCE.py', 'INTENT.json', 'AUDIT_REPORT.json'}:
                    a.write_once(directory / name, {})
                a.write_once(directory / 'AUDIT_REPORT.json', dict(status='CONTINUATION_INDEPENDENT_AUDIT_COMPLETE',
                    job_id=self.payload['job_id'], selection=self.payload['selection'], permission=a.PERMISSION,
                    next_stage_authorized=False, full_current_post_risk_exactly_reproduced=True,
                    global_scientific_hold=False, direct_compute_node_process_death_observed=False))
                a.write_once(directory / 'PUBLICATION.json', dict(status='CONTINUATION_AUDIT_PUBLISHED',
                    report=a.reference(directory / 'AUDIT_REPORT.json'),
                    files={p.name: a.reference(p) for p in directory.iterdir()}))
            else: a.write_once(directory / 'ERROR.json', dict(status='OFFLINE_BLOCKED_FIXTURE'))
            manifest = a.publish_bundle(directory, self.payload, success)
        return directory, manifest

    def test_complete_bundle_success_and_blocked_evidence(self):
        directory, manifest = self.make_bundle()
        with patch.object(a, 'ROOT', self.fake_root):
            value = a.validate_bundle(directory / 'BUNDLE.tar.gz', manifest, self.payload, self.root / 'download')
        self.assertTrue(value['bundle_verified']); self.assertFalse(value['audit_success'])
        self.assertTrue((self.root / 'download/ERROR.json').is_file())

    def test_complete_success_semantics(self):
        directory, manifest = self.make_bundle(True)
        with patch.object(a, 'ROOT', self.fake_root):
            self.assertTrue(a.validate_bundle(directory / 'BUNDLE.tar.gz', manifest, self.payload, self.root / 'download')['audit_success'])

    def test_truncated_bundle_retained_no_extract(self):
        directory, manifest = self.make_bundle(); raw = (directory / 'BUNDLE.tar.gz').read_bytes()[:20]
        bad = self.root / 'partial.tar.gz'; bad.write_bytes(raw)
        with patch.object(a, 'ROOT', self.fake_root), self.assertRaisesRegex(ValueError, 'Partial'):
            a.validate_bundle(bad, manifest, self.payload, self.root / 'download')
        self.assertEqual(bad.read_bytes(), raw); self.assertFalse((self.root / 'download').exists())

    def test_manifest_foreign_partial_duplicate_path_rejected(self):
        _, manifest = self.make_bundle(True)
        mutations = []
        wrong = copy.deepcopy(manifest); wrong['job_id'] += 1; mutations.append(wrong)
        wrong = copy.deepcopy(manifest); wrong['files'].pop('AUDIT_REPORT.json'); mutations.append(wrong)
        wrong = copy.deepcopy(manifest); wrong['bundle']['path'] = '/other/BUNDLE.tar.gz'; mutations.append(wrong)
        wrong = copy.deepcopy(manifest); wrong['files']['../evil'] = wrong['files'].pop('INCIDENTS.json'); mutations.append(wrong)
        with patch.object(a, 'ROOT', self.fake_root):
            for value in mutations:
                with self.assertRaises(ValueError): a.validate_manifest(value, self.payload)

    def test_tar_symlink_even_with_matching_bundle_hash_rejected(self):
        directory, manifest = self.make_bundle()
        path = self.root / 'symlink.tar.gz'
        with tarfile.open(path, 'w:gz') as archive:
            for name, ref in manifest['files'].items():
                info = tarfile.TarInfo(name); info.size = ref['bytes']
                if name == 'ERROR.json': info.type = tarfile.SYMTYPE; info.linkname = '/outside'
                archive.addfile(info, io.BytesIO((directory / name).read_bytes()))
        manifest['bundle'].update(sha256=a.sha(path), bytes=path.stat().st_size)
        with patch.object(a, 'ROOT', self.fake_root), self.assertRaises(ValueError):
            a.validate_bundle(path, manifest, self.payload, self.root / 'download')


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='satdose-driver-OFFLINE-'); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve(); self.payload = fake_payload()
        self.output = self.root / 'attempt'

    def argv(self):
        return ['--submission-receipt', str(self.root / 'FAKE.json'), '--submission-sha256', 'a' * 64,
                '--job-id', '9999999', '--start-seed', '11031', '--arm', 'T0', '--first-step', '3', '--local-dir', str(self.output)]

    def test_default_plan_has_zero_subprocess_and_files(self):
        with patch.object(a, 'local_payload', return_value=copy.deepcopy(self.payload)), \
                patch.object(a.subprocess, 'run', side_effect=AssertionError('No subprocess allowed')), redirect_stdout(io.StringIO()):
            self.assertEqual(a.main(self.argv()), 0)
        self.assertFalse(self.output.exists())

    def test_real_local_bootstrap_large_stdin_payload_no_large_argv(self):
        # Execute only a tiny inert probe through the real bootstrap on this Mac.
        # The SSH transport itself is replaced; no project runtime is imported.
        original_run = a.subprocess.run
        source = b"import sys,json,base64; value=json.loads(base64.b64decode(sys.argv[2])); print(len(value['probe']))\n"
        data = dict(probe='x' * 250000)
        def transport(argv, **kwargs):
            self.assertEqual(argv[0], 'ssh'); self.assertLess(len(argv[-1]), 1500)
            parts = shlex.split(argv[-1]); bootstrap = parts[parts.index('-c')+1]
            self.assertEqual(parts[-1], '--remote-worker')
            return original_run([sys.executable, '-I', '-B', '-c', bootstrap, '--remote-worker'], **kwargs)
        out = self.root / 'probe.out'; err = self.root / 'probe.err'
        with patch.object(a.subprocess, 'run', side_effect=transport):
            self.assertEqual(a.ssh_call(source, '--remote-worker', data, out, err), 0)
        self.assertEqual(out.read_bytes(), b'250000\n'); self.assertEqual(err.read_bytes(), b'')

    def test_network_failure_raw_retained_and_no_retry(self):
        calls = []
        def fake(source, mode, data, out, err):
            calls.append(mode)
            self.assertTrue((self.output / 'OPERATION_SOURCE.py').is_file())
            self.assertTrue((self.output / 'LOCAL_INTENT.json').is_file())
            out.write_bytes(b'partial\xff'); err.write_bytes(b'network failed\xfe'); return 255
        with patch.object(a, 'local_payload', return_value=copy.deepcopy(self.payload)), patch.object(a, 'ssh_call', side_effect=fake), redirect_stdout(io.StringIO()):
            self.assertEqual(a.main(self.argv() + ['--execute']), 2)
            with self.assertRaises(ValueError): a.main(self.argv() + ['--execute'])
        self.assertEqual(calls, ['--remote-worker'])
        self.assertEqual((self.output / 'AUDIT_STDOUT.log').read_bytes(), b'partial\xff')
        self.assertEqual((self.output / 'AUDIT_STDERR.log').read_bytes(), b'network failed\xfe')
        self.assertTrue((self.output / 'LOCAL_RECEIPT.json').is_file())

    def test_download_failure_retains_manifest_recovery_never_audits(self):
        payload = copy.deepcopy(self.payload); payload['operation_source_sha256'] = a.sha(Path(a.__file__))
        source_ref = dict(path=str(a.audit_for(payload['selection']) / 'OPERATION_SOURCE.py'),
                          sha256=payload['operation_source_sha256'], bytes=1)
        directory = a.audit_for(payload['selection'])
        manifest = dict(schema='sr-satdose-continuation-audit-manifest-v1', success=True,
            status='CONTINUATION_AUDIT_PUBLICATION_COMPLETE', permission=a.PERMISSION, job_id=payload['job_id'],
            selection=payload['selection'], payload_sha256=a.digest(a.canonical(payload)),
            operation_source_sha256=payload['operation_source_sha256'], next_stage_authorized=False,
            files={n: dict(path=str(directory / n), sha256='a'*64, bytes=1) for n in a.SUCCESS_FILES},
            bundle=dict(path=str(directory / 'BUNDLE.tar.gz'), sha256='a'*64, bytes=10))
        manifest['files']['OPERATION_SOURCE.py'] = source_ref; calls=[]
        def fake(source, mode, data, out, err):
            calls.append(mode); err.write_bytes(b'')
            if mode == '--remote-worker': out.write_bytes(a.MARKER + a.canonical(manifest)); return 0
            out.write_bytes(b'partial download'); return 255
        with patch.object(a, 'local_payload', side_effect=lambda args: copy.deepcopy(self.payload)), \
                patch.object(a, 'ssh_call', side_effect=fake), redirect_stdout(io.StringIO()):
            self.assertEqual(a.main(self.argv() + ['--execute']), 2)
            saved = self.output / 'REMOTE_MANIFEST.json'
            self.assertEqual(a.loads(saved.read_bytes()), manifest)
            self.assertTrue(a.loads((self.output / 'LOCAL_RECEIPT.json').read_bytes())['remote_audit_success'])
            self.output = self.root / 'recovery'
            self.assertEqual(a.main(self.argv() + ['--execute', '--recover-manifest', str(saved), '--recover-sha256', a.sha(saved)]), 2)
        self.assertEqual(calls, ['--remote-worker', '--remote-download', '--remote-download'])


class OriginalReceiptChain(unittest.TestCase):
    def setUp(self):
        fixture_module = runpy.run_path(str(OPERATIONS / 'tests/test_submit_continuation.py'))
        self.fixture = fixture_module['LocalDriver'](); self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.directory = self.fixture.directory; self.submitter = a.load_submitter()
        preparation = self.submitter.payload_from_receipt(self.fixture.input, a.sha(self.fixture.input))
        self.s = a.selection(11031, 'T0', 3); self.job = 9999999
        self.path = self.directory / 'FAKE_SUBMIT_RECEIPT.json'
        submitted = dict(status='CONTINUATION_SUBMITTED_RELEASE_CONFIRMED', job_id=self.job, job_name=a.job_name(self.s),
            selection=self.s, release=preparation['preparation']['release'], other_jobs_modified=False)
        response = dict(result=submitted, receipt=dict(path=str(a.schedule_for(self.s) / 'RELEASE_RESULT.json'), sha256='a'*64))
        source = dict(path=str(OPERATIONS / 'satdose_submit_continuation.py'), sha256=a.SUBMITTER_SHA)
        out = self.directory / 'submit.stdout'; out.write_bytes(b'SATDOSE_SUBMIT_JSON=' + a.canonical(response))
        err = self.directory / 'submit.stderr'; err.write_bytes(b'')
        self.receipt = dict(status='SUBMISSION_RESPONSE_RETURNED', returncode=0, response=response, source=source,
            stdout=out.read_text(), stderr='', raw_logs=dict(stdout=a.reference(out), stderr=a.reference(err)))
        self.intent = dict(status='SSH_SUBMISSION_ATTEMPT_UNKNOWN_NO_RETRY', source=source,
                           payload=dict(preparation, source_sha256=a.SUBMITTER_SHA))
        self.save()

    def save(self):
        self.path.write_bytes(a.canonical(self.receipt))
        self.path.with_name(self.path.name + '.intent.json').write_bytes(a.canonical(self.intent))

    def args(self): return SimpleNamespace(submission_receipt=self.path, submission_sha256=a.sha(self.path),
                                          job_id=self.job, start_seed=11031, arm='T0', first_step=3)

    def test_original_local_preparation_validator_and_submit_chain(self):
        self.assertEqual(a.local_payload(self.args())['selection'], self.s)

    def test_foreign_job_or_changed_intent_rejected(self):
        args = self.args(); args.job_id += 1
        with self.assertRaises(ValueError): a.local_payload(args)
        self.intent['payload']['source_sha256'] = 'b'*64; self.save()
        with self.assertRaises(ValueError): a.local_payload(self.args())

    def test_changed_raw_or_submit_source_rejected(self):
        self.receipt['source']['sha256'] = 'b'*64; self.save()
        with self.assertRaises(ValueError): a.local_payload(self.args())


class ChunkEvidence(unittest.TestCase):
    def setUp(self):
        self.s = a.selection(11031, 'T0', 3); self.key = '11031/T0'
        self.release = dict(trusted_commits={f'{seed}/{arm}': [fake_ref(f'{seed}-{arm}-{n}') for n in (1,2)]
            for seed in (11031,11032) for arm in ('T0','T1')}, original_train_release=fake_ref('ORIGINAL'),
            dev_endpoints={'OFFLINE_DEV': {}}, risk_clarification=fake_ref('CLARIFICATION'))
        self.result = dict(schema='sr-satdose-continuation-result-v1', status='CONTINUATION_CHUNK_COMPLETE',
            permission='NO_AUTOMATIC_SUCCESSOR_OR_MAIN', **self.s, process_exit_observed=False, main_authorized=False,
            train_responses=768, formal_optimizer_steps=6, scheduler_steps=6,
            original_train_release=self.release['original_train_release'], initial_commit=self.release['trusted_commits'][self.key][-1],
            commits=[fake_ref(f'COMMIT{n}') for n in range(3,9)],
            risk_assessments=[fake_ref(f'RISK{n}') for n in range(3,9)], post_risk=fake_ref('RISK8'))

    def test_completed_cursor_only_extends_selected_path(self):
        original = copy.deepcopy(self.release)
        trusted = a.validate_result(self.result, self.release, self.s)
        self.assertEqual([len(trusted[k]) for k in sorted(trusted)], [8,2,2,2])
        self.assertEqual(self.release, original)

    def test_wrong_count_or_manufactured_process_exit_rejected(self):
        for field,value in [('train_responses',767),('formal_optimizer_steps',True),('process_exit_observed',True),
                            ('main_authorized',True),('initial_commit',fake_ref('FOREIGN'))]:
            result=copy.deepcopy(self.result);result[field]=value
            with self.assertRaises(ValueError):a.validate_result(result,self.release,self.s)

    def test_historical_reports_exact_prefix_but_no_truncated_replay(self):
        sources=dict(risk_verifier=dict(path='/OFFLINE/risk.py',sha256='b'*64))
        values={}; prefixes=copy.deepcopy(self.release['trusted_commits'])
        for index,step in enumerate(range(3,9)):
            prefixes[self.key].append(self.result['commits'][index])
            values[f'RISK{step}']=dict(schema='sr-satdose-scientific-risk-assessment-v1',
                permission='RISK_ASSESSMENT_ONLY_NO_TRAINING_OR_MAIN_AUTHORITY',status='PENDING_REQUIRED_DEV',
                global_hold=False,hold_reasons=[],trusted_commits=copy.deepcopy(prefixes),
                dev_endpoints=self.release['dev_endpoints'],verifier=sources['risk_verifier'],clarification=self.release['risk_clarification'])
        c=SimpleNamespace(read=lambda ref: values[Path(ref['path']).name],
            bound=lambda ref: a.slot_for(self.s)/f'RISK_AFTER_STEP_{int(Path(ref["path"]).name[4:]):03d}.json')
        evidence=a.historical_risks(c,self.result,self.release,dict(execution_sources=sources),self.s)
        self.assertEqual(len(evidence),6)
        self.assertTrue(all('NOT_FULL_REPLAY' in x['verification'] for x in evidence))
        values['RISK4']['trusted_commits']['11032/T1'].append(fake_ref('INVENTED_OTHER_PATH'))
        with self.assertRaises(ValueError):a.historical_risks(c,self.result,self.release,dict(execution_sources=sources),self.s)

    def test_actual_capture_inventory_rejects_extra_complete_draw(self):
        with tempfile.TemporaryDirectory(prefix='satdose-captures-OFFLINE-') as temp:
            root=Path(temp).resolve()
            with patch.object(a,'ROOT',root):
                directory=a.slot_for(self.s)/'audit/steps/step-003';directory.mkdir(parents=True)
                refs=[]; values={}; bound={}
                for n in range(128):
                    stem=a.digest(str(n).encode());capture=directory/(stem+'.CAPTURE.json');capture.write_bytes(b'{}')
                    (directory/(stem+'.BEGIN.json')).write_bytes(b'{}')
                    complete=directory/(stem+'.COMPLETE.json');complete.write_bytes(b'{}')
                    ref=fake_ref('capture'+str(n));refs.append(ref);values[ref['path']]=dict(native_capture_path=str(capture),native_capture_sha256=a.sha(capture));bound[ref['path']]=complete
                commit=fake_ref('COMMIT');values[commit['path']]=dict(transaction=dict(evidence=fake_ref('EVIDENCE')))
                bound[commit['path']]=root/'step-003/COMMIT.json'
                values['execution_releases/ROLLOUT']=dict(complete_files=refs)
                c=SimpleNamespace(read=lambda ref: values[ref['path']],bound=lambda ref:bound[ref['path']],
                    tx=SimpleNamespace(bound=lambda parent,ref:root/'EVIDENCE',read_json=lambda path:dict(rollout_receipt=fake_ref('ROLLOUT'))))
                result=dict(commits=[commit],train_responses=128)
                self.assertEqual(a.capture_population(c,result,self.s),dict(BEGIN=128,CAPTURE=128,COMPLETE=128))
                (directory/('extra.COMPLETE.json')).write_bytes(b'{}')
                with self.assertRaises(ValueError):a.capture_population(c,result,self.s)

    def test_queued_successor_without_worker_slot_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='satdose-queued-OFFLINE-') as temp:
            root=Path(temp).resolve()/'prepared';root.mkdir()
            release=dict(prior_chunks={},dev_endpoints={f'{s}/{arm}/2':{} for s in (11031,11032) for arm in ('T0','T1')})
            with patch.object(a,'ROOT',root):
                schedule=root.parent/'scheduling'
                for seed in (11031,11032):
                    for arm in ('T0','T1'):
                        directory=schedule/'trained_dev_71018e5'/f'start-{seed}'/arm/'step-002';directory.mkdir(parents=True)
                        a.write_once(directory/'SUBMISSION.json',{});a.write_once(directory/'RELEASE_RESULT.json',dict(status='TRAINED_DEV_SUBMITTED_RELEASE_CONFIRMED'))
                directory=a.schedule_for(self.s);directory.mkdir(parents=True)
                a.write_once(directory/'SUBMISSION.json',{});a.write_once(directory/'RELEASE_RESULT.json',dict(status='CONTINUATION_SUBMITTED_RELEASE_CONFIRMED'))
                a.submission_population(release,self.s)
                (directory.parent/'chunk-009-016').mkdir()
                with self.assertRaisesRegex(ValueError,'successor'):a.submission_population(release,self.s)

    def test_actual_claim_resources_and_process_identity(self):
        payload=dict(selection=self.s,job_id=9999999,submitted=dict(release=fake_ref('RELEASE')))
        process=dict(host='gpu02',boot_id='aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',pid=25,start_ticks=100)
        result=dict(initial_commit=fake_ref('INITIAL'),process=process)
        fields=dict(JobId='9999999',JobName=a.job_name(self.s),JobState='RUNNING',Account='rose',QOS=a.QOS,
            Partition='cluster02',NumNodes='1',NumCPUs='8',NumTasks='1',Requeue='0',TimeLimit='1-00:00:00',UserId='varun024(123)',
            Command=str(a.CODE/'scheduling/run_continuation.sh'),WorkDir=str(a.CODE),NodeList='gpu02',
            StdOut=str(a.schedule_for(self.s)/'slurm-9999999.out'),StdErr=str(a.schedule_for(self.s)/'slurm-9999999.err'),
            AllocTRES='cpu=8,mem=66G,node=1,gres/gpu=2,gres/gpu:pro6000=2')
        claim=dict(schema='sr-satdose-continuation-claim-v1',status='ATTEMPTED_UNKNOWN_NO_RETRY',selection=self.s,
            release=fake_ref('RELEASE'),initial_commit=result['initial_commit'],process=process,
            allocation=dict(job_id='9999999',step_id='0',task_count=1,qos=a.QOS,observation=fields))
        self.assertEqual(a.check_claim(claim,result,payload),'gpu02')
        for field,value in [('QOS','normal'),('NumCPUs','16'),('NodeList','gpu03'),('StdOut','/another/job.log')]:
            bad=copy.deepcopy(claim);bad['allocation']['observation'][field]=value
            with self.assertRaises(ValueError):a.check_claim(bad,result,payload)


if __name__ == '__main__': unittest.main()
