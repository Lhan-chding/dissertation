import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / 'satdose_parallel_main_admission.py'
loader = importlib.util.spec_from_file_location('satdose_parallel_main_admission', SOURCE)
gate = importlib.util.module_from_spec(loader)
loader.loader.exec_module(gate)


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


class ParallelAdmissionTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve() / 'project' / 'prepared_data_v1'
        self.root.mkdir(parents=True)
        self.ep = dict(start_seed=11032, arm='T1', step=16, responses=108)
        self.devdir = self.root / 'execution_releases/trained_dev_audit_job_21_s11032_T1_step016'
        self.dev_result = self.root / 'trained_dev/start-11032/T1/step-016/RESULT.json'
        self.dev_submission = dict(job_id=21, release={'path': 'release-dev', 'sha256': 'c' * 64})
        self.selection = dict(start_seed=11032, arm='T1', first_step=9, last_step=16)
        self.chunkdir = self.root.parent / 'continuation_audits_namespace_v2/start-11032/T1/chunk-009-016'
        self.chunk_result = self.root / 'continuation_control/start-11032/T1/chunk-009-016/RESULT.json'
        self.chunk_submission = dict(job_id=1752141, release={'path': 'release-train', 'sha256': 'd' * 64})

    def dev_evidence(self):
        actual = dict(status='TRAINED_DEV_ENDPOINT_COMPLETE', responses=108, global_scientific_hold=False,
                      receipt={'path': 'evaluation_receipt', 'sha256': 'b' * 64},
                      endpoint_binding={'path': 'evaluation_binding', 'sha256': 'c' * 64})
        write(self.dev_result, actual)
        for name in gate.DEV_MEMBERS:
            write(self.devdir / name, {})
        (self.devdir / 'ORIGINAL_RESULT.json').write_bytes(self.dev_result.read_bytes())
        report = dict(status='TRAINED_DEV_INDEPENDENT_AUDIT_COMPLETE', schema='sr-satdose-independent-trained-dev-audit-v1',
                      job_id=21, endpoint=self.ep, result=gate.file_ref(self.dev_result, self.root),
                      release=self.dev_submission['release'], original_train_release=gate.ORIGINAL,
                      operation_source_sha256=gate.DEV_AUDITOR, responses=108, exact_post_risk_object_reproduced=True,
                      global_scientific_hold=False, next_stage_authorized=False, existing_global_hold=None)
        write(self.devdir / 'AUDIT_REPORT.json', report)
        files = {n: gate.file_ref(self.devdir / n, self.root) for n in gate.DEV_MEMBERS}
        write(self.devdir / 'PUBLICATION.json', dict(status='ANALYSIS_ONLY_ARTIFACTS_PUBLISHED', job_id=21,
              endpoint=self.ep, global_scientific_hold=False, report=files['AUDIT_REPORT.json'], files=files))
        terminal = SimpleNamespace(audit_spec={'dev_endpoints': {'11032/T1/16':
            dict(receipt=actual['receipt'], binding=actual['endpoint_binding'])}})
        return (self.ep, self.devdir, self.dev_result, self.dev_submission), terminal

    def chunk_evidence(self):
        write(self.chunk_result, dict(status='CONTINUATION_CHUNK_COMPLETE'))
        names = gate.TRAIN_MEMBERS | {'NAMESPACE_VERIFIER_SOURCE.py'}
        for n in names:
            write(self.chunkdir / n, {})
        (self.chunkdir / 'OPERATION_SOURCE.py').write_bytes(SOURCE.with_name('satdose_audit_continuation_v2.py').read_bytes())
        (self.chunkdir / 'ORIGINAL_RESULT.json').write_bytes(self.chunk_result.read_bytes())
        result_ref = gate.file_ref(self.chunk_result, self.root)
        report = dict(schema='sr-satdose-independent-continuation-audit-v1', status='CONTINUATION_INDEPENDENT_AUDIT_COMPLETE',
                      selection=self.selection, job_id=1752141, result=result_ref, original_train_release=gate.ORIGINAL,
                      release=self.chunk_submission['release'], operation_source_sha256=gate.NORMAL_AUDITOR,
                      scheduler_exit_observed=True, full_current_post_risk_exactly_reproduced=True,
                      global_scientific_hold=False, next_stage_authorized=False, added_commits=8, train_responses=1024)
        write(self.chunkdir / 'AUDIT_REPORT.json', report)
        files = {n: gate.file_ref(self.chunkdir / n, sized=True) for n in names}
        write(self.chunkdir / 'PUBLICATION.json', dict(status='CONTINUATION_AUDIT_PUBLISHED', selection=self.selection,
              job_id=1752141, report=files['AUDIT_REPORT.json'], files=files))
        files['PUBLICATION.json'] = gate.file_ref(self.chunkdir / 'PUBLICATION.json', sized=True)
        (self.chunkdir / 'BUNDLE.tar.gz').write_bytes(b'published bundle')
        write(self.chunkdir / 'MANIFEST.json', dict(schema='sr-satdose-continuation-audit-manifest-v1',
              success=True, status='CONTINUATION_AUDIT_PUBLICATION_COMPLETE', selection=self.selection, job_id=1752141,
              operation_source_sha256=gate.NORMAL_AUDITOR, next_stage_authorized=False, files=files,
              bundle=gate.file_ref(self.chunkdir / 'BUNDLE.tar.gz', sized=True)))
        entry = (self.selection, gate.NORMAL_AUDITOR, self.chunkdir, self.chunk_result, self.chunk_submission)
        terminal = SimpleNamespace(audit_spec={'continuation_results': {'11032/T1/chunk-009-016': result_ref}})
        return entry, terminal

    def test_wait_is_read_only(self):
        before = set(self.root.rglob('*'))
        x = gate.readiness(self.root)
        self.assertEqual(x['status'], 'WAIT_FOR_TRAINING_AND_AUDITS')
        self.assertEqual(sum(p.endswith('/COMMIT.json') for p in x['missing']), 96)
        self.assertEqual(before, set(self.root.rglob('*')))

    def test_global_hold_overrides_incomplete_population(self):
        write(self.root / 'training_control/HOLD.json', {})
        with self.assertRaisesRegex(ValueError, 'Retained incident'):
            gate.readiness(self.root)

    def test_known_job_cannot_be_replaced_even_while_waiting(self):
        path = self.root.parent / 'scheduling/continuation_71018e5/start-11032/T1/chunk-009-016/RELEASE_RESULT.json'
        write(path, dict(status='CONTINUATION_SUBMITTED_RELEASE_CONFIRMED', selection=self.selection, job_id=42))
        with self.assertRaisesRegex(ValueError, 'Wrong job'):
            gate.readiness(self.root)

    def test_dev_checks_all_copied_members_and_terminal_binding(self):
        entry, terminal = self.dev_evidence()
        key, record = gate.validate_dev(self.root, entry, terminal)
        self.assertEqual((key, record['job_id']), ('11032/T1/16', 21))
        terminal.audit_spec['dev_endpoints'][key]['binding'] = {}
        with self.assertRaisesRegex(ValueError, 'Terminal used another DEV'):
            gate.validate_dev(self.root, entry, terminal)
        (self.devdir / 'SCHEDULER.json').write_text('stale')
        with self.assertRaisesRegex(ValueError, 'Publication members'):
            gate.validate_dev(self.root, entry, None)

    def test_train_checks_selected_job_original_result_and_bundle_bytes(self):
        entry, terminal = self.chunk_evidence()
        key, record = gate.validate_train(self.root, entry, terminal)
        self.assertEqual((key, record['job_id']), ('11032/T1/chunk-009-016', 1752141))
        terminal.audit_spec['continuation_results'][key] = {}
        with self.assertRaisesRegex(ValueError, 'another result'):
            gate.validate_train(self.root, entry, terminal)
        (self.chunkdir / 'BUNDLE.tar.gz').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'bundle'):
            gate.validate_train(self.root, entry, None)

    def test_scheduler_completion_alone_is_insufficient(self):
        terminal = SimpleNamespace(root=self.root, spec={'original_train_release': gate.ORIGINAL},
            report=dict(status='TERMINAL_TRAINING_EVIDENCE_AUDITED_NOT_MAIN_ADMISSION',
                terminal_steps={f'{s}/{a}':24 for s in gate.SEEDS for a in gate.ARMS},
                formal_optimizer_steps=96, train_responses=12288, dev_endpoints=14, dev_responses=1512,
                global_hold=False, pending_required_dev=[]))
        with self.assertRaisesRegex(ValueError, 'Missing operator evidence'):
            gate.verify_admission(terminal, root=self.root)
        terminal.report['formal_optimizer_steps'] = 95
        with patch.object(gate, 'collect', side_effect=AssertionError('must reject before collection')):
            with self.assertRaisesRegex(ValueError, 'complete original terminal audit'):
                gate.verify_admission(terminal, root=self.root)


if __name__ == '__main__':
    unittest.main()
