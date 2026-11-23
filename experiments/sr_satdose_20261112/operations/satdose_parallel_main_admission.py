"""Read-only evidence gate for the two-lane MAIN operator.

The CPU launcher calls verify_admission with the original VerifiedTerminal
returned by the authenticated serial preparer's authenticate_remote. The
terminal publisher independently replays all TRAIN and DEV evidence twice.
This additional gate binds the corresponding operator audit publications and
their original scheduler identities to that exact verified training history.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path('/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/prepared_data_v1')
ORIGINAL = dict(path='execution_releases/FIRST_TWO_RELEASE_71018e5.json',
                sha256='22f3b7b260670628398154652455520cd25169234f66a827d3d117def924438f')
SEEDS = (11031, 11032)
ARMS = ('T0', 'T1')
CHUNKS = ((3, 8), (9, 16), (17, 24))
DEV_STEPS = (2, 8, 16)
NORMAL_AUDITOR = '23e6b906c8d89947dfe86b56450c7bd9a8752e59b1bb6c0a10996520e0d2b593'
RECOVERY_AUDITOR = 'a6a3d3265da51dad54f16f7027afd2b9cf8be2ce2af2bf23ceef863fd63fa4bd'
DEV_AUDITOR = '7c29df29ab6115b322505c59e44bb666d1ef79bf0fe6048c111b25826e0227e7'
TRAIN_MEMBERS = frozenset(('INTENT.json', 'OPERATION_SOURCE.py', 'INCIDENTS.json',
    'SCHEDULER_BEFORE.json', 'SCHEDULER_AFTER.json', 'OWN_JOB_STDOUT.log', 'OWN_JOB_STDERR.log',
    'SUBMISSION_EVIDENCE.json', 'ORIGINAL_CLAIM.json', 'ORIGINAL_RELEASE.json',
    'ORIGINAL_RESULT.json', 'RISK_REPLAY.json', 'AUDIT_REPORT.json', 'ORIGINAL_AUDITOR_SOURCE.py'))
DEV_MEMBERS = frozenset(('CLAIM.json', 'SCHEDULER.json', 'OWN_JOB_STDOUT.log', 'OWN_JOB_STDERR.log',
    'ORIGINAL_RESULT.json', 'ORIGINAL_POST_RISK.json', 'ANALYSIS.json', 'RISK_REPLAY.json', 'AUDIT_REPORT.json'))


def require(condition, message):
    if not condition:
        raise ValueError('HOLD_PARALLEL_MAIN_ADMISSION: ' + message)


def file_bytes(path):
    require(path.is_file() and path.resolve(strict=True) == path, 'Noncanonical file ' + str(path))
    return path.read_bytes()


def pairs(items):
    obj = dict(items)
    require(len(obj) == len(items), 'Duplicate JSON key')
    return obj


def read(path):
    return json.loads(file_bytes(path), object_pairs_hook=pairs,
                      parse_constant=lambda value: require(False, 'Nonfinite ' + value))


def file_ref(path, relative=None, sized=False):
    data = file_bytes(path)
    record = dict(path=str(path.relative_to(relative) if relative is not None else path),
                  sha256=hashlib.sha256(data).hexdigest())
    if sized:
        record['bytes'] = len(data)
    return record


def clear(directory):
    for name in ('HOLD.json', 'ERROR.json'):
        require(not os.path.lexists(directory / name), 'Retained incident ' + str(directory / name))


def publication_files(directory, names, *, relative=None, sized=False):
    publication = read(directory / 'PUBLICATION.json')
    expected = {name: file_ref(directory / name, relative, sized) for name in names}
    require(publication['files'] == expected and publication['report'] == expected['AUDIT_REPORT.json'],
            'Publication members or report binding differ at ' + str(directory))
    return publication, expected


def validate_train(root, entry, terminal):
    selection, source, directory, result_path, scheduling = entry
    first, last = selection['first_step'], selection['last_step']
    key = f"{selection['start_seed']}/{selection['arm']}/chunk-{first:03d}-{last:03d}"
    job = scheduling['job_id']
    last_member = 'KNOWN_FAILURE_BINDING.json' if source == RECOVERY_AUDITOR else 'NAMESPACE_VERIFIER_SOURCE.py'
    members = TRAIN_MEMBERS | {last_member}
    manifest = read(directory / 'MANIFEST.json')
    require(manifest['schema'] == 'sr-satdose-continuation-audit-manifest-v1'
            and manifest['success'] is True and manifest['status'] == 'CONTINUATION_AUDIT_PUBLICATION_COMPLETE'
            and manifest['selection'] == selection and manifest['job_id'] == job
            and manifest['operation_source_sha256'] == source and manifest['next_stage_authorized'] is False,
            'Incomplete/foreign continuation audit ' + key)
    publication, files = publication_files(directory, members, sized=True)
    pubref = file_ref(directory / 'PUBLICATION.json', sized=True)
    require(manifest['files'] == {**files, 'PUBLICATION.json': pubref}
            and manifest['bundle'] == file_ref(directory / 'BUNDLE.tar.gz', sized=True)
            and files['OPERATION_SOURCE.py']['sha256'] == source
            and publication['status'] == 'CONTINUATION_AUDIT_PUBLISHED'
            and publication['selection'] == selection and publication['job_id'] == job,
            'Incomplete continuation bundle or publication ' + key)
    if source == RECOVERY_AUDITOR:
        failure = read(directory / 'KNOWN_FAILURE_BINDING.json')
        old = root.parent / 'continuation_audits_71018e5/start-11031/T0/chunk-003-008'
        require(job == 1742958 and failure['selection'] == selection and failure['job_id'] == job
                and failure['original_failure_preserved'] is True and failure['original_directory'] == str(old)
                and failure['original_manifest_sha256'] == file_ref(old / 'MANIFEST.json')['sha256']
                and failure['original_error_sha256'] == file_ref(old / 'ERROR.json')['sha256'],
                'Historical recovery is missing its failed original')
    report = read(directory / 'AUDIT_REPORT.json')
    result = file_ref(result_path, root)
    require(report['schema'] == 'sr-satdose-independent-continuation-audit-v1'
            and report['status'] == 'CONTINUATION_INDEPENDENT_AUDIT_COMPLETE'
            and report['selection'] == selection and report['job_id'] == job
            and report['result'] == result and result['sha256'] == files['ORIGINAL_RESULT.json']['sha256']
            and report['original_train_release'] == ORIGINAL and report['release'] == scheduling['release']
            and report['operation_source_sha256'] == source and report['scheduler_exit_observed'] is True
            and report['full_current_post_risk_exactly_reproduced'] is True
            and report['global_scientific_hold'] is False and report['next_stage_authorized'] is False
            and report['added_commits'] == last - first + 1 and report['train_responses'] == 128 * (last - first + 1),
            'Continuation audit disagrees with actual result ' + key)
    if terminal is not None:
        require(terminal.audit_spec['continuation_results'][key] == result, 'Terminal used another result ' + key)
    return key, dict(job_id=job, publication=pubref, manifest=file_ref(directory / 'MANIFEST.json'),
                     result=result)


def validate_dev(root, entry, terminal):
    selection, directory, result_path, scheduling = entry
    job = scheduling['job_id']
    key = f"{selection['start_seed']}/{selection['arm']}/{selection['step']}"
    publication, files = publication_files(directory, DEV_MEMBERS, relative=root)
    actual = read(result_path)
    reference = file_ref(result_path, root)
    require(publication['status'] == 'ANALYSIS_ONLY_ARTIFACTS_PUBLISHED'
            and publication['job_id'] == job and publication['endpoint'] == selection
            and publication['global_scientific_hold'] is False
            and reference['sha256'] == files['ORIGINAL_RESULT.json']['sha256'],
            'Incomplete or foreign trained DEV publication ' + key)
    report = read(directory / 'AUDIT_REPORT.json')
    require(report['status'] == 'TRAINED_DEV_INDEPENDENT_AUDIT_COMPLETE'
            and report['schema'] == 'sr-satdose-independent-trained-dev-audit-v1'
            and report['job_id'] == job and report['endpoint'] == selection and report['result'] == reference
            and report['release'] == scheduling['release'] and report['original_train_release'] == ORIGINAL
            and report['operation_source_sha256'] == DEV_AUDITOR and report['responses'] == 108
            and report['exact_post_risk_object_reproduced'] is True and report['global_scientific_hold'] is False
            and report['next_stage_authorized'] is False and report['existing_global_hold'] is None
            and actual['status'] == 'TRAINED_DEV_ENDPOINT_COMPLETE' and actual['responses'] == 108
            and actual['global_scientific_hold'] is False,
            'DEV independent risk or original result differs ' + key)
    if terminal is not None:
        require(terminal.audit_spec['dev_endpoints'][key] ==
                {'receipt': actual['receipt'], 'binding': actual['endpoint_binding']},
                'Terminal used another DEV binding ' + key)
    return key, dict(job_id=job, publication=file_ref(directory / 'PUBLICATION.json', root), result=reference)


def collect(root, terminal=None):
    require(root.is_dir() and root.resolve(strict=True) == root, 'Noncanonical evidence root')
    clear(root)
    for family in ('training_control', 'first_two_launcher', 'first_two_control', 'training_transactions'):
        clear(root / family)
    missing = []
    train_entries, dev_entries = [], []

    def present(path):
        if not os.path.lexists(path):
            missing.append(str(path.relative_to(root.parent)))
            return False
        require(path.is_file() and path.resolve(strict=True) == path, 'Noncanonical evidence ' + str(path))
        return True

    job_ids = set()
    for seed in SEEDS:
        present(root / f'start_dev/start-{seed}/RESULT.json')
        for arm in ARMS:
            path = Path(f'start-{seed}') / arm
            for step in range(1, 25):
                step_dir = root / 'training_transactions' / path / f'step-{step:03d}'
                clear(step_dir)
                present(step_dir / 'COMMIT.json')
            for first, last in CHUNKS:
                suffix = path / f'chunk-{first:03d}-{last:03d}'
                selection = dict(start_seed=seed, arm=arm, first_step=first, last_step=last)
                slot = root / 'continuation_control' / suffix
                clear(slot)
                result = slot / 'RESULT.json'
                present(result)
                recovery = (seed, arm, first) == (11031, 'T0', 3)
                family = ('continuation_audits_namespace_recovery_v1_1742958' if recovery
                          else 'continuation_audits_namespace_v2')
                audit = root.parent / family / suffix
                clear(audit)
                present(audit / 'MANIFEST.json')
                schedule = root.parent / 'scheduling/continuation_71018e5' / suffix
                clear(schedule)
                if present(schedule / 'RELEASE_RESULT.json'):
                    submitted = read(schedule / 'RELEASE_RESULT.json')
                    job = submitted['job_id']
                    require(type(job) is int and job > 0 and job not in job_ids
                            and submitted['status'] == 'CONTINUATION_SUBMITTED_RELEASE_CONFIRMED'
                            and submitted['selection'] == selection, 'Unexpected continuation submission')
                    require((seed, arm, first) != (11032, 'T1', 9) or job == 1752141, 'Wrong job for 11032/T1/9')
                    job_ids.add(job)
                    train_entries.append((selection, RECOVERY_AUDITOR if recovery else NORMAL_AUDITOR,
                                          audit, result, submitted))
            for step in DEV_STEPS:
                suffix = path / f'step-{step:03d}'
                ep = dict(start_seed=seed, arm=arm, step=step, responses=108)
                result = root / 'trained_dev' / suffix / 'RESULT.json'
                clear(result.parent)
                present(result)
                schedule = root.parent / 'scheduling/trained_dev_71018e5' / suffix
                clear(schedule)
                if present(schedule / 'RELEASE_RESULT.json'):
                    submitted = read(schedule / 'RELEASE_RESULT.json')
                    job = submitted['job_id']
                    require(type(job) is int and job > 0 and job not in job_ids
                            and submitted['status'] == 'TRAINED_DEV_SUBMITTED_RELEASE_CONFIRMED'
                            and submitted['endpoint'] == ep, 'Unexpected trained DEV submission')
                    job_ids.add(job)
                    audit = root / 'execution_releases' / f'trained_dev_audit_job_{job}_s{seed}_{arm}_step{step:03d}'
                    clear(audit)
                    present(audit / 'PUBLICATION.json')
                    dev_entries.append((ep, audit, result, submitted))
    if missing:
        return missing, {}, {}
    trains = dict(validate_train(root, entry, terminal) for entry in train_entries)
    devs = dict(validate_dev(root, entry, terminal) for entry in dev_entries)
    require(len(trains) == len(devs) == 12, 'Incorrect independent audit population')
    return [], trains, devs


def readiness(root=ROOT):
    """Non-mutating prerequisite observation before any write-once terminal publication."""
    missing, _, _ = collect(root)
    return dict(schema='sr-satdose-parallel-main-readiness-v1',
                status='WAIT_FOR_TRAINING_AND_AUDITS' if missing else 'READY_FOR_TERMINAL_PUBLICATION',
                missing=missing)


def verify_admission(terminal, *, root=ROOT):
    """Use only with the authenticated original prepare_main.authenticate_remote result."""
    require(terminal.root == root and terminal.spec['original_train_release'] == ORIGINAL
            and terminal.report['status'] == 'TERMINAL_TRAINING_EVIDENCE_AUDITED_NOT_MAIN_ADMISSION'
            and terminal.report['terminal_steps'] == {f'{s}/{a}': 24 for s in SEEDS for a in ARMS}
            and terminal.report['formal_optimizer_steps'] == 96 and terminal.report['train_responses'] == 12288
            and terminal.report['dev_endpoints'] == 14 and terminal.report['dev_responses'] == 1512
            and terminal.report['global_hold'] is False and terminal.report['pending_required_dev'] == [],
            'Not the complete original terminal audit')
    missing, trains, devs = collect(root, terminal)
    require(not missing, 'Missing operator evidence: ' + ', '.join(missing))
    terminal.stable()
    return dict(schema='sr-satdose-parallel-main-admission-v1', status='TERMINAL_AND_OPERATOR_AUDITS_VERIFIED',
                original_train_release=ORIGINAL.copy(), terminal_admission=terminal.receipt,
                terminal_audit=terminal.spec['report'], continuation_audits=trains, trained_dev_audits=devs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--readiness', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(readiness(), sort_keys=True))
