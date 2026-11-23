"""CPU regressions for final cross-wave coverage and read-only audit transport."""

import copy
import gzip
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[1] / "satdose_audit_parallel_main.py"
spec = importlib.util.spec_from_file_location("parallel_audit_test", SRC)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
SHA = "a" * 64


def scheduler(job=101, seed=11031, arm="START", step=0):
    tres = "billing=16,cpu=8,gres/gpu:pro6000=2,gres/gpu=2,mem=66G,node=1"
    name = f"LH-SATDOSE-MAIN-{seed}-{arm}-s{step:02d}"
    parent = [
        str(job),
        "COMPLETED",
        "0:0",
        name,
        "rose",
        a.QOS,
        "cluster02",
        "varun024",
        "2026-11-21T07:50:15",
        "2026-11-21T18:11:51",
        "gpu-pro6000-5",
        tres,
        tres,
    ]
    rows = [parent]
    for suffix in ("batch", "extern", "0"):
        row = parent.copy()
        row[0] = f"{job}.{suffix}"
        row[3:8] = [suffix, "rose", "", "", ""]
        row[11] = ""
        rows.append(row)
    return rows


def bundle():
    endpoints, accounting, rows = {}, {}, []
    for index, (arm, step) in enumerate(a.WAVES):
        for seed in a.SEEDS:
            job = 101 + index * 2 + int(seed == 11032)
            n = 1952 if step in (0, 24) else 256
            ids = [f"SAT1/MAIN/{seed}/{arm}/s{step:02d}/synthetic/{i}" for i in range(n)]
            rows.extend(
                dict(
                    request_id=rid,
                    job_id=job,
                    complete_sha256=SHA,
                    native_capture_sha256=SHA,
                    score_sha256=SHA,
                )
                for rid in ids
            )
            endpoints[f"{seed}/{arm}/{step}"] = dict(
                job_id=job,
                independent_summary=dict(
                    metrics={"assigned": n},
                    evidence={"request_ids_sha256": a.digest(a.canonical(ids))},
                ),
            )
            accounting[str(job)] = {"rows": scheduler(job, seed, arm, step)}
    return dict(
        status=a.STATUS,
        authority="READ_ONLY_AUDIT_NO_SUCCESSOR",
        terminal={"path": "terminal", "sha256": SHA},
        release={"path": str(a.RELEASE.relative_to(a.ROOT)), "sha256": SHA},
        auditor_sha256=SHA,
        manifest={"sha256": a.MANIFEST_SHA},
        waves=[
            dict(
                jobs={str(seed): endpoints[f"{seed}/{arm}/{step}"]["job_id"] for seed in a.SEEDS},
                pair={
                    "path": str(
                        a.RELEASE.parent.relative_to(a.ROOT) / f"wave-{i:02d}" / "PAIR.json"
                    ),
                    "sha256": SHA,
                },
                released={
                    "path": str(
                        a.RELEASE.parent.relative_to(a.ROOT)
                        / f"wave-{i:02d}"
                        / "RELEASE_RESULT.json"
                    ),
                    "sha256": SHA,
                },
            )
            for i, (arm, step) in enumerate(a.WAVES)
        ],
        scheduler=accounting,
        endpoints=endpoints,
        rows=rows,
    )


class FinalMergeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = bundle()
        cls.fixture_hash = a.digest(a.canonical([r["request_id"] for r in cls.fixture["rows"]]))

    def setUp(self):
        p = patch.object(a, "FROZEN_IDS_SHA", self.fixture_hash)
        p.start()
        self.addCleanup(p.stop)

    def check(self, obj):
        return a.validate_bundle(obj, SHA, SHA)

    def test_frozen_total_and_per_job_hashes(self):
        self.check(self.fixture)

    def test_compensated_omission_or_duplicate_fails(self):
        items = ["a", "b", "c"]
        with self.assertRaisesRegex(ValueError, "missing 1.*duplicate IDs 1"):
            a.exact_membership(items, ["a", "b", "b"], "combined")

    def test_reordered_globally_complete_rows_fail(self):
        fixture = self.fixture.copy()
        fixture["rows"] = self.fixture["rows"][::-1]
        with self.assertRaisesRegex(ValueError, "ordered IDs"):
            self.check(fixture)

    def test_wrong_worker_without_count_change_fails(self):
        fixture = self.fixture.copy()
        fixture["rows"] = self.fixture["rows"].copy()
        fixture["rows"][0] = dict(fixture["rows"][0], job_id=102)
        with self.assertRaisesRegex(ValueError, "wrong lane"):
            self.check(fixture)

    def test_mismatched_per_endpoint_evidence_fails(self):
        fixture = self.fixture.copy()
        fixture["endpoints"] = copy.deepcopy(self.fixture["endpoints"])
        fixture["endpoints"]["11031/T1/16"]["independent_summary"]["evidence"][
            "request_ids_sha256"
        ] = "f" * 64
        with self.assertRaisesRegex(ValueError, "Endpoint count/order"):
            self.check(fixture)

    def test_wrong_job_name_wrong_gpu_failed_step_and_missing_accounting(self):
        for target, column, value in [
            (0, 3, "other"),
            (0, 12, "gres/gpu:pro6000=1"),
            (1, 1, "FAILED"),
            (3, 0, "101.batch"),
        ]:
            with self.subTest(target=target, column=column):
                rows = scheduler()
                rows[target][column] = value
                with self.assertRaises(ValueError):
                    a.scheduler_rows(
                        "\n".join("|".join(r) for r in rows), 101, "LH-SATDOSE-MAIN-11031-START-s00"
                    )

    def test_wave_cannot_substitute_the_other_lane_job(self):
        fixture = self.fixture.copy()
        fixture["waves"] = copy.deepcopy(self.fixture["waves"])
        jobs = fixture["waves"][2]["jobs"]
        jobs["11031"], jobs["11032"] = jobs["11032"], jobs["11031"]
        with self.assertRaisesRegex(ValueError, "Wave assigns different jobs"):
            self.check(fixture)

    def test_wrong_publication_pin_fails(self):
        with self.assertRaisesRegex(ValueError, "Foreign audit"):
            a.validate_bundle(self.fixture, "f" * 64, SHA)

    def test_missing_scorer_and_real_capture_is_not_success(self):
        fixture = self.fixture.copy()
        fixture["rows"] = list(fixture["rows"])
        fixture["rows"][-1] = dict(fixture["rows"][-1], score_sha256="", native_capture_sha256="")
        with self.assertRaisesRegex(ValueError, "missing capture"):
            self.check(fixture)

    def test_verify_existing_download_requires_ssh_content_digest_and_is_repeatable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            raw = a.canonical(self.fixture)
            (root / "BUNDLE.json.gz").write_bytes(gzip.compress(raw))
            (root / "INTENT.json").write_text(
                json.dumps(dict(release_sha256=SHA, auditor_sha256=SHA))
            )
            log = root / "SSH_STDERR.log"
            log.write_bytes(b"SSH disconnected")
            with self.assertRaisesRegex(ValueError, "digest"):
                a.verify_download(root, SHA, SHA)
            self.assertFalse((root / "AUDIT_RECEIPT.json").exists())
            log.write_text("SATDOSE_PARALLEL_AUDIT_SHA256=" + a.digest(raw) + "\n")
            self.assertEqual(a.verify_download(root, SHA, SHA), a.verify_download(root, SHA, SHA))
            (root / "AUDIT_RECEIPT.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Prior audit"):
                a.verify_download(root, SHA, SHA)

    def test_default_cli_does_not_read_cluster_or_start_jobs(self):
        with (
            patch.object(a, "audit", side_effect=AssertionError("no remote audit")),
            patch.object(a.subprocess, "run", side_effect=AssertionError("no SSH")),
        ):
            self.assertEqual(a.main(["--release-sha256", SHA]), 0)


if __name__ == "__main__":
    unittest.main()
