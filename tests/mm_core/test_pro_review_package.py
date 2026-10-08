"""Local-only adversarial and reproducibility tests for the review ZIP."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
import zipfile
import zlib
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/mm_core/build_pro_review_package.py"
SPEC = importlib.util.spec_from_file_location("build_pro_review_package", SCRIPT)
package = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(package)


class ProReviewPackageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.staging = self.root / "staging"
        self.staging.mkdir()
        self.output = self.root / "review.zip"
        self.exclusions = [{"category": "model_weights", "reason": "Not part of this review"}]
        (self.staging / "PACKAGE_EXCLUSIONS.json").write_text(
            json.dumps({"excluded": self.exclusions}), encoding="utf-8"
        )
        (self.staging / "README.md").write_text(
            "# Review\nLocal evidence only.\n", encoding="utf-8"
        )
        (self.staging / "data").mkdir()
        (self.staging / "data/回答.jsonl").write_text('{"answer": 42}\n', encoding="utf-8")
        (self.staging / "data/chart.png").write_bytes(b"\x89PNG\r\n\x1a\nchart fixture")

    def build(self):
        return package.build_package(self.staging, self.output)

    def rewritten(self, *, replacements=None, removed=(), extra=()):
        replacements = replacements or {}
        target = self.root / "tampered.zip"
        with zipfile.ZipFile(self.output) as source, zipfile.ZipFile(target, "w") as output:
            for info in source.infolist():
                if info.filename not in removed:
                    output.writestr(info, replacements.get(info.filename, source.read(info)))
            for name, data in extra:
                output.writestr(package.info_for(name), data)
        return target

    def test_package_manifest_checksums_crc_and_immutable_staging(self):
        before = {
            p.relative_to(self.staging): p.read_bytes()
            for p in self.staging.rglob("*")
            if p.is_file()
        }
        receipt = self.build()
        self.assertEqual(receipt["status"], "VERIFIED")
        self.assertTrue(receipt["crc_read_to_eof"])
        self.assertEqual(receipt["sha256"], hashlib.sha256(self.output.read_bytes()).hexdigest())
        self.assertEqual(receipt["excluded"], self.exclusions)
        self.assertEqual(json.loads(Path(str(self.output) + ".json").read_text()), receipt)
        self.assertEqual(
            before,
            {
                p.relative_to(self.staging): p.read_bytes()
                for p in self.staging.rglob("*")
                if p.is_file()
            },
        )
        with zipfile.ZipFile(self.output) as archive:
            manifest = json.loads(archive.read("MANIFEST.json"))
            self.assertEqual(manifest["hash_scope_excludes"], ["MANIFEST.json", "SHA256SUMS.txt"])
            self.assertNotIn("MANIFEST.json", manifest["files"])
            self.assertNotIn("SHA256SUMS.txt", manifest["files"])
            sums = archive.read("SHA256SUMS.txt").decode().splitlines()
            self.assertEqual(
                {row[66:] for row in sums}, set(archive.namelist()) - {"SHA256SUMS.txt"}
            )
            for row in sums:
                self.assertEqual(hashlib.sha256(archive.read(row[66:])).hexdigest(), row[:64])
            for info in archive.infolist():
                self.assertEqual(info.date_time, package.FIXED_TIMESTAMP)
            self.assertIsNone(archive.testzip())

    def test_reproducible_across_output_names_and_source_mtime(self):
        self.build()
        os.utime(self.staging / "README.md", (1700000000, 1700000000))
        other = self.root / "another.zip"
        package.build_package(self.staging, other)
        self.assertEqual(self.output.read_bytes(), other.read_bytes())

    def test_standalone_stdlib_verifier_with_no_repository_imports(self):
        self.build()
        standalone = self.root / "standalone"
        standalone.mkdir()
        with zipfile.ZipFile(self.output) as archive:
            (standalone / "verify_package.py").write_bytes(archive.read("verify_package.py"))
        result = subprocess.run(
            [sys.executable, "-I", str(standalone / "verify_package.py"), str(self.output)],
            cwd=standalone,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(json.loads(result.stdout)["status"], "VERIFIED")

    def test_cli_creates_zip_and_receipt(self):
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--staging",
                str(self.staging),
                "--output",
                str(self.output),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(json.loads(result.stdout)["new_model_calls"], 0)
        self.assertTrue(Path(str(self.output) + ".json").is_file())

    def test_missing_explicit_exclusion_declaration_is_rejected(self):
        (self.staging / "PACKAGE_EXCLUSIONS.json").unlink()
        with self.assertRaisesRegex(ValueError, "EXPLICIT_EXCLUSIONS_REQUIRED"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_empty_exclusion_declaration_is_explicit_and_allowed(self):
        (self.staging / "PACKAGE_EXCLUSIONS.json").write_text('{"excluded": []}')
        self.assertEqual(self.build()["excluded"], [])

    def test_output_inside_staging_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "OUTPUT_INSIDE_STAGING"):
            package.build_package(self.staging, self.staging / "review.zip")

    def test_refuses_overwrite_of_zip_or_receipt(self):
        for target in (self.output, Path(str(self.output) + ".json")):
            with self.subTest(target=target.name):
                target.write_bytes(b"must remain unchanged")
                with self.assertRaisesRegex(ValueError, "OUTPUT_EXISTS"):
                    self.build()
                self.assertEqual(target.read_bytes(), b"must remain unchanged")
                target.unlink()

    def test_staging_symlink_file_directory_and_root_are_rejected(self):
        target = self.root / "outside.txt"
        target.write_text("outside")
        for source in (target, self.root):
            link = self.staging / "linked"
            link.symlink_to(source)
            with self.assertRaisesRegex(ValueError, "SYMLINK_NOT_ALLOWED"):
                self.build()
            link.unlink()
        link = self.root / "linked-staging"
        link.symlink_to(self.staging)
        with self.assertRaisesRegex(ValueError, "SYMLINK_STAGING_NOT_ALLOWED"):
            package.build_package(link, self.output)

    def test_dangling_output_symlink_is_not_overwritten(self):
        self.output.symlink_to(self.root / "absent.zip")
        with self.assertRaisesRegex(ValueError, "OUTPUT_EXISTS"):
            self.build()
        self.assertTrue(self.output.is_symlink())

    def test_reserved_metadata_cannot_be_supplied_by_staging(self):
        for name in ("MANIFEST.json", "sha256sums.TXT", "Verify_Package.py"):
            with self.subTest(name=name):
                path = self.staging / name
                path.write_text("untrusted replacement")
                with self.assertRaisesRegex(ValueError, "RESERVED_METADATA_NAME"):
                    self.build()
                path.unlink()

    def test_weights_credentials_and_opaque_archives_rejected(self):
        for name in (
            "weights.PT",
            "model.safetensors",
            "key.pem",
            "private.key",
            ".env",
            "credentials.json",
            "weights.bin",
            "concealed.tar.gz",
        ):
            with self.subTest(name=name):
                path = self.staging / name
                path.write_bytes(b"do not publish")
                with self.assertRaisesRegex(ValueError, "FORBIDDEN_"):
                    self.build()
                self.assertFalse(self.output.exists())
                path.unlink()

    def test_private_key_content_under_innocent_suffix_is_rejected(self):
        (self.staging / "notes.txt").write_text("-----BEGIN PRIVATE KEY-----\nsecret\n")
        with self.assertRaisesRegex(ValueError, "PRIVATE_KEY_CONTENT"):
            self.build()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".pro-*")))

    def test_unsafe_staging_filename_is_rejected(self):
        path = self.staging / "unsafe\\name.txt"
        path.write_text("unsafe")
        with self.assertRaisesRegex(ValueError, "UNSAFE_PATH"):
            self.build()

    def test_verifier_rejects_path_traversal_absolute_and_control_names(self):
        self.build()
        for name in (
            "../escape.txt",
            "/absolute.txt",
            "C:/drive.txt",
            "a\\b.txt",
            "control\nname.txt",
            "a//b.txt",
            "a/./b.txt",
            "con.txt",
        ):
            with self.subTest(name=name):
                changed = self.rewritten(extra=[(name, b"unsafe")])
                with self.assertRaisesRegex(ValueError, "UNSAFE_PATH"):
                    package.verify_archive(changed)

    def test_verifier_rejects_case_unicode_and_prefix_collisions(self):
        self.build()
        additions = (
            [("Readme.md", b"different case")],
            [("caf\u00e9.txt", b"nfc"), ("cafe\u0301.txt", b"nfd")],
            [("DATA/other.txt", b"different directory case")],
        )
        for rows in additions:
            with (
                self.subTest(rows=rows),
                self.assertRaisesRegex(ValueError, "CASE_OR_UNICODE_COLLISION"),
            ):
                package.verify_archive(self.rewritten(extra=rows))

    def test_verifier_rejects_duplicate_entries(self):
        self.build()
        with self.assertWarns(UserWarning):
            changed = self.rewritten(extra=[("README.md", b"duplicate")])
        with self.assertRaisesRegex(ValueError, "DUPLICATE_FILE"):
            package.verify_archive(changed)

    def test_verifier_rejects_file_directory_collision(self):
        self.build()
        changed = self.rewritten(extra=[("README.md/subfile", b"bad")])
        with self.assertRaisesRegex(ValueError, "FILE_DIRECTORY_COLLISION"):
            package.verify_archive(changed)

    def test_verifier_rejects_zip_symlink(self):
        self.build()
        changed = self.rewritten()
        with zipfile.ZipFile(changed, "a") as archive:
            info = package.info_for("link")
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, "../outside")
        with self.assertRaisesRegex(ValueError, "NONREGULAR_ZIP_ENTRY"):
            package.verify_archive(changed)

    def test_verifier_rejects_tampered_extra_and_missing_payload(self):
        self.build()
        cases = (
            ({"replacements": {"README.md": b"changed"}}, "MANIFEST_HASH_OR_SIZE_MISMATCH"),
            ({"extra": [("extra.txt", b"extra")]}, "EXTRA_OR_MISSING_FILES"),
            ({"removed": ["README.md"]}, "EXTRA_OR_MISSING_FILES"),
            ({"removed": ["MANIFEST.json"]}, "MISSING_REQUIRED_METADATA"),
        )
        for arguments, error in cases:
            with self.subTest(arguments=arguments), self.assertRaisesRegex(ValueError, error):
                package.verify_archive(self.rewritten(**arguments))

    def test_verifier_checks_manifest_itself_and_embedded_verifier(self):
        self.build()
        with zipfile.ZipFile(self.output) as archive:
            manifest = json.loads(archive.read("MANIFEST.json"))
        manifest["annotation"] = "tamper"
        with self.assertRaisesRegex(ValueError, "CHECKSUM_MISMATCH:MANIFEST.json"):
            package.verify_archive(
                self.rewritten(replacements={"MANIFEST.json": package.encoded(manifest)})
            )
        with self.assertRaisesRegex(ValueError, "MANIFEST_HASH_OR_SIZE_MISMATCH:verify_package.py"):
            package.verify_archive(
                self.rewritten(replacements={"verify_package.py": b"# tamper\n"})
            )

    def test_verifier_rejects_crc_damage_during_read_to_eof(self):
        self.build()
        with zipfile.ZipFile(self.output) as archive:
            info = archive.getinfo("README.md")
        damaged = bytearray(self.output.read_bytes())
        offset = info.header_offset
        name_length = int.from_bytes(damaged[offset + 26 : offset + 28], "little")
        extra_length = int.from_bytes(damaged[offset + 28 : offset + 30], "little")
        data_offset = offset + 30 + name_length + extra_length
        damaged[data_offset + info.compress_size - 2] ^= 0xFF
        self.output.write_bytes(damaged)
        with self.assertRaises((zipfile.BadZipFile, zlib.error)):
            package.verify_archive(self.output)

    def test_failed_receipt_publication_removes_only_own_zip(self):
        real_link = os.link
        receipt = Path(str(self.output) + ".json")

        def race(source, target):
            if target == receipt:
                receipt.write_bytes(b"other writer")
                raise FileExistsError("another writer won")
            real_link(source, target)

        with patch.object(package.os, "link", side_effect=race), self.assertRaises(FileExistsError):
            self.build()
        self.assertFalse(self.output.exists())
        self.assertEqual(receipt.read_bytes(), b"other writer")
        self.assertFalse(list(self.root.glob(".pro-*")))

    def test_changed_staging_is_not_published(self):
        original_inventory = package.inventory
        calls = 0

        def changing_inventory(staging):
            nonlocal calls
            calls += 1
            if calls == 2:
                (staging / "README.md").write_text("concurrent edit")
            return original_inventory(staging)

        with (
            patch.object(package, "inventory", side_effect=changing_inventory),
            self.assertRaisesRegex(ValueError, "STAGING_CHANGED_DURING_BUILD"),
        ):
            self.build()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".pro-*")))

    def test_duplicate_json_policy_keys_are_rejected(self):
        (self.staging / "PACKAGE_EXCLUSIONS.json").write_text('{"excluded":[],"excluded":[]}')
        with self.assertRaisesRegex(ValueError, "DUPLICATE_JSON_KEY"):
            self.build()

    def test_unreadable_subdirectory_cannot_be_silently_omitted(self):
        blocked = self.staging / "read_error"
        blocked.mkdir()
        (blocked / "required.json").write_text('{"required": true}')
        real_scandir = os.scandir

        def fail_scandir(path):
            if Path(path).resolve() == blocked.resolve():
                raise PermissionError("injected directory read failure")
            return real_scandir(path)

        with (
            patch.object(package.os, "scandir", side_effect=fail_scandir),
            self.assertRaisesRegex(PermissionError, "injected directory read failure"),
        ):
            self.build()
        self.assertFalse(self.output.exists())
        self.assertFalse(Path(str(self.output) + ".json").exists())
        self.assertFalse(list(self.root.glob(".pro-*")))


if __name__ == "__main__":
    unittest.main()
