from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from adapter.registration import COUNTS, Registration, model_payload, register, strict_json

PACKAGE = Path(__file__).resolve().parents[1]


class RegistrationTests(unittest.TestCase):
    def test_frozen_slots_budget_and_model_boundary_without_evaluation_read(self):
        original_open = Path.open
        def guarded_open(path, *args, **kwargs):
            if path.name == 'evaluation_manifest.jsonl':
                raise AssertionError('formal evaluation manifest opened')
            return original_open(path, *args, **kwargs)
        with patch.object(Path, 'open', guarded_open):
            reg = Registration(PACKAGE)
            counts = Counter()
            request_ids = set()
            train_groups = Counter()
            first_smoke = None
            clarification = []
            for row in reg.requests():
                counts[row['scope']] += 1
                self.assertNotIn(row['request_id'], request_ids)
                request_ids.add(row['request_id'])
                self.assertEqual(row['phase'], row['scope'])
                self.assertEqual(hashlib.sha256(row['prompt'].encode()).hexdigest(), row['prompt_sha256'])
                self.assertEqual(hashlib.sha256(row['assistant_prefill'].encode()).hexdigest(), row['prefill_sha256'])
                self.assertTrue(row['image_file'].startswith('frozen/images/'))
                self.assertEqual(set(model_payload(row)),
                                 {'prompt', 'assistant_prefill', 'image_file', 'image_sha256'})
                self.assertFalse({'world', 'gold', 'audit', 'family', 'channel'} & row.keys())
                if row['scope'] == 'DEV_SMOKE' and first_smoke is None:
                    first_smoke = row
                if row['scope'] == 'CLARIFICATION':
                    clarification.append(row)
                if row['scope'] == 'TRAIN':
                    train_groups[(row['parent'], row['train_seed'], row['arm'], row['step'])] += 1
            self.assertEqual(dict(counts), COUNTS)
            self.assertEqual(len(request_ids), 37440)
            self.assertEqual(len(train_groups), 288)
            self.assertEqual(set(train_groups.values()), {128})
            smoke = json.loads((PACKAGE / 'frozen/receipt_plans/smoke_slots.json').read_text())[0]
            self.assertEqual(first_smoke['request_id'], smoke['request_ids'][0])
            self.assertEqual(first_smoke['seed'], smoke['seeds'][0])
            expected = [json.loads(line) for line in (PACKAGE / 'frozen/clarification_manifest.jsonl').read_text().splitlines()]
            self.assertEqual([{k: row[k] for k in expected[0]} for row in clarification], expected)

    def test_register_rerun_preserves_bytes_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'registration'
            summary = register(PACKAGE, out)
            stat_before = {p.name: p.stat().st_mtime_ns for p in out.iterdir()}
            self.assertEqual(register(PACKAGE, out), summary)
            self.assertEqual(stat_before, {p.name: p.stat().st_mtime_ns for p in out.iterdir()})
            (out / 'INDEX.json').write_text('{}\n')
            with self.assertRaisesRegex(ValueError, 'existing registration bytes differ'):
                register(PACKAGE, out)
            self.assertEqual((out / 'INDEX.json').read_text(), '{}\n')

    def test_reject_destination_inside_frozen_before_writing(self):
        with self.assertRaisesRegex(ValueError, 'destination cannot modify frozen'):
            register(PACKAGE, PACKAGE / 'frozen' / 'new_registration')
        self.assertFalse((PACKAGE / 'frozen' / 'new_registration').exists())

    def test_reject_mismatched_case_identity_and_unsafe_source(self):
        reg = Registration(PACKAGE)
        group = json.loads((PACKAGE / 'frozen/receipt_plans/smoke_slots.json').read_text())[0]
        with self.assertRaisesRegex(ValueError, 'case identity mismatch'):
            list(reg.group_requests(dict(group, root_id='wrong-root'), 'DEV_SMOKE'))
        with self.assertRaisesRegex(ValueError, 'unsafe source path'):
            reg.verify('../RELEASE.json', '0' * 64)
        with self.assertRaisesRegex(ValueError, 'source hash mismatch'):
            reg.verify('frozen/FREEZE.json', '0' * 64)

    def test_reject_duplicate_and_nonfinite_json(self):
        for raw in ('{"x":1,"x":2}', '{"x":NaN}'):
            with self.assertRaises(ValueError):
                strict_json(raw)


if __name__ == '__main__':
    unittest.main()
