import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("server_preflight", ROOT / "scripts/server_preflight.py")
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


class ServerPreflightTests(unittest.TestCase):
    def test_fixed_image_panel(self):
        rows = preflight.image_selection(ROOT / "data")
        self.assertEqual(sum(split == "SMOKE" for split, _ in rows), 8)
        self.assertEqual(sum(split == "MAIN" for split, _ in rows), 16)

    def test_inverse_patch_tensor(self):
        # Distinct spatial/channel values detect patch-order and axis mistakes.
        image = np.arange(8 * 12 * 3, dtype=np.float64).reshape(8, 12, 3) % 256
        patch, merge, temporal = 2, 2, 2
        ip = SimpleNamespace(patch_size=patch, merge_size=merge, temporal_patch_size=temporal,
                             do_normalize=True, do_rescale=True, rescale_factor=1 / 255,
                             image_mean=[0.4, 0.5, 0.6], image_std=[0.2, 0.3, 0.4])
        normalized = (image / 255 - ip.image_mean) / ip.image_std
        frames = np.stack([normalized.transpose(2, 0, 1)] * temporal)
        h, w = 4, 6
        patches = frames.reshape(1, temporal, 3, h // merge, merge, patch, w // merge, merge, patch)
        pixels = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(h * w, 3 * temporal * patch * patch)
        np.testing.assert_array_equal(preflight.reconstruct_pixels(pixels, [1, h, w], ip), image.astype(np.uint8))

    def test_history_no_open_of_sealed_path(self):
        with self.assertRaises(PermissionError):
            preflight._history_rows(Path("/missing/FORMAL/TASKS_PRIVATE.jsonl"))

    def test_historical_renaming_cannot_hide_overlap(self):
        row = json.loads((ROOT / "data/SMOKE_TASKS_PRIVATE.jsonl").read_text().splitlines()[0])
        row["world"]["categories"] = ["new_" + n for n in row["world"]["categories"]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "DEV.jsonl"
            path.write_text(json.dumps(row) + "\n")
            result = preflight.history_check(ROOT / "data", [path])
        self.assertEqual(result["status"], "FAIL_OVERLAP")
        self.assertTrue(any(r["positional"] and not r["exact"] for r in result["matches"]))

    def test_no_history_is_not_a_pass(self):
        self.assertEqual(preflight.history_check(ROOT / "data", [])["status"], "NOT_RUN")

    def test_bad_history_row_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "DEV.jsonl"
            path.write_text('{"qid":"missing_world"}\n')
            with self.assertRaises(ValueError):
                preflight.history_check(ROOT / "data", [path])


if __name__ == "__main__":
    unittest.main()
