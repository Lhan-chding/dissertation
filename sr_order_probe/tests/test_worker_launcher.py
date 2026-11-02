import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class WorkerLauncherTests(unittest.TestCase):
    def main_calls(self, fail=False):
        launcher = Path(__file__).resolve().parents[1] / 'scripts/run_main_four_workers.sh'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / 'srun'
            fake.write_text(
                '#!/bin/sh\n'
                '[ "$CUBLAS_WORKSPACE_CONFIG" = ":4096:8" ] || exit 31\n'
                'printf "%s\\n" "$*" >> "$LAUNCH_TEST_CAPTURE"\n'
                'case "$*" in *"--parent b0 --shard-index 2 "*) '
                '[ "$LAUNCH_TEST_FAIL" = 0 ] || exit 32;; esac\n'
            )
            fake.chmod(0o755)
            env = dict(os.environ, PATH=directory + ':' + os.environ['PATH'],
                       SLURM_MEM_PER_NODE='135168', LAUNCH_TEST_FAIL=str(int(fail)),
                       LAUNCH_TEST_CAPTURE=str(root / 'calls'))
            result = subprocess.run(['bash', str(launcher), directory, directory],
                                    env=env, capture_output=True, text=True)
            return result, (root / 'calls').read_text().splitlines()

    def test_main_partition_and_finite_postprocessing(self):
        result, calls = self.main_calls()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 16)
        self.assertTrue(all('--mem=33792M' in c and '--gpus-per-task=1' in c for c in calls))
        for index, parent in enumerate(('b0', 'b1', 'b2')):
            group = calls[index * 4:(index + 1) * 4]
            for shard in range(4):
                self.assertEqual(sum(f'--parent {parent} --shard-index {shard} --shard-count 4' in c for c in group), 1)
        self.assertIn('order_probe_server collect', calls[12])
        self.assertIn('score_answers.py', calls[13])
        self.assertIn('scripts/analyze.py', calls[14])
        self.assertIn('supplementary_report.py', calls[15])

    def test_main_failure_stops_successor_parents_and_analysis(self):
        result, calls = self.main_calls(fail=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(calls), 4)
        self.assertTrue(all('--parent b0 ' in c for c in calls))

    def test_required_environment_and_disjoint_worker_resources(self):
        launcher = Path(__file__).resolve().parents[1] / 'scripts/run_worker_pair.sh'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / 'srun'
            fake.write_text(
                '#!/bin/sh\n'
                '[ "$CUBLAS_WORKSPACE_CONFIG" = ":4096:8" ] || exit 31\n'
                'printf "%s\\n" "$*" >> "$LAUNCH_TEST_CAPTURE"\n'
            )
            fake.chmod(0o755)
            env = dict(os.environ, PATH=directory + ':' + os.environ['PATH'],
                       SLURM_MEM_PER_NODE='67584',
                       LAUNCH_TEST_CAPTURE=str(root / 'calls'))
            result = subprocess.run(['bash', str(launcher), directory, directory, 'SMOKE', 'b0'],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = (root / 'calls').read_text().splitlines()
            self.assertEqual(len(calls), 2)
            self.assertTrue(all('--mem=33792M' in c and '--gpus-per-task=1' in c for c in calls))
            self.assertTrue(any('--shard-index 0' in c for c in calls))
            self.assertTrue(any('--shard-index 1' in c for c in calls))


if __name__ == '__main__':
    unittest.main()
