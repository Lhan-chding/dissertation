import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class WorkerLauncherTests(unittest.TestCase):
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
