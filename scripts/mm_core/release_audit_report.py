import argparse
from pathlib import Path

from mm_core.release import release_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--template-root", type=Path, required=True)
    args = parser.parse_args()
    release_report(args.run_root, args.template_root)
