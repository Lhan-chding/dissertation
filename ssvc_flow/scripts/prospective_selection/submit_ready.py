"""CLI convenience entry; the registry journals every submission before sbatch."""

import sys

from src.prospective_selection.cli import main

if __name__ == "__main__":
    main(["submit-ready", *sys.argv[1:]])
