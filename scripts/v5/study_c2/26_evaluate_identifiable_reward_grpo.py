from _bootstrap import bootstrap_repo

bootstrap_repo()

from compensability_v5.study_c2.cli import run_registered

raise SystemExit(run_registered(26))
