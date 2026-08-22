from _bootstrap import bootstrap_repo

bootstrap_repo()

from compensability_v5.study_c2.cli import run_registered  # noqa: E402

raise SystemExit(run_registered(26))
