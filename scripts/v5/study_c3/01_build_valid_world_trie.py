from _bootstrap import bootstrap_repo

bootstrap_repo()

from compensability.study_c3.cli import run_stage  # noqa: E402

raise SystemExit(run_stage(1))
