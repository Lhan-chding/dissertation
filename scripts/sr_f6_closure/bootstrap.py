"""Isolated launcher: exactly the original producer, execution consumer and repair."""

import os
import sys
from pathlib import Path

project = Path("/projects/_ssd/varunssd/louis-ssvc")
code = Path(__file__).resolve().parents[2]
assert sys.flags.isolated and sys.dont_write_bytecode
assert code.parent == project and code.name.startswith("sr_f6_closure_code_")
roots = [
    project / "sr_f6_code_6e5d363/src",
    project / "sr_f6_followup_code_1e4355c/src",
    code / "src",
]
assert all(p.resolve(strict=True) == p and not p.is_symlink() for p in roots)
sys.path[:0] = [str(p) for p in roots]
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
from sr_f6_closure.runner import main  # noqa: E402

raise SystemExit(main())
