"""Start coverage in test-spawned Python processes when coverage requests it.

Python ignores ``.pth`` startup hooks inside this repository's hidden ``.venv``
directory.  Coverage's subprocess patch still propagates its serialized,
project-owned configuration, and this ordinary ``sitecustomize`` hook consumes
that configuration without affecting non-coverage test or production runs.
"""

from __future__ import annotations

import os

if os.getenv("COVERAGE_PROCESS_CONFIG") or os.getenv("COVERAGE_PROCESS_START"):
    try:
        import coverage
    except ImportError:  # pragma: no cover - coverage is optional outside dev/test installs
        pass
    else:
        coverage.process_startup(slug="sitecustomize")
