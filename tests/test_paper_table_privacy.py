"""Publication-boundary tests for accepted metric table rendering."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.build_paper_tables import _flatten, _reject_private_metrics


@pytest.mark.parametrize(
    "private_value",
    (
        "/tmp/private-run/result.json",
        "/private/tmp/private-run/result.json",
        "/var/folders/ab/private-run/result.json",
        "sk" + "-exampletokenvalue123456",
        "rk" + "-exampletokenvalue123456",
        "ghp" + "_exampletokenvalue123456",
        "github" + "_pat_exampletokenvalue123456",
    ),
)
def test_paper_table_privacy_rejects_local_paths_and_standard_secret_prefixes(
    private_value: str,
) -> None:
    flattened = _flatten({"published_metric": private_value})

    with pytest.raises(ValueError, match="privacy check rejected"):
        _reject_private_metrics(flattened, source=Path("accepted.json"))
