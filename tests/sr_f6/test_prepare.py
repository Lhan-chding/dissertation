import json

import pytest

from sr_f6.prepare import initialize, source_manifest


def test_explicit_source_recovery_preserves_first_attempt(tmp_path):
    config = tmp_path / "input.json"
    config.write_text('{"fixture":1}\n')
    root = tmp_path / "run"
    initialize(root, config, "a" * 40)
    original = (root / "PREPARATION_STARTED.json").read_bytes()
    with pytest.raises(FileExistsError):
        initialize(root, config, "b" * 40)
    initialize(root, config, "b" * 40, resume_source_failure=True)
    assert (root / "PREPARATION_STARTED.json").read_bytes() == original
    assert (
        json.loads((root / ("PREPARATION_RECOVERY_" + "b" * 12 + ".json")).read_text())[
            "previous_started_record"
        ]["revision"]
        == "a" * 40
    )


@pytest.mark.parametrize("existing", ["DATA", "CENSUS", "SOURCE_BINDING.json", "CPU_PREPARED.json"])
def test_source_recovery_cannot_overwrite_later_phase(tmp_path, existing):
    config = tmp_path / "input.json"
    config.write_text("{}")
    root = tmp_path / "run"
    initialize(root, config, "a" * 40)
    (root / existing).touch()
    with pytest.raises(PermissionError):
        initialize(root, config, "b" * 40, resume_source_failure=True)


def test_frozen_source_inventory_includes_shell_entrypoints(tmp_path):
    script = tmp_path / "scripts/sr_f6/run_pilot_pair.sh"
    script.parent.mkdir(parents=True)
    script.write_text("#!/usr/bin/env bash\nexit 0\n")
    before = source_manifest(tmp_path)
    assert before["files"][0]["path"] == "scripts/sr_f6/run_pilot_pair.sh"
    script.write_text("#!/usr/bin/env bash\nexit 1\n")
    assert source_manifest(tmp_path)["files"][0]["sha256"] != before["files"][0]["sha256"]
