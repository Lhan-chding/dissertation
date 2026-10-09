"""A lease handoff must not turn a live/unknown science task into completion."""

import importlib.util
from pathlib import Path


def test_controller_release_and_technical_boundary():
    path = Path(__file__).resolve().parents[2] / "scripts/operations/mm_dev_f2_controller.py"
    spec = importlib.util.spec_from_file_location("mmdev_controller_ops", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    live = {"ACTIVE", "UNKNOWN", "REGISTERED", "SUBMITTING"}
    assert module.should_finish({"phase": "RELEASED"}, live)
    for status in live:
        assert not module.should_finish(
            {
                "phase": "F2_FROZEN",
                "technical_blockers": ["another task"],
                "tasks": {"a": {"status": status}},
            },
            live,
        )
    assert not module.should_finish(
        {
            "phase": "F2_FROZEN",
            "technical_blockers": [],
            "tasks": {},
        },
        live,
    )
    assert module.should_finish(
        {
            "phase": "F2_FROZEN",
            "technical_blockers": ["technical failure"],
            "tasks": {"a": {"status": "BLOCKED"}},
        },
        live,
    )
