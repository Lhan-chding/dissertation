"""A lease handoff must not turn a live/unknown science task into completion."""

import importlib.util
import threading
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


def test_successor_waits_for_full_or_unknown_capacity():
    path = Path(__file__).resolve().parents[2] / "scripts/operations/mm_dev_f2_controller.py"
    spec = importlib.util.spec_from_file_location("mmdev_controller_ops", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Backend:
        def __init__(self, outcomes):
            self.outcomes = iter(outcomes)

        def submission_capacity(self, permission):
            assert permission == {"qos": "teacher"}
            outcome = next(self.outcomes)
            if isinstance(outcome, Exception):
                raise outcome
            return {"max_submit_jobs_per_user": 5, "queued_job_ids": outcome}

    class Stop:
        def is_set(self):
            return False

        def wait(self, seconds):
            assert seconds == 30

    reports = []
    assert module.wait_for_successor_slot(
        Backend([RuntimeError("query failed"), list("12345"), list("1234")]),
        {"qos": "teacher"},
        Stop(),
        reports.append,
    )
    assert [row["status"] for row in reports] == [
        "SUCCESSOR_CAPACITY_UNKNOWN",
        "WAITING_FOR_SUCCESSOR_SLOT",
        "SUCCESSOR_SLOT_AVAILABLE",
    ]
    stopped = threading.Event()
    stopped.set()
    assert not module.wait_for_successor_slot(Backend([]), {}, stopped, reports.append)
