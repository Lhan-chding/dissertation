"""QOS changes preserve the global project queue and explicit requeue policy."""

import pytest

from src.prospective_selection import cli, jobs


@pytest.mark.parametrize("qos,requeue", [(cli.TEACHER_QOS, False), (cli.PREEMPTIBLE_QOS, True)])
def test_submit_qos_keeps_global_queue(tmp_path, monkeypatch, qos, requeue):
    args = cli.parser().parse_args(
        [
            "submit-ready",
            "--root",
            str(tmp_path),
            "--runtime",
            str(tmp_path / "runtime.json"),
            "--python",
            "/python",
            "--project-root",
            str(tmp_path),
            "--cache-root",
            str(tmp_path),
            "--max-gpu-jobs",
            "7",
            "--available-gpus",
            "7",
            "--qos",
            qos,
        ]
    )
    commands = []

    def check_output(command, **kwargs):
        commands.append(command)
        if command[0] == "squeue":
            # All project jobs count, regardless of their QOS; unrelated jobs do not.
            assert "--qos" not in " ".join(command)
            return (
                "1|ps2-teacher|gres/gpu:pro6000:1\n"
                "2|ps2-preemptible|gres/gpu:pro6000:1\n"
                "3|other|gres/gpu:a40:1\n"
            )
        return "4\n"

    class Registry:
        def __init__(self, *args):
            pass

        def submit_ready(self, query, submit, **limits):
            assert query() == [{"job_id": "1", "gpus": 1}, {"job_id": "2", "gpus": 1}]
            assert limits == {"max_gpu_jobs": 7, "available_gpus": 7}
            return [submit({"task_id": "abcdef"}, tmp_path / "task.json")]

    monkeypatch.setattr(jobs, "TaskRegistry", Registry)
    monkeypatch.setattr(cli.subprocess, "check_output", check_output)
    result = cli.submit_ready(args, {})
    command = commands[-1]
    assert "--qos=" + qos in command
    assert ("--requeue" in command) == requeue
    assert ("--no-requeue" in command) != requeue
    assert "--open-mode=append" in command
    assert result["qos"] == qos and result["requeue"] == requeue
    assert result["submissions"] == ["4"]


def test_unapproved_qos_rejected():
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["submit-ready", "--qos", "arbitrary-account-qos"])
