"""Exact retirement and immutable history for the never-started rose recovery."""

import hashlib
import json
from pathlib import Path
from types import ModuleType

OLD_WORK = "operations/delivery_recovery/20261021_delivery_oom_cpu22"
RETIREMENT = "operations/delivery_recovery_handoff/216142_teacher"
COMMITS = (
    "249539afb0e3d8b25e750a0076207d7338deb06d",
    "ab7c9d45065d1af58f49d4d62385a5bd0eba14dc",
    "b77d7995716cbeafccf5dabbc406432456a12def",
)
DIRECTORIES = {
    OLD_WORK + "/commands",
    "operations/delivery_recovery_resume/216142_20261021",
    "operations/delivery_recovery_continue/216142_20261021",
}


def require(ok, message):
    if not ok:
        raise PermissionError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bounded(root, name):
    p = Path(name)
    require(not p.is_absolute() and ".." not in p.parts, "Unsafe history path")
    target = (root / p).resolve(strict=True)
    require(target.is_relative_to(root), "History path escapes")
    return target


def check_terminal(raw):
    rows = [line.split("|") for line in raw.splitlines() if line.strip()]
    rows = [r for r in rows if r[0] == "216142"]
    require(len(rows) == 1 and len(rows[0]) == 6, "Retired accounting ambiguous")
    _, state, start, elapsed, allocated, exitcode = rows[0]
    require(state == "CANCELLED" or state.startswith("CANCELLED by "), "Old recovery not cancelled")
    require(
        start in ("Unknown", "None") and elapsed == "0" and not allocated and exitcode == "0:0",
        "Old recovery ran or acquired resources",
    )
    return rows[0]


def verify_history(root, auth, *, require_retired=False):
    root = Path(root)
    require(auth.get("retirement_directory") == RETIREMENT, "Retirement namespace differs")
    require(
        set(auth.get("retired_directories", [])) == DIRECTORIES,
        "Retired directory coverage differs",
    )
    pins = auth.get("retired_files", {})
    require(isinstance(pins, dict) and pins, "Retired history missing")
    for commit, names in zip(
        COMMITS,
        (
            (
                "RECOVERY_AUTHORIZATION.json",
                "recover_delivery.py",
                "submit_delivery_recovery.py",
                "verify_recovered_migration.py",
            ),
            ("HELD_RESUME_AUTHORIZATION.json", "resume_held_delivery.py"),
            ("HELD_CAPACITY_AUTHORIZATION.json", "continue_held_delivery.py"),
        ),
        strict=True,
    ):
        require(
            all(f"operations/{commit}/{name}" in pins for name in names),
            "Retired source/authorization missing",
        )
    old_auth = read(root / "operations" / COMMITS[0] / "RECOVERY_AUTHORIZATION.json")
    require(old_auth["script"] in pins and old_auth["qos"] == "rose", "Old script identity missing")
    paths = [bounded(root, name) for name in pins]
    require(
        all(sha(path) == pins[str(path.relative_to(root))] for path in paths),
        "Retired historical bytes changed",
    )
    for name in DIRECTORIES:
        actual = {
            str(p.relative_to(root))
            for p in bounded(root, name).rglob("*")
            if p.is_file() and not p.name.endswith(".lock")
        }
        expected = {n for n in pins if n.startswith(name + "/")}
        require(actual == expected, "Retired directory inventory changed: " + name)
    for name in ("RELEASE_INTENT.json", "RELEASE.json", "STARTED.json", "PROCESS.json", "END.json"):
        require(not (root / OLD_WORK / name).exists(), "Old recovery advanced: " + name)
    journal = root / RETIREMENT
    proof = journal / "RETIREMENT.json"
    if not proof.exists():
        require(not require_retired, "Retirement incomplete")
        return paths
    record = read(proof)
    require(
        record.get("status") == "CANCELLED_UNSTARTED_RECOVERY" and record.get("job_id") == "216142",
        "Retirement identity differs",
    )
    require(record.get("historical_files") == pins, "Retirement historical binding differs")
    files = record.get("command_files", {})
    actual = {
        str(p.relative_to(journal)): sha(p)
        for p in journal.rglob("*.json")
        if p.name != "RETIREMENT.json"
    }
    require(files == actual, "Retirement command inventory differs")
    require(
        read(journal / "CANCEL_INTENT.json").get("job_id") == "216142",
        "Cancellation intent differs",
    )
    require(
        read(journal / "CANCEL_INTENT.json").get("authorization") == auth,
        "Cancellation authority differs",
    )
    paths.extend(journal / name for name in files)
    commands = []
    expected = [
        ("held_job", ["scontrol", "show", "job", "216142", "-o"]),
        ("stored_script", ["scontrol", "write", "batch_script", "216142", "-"]),
        (
            "held_accounting",
            [
                "sacct",
                "-n",
                "-P",
                "-j",
                "216142",
                "--format=JobIDRaw,State,Start,ElapsedRaw,AllocTRES",
            ],
        ),
        ("cancel", ["scancel", "--state=PENDING", "216142"]),
    ]
    last = 0
    for i, (label, argv) in enumerate(expected, 1):
        intent = read(journal / f"{i:03d}_{label}.intent.json")
        result = read(journal / f"{i:03d}_{label}.result.json")
        require(
            intent.get("argv") == result.get("argv") == argv and result.get("returncode") == 0,
            "Retirement command failed or changed",
        )
        require(last <= intent["at"] <= result["at"], "Retirement chronology differs")
        last = result["at"]
        commands.append(result)
    old_source = root / "operations" / COMMITS[0] / "submit_delivery_recovery.py"
    old_module = ModuleType("retired_rose_submission")
    old_module.__file__ = str(old_source)
    exec(compile(old_source.read_bytes(), str(old_source), "exec"), old_module.__dict__)
    old = old_module.Submission(
        root,
        root / "operations" / COMMITS[0] / "RECOVERY_AUTHORIZATION.json",
        validator=lambda *_: None,
    )
    old.validate()
    old.check_held_fields(old_module.fields(commands[0]["stdout"]), "216142")
    require(
        sha(root / old_auth["script"])
        == hashlib.sha256(commands[1]["stdout"].encode()).hexdigest(),
        "Old stored script differs",
    )
    rows = [r.split("|") for r in commands[2]["stdout"].splitlines() if r.strip()]
    require(
        rows == [["216142", "PENDING", "Unknown", "0", ""]]
        or rows == [["216142", "PENDING", "None", "0", ""]],
        "Old job had started",
    )
    target = record.get("terminal_poll")
    require(
        isinstance(target, str)
        and Path(target).parts[0] == "polls"
        and ".." not in Path(target).parts,
        "Terminal poll path invalid",
    )
    poll_directories = sorted((journal / "polls").iterdir())
    require(
        poll_directories and target == str(poll_directories[-1].relative_to(journal)),
        "Terminal poll not final",
    )
    for poll in poll_directories:
        receipt = read(poll / "RESULT.json")
        require(
            receipt.get("files")
            == {p.name: sha(p) for p in poll.glob("*.json") if p.name != "RESULT.json"},
            "Retirement poll inventory differs",
        )
        results = []
        for i, (label, argv) in enumerate(
            (
                (
                    "terminal",
                    [
                        "sacct",
                        "-n",
                        "-P",
                        "-j",
                        "216142",
                        "--format=JobIDRaw,State,Start,ElapsedRaw,AllocTRES,ExitCode",
                    ],
                ),
                ("queue", ["squeue", "-h", "-j", "216142", "-o", "%i|%T"]),
            ),
            1,
        ):
            intent = read(poll / f"{i:03d}_{label}.intent.json")
            result = read(poll / f"{i:03d}_{label}.result.json")
            require(
                intent.get("argv") == result.get("argv") == argv and result.get("returncode") == 0,
                "Terminal poll command changed or unknown",
            )
            require(last <= intent["at"] <= result["at"], "Terminal poll chronology differs")
            last = result["at"]
            results.append(result)
        if str(poll.relative_to(journal)) == target:
            require(receipt.get("status") == "TERMINAL", "Terminal poll incomplete")
            check_terminal(results[0]["stdout"])
            require(not results[1]["stdout"].strip(), "Retired recovery still queued")
        else:
            require(
                receipt.get("status") == "WAIT_TERMINAL", "Prior retirement poll status differs"
            )
    paths.append(proof)
    return paths
