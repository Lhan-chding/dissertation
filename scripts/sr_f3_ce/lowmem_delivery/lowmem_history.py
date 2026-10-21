"""Exact retirement and immutable history for the never-started 22-core teacher recovery."""

import hashlib
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

OLD_WORK = "operations/delivery_recovery/20261021_delivery_teacher"
RETIREMENT = "operations/delivery_recovery_handoff/216897_lowmem"
COMMITS = ("4401f8586c662ce66583c45a34dae724929ea2ef",)
DIRECTORIES = {OLD_WORK}


def require(ok, message):
    if not ok:
        raise PermissionError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bounded(root, name):
    p = Path(name)
    require(not p.is_absolute() and ".." not in p.parts, "Unsafe history path")
    target = (root / p).resolve(strict=True)
    require(target.is_relative_to(root), "History path escapes")
    return target


def check_terminal(raw):
    rows = [line.split("|") for line in raw.splitlines() if line.strip()]
    rows = [r for r in rows if r[0] == "216897"]
    require(len(rows) == 1 and len(rows[0]) == 6, "Retired accounting ambiguous")
    _, state, start, elapsed, allocated, exitcode = rows[0]
    require(state == "CANCELLED" or state.startswith("CANCELLED by "), "Old recovery not cancelled")
    require(
        start in ("Unknown", "None") and elapsed == "0" and not allocated and exitcode == "0:0",
        "Old recovery ran or acquired resources",
    )
    return rows[0]


def verify_pending_history(root, old_auth):
    """Require every old teacher query to have a known, completed audit outcome."""
    work = root / OLD_WORK
    submitted = read(work / "SUBMITTED.json")
    require(
        submitted.get("job_id") == "216897"
        and submitted.get("authorization_sha256")
        == sha(root / "operations" / COMMITS[0] / "RECOVERY_AUTHORIZATION.json")
        and submitted.get("script_sha256") == old_auth["script_sha256"],
        "Old teacher submission identity differs",
    )
    old_source = root / "operations" / COMMITS[0] / "submit_delivery_recovery.py"
    old_module = ModuleType("bound_teacher_pending_commands")
    old_module.__file__ = str(old_source)
    exec(compile(old_source.read_bytes(), str(old_source), "exec"), old_module.__dict__)
    expected_argv = dict(
        old_module.Submission.expected_commands(
            SimpleNamespace(user=old_auth["user"], script=root / old_auth["script"]), "216897"
        )
    )
    initial = submitted.get("invocation", "")
    require(
        initial.startswith("poll_submit_") and initial[12:].isdigit(),
        "Old teacher submit invocation differs",
    )
    directories = sorted((work / "commands").iterdir())
    require(any(d.name == initial for d in directories), "Old teacher submit audit missing")
    for directory in directories:
        if directory.name == initial:
            labels = ("qos", "assoc", "queue", "node", "sbatch")
            statuses = {"READY"}
        else:
            require(
                directory.name.startswith(("poll_continue_", "poll_submit_")),
                "Old teacher command namespace unknown",
            )
            labels = (
                ("held_job", "stored_script", "held_accounting", "qos", "assoc", "queue", "node")
                if directory.name.startswith("poll_continue_")
                else ("qos", "assoc", "queue", "node")
            )
            statuses = {"WAIT_CAPACITY", "WAIT_ACCOUNTING"}
        marker = read(directory / "RESULT.json")
        require(marker.get("status") in statuses, "Old teacher poll not a safe completed wait")
        files = {p.name: sha(p) for p in directory.glob("*.json") if p.name != "RESULT.json"}
        declared = dict(files)
        if directory.name == initial:
            declared.pop("005_sbatch.intent.json", None)
            declared.pop("005_sbatch.result.json", None)
        require(marker.get("files") == declared, "Old teacher poll inventory changed")
        # WAIT_ACCOUNTING/CAPACITY may return after an initial prefix of read-only queries.
        count = len(files) // 2
        require(
            count > 0
            and len(files) == 2 * count
            and (count == len(labels) if directory.name == initial else count <= len(labels)),
            "Old teacher command coverage incomplete",
        )
        expected = {
            f"{i:03d}_{label}.{kind}.json"
            for i, label in enumerate(labels[:count], 1)
            for kind in ("intent", "result")
        }
        require(set(files) == expected, "Old teacher command pair missing")
        for i, label in enumerate(labels[:count], 1):
            intent = read(directory / f"{i:03d}_{label}.intent.json")
            result = read(directory / f"{i:03d}_{label}.result.json")
            require(
                result.get("returncode") == 0
                and intent.get("argv") == result.get("argv")
                and isinstance(result.get("argv"), list),
                "Old teacher command unknown",
            )
            if label == "sbatch":
                require(
                    result["argv"]
                    == ["sbatch", "--hold", "--parsable", str(root / old_auth["script"])]
                    and result.get("stdout", "").strip().split(";")[0] == "216897",
                    "Old teacher sbatch identity differs",
                )
            else:
                require(
                    result["argv"] == expected_argv[label],
                    "Old teacher wait contains mutation",
                )


def verify_history(root, auth, *, require_retired=False):
    root = Path(root)
    require(auth.get("retirement_directory") == RETIREMENT, "Retirement namespace differs")
    require(
        set(auth.get("retired_directories", [])) == DIRECTORIES,
        "Retired directory coverage differs",
    )
    pins = auth.get("retired_files", {})
    require(isinstance(pins, dict) and pins, "Retired history missing")
    old_operation = root / "operations" / COMMITS[0]
    for name in (
        "RECOVERY_AUTHORIZATION.json",
        "recover_delivery.py",
        "submit_delivery_recovery.py",
        "verify_recovered_migration.py",
        "teacher_history.py",
    ):
        require(
            str((old_operation / name).relative_to(root)) in pins,
            "Retired source/authorization missing",
        )
    old_auth = read(old_operation / "RECOVERY_AUTHORIZATION.json")
    require(
        old_auth["script"] in pins
        and old_auth["qos"] == "soujanya-poria-startfund-2026-03"
        and old_auth.get("cpus") == 22
        and old_auth.get("memory_mib") == 67584
        and old_auth.get("recovery_directory") == OLD_WORK,
        "Old teacher script or resource identity differs",
    )
    for name, digest in {
        **old_auth.get("retired_files", {}),
        **old_auth.get("extra_sources", {}),
    }.items():
        require(pins.get(name) == digest, "Prior teacher historical binding omitted: " + name)
    paths = [bounded(root, name) for name in pins]
    require(
        all(sha(path) == pins[str(path.relative_to(root))] for path in paths),
        "Retired historical bytes changed",
    )
    # Preserve and revalidate the older 216142/216128 chain under its own authority.
    previous_source = old_operation / "teacher_history.py"
    previous = ModuleType("bound_original_teacher_history")
    previous.__file__ = str(previous_source)
    exec(compile(previous_source.read_bytes(), str(previous_source), "exec"), previous.__dict__)
    previous_paths = previous.verify_history(root, old_auth, require_retired=True)
    require(
        all(str(p.relative_to(root)) in pins for p in previous_paths),
        "Prior teacher retirement audit omitted",
    )
    verify_pending_history(root, old_auth)
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
        record.get("status") == "CANCELLED_UNSTARTED_RECOVERY" and record.get("job_id") == "216897",
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
        read(journal / "CANCEL_INTENT.json").get("job_id") == "216897",
        "Cancellation intent differs",
    )
    require(
        read(journal / "CANCEL_INTENT.json").get("authorization") == auth,
        "Cancellation authority differs",
    )
    paths.extend(journal / name for name in files)
    commands = []
    expected = [
        ("held_job", ["scontrol", "show", "job", "216897", "-o"]),
        ("stored_script", ["scontrol", "write", "batch_script", "216897", "-"]),
        (
            "held_accounting",
            [
                "sacct",
                "-n",
                "-P",
                "-j",
                "216897",
                "--format=JobIDRaw,State,Start,ElapsedRaw,AllocTRES",
            ],
        ),
        ("cancel", ["scancel", "--state=PENDING", "216897"]),
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
    old_module = ModuleType("retired_teacher_submission")
    old_module.__file__ = str(old_source)
    exec(compile(old_source.read_bytes(), str(old_source), "exec"), old_module.__dict__)
    old = old_module.Submission(
        root,
        root / "operations" / COMMITS[0] / "RECOVERY_AUTHORIZATION.json",
        validator=lambda *_: None,
    )
    old.validate()
    old.check_held_fields(old_module.fields(commands[0]["stdout"]), "216897")
    require(
        sha(root / old_auth["script"])
        == hashlib.sha256(commands[1]["stdout"].encode()).hexdigest(),
        "Old stored script differs",
    )
    rows = [r.split("|") for r in commands[2]["stdout"].splitlines() if r.strip()]
    require(
        rows == [["216897", "PENDING", "Unknown", "0", ""]]
        or rows == [["216897", "PENDING", "None", "0", ""]],
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
                        "216897",
                        "--format=JobIDRaw,State,Start,ElapsedRaw,AllocTRES,ExitCode",
                    ],
                ),
                ("queue", ["squeue", "-h", "-j", "216897", "-o", "%i|%T"]),
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
