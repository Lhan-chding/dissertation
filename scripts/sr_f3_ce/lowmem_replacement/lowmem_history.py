"""Exact retirement of the held low-memory job after rejected normalization."""

import hashlib
import json
from pathlib import Path
from types import ModuleType

OLD_WORK = "operations/delivery_recovery/20261021_delivery_lowmem"
RETIREMENT = "operations/delivery_recovery_handoff/216983_lowmem_replacement"
COMMITS = ("538aa3dd36e9d5a53f5aa0040b24d3999eab76e9",)
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
    rows = [r for r in rows if r[0] == "216983"]
    require(len(rows) == 1 and len(rows[0]) == 6, "Retired accounting ambiguous")
    _, state, start, elapsed, allocated, exitcode = rows[0]
    require(state == "CANCELLED" or state.startswith("CANCELLED by "), "Old recovery not cancelled")
    require(
        start in ("Unknown", "None") and elapsed == "0" and not allocated and exitcode == "0:0",
        "Old recovery ran or acquired resources",
    )
    return rows[0]


NORMALIZATION_COMMIT = "e184e572a1f184239c5a977e3ac268b881198ec2"
NORMALIZATION_AUDIT = "commands/poll_constraint_normalization_216983"


def original_submit(root):
    source = root / "operations" / COMMITS[0] / "submit_delivery_recovery.py"
    module = ModuleType("bound_original_lowmem_submit")
    module.__file__ = str(source)
    exec(compile(source.read_bytes(), str(source), "exec"), module.__dict__)
    original = module.Submission(
        root, source.parent / "RECOVERY_AUTHORIZATION.json", validator=lambda *_: None
    )
    original.validate()
    return original


def check_old_held(root, auth, fields):
    # The newly bound verifier accepts exactly the two equivalent spellings;
    # validate the old job using its original identity without changing raw text.
    source = Path(__file__).parent / "submit_delivery_recovery.py"
    require(
        auth["extra_sources"].get(str(source.relative_to(root))) == sha(source),
        "New held verifier source differs",
    )
    module = ModuleType("bound_replacement_held_validator")
    module.__file__ = str(source)
    exec(compile(source.read_bytes(), str(source), "exec"), module.__dict__)
    module.Submission.check_held_fields(original_submit(root), fields, "216983")


def check_pending_accounting(raw):
    rows = [line.split("|") for line in raw.splitlines() if line.strip()]
    exact = [row for row in rows if row[0] == "216983"]
    require(
        len(exact) == 1
        and exact[0][1:] in (["PENDING", "None", "0", ""], ["PENDING", "Unknown", "0", ""]),
        "Retired job has no zero-allocation proof",
    )


def verify_pending_history(root, auth, old_auth):
    work = root / OLD_WORK
    submitted = read(work / "SUBMITTED.json")
    original_auth = root / "operations" / COMMITS[0] / "RECOVERY_AUTHORIZATION.json"
    require(
        submitted.get("job_id") == "216983"
        and submitted.get("authorization_sha256") == sha(original_auth)
        and submitted.get("script_sha256") == old_auth["script_sha256"],
        "Old low-memory submission differs",
    )
    initial = submitted.get("invocation", "")
    require(
        initial.startswith("poll_submit_") and initial[12:].isdigit(), "Old initial poll differs"
    )
    norm_op = root / "operations" / NORMALIZATION_COMMIT
    norm_auth_path = norm_op / "NORMALIZATION_AUTHORIZATION.json"
    norm_source = norm_op / "normalize_lowmem_constraint.py"
    for path in (norm_auth_path, norm_source):
        require(
            auth["retired_files"].get(str(path.relative_to(root))) == sha(path),
            "Normalization source/authority missing",
        )
    norm_auth = read(norm_auth_path)
    require(
        norm_auth.get("root") == str(root)
        and norm_auth.get("job_id") == "216983"
        and norm_auth.get("source") == str(norm_source.relative_to(root))
        and norm_auth.get("source_sha256") == sha(norm_source)
        and norm_auth.get("original_authorization")
        == dict(path=str(original_auth.relative_to(root)), sha256=sha(original_auth)),
        "Failed normalization binding differs",
    )
    partial = bounded(root, norm_auth["partial_poll"]["path"])
    require(
        partial.parent == work / "commands"
        and partial.name.startswith("poll_continue_")
        and partial.name[14:].isdigit(),
        "Interrupted poll scope differs",
    )
    audit = work / NORMALIZATION_AUDIT
    require(
        set((work / "commands").iterdir()) == {work / "commands" / initial, partial, audit},
        "Old commands contain unknown attempt",
    )
    original = original_submit(root)
    expected = dict(original.expected_commands("216983"))

    def pair(directory, i, label, *, rc=0):
        intent, result = [
            read(directory / f"{i:03d}_{label}.{kind}.json") for kind in ("intent", "result")
        ]
        argv = (
            expected[label]
            if label != "normalize"
            else ["scontrol", "update", "JobId=216983", "Features=cpu_ok"]
        )
        require(
            intent.get("argv") == result.get("argv") == argv
            and result.get("returncode") == rc
            and intent["at"] <= result["at"],
            "Historical command unknown or changed",
        )
        return result

    first = work / "commands" / initial
    labels = ("qos", "assoc", "queue", "node", "sbatch")
    names = {
        f"{i:03d}_{label}.{kind}.json"
        for i, label in enumerate(labels, 1)
        for kind in ("intent", "result")
    }
    require(
        {p.name for p in first.iterdir()} == names | {"RESULT.json"},
        "Initial submit inventory differs",
    )
    marker = read(first / "RESULT.json")
    require(
        marker.get("status") == "READY"
        and marker.get("files")
        == {name: sha(first / name) for name in names if not name.startswith("005_")},
        "Initial submit marker differs",
    )
    for i, label in enumerate(labels, 1):
        result = pair(first, i, label)
        if label == "sbatch":
            require(
                result.get("stdout", "").strip().split(";")[0] == "216983",
                "Initial submit job differs",
            )
    names = {"001_held_job.intent.json", "001_held_job.result.json"}
    require(
        {p.name for p in partial.iterdir()} == names
        and norm_auth["partial_poll"].get("files") == {name: sha(partial / name) for name in names},
        "Interrupted poll changed or advanced",
    )
    response = pair(partial, 1, "held_job")
    from_fields = original.__class__.check_held_fields.__globals__["fields"]
    raw = from_fields(response["stdout"])
    require(raw.get("Features") == "cpu_ok&cpu_ok", "Interrupted poll has another failure")
    check_old_held(root, auth, raw)
    labels = ("held_job", "stored_script", "held_accounting", "normalize")
    names = {
        f"{i:03d}_{label}.{kind}.json"
        for i, label in enumerate(labels, 1)
        for kind in ("intent", "result")
    }
    require(
        {p.name for p in audit.iterdir()}
        == names | {"AUTHORIZATION.json", "SOURCE.json", "STOP.json"},
        "Normalization audit unknown or advanced",
    )
    require(
        read(audit / "AUTHORIZATION.json") == norm_auth
        and read(audit / "SOURCE.json")
        == dict(
            path=str(norm_source.relative_to(root)),
            sha256=sha(norm_source),
            utf8=norm_source.read_text(),
        ),
        "Normalization complete source/authority archive differs",
    )
    for i, label in enumerate(labels, 1):
        result = pair(audit, i, label, rc=1 if label == "normalize" else 0)
        if label == "held_job":
            raw = from_fields(result["stdout"])
            require(raw.get("Features") == "cpu_ok&cpu_ok", "Normalization precheck differs")
            check_old_held(root, auth, raw)
        elif label == "stored_script":
            require(
                hashlib.sha256(result["stdout"].encode()).hexdigest() == old_auth["script_sha256"],
                "Normalization stored script differs",
            )
        elif label == "held_accounting":
            check_pending_accounting(result["stdout"])
        else:
            require(
                result.get("stdout") == ""
                and "Changing features of a submitted job is not allowed; cancel and resubmit"
                in result.get("stderr", ""),
                "Not the known cluster normalization refusal",
            )
    stop = read(audit / "STOP.json")
    require(
        stop.get("status") == "NORMALIZATION_STOPPED"
        and isinstance(stop.get("error"), str)
        and stop["error"]
        == "PermissionError('Command failed; inspect preserved receipt: normalize')"
        and stop["at"] >= result["at"],
        "Normalization STOP differs",
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
        "lowmem_history.py",
    ):
        require(
            str((old_operation / name).relative_to(root)) in pins,
            "Retired source/authorization missing",
        )
    old_auth = read(old_operation / "RECOVERY_AUTHORIZATION.json")
    require(
        old_auth["script"] in pins
        and old_auth["qos"] == "soujanya-poria-startfund-2026-03"
        and old_auth.get("cpus") == 4
        and old_auth.get("memory_mib") == 12288
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
    previous_source = old_operation / "lowmem_history.py"
    previous = ModuleType("bound_original_teacher_history")
    previous.__file__ = str(previous_source)
    exec(compile(previous_source.read_bytes(), str(previous_source), "exec"), previous.__dict__)
    previous_paths = previous.verify_history(root, old_auth, require_retired=True)
    require(
        all(str(p.relative_to(root)) in pins for p in previous_paths),
        "Prior teacher retirement audit omitted",
    )
    verify_pending_history(root, auth, old_auth)
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
        record.get("status") == "CANCELLED_UNSTARTED_RECOVERY" and record.get("job_id") == "216983",
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
        read(journal / "CANCEL_INTENT.json").get("job_id") == "216983",
        "Cancellation intent differs",
    )
    require(
        read(journal / "CANCEL_INTENT.json").get("authorization") == auth,
        "Cancellation authority differs",
    )
    paths.extend(journal / name for name in files)
    commands = []
    expected = [
        ("held_job", ["scontrol", "show", "job", "216983", "-o"]),
        ("stored_script", ["scontrol", "write", "batch_script", "216983", "-"]),
        (
            "held_accounting",
            [
                "sacct",
                "-n",
                "-P",
                "-j",
                "216983",
                "--format=JobIDRaw,State,Start,ElapsedRaw,AllocTRES",
            ],
        ),
        ("cancel", ["scancel", "--state=PENDING", "216983"]),
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
    check_old_held(root, auth, old_module.fields(commands[0]["stdout"]))
    require(
        sha(root / old_auth["script"])
        == hashlib.sha256(commands[1]["stdout"].encode()).hexdigest(),
        "Old stored script differs",
    )
    rows = [r.split("|") for r in commands[2]["stdout"].splitlines() if r.strip()]
    require(
        rows == [["216983", "PENDING", "Unknown", "0", ""]]
        or rows == [["216983", "PENDING", "None", "0", ""]],
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
                        "216983",
                        "--format=JobIDRaw,State,Start,ElapsedRaw,AllocTRES,ExitCode",
                    ],
                ),
                ("queue", ["squeue", "-h", "-j", "216983", "-o", "%i|%T"]),
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
