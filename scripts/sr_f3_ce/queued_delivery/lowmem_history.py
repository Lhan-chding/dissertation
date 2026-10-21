"""Byte-bound retirement of 217029 and its complete inherited recovery history."""

import hashlib
import json
from pathlib import Path
from types import ModuleType

OLD_OPERATION = "dd1ebe3ad4af603df800cf24ff28df22e1efc484"
OLD_WORK = "operations/delivery_recovery/20261021_delivery_lowmem_replacement"
RETIREMENT = "operations/delivery_recovery_handoff/217029_user_cancel"
OLD_AUTH_SHA256 = "3ba0914d7d88210dd6d5f2dc456119cd114fbf22dd4f9a440648b4c1bb7b02c7"


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


def inventory(root, directory):
    return {
        str(p.relative_to(root)): sha(p)
        for p in bounded(root, directory).rglob("*")
        if p.is_file() and not p.name.endswith(".lock")
    }


def load(path, name):
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


def accounting(raw, *, terminal):
    rows = [line.split("|") for line in raw.splitlines() if line.strip()]
    rows = [r for r in rows if r[0] == "217029"]
    require(len(rows) == 1 and len(rows[0]) == 6, "Retired accounting ambiguous")
    row = rows[0]
    require(
        (row[1] == "CANCELLED" or row[1].startswith("CANCELLED by "))
        if terminal
        else row[1] == "PENDING",
        "Retired scheduler state differs",
    )
    require(
        row[2] in ("Unknown", "None") and row[3:] == ["0", "", "0:0"],
        "Retired job ran or acquired resources",
    )
    return row


def verify_retirement(root, old_auth, old_submit, pins):
    directory = root / RETIREMENT
    record = read(directory / "RETIREMENT.json")
    authority = read(directory / "AUTHORIZATION.json")
    require(
        record.get("status") == "CANCELLED_UNSTARTED_RECOVERY"
        and record.get("job_id") == authority.get("job_id") == "217029"
        and record.get("prior_authorization_sha256")
        == authority.get("prior_authorization_sha256")
        == OLD_AUTH_SHA256,
        "Retirement authority differs",
    )
    require(
        authority.get("user_instruction")
        == (
            "Cancel original 217029 and complete full package by another legitimate method; "
            "no experimental rerun"
        ),
        "Retirement user instruction differs",
    )
    fmt = "--format=JobIDRaw,State,Start,ElapsedRaw,AllocTRES,ExitCode"
    expected = [
        ("001_held", ["scontrol", "show", "job", "217029", "-o"]),
        ("002_script", ["scontrol", "write", "batch_script", "217029", "-"]),
        ("003_accounting", ["sacct", "-n", "-P", "-j", "217029", fmt]),
        ("004_cancel", ["scancel", "--state=PENDING", "217029"]),
        ("005_queue", ["squeue", "-h", "-j", "217029", "-o", "%i|%T|%R"]),
        ("006_terminal", ["sacct", "-n", "-P", "-j", "217029", fmt]),
    ]
    names = {"AUTHORIZATION.json", "PRECHECK.json", "RETIREMENT.json"} | {
        label + "." + kind + ".json" for label, _ in expected for kind in ("intent", "result")
    }
    require(
        {p.name for p in directory.iterdir() if p.is_file()} == names,
        "Retirement audit inventory differs",
    )
    prior = authority["at"]
    replies = {}
    for label, argv in expected:
        intent = read(directory / (label + ".intent.json"))
        result = read(directory / (label + ".result.json"))
        require(
            intent.get("argv") == result.get("argv") == argv and result.get("returncode") == 0,
            "Retirement command differs",
        )
        require(
            prior <= intent["at"] <= result["at"] <= record["at"],
            "Retirement command chronology differs",
        )
        prior = result["at"]
        replies[label] = result["stdout"]
    old_submit.check_held_fields(old_submit_module_fields(replies["001_held"]), "217029")
    require(
        hashlib.sha256(replies["002_script"].encode()).hexdigest() == old_auth["script_sha256"],
        "Retired stored script differs",
    )
    accounting(replies["003_accounting"], terminal=False)
    require(not replies["005_queue"].strip(), "Retired job remains queued")
    require(
        accounting(replies["006_terminal"], terminal=True) == record.get("terminal"),
        "Retirement terminal proof differs",
    )
    precheck = read(directory / "PRECHECK.json")
    require(
        precheck.get("status") == "UNSTARTED_ZERO_ALLOCATION"
        and precheck.get("original_files") == inventory(root, OLD_WORK),
        "Retired original bytes changed",
    )
    for name, digest in precheck["original_files"].items():
        require(pins.get(name) == digest, "Retired original not authorized")
    require(
        read(directory / "003_accounting.result.json")["at"]
        <= precheck["at"]
        <= read(directory / "004_cancel.intent.json")["at"],
        "Retirement precheck chronology differs",
    )


def old_submit_module_fields(raw):
    # Same parser as the preserved dd1 submit module; raw bytes remain archived.
    import re

    return dict(re.findall(r"(\S+?)=(\S*)", raw))


def verify_polls(root, old_auth, original):
    work = root / OLD_WORK
    submitted = read(work / "SUBMITTED.json")
    require(
        submitted.get("job_id") == "217029"
        and submitted.get("authorization_sha256") == OLD_AUTH_SHA256
        and submitted.get("script_sha256") == old_auth["script_sha256"],
        "Retired submission identity differs",
    )
    initial = submitted.get("invocation", "")
    require(
        initial.startswith("poll_submit_") and initial[12:].isdigit(),
        "Retired submission invocation differs",
    )
    commands = dict(original.expected_commands("217029"))
    for directory in (work / "commands").iterdir():
        require(
            directory.is_dir() and directory.name.startswith("poll_"),
            "Unexpected retired command directory",
        )
        record = read(directory / "RESULT.json")
        if directory.name == initial:
            require(record.get("status") == "READY", "Old submission not ready")
            labels = ("qos", "assoc", "queue", "node", "sbatch")
            sealed = 4
        else:
            require(
                record.get("status") in ("WAIT_CAPACITY", "WAIT_ACCOUNTING"),
                "Old poll not a complete safe wait",
            )
            labels = ("held_job", "stored_script", "held_accounting")
            if record["status"] == "WAIT_CAPACITY":
                labels += ("qos", "assoc", "queue", "node")
            sealed = len(labels)
        expected = {
            f"{i:03d}_{label}.{kind}.json"
            for i, label in enumerate(labels, 1)
            for kind in ("intent", "result")
        }
        require(
            {p.name for p in directory.iterdir()} == expected | {"RESULT.json"},
            "Retired poll inventory differs",
        )
        require(
            record.get("files")
            == {name: sha(directory / name) for name in expected if int(name[:3]) <= sealed},
            "Retired poll seal differs",
        )
        for i, label in enumerate(labels, 1):
            intent = read(directory / f"{i:03d}_{label}.intent.json")
            result = read(directory / f"{i:03d}_{label}.result.json")
            require(
                intent.get("argv") == result.get("argv") == commands[label]
                and result.get("returncode") == 0,
                "Retired query outcome unknown",
            )
            if label == "sbatch":
                require(
                    result.get("stdout", "").strip().split(";")[0] == "217029",
                    "Retired sbatch identity differs",
                )
            elif label == "held_job":
                original.check_held_fields(old_submit_module_fields(result["stdout"]), "217029")
            elif label == "stored_script":
                require(
                    hashlib.sha256(result["stdout"].encode()).hexdigest()
                    == old_auth["script_sha256"],
                    "Retired queried script differs",
                )


def verify_history(root, auth, *, require_retired=False):
    root = Path(root)
    require(auth.get("retirement_directory") == RETIREMENT, "Retirement namespace differs")
    pins = auth.get("retired_files", {})
    require(isinstance(pins, dict) and pins, "Retired history missing")
    paths = [bounded(root, name) for name in pins]
    require(
        all(sha(p) == pins[str(p.relative_to(root))] for p in paths),
        "Retired historical bytes changed",
    )
    operation = root / "operations" / OLD_OPERATION
    authority = operation / "RECOVERY_AUTHORIZATION.json"
    require(
        str(authority.relative_to(root)) in pins and sha(authority) == OLD_AUTH_SHA256,
        "Original authorization differs",
    )
    old_auth = read(authority)
    required = {**old_auth["retired_files"], **old_auth["extra_sources"], **old_auth["files"]}
    required[old_auth["script"]] = old_auth["script_sha256"]
    required[str((operation / "recover_delivery.py").relative_to(root))] = old_auth["source_sha256"]
    require(
        all(pins.get(name) == digest for name, digest in required.items()),
        "Inherited historical binding omitted",
    )
    directories = set(old_auth["retired_directories"]) | {OLD_WORK, RETIREMENT}
    require(
        set(auth.get("retired_directories", [])) == directories,
        "Retired directory coverage differs",
    )
    for directory in directories:
        require(
            inventory(root, directory)
            == {n: d for n, d in pins.items() if n.startswith(directory + "/")},
            "Retired directory inventory changed: " + directory,
        )
    previous = load(operation / "lowmem_history.py", "bound_dd1_history")
    inherited = previous.verify_history(root, old_auth, require_retired=True)
    require(
        all(str(p.relative_to(root)) in pins for p in inherited),
        "Inherited retirement audit omitted",
    )
    old_submit = load(operation / "submit_delivery_recovery.py", "bound_dd1_submission")
    original = old_submit.Submission(root, authority, validator=lambda *_: None)
    original.validate()
    for marker in (
        "ALLOCATION_VERIFIED",
        "CAPACITY_BEFORE_RELEASE",
        "RELEASE_INTENT",
        "RELEASE",
        "STARTED",
        "PROCESS",
        "END",
        "MEMORY_PROFILE",
    ):
        require(
            not (root / OLD_WORK / (marker + ".json")).exists(), "Retired job advanced: " + marker
        )
    verify_polls(root, old_auth, original)
    verify_retirement(root, old_auth, original, pins)
    return paths
