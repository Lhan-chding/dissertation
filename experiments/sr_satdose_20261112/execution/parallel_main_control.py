"""One fresh MAIN endpoint in a two-job wave, after verified terminal admission.

Each job holds a shared global worker lease (excludes legacy exclusive workers)
and an exclusive lease for its own seed. Pair authorization is written before
either job is released. An occupied endpoint is immutable, even after failure.
"""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager, nullcontext, suppress
from pathlib import Path

ROOT = Path("/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/prepared_data_v1")
SERIAL = "external_main_code_4190bc3"
EXTERNAL = "external_parallel_main_v1"
BATCH = "execution_releases/parallel_main_v1"
SERIAL_SHA = "e30ae9b2bcbf02d1b1ff788cffbf951d5ca48696170fbba5107d799f1e226c97"
TERMINAL_SHA = "3c43517e537cb76c4637e940418599325d088ee957476575cc418a9c53d3d95f"
TRAIN_RELEASE = dict(
    path="execution_releases/FIRST_TWO_RELEASE_71018e5.json",
    sha256="22f3b7b260670628398154652455520cd25169234f66a827d3d117def924438f",
)
SEEDS = (11031, 11032)
WAVES = (("START", 0), ("T0", 8), ("T0", 16), ("T0", 24), ("T1", 8), ("T1", 16), ("T1", 24))


def require(condition, message):
    if not condition:
        raise PermissionError("HOLD_MAIN_PARALLEL: " + message)


def load_source(name, path, expected):
    require(
        path.resolve(strict=True) == path
        and path.is_file()
        and re.fullmatch("[a-f0-9]{64}", expected),
        "Noncanonical source or digest",
    )
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == expected, "Source changed: " + str(path))
    require(name not in sys.modules, "Fresh interpreter required")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(data, str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def release_directory():
    return ROOT / BATCH


def wave_directory(wave):
    return release_directory() / f"wave-{wave:02d}"


def endpoint_keys(waves):
    return {f"{seed}/{arm}/{step}" for arm, step in waves for seed in SEEDS}


def bootstrap(path, digest):
    """Authenticate code and the original published terminal before model imports."""
    sys.dont_write_bytecode = True
    serial_root = ROOT / SERIAL
    m = load_source("satdose_sealed_main", serial_root / "execution/main_control.py", SERIAL_SHA)
    require(
        m.ROOT == ROOT and path == release_directory() / "RELEASE.json" and m.sha(path) == digest,
        "Wrong shared release",
    )
    release = m.read(path)
    expected = dict(
        schema="sr-satdose-main-parallel-release-v1",
        status="AUDITED_FOR_FIXED_MAIN_PARALLEL_ONLY",
        permission="FIXED_MAIN_ONLY_NO_TRAINING",
        experiment_id=m.EXPERIMENT,
        evidence_root=str(ROOT),
        design_freeze_sha256=m.FREEZE_SHA,
        config_sha256=m.CONFIG_SHA,
        original_train_release=TRAIN_RELEASE,
        external_root=str(serial_root),
        parallel_root=str(ROOT / EXTERNAL),
    )
    require(all(release.get(k) == v for k, v in expected.items()), "Untrusted parallel release")
    for root_key, sources_key in [
        ("external_root", "external_sources"),
        ("parallel_root", "parallel_sources"),
    ]:
        require(
            m.inventory(Path(release[root_key])) == release[sources_key],
            "External inventory differs",
        )
    here = Path(__file__).resolve(strict=True)
    require(
        here == ROOT / EXTERNAL / "execution/parallel_main_control.py"
        and m.sha(here) == release["parallel_sources"]["execution/parallel_main_control.py"],
        "Unbound controller",
    )
    authority = serial_root / "execution/terminal_authority.py"
    authority_ref = dict(path=str(authority), sha256=TERMINAL_SHA)
    spec = m.read(m.bound(release["terminal_spec"]))
    require(
        release["terminal_authority"] == spec["authority"] == authority_ref
        and spec["external_root"] == str(serial_root)
        and spec["external_sources"] == release["external_sources"]
        and spec["original_train_release"] == TRAIN_RELEASE,
        "Terminal release/source mismatch",
    )
    ta = load_source("satdose_sealed_terminal", authority, TERMINAL_SHA)
    terminal = ta.verify(spec)
    require(
        terminal.root == ROOT
        and terminal.package == ROOT / "train_code_71018e5"
        and terminal.original == m.read(m.bound(TRAIN_RELEASE)),
        "Wrong terminal TRAIN identity",
    )
    m._configure(ta, terminal)
    require(
        m.first_two.ROOT == m.storage.ROOT == ROOT and m.storage.FREEZE_SHA == m.FREEZE_SHA,
        "Original environment differs",
    )
    for key in ("terminal_publication", "terminal_manifest"):
        m.bound(release[key])
    admission = m.read(m.bound(release["terminal_admission"]))
    validate_admission(admission, terminal)
    coverage = load_source(
        "satdose_fixed_parallel_plan",
        ROOT / EXTERNAL / "operations/satdose_parallel_main_plan.py",
        release["parallel_sources"]["operations/satdose_parallel_main_plan.py"],
    )
    return m, release, terminal, coverage


def validate_admission(admission, terminal):
    require(
        admission.get("schema") == "sr-satdose-parallel-main-admission-v1"
        and admission.get("status") == "TERMINAL_AND_OPERATOR_AUDITS_VERIFIED"
        and admission.get("terminal_admission") == terminal.receipt
        and admission.get("terminal_audit") == terminal.spec["report"]
        and admission.get("original_train_release") == terminal.spec["original_train_release"],
        "Terminal/operator audit gate mismatch",
    )
    require(
        set(admission["continuation_audits"]) == set(terminal.audit_spec["continuation_results"])
        and len(admission["continuation_audits"]) == 12
        and set(admission["trained_dev_audits"])
        == {f"{s}/{a}/{t}" for s in SEEDS for a in ("T0", "T1") for t in (2, 8, 16)},
        "Independent audit population mismatch",
    )


class LaneLease:
    def __init__(self, seed):
        require(seed in SEEDS, "Unregistered lane")
        self.seed = seed
        self.stack = ExitStack()
        self.files = []

    def __enter__(self):
        locks = [
            (ROOT / "DIAGNOSTIC_WORKER.lock", fcntl.LOCK_SH),
            (release_directory() / f"LANE_{self.seed}.lock", fcntl.LOCK_EX),
        ]
        try:
            for path, mode in locks:
                require(path.resolve(strict=True) == path and path.is_file(), "Invalid lease path")
                fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
                self.stack.callback(os.close, fd)
                fcntl.flock(fd, mode | fcntl.LOCK_NB)
                self.files.append((fd, path))
            self.active()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def active(self):
        require(len(self.files) == 2, "Inactive lane lease")
        for fd, path in self.files:
            require(
                path.resolve(strict=True) == path and os.path.samestat(os.fstat(fd), path.stat()),
                "Replaced lane/global lease",
            )

    def __exit__(self, *exc):
        self.files.clear()
        return self.stack.__exit__(*exc)


def validate_pair(value, release_ref, wave, job_seed=None, job_id=None):
    arm, step = WAVES[wave]
    require(
        value.get("schema") == "sr-satdose-main-pair-v1"
        and value.get("release") == release_ref
        and type(value.get("wave")) is int
        and value["wave"] == wave
        and (value["arm"], value["step"]) == (arm, step),
        "Pair belongs to another release or wave",
    )
    jobs = value["jobs"]
    require(
        set(jobs) == {str(s) for s in SEEDS}
        and all(type(j) is int and j > 0 for j in jobs.values())
        and len(set(jobs.values())) == 2,
        "Invalid paired job IDs",
    )
    if job_seed is not None:
        require(str(jobs[str(job_seed)]) == job_id, "Foreign Slurm job for this lane")


def wave_healthy(wave):
    p = wave_directory(wave)
    if os.path.lexists(p):
        require(p.resolve(strict=True) == p and p.is_dir(), "Noncanonical wave directory")
        require(
            all(
                "HOLD" not in x.name and "ERROR" not in x.name and not x.is_symlink()
                for x in p.iterdir()
            ),
            "Wave incident or symlink",
        )


def check_slots(m, wave, seed, *, active, worker):
    slots = m._existing_slots()
    expected = endpoint_keys(WAVES[:wave])
    arm, step = WAVES[wave]
    own = m.endpoint_key(seed, arm, step)
    other = m.endpoint_key(next(s for s in SEEDS if s != seed), arm, step)
    if active:
        expected.add(own)
    require(
        set(slots) == expected or (worker and set(slots) == expected | {other}),
        "Missing predecessor, duplicate/unknown endpoint or unexpected wave",
    )
    return slots


def previous_endpoints(m, release_ref, terminal, registry, wave, slots):
    observed = m.TA.Observed(ROOT)
    previous = {}
    processes = []
    config = m.bound(m.read(registry.freeze_path)["artifacts"]["config"])
    for index, (arm, step) in enumerate(WAVES[:wave]):
        wave_healthy(index)
        pair_ref = m.ref(wave_directory(index) / "PAIR.json")
        pair = observed.json(pair_ref)
        validate_pair(pair, release_ref, index)
        released_ref = m.ref(wave_directory(index) / "RELEASE_RESULT.json")
        validate_released(observed.json(released_ref), pair, pair_ref)
        require(pair["previous_results"] == previous, "Previous wave chain changed")
        for seed in SEEDS:
            key = m.endpoint_key(seed, arm, step)
            slot = slots[key]
            result_ref = m.ref(slot / "RESULT.json")
            result = observed.json(result_ref)
            require(
                result.get("status") == "MAIN_ENDPOINT_COMPLETE"
                and (result["start_seed"], result["arm"], result["step"]) == (seed, arm, step)
                and result["responses"] == m.allocations()[seed, arm, step]
                and result["release"] == release_ref
                and result["terminal"] == terminal.receipt
                and result["pair"] == pair_ref
                and result["released"] == released_ref
                and result["job_id"] == pair["jobs"][str(seed)],
                "Prior endpoint identity or population differs",
            )
            require(
                observed.ref(result["receipt"]) == slot / "endpoint/RECEIPT.json"
                and observed.ref(result["endpoint_binding"]) == slot / "ENDPOINT_BINDING.json"
                and observed.ref(result["close"]) == slot / "CLOSE.json",
                "Prior artifacts relocated",
            )
            binding = observed.json(result["endpoint_binding"])
            ep = binding["endpoint"]
            require(
                (ep["start_seed"], ep["arm"], ep["step"]) == (seed, arm, step)
                and binding["checkpoint"] == result["checkpoint"],
                "Prior binding belongs to another endpoint",
            )
            if arm != "START":
                require(
                    binding["checkpoint"]["commit"]
                    == terminal.audit_spec["trusted_commits"][f"{seed}/{arm}"][step - 1],
                    "Prior endpoint used another trained checkpoint",
                )
            else:
                require(
                    binding["checkpoint"] is None,
                    "Starting endpoint substituted a trained checkpoint",
                )
            evaluated = m.evaluation.analyze_completed_endpoint(
                config,
                registry.freeze_path,
                ROOT,
                observed.ref(result["receipt"]),
                expected_receipt_sha256=result["receipt"]["sha256"],
                endpoint_binding_path=observed.ref(result["endpoint_binding"]),
                expected_endpoint_binding_sha256=result["endpoint_binding"]["sha256"],
            )
            require(
                evaluated["summary"]["scope"] == "MAIN"
                and evaluated["summary"]["metrics"]["assigned"] == result["responses"],
                "Prior independent population differs",
            )
            receipt = observed.json(result["receipt"])
            ids = []
            for reference in receipt["complete_files"]:
                path = observed.ref(reference)
                complete = m.read(path)
                ids.append(complete["request_id"])
                observed.path(
                    Path(complete["native_capture_path"]), complete["native_capture_sha256"]
                )
                observed.path(path.with_name(path.name.replace(".COMPLETE.json", ".BEGIN.json")))
            observed.ref(receipt["raw"])
            observed.ref(receipt["accounting"])
            observed.tree(slot)
            m._capture_inventory(slot / "endpoint", ids)
            require(result["process_identity"] not in processes, "Prior endpoints reused a process")
            processes.append(result["process_identity"])
            previous[key] = result_ref
    observed.stable()
    return observed, previous, processes


@contextmanager
def file_mutex(path, exclusive=True):
    """Coordinate reads with other parallel workers; do not lock TransactionStore itself."""
    require(path.resolve(strict=True) == path and path.is_file(), "Missing coordination file")
    fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        require(os.path.samestat(os.fstat(fd), path.stat()), "Coordination file replaced")
        yield
    finally:
        os.close(fd)


def preflight(path, digest, seed, arm, step, *, worker=False):
    require(
        type(seed) is int and seed in SEEDS and type(step) is int and (arm, step) in WAVES,
        "Unregistered endpoint",
    )
    m, release, terminal, coverage = bootstrap(path, digest)
    wave = WAVES.index((arm, step))
    release_ref = m.ref(path)
    require(release_ref["sha256"] == digest, "Release changed during bootstrap")
    m._hold_free()
    wave_healthy(wave)
    freeze = m.read(ROOT / "DESIGN_FREEZE.json")
    require(m.sha(ROOT / "DESIGN_FREEZE.json") == m.FREEZE_SHA, "Design freeze changed")

    def artifact(key):
        return m.bound(freeze["artifacts"][key])

    registry = m.backend.FrozenRegistry(
        artifact("config"),
        ROOT,
        ROOT / "DESIGN_FREEZE.json",
        [
            artifact(k)
            for k in (
                "train_manifest",
                "dry_manifest",
                "dev_manifest",
                "prefix_calib_manifest",
                "main_manifest",
            )
        ],
        artifact("inputs"),
        artifact("images"),
    )
    manifest_path = artifact("main_manifest")
    plan = coverage.build_plan(manifest_path)
    require(
        plan == m.read(m.bound(release["plan"]))
        and tuple(coverage.lane_endpoints(seed)) == tuple((seed, a, t) for a, t in WAVES),
        "Allocation differs from frozen plan",
    )
    selection, requests = m.runtime._selection(registry, seed, arm, step, "MAIN")
    rows = m.tx.rows(manifest_path)
    m.evaluation.validate_manifest(rows, "MAIN")
    require(
        m.evaluation.endpoint_allocations("MAIN") == m.allocations(), "Original allocation differs"
    )
    manifest = [r for r in rows if (r["start_seed"], r["arm"], r["step"]) == (seed, arm, step)]
    ids = [r["request_id"] for r in manifest]
    count = m.allocations()[seed, arm, step]
    registered = plan["lanes"][str(seed)]["endpoints"][wave]
    require(
        len(ids) == registered["responses"] == count
        and ids == [r["request_id"] for r in requests]
        and m.backend.object_sha(ids) == registered["request_ids_sha256"],
        "Incomplete or reordered requests",
    )
    pair_ref = released_ref = None
    job = None
    pair_path = wave_directory(wave) / "PAIR.json"
    if worker:
        pair_ref = m.ref(pair_path)
        pair = m.read(m.bound(pair_ref))
        job = os.environ.get("SLURM_JOB_ID")
        validate_pair(pair, release_ref, wave, seed, job)
        released_ref = m.ref(wave_directory(wave) / "RELEASE_RESULT.json")
        released = m.read(m.bound(released_ref))
        validate_released(released, pair, pair_ref)
        verify_scheduler_publication(m, pair, released)
    with file_mutex(pair_path, exclusive=False) if worker else nullcontext():
        slots = check_slots(m, wave, seed, active=False, worker=worker)
        if worker:
            verify_peer_claim(m, slots, wave, seed, pair_ref, pair, release_ref)
    prior, previous, processes = previous_endpoints(m, release_ref, terminal, registry, wave, slots)
    if worker:
        require(pair["previous_results"] == previous, "Prior RESULTS changed since Slurm admission")
    process = m._process()
    require(process not in processes, "Reused MAIN endpoint process")
    source = registry.config["start_bindings"][str(seed)]
    store = identity = commit = checkpoint = None
    policy = source["policy_sha256"]
    if arm != "START":
        commit = terminal.audit_spec["trusted_commits"][f"{seed}/{arm}"][step - 1]
        marker = m.read(m.bound(commit))
        identity = marker["identity"]
        require(
            identity["train_seed"] == seed
            and identity["arm"] == arm
            and identity["execution_sources_sha256"]
            == m.backend.object_sha(terminal.original["execution_sources"]),
            "Foreign original training checkpoint",
        )
        lock = m.read(artifact("source_lock"))
        native = m.fork_loader.load_native_ops(lock["native_src"], lock["native_files"])
        store = m.tx.TransactionStore(
            ROOT,
            ROOT / "DESIGN_FREEZE.json",
            m.FREEZE_SHA,
            identity,
            native=native,
            genesis_reference=marker["transaction"]["genesis_reference"],
        )
        with file_mutex(path):
            require(store.inspect_committed(step, commit["sha256"]) == marker, "Checkpoint changed")
        checkpoint = dict(commit=commit, state=m.ref(store.path / f"step-{step:03d}/state.pt"))
        policy = marker["policy_sha256"]
    endpoint = {k: v for k, v in selection.items() if k != "scope"} | dict(
        policy_sha256=policy, reference_sha256=source["reference_sha256"]
    )
    modules = dict(
        evaluation_runtime=m.runtime,
        evaluation_loader=m.runtime.loader,
        training_state=m.runtime.state,
        training_transaction=m.tx,
    )
    sources = {}
    for name, module in modules.items():
        p = Path(module.__file__).resolve()
        digest = m.sha(p)
        require(
            p.is_relative_to(terminal.package)
            and terminal.original["sources"][str(p.relative_to(terminal.package))] == digest,
            "Altered evaluation runtime source",
        )
        sources[name] = dict(path=str(p), sha256=digest)
    here = Path(__file__).resolve()
    sources["authority"] = dict(path=str(here), sha256=m.sha(here))
    budget = m.storage._budget(release["storage_budget"], m.storage._now())
    require(budget.get("scope") == "MAIN_ONLY", "Wrong storage budget")
    terminal.stable()
    prior.stable()
    return dict(
        m=m,
        release=release,
        release_ref=release_ref,
        terminal=terminal,
        freeze=freeze,
        registry=registry,
        selection=selection,
        requests=requests,
        manifest=manifest,
        count=count,
        endpoint=endpoint,
        sources=sources,
        sources_sha256=m.backend.object_sha(sources),
        start=seed,
        arm=arm,
        step=step,
        directory=m._slot(seed, arm, step),
        process=process,
        prior=prior,
        store=store,
        identity=identity,
        commit=commit,
        checkpoint=checkpoint,
        budget=budget,
        wave=wave,
        pair=pair_ref,
        released=released_ref,
        job_id=int(job) if job is not None else None,
        previous_results=previous,
    )


def verify_peer_claim(m, slots, wave, seed, pair_ref, pair, release_ref):
    other = next(s for s in SEEDS if s != seed)
    arm, step = WAVES[wave]
    peer = slots.get(m.endpoint_key(other, arm, step))
    if peer is not None:
        claim = m.read(m.bound(m.ref(peer / "CLAIM.json")))
        require(
            claim.get("pair") == pair_ref
            and claim.get("release") == release_ref
            and claim.get("job_id") == pair["jobs"][str(other)]
            and (claim["start_seed"], claim["arm"], claim["step"]) == (other, arm, step),
            "Unrecognized sibling claim",
        )


def check_peer_scheduler(job):
    """A killed job may be unable to publish its own HOLD."""
    env = {k: v for k, v in os.environ.items() if k != "SLURM_CLUSTERS"}
    result = subprocess.run(
        ["scontrol", "show", "job", "-o", str(job)],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    if result.returncode == 0:
        fields = dict(word.split("=", 1) for word in result.stdout.split() if "=" in word)
        require(fields.get("JobId") == str(job), "Foreign peer status")
        state, code = fields.get("JobState"), fields.get("ExitCode")
        require(
            state != "PENDING"
            or (
                fields.get("Reason") not in ("JobHeldUser", "JobHeldAdmin")
                and fields.get("Priority", "").isdigit()
                and int(fields["Priority"]) > 0
            ),
            "Sibling job is held",
        )
    else:
        result = subprocess.run(
            ["sacct", "-X", "-nP", "-j", str(job), "--format=JobIDRaw,State%40,ExitCode"],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
        lines = [line.strip().split("|") for line in result.stdout.splitlines() if line.strip()]
        require(
            result.returncode == 0
            and len(lines) == 1
            and len(lines[0]) == 3
            and lines[0][0] == str(job),
            "Peer job state unknown",
        )
        _, state, code = lines[0]
    require(
        state in ("PENDING", "CONFIGURING", "RUNNING", "COMPLETING", "COMPLETED")
        and (state != "COMPLETED" or code == "0:0"),
        "Peer job failed or held: " + str(state),
    )


def authority_for(ctx, lease):
    m = ctx["m"]

    class ParallelAuthority(m._authority_type()):
        # The original runtime verifies the exact file implementing verify().
        # This class is bound to the parallel source, not the frozen serial SHA.
        def __init__(self):
            require(type(lease) is LaneLease and lease.seed == ctx["start"], "Incorrect lane lease")
            lease.active()
            self.ctx = ctx
            self.lease = lease
            self.index = 0
            self.claim = None
            self.process = ctx["process"]
            self.output = ctx["directory"] / "endpoint"
            self.next_peer_check = 0.0
            m.runtime.AuthenticatedEvaluationAuthority.__init__(self, ctx["sources"]["authority"])

        def claim_value(self):
            return super().claim_value() | dict(
                pair=ctx["pair"], released=ctx["released"], job_id=ctx["job_id"]
            )

        def fast(self):
            self.lease.active()
            require(
                self.claim is not None and m._process() == self.process, "Lost claim or process"
            )
            m.bound(self.claim)
            m._hold_free()
            wave_healthy(ctx["wave"])
            m.bound(ctx["release_ref"])
            require(m.sha(ROOT / "DESIGN_FREEZE.json") == m.FREEZE_SHA, "Design freeze changed")
            for name in (
                "original_train_release",
                "terminal_spec",
                "terminal_publication",
                "terminal_manifest",
                "terminal_admission",
                "storage_budget",
                "plan",
            ):
                m.bound(ctx["release"][name])
            for base, files in (
                (Path(ctx["release"]["parallel_root"]), ctx["release"]["parallel_sources"]),
                (Path(ctx["release"]["external_root"]), ctx["release"]["external_sources"]),
                (ctx["terminal"].package, ctx["terminal"].original["sources"]),
            ):
                for name, digest in files.items():
                    require(m.sha(base / name) == digest, "Live MAIN source changed")
            m.bound(ctx["terminal"].receipt)
            m.bound(ctx["released"])
            for ref in ctx["previous_results"].values():
                m.bound(ref)
            pair_path = m.bound(ctx["pair"])
            pair = m.read(pair_path)
            with file_mutex(pair_path, exclusive=False):
                slots = check_slots(m, ctx["wave"], ctx["start"], active=True, worker=True)
                verify_peer_claim(
                    m, slots, ctx["wave"], ctx["start"], ctx["pair"], pair, ctx["release_ref"]
                )
            if time.monotonic() >= self.next_peer_check:
                other = next(s for s in SEEDS if s != ctx["start"])
                check_peer_scheduler(pair["jobs"][str(other)])
                self.next_peer_check = time.monotonic() + 30

        def boundary(self):
            self.next_peer_check = 0.0
            self.fast()
            require(self.index in (0, ctx["count"]), "Incomplete endpoint boundary")
            ctx["terminal"].stable()
            ctx["prior"].stable()
            m._capture_inventory(
                self.output, [r["request_id"] for r in ctx["requests"][: self.index]]
            )
            if ctx["store"] is not None:
                # Only the frozen transaction inspection needs serialization.
                with file_mutex(m.bound(ctx["release_ref"])):
                    ctx["store"].inspect_committed(ctx["step"], ctx["commit"]["sha256"])

        def verify(self, **kwargs):
            return super().verify(**kwargs)

    return ParallelAuthority()


def validate_released(value, pair, pair_ref):
    require(
        value.get("status") == "MAIN_PARALLEL_PAIR_RELEASE_CONFIRMED"
        and value.get("wave") == pair["wave"]
        and value.get("jobs") == pair["jobs"]
        and value.get("pair") == pair_ref
        and value.get("release") == pair["release"],
        "Unconfirmed pair release",
    )


def verify_scheduler_publication(m, pair, released):
    directory = wave_directory(pair["wave"])
    ordered_ids = [str(pair["jobs"][str(seed)]) for seed in SEEDS]
    path = m.bound(released["release_response"])
    require(path == directory / "RELEASE_RESPONSE.json", "Relocated Slurm release response")
    response = m.read(path)
    require(
        response["returncode"] == 0
        and response["argv"] == ["scontrol", "release", ",".join(ordered_ids)],
        "Paired Slurm release failed or named different jobs",
    )
    for seed in SEEDS:
        key = str(seed)
        held_path = m.bound(pair["held_refs"][key])
        require(held_path == directory / f"HELD_{seed}.json", "Relocated held-job identity")
        held = m.read(held_path)["fields"]
        post = released["post_release"][key]["fields"]
        expected = dict(
            JobId=str(pair["jobs"][key]),
            JobName=f"LH-SATDOSE-MAIN-{seed}-{pair['arm']}-s{pair['step']:02d}",
            QOS="soujanya-poria-startfund-2026-03",
            Account="rose",
            Partition="cluster02",
        )
        for fields in (held, post):
            require(
                all(fields.get(k) == v for k, v in expected.items())
                and {"gres/gpu=2", "gres/gpu:pro6000=2"} <= set(fields["ReqTRES"].split(",")),
                "Foreign job or GPU allocation in pair",
            )
        require(
            held.get("JobState") == "PENDING"
            and held.get("Reason") == "JobHeldUser"
            and held.get("Priority") == "0"
            and post.get("Reason") not in ("JobHeldUser", "JobHeldAdmin"),
            "Pair hold/release mismatch",
        )


def execute(path, digest, seed, arm, step):
    with LaneLease(seed) as lease:
        c = preflight(path, digest, seed, arm, step, worker=True)
        m = c["m"]
        tx = m.tx
        allocation = m.first_two._allocation()
        require(
            allocation["job_id"] == str(c["job_id"])
            and allocation["observation"].get("JobName")
            == f"LH-SATDOSE-MAIN-{seed}-{arm}-s{step:02d}",
            "Wrong running allocation",
        )
        authority = authority_for(c, lease)
        directory = c["directory"]
        output = directory / "endpoint"
        with file_mutex(m.bound(c["pair"])):
            # A peer reader sees either no slot or a complete, job-bound claim.
            check_slots(m, c["wave"], seed, active=False, worker=True)
            directory.mkdir(parents=True, exist_ok=False)
            tx.fsync_directory(directory.parent)
            output.mkdir()
            tx.fsync_directory(directory)
            tx.write_once(directory / "CLAIM.json", authority.claim_value())
            authority.activate(m.ref(directory / "CLAIM.json"))
        handle = None
        try:
            tx.write_once(
                directory / "STORAGE.json", m.storage_check(output, c["release"]["storage_budget"])
            )
            authority.boundary()
            binding = dict(
                schema="sr-satdose-evaluation-endpoint-binding-v1",
                scope="MAIN",
                endpoint=c["endpoint"],
                requests=c["count"],
                request_ids_sha256=m.backend.object_sha([r["request_id"] for r in c["requests"]]),
                design_freeze_sha256=m.FREEZE_SHA,
                config_sha256=m.CONFIG_SHA,
                source_lock_sha256=c["freeze"]["artifacts"]["source_lock"]["sha256"],
                manifest_sha256=c["freeze"]["artifacts"]["main_manifest"]["sha256"],
                execution_sources_sha256=c["sources_sha256"],
                checkpoint=c["checkpoint"],
            )
            tx.write_once(directory / "ENDPOINT_BINDING.json", binding)
            # The loader internally calls the original nonblocking global
            # TransactionStore inspection. Coordinate only this load phase.
            with file_mutex(path) if c["store"] is not None else nullcontext():
                handle = m.runtime.build_evaluation_runtime(
                    c["registry"],
                    m.bound(c["freeze"]["artifacts"]["source_lock"]),
                    start_seed=seed,
                    arm=arm,
                    step=step,
                    scope="MAIN",
                    output_dir=output,
                    authority=authority,
                    authority_source_reference=c["sources"]["authority"],
                    execution_sources=c["sources"],
                    expected_execution_sources_sha256=c["sources_sha256"],
                    commit_store=c["store"],
                    expected_identity=c["identity"],
                    expected_commit_sha256=None if c["commit"] is None else c["commit"]["sha256"],
                )
            require(handle.endpoint() == c["endpoint"], "Actual MAIN model endpoint differs")
            bridge = m.backend.GenerationBridge(c["registry"], output, handle, handle.endpoint)
            complete = []
            raw_path = output / "RAW.jsonl"
            descriptor = os.open(
                raw_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
            )
            with os.fdopen(descriptor, "wb") as raw:
                for manifest in c["manifest"]:
                    value = bridge.generate_one(
                        manifest["request_id"], terminal=m.read(m.bound(c["terminal"].receipt))
                    )
                    row = dict(
                        value,
                        **{k: v for k, v in manifest.items() if k != "request_id"},
                        scope="MAIN",
                        suffix_text=value["generated_text"],
                    )
                    raw.write(tx.canonical(row) + b"\n")
                    raw.flush()
                    os.fsync(raw.fileno())
                    complete.append(
                        m.ref(
                            output
                            / (
                                hashlib.sha256(manifest["request_id"].encode()).hexdigest()
                                + ".COMPLETE.json"
                            )
                        )
                    )
            after = handle.endpoint()
            closed = handle.close()
            handle = None
            require(
                after == c["endpoint"]
                and closed["attempted_requests"] == c["count"]
                and all(
                    closed[k] == 0
                    for k in ("optimizer_steps", "scheduler_steps", "parameter_updates")
                ),
                "MAIN changed policy/count",
            )
            authority.boundary()
            events = tx.rows(output / "accounting.jsonl")
            require(len(events) == 2 * c["count"], "MAIN accounting count differs")
            for i, manifest in enumerate(c["manifest"]):
                generated = m.read(m.bound(complete[i]))
                for event, kind, count in zip(
                    events[2 * i : 2 * i + 2],
                    ("completion_attempts", "generated_tokens"),
                    (1, len(generated["tokens"])),
                    strict=True,
                ):
                    require(
                        event["kind"] == kind
                        and event["count"] == count
                        and event["metadata"]["group_seed"] == manifest["seed"],
                        "Actual MAIN accounting differs",
                    )
            tx.write_once(directory / "CLOSE.json", closed)
            receipt = dict(
                schema="sr-satdose-evaluation-completion-v1",
                status="EVALUATION_ENDPOINT_CAPTURE_COMPLETE_NO_TRAINING_PERMISSION",
                scope="MAIN",
                endpoint_binding_sha256=m.sha(directory / "ENDPOINT_BINDING.json"),
                endpoint_before=c["endpoint"],
                endpoint_after=after,
                responses=c["count"],
                optimizer_created=False,
                scheduler_created=False,
                optimizer_steps=0,
                scheduler_steps=0,
                parameter_updates=0,
                extra_forward_sequences=0,
                raw=m.ref(raw_path),
                accounting=m.ref(output / "accounting.jsonl"),
                complete_files=complete,
            )
            tx.write_once(output / "RECEIPT.json", receipt)
            result = dict(
                schema="sr-satdose-main-result-v1",
                status="MAIN_ENDPOINT_COMPLETE",
                permission="NO_TRAINING_OR_SUCCESSOR_EXPERIMENT",
                start_seed=seed,
                arm=arm,
                step=step,
                responses=c["count"],
                design_freeze_sha256=m.FREEZE_SHA,
                release=c["release_ref"],
                terminal=c["terminal"].receipt,
                process_identity=c["process"],
                checkpoint=c["checkpoint"],
                receipt=m.ref(output / "RECEIPT.json"),
                endpoint_binding=m.ref(directory / "ENDPOINT_BINDING.json"),
                close=m.ref(directory / "CLOSE.json"),
                pair=c["pair"],
                released=c["released"],
                job_id=c["job_id"],
            )
            tx.write_once(directory / "RESULT.json", result)
            return result
        except BaseException as exc:
            if handle is not None:
                with suppress(Exception):
                    handle.close()
            tx.write_once(
                directory / "HOLD.json",
                dict(
                    status="HOLD_MAIN_TECHNICAL",
                    permission="NO_GENERATION_OR_TRAINING",
                    start_seed=seed,
                    arm=arm,
                    step=step,
                    error_type=type(exc).__name__,
                    error=str(exc),
                    recovery="AUDIT_EXISTING_ATTEMPT_NO_RESAMPLING",
                ),
            )
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-receipt", type=Path, required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--start-seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--arm", choices=("START", "T0", "T1"), required=True)
    parser.add_argument("--step", type=int, choices=(0, 8, 16, 24), required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    call = (args.release_receipt, args.release_sha256, args.start_seed, args.arm, args.step)
    try:
        if args.execute:
            value = execute(*call)
        else:
            c = preflight(*call)
            value = dict(
                status="MAIN_PARALLEL_PREFLIGHT_COMPLETE",
                start_seed=args.start_seed,
                arm=args.arm,
                step=args.step,
                responses=c["count"],
                previous_results=c["previous_results"],
                permission="NO_MODEL_CALLS",
            )
    except Exception as exc:
        print(
            json.dumps(
                dict(status="HOLD_MAIN_PARALLEL", error_type=type(exc).__name__, error=str(exc))
            )
        )
        return 2
    print(json.dumps(value, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
