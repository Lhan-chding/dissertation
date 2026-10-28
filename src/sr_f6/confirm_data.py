"""Independent, model-blind DEV_SCHEMA_CONFIRM data, never old census data.

Only the registered constructor may reject candidate worlds. Historical
collisions stop this build; they never trigger another seed or another bank.
CPU mathematical checks do not certify visual readability or model behavior.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

from sr_f4 import task_contract as native
from srf6_ref import design
from srf6_ref.core import FAMILIES, enc, fingerprint, heldout_tags, oracle_a, oracle_b

from .common import digest, read, safe_evidence
from .data import _write, all_tasks, canonical, render_source
from .interface_v3_protocol import VERSION as PROMPT_VERSION
from .interface_v3_protocol import WIRE_PROTOCOL, public_prompt
from .measurement import score_response
from .source_binding import inventory_file, verify_bound_files

VERSION = "srf6-confirm-data-v1"
SPLIT = "DEV_SCHEMA_CONFIRM"
MASTER_SEED = 20261028
ROOTS_PER_FAMILY = 8
FACTORS = ("P0", "P1")
IDENTITY_FIELDS = ("qid", "root_id", "split", "family", "factor", "variant")


def _canonical_root(root):
    path = Path(root)
    if not path.is_absolute() or path.resolve(strict=True) != path or not path.is_dir():
        raise PermissionError("An existing canonical run directory is required")
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise PermissionError("Symlinked data roots are forbidden")
    return path


def _rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _audit_roots(roots):
    expected = {f"{SPLIT}_{family}_{i:04d}" for family in FAMILIES for i in range(8)}
    _require(
        len(roots) == 48 and {r["root_id"] for r in roots} == expected, "Exact 48 roots required"
    )
    seen, world_roots, summaries = set(), {}, []
    oracle_checks = native_checks = 0
    for root in roots:
        rid, cert = root["root_id"], root["certificate"]
        _require(set(root["tasks"]) == set(FACTORS), "Both factors must be retained")
        anchor = root["tasks"]["P0"][0]
        world, query, family = anchor["world"], anchor["query"], anchor["family"]
        index = int(rid.rsplit("_", 1)[1])
        forced = index % 3 == 0
        _require(
            query == design.base_query(family, index, len(world["categories"]))
            and cert["root_id"] == rid
            and cert["engineered_collision_anchor"] is forced,
            "Registered public query or forced-anchor identity differs",
        )
        _require(
            [t["edit_descriptor"] for t in root["tasks"]["P0"]]
            == [t["edit_descriptor"] for t in root["tasks"]["P1"]]
            and root["tasks"]["P1"][0]["world"] == world,
            "Paired anchor or edit-cost bins differ",
        )
        pool = design.edit_pool(world, query, False, family)
        edits = {edit["edit_id"]: edit for edit in pool}
        rules = design.wrong_rules(query)
        pool_mask = 0
        for edit in pool:
            pool_mask |= edit["response_mask"]
        active = [h for i, h in enumerate(rules) if (pool_mask >> i) & 1]
        aliases = [
            h
            for h in active
            if design.wrong_value(world, query, h) == oracle_a(world, query)["answer"]
        ]
        _require(bool(active), "No active finite-library candidate")
        _require(
            cert["rules"] == rules
            and cert["active_rules"] == active
            and cert["uncovered_by_pool"] == [h for h in rules if h not in active]
            and cert["base_alias_rules"] == aliases
            and cert["pool_size"] == len(pool)
            and (not forced or "boundary_swap" in aliases),
            "Finite candidate inventory or collision witness differs",
        )
        masks = {}
        for factor, tasks in root["tasks"].items():
            _require(
                len(tasks) == 4
                and [t["variant"] for t in tasks] == list(range(4))
                and len({digest(t["world"]) for t in tasks}) == 4,
                "Every root/factor requires four distinct ordered views",
            )
            mask = 0
            for task in tasks:
                qid, current = task["qid"], task["world"]
                _require(
                    qid == f"{rid}_{factor}_v{task['variant']}"
                    and qid not in seen
                    and task["root_id"] == rid
                    and task["split"] == SPLIT
                    and task["family"] == family
                    and task["factor"] == factor
                    and task["query"] == query
                    and current["categories"] == world["categories"]
                    and task["engineered_collision_anchor"] is forced,
                    "Question lineage, public query or category order differs",
                )
                seen.add(qid)
                _require(
                    all(v in design.GRID for col in current["series"].values() for v in col)
                    and design.valid_task(current, query, False, family)
                    and task["heldout_tags"] == heldout_tags(current, query) == [],
                    "Original grid, validity or global holdout rule failed",
                )
                previous = world_roots.setdefault(fingerprint(current), rid)
                _require(
                    previous == rid, "Renamed/reordered world repeats across confirmation roots"
                )
                gold = oracle_a(current, query)
                _require(gold == oracle_b(current, query), "Independent oracle disagreement")
                oracle_checks += 1
                _require(
                    native.execute(current, query) == gold["answer"]
                    and native.independently_answer(current, query) == gold["answer"]
                    and set(native.selected(current, query)) == set(gold["indices"]),
                    "Native task oracle disagreement",
                )
                _require(
                    task["gold_answer"] == enc(gold["answer"])
                    and task["gold_selected"] == gold["selected"]
                    and task["gold_bindings"]
                    == dict(zip(gold["selected"], map(enc, gold["operands"]), strict=True))
                    and task["answer_delta"]
                    == enc(gold["answer"] - oracle_a(world, query)["answer"]),
                    "Stored gold or exact answer delta differs",
                )
                # A mathematical full-fact witness tests the unchanged strict
                # native scorer. It is never a sampled answer or visual PASS.
                witness = dict(
                    evidence=[
                        dict(series=s, category=c, value=current["series"][s][i])
                        for s in ("Alpha", "Beta")
                        for i, c in enumerate(current["categories"])
                    ],
                    selected=gold["selected"],
                    bindings=[
                        dict(category=c, operand=enc(v))
                        for c, v in zip(gold["selected"], gold["operands"], strict=True)
                    ],
                    aggregate=query["aggregate"],
                    answer=enc(gold["answer"]),
                )
                score = score_response(canonical(witness), current, query, finish_reason="balanced")
                _require(
                    all(score[k] == 1 for k in ("V", "A", "J", "S", "B", "P")),
                    "Native strict witness regression failed",
                )
                native_checks += 1
                if task["variant"]:
                    edit = edits.get(task["edit_id"])
                    _require(
                        edit is not None
                        and edit["world"] == current
                        and list(edit["descriptor"]) == list(task["edit_descriptor"]),
                        "Actual edit or cost descriptor differs from the original finite pool",
                    )
                    mask |= edit["response_mask"]
                else:
                    _require(
                        task["edit_id"] == "anchor" and task["edit_descriptor"] is None,
                        "Anchor edit differs",
                    )
            masks[factor] = mask
            _require(
                abs(cert[f"coverage_{factor}"] - mask.bit_count() / len(active)) < 1e-12
                and abs(cert[f"coverage_all_registered_{factor}"] - mask.bit_count() / len(rules))
                < 1e-12,
                "Finite-library coverage differs from actual selected views",
            )
        descriptors = [t["edit_descriptor"] for t in root["tasks"]["P0"]][1:]
        _require(
            canonical(cert["descriptors"]) == canonical(descriptors),
            "Registered matched descriptors differ",
        )
        bins = [[e for e in pool if list(e["descriptor"]) == list(d)] for d in descriptors]
        best = max(
            (a["response_mask"] | b["response_mask"] | c["response_mask"]).bit_count()
            for a, b, c in itertools.product(*bins)
        )
        _require(
            masks["P1"].bit_count() == best, "P1 is not the original exact matched-bin maximizer"
        )
        _require(
            cert["blind_P1"]
            == [
                h
                for i, h in enumerate(rules)
                if (pool_mask >> i) & 1 and not (masks["P1"] >> i) & 1
            ],
            "Finite-library blind inventory differs",
        )
        summaries.append(
            dict(
                root_id=rid,
                family=family,
                engineered_collision_anchor=forced,
                active_candidates=len(active),
                coverage_P0=cert["coverage_P0"],
                coverage_P1=cert["coverage_P1"],
                base_alias_candidates=len(aliases),
                blind_P1=len(cert["blind_P1"]),
                constructor_attempt=cert["attempt"],
                constructor_rejections=len(cert["rejections"]),
            )
        )
    _require(
        sum(r["engineered_collision_anchor"] for r in summaries) == 18,
        "Exactly 18 forced anchors required",
    )
    groups = {}
    for label, subset in (
        ("all", summaries),
        ("forced", [r for r in summaries if r["engineered_collision_anchor"]]),
        ("nonforced", [r for r in summaries if not r["engineered_collision_anchor"]]),
    ):
        gaps = [r["coverage_P1"] - r["coverage_P0"] for r in subset]
        groups[label] = dict(
            roots=len(subset),
            mean_coverage_gap=sum(gaps) / len(gaps),
            strict_gain_roots=sum(g > 1e-12 for g in gaps),
        )
    return dict(
        status="PASS",
        scope="CPU_MATHEMATICAL_ONLY",
        split=SPLIT,
        master_seed=MASTER_SEED,
        roots=48,
        questions=384,
        forced_anchors=18,
        family_questions=dict(Counter(t["family"] for t in all_tasks(roots))),
        dual_oracle_comparisons=oracle_checks,
        native_oracle_comparisons=oracle_checks,
        native_full_fact_regressions=native_checks,
        cost_matching="PASS",
        finite_candidate_audit=dict(
            scope="DECLARED_FINITE_LIBRARY_ONLY",
            model_error_coverage="NOT_RUN",
            G8_credit="NOT_RUN",
            unseen_combination_gain="NOT_RUN",
            strata=groups,
            roots=summaries,
        ),
        model_calls=0,
        optimizer_updates=0,
        source_visual_review="NOT_RUN",
        processor_readability="NOT_RUN",
    )


def _bind_pin(path, sha256, *, boundary, expected_bytes=None):
    path = Path(path)
    if path.resolve(strict=True) != path or any(p.is_symlink() for p in (path, *path.parents)):
        raise PermissionError("Historical world source must be canonical and not symlinked")
    bound = dict(
        path=str(path),
        resolved_path=str(path),
        bytes=path.stat().st_size if expected_bytes is None else expected_bytes,
        sha256=sha256,
        symlink=False,
    )
    checked = verify_bound_files([bound], boundary=boundary)
    bound["stat_cache_key"] = checked["files"][0]["stat_cache_key"]
    return bound


def _historical(roots, historical_roots):
    _require(
        isinstance(historical_roots, list) and len(historical_roots) == 4,
        "Explicit original producer plus three historical roots required",
    )
    previous = [_canonical_root(p) for p in historical_roots]
    _require(
        len(set(previous)) == 4 and len({p.parent for p in previous}) == 1,
        "Four distinct historical project siblings required",
    )
    boundary, original = previous[0].parent, previous[0]
    identity_path = safe_evidence(original, "IDENTITY.json")
    identity = read(identity_path)
    ref = identity["data_manifest"]
    _require(
        ref["path"] == "DATA/MANIFEST.json",
        "Original identity must bind the original DATA manifest",
    )
    manifest_path = safe_evidence(original, ref["path"])
    bindings = [
        inventory_file(identity_path, boundary),
        _bind_pin(manifest_path, ref["sha256"], boundary=boundary),
    ]
    manifest = read(manifest_path)
    members = manifest["files"]
    _require(
        len(members) == len({r["path"] for r in members}), "Duplicate original data manifest member"
    )
    source_refs = [
        (original, item)
        for item in members
        if len(Path(item["path"]).parts) == 3
        and Path(item["path"]).parts[0] == "DATA"
        and Path(item["path"]).name == "TASKS_PRIVATE.jsonl"
    ]
    # All original task panels, including tests, are used only as world
    # exclusions. No historical query, label or sampled answer is returned.
    _require(bool(source_refs), "Original manifest has no task world sources")
    history_path = safe_evidence(original, "PREFLIGHT/HISTORICAL_WORLD_AUDIT.json")
    history = read(history_path)
    bindings.append(inventory_file(history_path, boundary))
    _require(
        history.get("status") == "PASS", "Original historical source inventory is not verified"
    )
    expected = [
        previous[1] / "DATA/TASKS_PRIVATE.jsonl",
        previous[2] / "DATA/TASKS.jsonl",
        previous[3] / "DATA/TASKS.jsonl",
    ]
    old = history["sources"]
    _require(
        len(old) == 3 and {r["path"] for r in old} == set(map(str, expected)),
        "Explicit historical roots differ from the original bound inventory",
    )
    for item in old:
        path = Path(item["path"])
        base = path.parent.parent
        source_refs.append((base, dict(path=str(path.relative_to(base)), sha256=item["sha256"])))
    shapes, sources = set(), []
    for base, item in source_refs:
        path = safe_evidence(base, item["path"])
        bound = _bind_pin(path, item["sha256"], boundary=boundary, expected_bytes=item.get("bytes"))
        bindings.append(bound)
        count = external = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                world = row.get("world")
                if world is None:
                    _require(
                        row.get("pool") == "CHARTQA", "Unclassified historical row without world"
                    )
                    external += 1
                    continue
                _require(
                    isinstance(world.get("categories"), list)
                    and all(
                        len(world["series"][s]) == len(world["categories"])
                        for s in ("Alpha", "Beta")
                    ),
                    "Malformed historical world",
                )
                shapes.add(fingerprint(world))
                count += 1
        _require(count > 0, "Empty historical synthetic world source")
        verify_bound_files([bound], boundary=boundary)
        sources.append(
            dict(
                path=str(path),
                sha256=bound["sha256"],
                world_rows=count,
                chartqa_rows_without_world=external,
            )
        )
    collisions = [t["qid"] for t in all_tasks(roots) if fingerprint(t["world"]) in shapes]
    return dict(
        status="FAIL" if collisions else "PASS",
        comparison="rename/order invariant Alpha/Beta fact multiset",
        checked_question_rows=384,
        unique_historical_worlds=len(shapes),
        collisions=collisions,
        sources=sources,
        bound_files=bindings,
        boundary=str(boundary),
        historical_labels_returned=False,
        reseeding_permitted=False,
    )


def build(root, *, historical_roots: list[Path]) -> dict:
    """Create only a fresh DATA subtree; the caller fixes historical root identities."""
    root = _canonical_root(root)
    _require(
        all(
            root != Path(p)
            and not root.is_relative_to(Path(p))
            and not Path(p).is_relative_to(root)
            for p in historical_roots
        ),
        "New and historical run roots must be separate",
    )
    destination = root / "DATA"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Preserve existing DATA, including partial or failed builds")
    roots = design.build_bank(SPLIT, ROOTS_PER_FAMILY, MASTER_SEED)
    audit = _audit_roots(roots)
    historical = _historical(roots, historical_roots)
    destination.mkdir()
    audit["historical_world_exclusion"] = historical
    if historical["status"] != "PASS":
        audit["status"] = "HISTORICAL_COLLISION_STOP"
        _write(destination / "AUDIT.json", audit)
        _write(destination / "CPU_AUDIT.json", audit)
        _write(destination / "ROOTS.jsonl", roots, rows=True)
        raise RuntimeError(
            "Historical world collision; preserve fixed bank and stop without reseeding"
        )
    tasks = all_tasks(roots)
    images, geometry, public = {}, [], []
    for task in tasks:
        pin = digest(task["world"])
        if pin not in images:
            payload, metrics = render_source(task["world"])
            image_file = f"DATA/images/{pin}.png"
            path = root / image_file
            path.parent.mkdir(exist_ok=True)
            with path.open("xb") as stream:
                stream.write(payload)
            images[pin] = dict(
                image_file=image_file, image_sha256=hashlib.sha256(payload).hexdigest()
            )
            geometry.append(dict(world_sha256=pin, **images[pin], **metrics))
        prompt = public_prompt(list(task["world"]["categories"]), dict(task["query"]))
        public.append(
            dict(
                **{k: task[k] for k in IDENTITY_FIELDS},
                protocol=WIRE_PROTOCOL,
                prompt_version=PROMPT_VERSION,
                model_input=dict(prompt=prompt, **images[pin]),
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                source_image_status="GENERATED",
            )
        )
    _write(destination / SPLIT / "TASKS_PRIVATE.jsonl", tasks, rows=True)
    _write(destination / SPLIT / "MODEL_INPUTS.jsonl", public, rows=True)
    _write(destination / "MODEL_INPUTS.jsonl", public, rows=True)
    for name in ("ROOTS.jsonl", "ROOT_CERTIFICATES_PRIVATE.jsonl"):
        _write(destination / name, roots, rows=True)
    _write(destination / "RENDER_GEOMETRY.jsonl", geometry, rows=True)
    review = [
        dict(
            qid=t["qid"],
            family=t["family"],
            **images[digest(t["world"])],
            processor_image=None,
            processor_routing=None,
            visual_review="NOT_RUN",
        )
        for t in tasks
        if t["factor"] == "P1" and t["variant"] == 0
    ]
    _write(
        destination / "PROCESSOR_REVIEW_PLAN.json",
        dict(
            items=review,
            count=48,
            selection=(
                "All eight new DEV_SCHEMA_CONFIRM P1 anchors per family; "
                "deterministic and model-blind"
            ),
            source_visual_review="NOT_RUN",
            processor_visual_review="NOT_RUN",
            historical_visual_review_reused=False,
        ),
    )
    for name in ("AUDIT.json", "CPU_AUDIT.json"):
        _write(destination / name, audit)
    files = []
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            bound = inventory_file(path, root)
            files.append(dict(bound, path=str(path.relative_to(root))))
    manifest = dict(
        schema="srf6-confirm-data-manifest-v1",
        data_version=VERSION,
        split=SPLIT,
        master_seed=MASTER_SEED,
        roots=48,
        questions=384,
        prompt_version=PROMPT_VERSION,
        files=files,
    )
    _write(destination / "MANIFEST.json", manifest)
    manifest_pin = inventory_file(destination / "MANIFEST.json", root)["sha256"]
    receipt = dict(
        status="CPU_PREPARED_NOT_RELEASED",
        data_version=VERSION,
        split=SPLIT,
        master_seed=MASTER_SEED,
        question_rows=384,
        roots=48,
        forced_anchors=18,
        unique_source_images=len(images),
        data_manifest=dict(path="DATA/MANIFEST.json", sha256=manifest_pin),
        data_manifest_sha256=manifest_pin,
        manifest_sha256=manifest_pin,
        cpu_audit=audit,
        historical_sources=historical["bound_files"],
        source_images="GENERATED",
        source_visual_review="NOT_RUN",
        processor_readability="NOT_RUN",
        model_calls=0,
        optimizer_updates=0,
    )
    _write(destination / "BUILD_RECEIPT.json", receipt)
    return receipt


def load_inputs(root):
    """Load the exact 384 new questions, with public/private linkage checked."""
    root = _canonical_root(root)
    public = _rows(safe_evidence(root, f"DATA/{SPLIT}/MODEL_INPUTS.jsonl"))
    tasks = _rows(safe_evidence(root, f"DATA/{SPLIT}/TASKS_PRIVATE.jsonl"))
    private = {r["qid"]: r for r in tasks}
    expected = {
        f"{SPLIT}_{f}_{i:04d}_{p}_v{v}"
        for f in FAMILIES
        for i in range(8)
        for p in FACTORS
        for v in range(4)
    }
    _require(
        len(public) == len(tasks) == len(private) == 384
        and {r["qid"] for r in public} == set(private) == expected,
        "Exact 384 distinct confirmation questions required",
    )
    for row in public:
        task = private[row["qid"]]
        prompt = public_prompt(task["world"]["categories"], task["query"])
        _require(
            all(row[k] == task[k] for k in IDENTITY_FIELDS)
            and row["split"] == SPLIT
            and row["protocol"] == WIRE_PROTOCOL
            and row["prompt_version"] == PROMPT_VERSION
            and row["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest(),
            "Confirmation public/private lineage or prompt differs",
        )
        model = row["model_input"]
        _require(
            set(model) == {"prompt", "image_file", "image_sha256"}
            and model["prompt"] == prompt
            and model["image_file"] == f"DATA/images/{digest(task['world'])}.png",
            "Bound confirmation model input differs",
        )
        safe_evidence(root, model["image_file"])
    _require(
        Counter((r["root_id"], r["factor"]) for r in public)
        == Counter(
            {(f"{SPLIT}_{f}_{i:04d}", p): 4 for f in FAMILIES for i in range(8) for p in FACTORS}
        ),
        "Four views per root/factor required",
    )
    return public, private


def verify_data(root):
    """Revalidate only new DATA and the already-bound historical world files."""
    root = _canonical_root(root)
    manifest = read(safe_evidence(root, "DATA/MANIFEST.json"))
    _require(
        manifest.get("schema") == "srf6-confirm-data-manifest-v1"
        and manifest.get("data_version") == VERSION
        and manifest.get("split") == SPLIT
        and manifest.get("master_seed") == MASTER_SEED
        and manifest.get("roots") == 48
        and manifest.get("questions") == 384
        and manifest.get("prompt_version") == PROMPT_VERSION,
        "Confirmation data manifest identity differs",
    )
    members = manifest["files"]
    _require(
        len(members) == len({r["path"] for r in members}), "Duplicate confirmation manifest member"
    )
    bounds = []
    for member in members:
        path = safe_evidence(root, member["path"])
        _require(Path(member["path"]).parts[0] == "DATA", "Manifest outside new DATA")
        bounds.append(dict(member, path=str(path)))
    wanted = {root / r["path"] for r in members} | {
        root / "DATA/MANIFEST.json",
        root / "DATA/BUILD_RECEIPT.json",
    }
    _require(
        {p for p in (root / "DATA").rglob("*") if p.is_file()} == wanted,
        "New DATA manifest coverage differs",
    )
    binding_check = verify_bound_files(bounds, boundary=root)
    public, private = load_inputs(root)
    roots = _rows(safe_evidence(root, "DATA/ROOTS.jsonl"))
    _require(
        _rows(safe_evidence(root, "DATA/ROOT_CERTIFICATES_PRIVATE.jsonl")) == roots
        and {t["qid"]: t for t in all_tasks(roots)} == private,
        "Root certificates differ from actual confirmation tasks",
    )
    audit = _audit_roots(roots)
    stored = read(safe_evidence(root, "DATA/AUDIT.json"))
    _require(
        all(stored.get(k) == v for k, v in audit.items())
        and read(safe_evidence(root, "DATA/CPU_AUDIT.json")) == stored,
        "Mathematical audit differs from actual confirmation data",
    )
    history = stored["historical_world_exclusion"]
    _require(
        history["status"] == "PASS"
        and history["collisions"] == []
        and history["checked_question_rows"] == 384,
        "Historical exclusion did not pass",
    )
    history_check = verify_bound_files(history["bound_files"], boundary=Path(history["boundary"]))
    pins = {r["path"]: r["sha256"] for r in members}
    _require(
        all(
            pins[r["model_input"]["image_file"]] == r["model_input"]["image_sha256"] for r in public
        ),
        "Image route differs from manifest",
    )
    return dict(
        status="PASS",
        scope="CPU_MATHEMATICAL_AND_BOUND_INPUTS_ONLY",
        split=SPLIT,
        questions=384,
        roots=48,
        data_binding=binding_check,
        historical_binding=history_check,
        model_calls=0,
        processor_readability="NOT_RUN",
    )
