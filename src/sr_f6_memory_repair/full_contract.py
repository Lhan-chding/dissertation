"""V7 full-validation provenance; metadata entry points import only the stdlib.

Historical tensors are not consumed by metadata. The complete original closure
and actual full-batch tensors are checked inside the GPU validation allocation.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from . import execution as legacy

SCHEMA = "srf6-memory-full-contract-v7"
RESULT_SCHEMA = "srf6-memory-full-validation-v7"
STATUS = "VERIFIED_ACTUAL_STORAGE_REPAIR_BATCH"
PROJECT = legacy.PROJECT
OPS = PROJECT / "sr_f6_operations_20261027/PILOT_MEMORY_REPAIR_V7_20261030"
V4_OPS = legacy.OPS
V6_OPS = PROJECT / "sr_f6_operations_20261027/PILOT_MEMORY_REPAIR_V6_20261030"
V4_SHA = "dfd8d792aed9fb2dc0d9a33ed4962afefb410609b974c92a34659efccc444a34"
V6_SHA = "ec1d146141f67794b0bb4c78a9a8164e5c67038044f738e3dad19f758f6b150c"
V6_RESULT_SHA = "0a5b59becd74a3286be5c16d2dcc19937c340a28b3e2c05bcdde801739b83aca"
V6_COMPLETE_SHA = "b58685ec52a75d75cd0b20cdd21830a2cfca957df4f15e373be9dde342422d03"
V6_COLD_SHA = "b16caac348f2efa731f53196a7606de8c888defc2dcc62210cf5ec438b6eafbd"
V6_REVISION = "1af637e0b4aaded4e444a4904091ec309927a9ff"
V6_BINDING = "95f4b5c31c4e42c35afb3734815c5b0328a765a8b934b81468982da8c87ab50a"
HOST_BYTES, DISK_BYTES, DEVICE_BYTES = 8 * 2**30, 96 * 2**30, 44 * 2**30
WINDOW = legacy.WINDOW
SOURCE_NAMES = (
    "__init__",
    "execution",
    "validation",
    "offload",
    "arena",
    "residency",
    "smoke",
    "smoke_reuse",
    "full_contract",
    "validation_full",
)
ADDITIONS = frozenset(
    (
        "prior_full_contract_ref",
        "prior_smoke_contract_ref",
        "prior_smoke_result_ref",
        "prior_smoke_complete_ref",
        "prior_smoke_cold_ref",
    )
)
MUTABLE = frozenset(("schema", "repair_source", "max_device_bytes"))
PLAN = dict(
    baseline_rows=[0],
    offload_rows=list(range(128)),
    archived_forward_rows=list(range(27)),
    failing_row=26,
    diagnostic_rows="original_registered_64",
    loss_denominator=128,
    atol=1e-6,
    rtol=1e-5,
    generated_responses=0,
    optimizer_updates=0,
    full_128_paired_gradient_equivalence=False,
)
require = legacy.require
strict_ref = legacy.strict_ref


def _same(left, right):
    """JSON identity distinguishes booleans, integers and floating-point values."""
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False
    )


def _read(ref):
    path = legacy.ref_path(PROJECT, ref)
    require(path.stat().st_size <= 16 * 1024**2, "Bounded metadata required")
    return legacy._json(strict_ref(path, ref["sha256"]))


def predecessor_refs():
    return dict(
        prior_full_contract_ref=dict(path=str(V4_OPS / "CONTRACT.json"), sha256=V4_SHA),
        prior_smoke_contract_ref=dict(path=str(V6_OPS / "CONTRACT.json"), sha256=V6_SHA),
        prior_smoke_result_ref=dict(path=str(V6_OPS / "SMOKE/RESULT.json"), sha256=V6_RESULT_SHA),
        prior_smoke_complete_ref=dict(
            path=str(V6_OPS / "smoke_COMPLETE.json"), sha256=V6_COMPLETE_SHA
        ),
        prior_smoke_cold_ref=dict(
            path=str(V6_OPS / "SMOKE/COLD_VERIFICATION.json"), sha256=V6_COLD_SHA
        ),
    )


def _prior_success(contract, prior):
    from . import smoke_reuse

    # This is the unchanged, stdlib-only metadata verifier, not a tensor replay
    # or a claim that the historical verifier was loaded from its old deployment.
    checked = smoke_reuse.verify_smoke_metadata(
        V6_OPS / "CONTRACT.json", V6_SHA, contract["prior_smoke_result_ref"]
    )
    require(_same(checked["contract"], prior), "V6 metadata contract differs")
    for key in smoke_reuse.INHERITED_KEYS:
        if key in contract:
            require(_same(contract[key], prior[key]), "Original v6 binding changed: " + key)
    require(
        prior["repair_source"]["revision"] == V6_REVISION
        and checked["result"]["cold_verification_ref"] == contract["prior_smoke_cold_ref"],
        "V6 source/cold proof changed",
    )
    complete = _read(contract["prior_smoke_complete_ref"])
    require(
        complete.get("status") == "COMPLETED_STAGED_SMOKE_NOT_FULL_BATCH_ADMISSION"
        and complete.get("job_id") == "235771"
        and complete.get("contract_ref") == contract["prior_smoke_contract_ref"]
        and complete.get("result_ref") == contract["prior_smoke_result_ref"]
        and complete.get("source")
        == dict(
            revision=V6_REVISION,
            deployment=prior["repair_source"]["deployment"],
            binding_sha256=V6_BINDING,
        ),
        "V6 actual completion binding differs",
    )
    proof = _read(contract["prior_smoke_cold_ref"])
    require(
        proof.get("status") == "VERIFIED_ACTUAL_TWO_ROW_REUSED_REFERENCE_SMOKE"
        and proof.get("admission") == "NOT_FULL_BATCH_ADMISSION",
        "V6 is only the completed two-row technical predecessor",
    )
    require(
        not any(
            (V6_OPS / name).exists()
            for name in (
                "smoke_FAILURE.json",
                "SMOKE/FAILURE.json",
                "WATCHDOG.json",
                "WATCHDOG_FAILURE.json",
                "smoke_WATCHDOG.json",
                "smoke_WATCHDOG_ATTEMPT.json",
                "smoke_WATCHDOG_RESULT.json",
            )
        ),
        "Preserve failed V6; no full successor",
    )


def _sources(contract, prior):
    repair = contract["repair_source"]
    revision = repair["revision"]
    require(
        isinstance(revision, str)
        and re.fullmatch(r"[0-9a-f]{40}", revision)
        and revision not in {V6_REVISION, legacy.PRODUCER_REVISION},
        "Separate actual committed full-validation source required",
    )
    deployment = legacy.plain_path(repair["deployment"])
    require(
        deployment != Path(prior["repair_source"]["deployment"])
        and deployment != legacy.PRODUCER_DEPLOYMENT,
        "Independent full-validation deployment required",
    )
    listed = {v["path"]: v for v in repair["files"]}
    expected = {"src/sr_f6_memory_repair/" + n + ".py" for n in SOURCE_NAMES} | {
        "scripts/sr_f6_memory_full_ops/operations.py",
        "scripts/sr_f6_memory_full_ops/launch.sbatch",
    }
    require(
        len(listed) == len(repair["files"]) and set(listed) == expected,
        "Exact full-validation source closure required",
    )
    for name, ref in listed.items():
        path = legacy.ref_path(deployment, ref)
        require(path == deployment / name, "Source path escaped deployment")
        strict_ref(path, ref["sha256"])
    require(
        Path(__file__).resolve() == deployment / "src/sr_f6_memory_repair/full_contract.py"
        and Path(legacy.__file__).resolve() == deployment / "src/sr_f6_memory_repair/execution.py",
        "Full contract helpers loaded from another deployment",
    )
    old_files = {v["path"]: v for v in prior["repair_source"]["files"]}
    old_deployment = legacy.plain_path(prior["repair_source"]["deployment"])
    for name in SOURCE_NAMES[:8]:
        key = "src/sr_f6_memory_repair/" + name + ".py"
        require(
            listed[key]["sha256"] == old_files[key]["sha256"],
            "Frozen historical dependency changed: " + name,
        )
        strict_ref(old_deployment / key, old_files[key]["sha256"])


def validate_metadata(path, expected_sha, *, source=True):
    path = legacy.plain_path(path)
    require(
        path in (OPS / "CONTRACT.json", OPS / "PREFLIGHT_CONTRACT.json"),
        "Unique V7 contract or metadata preflight required",
    )
    contract = _read(dict(path=str(path), sha256=expected_sha))
    require(contract.get("schema") == SCHEMA, "Full contract schema differs")
    for key, ref in predecessor_refs().items():
        require(contract.get(key) == ref, "Exact completed predecessor required: " + key)
    original = _read(contract["prior_full_contract_ref"])
    require(original.get("schema") == legacy.SCHEMA, "Original complete V4 contract required")
    require(set(contract) == set(original) | ADDITIONS, "Full contract fields differ")
    require(
        _same(
            {k: v for k, v in contract.items() if k not in MUTABLE | ADDITIONS},
            {k: v for k, v in original.items() if k not in MUTABLE},
        ),
        "Original full evidence/scientific registration changed",
    )
    require(
        all(
            type(contract.get(k)) is int and contract[k] == n
            for k, n in (
                ("max_host_bytes", HOST_BYTES),
                ("max_disk_bytes", DISK_BYTES),
                ("max_device_bytes", DEVICE_BYTES),
            )
        )
        and contract.get("storage_backend") == "arena_v3"
        and _same(contract.get("validation_window"), WINDOW),
        "Fixed full-validation storage/window required",
    )
    # These historical routines read retained failure/attempt/disposition JSON;
    # they do not read removed V2/V3 partial tensors or scratch.
    legacy._prior_allocation_failure(original)
    legacy.predecessor_evidence(original)
    prior = _read(contract["prior_smoke_contract_ref"])
    _prior_success(contract, prior)
    if source:
        _sources(contract, prior)
    return contract


def verify_contract(path, expected_sha):
    require(legacy.plain_path(path) == OPS / "CONTRACT.json", "Execution requires final contract")
    contract = validate_metadata(path, expected_sha)
    legacy._source(contract)
    legacy._original_evidence(contract)
    return contract


def verify_validation_metadata(path, expected_sha, result_ref):
    """Check bound receipts only, never claim a fresh tensor recomputation."""
    contract = validate_metadata(path, expected_sha)
    require(Path(path) == OPS / "CONTRACT.json", "Final full contract required")
    require(
        legacy.ref_path(PROJECT, result_ref) == OPS / "VALIDATION/RESULT.json",
        "Wrong full result path",
    )
    result = _read(result_ref)
    require(
        result.get("schema") == RESULT_SCHEMA
        and result.get("status") == STATUS
        and result.get("contract_sha256") == expected_sha
        and _same(result.get("plan"), PLAN)
        and result.get("prior_smoke_result_ref") == contract["prior_smoke_result_ref"],
        "Full result identity differs",
    )
    files = {v["path"]: v for v in contract["repair_source"]["files"]}
    key = "src/sr_f6_memory_repair/validation_full.py"
    require(
        result.get("verifier_source_ref")
        == dict(
            path=str(Path(contract["repair_source"]["deployment"]) / key),
            sha256=files[key]["sha256"],
        ),
        "Actual full verifier differs",
    )
    for field, name in (("evidence_ref", "EVIDENCE.json"), ("tensors_ref", "TENSORS.pt")):
        ref = result[field]
        target = legacy.ref_path(PROJECT, ref)
        require(target == OPS / "VALIDATION" / name and target.is_file(), "Full artifact absent")
        require(
            isinstance(ref["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]),
            "Invalid full artifact hash",
        )
    evidence = _read(result["evidence_ref"])
    require(
        evidence.get("scope") == "ACTUAL_CUDA_STORAGE_VALIDATION"
        and evidence.get("contract_sha256") == expected_sha
        and evidence.get("repair_revision") == contract["repair_source"]["revision"]
        and evidence.get("producer_revision") == contract["producer"]["revision"]
        and evidence.get("prior_smoke_result_ref") == contract["prior_smoke_result_ref"],
        "Full evidence identity differs",
    )
    cold_path = legacy.plain_path(OPS / "VALIDATION/COLD_VERIFICATION.json")
    require(
        cold_path.is_file() and cold_path.stat().st_size <= 16 * 1024**2,
        "Actual full cold verification absent",
    )
    cold_ref = dict(path=str(cold_path), sha256=hashlib.sha256(cold_path.read_bytes()).hexdigest())
    cold = _read(cold_ref)
    require(
        cold.get("schema") == RESULT_SCHEMA
        and cold.get("status") == STATUS
        and cold.get("contract_sha256") == expected_sha
        and cold.get("validation_ref") == result_ref
        and cold.get("verifier_source_ref") == result["verifier_source_ref"]
        and cold.get("repair_revision") == contract["repair_source"]["revision"]
        and cold.get("producer_revision") == contract["producer"]["revision"]
        and cold.get("offload_rows") == 128
        and cold.get("compared_gradient_rows") == [0]
        and cold.get("generated_responses") == cold.get("optimizer_updates") == 0
        and _same(cold.get("scope"), PLAN),
        "Actual full cold verification binding differs",
    )
    require(
        not (OPS / "VALIDATION/FAILURE.json").exists()
        and not (OPS / "validate_FAILURE.json").exists(),
        "Failed full validation preserved",
    )
    return dict(
        status=STATUS,
        contract=contract,
        result=result,
        validation_ref=result_ref,
        cold_verification_ref=cold_ref,
    )


verify_result_metadata = verify_validation_metadata
