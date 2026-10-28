"""Close the confirmation gates from verified facts, without granting training.

The receipt belongs to the new followup consumer and the actual 6e producer.
It cannot be supplied to the historical PILOT admission API. A future pilot
must register its own producer, v3 input overlay and runtime access contract.
No caller-provided PASS, selected receipt subset, or persisted cache authorizes
this assessment. The review and numerical consumers enumerate their attempts.
"""

from __future__ import annotations

import argparse
import copy
import json

from sr_f6.common import digest

from . import source

SCHEMA = "srf6-followup-confirmation-admission-v1"
STATUS = "CONFIRMATION_GATES_PASS"
DIRECTORY = "ADMISSION"
SCOPE = "CONFIRMATION_COMPLETE_REQUIRES_SEPARATE_PILOT_REGISTRATION"


def _require(condition, message):
    if not condition:
        raise PermissionError(message)


def _synthetic(value):
    if isinstance(value, dict):
        return any(
            (
                key in ("scope", "execution_origin", "status")
                and isinstance(item, str)
                and item.startswith("SYNTHETIC")
            )
            or _synthetic(item)
            for key, item in value.items()
        )
    return isinstance(value, (tuple, list)) and any(_synthetic(item) for item in value)


def _facts(root):
    # Imports stay lazy: source verifies the actual dual-package deployment
    # before any evidence can be promoted. No GPU or sampling entrypoint runs.
    from . import t03

    context = source.require_support(root, purpose="admission")
    review = context["review"]
    support = context["support_binding"]
    _require(
        support["status"] == "VERIFIED_SUPPORT_PASS_REVIEW_PENDING"
        and support["source_rows"] == 9216
        and len(context["records"]) == len(context["raw_by_row"]) == 9216,
        "Complete independently verified confirmation required",
    )
    _require(
        review["status"] == "REVIEW_COMPLETE_NOT_GATE_SIGNED",
        "Complete actual independent review required",
    )
    numerical = t03.verify_completed(root)
    _require(
        numerical["scope"] == "SERVER_ACTUAL",
        "Only actual server numerical evidence can close confirmation",
    )
    _require(
        set(numerical["parents"]) == {"b0", "b1", "b2"},
        "All three registered numerical parents required",
    )
    _require(
        not _synthetic((support, review, numerical)),
        "Synthetic fixtures cannot produce actual confirmation admission",
    )
    # Revalidate after reading downstream facts. The producer's private warm
    # cache reuses unchanged facts and checks the complete evidence membership.
    after = source.verify_registered(root, require_support=True)
    _require(
        after["consumer"] == context["consumer"]
        and after["support_binding_ref"] == context["support_binding_ref"],
        "Source or support binding changed during admission",
    )
    return context, review, numerical


def assess(root):
    """Read actual evidence; never run a model, write a receipt, or authorize."""
    try:
        context, review, numerical = _facts(root)
    except (OSError, ValueError, KeyError, TypeError, ImportError, PermissionError) as exc:
        return dict(
            schema=SCHEMA,
            status="UNKNOWN",
            scope=SCOPE,
            training_allowed=False,
            generation_allowed=False,
            evaluation_allowed=False,
            gate_status="NOT_SIGNED",
            error=type(exc).__name__ + ": " + str(exc),
        )
    states = [numerical["status"]] + [numerical["parents"][p]["status"] for p in ("b0", "b1", "b2")]
    status = (
        STATUS
        if all(state == "PASS" for state in states)
        else "FAIL"
        if "FAIL" in states
        else "INCONCLUSIVE"
        if "INCONCLUSIVE" in states
        else "UNKNOWN"
    )
    return dict(
        schema=SCHEMA,
        status=status,
        scope=SCOPE,
        execution_origin="ACTUAL_EVIDENCE_VALIDATION",
        consumer=copy.deepcopy(context["consumer"]),
        producer=copy.deepcopy(context["support_binding"]["producer"]),
        support_binding=copy.deepcopy(context["support_binding_ref"]),
        review_sha256=digest(review),
        numerical_sha256=digest(numerical),
        checks=dict(
            source_data_processor_visual="VERIFIED_BY_FROZEN_PRODUCER",
            complete_confirmation_support="PASS",
            independent_response_review=review["status"],
            numerical={p: numerical["parents"][p]["status"] for p in ("b0", "b1", "b2")},
        ),
        training_allowed=False,
        generation_allowed=False,
        evaluation_allowed=False,
        gate_status="NOT_SIGNED",
        new_model_calls=0,
        optimizer_updates=0,
        limitations=[
            "Confirmation exposure and measurement are not demonstrated learning benefit",
            "Independent reviewer identities are recorded, not cryptographically authenticated",
            "A new pilot input overlay, producer and runtime admission remain required",
        ],
    )


def seal(root):
    """Publish one immutable confirmation result, including unresolved outcomes."""
    context = source.verify_registered(root)
    root = context["root"]
    output = source._path(root / DIRECTORY, exists=False)
    _require(not output.exists(), "Preserve existing or partial admission; no automatic retry")
    output.mkdir(mode=0o700)
    source._write(
        output / "ATTEMPT.json",
        dict(schema=SCHEMA, status="STARTED", consumer=context["consumer"]),
    )
    assessment = assess(root)
    source._write(output / "ASSESSMENT.json", assessment)
    receipt = dict(
        schema=SCHEMA,
        status=assessment["status"],
        scope=SCOPE,
        assessment=source._reference(output / "ASSESSMENT.json"),
        consumer=context["consumer"],
        gate_status="CONFIRMATION_FACTS_SEALED" if assessment["status"] == STATUS else "NOT_SIGNED",
        training_allowed=False,
        generation_allowed=False,
        evaluation_allowed=False,
    )
    source._write(output / "RECEIPT.json", receipt)
    return receipt


def require_confirmation(root):
    """Validate the fixed receipt against freshly reconstructed current facts."""
    context = source.verify_registered(root)
    root = context["root"]
    directory = source._path(root / DIRECTORY)
    _require(
        {p.name for p in directory.iterdir()}
        == {"ATTEMPT.json", "ASSESSMENT.json", "RECEIPT.json"},
        "Incomplete or unregistered admission execution evidence",
    )
    attempt = source._read(source._path(directory / "ATTEMPT.json"))
    _require(
        attempt == dict(schema=SCHEMA, status="STARTED", consumer=context["consumer"]),
        "Admission attempt source differs",
    )
    receipt = source._read(source._path(directory / "RECEIPT.json"))
    expected_ref = source._reference(directory / "ASSESSMENT.json")
    _require(
        receipt
        == dict(
            schema=SCHEMA,
            status=STATUS,
            scope=SCOPE,
            assessment=expected_ref,
            consumer=context["consumer"],
            gate_status="CONFIRMATION_FACTS_SEALED",
            training_allowed=False,
            generation_allowed=False,
            evaluation_allowed=False,
        ),
        "Complete unchanged confirmation receipt required",
    )
    saved = source._bound(expected_ref)
    current = assess(root)
    _require(current["status"] == STATUS and saved == current, "Confirmation facts changed")
    return dict(receipt, receipt_ref=source._reference(directory / "RECEIPT.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("assess", "seal", "verify"))
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    result = {"assess": assess, "seal": seal, "verify": require_confirmation}[args.action](
        args.root
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
