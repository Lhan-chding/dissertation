# PILOT_V3 CPU preparation operations

These operator sources call the already frozen `7631725` production deployment.
They do not replace that deployment, its identity, or any scientific gate. Their
Git revision and exact program/input bytes are recorded separately at submission.

Private inputs and actual receipts remain in `.local/pilot_v3_20261029`. The
prospective source declaration covers the actual S0 manifest and the deployed
763 manifest, not a later checkout. Actual confirmation delivery is required;
missing evidence is never filled with a placeholder. No template is executable
as an activated submission.

1. Validate with `build_prepare_submission.py --validate-only`. This performs
   local syntax/source-declaration checks only.
2. After the existing three-parent T03 job and unique confirmation admission
   both complete, deliver their original evidence using the established
   confirmation delivery script. Preserve all failed and partial records.
3. Run `build_prepare_submission.py` with the project's existing Python 3.12
   environment. It requires committed operator sources, actual delivered
   `CONFIRMATION_GATES_PASS`, terminal prerequisite jobs, and bound receipt bytes.
   It creates a private request once and does not contact the server.
4. Run `run_remote.py submit_prepare_remote.py`. It checks actual upstream
   records, an absent new phase/attempt, and live teacher reservations/slots.
   It submits exactly one 0-GPU job and never retries an unknown submission.
5. The CPU job runs 103 existing packaged native regression cases, including
   shared-filesystem publish/recovery and interprocess lock cases. These are
   synthetic CPU checks, not numerical or scientific PASS. It then calls the
   frozen source registration, isolated historical confirmation, fixed public
   overlay, nine-path registration, pilot admission, and two-worker plan APIs.

The CPU job does not generate answers, update a model, or submit GPU work. A
failed, incomplete, or unknown preparation preserves its attempt and stops.
Only `READY_FOR_REGISTERED_PILOT_ONLY` allows the next separately guarded
operator to consider the fixed GPU pilot. FORMAL and TEST remain closed.

The production module inventory, native tests and scientific APIs remain those
in the frozen 763 archive. The new six local operator regression cases verify
request assembly refuses missing/non-PASS confirmation and unexplained source
changes; they are not part of a scientific acceptance criterion.
