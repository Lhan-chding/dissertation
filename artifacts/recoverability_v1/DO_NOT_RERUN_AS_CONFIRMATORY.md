# Frozen v0.3 negative pilot

`CVA-Chart-Pilot-v0.3` is a completed, failed preregistered calibration. Its
200-response summary is fixed in
`configs/recoverability/v0_3_negative_pilot.yaml`. The original Pilot A and
Pilot B were terminated without training.

Do not rerun v0.3 as a confirmatory attempt, change its parser or gates, or
select a preferred outcome from repeated calibrations. The server evidence
capture step only validates and hashes the existing records, summary, data log,
and calibration log. It performs no model call.

Recoverability v1 is a new preregistered protocol with a different claim and
must remain `PREREGISTERED_NOT_RUN` until the fixed server bridge begins.
