# RTX 4090 GPU Pilot protocol

The pilot is intentionally smaller than the later A100/H100 multi-interface
study. It runs only after preflight, offline model smoke, data generation, and
base calibration.

The active dataset is `CVA-Chart-Pilot-v0.2`. It retains the frozen v0.1
tasks, splits, values, and audit prefix, but replaces printed point values with
a visible integer y-axis scale. The legacy `configs/data/cva_chart_pilot.yaml`
and v0.1 renderer remain available for byte-level replay of the first failed
calibration; new collection and training accept only v0.2.

## Pilot A: fixed natural mediator

The image is removed. A naturally generated erroneous evidence mediator is
held fixed and language LoRA is trained with outcome-only reward. This tests
whether downstream reasoning can systematically compensate for operational
natural evidence errors. It does not test whether RL changes the model's visual
error distribution. Only validated `visual_error` and
`compensated_visual_error` first responses enter this treatment; parse failures
and reasoning-only errors remain in the audit but are not treated as fixed
visual mediators. The mediator is canonical parsed JSON, not raw model text,
and the prompt remains a normal text-only Qwen chat message.

## Pilot B: online image, LM-only LoRA

The image remains online while the vision tower, visual merger/projector, and
base language weights are frozen. Only language LoRA is trainable. This can
support claims about operational readout, reasoning, and compensation, not
visual acquisition improvement.

## Calibration gates

Training is prohibited unless calibration records:

- base answer accuracy in `30--75%`;
- natural perception error in `15--50%`;
- structured evidence parse rate at least `95%`;
- at least three natural error families with at least ten observations each.

If a gate fails, adjust the renderer/task difficulty and rerun calibration. Do
not start training merely because the model loads. On a reviewed rerun, the
collector moves the complete failed attempt into a unique
`trajectories/natural/attempts/failed-*` directory before publishing the new
attempt. It never overwrites or archives an accepted calibration.

The smoke and natural collector preserve the strict three-tag parser. The
smoke permits at most two deterministic, format-only retries. Its retry prompt
never quotes the prior model output, and every raw attempt is retained in the
local evidence record. The natural collector never resamples a parse failure:
its first response remains the canonical natural trajectory, preventing
parse-conditioned selection from changing the measured error distribution.
Exhausting the smoke retries makes it exit nonzero; partial JSON is never
reinterpreted as a valid trajectory. A valid pilot trajectory must have exactly
one integer `values` array of the expected length, exactly the requested
`operation`, and a non-boolean finite numeric answer. The smoke passes only
when this closed schema and the known-answer check both pass.

For every four-value pilot chart, perception must report A, B, C, and D in
that order even when the arithmetic question uses only A and B. Two-value
evidence is rejected rather than reinterpreted. The prompt states that `sum`
means A+B and `difference` means A-B, and prohibits escaped line breaks inside
the perception JSON. These instructions improve format compliance without
loosening the parser or changing the natural-error taxonomy.

The training launchers do not trust summary booleans. Immediately before any
training import, they rerun the live hardware audit and known-answer smoke,
then bind the required local model bytes, stage/path/model/data configs,
dataset manifest, records, counterfactuals, image bundle, smoke report, and
calibration trajectory by SHA-256. They regenerate all 2,950 PNGs from the
committed seed and renderer for byte-exact comparison, strictly replay all
2,800 records and 200 calibration responses, and recompute every calibration
gate. Natural-collection summaries bind both the model snapshot and canonical
dataset before and after collection. Pilot A additionally replays all 1,200
first-response natural trajectories.

Training also requires a clean Git commit and the exact registered stage
budget, paths, freeze policy, and claim boundary. Each completed stage keeps
stage-local copies of its volatile authorization reports, raw completions and
rewards in `rollouts.jsonl`, trainer history in `metrics.jsonl`, timestamps,
package/environment provenance, and SHA-256 hashes for the final adapter and
evidence files. The current analysis entrypoint nevertheless remains
unconditionally blocked: semantic replay and authenticated cross-stage
comparison are post-GPU work, so neither file existence nor self-hashes permit
a scientific claim yet.

## Ordered entrypoints

```text
00_preflight.py
01_smoke_qwen.py
02_generate_pilot_data.py
03_base_calibration.py
04_collect_natural.py
05_pilot_a.py
06_pilot_b.py
07_analyze.py
```

Pilot A/B require both `--execute` and the exact
`COMPBIAS_GPU_EXECUTION_ACK` value documented in the server runbook. This
double gate prevents a config parse or repository clone from starting training.

The launcher enforces code, model, data, trajectory, live-hardware, and
known-answer inference gates. The acknowledgement is also the operator's
manual assertion that the reviewed GPU lock and vulnerability review are
complete. Those external supply-chain approvals are not
authenticated by this repository and must not be described as automated or
third-party authorization.
