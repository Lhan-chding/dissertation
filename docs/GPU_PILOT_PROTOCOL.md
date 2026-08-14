# RTX 4090 GPU Pilot protocol

The pilot is intentionally smaller than the later A100/H100 multi-interface
study. It runs only after preflight, offline model smoke, data generation, and
base calibration.

## Pilot A: fixed natural mediator

The image is removed. A naturally generated erroneous evidence mediator is
held fixed and language LoRA is trained with outcome-only reward. This tests
whether downstream reasoning can systematically compensate for operational
natural evidence errors. It does not test whether RL changes the model's visual
error distribution.

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
not start training merely because the model loads.

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
