# Research question

Does outcome-only multimodal reinforcement learning repair local visual and
reasoning errors, or can it obtain a correct final answer by selecting natural
visual errors that downstream reasoning compensates?

The project distinguishes three quantities:

- `c_sel`: success conditional on a naturally occurring error trajectory;
- `c_fork`: downstream success after replaying a naturally sampled mediator
  with image access removed;
- `c_syn`: downstream success after injecting a synthetic mediator.

The project does not assume a unique perception/reasoning boundary. It validates
candidate interfaces and reports an admissible multi-interface band.

## Current status

- CPU: controlled mechanism and v2 estimator software established.
- GPU Pilot: Qwen2.5-VL-3B-Instruct on an RTX 4090 with 47.37 GiB VRAM,
  pending server preflight and offline smoke.
- No Qwen training result is currently recorded.

## Explicit non-claims

Current evidence does not establish general VLM compensation, recover a unique
internal perception/reasoning decomposition, show improved visual acquisition,
or show that synthetic error represents natural error.
