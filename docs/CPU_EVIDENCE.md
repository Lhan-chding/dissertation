# CPU evidence

Only compact summaries and reproducibility code are versioned. Generated image
trees, per-trajectory tables, activations, and checkpoints are excluded.

## Theory and finite policy

- Seven legacy property families, 7,000 finite random checks, maximum absolute
  error approximately `5.17e-11`.
- Exact distribution optimization, REINFORCE, PPO-like, and GRPO-like fixed
  reasoner experiments use 20 seeds.
- Matched-mean reasoner scaling changes severity by approximately `-0.0477`,
  `0`, and `+0.1249` for truth-, uniform-, and error-directed gain.
- Tabular coordination, differentiable two-sigmoid, and PIL CNN-to-MLP
  mechanism experiments remain reproducible from `scripts/` and `configs/`.

## Natural versus synthetic replay

The confirmatory CPU run used 1,000 semantic states, 8,000 images, 640,000
natural mediators, 20,480,000 fork continuations, 20,000 synthetic mediators,
20 seeds, and five error families.

`c_fork` was approximately `0.957--0.960`; `c_syn` was approximately
`0.885--0.888`. The paired `c_syn-c_fork` interval excluded zero in all five
families, with means approximately `-0.071` to `-0.073`.

This is controlled small-neural evidence that injected and naturally produced
states are not interchangeable. It is not a Qwen result.

The publishable summaries are indexed in `experiments/cpu/summaries/README.md`.
