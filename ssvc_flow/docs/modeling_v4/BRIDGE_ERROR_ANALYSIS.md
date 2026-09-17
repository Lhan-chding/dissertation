# B-stage bridge error analysis

This post-hoc analysis reads the two completed B origins. It does not load model
weights, collect new samples, fit response predictors, modify measured results,
or advance the GPU campaign. Run it on an allocated server CPU. The running GPU
workers retain their original source snapshot.

## Inputs and outputs

Inputs are `tasks_initial.json`, each B task's `COMPLETE.json`, `ALL_FORKS.json`,
`PAIR_DIAGNOSTICS.json`, and the referenced shared sample and score chunks.
Only the nonalias primary contrast selected by the original bridge is analyzed
numerically. All four banks' alias and parameter-difference metadata remain in
the summary. Neither aliases nor repeated scores count as independent evidence.

Six independent sampling streams are retained separately: `work`, `reference`,
`direct_work`, endpoint mixture, and independent left/right endpoint counts.
The third origin packet `direct_work` is a likelihood-ratio direct measurement;
it is not the separate two-endpoint `COUNT` baseline.

The implementation checks complete per-prompt draw indices, sample/seed/namespace
disjointness, event partitions, proposal identities, score/input/token alignment,
recorded generation parity, prefix-scoring execution, and the 64-token horizon.
Recorded chunk hashes are indexed without repeatedly hashing all original data.
Chunk counts and actual read sizes are checked. The original diagnostic means
and variance estimates must be reproduced within `1e-12`.

Outputs:

- `EVENT_COUNTS.csv`: X/S/W/I counts, EOS counts and lengths by sampling packet.
- `EVENT_ESTIMATES.csv`: RAW4, PRESERVE_XI, crossfit and endpoint-count estimates,
  standard errors, descriptive intervals and frozen-scale labels.
- `METHOD_DIFFERENCES.csv`: independent-packet and estimator differences.
- `MASS_RESIDUALS.csv`: sum of four raw probability-change estimates and its SE.
- `ENDPOINT_OVERLAP.csv`: weight means, ESS and maximum normalized weights.
- `PACKET_STATISTICS.json`: complete four-event covariance and crossfit diagnostics.
- `SUMMARY.json`, `INPUT_INDEX.json`, `COMPLETE.json`: scope, identity and outputs.

## Statistical interpretation

The signed target is always `joint_1 - joint_0`. Events are a partition:
X is exact reconstructed world, S is another world with the same executor result,
W is a valid world with a different executor result, and I is invalid output.
The target distribution uses the original non-thinking input, full softmax,
actual EOS or truncation at 64 tokens. It is not generic QA accuracy.

ORIGIN and MIX use their original, unnormalized importance contributions. Both
endpoint scores are evaluated on the same sampled action, preserving covariance.
PRESERVE_XI is the pre-existing fixed correction. Crossfit splits by an outcome-
independent seed; every bootstrap replicate resamples original draws within each
fold and refits both coefficients. A thousand refits estimate conditional
sampling uncertainty, not variation across training seeds. No reference labels
are used to select a method in this analysis.

Normal intervals and nonparametric bootstrap intervals are descriptive. A packet
with no examples of an event cannot reveal that event's unseen tail by bootstrap.
Zero empirical variance, tiny bootstrap width or exact zero sum is not accuracy
certification. The frozen numerical precision labels retain this limitation.

The count baseline also gets conservative Clopper-Pearson intervals. Each
endpoint gets a two-sided 97.5% interval; subtracting endpoints gives at least
95% coverage for one difference by the union bound, under the iid Bernoulli
sampling model. A second version divides alpha by 96 (two origins, twelve
prompts, four events). It covers this fixed count family, not new prompts or
training seeds. These intervals stay nonzero at zero or saturated counts.
Method: [SciPy exact proportion interval documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html).

The additional normal family threshold applies to 96 cells **per fixed
method/comparison**, not all twelve comparison sets jointly. It is still an
approximation. The historical bridge threshold uses
`Phi^-1(1 - .01/(8*12)) = 3.708691...` and an added absolute `1e-4` tolerance;
passing that broad diagnostic does not prove equivalence.

Historical `UNRESOLVED_NOT_CERTIFIED` labels were unconditional strings in
`pair_observation_diagnostics()`, not the outcomes of 24 precision tests. The
analysis computes actual descriptive scale labels without rewriting those
original receipts. With no exact VLM event truth, it does not report measured
bias, prediction RMSE, scientific reliability or online SSVC effectiveness.

## Conditional complete-action support bounds

`src.modeling_v4.bridge_support_bounds` adds a CPU-only diagnostic using the
union of complete actions already scored at both endpoints in the four packets.
It deduplicates token sequences including EOS/truncation, not decoded strings.
For event c, let a_Lc and a_Rc be observed probability masses and let t_L and t_R
be the remaining total endpoint mass. Conditional on accurate normalized
numerical scores, the event response lies in
`[a_Lc - a_Rc - t_R, a_Lc - a_Rc + t_L]`.

The union can be selected from the measured data; the containment is pointwise
and retains all unseen mass. It does not mistake a canonical output probability
for the entire event probability. A repeated complete action with differing
scores, conflicting labels/inputs, or mass exceeding one invalidates the bound.
The bounds are not confidence intervals or independent-test references. They
do not bound BF16/systematic scoring error, and ordinary floating arithmetic is
not outward-rounded interval arithmetic. Being inside a wide bound is not
evidence of accuracy. Comparing estimates to these bounds is post-hoc and must
not be used as locked-test model selection.

Run `python -m src.modeling_v4.bridge_support_bounds --campaign <campaign>
--out <new_output>` on a CPU Slurm allocation with CUDA hidden and both analysis
modules installed. Alternatively load both standalone modules with importlib
beside the immutable source snapshot. The actual launch argv is retained with
the server analysis receipts.

## Server CPU command

Use the frozen experiment checkout for imports, and upload the new analysis
module and batch script separately. This avoids altering running GPU code.

```bash
sbatch --account=rose --qos=soujanya-poria-startfund-2026-03 \
  --job-name=ssvc-v4-bridge-error-analysis \
  --output=/absolute/analysis_%j.out --error=/absolute/analysis_%j.err \
  /absolute/modeling_v4_bridge_analysis.sbatch \
  /absolute/python /absolute/frozen/ssvc_flow \
  /absolute/bridge_analysis.py /absolute/campaign /absolute/new_output
```

The batch script requests no GPU and explicitly hides CUDA. It refuses an
existing nonempty output directory and writes only a new analysis directory.
