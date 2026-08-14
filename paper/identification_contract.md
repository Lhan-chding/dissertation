# Identification contract for v2 compensability

Last updated: 2026-08-14 (Asia/Singapore, UTC+08).

## What is identified

1. Final answers identify outcome behavior only.
2. Natural mediator records identify selection associations conditional on
   sample and interface.
3. Image-cut fork replay identifies causal downstream value only under the
   replay, exclusion, sufficiency, naturalness, and parser gates.
4. Model/oracle crossed interventions identify `D_P`, `D_R`, and `Gamma` at
   that interface.
5. Multiple valid interfaces identify an interval, not a unique anatomical
   split.

## Interface gates

An interface is valid only if every registered gate passes:

| Gate | Threshold |
|---|---:|
| Oracle loss | <= 0.01 |
| Replay answer-distribution JS | <= 0.05 |
| Replay accuracy gap | <= 0.03 |
| Image replacement/removal effect after cut | <= 0.01 |
| Parser/state reliability | >= 0.95 |
| Natural states per core error | >= 200 |
| Natural inputs per core error | >= 50 |

Invalid interfaces remain visible with failed reasons but are excluded before
computing the simultaneous max-stat interval. A favorable `Gamma` cannot rescue
an invalid interface.

## Evidence records

Natural, forked, synthetic, crossed-risk, and checkpoint-distribution records
are immutable and retain source kind, sample, checkpoint, interface, seed, and
provenance. Continuations are nested repeats: estimates average within mediator
and input before treating sample IDs as bootstrap clusters.

## Claim language

- Frozen visual tower: say `readout`, `reasoning`, or `interaction` change; do
  not say visual acquisition improved.
- Trainable visual tower: `operational perception change` is allowed; a unique
  internal factorization is not.
- Black-box API: use exactly `operational compensation certificate under
  interface family M`.
- Synthetic mediator: label `off-support stress test` unless transport audits
  justify approximation to natural states.

## Stop boundary

The four Qwen regimes produce private, non-executable plans only. Large-GPU
training remains blocked until Phase-D human review, a target CUDA smoke,
pinned local model snapshot verification, hardened container/SBOM/vulnerability
evidence, and an authenticated execution extension exist.
