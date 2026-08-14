# Experiment registry v2

Last updated: 2026-08-14 (Asia/Singapore, UTC+08).

| ID | Stage | Status | Evidence / blocker |
|---|---|---|---|
| V2-SYS | A--E | CPU_RECORDED | Current commit `8eed5548b2f71f1cf60b0ba4a4d231fe24b5c146`: full suite 1,061 passed, 0 failed/errors in 982.39 seconds. Combined project coverage 81.8862%, above the 80% gate. `artifacts/metrics/v2_readiness_summary.json` SHA-256 `7e1f501bb3f8359b9d60602749e56254a68d4c2539ef826ac639bc0b166fb9af` records the instrumented timeout and isolated instrumented rerun transparently. |
| V2-THEORY | A | CPU_RECORDED | Natural selection, transport, task pseudometric, non-identifiability, crossed risk, frozen mediator, density ratio, new-support, evidence-audit, and GPU-stop tests: 32 passed from a clean start at commit `40e0baaf762898e3c937ad754c3c7c0fb4e07787`. `artifacts/theory_v2/property_tests.json` SHA-256 `6c128ee493af42e8e35d8b0074115feeb86dec3fab1a6551df2157fecad2328c`. |
| V2-CLAIM-AUDIT | A | CPU_RECORDED | Seven legacy claim classes are assigned keep/demote/retract/rerun in `artifacts/claim_evidence/v1_claim_audit.csv`, SHA-256 `970e39f7af06a6f23d26eecd71bfbb671bc0358a6374d548717dea6b6dece94d`; the source available in-repo is the v1 theorem note, not a separate v1 plan file. |
| V2-SMALL-NATURAL | B | CONFIRMATORY_CPU | Clean run at commit `a9c7ae017d3fde4f7ed117f0bea65c1b3419fdfa`, config SHA-256 `946ff9dbec15ac8bbf926e871e678d3e21519c34f52d69cc26f49f3c1eb8381e`, summary SHA-256 `809e93239126d38ea6f8d4414a6593826a08b113b127f53728ef24ed853fa8ac`. |
| V2-LM-ONLY | C | BLOCKED_LARGE_GPU | Private plan generated; minimum planning VRAM 24 GB. Vision/projector frozen, language LoRA. No training started. |
| V2-PROJECTOR-LM | C | BLOCKED_LARGE_GPU | Private plan generated; minimum planning VRAM 32 GB. Acquisition frozen; projector/readout and language can change. No training started. |
| V2-VISION-LORA | D | BLOCKED_LARGE_GPU | Private plan generated; minimum planning VRAM 48 GB. No training started. |
| V2-MAX-E2E | D | BLOCKED_LARGE_GPU | Private plan generated; minimum planning VRAM 80 GB; multi-GPU is operationally preferred. No training started. |
| V2-PARTIAL-ID | D | IMPLEMENTED_NO_VLM_INPUT | Strict validity and simultaneous max-stat certificate builder implemented; no VLM interface records exist. |
| V2-BLACKBOX | E | PREREGISTERED_NOT_RUN | Two-pass behavioral interfaces and wording are frozen; no provider/API execution is configured. |

## Confirmatory small-neural result

The clean CPU run used 1,000 semantic states, 8,000 rendered PIL images,
640,000 natural mediators, 20,480,000 image-cut forked continuations, 20,000
synthetic mediators, 20 seeds, and five error families. Every crossed-risk
identity passed.

Across families, `c_sel` and `c_fork` were about 0.957--0.960 and their paired
seed gap was close to zero. `c_syn` was about 0.885--0.888. The paired
`c_syn-c_fork` intervals were strictly negative for every family, with means
from -0.0711 to -0.0734. This establishes, in the controlled small-neural
system, that injected states are not interchangeable with naturally produced
mediators. It does not establish the same magnitude or direction in Qwen.

## GPU planning, not execution evidence

The config minima are screening values, not measured capacity results. For the
veRL/GRPO pilot, the operational recommendation is 48 GB for LM-only,
48--80 GB for projector+LM, 80 GB for vision LoRA, and two 80 GB GPUs for the
maximal end-to-end regime. Actual batch size and placement remain blocked until
the exact target machine passes the CUDA/container smoke.
