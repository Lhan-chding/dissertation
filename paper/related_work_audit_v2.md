# Related-work and originality audit v2

Audit date: 2026-08-14 (Asia/Singapore, UTC+08).

This is a bounded, adversarial novelty audit for the v2 natural-trajectory and
partial-identification protocol. It is not proof that no equivalent work
exists. The audit searched primary arXiv, PMLR, and official proceedings
records using combinations of:

- `multimodal RL perception reasoning reward coupling`;
- `VLM natural mediator replay fork causal intervention`;
- `synthetic error natural error transport`;
- `partial identification perception reasoning VLM`;
- `activation replay image cut causal mediator`;
- `checkpoint density ratio error distribution outcome RL`.

The search was deliberately split into three threat classes: multimodal credit
assignment, causal representation/interchange interventions, and isolated
intervention validity. Exact-phrase searches for the proposed combination of
natural mediator replay, forked continuations, synthetic transport gaps, and
multi-interface partial identification did not identify a direct match. This
negative search result is time-bounded and must be refreshed before submission.

## Executive originality verdict

The broad thesis is **not novel enough to claim**:

> Outcome-only multimodal RL can favor reasoning over perception, and separating
> or separately rewarding perception can improve visual reasoning.

Several 2025--2026 papers already diagnose that problem or propose
perception-aware rewards, role separation, process rewards, or alternating
perception/reasoning optimization. The older VQA literature also documents
correct answers paired with failed perceptual subquestions.

The defensible candidate contribution is the narrower identification bundle:

1. distinguish natural selection success `c_sel`, natural-state fork replay
   `c_fork`, and synthetic-injection success `c_syn`;
2. use natural image-cut replay to test whether a naturally occurring mediator
   itself supports downstream compensation;
3. measure the synthetic-to-natural transport gap instead of assuming that an
   injected error represents a natural error;
4. report crossed interventional risk and coordination gaps without a linear
   hidden-state error model;
5. acknowledge non-identifiability of a unique perception/reasoning boundary
   and aggregate valid operational interfaces with simultaneous partial
   identification;
6. separate frozen sensory acquisition from trainable visual readout and
   downstream compensation;
7. test the trajectory-selection law against checkpoint error-density ratios.

No single work found in this audit combined those seven elements. This supports
presenting the bundle as a **candidate methodological contribution**, not using
an unconditional "first" claim. A real-VLM result remains required.

## Closest overlap: multimodal RL and credit assignment

### Asymmetric optimization of reasoning and perception

- **Source:** Xueqing Wu, Yu-Chi Lin, Kai-Wei Chang, and Nanyun Peng,
  *On Asymmetric Optimization of Reasoning and Perception in Vision-Language
  Model Post-Training*, arXiv:2605.29496, submitted 2026-05-28.
- **Primary URL:** https://arxiv.org/abs/2605.29496
- **Direct overlap:** controlled synthetic tasks separate perception from
  reasoning; the paper attributes RL asymmetry to outcome-reward coupling and
  evaluates a perception-aware reward.
- **Boundary:** CompBias cannot claim the first diagnosis that outcome rewards
  correlate more strongly with reasoning than perception, or the first
  perception-aware intervention. Its proposed difference is to identify what
  natural error trajectories are selected and then distinguish natural replay
  from synthetic injection.
- **Threat level:** very high for the broad framing; incomplete overlap with the
  v2 identification design.

### Bad Seeing or Bad Thinking / MoCA

- **Source:** Haozhe Wang et al., *Bad Seeing or Bad Thinking? Rewarding
  Perception for Multimodal Reasoning*, arXiv:2605.14054; the arXiv record says
  ICML 2026 Oral.
- **Primary URL:** https://arxiv.org/abs/2605.14054
- **Direct overlap:** explicitly decomposes interleaved perception and reasoning
  steps, uses a blindfolded reasoning proxy, verifies perceptual fidelity, and
  routes rewards by error source.
- **Boundary:** CompBias cannot claim the first perception/reasoning credit
  decomposition, blindfolded reasoner, or modality-aware reward. The v2 claim
  is diagnostic and identification-focused: it does not assume that one
  decomposition is the unique internal truth.
- **Threat level:** very high for decomposition and reward-routing claims.

### Perceive-to-Reason / PRA-GRPO

- **Source:** Hongxing Li et al., *Perceive-to-Reason: Decoupling Perception and
  Reasoning for Fine-Grained Visual Reasoning*, arXiv:2607.01191.
- **Primary URL:** https://arxiv.org/abs/2607.01191
- **Direct overlap:** a Perceiver/Reasoner two-stage formulation and alternating
  perception/reasoning GRPO using final-answer supervision.
- **Boundary:** role separation and alternating updates are occupied prior art.
  CompBias's frozen regimes are controls for identifying acquisition, readout,
  and compensation changes, not an architecture contribution.
- **Threat level:** high for any claimed novelty in staged roles or alternating
  multimodal RL.

### Perception-Aware Policy Optimization

- **Source:** Zhenhailong Wang et al., *Perception-Aware Policy Optimization for
  Multimodal Reasoning*, arXiv:2507.06448.
- **Primary URL:** https://arxiv.org/abs/2507.06448
- **Direct overlap:** identifies perception errors as a multimodal RL bottleneck
  and adds an implicit-perception KL loss to policy optimization.
- **Boundary:** a perception-aware loss, better perception under RL, and reduced
  perception error are not CompBias novelty claims.
- **Threat level:** high for intervention/method claims; lower for causal
  identification of naturally selected errors.

### SQuINTing at VQA Models

- **Source:** Ramprasaath R. Selvaraju et al., *SQuINTing at VQA Models:
  Introspecting VQA Models with Sub-Questions*, CVPR 2020 Oral,
  arXiv:2001.06927.
- **Primary URL:** https://arxiv.org/abs/2001.06927
- **Direct overlap:** models can answer a reasoning question correctly while
  failing its low-level perception subquestion, i.e. a right-answer/wrong-reason
  consistency failure.
- **Boundary:** error cancellation and a correct final answer hiding perceptual
  failure are established observations, not new discoveries here.
- **Threat level:** decisive against broad historical-first claims.

The broader metadata ledger in `paper/related_work_audit.md` additionally
covers PRCO, PRPO, PCD, Perception-R1, PeRL-VL, VisualFLIP, visual-thinking
faithfulness, binary-reward degeneracy, success conditioning, and Boltzmann
projection. Those components cannot be relabeled as v2 novelty.

## Adjacent causal intervention and representation work

### Interchange Intervention Training

- **Source:** Atticus Geiger et al., *Inducing Causal Structure for
  Interpretable Neural Networks*, ICML 2022.
- **Primary URL:** https://proceedings.mlr.press/v162/geiger22a.html
- **Overlap:** aligns high-level causal variables with neural representations
  and trains against counterfactual interchange interventions.
- **Boundary:** activation interchange and causal abstraction are established.
  V2's possible distinction is replaying naturally sampled multimodal
  mediators to estimate selection/compensation quantities, not inventing
  activation intervention.

### Interventional causal representation learning

- **Source:** Kartik Ahuja et al., *Interventional Causal Representation
  Learning*, ICML 2023.
- **Primary URL:** https://proceedings.mlr.press/v202/ahuja23a.html
- **Overlap:** formal identification of latent causal factors using
  interventional data, including weaker block-affine identification under
  imperfect interventions.
- **Boundary:** interventional identifiability is prior art. V2 uses a more
  conservative operational-interface partial-identification claim because a
  black-box VLM generally does not supply the assumptions needed for unique
  latent identification.

### Isolated causal effects of language

- **Source:** Victoria Lin, Louis-Philippe Morency, and Eli Ben-Michael,
  *Isolated Causal Effects of Natural Language*, ICML 2025.
- **Primary URL:** https://proceedings.mlr.press/v267/lin25k.html
- **Overlap:** shows that interventions intended to isolate one language feature
  can be biased when non-focal content is poorly approximated, and audits
  fidelity and overlap.
- **Boundary:** this is strong adjacent prior art for the synthetic-transport
  problem. V2 applies the concern to naturally occurring versus injected VLM
  mediators and estimates `c_syn-c_fork`; it must cite rather than claim the
  general intervention-validity insight.

## Claim decisions

| Candidate statement | Decision | Reason |
|---|---|---|
| Outcome-only multimodal RL under-trains perception | Do not claim as novel | Directly occupied by asymmetric-post-training and perception-aware RL papers. |
| Perception and reasoning should be separated | Do not claim as novel | MoCA, P2R, PRCO, SQuINT, and related work already operationalize this idea. |
| Correct answers can hide perception errors | Do not claim as novel | Explicitly documented by SQuINT and later faithfulness work. |
| Perception-specific rewards improve VLMs | Control only | PAPO, Perception-R1, Perceval, and related methods occupy the method space. |
| Synthetic mediator injection equals natural error | Reject as an assumption | The controlled v2 CPU result directly contradicts interchangeability in its registered system. |
| `c_sel`, `c_fork`, `c_syn` should be separately estimated | Candidate contribution | No exact combination was identified in the bounded search. |
| Natural mediator fork replay diagnoses co-state dependence | Candidate contribution | Activation intervention exists, but this selection-versus-natural-replay use was not found. |
| Multi-interface partial identification of compensation | Candidate contribution | General causal representation identification exists; the VLM operational certificate combination was not found. |
| Exact originality of the full bundle | Not yet established | Requires submission-time search and successful real-VLM evidence. |

## Falsification and update policy

The candidate novelty must be narrowed or withdrawn if a work is found that
already combines all of the following on multimodal RL trajectories:

1. natural error-conditioned selection moments;
2. mediator-cut forked continuations from natural states;
3. an explicit natural-versus-synthetic transport certificate;
4. crossed perception/reasoning risks or an equivalent interaction estimand;
5. multi-interface partial identification under boundary non-uniqueness; and
6. checkpoint error-density-ratio validation.

Before manuscript submission, rerun forward/backward citation search from the
four closest 2026 papers, inspect newly published ICML/CVPR/NeurIPS records, and
check titles/venues/authors again. The manuscript should use wording such as
"we introduce a framework that jointly..." rather than "the first framework"
unless that later audit supplies stronger evidence.
