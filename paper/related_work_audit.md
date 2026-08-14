# Related-work metadata and claim-boundary audit

Audit date: 2026-08-14 (Asia/Singapore, UTC+08).

This file is a metadata ledger, not a finished bibliography. Titles and authors
below were checked against arXiv metadata on the audit date; venue is omitted
unless an official venue page or the arXiv record explicitly supports it. A URL
to an arXiv abstract does not imply peer-reviewed acceptance.

## Closest multimodal measurement and credit work

### MathLens / capability decomposition

- **Verified title:** *What MLLMs Learn about When they Learn about Multimodal
  Reasoning*
- **Authors:** Jiwan Chung, Neel Joshi, Pratyusha Sharma, Youngjae Yu, Vibhav
  Vineet.
- **Status:** arXiv:2510.01719, v4 dated 2026-05-07; no conference venue claimed
  here.
- **URL:** https://arxiv.org/abs/2510.01719
- **Boundary:** The abstract says MathLens operationally decomposes performance
  into perception, reasoning, and multimodal-specific components. This project
  must not claim the first such decomposition; its candidate addition is the
  interventional error-selection/coupling mechanism.

### MeasureBench

- **Verified title:** *Do Vision-Language Models Measure Up? Benchmarking Visual
  Measurement Reading with MeasureBench*
- **Authors (arXiv v2):** Fenfen Lin, Yesheng Liu, Haiyu Xu, Chen Yue, Zheqi
  He, Mingxuan Zhao, Miguel Hu Chen, Jiakang Liu, JG Yao, Xi Yang.
- **Authors (CVF proceedings):** Fenfen Lin, Yesheng Liu, Haiyu Xu, Yue Chen,
  Zheqi He, Mingxuan Zhao, Miguel Hu Chen, Jin-Ge Yao, Xi Yang. The two
  official records differ, so this audit preserves their source-specific
  author lines instead of silently reconciling them.
- **Status:** arXiv:2510.26865, v2 dated 2026-03-24; CVPR 2026 is supported by
  the CVF Open Access proceedings copy.
- **URLs:** https://arxiv.org/abs/2510.26865 and
  https://openaccess.thecvf.com/content/CVPR2026/papers/Lin_Do_Vision-Language_Models_Measure_Up_Benchmarking_Visual_Measurement_Reading_with_CVPR_2026_paper.pdf
- **Official project/repository:** https://flageval-baai.github.io/MeasureBenchPage/
  and https://github.com/flageval-baai/MeasureBench
- **Boundary:** It supplies real and synthesized measurement-reading tasks and a
  synthesis pipeline. This project must not claim the first observation of
  measurement-reading failures or error cancellation. The untested addition is
  whether pre-RL interventional compensability predicts post-RL error selection
  and shift fragility.

### PRCO

- **Verified title:** *Seeing with You: Perception-Reasoning Coevolution for
  Multimodal Reasoning*
- **Authors:** Ziqi Miao, Haonan Jia, Lijun Li, Chen Qian, Yuan Xiong, Wenting
  Yan, Jing Shao.
- **Status:** arXiv:2603.28618, v2 dated 2026-04-09; no conference venue claimed
  here.
- **URL:** https://arxiv.org/abs/2603.28618
- **Boundary:** The abstract introduces PRCO with observer/solver roles and
  role-specific rewards. CompBias studies the outcome-only selection law and
  joint semantic ambiguity; it does not present another role-specific reward as
  its contribution.

### PRPO

- **Verified title:** *PRPO: Perception-Reinforced Policy Optimization via
  Token-Level Dynamic Advantage Reshaping*
- **Authors:** Qiming Li, Tianlun Li, Xiaolong Cheng, Hangyu Li, Ruiyan Gong,
  Kangning Niu, Kaitao Jiang, Mu Xu.
- **Status:** arXiv:2606.08708, v1 dated 2026-06-07; no conference venue claimed
  here.
- **URL:** https://arxiv.org/abs/2606.08708
- **Boundary:** PRPO explicitly reshapes token-level advantage for perceptual
  tokens. CompBias must not claim token-level credit assignment as new and does
  not add such a module to its main method.

### Perception-Correction Distillation (PCD)

- **Verified title:** *Correcting What You Cannot See: Credit Assignment for
  Perception Distillation in Multimodal Reasoners*
- **Authors:** Feng Xiong, Leyan Xue, Hongyu Lin.
- **Status:** arXiv:2607.28336, v2 dated 2026-08-01; no conference venue claimed
  here.
- **URL:** https://arxiv.org/abs/2607.28336
- **Boundary:** PCD identifies correctable perception failures for distillation.
  CompBias cannot claim the first diagnosis that trajectory-level reward fails
  to locate perception errors; its target is the induced error marginal and
  co-adapted compensation equilibrium.

### Asymmetric post-training

- **Verified title:** *On Asymmetric Optimization of Reasoning and Perception in
  Vision-Language Model Post-Training*
- **Authors:** Xueqing Wu, Yu-Chi Lin, Kai-Wei Chang, Nanyun Peng.
- **Status:** arXiv:2605.29496, v1 dated 2026-05-28; no conference venue claimed
  here.
- **URL:** https://arxiv.org/abs/2605.29496
- **Boundary:** The abstract studies unequal perception/reasoning gains and
  reward coupling. CompBias must not claim discovery of this asymmetry; it must
  quantitatively distinguish reduced local error from changed cross-error
  coupling.

### Perception-R1

- **Verified title:** *Perception-R1: Advancing Multimodal Reasoning Capabilities
  of MLLMs via Visual Perception Reward*
- **Authors:** Tong Xiao, Xin Xu, Zhenya Huang, Hongyu Gao, Quan Liu, Qi Liu,
  Enhong Chen.
- **Status:** arXiv:2506.07218, v3 dated 2026-03-03; no conference venue claimed
  here.
- **URL:** https://arxiv.org/abs/2506.07218
- **Boundary:** This is direct prior art for adding a perception reward. Such a
  reward is a CompBias control that may break a compensatory equilibrium, not a
  proposed contribution.

### PeRL-VL

- **Verified title:** *More Than the Final Answer: Improving Visual Extraction
  and Logical Consistency in Vision-Language Models*
- **Authors:** Hoang Anh Just, Yifei Fan, Handong Zhao, Jiuxiang Gu, Ruiyi Zhang,
  Simon Jenni, Kushal Kafle, Ruoxi Jia, Jing Shi.
- **Status:** arXiv:2512.12487, v1 dated 2025-12-13; no conference venue claimed
  here.
- **URL:** https://arxiv.org/abs/2512.12487
- **Boundary:** The abstract names its framework PeRL-VL and separates visual
  extraction reward from reasoning SFT. The plan's shorthand title was not the
  verified paper title. CompBias cannot claim the first decoupled improvement
  strategy.

## Visual evidence dependence and compensatory behavior

### VisualFLIP

- **Verified title:** *VisualFLIP: Do Predictions Depend on Task-Critical Visual
  Evidence in Multimodal Reasoning?*
- **Authors:** Didi Zhu, Changrui Chen, Stefanos Zafeiriou, Jiankang Deng.
- **Status:** arXiv:2606.07872, v1 dated 2026-06-05; no conference venue claimed
  here.
- **URL:** https://arxiv.org/abs/2606.07872
- **Boundary:** VisualFLIP uses same-question perturbation pairs whose answers
  deterministically flip. CompBias must not claim the first evidence-dependence
  test; it asks whether coupling contribution predicts paired/OOD failure.

### Visual-thinking faithfulness

- **Verified title:** *On the Faithfulness of Visual Thinking: Measurement and
  Enhancement*
- **Authors:** Zujing Liu, Junwen Pan, Qi She, Yuan Gao, Guisong Xia.
- **Status:** arXiv:2510.23482, v1 dated 2025-10-27; no conference venue claimed
  here.
- **URL:** https://arxiv.org/abs/2510.23482
- **Boundary:** The abstract reports correct answers with inaccurate/ignored
  visual information and intervention-based faithfulness tests. CompBias cannot
  equate an explicit chain of thought with an internal causal process.

### Cognitive mismatch

- **Verified title:** *Cognitive Mismatch in Multimodal Large Language Models for
  Discrete Symbol Understanding*
- **Authors:** Yinghui Li et al.; the current arXiv record lists 14 authors.
- **Status:** arXiv:2603.18472, v2 dated 2026-04-09; no conference venue claimed
  here.
- **URL:** https://arxiv.org/abs/2603.18472
- **Boundary:** The abstract reports recognition-reasoning inversion and
  compensation by language priors/procedures. CompBias's candidate distinction
  is a conditional relative-gain law plus a trainable coordination equilibrium,
  which still requires originality and empirical validation.

## KL, binary reward, optimization, and multimodal-learning context

### Binary-reward degeneracy

- **Verified title:** *Binary Rewards and Reinforcement Learning: Fundamental
  Challenges*
- **Author:** Marc Dymetman.
- **Status:** arXiv:2605.02375, v1 dated 2026-05-04; no conference venue claimed
  here.
- **URL:** https://arxiv.org/abs/2605.02375
- **Boundary:** This is direct prior art for binary-reward degeneracy and the
  KL-selected filtered/Boltzmann distribution. Exponential tilting is explicitly
  not a CompBias novelty claim.

### Success conditioning

- **Verified title:** *Success Conditioning as Policy Improvement: The
  Optimization Problem Solved by Imitating Success*
- **Author:** Daniel Russo.
- **Status:** arXiv:2601.18175, v2 dated 2026-06-02; no conference venue claimed
  here.
- **URL:** https://arxiv.org/abs/2601.18175
- **Boundary:** Success-conditioned policy improvement is neighboring theory.
  CompBias must isolate what follows specifically after partitioning trajectories
  by controlled perception error.

### Reference-sampled Boltzmann projection

- **Verified title:** *Reference-Sampled Boltzmann Projection for KL-Regularized
  RLVR: Target-Matched Weighted SFT, Finite One-Shot Gaps, and Policy Mirror
  Descent*
- **Authors:** Yao Shu, Chenxing Wei, Hongbin Lin, Shuang Qiu, Hui Xiong.
- **Status:** arXiv:2605.02469, v1 dated 2026-05-04; no conference venue claimed
  here.
- **URL:** https://arxiv.org/abs/2605.02469
- **Boundary:** The abstract explicitly identifies the standard Boltzmann target
  and policy mirror descent. CompBias may use these as an oracle/baseline, not
  claim their invention.

### Gradient starvation

- **Verified title:** *Gradient Starvation: A Learning Proclivity in Neural
  Networks*
- **Authors:** Mohammad Pezeshki, Sékou-Oumar Kaba, Yoshua Bengio, Aaron
  Courville, Doina Precup, Guillaume Lajoie.
- **Status:** arXiv:2011.09468, v4 dated 2021-11-24; the arXiv record states
  NeurIPS 2021 proceedings.
- **URL:** https://arxiv.org/abs/2011.09468
- **Boundary:** Feature-learning imbalance is broader prior art. It does not by
  itself establish the prompt-conditional compensability selection law.

### Proximal Policy Optimization

- **Verified title:** *Proximal Policy Optimization Algorithms*
- **Authors:** John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford,
  Oleg Klimov.
- **Status:** arXiv:1707.06347, v2 dated 2017-08-28.
- **URL:** https://arxiv.org/abs/1707.06347
- **Boundary:** PPO/GRPO are approximate optimization baselines. An exact
  distribution-space theorem cannot be described as an exact per-step guarantee
  for these algorithms.

### veRL / HybridFlow

- **Verified title:** *HybridFlow: A Flexible and Efficient RLHF Framework*
- **Authors:** Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang,
  Ru Zhang, Yanghua Peng, Haibin Lin, Chuan Wu.
- **Status:** arXiv:2409.19256, v2 dated 2024-10-02; the arXiv record links DOI
  https://doi.org/10.1145/3689031.3696075.
- **URLs:** https://arxiv.org/abs/2409.19256,
  https://github.com/verl-project/verl, and
  https://verl.readthedocs.io/en/latest/algo/grpo.html
- **Boundary:** veRL is the planned execution framework. The local config builder
  emits only a reviewed subset of official keys. A preflight plan does not show
  that veRL or GRPO training ran.

### Qwen2.5-VL

- **Verified title:** *Qwen2.5-VL Technical Report*
- **Authors:** Shuai Bai et al.; the arXiv record lists 27 authors.
- **Status:** arXiv:2502.13923, v1 dated 2025-02-19.
- **URLs:** https://arxiv.org/abs/2502.13923,
  https://github.com/QwenLM-corp/Qwen2.5-VL, and the pinned model tree
  https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/66285546d2b821cf421d4f5eb2576359d3770cd3
- **Boundary:** Qwen2.5-VL-3B-Instruct is a planned test subject, not a new model
  contribution. No Qwen training or evaluation is recorded in the project
  registry.

### Uni-modal feature learning

- **Verified title:** *On Uni-Modal Feature Learning in Supervised Multi-Modal
  Learning*
- **Authors:** Chenzhuang Du, Jiaye Teng, Tingle Li, Yichen Liu, Tianyuan Yuan,
  Yue Wang, Yang Yuan, Hang Zhao.
- **Status:** arXiv:2305.01233, v3 dated 2023-06-23; no conference venue claimed
  here.
- **URL:** https://arxiv.org/abs/2305.01233
- **Boundary:** This is relevant to modality-wise learning imbalance but does
  not support an outcome-RL compensability or error-coupling claim.

## Current originality assessment

The classical tools are already occupied. The candidate contribution is the
full, still-unconfirmed chain

```text
controlled visual-error intervention
  -> compensability landscape
  -> outcome-induced error selection
  -> truthful/compensatory joint equilibria
  -> coupling-linked shift fragility.
```

This is a search hypothesis, not a guarantee of novelty. Before each paper
phase, search and record at least these queries: `multimodal RL compensability`,
`perception reasoning error coupling`, `spurious compensation equilibrium`,
`outcome reward error cancellation`, `visual error selection law`, and
`co-adaptation truthful equilibrium`. If a directly equivalent law/equilibrium
appears, narrow or remove the corresponding claim rather than changing names.

## Metadata still requiring later audit

- Formal BibTeX keys and author formatting for manuscript use.
- Venue/acceptance status for every 2025--2026 arXiv-only item.
- Licenses and stable code/data releases for VisualFLIP and other optional
  external benchmarks.
- MathLens, Geo3K, and other dataset licenses/scene annotations before any data
  are downloaded or adapted.
- A submission-date refresh of all records and an adversarial search for work
  equivalent to the selection and joint-equilibrium results.
