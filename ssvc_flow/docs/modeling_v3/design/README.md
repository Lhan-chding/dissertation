# SSVC Modeling V3 — Observation Geometry, Calibration Coverage, Real 9B Validation

本包是下一阶段面向Codex的实验设计。先阅读 `START_HERE_FOR_CODEX.md`。

- `CODEX_EXPERIMENT_PLAN_zh.md`：主实验规范。
- `THEORY_AND_ESTIMATORS_zh.md`：数学对象、推导与数值边界。
- `GPU_RUNBOOK_zh.md`：两张PRO6000的分阶段并行与准确接口契约。
- `CPU_ACCEPTANCE_CHECKLIST_zh.md`：测试和科学验收。
- `protocol.json`：机器可读配置与派生工作量。
- `SOURCES_AND_NOVELTY_zh.md`：来源和相关工作边界。
- `reference/`：独立数学参考与测试，非生产训练器。
- `verification/`：本次实际执行的参考测试记录。
- `evidence/`：上一轮已公开结果摘录，只作开发依据。

本次没有修改GitHub、加载9B、提交Slurm或运行V3研究实验。参考数学测试的通过只表明其对应恒等式/契约实现自洽。
