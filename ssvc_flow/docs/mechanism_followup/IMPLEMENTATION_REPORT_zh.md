# SSVC mechanism-followup-v2 本地实现报告

按 2026-09-14 交接包完成本地实现、CPU 验收与服务器命令准备，停止在首次服务器原件核验和 GPU smoke 之前。实际状态由 [machine_readable_readiness.json](machine_readable_readiness.json) 给出。CPU fake adapter 的采样用于验证软件链路，梯度、Adam、checkpoint 和恢复使用真实 PyTorch CPU；没有产生新的 Qwen 实验结论。

## 版本、来源与变更范围

- 分支：`codex/ssvc-mechanism-followup-20260914`；工作目录：仓库 `ssvc_flow/`。
- 起点与测试时 HEAD：`7e405faa241e0085f701cdc858ba423ea76ba5a8`。测试记录同时保存当时 dirty state 与源码/测试 SHA256，不能把该 HEAD 单独当作新代码版本。
- 实现 commit：`d261d67daba5cac54a6736e366d951ede8d19932`；后续交付标记提交仅补充文档版本，不改变已验收源码。最终实际源码文件见 [IMPLEMENTATION_SOURCE_MANIFEST.json](IMPLEMENTATION_SOURCE_MANIFEST.json)。相对起点的完整交付 diff 保存在本地 `runs/mechanism_followup_cpu/delivery.diff`，也可用 `git diff 7e405faa241e0085f701cdc858ba423ea76ba5a8 d261d67daba5cac54a6736e366d951ede8d19932 -- ssvc_flow` 复核。
- 原附件的12个清单负载文件已核对大小和 SHA256；[design/](design/) 保留其原始字节、配置和参考数学。设计输入中的旧 `status` 是原始文档状态，当前实现状态在独立 readiness 文件。
- 新增9个 `src/followup_*.py` 模块、9个测试模块、1个 CPU adapter、冻结配置、3个 Slurm 模板及1个测试记录脚本。没有修改已有生产模块、旧测试或 main；原有未提交文档和未跟踪文件保留并排除于提交。
- 原 warm runtime 记录的 commit 是 `9cb60cd8e4e26f881e83f80f20b55999f8b8ecde`；其63个源码文件逐一与本地比对，全部 SHA256 一致。Git blob ID、文件 SHA256、canonical JSON hash 和 tensor state hash 分开记录。

[源码核对报告](SOURCE_DIFF_REVIEW_zh.md) 及本地 `runs/mechanism_followup_cpu/s0/` 保留核对证据。R4 字段 `decision_sha256` 按现有 `r4_continuation.py` 实际契约使用 canonical JSON hash，不能与文件字节 SHA256 混比；此次对应绑定通过。

## 已实现的行为

| 文件 | 实际职责 |
|---|---|
| `followup_protocol.py` / `followup_cli.py` | 冻结设计验证、精确预算、无模型 plan/audit/dry-run、显式 GPU/训练标志及 smoke/S1 证据门禁、Slurm 渲染 |
| `followup_parent.py` | 目录/受限 ZIP 只读审计、大小/重复/越界限制、固定三份元数据与 phase/step/model/origin 绑定；compact 与 raw 状态分离 |
| `followup_inputs.py` | 原 bank 与 composite 精确装配、token/old-score/prepared-input 身份、train/control 防泄漏、seed17 旧顺序桥接和29/41确定性排序 |
| `followup_updates.py` | joint 与 no_x_off 前归一化奖励、固定 PPO 分母、显式零梯度、真实 Adam/global clip、完整参数/Adam/RNG/模式/buffer 恢复与故障证据、CPU FP64向量几何 |
| `followup_statistics.py` | X/S/W/I 与 pX/v/qX/qS、配对覆盖、场景共同 bootstrap、固定面板边界、OIS/SNIS和支持状态、alias与多seed汇总 |
| `followup_runtime.py` | 六个 S1 bank 的候选和重放、独立训练 checkpoint、稳定完整推理指纹、精确 alias、逐样本账本、计数/响应/几何、no-clobber、续跑及物理更新配额 |
| `followup_train.py` | S2 三臂多seed on-policy CPU/真实入口、原初始化绑定、训练/评估隔离、N/L/OOD/R2各自契约和固定预算、检查点及中断恢复 |
| `followup_backend.py` | 现有真实 adapter 的 generate/logprobs/parser 接线、递归上游原件验证、固定本地9B snapshot、旧/新GPU smoke，以及历史两臂终点 R2 只评估入口 |

旧数学与 Adam 路径保持原样；新 joint 回归使用相同 FP32 production loss 契约，纯数学及向量距离参考使用 FP64。没有调整 epsilon、clip、奖励阈值来追求预期方向。全零优势仍对全部 trainable 参数显式赋零并执行 Adam，历史动量可产生更新。

恢复包括每个子模块的 training flag、buffer/forward状态及 RNG；更新异常保留 failure 并恢复 origin，恢复失败将对象置为不可继续。推理 fingerprint 排除非语义运行时长，包含实际推理状态与语义配置；参数相同但 Adam 不同的候选仍有独立 checkpoint。

S2的R2摘要保留三个诊断condition和family/condition分项，避免相同interface标签合并两个不同诊断条件；采样量保持不变。

历史终点完成记录按 manifest → binding → completed 标记顺序发布。最后两处发布中断的回归测试验证：已采864条可完整复用，续跑不新增生成，不执行 Adam。CPU fixture 同时核验标签和实际参数设备。

## CPU 验收记录

| 验证调用 | 通过 / 失败 / 跳过 / 错误 | 实际记录 |
|---|---:|---|
| 完整套件（最后复核修复前） | 1228 / 0 / 0 / 0 | `runs/mechanism_followup_cpu/final/result.json` |
| 最后修复后全部新增 followup 测试 | 205 / 0 / 0 / 0 | `runs/mechanism_followup_cpu/final_followup/result.json` |
| 附件标准库数学参考 | 15 / 0 / 0 / 0 | `runs/mechanism_followup_cpu/reference_verified/result.json` |
| CLI/Slurm/计划/审计/dry-run | 16条命令 exit 0 | `runs/mechanism_followup_cpu/non_pytest_checks/results.json` |
| 最终 Ruff/格式/脚本语法/文件保持 | 全部通过 | `runs/mechanism_followup_cpu/final_static_checks/results.json` |

完整套件包含旧1028项及新增200项；之后仅修复历史完成标记、fixture设备检查及S2的R2条件分项摘要，并重跑最终全部205项新增测试。两次源码快照在各自运行期间均未变化；没有把后一次修复宣称为在前一次全套测试中执行过。最终相关验证无失败、跳过或错误。

所有调用使用 Python 3.12.14、torch 2.13.0（CPU）、numpy 2.5.2、pytest 9.0.3，PyYAML 6.0.3；设置模型离线和空 CUDA_VISIBLE_DEVICES。完整版本与日志 SHA256 见 [CPU_VERIFICATION_RUNS.json](CPU_VERIFICATION_RUNS.json)。

逐项验收见 [CPU_ACCEPTANCE_RESULTS_zh.md](CPU_ACCEPTANCE_RESULTS_zh.md)，覆盖附件77项。每个最终 pytest 调用保留 `invocation.json`、`result.json`、`junit.xml`、`pytest.log` 和 fixture 产物；记录 cwd、Python路径/版本、库版本、HEAD、dirty state、完整命令、环境变量、退出码、passed/failed/skipped/errors、日志哈希和运行期间源文件是否变化。公开清单只保存相对路径及哈希；原始记录留在本地忽略的 `runs/`，不上传权重、原始训练输出或私有绝对路径。

关键端到端实测：S1 六个 bank 共47次真实 CPU Adam，18个逻辑直接候选在小型 fixture 中形成13个唯一推理策略、104条 fake 输出（4题×2次/唯一策略），检查点与alias分别保留。S2 至少2 seeds×3 arms、每run2步的实际 CPU 更新、评估和恢复链通过。历史端点 fixture 执行864条、0次 Adam。上述数字是软件验收，不能当作实际48题×16次的 S1 或64步9B训练结果。

复现实验从 `ssvc_flow/` 执行，使用已安装的同一 CPU 环境和新的输出目录：

```bash
: "${PY:?设置已核验 CPU 环境的 Python}"
"$PY" scripts/verify_followup_cpu.py \
  --out runs/mechanism_followup_cpu/recheck_full tests
"$PY" scripts/verify_followup_cpu.py \
  --out runs/mechanism_followup_cpu/recheck_reference \
  docs/mechanism_followup/design/reference/test_reference_math.py
```

实现过程保留 RED/GREEN 和中间失败日志。曾发生测试收集/接口接线及诊断 parser 问题，修复后由最终回归覆盖；没有将依赖或命令路径错误记作有效数学 RED。静态检查只对本轮新 Python 文件进行。原附件 Markdown 有6行标准双空格换行，因需保持清单哈希而原样保留；排除这份原始附件后的 scoped `git diff --check` 通过。

## 元数据与仍未核验的原件

三份固定元数据的实际 SHA256：

| 文件 | SHA256 |
|---|---|
| `runtime_lock.json` | `14ad5ec1bd7596b93720404e77f6ed220fa0123963c7a026438fcfdab81a545c` |
| `bank_manifest.json` | `dd4e543c6c8ae5076ce20ad46df7e51afd4cad7750f33b028e46fbf61536a416` |
| `candidate_manifest.json` | `01735c725d06f99a74c3c18d45f380153c73f9944a9658679a8828e2e47d0d12` |

目录与ZIP一致，warm step64、Qwen3.5-9B revision、B4/K8和来源链元数据一致；相关60行 warm 梯度表、128行 R4学习曲线及8项绑定检查通过。模型状态哈希只是已核对的元数据字段，不表示已从缺失张量重新算出。

当前紧凑包缺少真实 `origin.pt`、`samples.jsonl`，也未核验候选 checkpoint 张量、原始数据/图片及 prepared model tensors。状态是 `BLOCKED_MISSING_PARENT_RAW`；可交给服务器核验，但不能启动真实计算。`parent_raw_verified`、`raw_tensors_verified`、`gpu_smoke_passed`、`gpu_started`、`training_started` 均保持 false。

## 服务器边界、预算与限制

准确命令、原件字段、已核验的 CLI、Slurm 变量、每个 run、续跑/停止和回传文件见 [SERVER_HANDOFF_zh.md](SERVER_HANDOFF_zh.md)。本次没有 SSH、9B加载、模型下载、GPU计算、sbatch 或真实训练。第一步是填入私有实际路径并执行无模型 `prepare-server`；原件核验后，首次需要 GPU 的操作是4次 scratch Adam 的旧/新 smoke。

| 后续阶段 | 冻结工作量 |
|---|---|
| S1 | 原 bank00/03/04/05/11 + composite；41候选+6重放=47 Adam、1,504反向序列、直接逻辑输出上限13,824 |
| 新增 S2 | seed17仅新arm；seed29/41各3arms；共7runs×64步，每run65份完整续跑checkpoint，训练输出14,336、评估输出62,272 |
| 历史 seed17 两臂 | 只补终点 R2，各864输出，总1,728；不重训 |

S1失败更新尝试按可能已经消耗一次 Adam 保守记账；配额耗尽拒绝继续。S1测量链通过只证明技术链完整，不要求研究效果向预期方向变化。后续先读取已完成结果再决定下一步，不启用自动作业接续。

服务器实际吞吐、峰值显存和 walltime 尚未测量；完整冻结参数哈希与非有限检查的9B成本需由 smoke 实测。当前沿用上游严格认证数据路径与已有 processor 缓存规则，路径迁移需要独立桥接；不会静默放宽证书。相同seed跨arm的step0评估分别执行，未扣减其预算。所有区间、重要性采样和支持诊断保留 `NOT_CERTIFIED`；CPU通过不构成训练效果、模型尺度、安全性或因果证据。S3、seed53、追加采样和其他模型保持关闭。
