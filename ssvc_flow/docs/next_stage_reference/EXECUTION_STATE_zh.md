# SSVC 后续阶段执行状态

更新时间：2026-09-10。此文件记录实际进度；方案以 `CODEX_NEXT_STAGE_zh.md` 和 `configs/next_stage.yaml` 为准。

## 授权与范围

用户已明确授权后台按 **R2 → R3-cold → R4 → R3-warm** 顺序继续，包括必要实现、检查、GPU 作业和代码提交推送。后台任务 `ssvc` 每 15 分钟检查并继续可执行工作；状态无变化时静默，完成、失败、需要用户操作时通知。

只使用固定 Qwen3.5-9B revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`。R4 只运行 X_BASE/X_VALID，各 64 step，seed 17。R3-warm 后按文档停止；不开展 R5、额外模型或额外训练臂，不升级依赖。

## 已验收证据

- 原始 R1 作业 `146571`：120 条 reference、128 条 calibration on-policy smoke、四个主更新通过，生产路径为 `uncached_prefix_recompute`。
- 闭环作业 `146806`：Slurm COMPLETED 0:0；R0 全部 15 项 PASS；R1 补充真实 CUDA 控制 PASS，18 backward / 9 optimizer step / 0 新 rollout。详细数值见 `R0_R1_CLOSURE_20260910_zh.md`。
- 后续门禁同时验证原始 R1、R0 闭环和 R1 补充的 manifest、状态、数据和源码绑定，不能仅依赖摘要 PASS。
- 原始 N/L/P1、成功证据与失败诊断保持可追溯。已授权清理冗余失败结果，但不能删除其中独有的成功证据或借删除失败样本维持 PASS。

## 当前实施

- R2 输入、真实推理运行器和共享门禁已完成本地联调；**尚未提交 R2 GPU 作业**。
- R2 固定 72 calibration 场景，六个主条件共 2,160 输出；12 场景的长度/thinking 分支共 72 输出。每条输出落盘，严格核验身份后续跑；thinking 使用真实 processor 模板/token 校验。
- R3 直接梯度与类别梯度复用模块已通过 26 项新测试，尚未在 CUDA 上证明复用成立。只允许真实未裁剪梯度及实际 Adam 对照通过后采用；否则回退直接计算。
- R3 完整运行器、R4 完整训练/评估运行器仍需实现；现有 `optimizer_fork.py` / pilot 的 CPU 入口不能当成实际服务器实验。

本轮全套本地验证：`OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ../.venv/bin/python -m pytest tests -q`（cwd `ssvc_flow`），**463 passed，59.43 秒**。Ruff check、九个修改/新增 Python 文件的 format check、`bash -n scripts/ntu_r2.sbatch` 均通过。CPU fake 测试验证了 2,232 条完整覆盖、中断后续跑和再次续跑零重采样；不能将其算作模型实验。

服务器 CPU processor 预检：真实 `AutoProcessor.from_pretrained(snapshot, local_files_only=True)` 返回模板文本 SHA256 `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`，`<think>` / `</think>` token 为 248068 / 248069。thinking 的 generation prefix 以 `<think>\n` 结尾，非 thinking 以 `<think>\n\n</think>\n\n` 结尾；未加载模型权重。

原始 R1 环境文件只保存 Python 和 Transformers 版本；新门禁比较这两项，并明确原文件缺少其他依赖的历史版本。每次真实运行保存当前九个关键包版本，断点续跑要求精确一致，不回填历史证据。

## 下一步及依赖

1. 完成 R2 本地测试、源码审查、服务器 CPU processor/证据预检，提交并验证 push。
2. 无 SSVC 作业运行时更新服务器 checkout，提交 `scripts/ntu_r2.sbatch`；记录 job ID、commit、运行路径。运行期间禁止更新该服务器 checkout。
3. 验证 R2 完整样本覆盖、模型冻结、manifest 和配对统计。低正确率不是执行失败；数据/索引/图像传入缺陷阻塞受影响正式训练。
4. 实现并执行 R3-cold：48 train prompts、12 个 B4K8 bank、五个 λ 候选；control 只用于测量。类别梯度复用须两组实际直接梯度/Adam 对照通过。候选完全恢复参数、Adam、RNG，不复用 P3 rollout。
5. R4 两臂分别从 fresh LoRA/Adam 开始；各自 on-policy 轨迹。实施文档要求的 checkpoint、控制集 KL 门禁和 step 0/32/64 评估。不可从 R1/R3 scratch 起步。
6. R3-warm 绑定 X_BASE step64 的完整 Adam 状态，在相同 prompt IDs 上重新采样；执行要求的 3,072 条独立直接验证输出。验收并分析后停止后台任务。

## 固定服务器位置与运行约束

- SSH：`eee-cluster`；项目根：`/projects/varunssd/louis-ssvc`，真实路径可能解析为 `/projects/_ssd/varunssd/louis-ssvc`。
- checkout：`/projects/varunssd/louis-ssvc/dissertation-ssvc`；Python：`/projects/varunssd/louis-ssvc/envs/ssvc-py312/bin/python`。
- data：`reviewed-data-20260908/generated`；原 R1：`runs/NEXT_20260909/R1_server_146571_attempt_0`。
- R0：`runs/NEXT_20260909/R0_R1_close_146806_attempt_0/R0`；补充：同级 `R1_supplement`。
- 使用 Slurm、命名 GPU `pro6000`，沿用 account rose 和已有 QoS；不设置 `--mem` / `--cpus-per-task`。所有缓存、日志、临时文件放个人 project 目录。
- 真实加载器会核验 Hub revision；不可强制 `HF_HUB_OFFLINE` 或 `TRANSFORMERS_OFFLINE`。允许使用原有缓存，禁止依赖升级。
- 不操作其他项目的作业。保留本地无关 `reports/` 和三个带 ` 2.json` 后缀的 ownership 文件。
- 每次代码修改完成后审查限定 diff、运行相关检查，再 commit/push 并核实远端 hash；记录代码版本和实验版本。

R2 新作业使用独立目录 `runs/NEXT_20260909/R2_server_JOBID_attempt_0/R2`，24 小时时限。若需要同版本续跑，将环境变量 `SSVC_R2_RESUME_STAGE` 指向原 R2 stage，再提交相同脚本；新 wrapper 保存旧 stage 引用，并将完整 stage 一并归档。不得修改原 ledger 的失败记录或将已失败的执行样本视作完成；正常中断的未完成请求才可补齐。
