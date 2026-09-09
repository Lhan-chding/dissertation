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

- R2 输入、真实推理运行器和共享门禁已完成；**真实 R2 作业 `146853` 已启动**，节点 `gpu-pro6000-4`。运行源码 `518b23f4fb6803706a7ded57a2b5bfbefbf2d6ab`，启动时间 `2026-09-09T17:43:44Z`。输出根为 `/projects/varunssd/louis-ssvc/runs/NEXT_20260909/R2_server_146853_attempt_0`，stage 为其 `R2` 子目录；尚未验收完成。
- R2 固定 72 calibration 场景，六个主条件共 2,160 输出；12 场景的长度/thinking 分支共 72 输出。每条输出落盘，严格核验身份后续跑；thinking 使用真实 processor 模板/token 校验。
- R3-cold 已在本地实现输入选择、实际梯度/Adam fork、配对 control 响应统计、完整运行器和 `scripts/ntu_r3_cold.sbatch`，尚未提交 GPU 作业。新增共享门禁检查 R2 的真实执行身份、2,232 条完整 ledger、冻结状态、manifest、数据与原始 R0/R1 绑定。
- R3-cold 固定 48 train prompts、12 个 B4K8 bank、五个 λ；复用验证 bank 索引预注册为 `[1, 11]`，两组均通过真实未裁剪梯度及实际 Adam 对照后才允许复用，否则直接计算。每 bank 五个候选使用独立 ID；另做 λ=0 完整状态重放。A_BASE/A_VALID 只测梯度，不产生额外 Adam 候选。
- Control 固定 24 base scenes × 两接口 × 16 输出，共 768 条 proposal；响应 bank `[0, 6]` 的十个候选共 7,680 条评分。使用未裁剪、非自归一化 IS 和 5,000 次按 family 分层的 base-scene 配对 bootstrap；记录 ESS、最大归一化权重和类别支持，缺失支持不补成有效估计。SGD 仅报告未裁剪参数空间参考，不声称测得其概率响应。
- R3-cold 支持逐输出 ledger、不可变验证/bank attempt 和逐序列 control 评分续跑；失败原始输出和未提交 attempt 均保留，完成单元复验后复用。所有候选恢复参数、Adam、RNG，实际计数另记录验证、重放和中断重算成本。
- R4 完整训练/评估运行器和 R3-warm 仍需实现。当前 R3 CLI 对 warm 明确返回 BLOCKED，不能将 R3-cold 或 CPU fixture 当成 warm/GPU 完成证据。

最新全套本地验证：`OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ../.venv/bin/python -m pytest tests -q`（cwd `ssvc_flow`），**514 passed，70.64 秒**。Ruff check、十个修改/新增 Python 文件的 format check、`bash -n scripts/ntu_r3_cold.sbatch` 和 diff 检查均通过。R3 CPU tiny-torch 集成覆盖 1,152 条生成、60 个独立候选、7,680 条 control 评分；注入采样及 bank 提交中断后续跑，实测 86 次 Adam 更新（60 候选 + 12 重放 + 8 复用验证 + 6 重算），再次续跑新增计算为零。这些是实现验证，不能算作 Qwen/CUDA 实验。

本地审查修复了 origin optimizer param-group 未核验、跨 bank 候选 ID 重复和 raw row 缺少 run ID 的问题；相关回归和全套测试已通过。输入选择只依据固定数据元信息，实际 plan hash 为 `e84efd66534321fc8bb2c4546f4a9f7d10e70714beaa3999a48b97a9cb218974`。

服务器 CPU processor 预检：真实 `AutoProcessor.from_pretrained(snapshot, local_files_only=True)` 返回模板文本 SHA256 `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`，`<think>` / `</think>` token 为 248068 / 248069。thinking 的 generation prefix 以 `<think>\n` 结尾，非 thinking 以 `<think>\n\n</think>\n\n` 结尾；未加载模型权重。

原始 R1 环境文件只保存 Python 和 Transformers 版本；新门禁比较这两项，并明确原文件缺少其他依赖的历史版本。每次真实运行保存当前九个关键包版本，断点续跑要求精确一致，不回填历史证据。

首次服务器 CPU 门禁发现新代码将 R1 `environment_lock.config_sha256` 误当作 canonical JSON hash。核验原写入代码 `audit_r1_runtime.py` 后确认该字段是原 YAML 字节哈希 `c50fe86c280f89f0166f110908ed73837d80e011eabdabc4e8df6c29ef630753`。仅修正新门禁与测试 fixture，原证据不变；回归先 RED 后 GREEN，相关 40 项测试通过，未因此消耗 GPU 作业。

修正后真实服务器 CPU 门禁 **PASS**：72 场景 / 2,232 请求，Python 3.12.14、Transformers 5.14.1。新记录包版本为 torch 2.13.0+cu130、peft 0.19.1、tokenizers 0.22.2、accelerate 1.14.0、huggingface-hub 1.30.0、numpy 2.5.2、Pillow 12.3.0、safetensors 0.8.0。预检与提交记录已取回本地 `runs/NEXT_20260909/verified_146853`；不包含尚未完成的模型结果。

已提交并验证 GitHub 同分支 push 的 R2 版本：`6c40dda` 为 R2/梯度实现，`518b23f` 为哈希兼容修正。后续 R3/R4 实现仅更新本地/GitHub；**运行中的服务器 checkout 继续固定 `518b23f`，不要 pull**。最近一次现场检查，R2 作业 `146853` 为 RUNNING，elapsed 46:52，已完成 712/2,232；这只是进度快照，后续以现场文件为准。

## 下一步及依赖

1. 已完成 R2 实现、源码审查、服务器 CPU processor/证据预检、提交及 push 验证。
2. 跟踪 `146853`：查看 Slurm、`R2/progress.json`、`checks/r2.log` 和 `result.txt`。运行期间可本地准备 R3/R4，但禁止更新该服务器 checkout。
3. R2 结束后验证完整样本覆盖、模型冻结、manifest 和配对统计，取回归档并核验 SHA256。低正确率不是执行失败；数据/索引/图像传入缺陷阻塞受影响正式训练。
4. R2 完成并验收后，确认没有 SSVC 作业运行，再部署已测试的 R3-cold 代码，先做服务器 CPU 门禁和输入预检，再执行真实 R3-cold。Control 只用于测量；类别梯度复用须两组实际直接梯度/Adam 对照通过。候选完全恢复参数、Adam、RNG，不复用 P3 rollout。
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

R3-cold 脚本使用 `runs/NEXT_20260909/R3_cold_server_JOBID_attempt_RESTART/R3_cold`，71 小时时限（低于 partition 的 72 小时上限；这是预算上限，不是完成时间预测）。同版本续跑使用 `SSVC_R3_RESUME_STAGE`，新 wrapper 保留旧 stage 引用并一并归档；未通过 R2 验收前不能提交该作业。
