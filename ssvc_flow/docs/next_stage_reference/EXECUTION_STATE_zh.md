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

- R2 输入、真实推理运行器和共享门禁已完成；**真实 R2 作业 `146853` 已完成**，节点 `gpu-pro6000-4`。运行源码 `518b23f4fb6803706a7ded57a2b5bfbefbf2d6ab`，启动时间 `2026-09-09T17:43:44Z`。输出根为 `/projects/varunssd/louis-ssvc/runs/NEXT_20260909/R2_server_146853_attempt_0`，stage 为其 `R2` 子目录；Slurm COMPLETED 0:0，elapsed 02:42:15，2026-09-09T20:25:59Z 结束；2,232/2,232 输出与模型冻结检查 PASS，完整独立 CPU postflight 待执行。
- R2 固定 72 calibration 场景，六个主条件共 2,160 输出；12 场景的长度/thinking 分支共 72 输出。每条输出落盘，严格核验身份后续跑；thinking 使用真实 processor 模板/token 校验。
- R3-cold 已在本地实现输入选择、实际梯度/Adam fork、配对 control 响应统计、完整运行器和 `scripts/ntu_r3_cold.sbatch`，尚未提交 GPU 作业。新增共享门禁检查 R2 的真实执行身份、2,232 条完整 ledger、冻结状态、manifest、数据与原始 R0/R1 绑定。
- R3-cold 固定 48 train prompts、12 个 B4K8 bank、五个 λ；复用验证 bank 索引预注册为 `[1, 11]`，两组均通过真实未裁剪梯度及实际 Adam 对照后才允许复用，否则直接计算。每 bank 五个候选使用独立 ID；另做 λ=0 完整状态重放。A_BASE/A_VALID 只测梯度，不产生额外 Adam 候选。
- Control 固定 24 base scenes × 两接口 × 16 输出，共 768 条 proposal；响应 bank `[0, 6]` 的十个候选共 7,680 条评分。使用未裁剪、非自归一化 IS 和 5,000 次按 family 分层的 base-scene 配对 bootstrap；记录 ESS、最大归一化权重和类别支持，缺失支持不补成有效估计。SGD 仅报告未裁剪参数空间参考，不声称测得其概率响应。
- R3-cold 支持逐输出 ledger、不可变验证/bank attempt 和逐序列 control 评分续跑；失败原始输出和未提交 attempt 均保留，完成单元复验后复用。所有候选恢复参数、Adam、RNG，实际计数另记录验证、重放和中断重算成本。
- R3-cold 原始 rollout 现在保存完整 `prepared_hash`，梯度组重建时再次检查；R4 前置门禁重建 12 个 bank 与 60 个候选的样本/状态绑定，核验 tokenizer、原文、EOS、old-logprob、实际 Adam 和两个复用验证 bank。统计门禁独立重算响应 bank `[0, 6]` 的 5,000 次 bootstrap，核验 280 条 CSV 与 60 条 Parquet 候选；缺失、篡改或不可复算的产物均阻塞 R4。
- R4 已在本地实现固定输入、两臂训练/评估运行器、统计和 `scripts/ntu_r4.sbatch`，尚未在服务器部署或提交。`--allow-training` 显式授权实际两臂执行；锁定 YAML 保持原值。R3-warm 已在本地实现，入口为 `src.r3_warm_runtime` 和 `scripts/ntu_r3_warm.sbatch`；尚未服务器执行。不能将 R3-cold 或 CPU fixture 当成 warm/GPU 完成证据。
- R4 训练池保持原有 576 prompts，每臂使用相同 seed17 的固定 64×B4 顺序；每六步各群体四个 prompt slots。两臂均从证书绑定的初始 LoRA 和空 Adam 恢复，再分别生成当前策略 K8 轨迹。保存全部 128 次更新的不可变 attempt、逐组奖励/优势统计和完整 checkpoint；step64 每臂包含累计 2,048 个已完成训练 sample keys。
- R4 共计划 14,400 条新输出：训练 4,096、共享 step0 576、step32 1,152、step64 N/L/OOD 8,576。L 保持原有 48-token lock 与解析器；N 和 OOD 使用 64-token lock。graph-OOD 从原已审计的 graph_structure 池按 path/cycle × bar/line × 三 operation × 三实例固定选出 36 base scenes，全部通过唯一解、ID 范围和真值/图像隔离检查；不读取 confirm 样本。
- R4 的固定 step0 KL probe 在查看 R4 结果前选择每个 R3-cold control prompt 的 `sample_index=0`，共 48 条；只在基座、初始 LoRA、空 Adam、prompt、图像、generation lock 完全一致时复用，否则阻塞。每步 48 条、共 6,144 次 control 评分，不新增 control 生成；另外如实记录 4,096 次更新前 parity 和 4,096 次更新后训练评分。单条 proposal/题的噪声限制在报告中注明。
- R4 任一实际 KL 工程阈值超线保存出错步骤和全局停止标记，不能通过 resume 跳过；返回的非有限 likelihood、数值参数或策略污染保存原始证据并阻断新 attempt。正常中断保留已生成轨迹，从前一完整状态重算未提交更新并计入成本。初始 N/L 比较只使用来源和协议均验证的 P3 样本；无法对齐时明确标记未测量，来源哈希损坏则失败。
- R4 rollout 逐条保存文档 10.1 的来源、scene、输入、token、logprob、计时和内存字段，完整 `prepared_hash` 每个 prompt 计算一次。L 使用原始 truth/observation/error_index 和 semantic parser；N cue、唯一修复 solver、图像与 constraint_results 对 L 不适用时显式记录，不伪造 N 语义。FAIL/BLOCKED 也生成当前 `report_zh.md`，不能遗留旧 PASS 报告。
- R4 主分析使用配对 base-scene bootstrap；六群体共同区间限定为单指标的预声明群体范围。L 主整体权重固定为原 176 prompts 的组成，等 family 权重仅单列敏感性分析。退化或小样本区间标为不稳定，不据此宣称排除了下降或证明安全。

最新全套本地验证：`OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ../.venv/bin/python -m pytest tests -q`（cwd `ssvc_flow`），**763 passed，222.48 秒**。随后对 Git 来源读取的子目录 pathspec 修复运行全部 R4 gate 回归，**36 passed，42.44 秒**；其中真实临时 Git 仓库先 RED 后 GREEN，并从项目子目录实测读取原 `2bd53ae` 的57个 Python文件。Ruff 全 src/tests 检查、17个修改/新增 Python文件的 format check、三个 R3-cold/R4/R3-warm Slurm脚本的 `bash -n` 和限定 diff 检查均通过。两个warm统计文件仅格式化，AST逐项一致。

R3 CPU tiny-torch 集成覆盖 1,152 条生成、60 个独立候选、7,680 条 control 评分；注入采样及 bank 提交中断后续跑，实测 86 次 Adam 更新（60 候选 + 12 重放 + 8 复用验证 + 6 重算），再次续跑新增计算为零。R4 CPU tiny-torch 集成覆盖 14,400 条生成、128 次有效更新，注入一次实际 Adam 后中断并重算，实测 129 Adam / 4,128 backward；L 2,816 条均保留 48-token 配置，130 个 checkpoint 引用含八个里程碑，step64 各保留 2,048 个训练 keys。完整完成后的再次续跑新增 generation/forward/backward/Adam 均为零。R3-warm CPU tiny-torch验证4,224条输出和80次实际Adam调用（60候选＋12重放＋8复用验证），起点Adam64、候选65，direct中断恢复后完整重复运行新增计算为零，R4来源checkpoint字节不变。这些是实现验证，不能算作 Qwen/CUDA 实验。

本地审查修复了 origin optimizer param-group 未核验、跨 bank 候选 ID 重复和 raw row 缺少 run ID 的问题；相关回归和全套测试已通过。输入选择只依据固定数据元信息，实际 plan hash 为 `e84efd66534321fc8bb2c4546f4a9f7d10e70714beaa3999a48b97a9cb218974`。

R4 代码审查另修复了加载耗时进入 resume 身份、来源损坏被降为未测量、评分污染在恢复后漏检，以及 KL 超线诊断与原子步骤提交之间的中断窗口。停止记录先于局部结果和完成标记写入；即使随后中断，恢复仍在模型加载前阻断。

服务器 CPU processor 预检：真实 `AutoProcessor.from_pretrained(snapshot, local_files_only=True)` 返回模板文本 SHA256 `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`，`<think>` / `</think>` token 为 248068 / 248069。thinking 的 generation prefix 以 `<think>\n` 结尾，非 thinking 以 `<think>\n\n</think>\n\n` 结尾；未加载模型权重。

原始 R1 环境文件只保存 Python 和 Transformers 版本；新门禁比较这两项，并明确原文件缺少其他依赖的历史版本。每次真实运行保存当前九个关键包版本，断点续跑要求精确一致，不回填历史证据。

首次服务器 CPU 门禁发现新代码将 R1 `environment_lock.config_sha256` 误当作 canonical JSON hash。核验原写入代码 `audit_r1_runtime.py` 后确认该字段是原 YAML 字节哈希 `c50fe86c280f89f0166f110908ed73837d80e011eabdabc4e8df6c29ef630753`。仅修正新门禁与测试 fixture，原证据不变；回归先 RED 后 GREEN，相关 40 项测试通过，未因此消耗 GPU 作业。

修正后真实服务器 CPU 门禁 **PASS**：72 场景 / 2,232 请求，Python 3.12.14、Transformers 5.14.1。新记录包版本为 torch 2.13.0+cu130、peft 0.19.1、tokenizers 0.22.2、accelerate 1.14.0、huggingface-hub 1.30.0、numpy 2.5.2、Pillow 12.3.0、safetensors 0.8.0。预检与提交记录已取回本地 `runs/NEXT_20260909/verified_146853`；不包含尚未完成的模型结果。

已提交并验证 GitHub 同分支 push 的 R2 版本：`6c40dda` 为 R2/梯度实现，`518b23f` 为哈希兼容修正；R3-cold 初版为 `ef91770`，R4 实现为 `2bd53ae`。R2 运行期间服务器 checkout 一直保持完整 `518b23f4fb6803706a7ded57a2b5bfbefbf2d6ab`，该 hash 在作业结束后再次实测。

R2 原始归档已取回本地 `runs/NEXT_20260909/verified_146853`，SHA256 为 `375348e4311fdf6847a7a1fcce9205dc6b54be37959378f58a78bf3f53dfc983`；安全解包后独立验证18个 manifest 文件、47个 Git 历史源码文件和2,232条唯一、执行检查通过的 raw。六主条件各360条，SYM_LONG/SYM_THINKING各36条；原始参数、基座、LoRA前后哈希一致，0 backward、0 Adam。该归档/源码检查独立于下一步 CPU 输入/语义/统计复算。

## R3-warm 与独立验收实现

R3-warm 在通过完整 R4 门禁后载入 X_BASE step64 的完整 LoRA、Adam、RNG、sampler 和 2,048 条累计训练 keys。可选 `--checkpoint` 必须精确等于该门禁选定的文件，不能替换成摘要、参数文件或其他训练臂。重新生成同一 cold 训练/control prompt IDs 的 1,152 条当前策略轨迹，执行 60 个候选、12 次 λ0 重放及八次复用验证更新；每个候选均从同一 Adam64 起点独立更新一次，实际候选 Adam step 为65。

直接验证固定 banks `[0, 6]` × λ `[0, 1]`，每候选 48 prompts × K16，总计3,072条新输出；warm 总新输出为4,224。共同随机种子的键排除 λ，独立样本键仍包含完整候选身份；可变 token 路径不保证天然序列对应。报告 λ0/λ1 相对 warm θ 的绝对响应，以及 λ1−λ0 辅助响应的预测值、直接观察值、残差和配对 base-scene bootstrap 区间；保留支持不足、小 panel 和重叠警告。

共享 R3 运行器增加持久数值/污染阻断：返回的异常 likelihood 保存原值、token、prepared hash 和 attempt；在引擎恢复 scratch 前写入全局标记，同 stage resume 在模型加载前拒绝。采样/评分异常路径也在恢复前检查参数、Adam 和冻结参数版本。正常中断保留原始轨迹并允许有计数的未完成 attempt 重算。

新增 R2 独立 CPU postflight：依据原 R1 processor/EOS 证书加载本地 pinned processor/config，不加载模型权重；重建请求、每个输入及图像张量哈希，重新解码 token 并重算所有标注和六份表格/JSON/模板差异文件。输出写入原 stage 外的新目录，原文件哈希必须保持不变。已在运行中的真实 R2 上只读抽验 SYMBOLIC、IMAGE_CUE、SYM_THINKING 各一条，共三条全部通过；这仅证明处理器重建接口可用，完整2,232条独立 postflight 将在部署已测试代码后执行。

R4 门禁独立核验14,400条输出、两臂各64步的完整 checkpoint/Adam/RNG 链、128次更新、6,144条固定 control 评分、更新前后训练评分、CPU 重建输入及报告重算，才返回完整 warm checkpoint 绑定。统计重算另修复了 L 敏感性整体加权的浮点加法顺序，跨 Python hash seed 的结果逐位一致；没有修改奖励、优化器、冻结模块或统计样本定义。

## 下一步及依赖

1. 已完成 R2 实现、源码审查、服务器 CPU processor/证据预检、提交及 push 验证。
2. 跟踪 `146853`：查看 Slurm、`R2/progress.json`、`checks/r2.log` 和 `result.txt`。运行期间可本地准备后续阶段，但禁止更新该服务器 checkout。
3. R2 结束后执行 `python -m src.r2_result_audit`，验证完整样本覆盖、模型冻结、manifest、原始输入/token/标注和配对统计，取回归档并核验 SHA256。低正确率不是执行失败；数据/索引/图像传入缺陷阻塞受影响正式训练。
4. R2 完成并验收后，确认没有 SSVC 作业运行，再部署已测试的 R3-cold 代码，先做服务器 CPU 门禁和输入预检，再执行真实 R3-cold。Control 只用于测量；类别梯度复用须两组实际直接梯度/Adam 对照通过。候选完全恢复参数、Adam、RNG，不复用 P3 rollout。
5. R3-cold 完整验收后，运行 `python -m src.next_stage_preflight --phase R4` 的只读 CPU 门禁（包括 R3 原始样本、实际 checkpoint、统计产物重算和 R4 固定输入）。通过后设置 `SSVC_R3_COLD_STAGE` 为已验收 stage，再提交 R4 脚本。两臂分别从 fresh LoRA/Adam 开始；不可从 R1/R3 scratch 起步。
6. R4 全部完成后运行 `--phase R3-warm --r3-dir <已验收cold> --r4-dir <已验收R4>` CPU 预检。通过后设置 `SSVC_R3_COLD_STAGE` 和 `SSVC_R4_STAGE`，提交 warm 脚本；绑定 X_BASE step64 完整 Adam，在相同 prompt IDs 上重新采样并执行3,072条直接验证。验收及最终汇总完成后停止后台任务。

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

R4 脚本使用 `runs/NEXT_20260909/R4_server_JOBID_attempt_RESTART/R4`，同为 71 小时时限；同版本正常中断续跑使用 `SSVC_R4_RESUME_STAGE`。提交必须显式提供项目目录内已验收的 `SSVC_R3_COLD_STAGE`。新 CPU preflight CLI 支持 `--phase R3-cold|R4|R3-warm`，输出独立 JSON，拒绝覆盖旧记录；它不加载模型权重或请求 GPU，不能代替随后真实运行的已加载模型检查。

R3-warm 脚本使用 `runs/NEXT_20260909/R3_warm_server_JOBID_attempt_RESTART/R3_warm`，71小时时限；同版本正常中断使用 `SSVC_R3_WARM_RESUME_STAGE`。必须同时提供项目内已验收的 cold/R4 stage，完整完成后的 resume 不新增生成、forward、backward 或 Adam。
