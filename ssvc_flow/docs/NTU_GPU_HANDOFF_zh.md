# NTU：第一次 GPU 测试交接

2026-09-08 更新：修复后的真实 P1 作业 `144911` 已完整通过，退出 `0:0`。
实测结果、数值路径修复、验收后计数补正及证据见 [P1 实测记录](P1_FAILURE_144840_zh.md)。
下文保留环境准备和独立复验步骤。

如需独立复验，请把分支 `codex/ssvc-flow-ntu` 的最新代码放到获授权的 NTU 服务器，
在按 NTU 规则分配到的 CUDA GPU 作业中运行 P1。代码没有旧服务器地址、工作目录或模型快照路径默认值。
本轮不自动进入 P3 冻结评估、pilot、confirm 或完整 SSVC 训练。

## 取得代码与准备环境

以下命令从您在 NTU 选定的工作目录开始；不预设登录节点、partition、account 或绝对路径。

```bash
git clone --branch codex/ssvc-flow-ntu --single-branch https://github.com/Lhan-chding/dissertation.git dissertation-ssvc
cd dissertation-ssvc/ssvc_flow
bash scripts/setup_ntu.sh
```

准备 Python 3.12。setup 在 `ssvc_flow/.venv` 建立独立环境，已有环境时拒绝覆盖。
若 NTU 提供了自己的兼容 CUDA/PyTorch 环境，可在其中安装 requirements，并通过
`SSVC_PYTHON` 指定解释器；P1 会记录实际依赖版本。bootstrap 文件是候选依赖，不能当作 GPU 已验证锁。
bootstrap 中 PyTorch2.13.0 与 torchvision0.28.0 按
[官方安装矩阵](https://pytorch.org/get-started/previous-versions/) 配对。
Qwen 的 AutoProcessor 即使只处理图片，也需要导入视频 processor 的 torchvision 依赖。
主模型已固定到首个真实 P1 使用的 revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`；固定 revision 本身不代表 P1 已通过。
若服务器驱动需要指定 CUDA wheel，可先按该矩阵在新环境中安装对应组合，再装其余依赖；
不要在未知驱动条件下假定默认 wheel 可用。

研究文档建议优先评估48GB单卡、BF16基座与FP32 LoRA。实际需要多少显存和时间目前未知；
P1 将记录加载、单次/K=8生成及反向传播的峰值、吞吐和更新耗时。
使用足够磁盘空间的 Hugging Face 缓存；缓存目录由您的新 NTU 环境决定。

## 只运行 P1

### Louis 的已建环境：用 sbatch 离线排队

已在 `/projects/varunssd/louis-ssvc/envs/ssvc-py312` 安装依赖时，不必再运行
`setup_ntu.sh`。更新代码后，从登录节点提交以下批处理脚本：

```bash
(
  set -e
  cd /projects/varunssd/louis-ssvc/dissertation-ssvc
  git pull --ff-only origin codex/ssvc-flow-ntu
  mkdir -p /projects/varunssd/louis-ssvc/logs
  sbatch ssvc_flow/scripts/ntu_p1.sbatch
)
```

脚本默认申请 `cluster02` / `rose` 账号 / `soujanya-poria-startfund-2026-03` QoS，
1 张 `pro6000`，运行上限 6 小时。此 QoS 来自老师在
`/projects/varunssd/slurm_train.sh` 和 `start_gpu_job.sh` 的提交示例；
2026-09-05 的 `sacctmgr show assoc` 输出也确认 `varun024` 的 `rose` 账号获准使用该 QoS。
CPU/内存由集群按 GPU 型号分配，不照搬旧示例的 `--mem=256G`。
2026-09-08 已确认该项目 QoS 拒绝 A6000、允许 PRO6000；作业 `144840`
使用 `--gres=gpu:pro6000:1` 实际启动。脚本默认值与这组已验证的提交参数一致。
6 小时是运行上限，不是等待时间或耗时承诺；结束后立即释放资源。

这是老师指定的项目 QoS，不能沿用此前 `override-limits-but-killable` 的
“不计配额、可抢占”说明或等待时间估计。项目额度、
作业数及抢占规则尚未核实，不承诺立即启动。需要时可只读查询：

```bash
sacctmgr show qos where name=soujanya-poria-startfund-2026-03 \
  format=Name%45,Priority,Flags%60,GrpTRES%100,GrpTRESMins%100,MaxTRESPU%100
```

已经提交的作业不会因 `git pull` 改变 QoS；更新后的脚本仅对新提交生效。
若本次 P1 已提交，先查看它的 `scontrol show job JOBID`，不要重复提交。

看到 `Submitted batch job <编号>` 后，即可断开 SSH、VPN 或关闭电脑。
原来用 `srun --pty` 排队的交互申请是另一个作业，应在其原终端按 Ctrl+C 结束；
不要使用按用户批量取消的命令，以免影响同账号的其他实验。

批处理按顺序执行：

1. 检查 Slurm 上下文、独立解释器，记录源码 commit、GPU、实际依赖版本。
2. `pip check`、模型依赖导入、CUDA/BF16 矩阵乘法与反向传播、FP32 参数更新。
3. 在 CPU 上运行 SSVC 回归测试（禁止下载），记录 JUnit 结果。
4. 在本次作业目录生成 seed=17 的 3,632 个场景，记录预算 dry-run。
5. 运行原有锁定的 Qwen3.5-9B P1：36 个 prompt、K=8、2 次真实更新，
   包含概率一致性、缓存、图像输入、基座冻结、梯度和恢复重放检查。
6. 核对真实 CUDA P1 的 PASS 与验证锁，生成报告、证据包和 SHA-256。

2026-09-08 的概率修复保持 BF16 基座、FP32 LoRA 和原来的 0.02 报警值：
采样关闭 cache，teacher-forcing 对每个实际生成 token 的相同前缀做无 cache 重算，
包括有梯度的 likelihood。完整／分段／逐 token cache 仍单独审计。
模型 EOS 与官方 tokenizer 的聊天 EOS 合并；仅剥离最终 EOS，严格答案 parser 不做修补。
这比整段一次打分需要更多前向计算，dry-run 现在区分序列数与逐 token 前向上界；
真实耗时由本次 P1 测量。修复依据见 [P1 失败定位记录](P1_FAILURE_144840_zh.md)。

任一前置检查失败就停止后续模型工作，返回非零退出码；不会安装软件或执行 P3–P9。
输出目录为 `/projects/varunssd/louis-ssvc/runs/slurm-<编号>-attempt-0/`，
其中 `checks/` 保存分步日志，`runs/P1/` 保存全部模型记录与检查点，
`data/generated/` 保存本次数据，`reports/` 保存报告，`result.txt` 保存最终状态。
同名目录已存在时拒绝覆盖。证据包与 `.sha256` 文件放在该目录旁边。

当前脚本使用 `--no-requeue`：被抢占、超时或节点失败后保留已有文件，
不自动从头重跑或冒险恢复部分写入的状态。强制终止时可能来不及生成报告、
证据包或最终状态，此时以 `sacct`、Slurm 日志和原始目录为准。
恢复前按下文的 identity/revision/config 检查要求审阅原始 P1 目录。

重新连接后查看（把 `JOBID` 换成提交返回的编号）：

```bash
squeue -j JOBID -o "%.12i %.2t %.10M %.10l %R"
sacct -X -j JOBID --format=JobID,JobName%28,State%20,ExitCode,Elapsed,NodeList
tail -n 80 /projects/varunssd/louis-ssvc/logs/p1-JOBID.out
cat /projects/varunssd/louis-ssvc/runs/slurm-JOBID-attempt-0/result.txt
```

排队期间尚无输出目录和日志是正常现象；只有作业取得资源并开始执行后才生成。
`result.txt` 的 `state=PASS` 需要所有批处理检查成功；`runs/P1/status.json`
单独记录模型 P1 是否通过。若 P1 通过而打包失败，批处理仍返回失败并保留原始证据。

### 手动交互调试

可以先执行不加载模型的预算查询：

```bash
.venv/bin/python -m src.rollout --phase smoke --model qwen35_9b --dry-run --out runs/P1_budget
```

在已经分配 GPU 的作业 shell 中执行：

```bash
bash scripts/ntu_smoke.sh
```

首次缺少数据时会确定性生成全部3,632个场景和图像。P1只读取18个 calibration 场景的两接口，
不会查看 confirm 模型结果。正式 smoke 使用36个 prompt、K=8；默认2次真实更新，
加上恢复/同状态重放检查。预算计入policy与reference打分以及更新forward/backward，
混合缓存审计的额外forward计数由实际运行记录。

对已存在的运行目录，普通执行会拒绝覆盖。中断后使用：

```bash
bash scripts/ntu_smoke.sh --resume
```

resume 必须核对模型revision、数据/实际图像、配置和保存状态。不一致时保留失败原因，
使用新的输出目录重新进行独立 smoke，而非修改历史记录。如果读取的文件已损坏，先保留原目录。

## 把这些文件带回本地

保留整个 `runs/P1/`（包括状态、模型审计、逐样本JSONL、逐步更新记录、checkpoint、依赖冻结、错误日志）
以及 `data/generated/manifest.json`。完整checkpoint用于恢复，勿仅上传最终PASS截图。

```bash
.venv/bin/python -m src.report --run-root runs --out reports
tar -czf ssvc_ntu_P1_evidence.tar.gz runs/P1 reports data/generated/manifest.json
sha256sum ssvc_ntu_P1_evidence.tar.gz
```

若P1失败，仍保留OOM、兼容性错误、空支持或cache不支持记录。I4缓存分支可能为unsupported；
不得用部分KV复制宣称完整混合状态。P1必须通过实际关键检查才会写PASS。
通过后先审阅锁定环境和测量结果，再实现／执行下一阶段，不能根据是否出现预期负面现象更换模型。

## 已知待补项

- 研究包引用的 `THEORY_REVIEW` 未提供；对应原文反例核对保持未完成。
- 独立人工36张图检查尚待研究者记录；联系表可用
  `python scripts/build_contact_sheets.py --dataset data/generated --out runs/P0` 生成。
- 旧权重缺失及8项历史运行语义缺失已在P0报告中列明，不能声称C3精确复现。

官方接口依据：[Qwen3.5模型文档](https://huggingface.co/docs/transformers/model_doc/qwen3_5)、
[9B配置](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json)、
[官方chat template](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/chat_template.jinja)。
这些网页引用解释实现来源；真实运行会将模型revision解析为不可变commit，并保存模板/processor/tokenizer哈希。
