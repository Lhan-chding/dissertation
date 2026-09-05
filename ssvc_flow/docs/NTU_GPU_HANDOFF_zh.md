# NTU：第一次 GPU 测试交接

当前需要您把分支 `codex/ssvc-flow-ntu` 的最新代码放到新获授权的 NTU 服务器，
在您按 NTU 规则分配到的 CUDA GPU 作业中运行 P1。代码没有旧服务器地址、工作目录或模型快照路径默认值。
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
若服务器驱动需要指定 CUDA wheel，可先按该矩阵在新环境中安装对应组合，再装其余依赖；
不要在未知驱动条件下假定默认 wheel 可用。

研究文档建议优先评估48GB单卡、BF16基座与FP32 LoRA。实际需要多少显存和时间目前未知；
P1 将记录加载、单次/K=8生成及反向传播的峰值、吞吐和更新耗时。
使用足够磁盘空间的 Hugging Face 缓存；缓存目录由您的新 NTU 环境决定。

## 只运行 P1

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
