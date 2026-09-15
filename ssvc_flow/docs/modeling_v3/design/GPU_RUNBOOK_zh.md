# 两张 PRO 6000 的执行与交接规范

## 1. 并行原则

默认两张卡各运行一个独立9B副本，不先把一个9B训练拆成两卡DDP。两卡主要并行：不同seed/arm的训练，或冻结候选的生成与评分。

| 阶段 | GPU worker A | GPU worker B | CPU工作 |
|---|---|---|---|
| Q4 | 相同策略的生成/评分桥接 | 配对副本与跨卡零处理检查 | 标签、协方差、数值复算 |
| Q5源轨迹 | 一个seed/arm | 另一个独立seed/arm | 监测、manifest、压缩归档 |
| Q5响应 | 某原点候选分支/评分 | 另一原点或同对比另一端评分 | pilot、bank选择、预测拟合 |
| Q5参考 | 独立参考packet生成 | 端点评分/MIX复核 | 更新参考精度报告 |
| Q6 | 冻结窗口参考 | 另一独立窗口 | 离线跟踪与误差分析 |

共同随机数依靠每条请求的RNG key，不依靠两个进程“恰好同时”生成。跨GPU结果只有通过数值与身份一致性检查后才能合并。

如果源模型单卡真实训练OOM，依次测试microbatch1、梯度checkpointing等不改变目标的配置；分布式FSDP是独立新后端，需要新的概率/梯度/优化器一致性与吞吐检查，不能自动替换。

## 2. 服务器前置清单

- 查看已排队/运行作业，整个项目现有任务和本轮新任务合计最多用两张GPU；不scancel旧任务、不更改其输出目录。
- 读取现有通过的Slurm模板与QOS权限；历史记录使用 `soujanya-poria-startfund-2026-03`，是否仍有效由当前账户与模板核实。
- 当前snapshot、模型revision、tokenizer/processor、LoRA mask、dtype、Python/PyTorch/Transformers和CUDA版本按实际锁记录。
- 不假定 physical GPU index 等于进程内编号。使用Slurm分配的CUDA_VISIBLE_DEVICES；不手动占用未分配卡。
- 核验可用RAM、scratch空间、checkpoint大小、峰值显存。旧3/6GiB研究输出上限不再适用，但物理配额依然不能超用。
- 源码和测试在独立worktree中，旧run只读；保存parent HEAD和本次源码文件hash。
- 主配置、选择锁、数据清单、阶段任务清单冻结后再提交。

### 新旧证书桥接

旧R1/R4证书原件保持不变。新模块、128步调度、新seed和新配置需要新的V3执行锁与smoke，比较继承的数学/采样/LoRA契约并记录新增字段。不能编辑旧证书hash或放宽旧函数常量来让新协议冒充历史运行；也不能因为新配置hash自然不同就无限停在旧门禁。Codex应实现显式的V3兼容性审计与自己的真实性门禁。

## 3. 执行授权

本次ChatGPT只交付设计与参考测试，没有登录服务器、调用模型或提交job。

Codex完成新CLI与CPU验收后，先给用户：准确任务列表、two-worker安排、实测smoke时间/显存、预计总计算量、提交命令。首次新GPU任务须获得明确执行授权。若用户已明确授权整个新V3阶段，后续可按该阶段冻结清单执行，不要求每个样本重复确认；不能把其他旧实验的授权当作本轮授权。

不得根据结果正负跳过基线或停止不利方法。技术错误、模型身份漂移、原件破坏和配额不足会停止相应单元；科学空结果正常交付。

## 4. 新CLI契约

这些入口需要Codex实现和测试，不是旧仓库现成命令。

```bash
# 一律先无模型规划
"$PY" -m src.modeling_v3.cli plan-gpu \
  --config "$CFG" --out "$RUNROOT/plan_gpu"

# GPU桥接，使用已有原件绑定
"$PY" -m src.modeling_v3.cli vlm-smoke \
  --config "$CFG" --bindings "$BINDINGS" \
  --device cuda:0 --out "$RUNROOT/Q4/smoke_A" \
  --allow-gpu --acknowledge-new-experiment

# 每个源轨迹独立作业/输出目录
"$PY" -m src.modeling_v3.cli train-source \
  --config "$CFG" --bindings "$BINDINGS" \
  --seed 41001 --arm X_BASE --device cuda:0 \
  --out "$RUNROOT/Q5/source/seed41001/X_BASE" \
  --allow-gpu --allow-training --acknowledge-new-experiment

# 从已测源状态制作scratch候选
"$PY" -m src.modeling_v3.cli make-forks \
  --config "$CFG" --checkpoint "$CHECKPOINT" --bank-plan "$BANKPLAN" \
  --device cuda:0 --out "$RUNROOT/Q5/forks/$ORIGIN" \
  --allow-gpu --acknowledge-new-experiment

# 生成/评分职责可按冻结任务清单分片
"$PY" -m src.modeling_v3.cli observe-vlm \
  --config "$CFG" --task-manifest "$TASKS" --worker-index 0 \
  --workers 2 --device cuda:0 --out "$RUNROOT/Q5/observations" \
  --allow-gpu --acknowledge-new-experiment
```

`--resume`只能在当前run身份与源码锁一致时使用。CLI不自行调用sbatch，也不在import时加载模型。Slurm模板从已验证模板渲染，替换为当前准确CLI；不要交付不可运行的REPLACE_WITH_HASH脚本当完成。

## 5. 工作量和阶段化调度

主配置源训练：12seeds×2arms×128steps，合计3072次主Adam、98304条训练输出。
响应原点：24个X_BASE主节点＋6个X_VALID移位节点，共30。
每原点36个bank×3候选=108次scratch Adam；全部3240次。它们不是提交到源轨迹的新训练步。

探针36题，每原点主总量1024的共享origin proposal只有36864条新生成；但是在未去重情况下108候选需要约398万条“序列评分”。**评分不是免费，也不等于只做398万次单token forward。** certified prefix路径会按长度产生多次forward，必须实测。

参考主packet可共用生成：4096/题对应每原点147456条输出。两项MIX spotcheck另需每原点294912条；alias、跨目标共用和只对需复核单元追加可以降低实际量。若为全部heldout两对比都独立做MIX参考，则每原点3538944条，这不是默认做法，仅作为未共享的上限对照。

不得一次把这个上限笛卡尔乘积全部提交。顺序是：

1. Q4两个开发原点的小规模桥接，确定实际每条生成/评分成本；
2. 3个development seed的源训练与所需原点，完成观测与选择规则开发；
3. 锁定方法后完成3个calibration seed；
4. 最后解锁6个test seed，以及相应6个X_VALID移位原点。

同一阶段内部并行，最多两卡；估计吞吐只用于安排分片和墙钟时间，不将“比想象慢”转成科学失败。若总时长很长，清楚报告，不静默删掉方法、seed或参考精度要求。

## 6. 双卡任务调度和原子交付

先生成不可变 `tasks.jsonl`，每条任务有原点、candidate、prompt集合、role、draw区间、预期输入hash、输出路径和request keys。

由CPU规划器按任务hash/测得成本分到两个worker。每个worker独占自己的输出shard，不能两个进程同时写一个samples.jsonl。重启只补缺失request keys；失败的partial shard保留。

两卡之间用checkpoint/score shard的明确文件引用或直接本地通信，不通过GitHub上传权重和原始轨迹。两卡位于不同节点时，读写共享文件系统的原子语义和带宽必须测量。

同candidate连续评分一组requests，避免每条样本reload权重。可先分别对A/B整批评分，CPU再构造log-mixture。缓存只能复用相同inference fingerprint、相同input和token；不同Adam但相同参数允许推理alias，不允许训练resume alias。

## 7. 原始证据与发布

必须保留：
- raw completions、token IDs、每token logp、四事件标签；
- proposal source、pilot/main/reference独立RNG与角色；
- 实际候选e或足以精确重建e的checkpoint；不能只留范数；
- 完整source checkpoint恢复状态及参数/Adam/RNG hash；
- 每次评分的模式、生成上限、EOS、视觉input与source code hash；
- 高精度reference的批次级估计、SE、尾部及MIX交叉检查；
- forward/backward/Adam次数、生成/评分token数、GPU时间、显存、I/O。

公共仓库只提交代码、配置、经过审查的紧凑表及manifest。大原件保留私有storage；交接包必须包含完整的“原件索引＋具体复算命令”，不能再只有几张聚合表。

完成标记顺序：数据shard结束 → 所有预期request keys核对 → 文件hash manifest → 汇总表 → 完成标记。任一旧数据hash改变即拒绝续跑。

## 8. 结束时交给用户的内容

1. 实际代码commit和diff、工程测试结果；
2. 两GPU每阶段已运行／未运行任务清单与日志；
3. 分层科学结果：观测、方向覆盖、低秩、真实响应、离线窗口；
4. `MODELING_DECISION_V3_zh.md`：哪个方案可以用、在哪些事件/节点不可辨识；
5. 下一步是否有足够理由做实时/事件触发检测；没有资格时不得自动开始在线SSVC。
