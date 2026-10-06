# 4–5张PRO6000执行手册

**本文件定义待实现的CLI合同，不是已经安装好的运行工具。** Codex先完成新模块和针对性验收，再给服务器准确命令。不要复制旧冻结推理CLI去开启训练。

## 1. 一次性准备

在独立worktree建立`src/verified_discovery_transfer/`。从当前完成协议探针的代码继续；记录实际HEAD、Python环境、模型revision、LoRA白名单一次。优先沿用现有工作环境，不自动pip升级torch/transformers/peft。

把`templates/machine.example.json`复制为实际machine配置。检查S96、REP96的历史COMMIT和state路径能否读取；`sources/checkpoints_historical.json`仅证明导出时记录过这些位置，不能替代当前检查。

只在开始恢复每个父状态时核对参数名、shape、dtype、必要forward状态和模块数。推理以后所有参数冻结；SFT必须重新开放精确的原LoRA白名单，不重新初始化。固定teacher对象不与student共享可变张量。

## 2. 推荐CLI合同

以下命令均在仓库`ssvc_flow/`目录运行；参数名由Codex按此实现后确认，不可声称此包已经能直接启动GPU：

```bash
python -m src.verified_discovery_transfer.cli preflight \
  --plan /path/to/protocol.json --machine /path/to/machine.json --out /path/to/run

python -m src.verified_discovery_transfer.cli prepare-data \
  --plan /path/to/protocol.json --machine /path/to/machine.json --out /path/to/run

python -m src.verified_discovery_transfer.cli bridge \
  --plan /path/to/protocol.json --machine /path/to/machine.json --out /path/to/run

python -m src.verified_discovery_transfer.cli build-queue \
  --plan /path/to/protocol.json --machine /path/to/machine.json --out /path/to/run

python -m src.verified_discovery_transfer.cli worker \
  --queue /path/to/run/queue.sqlite --worker-id lane0 --device cuda:0

python -m src.verified_discovery_transfer.cli report \
  --run /path/to/run --reveal-test-after-completion
```

采用Slurm时，每个worker申请一张实际分配的GPU；程序内部使用Slurm映射后的`cuda:0`，不要把节点物理卡号误当可见编号。partition、qos、account、conda路径从实际可用配置取得，本包不猜测当前授权。

## 3. 队列的依赖关系

- 数据/角色冻结 → teacher V/T/E可执行。
- SFT数值与mask验收 → GOLD_ALL/REPLAY_ONLY可先跑，不依赖发现完成。
- teacher V完成 → 固定各parent的b16单协议、b2单协议/组合。
- teacher T repeat完成＋V选择冻结 → 对应SELF数据view和GOLD_MATCH可构造。
- view构造完毕 → 严格alias检查 → 学生任务入队。
- 学生达到64/128 → V诊断与训练sentinel；达到256 → E/G任务。
- E/G可计算封存，但结果不得被选臂/调参进程读取。
- 所有登记主模型完成或记录明确技术缺失 → 最终统一揭晓。

空发现集不产生学生训练作业，记录返回父模型并以已有父样本作final别名；这仍是一个完成的端到端分支，不能从平均中删除。

## 4. 五卡安排

起始：两卡分别常驻S96与REP96做V和T；其余卡做SFT桥接与GOLD_ALL。teacher或学生任务完成后动态领取队列，不固定浪费空闲卡。

S96 repeat0的SELF_O0/MIX/SINGLE/ALL是第一批完整比较。先汇报其发现、训练与V诊断，不提前揭E挑选下一步。如果两个SELF视图严格相同，合并任务并保留逻辑别名，不人为制造不同随机种子。

四卡时只减少并发。优先一GPU一模型，不先采用4/5卡分布式，避免把独立方法比较变成昂贵同步作业。只有实际显存不足、且微批调整不能解决时，再讨论模型并行；不能静默更换量化模型或LoRA秩。

## 5. 两种模型执行路径分别验收

### 冻结生成

复用已完成的raw-only、uncached-prefix路径，保留64token上限、无thinking、T=1、top_p=1、top_k=0。各次请求重置position/cache，不继承上一条输出。模型返回后先按登记顺序逆排列，再公开验证，全部失败也入库。

短期不引入vLLM等新引擎；若以后桥接加速，需另行记录其生成分布和数值条件，不能一半方法新引擎、一半旧引擎。chosen-token logp若原路径已经得到即可保存，不另跑整套候选评分。

### SFT

使用全序列causal teacher forcing，新模块显式计算completion-only NLL。它不要求旧PPO ratio=1，但必须证明target shift、future mask、attention隔离、EOS和梯度累积正确。

技术桥接仅用train-only固定8题、最多128次生成。先做CPU小模型mask测试，再真实Qwen单例/批量loss、分microbatch梯度、future-suffix不影响早期logit测试。通过以后所有臂同一路径。

若prefix/full路径在BF16下不同，报告实际每token误差及梯度关系，不把0.16之类旧差值直接认定“允许”；也不制定脱离数值基线的全局1e-12要求。真实计算模式、核和cache因果性有问题时先排查，不能用旧记录的“曾通过”代替新目标验收。

## 6. 精确的累积与scheduler

SFT每步12focus＋4replay。微批4时可以3个focus微批＋1个replay微批，每个实际序列loss系数1/16；REPLAY_ONLY只执行最后4例仍除16。降低微批保持这些系数，全部backward完后统一clip和step。

第k次更新使用`lr=1e-5*min(k/8,1)`；k从1开始。不要通过scheduler调用顺序使第一步意外使用0或完整lr。checkpoint保存已提交update索引，恢复时下一步为k+1。

R0_RESET32同样8步warmup，但采用旧on-policy loss而不是SFT；每组K8、B4。遇全零优势可跳过反向图，仍按照明确零梯度执行Adam/scheduler一次，不能拿`grad=None`代替`grad=0`而改变momentum语义。

## 7. 可恢复输出与缓存

每条原子生成请求有稳定ID：parent、split、task、protocol、role、repeat、draw、模板版本。新任务写临时chunk，完整后原子提交；恢复只认已提交记录，失败attempt保留但不合并重复行。

同一请求已完成可复用；不同协议、不同teacher或新权重不复用生成。相同规范答案也不能作为生成缓存键。

SFT保存LoRA、fresh-AdamW状态、scheduler、sampler/cursor、全部RNG、forward必要状态和训练view。基座只引用revision，不反复复制。checkpoint命名含parent/arm/repeat/step，学生与teacher严格分目录。

不要每条样本fsync整棵目录、每步重算全模型hash、每个分支重跑全仓测试。首次数据导入查契约；阶段结束查数量、唯一请求和文件可读性；崩溃恢复时查对应chunk。

## 8. 吞吐、存储与进度

`workload.json`是所有臂互不alias时的上限。代价分开记录：生成token与GPU秒、teacher评分（预期新增0）、SFT processed targets/token、R0 rollout与更新、evaluation。

symbolic和image单独测；SFT full-sequence和旧GRPO更新也分开测。先用代表性完成任务给预计剩余时间范围，不承诺5张卡一定在某天完成。

模型checkpoint仅留服务器，交接包带路径与必要metadata；全部新样本、规范世界和失败类型用压缩JSONL。旧大包只引用，不重复打包三份。

若磁盘不足，只停止新的写入任务并报告路径/空间；不可为腾空间删除未导出的原始数据或唯一checkpoint。

## 9. 何时停、何时继续？

立即处理：数据/验证契约不一致、E泄漏、恢复错模型、数值非有限、无LoRA梯度、future-token泄漏、错误逆排列。

继续并记录：某协议表现很差、某家族没发现成功、PTLC不匹配、GOLD不提升、区间很宽。结果不好不是修改指标、扩大步数、增加一个新奖励的授权。

所有结果出来以后再决定是否需要失败条件路由、SFT后RL、在线权重调节或新协议。它们不在本轮默认队列。
