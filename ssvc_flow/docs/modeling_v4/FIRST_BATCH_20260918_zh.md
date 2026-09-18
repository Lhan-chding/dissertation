# V4 首批测量与切换清单

## 当前决定

根据用户 2026-09-18 的要求，先交付小批完整测量，再决定扩展。
本次为独立的探索性测量，不改写原 C 阶段 4 原点、每原点 48 bank 的冻结定义，
不把首批结束标记成完整 C 完成。原 source、候选、样本、Adam 状态与旧代码保留。
用户于 2026-09-18 明确授权停止旧实验、保留已完成数据并启动本首批实验；切换已执行。

## 固定首批

- 原点：development seed41001 / X_BASE / step32。
- 从已完成 calibration_000..007 中，按原编号选首个主对比精确别名及首个非别名正范数组；不读概率响应选择。
- 实际选中 calibration_000（主对比 norm=0）和 calibration_001（norm=0.003309377317662962）。
- 两组的 joint_0、joint_1、no_x_off_1 原候选全部保留；不新做 Adam 更新。
- 从冻结的首24题中按 family/interface 每层取首题，共6题，顺序和完整题目记录写入 PLAN.json。
- 每题 work、reference、direct_work 各64次，主非别名对 MIX 64次、两个端点各直接生成64次。
- 保留已有重叠诊断规则；若次对比触发预定重叠条件，补该题的 MIX 测量并记录，不能按结果好坏增删。
- work/reference/direct/MIX 使用不同 RNG 流；preview 的观察原点命名空间与原 C 分开，不把相同种子重复当独立数据。
- 固定使用 uncached_prefix_recompute 评分，继续核验输入、token/EOS、策略身份、原始训练 dtype。
- 每完成一题立即保存原始样本、候选评分、DIAGNOSTICS.json、COMPLETE.json，更新 FIRST_RESULTS_zh.md。
- 六题结束后退出，无自动后续作业、无新增种子、无在线控制。

首批回答“概率变化及其测量误差是什么”。不拟合预测器，不使用 query/test 标签，
不将校准组改称测试组。后续预测验证仍须保留独立测试、强直接测量基线和非退化维数比较。

## 产物和限制

服务器输出：
`/projects/varunssd/louis-ssvc/modeling_v4_20260916/preview_20260918/`

入口 `PLAN.json` 冻结旧候选来源及新测量来源，`FIRST_RESULTS_zh.md` 是逐题更新的读数表，
`prompts/00/COMPLETE.json` 是第一题完整测量提交；`LATEST.json` 仅表示运行进度。
原 campaign 的 worker LATEST 可能滞后，不能据此判定当前阶段。

表中概率差以百分点报告，ORIGIN/MIX 的经验标准误差、COUNT 的保守单事件95%区间分列。
零经验方差不代表零真实误差；条件支持界仍以记录评分准确为条件，不包含 BF16 系统误差上界。
逐题区间不代表六题同时覆盖保证。小批样本不自动满足完整方案的参考精度。

## 基于已完成桥接的运行估计

不是新的性能探针，也不是已运行首批的实测：历史 B_origin_0 在12题/每流64次下，
真实生成与评分阶段耗时60,658.73秒（约16小时51分），生成4,608条回答。
本次6题主路径生成2,304条；若唯一的非别名次对比每题均触发 MIX，上限为2,688条。
候选只有两组，评分会按完整推理 fingerprint 共享。

用历史整段测量耗时作粗略量级参考：第一题约1–2小时，六题约8–12小时；
题目长度、候选评分和重叠补测会改变速度，不保证该区间。
Slurm 申请24小时是时间上限，不是预计工期；排队时间另计。
首题实际完成后，用真实耗时报告，不启动额外性能测试。

## 准备与准确启动命令

代码部署目录：
`/projects/varunssd/louis-ssvc/modeling_v4_20260916/code_preview_20260918/ssvc_flow`

只读导入元数据、冻结新预览清单，不启动模型：

```bash
cd /projects/varunssd/louis-ssvc/modeling_v4_20260916/code_preview_20260918/ssvc_flow
/projects/varunssd/louis-ssvc/envs/ssvc-py312/bin/python -m src.modeling_v4.measurement_preview prepare \
  --tasks /projects/varunssd/louis-ssvc/modeling_v4_20260916/tasks_initial.json \
  --collection-code /projects/varunssd/louis-ssvc/modeling_v4_20260916/code_161c591/ssvc_flow \
  --out /projects/varunssd/louis-ssvc/modeling_v4_20260916/preview_20260918
```

用户确认切换后，先结束旧批量作业的 allocation（仅这两个 SSVC 作业），
保留全部已提交原件；未完成的当前候选不计作完成，未来如恢复可利用已提交分块。
这不是删除实验，也不是让暂停的进程继续占卡：

```bash
scancel 158482 157057
```

确认它们均退出后，仅提交下面一个 GPU 作业：

```bash
sbatch --parsable \
  --partition=cluster02 --account=rose --qos=soujanya-poria-startfund-2026-03 \
  --gres=gpu:pro6000:1 --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=48G \
  --time=1-00:00:00 --no-requeue --job-name=ssvc-v4-first-batch \
  --output=/projects/varunssd/louis-ssvc/modeling_v4_20260916/preview_20260918/slurm-%j.out \
  --error=/projects/varunssd/louis-ssvc/modeling_v4_20260916/preview_20260918/slurm-%j.err \
  /projects/varunssd/louis-ssvc/modeling_v4_20260916/code_preview_20260918/ssvc_flow/scripts/modeling_v4_preview.sbatch \
  /projects/varunssd/louis-ssvc/envs/ssvc-py312/bin/python \
  /projects/varunssd/louis-ssvc/modeling_v4_20260916/code_preview_20260918/ssvc_flow \
  /projects/varunssd/louis-ssvc/modeling_v4_20260916/preview_20260918 \
  /projects/varunssd/louis-ssvc/modeling_v4_20260916/preview_20260918/PLAN.json
```

如果之后确需恢复同一预览，在完全相同命令末尾加 `--resume`；不自动重新提交。
切换前再次核验作业身份，不对其他用户或项目作业操作。

## 本次实现验证

- 受影响的采集、测量区间、支持界、预览选择/身份/恢复测试：47 passed。
- 新增实际小型 Torch 端到端采集、旧 checkpoint/新命名空间、诊断及恢复测试：1 passed。
- Ruff、脚本语法及 scoped diff 检查。
- 这些是本地 CPU 实现验证，不是新9B实验结果；未重复全仓测试。

## 实际切换回执（2026-09-18）

- 旧作业 158482、157057 于 14:41:54 UTC（新加坡22:41:54）按用户授权取消。
- 停前记录2082个已完成结果及其引用文件；停后全部仍在，文件大小、修改时间、inode相同，JSON元数据字节哈希相同。
- 检查没有重哈希大模型权重或全部原始样本，不能把文件状态核对描述成一次新的全量内容哈希审计。
- 两路源训练各128步、38个新候选组（27+11）及已完成桥接数据保留；正在进行的未完成候选不计入38组，没有删除其产物。
- 新作业 **159870** 于14:42:53 UTC（新加坡22:42:53）启动，状态已核实为RUNNING。
- 节点gpu-pro6000-5，账户rose，QoS soujanya-poria-startfund-2026-03，gpu:pro6000:1；调度器实际分配4 CPU、33 GiB主存。
- 代码提交：0002af0460d219d11d1479279ccaed4171bbf964。
- 冻结清单哈希：4ca7bd635dbe60023012d3aca2ca50b1725d6b06d52f33dc27497146879df684。
- 测量源码哈希：51b935d5a7852abe30e07307972e104af0826232cd0114ac8809f54e7dc563b7。
- `preview_20260918/LAUNCH.json`记录实际提交参数、授权和旧作业预期取消；`PRESERVATION_BEFORE.json`、`PRESERVATION_AFTER.json`记录保留检查。
- 本地小回执镜像：`ssvc_flow/reports/modeling_v4/preview_20260918/`；原始大数据仍在服务器。

作业启动不等于首题测量完成。监控每小时只读检查一次，在首题完整测量、六题完成、失败或需要用户决策时通知。
