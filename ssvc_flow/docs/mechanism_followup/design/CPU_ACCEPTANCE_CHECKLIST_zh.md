# CPU 验收清单

对应主文档 `mechanism-followup-v2`。每一项需要一个自动测试或一份可复核的命令输出；不能仅在报告中写“已检查”。

## 一、来源与范围

| ID | 测试/检查 | 通过条件 |
|---|---|---|
| P01 | 工作区与分支 | 记录 HEAD、dirty state；保留旧修改；未改 main |
| P02 | 旧源代码 | 关键模块与旧 runtime source hashes 对照；差异逐项说明 |
| P03 | 元数据身份 | warm phase、step64、模型 revision、B4/K8 与三份旧 JSON 绑定一致 |
| P04 | parent 路径 | 目录与受限 ZIP 两种只读读取；越界、重复、缺失、hash错均拒绝 |
| P05 | 状态分级 | compact metadata 通过不把 raw/checkpoint/GPU gate 置真 |
| P06 | 无 GPU 默认路径 | plan/audit/import/tests 不加载权重、不下载模型、不调用 sbatch |

适用文件与读取方式以当前源码为准。若读到原件缺失，测试“明确拒绝”行为；不得凭空创建看似真实的 parent 文件。

## 二、奖励与优势

| ID | 输入/操作 | 通过条件 |
|---|---|---|
| A01 | X/S/W/I 奖励 | joint0=(2,0,0,0)；joint1=(3,1,1,0) |
| A02 | all-valid：混合 X/S/W、多个 lambda | 加常数后优势不变；含数值容差说明 |
| A03 | all-X、all-W、all-I | 优势全部0；mean/variance/epsilon有限 |
| A04 | 4W+4I | lambda>0 时 W正I负；无 X 的 joint0 为0 |
| A05 | 4S+4I | no_x_off 同样关闭；不能误把 S 当 X |
| A06 | 4W+4I、epsilon=1e-4、微小lambda | 符合 lambda/sqrt(lambda²sigmaV²+epsilon²)；非真正不连续开关 |
| A07 | no_x_off 无 X 组 | 先把辅助系数设0再联合归一化；不是删样本 |
| A08 | no_x_off 有 X 组 | 等于相同 lambda 的 joint |
| A09 | composite | gate 按每个 group，不按整个 bank 是否存在 X |
| A10 | 同组排列、同bank组顺序 | 对应重排后的优势一致；顺序不影响统计 |
| A11 | 非有限值、未知类别、bool参数、空组 | 清楚失败，不产生 NaN 的伪结果 |
| A12 | reference 与仓库纯函数 | 原 joint 的奖励/均值/总体方差/优势一致 |
| A13 | 有限差分 | joint 优势的 lambda 导数与解析公式在非边界点一致 |

不修改 `epsilon_r` 来让微小权重产生预设结论。常量组保留明确的零方差分支。

## 三、真实 PyTorch CPU 梯度与 Adam

构造小型、可微分、无需下载的模型。应有多个可训练参数、冻结参数、buffer 和不同子模块模式；生成可使用 fake adapter，但梯度和 Adam 必须是真实 PyTorch。

| ID | 测试 | 通过条件 |
|---|---|---|
| G01 | 新 joint 对旧 direct_loss_gradients | 同组、同old分数、同模型时梯度一致 |
| G02 | 损失分母 | microbatch=1累积等于完整batch；分母固定B*K*64 |
| G03 | EOS/padding/截断 | 真EOS参与，padding不参与；不生成假EOS，不删I样本 |
| G04 | PPO clipping | 正负优势各测试 clip 边界及 boundary 外分支 |
| G05 | no_x_off | 被抑制组梯度0；其他组与joint一致；不改总分母 |
| G06 | 全零当前梯度+warm Adam | 仍执行一次step；可观察历史动量导致的更新 |
| G07 | None vs 0 对照 | 验证两种语义不同；生产使用显式0覆盖全trainable集合 |
| G08 | global clip | clip前后norm正确；固定阈值；未对每组单独clip |
| G09 | optimizer ownership | 重复参数、遗漏trainable、包含frozen时拒绝 |
| G10 | 非有限梯度/参数/state | 停止并保存错误，不写成功标记 |
| G11 | 同范数不同方向 | 例如(1,0)和(0,1)：范数同，向量差非零，cosine=0 |
| G12 | CPU FP64距离累计 | 向量范数、差、夹角与独立参考计算一致 |

CPU 纯数学先用 FP64；旧 production loss 本身有 FP32 转换时，新旧回归必须使用同一计算契约，不能比较不同精度路径后随意扩大容差。

## 四、完整状态恢复

| ID | 测试 | 通过条件 |
|---|---|---|
| F01 | origin A→candidate1→restore→candidate2 | 每次起点参数/Adam/RNG/模式均一致 |
| F02 | 候选执行顺序反转 | 每个候选结果保持一致 |
| F03 | lambda0重复 | 参数、Adam结构与统计一致；bitwise与allclose分开报告 |
| F04 | 注入中途异常 | 保存failure；finally恢复origin；不掩盖原异常 |
| F05 | buffer/position state | 显式恢复forward相关状态；缓存清理可验证 |
| F06 | 子模块模式混合 | 恢复每个子模块的training flag，不只恢复顶层bool |
| F07 | checkpoint往返 | weights_only加载；identity/hash/type/shape均校验 |
| F08 | optimizer步数 | 原64、候选65、未提交训练三项分开 |
| F09 | 参数相同但Adam不同 | 推理可标alias；训练checkpoint仍独立保存 |
| F10 | 恢复失败 | 状态变为不可继续；不得用下一候选覆盖故障 |

## 五、bank 与 prompt 绑定

| ID | 测试 | 通过条件 |
|---|---|---|
| B01 | 原bank0/3/4/5/11 | 按parent manifest装配；所有group K8，精确组构成匹配 |
| B02 | composite | 来源为3/2、11/2、0/0、0/1（零基）；原token与old分数不变 |
| B03 | 来源策略混合 | origin/model/parser/数据绑定不同立即拒绝 |
| B04 | control泄漏 | train与control场景重叠、重复prompt拒绝 |
| B05 | 样本完整性 | 重复sample key、缺样本、sample index冲突拒绝 |
| B06 | prepared input | prompt/tokenized/image/prepared hash可比且绑定正确 |
| B07 | 结果导向选bank | 选取函数不读取新的control/dev结果 |
| B08 | 旧结果只读 | parent目录内容运行前后hash不变 |

## 六、统计

| ID | 测试 | 通过条件 |
|---|---|---|
| S01 | 四类计数 | 总数与n一致；每prompt权重固定；未知/负计数拒绝 |
| S02 | ratio-of-weighted-means | qX=加权pX/加权v，不等于平均逐题qX |
| S03 | baseline方向 | right-left或candidate-base明确；单位probability/pp正确 |
| S04 | paired coverage | 候选prompt/scene/interface/seed block不匹配时拒绝 |
| S05 | 场景bootstrap | 同景两个接口共同抽样；家族分层；所有候选共同索引 |
| S06 | 全零/分母0 | 点估计与支持状态分开；q undefined；不输出假安全 |
| S07 | 固定面板边界 | 独立std参考验证Hoeffding半宽、union M与ratio传播 |
| S08 | 均匀16权重 | ESS=16、ESS/n=1、maxw=1/16、nmaxw=1；不因0.05误拒绝 |
| S09 | 稀有X未出现 | event support标UNOBSERVED，ESS高不能自动安全 |
| S10 | 极端logweights | log-space稳定；非有限输入拒绝 |
| S11 | OIS/SNIS/直接采样 | 名称与公式区分；不静默裁剪OIS |
| S12 | alias样本 | 不重复增加n、不计为独立策略/训练seed |
| S13 | 多seed汇总 | per-seed paired delta可复算；17标探索，29/41标前瞻 |
| S14 | 少场景区间 | 报告不可估计/不稳定情况，不通过删除异常bootstrap补出区间 |

固定面板区间和场景bootstrap须分开输出。S1不实现“只要置信区间过线就在线提交参数”的控制器。

## 七、端到端、CLI、恢复与预算

| ID | 测试 | 通过条件 |
|---|---|---|
| E01 | fake adapter完整S1 | 六bank候选→checkpoint→rawledger→counts→response齐全 |
| E02 | 真实语义I | 保存为正常样本；generation/parser执行故障另行失败 |
| E03 | 输出中断恢复 | 已有sample精确复用，不重复、不删除，不换seed |
| E04 | 恢复身份变化 | 变model/parser/config/origin/arm/seed等拒绝 |
| E05 | 文件no-clobber | 不覆盖已有completed或失败证据；目录逃逸拒绝 |
| E06 | 计划预算 | 41候选+6重放=47更新；直接逻辑18×48×16=13824 |
| E07 | 精确alias | 验证完整推理指纹，禁止norm/allclose去重 |
| E08 | S2 seed17排序 | 与旧R4 train schedule逐prompt/hash一致 |
| E09 | S2新seed排序 | seed29/41确定性，组间同seed顺序同、各自on-policy |
| E10 | S2 resume与评估隔离 | 评估不改变训练RNG/Adam/采样器；中断续跑一致 |
| E11 | CLI帮助/dry-run | 所有文档命令已实现；无GPU标志不能加载模型 |
| E12 | Slurm语法 | 生成脚本能bash -n；不执行sbatch |
| E13 | 小型多seed/多arm训练 | 至少2seeds×3arms的短CPU fake流程真实更新与读写链通过 |
| E14 | 旧相关测试 | 原GRPO/fork/optimizer/R3/R4输入与恢复测试回归，失败逐项处理 |

## 八、测试记录格式

每条命令记录：工作目录、Python路径/版本、库版本、git HEAD、dirty state、完整命令、退出码、通过/失败/skip数量、日志SHA-256。

```json
{
  "code_implemented": false,
  "cpu_math_tests_passed": false,
  "cpu_torch_tests_passed": false,
  "cpu_end_to_end_tests_passed": false,
  "legacy_regression_passed": false,
  "parent_metadata_verified": false,
  "parent_raw_verified": false,
  "gpu_smoke_passed": false,
  "gpu_started": false,
  "training_started": false,
  "ready_for_server_review": false,
  "blocking_items": []
}
```

此处 false 是空模板，不是当前运行结果。只能依据实际执行改写。全部本地项完成而仅缺服务器资源时，`ready_for_server_review` 可为 true，但 `gpu_smoke_passed` 仍为 false。
