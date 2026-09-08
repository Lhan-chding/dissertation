# 实现范围与验收口径

下表记录 2026-09-05 的本地交付边界：当时没有连接服务器、下载 9B 权重或执行 GPU 训练。
2026-09-08 的真实 NTU P1 失败诊断、修复和复验另见
[P1 实测记录](P1_FAILURE_144840_zh.md)。`local_evidence/` 保留原本地证据，不回写成 GPU 结果。

| 项目 | 本轮实现 | 尚未实测或缺失 |
| --- | --- | --- |
| P0 历史审计 | 旧源码、配置、数据、证据包只读审计；哈希核验；L/N 差异表 | 历史 checkpoint 权重与部分运行语义缺失，`legacy_exact_reproduction=false` |
| P0 新数据 | 3,632 个唯一场景；严格 parser；独立 396 候选 solver；纤维计数；原始图像；划分平衡 | processor 实际尺寸/视觉 token 待 P1；独立人工 36 图检查待记录 |
| P2 数学基准 | K=2/4/8/16/32 精确枚举，K=64/128 单独标记 Monte Carlo；有限步、ODE/Taylor、导数和 decoder toy | 缺失引用的 `THEORY_REVIEW`；自推定义与自建非闭合例不能冒充原文验证 |
| P1 兼容性代码 | 官方模型类、非 thinking 模板、语言 MLP LoRA、纯 softmax 采样、likelihood、实际 Adam 更新与恢复审计入口 | 9B CUDA、BF16、混合缓存、显存与吞吐必须到 NTU 实测 |
| P3–P9 | 保留实验设计与显式运行门禁；部分纯数学工具已实现 | 冻结评估、短训练、多种子确认、真实 optimizer-fork 科学实验、完整 FSA/自然接口、kernel/SSVC 控制器均未运行或交付 |

新协议所有执行路径均为仓库相对路径或显式参数。旧服务器路径只可能出现在只读历史证据本身，不作为运行默认值。

数据固定划分：calibration144、train576、control144、dev144、confirm288、ood288、natural_pool2048。
train 的两接口各288，按三家族×两图形×三运算精确平衡。所有真值数组跨 split 去重。
`natural_pool` 不要求三家族均衡，实际分配 duplicate880、cross-series880、trend288；原因是0–49中只有784个非恒定四项等差数列，须先满足其他划分并保持全局去重。
OOD 的图结构、数值平移、渲染扰动各96；图结构部分只含 cross_series，path/cycle各48。

P2 输出中的符号明确由本地实现定义：`U=(N/K)A`、`H_K=E[U]`、`Xi_K=E[UU^T]`、
`B_eta,K=E[softmax(log p+eta U)]-p`。`Xi_K` 为二阶矩。
`E[q(p+)]` 与 `q(E[p+])` 分开保存。缺少原文对应项使 P2 状态保留 `PARTIAL`。

本轮统计报告没有填充真实模型的概率表或训练表。对应表名、状态和原因记录在
`reports/table_status.json`，以免将零行、零概率、缺失支持和未执行混淆。

测试包含单元测试、模块集成测试、完整 CPU 命令流程及 fake/tiny 模型训练与恢复验证。
fake/tiny 结果只证明相应代码路径，不证明真实 9B 通过。测试覆盖率也不代替真实 GPU 验收。

P1 的数值监测在 GPU 测试前固定为 `max_empirical_sequence_kl=1.0`。
该量来自用于更新的同一批序列，是明确标记的经验代理，不是独立估计的总体 KL 或安全证书。
越界保留异常 checkpoint 并中止，resume 也必须继续报告该异常。
