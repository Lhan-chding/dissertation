# R0 闭合与 R1 定向补充

本轮继续使用 `CODEX_NEXT_STAGE_zh.md` 的阶段顺序。已完成的 R1 作业
146571、原始 P3-N/L 和旧 R0 审计保持只读；新证据写入独立目录。

## R0

入口为 `python -m src.finalize_r0`。旧脚本曾把 calibration 的 36 张人工
检查图与仅包含 dev 场景的 P3-N 记录连接，因无交集而记为 INCONCLUSIVE。
新入口分别执行 calibration 图像检查和原始 P3 场景输入重放。

- 重新验证旧 R0 和 P3 的 manifest、文件大小和 SHA-256；复用已绑定的
  分类、独立求解器、组统计和场景 bootstrap 证据。
- 按数据 manifest 检查全部文件、七个必需 split 和各 split 行数，防止
  缺失整个 train/control/confirm 分片时仅扫描剩余文件而误放行。
- 重新验证 N=4,608 sampled + 288 greedy、L=2,816 sampled + 176 greedy，
  原始 record hash、唯一 sample key 和逐题样本索引。
- 从锁定缓存加载 tokenizer、processor 和 chat template，重现历史视觉
  像素上限并核对三者内容哈希；不加载模型权重。
- 对 N/L 全部原始 token 序列重新解码；按实际原始场景重建每个 prompt，
  比较所有原始记录的文本、token ids、输入 tensor、图像 tensor、grid、
  尺寸和视觉 token 数。复用同一 prompt 的 CPU 处理结果，逐记录比较。
- 检查 calibration 的 36 张图和 processor 实测值，并将原有人审文件的
  哈希、场景集合、contact manifest 与原始 P3/P1 引用绑定。
- 重查 split 碰撞和 SYMBOLIC prompt 的未声明标签。

所有检查通过才写新的 R0 PASS，旧 INCONCLUSIVE 文件不会被改写。
任何缺文件、哈希变化、输入重放差异或未覆盖的旧审计问题均阻止后续执行。

## R1 补充

入口为 `python -m src.r1_supplement --r1-run <146571目录> --out <新目录>`。
使用原始 calibration bank 和 step1 非空 Adam 的 `resume_probe.pt`，
核验其与样本生成时的 adapter、old likelihood、原始代码和配置一致。
不重新生成样本，不把 P3 样本用于训练。

补充检查包括整批零优势对应的显式零梯度与 Adam 动量/步数行为、独立
X_BASE 奖励实现与通用 λ=0 路径的一致性，以及恢复同一起点后反转候选
执行顺序的一致性。检查结束恢复初始 scratch adapter、空 Adam 和 RNG。

补充记录实际 backward、optimizer step、顶层 forward 和 backward 期间
语言层重算调用。内部层调用与顶层调用分开计数。CPU fixture 只用于软件
回归；真实 GPU 结果必须由独立服务器产物证明。

此前 batch/cache 优化的失败记录保留，生产路径仍为经验证的逐样本
`uncached_prefix_recompute`。这次补充不宣称真实 batched model forward
通过，不采纳未通过数值检查的 batch/cache 路径。

## 顺序执行

`scripts/ntu_r0_r1_close.sbatch` 申请 1 张 PRO6000、6 小时，先执行 R0。
R0 返回非零时停止；通过后再加载唯一模型执行 R1 补充。每个作业尝试
使用独立目录，失败不会覆盖旧结果，退出时生成证据压缩包及 SHA-256。

R2 与 R3-cold 的真实执行仍依赖 R0/R1 结果验收；该脚本不启动它们。
R4 与 R3-warm 继续遵循后续顺序，完整在线 SSVC 未启用。
