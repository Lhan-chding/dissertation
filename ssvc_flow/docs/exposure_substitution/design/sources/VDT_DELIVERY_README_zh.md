# SSVC 可验证协议发现与 O0 监督回迁：最终证据包

本目录为现存输出分析包，不是可独立训练重跑包。冻结实验源码为 ed1d24839dc6bd374d17a92bd29678b7185c7605；运行身份为 6e93b2305fdf089109b918945742f962d9f776ba2c432674504f14deb478ed27。

## 读取顺序

1. FINAL_FINDINGS_zh.md：发现、拟合、迁移、保持性结论和限制。
2. run_v2/TEST_RELEASE.json、EXECUTION_SCOPE.json：统一揭晓及完整矩阵范围。
3. run_v2/final_analysis：三组补充分析的 Markdown、CSV、JSON，均为已揭晓现存输出的离线统计。
4. run_v2/FINAL_COMPARISONS.json：冻结代码生成的预登记主比较与关键次比较。
5. ssvc_training_prerelease_audit.json、postrelease_atomic_audit.json、r0_rollout_audit.json：独立技术核验；PASS 不代表科学假设成立。
6. frozen_source_ed1d248.tar.gz：冻结实现、测试、设计文件。ssvc_*analysis.py 和 statistics_frozen.py 是离线分析代码，不改变实验运行身份。

## 收录与省略

收录 T/V/E/G/R 清单、公开输入及已揭晓审计标签、全部正式回答 chunk（含失败回答）、R0 rollout、逐步训练指标/尝试日志、发现与选择记录、恢复回执、Slurm 日志、发布与分析输出。

未收录模型权重、113 个 .pt 检查点、本地/服务器依赖环境、逐调用重复 atomic 包装、临时 partial/tmp 文件、锁文件。它们没有从服务器删除；检查点文件 SHA-256 与完整恢复状态验证载于审计回执。chunk 保存正式原始回答与请求字段，逐调用包装的一致性另有核验。中断前未能持久化的计算成本不可由这些输出补造。

服务器保留目录：/projects/varunssd/louis-ssvc/verified_discovery_transfer_20261006/run_v2。

## 解释边界

两父是已有来源轨迹，repeat 嵌套在父内；不能当作四条独立训练谱系。场景 bootstrap 条件于这两个父和当前冻结数据分布，不是跨父总体推断。有限采样未成功不等于真实成功概率为零。G 图像与 duplicate 分层，并以共享 base scene 对齐 E/G；不并入一个最终总分。

原始 BF16 桥接 NUMERICAL_REVIEW_REQUIRED 保留。接受的是共同固定 full-sequence microbatch=4 路径，不证明旧 prefix 路径等价。
