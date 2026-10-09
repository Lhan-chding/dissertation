# 本轮用户执行修订

本文件记录用户对原始 MM_CORE_CODEX_FINAL_20261008.zip 的直接补充，优先于包内对应条款。
原始 `design/` 文件及其 manifest 保持字节一致，不能用旧原文覆盖本次明确指示。

- 资源按最多同时五张 NVIDIA Pro 6000 考虑。单个分片或训练接口使用一张卡；
  实际分配仍服从账户、QoS、分区和现场可用设备。
- GPU 小时不设置为限制或门槛。`max_allocated_gpu_hours=null`，
  `gpu_hours_policy=ACCOUNTING_ONLY`。记录请求时限估计、实际设备数乘占用时长、
  加载/空闲/失败及排队时间；累计小时不决定启动、停止、通过或是否需要用户批准。
  调度系统要求的单作业 walltime 属于调度参数，并非研究总 GPU 小时门槛。
- 当前模型固定为 **Qwen/Qwen3.5-9B**，以实际服务器快照 revision 和逐文件哈希冻结。
  原始文档中的 Qwen2.5-VL-3B 不再是本轮目标模型。

模型变更发生在 MM-CORE 首次 GPU 作业提交和模型输出之前。新模型使用独立运行根
`mm_core_f1_qwen35_20261008`，此前 3B 准备记录仅作为历史保留，不继承其 processor
审查、预推理冻结或恢复认证。新运行重新验证原生图像 processor、模板及真实视觉路径。
固定 `enable_thinking=false`，继续采用同次生成的 readings/answer JSON 协议；不根据输出调参。

Qwen3.5-9B 为混合注意力结构。保持原约定的语言层 q/v LoRA：仅在 8 个完整注意力层
（索引3/7/11/15/19/23/27/31）的16个 `q_proj/v_proj` 模块施加 rank8/alpha16 LoRA；
原生 `q_proj` 同时包含 query 与 gate 通道，这一点在训练配方中显式记录。
线性注意力与视觉权重冻结，不将 fused 投影临时替换为另一套适配器。

本轮授权阶段及科学门槛保持原契约：资产/CPU 核验、MM-AUDIT、条件触发的一次16更新
公共格式桥接及必要的隔离 ENGINE 恢复检查。总生成尝试4096、物理更新64、额外评分前向8192
仍为上限。全部不利结果及技术事件保留，完成报告后 `STOP_FOR_REVIEW`。
MM-DEV、MM-LOCK、MM-CAL、MM-ONLINE 及新符号训练未获授权。
