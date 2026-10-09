# SR-F1 当前执行状态

更新时间：2026-10-09（Asia/Singapore）。

状态：`LOCAL_IMPLEMENTATION_VERIFIED_SERVER_PREFLIGHT_PENDING`。科学训练尚未开始，没有本轮能力效果结论。

- 原 ZIP 安全路径、CRC、76个包内哈希通过；原包111项测试通过；10项清单重建逐字节匹配。
- 新SR-F1训练、评价、恢复、调度和释放入口已实现；最终完整本地检查见 [FINAL_LOCAL_CHECKS.json](validation/FINAL_LOCAL_CHECKS.json)。这些测试不替代实际GPU概率/梯度/恢复门禁。
- 本地实际渲染2890张图及1张白图，最小主标签字高13px；原生服务器processor仍待实际运行。
- 已取得并校验ChartQA全量2500题、1509唯一图像，固定作者数据revision与字节hash，尚未模型评价。
- 服务器固定9B revision目录存在，获批账户/QoS已实时读取；逐权重字节及完整processor验收仍待正式作业。
- 源码上传首次被自动审批拒绝；用户随后明确授权上传及继续执行，部署已成功。
- CPU预检195389最初提交rose，排队预计次日；Slurm拒绝原地更换QoS，已仅取消该尚未运行作业（CANCELLED、Elapsed=0）并保留回执。将按已获批teacher QoS重提同一逻辑预检，不重复并行运行。
- 无旧F2结果参与起点/超参数选择；旧F2及SER-J23均未恢复或改动。

服务器运行根：`/projects/_ssd/varunssd/louis-ssvc/sr_f1_20261009`。
独立分支：`codex/sr-f1-20261009`。

## 放行顺序

实际CPU处理器/权重核验 → 执行冻结 → 公共原模型FORMAT及必要的一次桥接 → ENGINE真实4对2+2 → 固定MONITOR基线 → 15条96步训练 → 全部注册评价 → 原文独立重评分与统一中文报告。

72小时是Slurm租约，完整状态续跑；没有GPU累计时长或研究墙钟停止阈值。训练总步数和注册题流仍不可延长。
