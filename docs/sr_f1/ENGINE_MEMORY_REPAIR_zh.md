# SR-F1.1 ENGINE 显存故障与技术修复

## 已核实故障

2026-10-10 03:34（新加坡时间）的监控发现 ENGINE GPU 作业196085以 `FAILED / 1:0` 结束，控制器196046以 `FAILED / 2:0` 结束。两者均已退出队列。独立格式确认此前为500/512，`PASS_95`；此次故障发生在 ENGINE 工程验收，科学训练尚未开始，TEST继续封存。

原始子进程日志为服务器新运行根下的 `engineering/engine/natural/logs/continuous-517106.log`。首条梯度序列的 prompt 为1023 tokens、completion为259 tokens；梯度路径完成248次模型前向后，在第249次模型调用的 eager attention `repeat_kv` 分配处发生 `torch.OutOfMemoryError`。日志记录GPU容量95GiB、PyTorch已分配91.67GiB、已保留未分配2.57GiB、可用18.25MiB，下一次申请20MiB。此前无梯度参考路径完成259 tokens。该路径保留逐token原生缓存和完整反向图，原实现将其反向传播保存张量持续留在GPU。

故障时实际优化器更新为0，已保存128条首批rollout和完整第0步检查点。CPU独立审计作业196123为 `COMPLETED / 0:0`，核验全部原始槽位、种子、记录hash、采样logprob、输入/图像/策略/运行身份、完整参数/参考参数/Adam/调度器/RNG字段以及checkpoint文件hash，结果为 `PASS_ZERO_UPDATE_FULL_STATE`。审计未调用模型、未初始化CUDA。

- 原检查点文件SHA256：`0642ca038a0e99b1bdda1276b28f9f8a2eca06184f7072142d3d8b3eb2b21632`
- 原完整状态hash：`6ebf19f2a38638988671ac8591de83d770753f42de25c51dfcac3940ee192c33`
- 原源码提交：`023a56699a6faad7b7b2521706996b82b0c61496`
- 新实验冻结SHA256：`c7cf051d3aa6fb45889324e64b08d31a008137668aff2382283e4e19f99ff2b8`

服务器 `technical_incidents/engine_memory_20261010/evidence/` 保存337项原始文件，合计15,942,920字节，逐文件核验复制前后SHA256；含失败日志、原回答、完整检查点、调度记录及冻结回执。`code_before_engine_memory_20261010/` 保存原源码。原冻结、原QoS修复回执及旧SR-F1运行根不改写。

## 技术变更边界

梯度前向期间，使用PyTorch saved-tensor hooks把反向传播保存的非参数张量按原dtype复制到CPU，反向使用时恢复原device。参数storage的视图继续驻留GPU，并检查参数版本；避免逐token复制同一模型权重而耗尽主机内存。策略名为 `saved_activation_cpu_parameter_storage_resident_v1`。原生forward/backward、可微循环缓存、损失、梯度累加和两次反向流程不变；不截断历史图、不缩短completion、不改采样或学习率。保存与恢复字节数写入技术记录。

本实验GPU worker申请384GiB主机内存、单GPU及老师QoS `soujanya-poria-startfund-2026-03`。CPU worker资源不变，不增加账号范围GPU上限，也不修改无关作业。

独立 `ENGINE_MEMORY_REPAIR.json` 绑定原冻结、原QoS回执、原源码快照、故障证据、CPU检查点审计以及新源码身份。生产代码变更仅限runtime、ENGINE恢复、调度恢复、入口及冻结链验证五个文件；数据、奖励、五臂×三seed×96步、768-token上限、lr=1e-4和TEST释放门禁不变。

恢复仅消费一次经过核验的第0步状态与128条首批回答。原失败整段保留，恢复目录不带旧PROCESS记录。恢复的连续路径仍须在一个新进程中完成4次更新；拆分路径仍须在两个进程中完成2+2次更新。ENGINE原有概率、梯度、参数、Adam和RNG验收门槛保持。若恢复再次中断，不能把这次预置恢复授权重复消费。

## 验证与当前边界

本地FP32与BF16回归检查已核对原路径与CPU保存张量路径：逐token logprob和两次反向传播的所有LoRA梯度逐位相同，保留历史token梯度，参数变异检测及失败计数有效。第0步及128条原回答已完成服务器CPU验证。全套测试、部署身份、实际GPU进度和ENGINE最终验收分别记录于独立验证回执；CPU测试通过不代表真实9B GPU验收通过。
