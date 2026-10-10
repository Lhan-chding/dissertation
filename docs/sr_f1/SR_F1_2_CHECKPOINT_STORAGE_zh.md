# SR-F1.2 后续种子检查点存储预置

2026-10-10 22:57巡检发现SSD真实Ceph目录额度为750,000,000,000字节，已用约714.62GB，余量约35.38GB。单份step>0完整检查点实测347,111,205字节，step0为173,569,979字节。当前四臂均已提交step32；按固定每8步保留完整状态推算，剩余12路径检查点约需45.82GB，尚未计入adapter、原回答及最终评价。全局statvfs空闲不能替代目录配额。

这不是当前训练失败。为避免后续写盘不足，仅为尚未提交的71002/71003八条路径预置`runs/<model>/checkpoints`目录链接，目标固定为`/projects/_hdd/varunhdd/louis-ssvc/sr_f12_20261010_checkpoints/<model>`。预计约34.71GB未来检查点写入HDD。现有71001路径、全部已生成数据、四个GPU及CPU控制器保持原状，不删除、不搬动现有科学结果。

科学配置、注册、冻结、生产源码6cd2、采样顺序、优化器和GPU数量均不修改。整目录链接使生产临时写入、fsync、hardlink、原子rename和commit/LATEST位于同一文件系统。恢复、主任务完成和端点验证以解析后的checkpoint目录为边界，仍检查字节、完整状态及字段哈希；内部逃逸路径仍被拒绝。

操作入口归档于`validation/f12_ops/prepare_future_checkpoint_storage.py`：仅CPU Slurm环境运行；核对四份冻结/源码清单文件及458项源文件、真实SSD/HDD额度、八条路径无注册/队列/任务目录/输出目录；写不可变INTENT后做真实HDD合成commit/load、完整Adam/调度器/cursor/RNG恢复及篡改拒绝，再次核对未启动状态后创建八个空目录链接，写PROBE/APPLIED。遇部分执行须审查回执，不得盲目重跑或重指已有链接。

本地针对目录链接完整恢复、八条外置检查点的末端验证及原训练/端点相关测试：68项通过。真实HDD验证和实际预置结果以服务器`deployment/checkpoint_storage_20261010/{INTENT,PROBE,APPLIED}.json`为准；CPU测试通过不等于科学训练完成。

后续监控、故障保全和交付必须显式遍历APPLIED中八个目标目录，通用`runs.rglob`可能遗漏符号链接中的内容。持续检查SSD与HDD真实额度；本措施不能保证同目录其他用户工作负载未来不再消耗额度。

实际执行：普通CPU197229在15:04:33Z启动、15:04:41Z COMPLETED0:0，排队18秒；68项服务器测试和真实HDD探针通过，八个空链接完成并独立核验。原生产和冻结哈希不变；15:05:32Z四个原GPU均继续RUNNING。不可重复执行脚本。完整证据见[2305核验回执](validation/SR_F1_2_CHECKPOINT_STORAGE_20261010_2305.json)。
