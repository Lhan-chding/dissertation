# PILOT_V3 Round A执行

上游233935实际COMPLETED0:0/12:32，READY_FOR_REGISTERED_PILOT_ONLY；50份原件一次交付、内部引用核验完成。模型生产者固定b7e7b067f237b360b1dfb2c3075d3f14be7c281f与既有phase/spec/admission/jobs-plan，原6e/1e/1879身份链、旧UNKNOWN和全部科学设计不变。详见[PILOT_CLOSURE_EXECUTION_zh.md](PILOT_CLOSURE_EXECUTION_zh.md)。

新增独立外层操作器只提交Round A：一个LH-SRF6-PilotV3-A作业，2 PRO6000、8CPU、66G、23小时、no-requeue。该壁钟限制沿用原确认/T03的23小时配置，不增加固定回答数或更新步数。两个独立单GPU worker严格使用原b7启动器与活会话/模型准入，按已冻结顺序分别执行b0的U11→C_SHUF→U10，以及b1的U10→C_SHUF→U11。每个任务新模型进程，没有显式恢复参数，不自动补抽/重提。

操作器源码将以独立实际Git版本/部署/内容binding记录，不能写入或改变b7生产部署与阶段清单；提交前只做精确登记/封存/操作来源检查，完整生产准入仍由原worker在分配内执行。提交前核teacher全用户Req/Alloc运行、待运行和未知预留的8GPU总额及varun024五提交槽，唯一锁与意图先于sbatch；未知提交保留且不自动重试。

B仅在A完整后才能准备，EVAL仅在九路主实验及唯一恢复完整后开放；本轮外层入口不开放B/EVAL/正式训练/最终测试。题组有限候选覆盖、真实错误UNKNOWN、G8信用机会与未见组合学习收益继续分开记录，不根据结果修改题、seed、臂、lambda、步数或验收门。

外层操作器已完成30项本地回归，覆盖精确身份、固定资源、满槽/配额/未知预留、重复或不明提交、错误分配、启动环境及半写回执；Ruff、格式、bash语法和diff检查通过。独立复核已修正两项问题：提交回执采用同目录fsync后不覆盖原子发布，避免快启动读取半份JSON；作业内构造固定环境后设置SLURM_EXPORT_ENV=ALL，让内部srun继承实际环境，符合[Slurm官方export说明](https://slurm.schedmd.com/sbatch.html#OPT_export)。原b7代码不变。

本段登记时仍无新操作源码部署、GPU意图或GPU作业。工程检查通过不等于真实GPU执行或先导科学PASS；实际操作commit、部署与提交回执将在后续事实段补充。

## 11:42 UTC实际唯一提交

独立操作源码`74a95f5b94e5c4e1d92ba7a5ba8b250b850ac4e8`已commit、push并由ls-remote核实；源身份与b7模型生产者分开。一次部署`sr_f6_pilot_ops_code_74a95f5`，四文件25,234bytes，实际标准库导入、来源绑定、封存元数据及无既有worker输出检查通过，部署阶段0模型调用/0作业。operation binding为`0ed394b9ebabc4f752b0f7a44df67acbc5b4fdcfa601146be0ef5434dddb7c64`。

11:42:41 UTC，已核实233935完成离队且teacher全用户实际1GPU/varun024的1槽，唯一提交`233996/LH-SRF6-PilotV3-A`。明确SUBMITTED回执、A_ATTEMPT与A_SUBMISSION均已保存，intent sha为`e27dc5c2eaa83cecc1592139df3df22f1168199f7710999b7e2609287e406eee`。提交后teacher已知3GPU预留/2槽；其它项目只读核额、未修改。

11:42:56 UTC实际PENDING(Resources)，尚无A_STARTED、worker或任务产物。这是排队状态，不是先导完成或科学PASS。禁止再次deploy/submit/prepare/重采；只用私有`.local/pilot_round_a_20261029/run_operation.py read`读取唯一233996及原回执，续接见同目录CONTINUE。旧9216回答、360审核、24T03和全部FAIL/UNKNOWN原件保留。后续文档提交不改变实际操作74a95f5或模型生产b7e7b06。
