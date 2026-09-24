# 前瞻选择实施状态

继承提交：b52f55993791a9954be07a0195eb656441b53651。工作分支：codex/prospective-reward-selection-20260924。

本轮增加独立的 prospective_selection 模块：元数据划分、共同题序、修复标签、四层权限包、核岭选择器、精度规划、统计、完整训练恢复、原始评估、持久化任务队列及CLI。历史训练器仅增加显式 advantage_callback，默认原行为保持。DIRECT_REPAIR_R4使用同一个PPO损失及Adam路径。

CPU验证：当前188通过、1跳过（既有可选原件测试），耗时23.53秒；ruff与git diff --check通过。CPU通过不代表真实9B通过。真实smoke、开发、冻结和最终结果分别记录，不能用设计包自带日志当作本次执行证据。

真实元数据准备已验证source384、continuation192、P72、D144和场景/真值不重叠。T元数据已完成已登记历史范围的暴露审计，排除2个CPU代理真值别名，冻结用候选面板为288题。详见CONFIRM_EXPOSURE_AUDIT_zh.md；这不证明审计范围以外从未暴露。当前GPU仍绑定原开发数据；最终冻结和测试尚未进行。

P0已重读旧16个H32端点的36,864条P输出，6类恒等式精确成立，42个同题同奖励桶呈现修复结构差别。所有16个E复核尚待真实运行，不能把P重复称为E。

新source共同使用字面LoRA初始化seed17。旧D2来源初始化使用历史hash派生seed；这是本方案明确要求的新source设置，历史轨迹保持原状。B4/K8、生成过滤、LoRA布局、Adam和固定Lnorm继承核对通过。新的生成计数路径省去每条输出的额外完整重评分，必须在真实smoke独立比较行为logp与既有prefix评分后才启动source。

前4条开发lineage为61001–61004，包含8个原点、88个32步分支；每原点11个配方。最终测试任务不得在冻结及decision之前创建；后续测试只执行选择和固定对照的去重并集。

尚未完成的科学交付：旧E复核、前4条开发轨迹、实测工期、完整开发/调参、冻结、最终独立测试以及有结果支持的FINAL_DECISION_zh.md。

当前真实GPU smoke为Slurm169593，代码快照414df78，4条行为logp与独立prefix评分比较差值均为0；完整R0/R1各2步及恢复检查已通过，SlurmCOMPLETED31分08秒。4条开发source及1个旧E复核已在bb23e9b启动；首批结果尚未完成。用户已授权每20分钟持久续跑，automation `ssvc` 已激活；阶段进度以EXECUTION_STATE_zh.md及服务器原件为准。
