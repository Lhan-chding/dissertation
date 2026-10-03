# 四条小实验交付回执

当前目标已完成：首四条的数据收集、探索性嵌套留出闭环、原始数据核验和本地交付。后续完整矩阵保持暂停，等待用户利用本批数据分析并确定新的idea/算法设计。

- [完整分析数据包（264.63 MB）](../../../artifacts/pilot_four_20261004/SSVC_FOUR_LINEAGE_FULL_DATA_20261004.zip)
- [核心复算包（5.94 MB）](../../../artifacts/pilot_four_20261004/SSVC_FOUR_LINEAGE_ANALYSIS_CORE_20261004.zip)
- [主报告](PILOT_REPORT_zh.md)
- [阅读入口](START_HERE_zh.md)
- [包校验回执](PACKAGE_VERIFICATION.json)
- [独立解压复算校验](PACKAGE_PORTABILITY_VERIFICATION.json)

完整包包括首四条88分支、源训练、前置观测、历史16端点E复核的原始分析输出，以及题面、720幅图像、训练代码快照和完整选择器模型/预测/调参记录。42605个原件文件全部通过SHA256及gzip完整读到EOF核验，395776条权威成功生成记录和3200条有限训练更新的绑定与数量核验通过。权重、Adam张量和缓存留服务器，包内列明省略项与位置；该包不是独立从零重训权重包。

两ZIP全部成员的CRC、SHA256及唯一安全路径通过；核心包解压到独立目录后，39个复算输出逐字节相同。27项针对性测试通过。包生成后的外部回执PACKAGE_VERIFICATION、PACKAGE_PORTABILITY_VERIFICATION和本文件不包含在已经校验的ZIP内部，避免自引用哈希。

代码及分析提交7af92df。研究数据历史的GitHub推送仍受先前自动审批拒绝限制，本轮没有重试或绕过。完整结果只在本地和已授权服务器保存。
