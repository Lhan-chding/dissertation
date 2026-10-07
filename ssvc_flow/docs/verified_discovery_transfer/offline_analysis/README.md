# 已揭晓现存输出分析

仅使用Python标准库。实验源码始终冻结在ed1d248；本目录为最终离线统计，不调用模型、不写服务器实验目录。先验证TEST_RELEASE身份和完整性，再对交付快照运行。

```sh
python3 ssvc_discovery_fit_analysis.py /path/to/run_v2 --output /path/to/new_discovery_output
python3 ssvc_retention_analysis.py /path/to/run_v2 --released --output /path/to/new_retention_output
python3 ssvc_transfer_analysis.py --self-test
python3 ssvc_transfer_analysis.py /path/to/copy_of_run_v2
```

迁移脚本写入快照的final_analysis/transfer*；使用没有现存transfer输出的快照副本，脚本拒绝覆盖既有分析。其他两个脚本使用全新的输出目录。statistics_frozen.py按字节复制自冻结实现。所有生成CSV/JSON和来源哈希保存在相邻final_analysis目录；本目录不替代预登记比较，额外父内区间及分层属于探索统计。
