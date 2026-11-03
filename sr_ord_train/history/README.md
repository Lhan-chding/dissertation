# 可追溯的历史输入与适用范围

母包：[SR_ORDER_PROBE_COMPLETE_20261103.zip](https://chatgpt.com/library?file_id=libfile_ecdfc0246d30819181389bc7973523f4&account_id=personal)。历史工作记录另见[合并分析](https://chatgpt.com/library?file_id=libfile_db5bba3893ec8191954f8500556ba968&account_id=personal)，[Claude分析](https://chatgpt.com/library?file_id=libfile_0c07b31f496881918493d062a7bc683f&account_id=personal)。
    
source_inventory.json 记录本包从 F6 运行源码快照、探查参考构造/测量/制图模块和回执逐字提取文件的原路径及 SHA256。vendor下只提供用于复核的CPU模块，不是可操作远端调度器；空命名空间初始化文件用于导入。runtime_contract.py 另列实际要匹配的7个评分源与5个关键内核源。historical_sr_f6.json 连同旧72步、6臂/旧准入条件原样保留作为历史资料；本轮不执行该配置。323个开放事实表指纹用于去重，不代表重跑过旧实验或读过封存结果。

96个旧保持视图是已开放的 DEV_EVAL/P0/原v3 overlay；本地从历史构造器重放原数值与题号，不重新选择模型答得好/坏的旧题。实际历史图像以历史原输入清单的图片哈希绑定，不混用本次探查式制图器。见 tools/bind_retention.py 。

布置新运行前核验历史源、父模型和各实际回执。授权范围内的远端历史输入若无法读取，保留日志并报告缺失；不能自行猜测数据、权重、checkpoint 或闭源文件的内容。
