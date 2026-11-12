# SR-SATDOSE-1 准入报告（2026-11-12）

状态：HOLD_MISSING_HISTORICAL_BUNDLE。没有获得模型调用或正式训练准入。

## 已执行

- 原执行包38个登记文件大小与SHA全部吻合，原始包内容保持不变。
- 服务器指定23个源码文件和两个R24 state.pt文件SHA全部吻合（见 intake/SOURCE_FILE_AUDIT.json）。这只是文件核验，不代表已加载并核验policy/reference张量。
- 本地CPU测试59项通过、11项因缺少原生依赖跳过；初次macOS临时目录路径别名导致1项失败，使用真实路径TMPDIR=/private/tmp后通过，未修改测试。
- 服务器使用旧完整package_v2原生依赖运行新包70项测试：全部通过、零跳过。
- 服务器旧完整项目数据生成、评分与runtime contract测试22项全部通过，零跳过。原8类别data_design.py真实存在。
- 固定训练槽位检查通过：4路径、96更新、1536组、768个新训练root槽位、12288个回答预算。这不代表任务/图像已生成或冻结。
- 建立新服务器目录 /projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112，保存用户原始输入zip、解压原包与本次CPU测试日志。

## 缺失与门禁

本包 evidence/SOURCE_FILES.json 绑定的历史包为 SR_ORD_TRAIN_RESULTS_20261111(1).zip，71366142字节，SHA256 b6a560fd0d74e9fbca0d0000b74dcc622b749c400e4f45fb211f6301e29d89d2。

在已查本地Downloads/Documents和服务器指定范围未找到该包或完整解压副本，详见 intake/HISTORICAL_BUNDLE_SEARCH.json。不能用本包附带的旧审计报告代替本次重新执行 tools/audit_historical.py，也不能重新拼造同名文件冒充原始包。

以下均为PENDING：历史全结果重放审计、张量身份和原reference加载、数据/图像及跨历史数据排重、DESIGN_FREEZE、DRY、PREFIX_CALIB、GPU梯度/数值/影子Adam验收、正式两步canary、后续训练与MAIN。

本次新增模型生成0，影子更新0，正式更新0；没有提交新GPU作业，没有调用旧CLI绕过门禁，没有解封MAIN。

## 恢复入口

取得指定历史包或其真实完整解压目录后，先验证原zip SHA/文件清单，再运行原包 tools/audit_historical.py。之后按 CODEX_START_HERE_zh.md 顺序实施。当前仅纳入原始执行包和预检证据，尚未实现服务器训练适配器；不能把CPU PASS作为训练许可。

## 清理

已删除66个可再生缓存普通文件，共2025323字节（约1.93 MiB），删除后上述范围普通文件为0，错误为0；详见 intake/CLEANUP_RECEIPT.json。只删除已确认可再生缓存，保留模型、检查点、原始结果、唯一审计证据及活动任务依赖；不能据目录名cache推断内部模型权重可删除。
