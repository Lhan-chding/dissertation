# SR-SATDOSE-1 准入报告（2026-11-13，DRY运行中）

状态：DESIGN_FROZEN_DRY_RUNNING_PENDING_GPU_ADMISSION。用户已授权AI可视核验替代人工，事前冻结已完成。VPN恢复后于2026-11-13 05:48 UTC（新加坡13:48）补查：唯一DRY作业1736279在gpu-pro6000-9运行；随后计数为11031完成150/640、11032尚未开始，总150/1280。没有发现HOLD或完整校准回执。尚未执行影子Adam或正式更新。此为有时间戳的观测，不代表后续实时状态。

## 已闭合的证据

- 用户提供 `.worktrees/sr-ord-train-20261103/sr_ord_train/execution/final_delivery_20261111/SR_ORD_TRAIN_RESULTS_20261111.zip`，SHA256 为 `b6a560fd0d74e9fbca0d0000b74dcc622b749c400e4f45fb211f6301e29d89d2`，与执行包绑定完全吻合。原文件名中的 `(1)` 不影响身份。
- 本地Python3.12与服务器分别运行历史审计：1099个文件、36864条训练记录、4608个G8和26624条评估记录核验通过；复现14条J=1但选择不正确，未改原生J定义。此前“缺少历史包”阻塞已解除。
- 原包38个登记文件大小/SHA、服务器指定23个源码文件和两个R24 state.pt SHA均吻合。
- 服务器真实CPU张量审计：两个R24分别496个张量，policy SHA及原b0 reference tensor SHA全部吻合；原b0两文件SHA通过。未构造Qwen实例。
- 完整运行源码锁核验570个native源码、124个legacy执行源码、13个模型文件和2个parent文件；数据构造依赖28文件本地/远端完全一致。
- 原执行包70项测试全部通过，零跳过；旧完整项目22项测试全部通过。新runtime 66项测试全部通过、零跳过，见 intake/runtime_cpu_tests_final.log；独立审查所发现的reference参数临时交换误报已经修复并加真实原生回归。
- 服务器 prepared_data_v1 已生成1376张图片，2656次oracle检查通过，全部28936个请求ID唯一且预算吻合：TRAIN12288、DRY1280、PREFIX_CALIB96、DEV1512、MAIN13760。
- PNG结构、柱高、元数据、历史DEV图片SHA、排重和两臂仅清单变化检查通过；随后已完成事前冻结。

## 可视核验与后续门禁

计划第3.3节要求“随机种子固定选32张人工可视检查”。固定32图与私有核验数值位于 visual_review/index.html。已由AI查看8页联系表、32张图，未见裁切、重叠、图例缺失或答案提示；AI_VISUAL_REVIEW.json 明确标注非人工检查，不冒充人工验收。

用户明确答复“授权以AI可视核验替代”；VISUAL_ACCEPTANCE.json如实记录reviewer_kind=AI及替代授权，没有冒称人工检查。DESIGN_FREEZE.json已在首次模型调用前生成，SHA256为6c4b948c533fdd736e089bf3836378f4dbcb76ac66f1d480fa979eaca56c84e0。

已实现只读R24 runtime、原token生成与捕获、梯度诊断、原生reference加载、CPU冻结/源码锁工具和DRY入口；它们通过CPU检查不等于真实GPU验收。真实DRY运行中；PREFIX_CALIB、梯度/内存/影子Adam验证、生产loss独立重放、canary、正式训练与MAIN均未执行。正式训练事务发布、canary工程整合及终态MAIN runner尚未端到端完成；禁止旧CLI绕过门禁。

提交时新增模型生成0，shadow optimizer调用0，正式更新0。唯一两GPU worker 1736279使用live核验的rose账户/QoS，固定依次执行11031/11032的DRY各640条；不自动PREFIX、训练或MAIN，Requeue=0。已从Priority排队转为运行，具体状态以队列及实际回执为准，禁止重复提交。见intake/DRY_SUBMISSION.json。

用户要求每小时监控一次。03:15、04:15（新加坡时间）连接超时期间没有重提作业；用户告知VPN恢复后已补查。后续从13:48检查起按一小时间隔核查队列、完成计数及HOLD/完成回执。

新增完成后CPU分析工具已通过完整90项本地测试、零跳过，日志为intake/calibration_analysis_cpu_tests_20261113.log。独立审查发现的源码锁与逐请求完成证据缺口已修复并增加拒绝测试；测试间原生模块污染已隔离，生产源码身份检查保留。没有读取当前部分raw来选择门槛，冻结runtime没有修改。

PREFIX文档“严格或限定补括号后”在两套整套读法结论不同时存在歧义；在任何PREFIX生成前询问后，用户回复“你认为该如何选择”。代理向用户说明并按原文“或”解释为：任一整套通过即可，不混view/指标，双套完整报告。这不是用户显式选择选项的批准，澄清记录明确区分。2026-11-13 06:28 UTC服务器两次确认PREFIX分配及运行目录均为空，随后将独立澄清文件write-once保存至prepared_data_v1/protocol_clarifications；原DESIGN_FREEZE未改。对应原始字节在本地protocol_clarifications镜像，远端写入回执在intake/PREFIX_PROTOCOL_REMOTE_RECEIPT_20261113.json。分析器须绑定此凭据才能判定PREFIX指标；缺失时PENDING，任何情况下均不直接授予训练许可。接入后全套104项CPU测试通过、零跳过，见intake/prefix_clarification_cpu_tests_20261113.log。

等待DRY期间新增execution/production.py独立原生数值核心，供后续生产事务与诊断比较共用生产路径。它没有optimizer能力，捕获裁剪前后梯度，保留原128等权累计和数值哨兵；独立审查已核对原实现。加入该模块后全套101项CPU测试通过、零跳过，见intake/production_core_cpu_tests_20261113.log。它尚未部署/运行GPU；不能把合成CPU一致性当作真实生产梯度GPU验收。真实两次恢复的影子Adam、短序列单卡/两卡对照、1536压力验收及完整GPU诊断driver仍待实现与执行。

## 服务器位置

`/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112`：原输入包、已SHA验证历史包及解压副本、CPU审计、implementation_preflight、prepared_data_v1。

`prepared_data_v1/runtime_source_lock.json` 为真实来源锁，与DESIGN_FREEZE分开保存。已将科学代码提交86839fffbbbdb84d68fbce8995da8f19acc60bd9的完整runtime与配置部署到prepared_data_v1/code，并独立逐文件验证SHA一致，见intake/IMPLEMENTATION_DEPLOYMENT.json。DESIGN_FREEZE及AI替代授权现已闭合，GPU正在执行零更新DRY。

## 清理与Git

第三轮实际删除9702203395字节冗余归档，净逻辑减少9659735435字节；此前缓存清理2025323字节另计。原9.25GB归档中的唯一历史RUN_REGISTRY、完整成员及链接元数据和恢复说明均先保存、独立SHA核验后才删除归档。清理各轮实际结果见 intake/CLEANUP_RECEIPT.json、CLEANUP_SECOND_PASS.json、CLEANUP_THIRD_PASS.json。第二轮VPN超时没有产生新增删除；后续恢复后继续核验和删除冗余归档。保留新实验所有依赖、唯一科研原始记录、必要检查点与模型。

Git远程已核实为PUBLIC。此前推送包含服务器路径/日志/审计证据的提交被自动审批拒绝，公开外发许可仍未得到明确答复；本次不绕过拒绝。代码和证据可本地提交，但不能称已推送。
