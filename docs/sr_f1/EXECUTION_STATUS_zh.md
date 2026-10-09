# SR-F1 当前执行状态

实测时间：`2026-10-09T13:46:59.531095+00:00`（新加坡时间21:46:59）。此文件是时间点快照，实时状态以服务器调度器及回执为准。

状态：`FORMAT_REPAIR_BRIDGE_UPDATES_VERIFIED_RUNNING`。技术修复已经审计、部署并激活，GPU已完成桥接第1/16步真实参数更新并保存完整检查点，正在处理下一步。原FORMAT只覆盖93/256，桥接后格式门禁尚未确认；ENGINE与科学训练未开始，没有本轮能力效果结论。

## 此次问题与修复

- 原256条回答全部生成完成，其中93条可按字段评分、22条达到768-token上限、234条正常EOS结束。正常结束的回答仍有141条违反严格JSON/字段协议，不能把问题全部归因于截断。
- 实现把“出现任意截断”直接判为停止，错误阻止了合同允许在排除技术原因后执行的唯一公共格式桥接。修复只改变这道过严门禁，并补齐桥接后实际权重身份的发布时机。详见 [技术修复说明](FORMAT_GUARD_REPAIR_zh.md)。
- 原生CPU审计 **195666 COMPLETED，退出码0，24秒**；18项检查全部通过，排除输入路由、模板、停止和解析技术错误。三个相关题池192个标准答案最长118 tokens（含EOS），均可在原768上限内表达。审计无模型调用、无权重加载、无CUDA上下文。
- 部署及一次性恢复预检 **195668 COMPLETED，退出码0，52秒**。原冻结、原源码真实副本、失败attempt0000及原256条回答全部保留；新attempt0001复用原FORMAT，无新增before采样。见 [服务器修复审计回执](validation/VERIFIED_FORMAT_REPAIR_20261009.json)。
- 修复代码提交 `e61e9de86e6954c94c751b6383b53da68718185b` 已推送并核实GitHub同名分支，也是服务器实际运行的源码提交；后续文档提交不改变该部署身份。完整本地检查 **276 passed，3 subtests passed**，ruff、格式、shell及原包76项哈希检查通过，见 [修复本地检查](validation/REPAIR_LOCAL_CHECKS.json)。其中“服务器待执行”字段是本地检查时点的历史状态，后续服务器结果见独立回执。
- 768-token上限、95%门槛、原提示、16×8公共桥接、五臂×三seed×96步及所有冻结科学设置不变。一次桥接后若独立FORMAT_CONFIRM仍不足95%，按原合同停止，不追加训练。

## 已核实

- 原ZIP安全路径、CRC、76个包内哈希通过；原包111项测试通过；10项清单重建逐字节匹配。
- SR-F1训练、评价、恢复、调度和释放入口已实现；初版223项本地测试回执保留在 [初版本地检查](validation/FINAL_LOCAL_CHECKS.json)。这些测试不替代实际GPU概率/梯度/恢复门禁。
- 初始冻结源码提交 `c3001259acfb2347180bf9f9d62227c3997a3e40` 及134项部署清单保留在 [原部署身份](validation/DEPLOYED_SOURCE_IDENTITY.json)。当前修复通过独立源码修订回执绑定原冻结，并未改写原部署历史。
- CPU预检 **195408 COMPLETED，退出码0，耗时6分10秒**。服务器实际核验全部4800题、2890张图及1张白图；保持1024×768，最小主标签有效字高13px，像素重建和七项输入隔离检查均通过。取回的回执及14张联系表与冻结hash相符，并抽查4张联系表。见 [服务器视觉核验](validation/SERVER_PROCESSOR_QA_20261009.json)。
- 固定9B revision及复合权重hash已按服务器实际字节重新核对：`921fb7dbfce93fbcf3ab0ccc4fe63bdc41269e7b6ce8f8d6fc315c042fee5511`。
- ChartQA服务器状态 `AVAILABLE_VERIFIED`：全量2500题，human/augmented各1250，1509唯一图像；固定作者revision、图像、清单和评分实现均校验。尚未模型评价。
- 执行冻结 `FROZEN`；冻结文件SHA256 `b84d6d1ad26761cb07e3db706be0b8ca2b4e4a8d436473b0ad3e964a81f463f9`。
- 旧F2及SER-J23未恢复或改动，未读旧F2结果选起点或超参数。

## 当前作业与后续

- **195677**：恢复CPU实验调度器 `RUNNING`，在 `cpu-2`。
- **195679**：`COMMON_START_attempt0001`，单张PRO6000，在 `gpu-pro6000-12` 运行。已保存桥接第1/16步完整检查点及对应8条样本的更新记录，损失与梯度范数均有限，实际参数hash已不同于零LoRA；确认和复测面板均0。见 [GPU恢复快照](validation/GPU_REPAIR_RECOVERY_20261009.json)。
- 公共零输出LoRA已验证 `zero_output_exact=true`；原FORMAT256条仍在，activation回执记录原回答复用及新before生成0。
- 后续顺序：一次16步共同桥接及独立格式确认、原题复测 → ENGINE实际4对2+2 → 固定MONITOR基线 → 15条96步训练 → 全部注册评价 → 原文独立重评分与统一中文报告。门禁失败保留技术记录并阻止下游；不跳过门禁。
- 调度器跨运行根统计当前用户所有运行/排队GPU和未知提交，最多共享5张单卡。72小时是Slurm租约，按完整状态续跑；没有GPU累计时长或研究墙钟停止阈值。
- TEST保持封存；当前进入GPU协议与工程阶段；科学更新尚未开始。CPU通过与源码上传均不是科学完成。

原始启动快照保留在 [服务器启动回执](validation/SERVER_LAUNCH_20261009.json) 和 [初次GPU启动回执](validation/GPU_START_20261009.json)，不能据此推断旧作业仍在运行。后续运行持续发生在服务器；本地记录不会自行成为实时状态或最终报告。

## 保留的技术历史

- 初始SSH超时，用户确认网络恢复后连接成功。
- 初次源码传输被自动审批拒绝，用户随后明确授权上传及继续执行；后续部署成功。
- 初始CPU预检195389使用rose QoS，未开始执行。Slurm拒绝原地更换QoS后，仅取消该尚未运行作业（`CANCELLED`、`Elapsed=0`），按已核实的teacher QoS提交195408；保留取消、迁移、提交及释放回执。
- 原GPU **195415 FAILED/1:0，耗时1:24:35**；原调度器 **195409 FAILED/2:0，耗时1:26:14**。该失败是格式门禁阻塞，未执行桥接、ENGINE或科学训练。失败attempt0000、日志、注册及journal均保留。
- 旧预冻结部署保留在服务器 `code_precommit_attempt01`；冻结原源码保留在 `code_before_format_guard_repair` 及 `code_displaced_format_guard_repair`；当前修复部署为 `code`。未覆写原实验回答。

服务器运行根：`/projects/_ssd/varunssd/louis-ssvc/sr_f1_20261009`。
独立分支：`codex/sr-f1-20261009`。
