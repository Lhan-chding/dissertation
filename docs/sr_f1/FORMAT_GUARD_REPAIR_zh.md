# FORMAT 截断门禁技术修复

## 实测问题

原FORMAT的256条真实回答中，字段可评分覆盖93条（36.328125%），22条在768-token处自然截断，234条以EOS结束。后者中141条仍不满足严格JSON/字段协议。原GPU作业195415以FAILED/1:0结束，调度器195409在记录COMMON_START阻塞后以FAILED/2:0结束；桥接、ENGINE和科学训练均未执行。

实现把“存在任何截断”直接当成永久阻塞，这比合同§4.3和§7.2合读后的要求更严格。合同要求保留真实截断失败和原分母，并先排除输入、parser、停止及路由技术错误，再判断是否进入唯一公共桥接。关闭thinking不会自动强制JSON，不能据此把所有解释性输出认作模板错误。

即使把22条截断全都视为格式成功，上界仍只有115/256=44.921875%，低于固定95%要求。这个上界只用于排除“截断足以解释格式不足”，正式覆盖率仍是93/256。桥接不由答案或读数准确率触发。

## 修复与原有证据

- 新增CPU技术审计：固定模型tokenizer重解码256条原token，核验EOS/768上限、原始采样概率、输入/模型/adapter/槽位/seed/hash；原生processor重建32题输入并与冻结及GPU记录比对。检查FORMAT、FORMAT_CONFIRM和BRIDGE全部192个程序gold completion含EOS能容纳在768内。
- 审计不产生新回答、不加载权重、不执行训练。审计只有在证据完整且排除技术原因后才发布`VERIFIED_PROTOCOL_NONADHERENCE`；这是唯一桥接的技术许可，不是最终FORMAT通过或科学结论。
- 保留原`EXECUTION_FREEZE.json`、原源码真实副本、失败attempt0000、完整journal和256条原始记录。`TECHNICAL_REPAIR.json`绑定原冻结、原/新源码commit及逐文件hash、具体文件白名单和技术审计。
- 显式一次性恢复只能作用于该已终止的COMMON_START失败，且桥接与下游尚未开始。新attempt0001复用原FORMAT；不会重新采样before面板，也不会清空旧失败。
- 桥接后先发布`SRF1_FORMAT_BRIDGED`实际权重身份，再执行确认及同题复测，修复原实现的“权重已变但model_identity仍标零LoRA”记录缺陷。

768-token上限、严格解析、原提示、95%门槛、16×8公共桥接、科学五臂/三seed/96步、数据及所有随机流均保持不变。若一次桥接后的独立确认仍不足，保持协议阻塞，不追加桥接。

## 实际执行边界

服务器CPU技术审计195666和部署激活195668均已COMPLETED/0:0。18项原生审计全部通过；三个题池192个标准completion最长118 tokens（含EOS）。原冻结SHA256保持`b84d6d1ad26761cb07e3db706be0b8ca2b4e4a8d436473b0ad3e964a81f463f9`，修复源码提交`e61e9de86e6954c94c751b6383b53da68718185b`。

截至2026-10-09 21:46:59新加坡时间，恢复控制器195677和GPU作业195679均RUNNING。桥接第1/16步已真实更新参数、写出完整检查点；8条序列的损失与梯度均有限，参数hash不同于公共零LoRA。原FORMAT256条复用，新增before生成0。技术阻塞列表为空，任务中的旧blocker字段保留attempt0000失败历史。

当前只能确认技术恢复和首步实际训练；独立FORMAT确认、原题复测、ENGINE及科学训练尚未完成。详见 [原生审计回执](validation/VERIFIED_FORMAT_REPAIR_20261009.json)、[GPU恢复快照](validation/GPU_REPAIR_RECOVERY_20261009.json) 和 [当前状态](EXECUTION_STATUS_zh.md)。
