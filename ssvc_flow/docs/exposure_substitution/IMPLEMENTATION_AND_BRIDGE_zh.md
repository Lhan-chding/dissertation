# 实现与桥接记录

新模块位于 `ssvc_flow/src/exposure_substitution`，不修改旧VDT运行器和历史实验。CLI提供prepare、cpu-check、build-queue、bridge、train、eval-diagnostic、eval-confirm-sealed、worker和release-and-analyze。

训练采用显式step/slot；common、donor、replay不能重新shuffle。五个microbatch固定为4/4/3/1/4，逐序列回答NLL求和后除以16；目标含唯一EOS，prompt和padding被mask。学生保留完整Adam、学习率步号、RNG、前向状态、参数和schedule cursor；保存0/64/128/192/256。零有限梯度仍执行Adam更新。

训练与评估使用真实父模型绑定和继承的冻结生成路径。逐回答文件具有不可变身份、固定seed、内容hash和独立失败attempt。E_CONFIRM不提前评分。全部113个登记作业终态后检查实际检查点、原始样本覆盖和分析源码hash才允许揭晓。直接CLI调用与worker共享桥接及STOP约束。

分析固定Gamma、Delta3/4、四项A对照及各父基线，128根配对bootstrap保持所有条件，cross16格和trend4位置使用固定宏权重。输出原始值出现、额外错误、混合锚点、重叠操作数签名、公开删改投影和每条原始回答的审计派生记录。

## 已完成验证

- 115项集成及继承回归测试、35项subtest通过；实际旧产物和本地冻结数据参与测试。
- 附件51项参考测试通过。
- 新代码与测试的Ruff检查、格式检查通过。
- 独立只读复核覆盖恢复、数值权重、样本身份、统计和揭晓条件；发现的问题已修复并加回归测试。

精确命令与日志在[本地验证回执](evidence/LOCAL_VALIDATION.json)。CPU fixture结果不作为模型实验结果。

## 真实GPU桥接

当前状态：源码已提交并推送，源码包已传输；服务器prepare建立SSH连接前网络不可达，因此真实桥接未执行，本次未提交GPU作业。见[完整范围状态](EXECUTION_SCOPE.json)。内部网络及部署路径回执保留在本地，不加入公开仓库的这份状态更新。

桥接计划为连续8步与4步保存后恢复至8步，最多16次技术更新；不访问E_CONFIRM。加载检查点前主动还原起点权重，验证加载确实覆盖参数。最终PASS在推理权重还原核验结束后才发布。恢复差异、实测吞吐、GPU型号与调度记录需在真实执行后补入；当前没有新模型结果或结构效应结论。
