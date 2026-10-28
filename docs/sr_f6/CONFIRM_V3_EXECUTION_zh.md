# Interface-v3独立确认的生产合同

本文仅规定方法、来源和执行边界，不包含实际模型回答或私有结果。原interface-v3筛查的
完整冻结SCREEN_PASS是进入条件；确认使用新的实际源码部署和独立运行根，不能把新结果
改名成DEV_CAL/CENSUS_S0，不能重签原版或v2的失败。运行事实以本地执行状态为准。

## 数据与提示先于回答冻结

技术修复后的独立根为`/projects/_ssd/varunssd/louis-ssvc/sr_f6_confirm_v3_20261028_r1`，阶段为
`SCHEMA_CONFIRM_V3`。直接调用原生成器
`build_bank("DEV_SCHEMA_CONFIRM", 8, 20261028, ood=False, zero_alias=False)`：六家族各
索引0至7，48根、每根P0/P1各四视图，共384题。原model-blind拒绝规则全部保留。
原规则产生18个强造根，占37.5%；不把它说成筛查的50%或TRAIN总体比例。

新数据须核对两独立oracle、原生评分、候选证书、匹配编辑成本及与原生产者和固定历史源的
世界重复。只读必要的既有world及来源绑定，不查看测试结果；碰撞或任何审核失败都保留并
停止，不换种子或补题。新世界、图像、私有任务和公开输入只写新根，原CONFIG字节不变。

唯一提示转换为`interface_v3_protocol.public_prompt(categories, query)`，只读取公开类别与
query，不读取数值、gold、答案、候选证书或模型输出。在第一条确认回答之前登记面板映射：

| 面板 | 输入协议 |
| --- | --- |
| DEV_SCHEMA_CONFIRM、DEV_PILOT、TRAIN、DEV_EVAL | 同一v3公开转换 |
| TEST_ID、TEST_COMPOSITION、TEST_CLASSOUT | 同一v3公开转换 |
| NATURAL_INTERFACE、LEGACY、CHARTQA | 保留原已登记原生协议 |

同一面板的父模型baseline与训练endpoint必须使用相同输入。后续通过独立、不可变的输入
覆盖层机械转换，保持world/image/qid/root/factor、seed、臂、lambda、预算、步数和严格
scorer不变。旧默认提示不全局替换，历史重放不改写。测试转换程序只返回绑定及计数，无需
向执行代理显示测试内容。规则登记本身不授权任何后续训练或测试访问；消费者必须先完成
实现和验证。

## 来源、预检与采集

技术预检失败的旧确认登记与证据保持原件。只有在旧根没有任何模型执行或回答、且故障确认为部署遗漏时，才允许在独立修复根登记；新数据、图像、公开输入、配置及全部种子/分片/科学门必须与旧登记逐项一致。修复仅改变实际源码和运维来源，不能据结果换题、换种子或重抽。缺失的配置须作为部署资产和来源清单的一部分验证。

确认登记保留三种来源：原模型/父adapter与数据生产者、冻结筛查5ca3fe9、新确认部署。
完整筛查由其原冻结目录中的隔离解释器只读验证，再绑定报告、封存和文件状态；新部署
不得冒用旧源码身份。未变化模型文件复用既有内容缓存，只在必要绑定和变化节点读取内容。

确认必须有384项实际processor路由、三个tokenizer回转样例、新48个P1 anchor的实际处理后
像素，以及逐图独立MODEL_VISUAL或HUMAN审核和可追溯观察记录。原筛查的48图不能授权新
世界。CPU代码测试不替代实际processor、视觉审核或GPU数值门。只有预检与视觉共同封存后
才能生成这9,216条回答。

三个父各384题，每题完整G8；每父两worker，按排序qid奇偶分片，每worker192题/1,536回答。
每draw种子为
`seed_for(20261028,"srf6-confirm-v3","confirm",parent,root_id,variant,draw)`，draw为0至7，
factor不进入种子。SHUF沿用
`seed_for(20261028,"census_shuffle",parent,factor,parent+"/"+qid)`。全部9,216种子、分片和
每worker首G8在首次输出前冻结；重复anchor、共同随机流和同根视图仍相关。

FP32原生采样、温度1、top-p1、top-k0、1536 token和原停止规则不变。每draw立即保留原文、
token、sampler logps、原停止、实际模型与输入身份以及调用账本。严格scorer与内容诊断分开；
EOS不改成balanced，UNKNOWN不算错误也不从分母删除。固定shard执行slot在模型构造前消费，
重复、部分输出、技术失败或未知提交状态均保留并停止自动补抽。

每worker首个固定G8仅允许一次同token概率对齐，8条单序列FP32 forward，总48条，0新增
回答/0优化器更新。平均绝对差≤0.01 nat、最大差≤5 nat、TIS截断比例≤0.01；原轨迹与逐条
概率向量保存，参数与RNG状态必须保持。此检查不是后续完整loss/梯度数值准入门。

## 完整核验和固定科学门

完整证据消费者须独立读取全部原始G8重建标签与J/COND/GLOBAL/SHUF信用，重演token解码和
停止，验证六worker覆盖、来源、首G8实际trace及逐事件调用账本。文件数、Slurm完成、组内
自报PASS或仅有summary都不足以通过。动态证据在最终绑定节点读一次，保留全部不利行。

确认支持条件沿用原合同，不新增经验阈值：

- 六个parent×factor格各1,536行，V、S-known、B-known分别≥1,383，共18项；UNKNOWN计入分母。
- 每因子576个G8中，COND非零≥24组，覆盖≥12根、≥3家族、≥2父。
- SHUF实际改变≥12组、≥6根。
- 原288个TRAIN根的有限候选覆盖平均增量≥0.05、严格改善≥96根。必须绑定原TRAIN证书，
  不能把新确认集覆盖值替代TRAIN，或把新数据伪装成旧DEV_CAL审核。

支持失败即保留负结果并停止本版本扩展；支持通过仍只到待独立审核状态。新的360条审核
必须从完整新9216原文按原算法抽取，seed2026102701，36层各10条、同格每根最多4条和原
精确文本去重规则保留；缺额不能新增采样。保留独立盲提取、world/query判定及裁决轨迹，
不得复用旧审核或用程序评分代替独立审核，不凭空增加一致率门。

完整数值准入固定选择新确认排序首qid、每父其全部G8，共24条原token，执行原J优势及固定
正探针的loss/梯度比较。原atol=1e-6、rtol=1e-5、KL=.02、概率门及完整状态恢复规则保留。
全零导致的INCONCLUSIVE不能改成PASS或换qid。新审核、新数值和新admission消费者缺失时
先实现并验证，不接受旧身份的许可。全部原门通过后才按合同进行12步pilot、恢复、54路72步
及完整评测。

新消费者的双来源绑定、固定360真实审核、固定24条数值重放及确认完成证明，见
[CONFIRM_FOLLOWUP_EXECUTION_zh.md](CONFIRM_FOLLOWUP_EXECUTION_zh.md)。该完成证明不能替代
未来pilot自身的源码、v3输入覆盖层和运行时准入登记。

## 题组结论与运维边界

分别报告有限候选覆盖、真实错误匹配及UNKNOWN、完整G8信用机会、最终未见COMPOSITION的
D0学习收益。候选兼容不等于模型内部程序；强造层收益、双端错误但差值正确、P1局部退步和
未匹配错误全部保留。确认支持通过只证明达到登记暴露条件，不证明训练会受益。

SSH为eee-cluster，服务器写入只在canonical项目根。提交前实时检查teacher QoS全用户
运行、待运行及未知Req/Alloc最大预留，8 PRO6000总配额和varun024五提交槽分开计数，
其它QoS单列；不绕过限制、不修改其它项目。新GPU名LH-SRF6-，每作业两独立单GPU worker。
代码及纯方法可commit/push既有分支；真实数据、日志、审核和本地状态不上传Git。
