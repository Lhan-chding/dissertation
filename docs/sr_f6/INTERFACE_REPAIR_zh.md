# SR-F6 interface-v2 修复范围与使用方式

2026-10-28。依据用户提供的 Claude 分析和 GPT Pro 独立诊断进行代码修复。
两份意见是审查输入，不代替实验授权或原始证据。本修复没有改变旧版 FAIL，
没有新增模型回答，没有启动优化器，也没有恢复已关闭的自动跟进。

## 已采用的修复

- **共同提示。** `interface_protocol` 由同一字段说明生成提示及形状 schema。
  selected 明确为字符串数组；operand 明确为算出的数值，不能是列名或算式；
  差值保留符号；TOPK 和区间过滤的证据要求分别说明。work 可选，建议位于最终
  声明之前，answer 最后。字段顺序仍是提示建议，不是额外奖励条件。
- **生产输入分支。** `data.public_prompt(..., prompt_version="srf6-interface-v2")`
  才使用新提示。默认旧提示和自然接口不变。提示仅接收公开类别和完整公开 query，
  不读取数值 world、gold、P0/P1 因子或候选证书。wire 仍为 `structured_v4`，
  因为响应语法没有改变；新输入另记 prompt_version 及内容身份。
- **一图多题。** `coalesce_asset_inventory` 保留每个 qid 引用，把唯一逻辑文件交给
  原核验器；同路径不同内容、目标、大小或元数据失败，不同路径同内容不合并。
  通用 `verify_bound_files` 的路径及变更保护没有放宽。
- **视觉回执别名。** 仅识别 source_viewed→reviewed_source、
  processed_viewed→reviewed_processed、defects→issues。原状态、观察、文件身份
  保留；缺字段、类型错误、冲突和负观察不得被改成通过。派生映射明确不是新的看图审核。
- **测量诊断。** `diagnose_response` 保留整个原 wire_score，新增 J_core_diag、
  各字段语法/可判性/真值、多项错误、证书真值/充分性和字段顺序。Jcore 仅删除严格
  JSON 对象中的原始成员，保留数字字面量；不重序列化 Decimal，不并入绑定证据。
  原始 EOS 与评分完成分开，仍用原普查 balanced-only 规则。
- **题组诊断。** `group_interface_audit` 分开报告全部注册候选的响应覆盖、绝对区分、
  anchor 碰撞揭露、具体语义边界、全 8×8 回答配对、同一可评子集的 oracle-only 对照。
  原 active coverage 不替换；UNKNOWN、库外和不利结果保留。64 对组合不是64个独立样本。

`admission._source/_processor` 仅显式传入
`consumer_version="srf6-receipt-consumer-interface-v2"` 才启用新适配。
旧 `assess/seal` 默认行为不变，不能用新兼容代码重签旧版失败。

## 没有直接采用的建议

1. 不将宽松 JSON、对象数组自动转字符串、从文本正则找 answer 接入主评分。
   当前 A 可忽略合法 JSON 中的新增字段错误，但其他字段破坏整段 JSON 时仍受影响；
   **不能声称 A 已完全不依赖协议语法**。若改变这个终点，需单独规定无歧义抽取语法，
   覆盖嵌套 answer、字符串内 answer、重复键、多对象、截断等反例后另行登记。
2. 不把 bindings 补进 evidence 来改变 J；J′ 可作另行定义的后验敏感性，不能充当奖励。
   提示修复能提高多少可判率、证书充分性或答案表现，必须由新回答验证，不能预报95%。
3. 不新增“P1 至少多出15%非零信用或脆弱响应”的科学门。该方向不是数据学习价值的
   必要条件。也不自动缩为36路：这还会把 C_GLOBAL/C_SHUF 从 P1 移到 P0，改变对照对象。
4. 不根据当前输出改生成器、seed、λ、臂、步数、长度、采样或数值精度。
   不启用受限解码、格式 SFT 或提示反复搜索。

## 已实现的筛查准备入口

```bash
PYTHONPATH=src python -m sr_f6.interface_repair \
  --source-root /path/to/existing/run \
  --output /path/to/new/repair-input-plan
```

入口只读完整 DEV_CAL，从每家族固定索引0/1的 root 选择96个题目/因子输入，
保留原图片引用和任务，仅改共同提示。输出完整的三父×两draw **576条输入/seed计划**、
schema 和提示文本；相同 root/variant/draw 的 P0/P1 共用 seed。
每父×因子96条，拟议工程筛查的 V/S-known/B-known 各须至少87条。
只有一个提示候选，失败后停止并保留全部行。

**状态是 INPUT_PLAN_ONLY_NOT_DEPLOYED，不是服务器冻结或可运行的GPU提交入口。**
本模块不加载模型、不构造确认集、不读正式测试、不签门、不发作业。
任何输出目录已存在时拒绝覆盖，输出不得位于原运行目录内。

## 后续实际执行边界

先在服务器以新消费者运行原文件核验和实际 processor/tokenizer 检查，绑定新的源码、
提示、任务、模型和运行根；不重扫未变化的模型权重。不因旧图像可复用就声称新提示的
token 输入已经核验。需补齐新版本实际运行登记和受控采样接线后，才能执行576条筛查。
小筛查是工程检查，不是总体90%的置信保证，也不是完整G8信用普查。

通过后才可另行冻结独立确认开发集及其来源，完整普查仍为9,216回答、每格1,536条，
各可判性要求至少1,383条，其他原支持条件不变。原360条审核、数值及恢复要求不能
由小筛查代替。先导、54路训练和正式测试均不由本补丁自动批准。

验证分开记录：ENGINEERING_VALID、MEASUREMENT_VALID、COMPARISON_SUPPORTED、
EFFECT_ESTIMATED。代码测试通过只支持对应工程行为；题组学习价值仍待正式实验检验。
