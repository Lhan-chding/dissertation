# SR-F6 数据与测量独立审计

2026-10-27。范围：原实施包的 CPU 参考与已有 F5 评分代码；本文件不代表服务器来源绑定、真实模型审核或 GPU 实验通过。原包保留在 `package/`，未修改旧 F5。

## 结论与纠错边界

发现两个可复现的参考实现问题，需要在生产实现修正；没有发现必须改变六臂、固定剂量、种子或 72 步预算的数学矛盾。

1. **原生 J 投影会改变数字类型。**原包 `src/srf6_ref/measurement.py:93–97` 用 `json.dumps(default=str)` 投影，JSON 数字经 Decimal 解析后变成字符串。正确答案写成 `N.` 加 80 位零时，F6 V 合法、原生 F5 J=1，但参考投影后 J=0。原因是原生评分器接受精确 Decimal，而字符串另有 64 字符上限。新的 `sr_f6.measurement.lossless_legacy_projection` 只删除顶层 `bindings`、`aggregate` 及必要分隔符，保留原始数字、转义、`work` 与其他字段，再调用原有 `sr_f5.measurement.score_response`。不修改原生 J。
2. **确认测试的普通视图受到未使用优化视图的验收影响。**原包 `src/srf6_ref/design.py:161–173` 对 COMPOSITION 同时构造、验收 P0/P1；即使实际测试取 P0，未使用 P1 出现候选答案碰撞也会拒绝整个 root。保持 P0 提案不变、仅将验收中的未使用 P1 替换为 P0，登记 192 个 roots 中有 7 个发生改变。对应 `(family,index,原attempt,仅P0验收attempt)` 为 BINDING/7/1/0、TIES/0/3/1、INTERVAL/13/1/0、INTERVAL/20/2/0、K_DIRECTION/16/4/2、K_DIRECTION/18/11/7、K_DIRECTION/27/5/1。生产数据适配须执行文档 02 §2.7 的独立普通测试规则，只以实际测试视图与登记约束决定接受，不使用优化 P1 验收筛选。此项在数据适配代码中修复，不在测量模块内变更生成器。

另有一个明确的**生产校验缺口**：参考 `census.summarize` 只验证各 G8，未验证完整六个父×因子格、冻结 DEV_CAL 题目清单、每格行数及同父跨组 policy 身份。它也明示自身不是签署门。生产门必须核对完整期望清单，不能仅遍历摘要中已经出现的格子。

## 字段可判与信用资格分开

`sr_f6.measurement.parse(text)` 不接收 world、query 或 gold。它分别返回单个合法 JSON 是否可读，以及共同协议 V 是否有效。重复键、非法 JSON、多对象与尾随文本不做有利修复。

在完整生成、单个 JSON 可读时，S、B、P、E 按各自字段独立测量。例如 `bindings=null` 导致 V=0、J=0、B=UNKNOWN，但明确可读的正确 selected 仍记 S=1；无效 aggregate 不抹去正确 S/B。缺失或不可解释 operand 保持 B=UNKNOWN。完整可读的空集合、重复成员、错误对象或错误数值是语义 FAIL。B 始终核验题目真正需要的对象与操作数，不以任意声明对象为金标。

**这些可判诊断不能绕过协议门。**`credit_eligible` 必须同时满足 V、完成标记与原生完成条件；`eligible_components` 再与 S/B 的 known mask 相交。现有参考信用函数本身也检查 V 与 finished。没有完成的生成中，S/B/P/E 记 UNKNOWN；A 仍仅由完整合法 JSON 的 answer 决定。A 不因新增字段缺失、错误或无法解释而自动失败。

P 沿用原生事实语法，只衡量已提交事实的真实性。空事实、无效事实或无效实体是 UNKNOWN，不代表完整感知能力。E_raw 检查声明操作数的自洽，E_anchored 保持 `B ∧ 正确运算 ∧ A`；两者均不额外进入信用。

## 实际执行的检查

- 在内存中重建原参考的 792 roots、6,144 tasks；逐条与 F5 两个独立 oracle 比较答案和最终集合，不一致 0；跨 split 的名称/顺序不变事实指纹重复 0。未写入或修改原包数据。
- `PYTHONPATH=src /Users/louis/Documents/ChatGPT/dissertation-ntu/.venv/bin/python -m pytest -q tests/sr_f6/test_measurement.py`：26 项通过。包含六家族各 50 个、共 300 个程序夹具（含原生 J=0 与 J=1），每个再做独立的 selected/bindings 内容变异，共 600 次 J 不变性比较。
- 同一测试文件还覆盖 720 种顶层字段顺序、80 位尾零 Decimal、原始 Unicode/work 文本、各字段非法时的独立可判性、null operand、错误/重复/空范围、重复 JSON 键、截断、完成原因冲突、仅 answer 的合法主答案评分、输入不变性和未知任务扩展拒绝。
- 初次样式检查发现两处格式问题（集合比较方向、长行）；已修复后重跑。没有改动评分契约以迎合测试。

程序夹具不是 360 条真实回答盲审核，也不是实际服务器 J 身份或 tokenizer/processor 审核。后者仍需服务器证据。完整正式数据隔离、48 张真实 processor 图像审核、普查与训练保持各自阶段状态。

## 解释限制

P1 数据效应包含该覆盖生成策略带来的难度变化，不能单独归因于覆盖数字。B 与 S 可以相关，但不要求两项始终一致；它们是可观察声明，不证明内部计算过程。共享新协议内的原生 J 对照成立，也不使 F5 历史准确率成为同接口直接对照。以上限制不通过改变标签或删除不利样本消除。
