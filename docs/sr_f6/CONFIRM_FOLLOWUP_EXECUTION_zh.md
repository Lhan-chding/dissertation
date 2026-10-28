# 新确认的独立审核、数值重放与准入消费者

本文只规定方法及执行边界，不包含实际回答、审核结果或运行许可。实际状态在本地执行记录中
单独保留。实现和合成测试通过不代表真实审核、服务器数值门或科学门通过。

## 两种实际源码身份

9,216条确认的实际生产者保持为
`6e5d36351b8c1f6f18c61661246191772ba6a6dd`，独立根保持
`/projects/_ssd/varunssd/louis-ssvc/sr_f6_confirm_v3_20261028_r1`。已经提交的采集作业及原始
记录不改写、不补抽。新消费者使用独立包`sr_f6_followup`、新的不可变源码部署和独立运行根
`/projects/_ssd/varunssd/louis-ssvc/sr_f6_confirm_followup_v1_20261029`。

启动时冻结6e部署的`src`排在新消费者部署的`src`之前。两个包分别验证自己的实际导入路径
和内容清单，包括作为`__main__`执行的入口；新包不能冒用6e的生产者身份，冻结包也不能被
新checkout中的同名模块覆盖。注册只绑定来源，不授予模型访问权限。

完整确认分析仍只由原6e的`confirm_analysis.assess`执行一次。后续绑定读取已封存的分析，
用原验证器重建全部9,216条原回答、原token/停止、调用账本、六worker和48条首G8对齐，
再调用纯分析函数核对所有报告。不会重复运行写入式`assess`。只有实际
`CONFIRM_SUPPORT_PASS_REVIEW_PENDING`才允许新审核；18项门、COND/SHUF和原288个TRAIN根
覆盖要求均不变。未知、部分、技术失败和科学失败不能靠磁盘上的PASS字符串跨越。

完整证据首次绑定及冷进程验证使用内容检查。进程内只复用经过验证的私有缓存，并核对整个
依赖闭包及目录成员未变；未变化模型沿用原内容绑定。禁止每条审核或数值轨迹重新扫描全模型。

## 新360条独立真实审核

从真正的`SCHEMA_CONFIRM_V3 / DEV_SCHEMA_CONFIRM`完整矩阵抽取，不把新phase或qid改名为旧
DEV_CAL。保留原seed`2026102701`、36层各10条、每parent/factor格同根最多4条、原精确文本
去重、稀有状态优先级和原哈希域。去重范围为同parent/factor/qid；不足360保留缺额，不补抽、
不放宽根上限、不从旧回答填充。

`REVIEW/PACKAGE`中的盲抽取包只含匿名ID、原回答文本、协议和原停止信息。实际提取完成并
封存之后，判断者才收到已封存提取和world/query。parent、factor、程序标签、gold和私有
选择账本不能进入盲审核输入。程序评分只用于比较，不能替代真实独立阅读。

每次实际审核在开始前登记独占attempt，保留真实MODEL_TEXT或HUMAN身份、方法、实际可见
输入输出及顺序时间。不能把模型审核写成HUMAN，也不能伪造平台未暴露的模型身份。审核来源
记录不等于密码学身份认证。原UNKNOWN、分歧、裁决、oracle错误、假PASS和泄漏规则完整保留，
不增加经验一致率门。所有未结束、孤立或失败attempt都必须被枚举，不能只选择有利回执。

`verify_completed`重算固定包、全部审核与裁决以及已封存汇总。终态仅为
`REVIEW_COMPLETE_NOT_GATE_SIGNED`；没有训练、采样或评测权限。

## 固定24条实际数值重放

仅在原支持门及新360条真实独立审核完整通过后加载GPU模型。选择确认登记中的排序首qid，
三个父各原G8，共24条原始token序列。每条只执行一次reference和一次current forward，总48次，
比较原actual-J优势和固定positive probe，总48组loss及全adapter梯度；新增回答和优化器更新
均为0。

数学核沿用冻结6e的`numerical_replay._execute`及原原生目标函数。新认证session、runtime和
审计路径属于新消费者；原raw的phase、run_root和来源保留原值。输入从已绑定生产根读取，
输出只进入新的T03目录。加载FP32基础模型和原父adapter，仅adapter求梯度，不创建优化器。
实际adapter内容必须与确认采集的原父tensor身份一致。

保持原atol`1e-6`、rtol`1e-5`、KL`.02`、概率/TIS门及完整状态恢复规则。全零探针导致的
INCONCLUSIVE保持原样，不能换qid。异常后核对adapter/reference、随机数、梯度、requires-grad
及训练模式；冻结base或buffer异常时标记runtime不可复用，不能宣称已恢复。

调度使用一个两GPU作业：一条单GPU通道依次执行b0、b2，另一条执行b1；每父独立进程。
任何后续父开始前重新检查失败与准入。提交前实时核teacher全用户8GPU预留及varun024五槽，
保留唯一意图和回执，未知提交不自动重试。固定三父的全部attempt、失败、账本和结果均由
`verify_completed`枚举，不能用调用方选取的回执子集或合成CPU结果通过实际数值门。

未知、额外或损坏的执行证据记为TECHNICAL_INVALID/INCOMPLETE；只有原数学核对完整实际
数值轨迹判定FAIL时才记为科学数值失败。二者都不能自动重跑，但不可混淆其科学含义。

## 确认完成证明与后续边界

`admission.assess`重新读取来源、完整支持、新360审核及三父实际数值结果；`seal`把本次结果
写入唯一`ADMISSION`目录。只有全部实际事实通过才能得到`CONFIRMATION_GATES_PASS`。
FAIL、INCONCLUSIVE、UNKNOWN及部分执行原样保留，不自动补跑。

操作时先用只读`review_audit.aggregate_review`检查真实审核和分歧，完成必要的独立裁决，
再执行一次`review_audit.seal`。裁决须绑定完整原审核、分歧及实际decision evidence，开始
时间晚于原判断，执行实例也须独立；不能换审核者名字后由原实例自裁决。同样先只读检查
admission，再一次封存；负结果和部分封存都不能反复覆盖试至PASS。

0GPU启动器为`scripts/sr_f6_followup/followup_cpu.sbatch`，仅接受显式register、bind-support、
review-package、review-seal、t03-freeze、admission-seal动作。新360审核封存后，t03-freeze把
原固定24条的实际来源写入T03/SPEC.json，随后才可使用一次性GPU dispatcher。CPU提交同样
需要唯一意图、明确回执和实时
五槽检查。源码部署与注册不是任何门的PASS，也不授权执行后续动作。真实审核记录由实际
独立审阅产生，启动器不生成审核结论。

该证明始终保持generation/training/evaluation为false。未来12步pilot必须另行实现并登记
真实生产源码、已冻结的v3公开输入覆盖层、实际运行时访问控制和相应准入消费者；旧PILOT
许可不能用于新确认。此后才按原合同处理恢复验证、54路72步和完整评测。

题组结论分四层报告：有限候选覆盖、实际模型错误与UNKNOWN、完整G8可用信用、最终未见
COMPOSITION的D0学习收益。确认门通过只证明达到登记暴露与测量条件，不能提前声称训练收益。
所有不利家族、P1退步、强造层和未匹配错误保留；不得据结果调题、seed、lambda、臂或步数。
