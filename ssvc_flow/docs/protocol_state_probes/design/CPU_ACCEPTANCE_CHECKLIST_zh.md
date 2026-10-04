# 实施与验收清单

## A. 本包已经实际检查

运行：

```bash
cd reference
python -m unittest -v test_contracts
```

46项确定性测试，包含全部元数据上的19,008次程序断言；范围见`verification/EXECUTION_SCOPE.json`。这不包含GPU推理、模型正确率、服务器权重实际加载或Slurm集成。

## B. Codex接入必须检查

### 数据与提示
- O0文本逐字保持；A1只改终端输出顺序。
- L00/L10变量—数值字典一致，L01/L11只改输出顺序。
- B0/B1行配对，BR仅改行排列，B1的TH/Tb精确相等。
- prompt compiler仅接受公开字段；audit和程序预测没有进入聊天消息。
- U22在服务器实际执行记录中仍未用；P旧样本不伪装成本轮O0。
- 执行对照单独kind、单独分母。

### 程序与评价
- 原题、反序题都先规范还原；sum/difference/range遵守原operation。
- 完整JSON严格解析，不接受bool/float/截取数组；越界保留诊断但主事件I。
- C_orig与C_display并存；原奖励C在I上为0；C_alg只作诊断。
- PTLC V1/V2冲突、域回退、Fraction结果不擅自修改。
- match_mask互斥，copy/PTLC重合不双算；无法解析字段为null。
- B1风险必须是原新程序实际不同；保存原新完整向量。
- 原始程序正例预测truth的匹配不单独宣称机制证据。

### 推理与恢复
- 正确旧checkpoint加载，参数和影响forward的配置匹配一次。
- 模型eval/no_grad，无backward/optimizer.step。
- raw-only后端，不提前按旧解析器给新协议判分。
- 固定seed包含checkpoint、公开case、role、draw，不依赖worker与时间。
- 缓存key区分变换后的提示和输出顺序；O0/B0只有文本完全相同时共享。
- 同key恢复不会双计；不同内容冲突会显式报错。
- 非法回答不重试，技术失败与模型错误分开。
- 新生成路径未通过桥接时，不混用teacher、KV缓存或量化输出。

### 统计与报告
- 固定题和家族权重；D/U分开；训练lineage数不等于回答数。
- alias比较的差值和方差均为0，共享后验变量。
- 0/n、n/n有边界不确定性，不把SE=0当确定。
- T1/T2使用同题完整四端点，缺单元不填0。
- paired-scene bootstrap不重复嵌套抽completion；小子集报告逐题。
- 效果不佳不是程序失败；不事后调整PTLC或样本量。
- 结果包包含缺项、反例、other质量以及所有预定比较。

## C. 一次有针对性的GPU回归即可

对新模块运行上述单元测试；对旧生成/加载组件运行其相关smoke。仅当修改了共享训练底层时才扩大回归范围；本轮原则上不修改训练底层。不要每个worker、每个case都重复完整仓库测试。
