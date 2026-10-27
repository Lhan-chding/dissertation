# 08｜Codex实施任务与交付

## 8.1 这包是什么、不是什么
这里的`srf6_ref`是可执行CPU科学参考，不是下载即可在服务器训练9B的完整生产框架。`server_contract.py`故意不绑定真实模型；Codex必须在已有F5 runtime上实现适配，而不是假装参考fixture已经跑过GPU。

按以下依赖顺序执行，不直接提交54路：

|任务|修改/新增|验收产物|
|---|---|---|
|T00 来源与环境|只读F5；绑定本轮独立run根、父adapter、tokenizer、旧scorer|SOURCE_BINDING回执+内容hash+实际环境版本|
|T01 数据与图像|逐字移植生成规则；生产渲染和processor可视化；补全外部ID面板|任务/图像manifest、双oracle及隔离证明、视觉审核|
|T02 测量与J|新增外层bindings；A与V分开；S/B严格oracle；旧J投影|先夹具/native J parity预检查；T04普查后补360条真实审核并签最终PASS|
|T03 训练适配|信用COND/GLOBAL/SHUF；相同序列归一化；完整原生logprob路径|λ0原生loss/梯度对照、实际token回归、source hash|
|T04 普查|DEV_CAL推理＋反事实信用，不更新|逐回答/组/父/家族日志、可实施门、异常定位|
|T05 先导|9条12步+一条恢复；无效果门|完整概率/梯度/事务/恢复回执、开发端点|
|T06 冻结|真实证据身份锁；全矩阵顺序锁；测试访问锁|五阶段有效回执、REGISTERED_PROTOCOL.json|
|T07 正式训练|54条72步，恢复不重计曝光|全部端点、训练账本、完成/异常清单|
|T08 评测统计|57checkpoint同面板；固定聚合与多重比较|完整表、配对差、区间、原文/标签/成本与结论范围|

## 8.2 已存在的本地命令
```bash
cd SR_F6_CODEX_20261027
python -m pip install -r requirements.txt
PYTHONPATH=src python -m unittest discover -s tests -v
python scripts/math_checks.py
python scripts/budget.py
python scripts/audit_data.py
python scripts/summarize_census.py --input YOUR_REAL_CENSUS.jsonl --output YOUR_REPORT.json
python scripts/launch_guard.py --receipts templates/gates --config configs/experiment.json
```
最后一条对模板**应非零退出**，不是安装失败。`YOUR_REAL_CENSUS.jsonl`不存在时不生成虚拟PASS。绘图示例命令见README，输出不能替代真实模型processor审核。

## 8.3 生产CLI是需实现的接口，当前包不存在这些训练命令
Codex需提供独立命名空间，例如`sr_f6_runtime`，实现`bind/check-data/census/pilot/freeze/train/evaluate/report`，并将实际help文本和代码hash纳入回执。不要对用户声称当前参考包已经有`train`可运行入口。

`train`必须检查：父adapter已绑定，五门真实证据齐全，current config digest吻合，phase='FORMAL'，run_id唯一锁，数据split无测试泄露，恢复事务有效。不提供`--ignore-gates`、`--try-another-seed`或`--best-checkpoint`开关。

## 8.4 最小记录契约
回答JSONL：experiment/phase/run_id/parent/repeat/arm/factor/step/group/qid/root/family/variant/draw，policy/source/data/prompt/image/scorer hashes，sampler seed，raw_text，raw_token_ids，sample_logprobs，finish reason，A/J/V/P/S/B/E_raw/E_anchored，known mask，每项判定见证/UNKNOWN原因。

信用JSONL：以上身份+G8成员ID，native优势，组件均值，stratum成员与大小，c原始/施加，mode，λ/weights，置换seed/映射，nonzero、exchangeable、changed分别记录；记录eligible分母而不只保存成功行。

更新JSONL：step输入/输出checkpoint摘要、128行ID、耗时/token数、各loss/梯度、probability分位数、TIS触顶、optimizer_count、参数变化、COMMIT与RNGhash。恢复记录不能只含一个bool。

评测JSONL：继承回答schema；额外primary_answer_parse_status、original_legacy_score、chartqa_v1/v2版本。不存在的指标null+reason，不填0伪装失败。

## 8.5 最终返回给用户的紧凑包
必须包含：中文报告，独立主表与控制表，四格及交互，全部父/重复差值，统计脚本，配置/代码/数据hash，完整错误/停止记录，oracle审核覆盖，成本账本，至少所有罕见关键事件原文；大体量raw token可留服务器，但给可定位路径和最小定向提取脚本。

不得只提供“PASS 51 tests / completed”或一份结果摘要。正式未运行就写NOT_RUN；只完成普查则只报告测量与信号，不推断训练优劣。
