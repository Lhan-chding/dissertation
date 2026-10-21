# 02 源材料、审查边界及生产适配

## 1. 实际取得的证据
这次环境中完整 ZIP 已可读取，不沿用此前“上传失败”的结论。`release/ENDPOINT_SCORES.jsonl` 已逐行读完：233,340 行；SHA256=4c0c644c22859db1f17096d5e5d7ca3d5246baa202ea736f06fcfb7187fcee84；(model_id,pool,protocol,qid,draw) 无重复。
结构化 REPAIR，三块等权准确率约为 PARENT84.48%、RL93.51%、PREFIX90.59%、EXEC79.51%、GOLD54.54%。精确逐块分子/分母见 LOCAL_ENDPOINT_RECHECK.json。它验证评分文件聚合，不是从图像与回答原文重新评分，也没有重新确认全部统计区间。

## 2. f9 的确认事实与尚不能推出的结论
`gpt2_checks.py:127–143`：v7 过程检查收集全文整数减法等式操作数，并检测 /2、/3 或 count。它没有验证每个等式右端，也没有追踪“先错后修正”的最终采用分支；相同操作数可能对应多类别。故其 237/251 是强候选过程证据，但不是可直接部署的通用选择验证器。
`fingerprint2.py:25–42,59–73`：0/2/0 来自答案数值的候选程序分类，不来自 v7 work 解析，更不是四维逐组过程标签的完整联结。
`fingerprint2.py:53`：歧义计数键是 (family,variant)，不含 block；输出中 INTERVAL7 的 93 是三块汇总，不能直接写为 b2 的 93/251。
`gpt2_checks.py:118`（以附带源文件为准）：证据异质性使用 `score.get('certificate',0)`，不是 `certificate_truth`。必须回查聚合关系，再判定 72 组的差异到底是真值、充分性、格式门控中的哪一种。
`fingerprint2.py:6–11` 依赖旧 fingerprint.py，通过 exec 载入；gpt2_checks.py 依赖 scores.pkl、固定 /home/claude 路径；f9 输出含旧异常和追加重跑片段。它们是证据附件，不作为生产入口。生产实现必须结构化导入、明确路径、以 JSON/JSONL 交换并记录单次运行 ID。

## 3. Codex 必须补齐的原生契约
按现有清单，优先检查：
- `code/src/sr_f3_ce/{contract,protocol,evaluation,training,runtime,masks,prompts,data,bootstrap}.py`。
- `code/src/mm_core/scoring.py` 及实际引用到的旧模块。
- `EXECUTION_FREEZE.json`、`RUN_IDENTITY.json`、`plan/manifests/TASKS_AUDIT_ONLY.jsonl`、三块父模型 checkpoint。
这些是源成员名/查找提示，不是假定当前仓库目录布局。解析实际调用链，不靠文件名猜签名。

必须写出：J 的完整逻辑、每个分量的定义和值域；valid/finish/presentation/certificate/certificate_truth/certificate_sufficient 的关系；null/undefined 的处理；真实 advantage 公式和 epsilon 位置；TIS/ratio clipping 的顺序；序列内 token reduction、序列间归约、梯度累计分母；KL 的估计器及参考权重；更新前 rollout 的时点。
训练 J 若非二元：A 可继承原实现；C 的二元分层假设不成立，自动 BLOCK_C，不把连续 J 四舍五入成二元。

## 4. Golden tests
至少从三块×四题族×{全零,混合,全对} 选择所有可得类型，累计不低于 36 个历史组；不可得类型明确记录。原奖励、优势、样本顺序和分母需与已保存日志逐项一致；若原库为浮点32，容差1e-6（相对1e-5），并同时报告最大差与异常行。
A 的 uniform 特例必须与本轮参考采样器吻合；若旧 sampler 是分层/无放回而不同，A0 标为 `J_UNIFORM_NEW_REPEAT`，不能声称纯随机种子复现。旧实现可另用CPU日程比对，不偷偷添加训练臂。
C 的 J 臂必须在相同 batch/state 下与未接入 C 的原 trainer loss、gradient（浮点容差预登记）一致。附加项不能经过第二次 whitening。SELECT_SHUF 保持每组最终 residual multiset。

## 5. 数据恢复
优先使用服务器已存在的完整 restored 目录。若只有完整小包，使用所附 compact_analysis.py/compact_codec.py 的官方恢复程序；COMPLETE_DATA.xz 不是 tar。
跨 Python/zlib 版本可能无法逐字节复现原 gzip。若发生，只能报告容器字节不一致及逐逻辑记录核验结果，不能伪报 PASS_ALL_ORIGINAL_MEMBERS_BYTE_EXACT。恢复只读取数据，不下载模型，不重跑旧实验。
