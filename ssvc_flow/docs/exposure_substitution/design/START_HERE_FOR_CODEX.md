# Codex入口

## 阅读

1. `OVERVIEW_zh.md`
2. `CODEX_EXECUTION_PLAN_zh.md`（主规范）
3. `THEORY_AND_STATISTICS_zh.md`
4. `GPU_RUNBOOK_zh.md`
5. `protocol.json` 与 `manifests/BUILD_REPORT.json`

## 现有与待实现

本包已经：复现最新Claude脚本、读取原VDT代码与当前GitHub分支、生成新任务/审计/目标/三个显式schedule、提供CPU数学与数据契约测试。

本包没有：修改仓库、加载模型、连接服务器检查点、执行新Qwen回答或训练。`reference/`不是完整GPU训练实现。

## 第一步

从真实VDT输出读取父S96/REP96、原T和R_replay。以新worktree实现显式slot采样；不要继续按每臂focus列表的长度随机shuffle。若本地有本包排除索引之后的新任务，先补一次root排除，不能在看到新结果后再筛题。

可重建本包清单：

```
python reference/build_manifests.py \
 --results-zip /actual/path/SSVC_VERIFIED_DISCOVERY_TRANSFER_RESULTS_20261007.zip \
 --out /new/path/manifests
python -m unittest discover -s reference -p 'test_contracts.py' -v
```

不要用产物名字猜sandbox路径；以上路径由实际环境填写。

## 第二步

完成CPU集成测试、一次真实GPU桥接，然后启动S96/block0的A/B/C。默认允许本计划中的18次SFT与评估，不启动计划外RL、负例训练、发现或剂量扫描。

特别注意：A/B/C只在供体关系上不同；供体的观察、目标、位置j2必须一致。供体单独microbatch，所有序列仍各占1/16。恢复学生时必须恢复Adam，不能fresh重启。

## 最终回传

主结果必须有Gamma、两个接收单元的原始效应、A/父模型对照、每父/每schedule的完整结果，以及供体叶收益、其他中心保持、混合锚点、值出现和投影读数。

保留完整原始回答、训练slot/NLL及检查点位置。无增益或区间宽仍然是结果，不改主比较、不删臂、不自动延长。

最终以 `FINAL_FINDINGS_zh.md` 回答：**结构匹配的训练曝光替换，是否真实改变了相应受保护任务的行为；哪些现有解释得到支持，哪些仍未区分。**
