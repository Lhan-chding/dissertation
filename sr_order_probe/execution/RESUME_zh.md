> 最新：原三项烟测因启动环境缺项在模型加载前失败，零回答；恢复作业为242525/242526/242527。入口改用运行根下 `deploy/run_worker_pair_envfix.sh`。先读INCIDENT_LAUNCH_ENV.json及SUBMISSIONS_SMOKE_ENVFIX.json，以下旧提交编号仅作历史记录。

# 继续本次已提交探查

当前完整任务尚未完成。先读取 `EXECUTION_STATE_zh.md`、`SUBMISSIONS_SMOKE.json`，检查原有作业；不要再次提交216条烟测。

- SSH：`eee-cluster`。
- 新运行根：`/projects/_ssd/varunssd/louis-ssvc/sr_order_probe_20261102`。
- 实际入口源码：上述根的 `deploy/v2`，加已记录的 `deploy/presmoke-patch.tar.gz`。
- Python：`/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12`。
- b0/b1/b2 烟测作业：242490/242491/242492，每父两卡、两个固定分片，共72条。
- 三作业最近均PENDING(Priority)。不修改旧F5/F6文件、不操作其他作业、不打开封存TEST内容。

## 烟测完成后

1. 检查三个作业 `sacct`、stdout/stderr、`ATTEMPTS.jsonl`、`RECORDS/SMOKE`、`WORKERS/*/{START,END,RECOVERY}.json`；必须真实父身份、全FP32、模型不变、完整216 ID、同assistant长prefix、真实suffix与终止一致。无输出质量门。
2. 在实际入口目录执行 `python -m order_probe_server seal-smoke --run-root <新运行根>`。该命令重验原输出及预检证据，不能手写PASS。
3. 使用 `scripts/score_answers.py --inputs <deploy/v2/data> --answers <运行根/RAW/SMOKE.jsonl> --split SMOKE --out <新烟测评分路径>`。检查UNKNOWN、length、技术错误与分组指标；低正确率和非法模型输出保留，不重采、不删条件。
4. 只有技术路径通过，才重新查全部实际/等待/不明GPU分配和五作业上限，为MAIN提交各父同样的两分片入口 `scripts/run_worker_pair.sh <code> <root> MAIN b0|b1|b2`。预计主请求每父1440、每分片720。所有seed固定。
5. 断点续跑只使用同split/parent/shard，既有完整输出自动复用；同ID系统故障最多一次恢复。已保存技术异常原文不能被当作无输出重采。

## 主探查完成后

先检查实际日志与成本，再 `python -m order_probe_server collect --run-root <root> --split MAIN`；固定评分4320条，运行原 `scripts/analyze.py` 与 `scripts/supplementary_report.py`。补充脚本的审核seed=2026110203，首条烟测前已记录于 `SUPPLEMENT_MANIFEST.json`。

按 `docs/04` 完成72条有限独立复核（先读BLIND原文抽字段，后读PRIVATE_TASKS判断；注明代理审核，不称人类双盲）。交付中文报告、全部原文/token、评分/配对统计、实际身份/成本、输入源码摘要与技术事件。完整保留阴性父模型、UNKNOWN和不确定性。最终Git提交并push，核对远端。

旧原生J仅支持最多8类别；本轮10类主指标使用冻结probe评分器。17项8类/前缀固定夹具验证的是共有A/S/B规则，不是十类原J认证。CPU-only processor身份中的旧1024限制未用于采样，实际GPU后端身份与生成配方必须均1536。

## 已保留的技术事件

CPU作业242456因抓取源码/测试时不同步失败，无模型调用；242476使用统一快照完成89项测试及实际processor/历史排重。之后13项原生后端和6项补充报告服务器定向测试通过，本地总95项通过。预检回执、48幅源/处理图和目视检查均存于 `execution/PREFLIGHT_CPU_v2`。原部署归档与失败日志留在新运行根，不覆盖失败状态。
