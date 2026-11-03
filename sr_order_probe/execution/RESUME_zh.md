> 最终更新：主实验与72条审核已完成。先读 FINAL_STATUS_zh.md 与 ../results_20261103/RESULTS_REPORT_zh.md。下文提交/排队/待审核状态为历史记录，禁止据此重复运行。

# 继续本次已提交探查

当前完整任务尚未完成。216条烟测已真实完成并通过技术验收，主实验作业242982已提交。先读取 `SUBMISSION_MAIN.json`、`SMOKE_RECEIPT.json`、`SMOKE_REVIEW.json`；不要重复提交烟测或MAIN。

- SSH：`eee-cluster`。
- 新运行根：`/projects/_ssd/varunssd/louis-ssvc/sr_order_probe_20261102`。
- 实际入口源码：上述根的 `deploy/v2`，加已记录的 `deploy/presmoke-patch.tar.gz`。
- Python：`/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12`。
- 恢复烟测242525/242526/242527均COMPLETED 0:0，每父72条。原失败作业及原因保留于INCIDENT_LAUNCH_ENV.json。
- MAIN作业242982：LH-SROP-MAIN-4W；单作业4卡，每父4个固定分片，各360条，依次b0、b1、b2，共4320条。入口是运行根下deploy/run_main_four_workers.sh。
- 不修改旧F5/F6文件、不操作其他作业、不打开封存TEST内容。

## 已验收烟测与当前主探查

全部216条与六个worker结束身份已复核，SMOKE_TECHNICAL_PASS。技术错误0、采样token 92,159、参数更新0；终止37 eos与179 balanced，无length。三个长prefix条件实际强制274–278 tokens，同assistant。b0/b1/b2答案正确45/48/49条（每父72）；这不是主实验效果证据，也没有用于放行或改参数。

2026-11-03 00:43新加坡时间提交主实验242982。提交前同QoS四项其他作业共4卡，账号4个作业，因此申请一个名额和4卡，总计8卡、5作业。更改仅为确定性请求分片，冻结输入、模型、种子、batch=1及评分不变。每个worker得到33GiB host内存。

先查242982的squeue/sacct和MAIN记录。任何worker失败均停止后续父模型及统计；保留原输出/失败。断点只续原MAIN/parent/4分片组合，已落盘回答不得重采，同ID系统故障最多一次恢复。不要无条件重启整个作业。

入口在全部worker完成后按顺序collect、score、analyze、supplement，输出RAW/MAIN.jsonl、SCORES_MAIN.jsonl、ANALYSIS_MAIN、SUPPLEMENT_MAIN。这是有限流水线，没有定期监控。若已有输出，先验证，不能覆盖或盲目重复分析。

## 主探查完成后

先检查实际日志与成本，再 `python -m order_probe_server collect --run-root <root> --split MAIN`；固定评分4320条，运行原 `scripts/analyze.py` 与 `scripts/supplementary_report.py`。补充脚本的审核seed=2026110203，首条烟测前已记录于 `SUPPLEMENT_MANIFEST.json`。

按 `docs/04` 完成72条有限独立复核（先读BLIND原文抽字段，后读PRIVATE_TASKS判断；注明代理审核，不称人类双盲）。交付中文报告、全部原文/token、评分/配对统计、实际身份/成本、输入源码摘要与技术事件。完整保留阴性父模型、UNKNOWN和不确定性。最终Git提交并push，核对远端。

旧原生J仅支持最多8类别；本轮10类主指标使用冻结probe评分器。17项8类/前缀固定夹具验证的是共有A/S/B规则，不是十类原J认证。CPU-only processor身份中的旧1024限制未用于采样，实际GPU后端身份与生成配方必须均1536。

## 已保留的技术事件

CPU作业242456因抓取源码/测试时不同步失败，无模型调用；242476使用统一快照完成89项测试及实际processor/历史排重。之后13项原生后端和6项补充报告服务器定向测试通过，本地总95项通过。预检回执、48幅源/处理图和目视检查均存于 `execution/PREFLIGHT_CPU_v2`。原部署归档与失败日志留在新运行根，不覆盖失败状态。

## 最新调度状态

新加坡2026-11-03 00:44实查：242982为PENDING，原因QOSGrpGRES，MAIN仍0/4320。QoS总限8张PRO6000，其他可见请求4张，本作业请求4张；可见请求合计不能完整解释调度器准入，不能承诺立即开始。不重复提交或改变其他作业。见MAIN_QUEUE_SNAPSHOT.json。
