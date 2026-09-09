# R2 真实实验结果与验收

记录日期：2026-09-10（Asia/Singapore）。原始运行与归档检查 PASS；完整 CPU postflight 待记录。

## 执行事实

- Slurm `146853`：COMPLETED，exit `0:0`，节点 `gpu-pro6000-4`，耗时 `02:42:15`，结束 `2026-09-09T20:25:59Z`。
- 源码 `518b23f4fb6803706a7ded57a2b5bfbefbf2d6ab`；固定 Qwen3.5-9B revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`，BF16、单序列 uncached。
- 六主条件：72 calibration 场景，每条件288 sampled＋72 greedy；长度/thinking：12场景，每分支24 sampled＋12 greedy。总输出2,232，无执行检查失败。
- 基座、LoRA及全参数哈希前后一致，0 backward，0 optimizer step。
- 归档 SHA256：`375348e4311fdf6847a7a1fcce9205dc6b54be37959378f58a78bf3f53dfc983`。本地独立核对18个 manifest 文件与47个历史 Git 源文件，开始/结束源码一致。

## 严格完整动作正确率

下表 pX 使用完整输出的原始严格解析。每个主条件采样分母288，greedy分母72；三家族各24场景，按题等权。

|条件与信息变化|sample X/288|sample pX|greedy X/72|greedy pX|sample 相对原版差值及95%区间（百分点）|
|---|---:|---:|---:|---:|---:|
|SYM_ORIGINAL：原符号基准|78|27.08%|37|51.39%|基准|
|SYM_CLEAR：表达改写，相同信息|89|30.90%|37|51.39%|3.82 [-2.08, 9.38]|
|SYM_NO_OPERATION：移除下游操作语句|89|30.90%|39|54.17%|3.82 [-1.39, 8.68]|
|ORACLE_INDEX：加入真实错误位置|126|43.75%|45|62.50%|16.67 [10.42, 22.92]|
|IMAGE_CUE：符号修复＋真实图表|224|77.78%|66|91.67%|50.69 [44.44, 56.94]|
|IMAGE_ONLY：直接读图，移除观察与cue|285|98.96%|72|100.00%|71.88 [67.36, 75.69]|

区间来自原始配对结果：按家族分层，以 base_scene 为单位配对 bootstrap 5,000次，seed20260909；不对单条输出二次抽样。区间只反映场景抽样，不包含训练seed不确定性。IMAGE_CUE、IMAGE_ONLY、ORACLE_INDEX改变了信息或任务；不把差值当作同信息条件下的机制证明。

## 12场景长度与 thinking 诊断

- sample：256-token 非thinking 正确 1/24；相同12场景的64-token原版 pX=0.00%。1024-token thinking 截断 24/24，可靠final 0/24，diagnostic_final_X 0/24（全输出分母）。
- greedy：256-token 非thinking 正确 2/12；相同12场景的64-token原版 pX=16.67%。1024-token thinking 截断 12/12，可靠final 0/12，diagnostic_final_X 0/12（全输出分母）。

所有 thinking 原始文本/token 均保留；无法定位官方 final 分隔符时不从推理文本抽取答案。长输出诊断只含12个场景，不能用其结果替代72场景主分析。

## 按文档继续的依据

SYM_CLEAR sampled差值区间包含0，greedy正确数与原版同为37/72；原 N 协议保持。IMAGE_ONLY sampled为285/288、greedy为72/72，按文档继续保留独立接口结果。ORACLE_INDEX sampled较原版高48/288，但不据此把错误定位认定为唯一机制。R3仍使用预注册训练/control IDs及原奖励与生成协议。正式进入R3须完成完整CPU输入/语义/统计验收。

## 证据位置

- 服务器原始stage：`/projects/varunssd/louis-ssvc/runs/NEXT_20260909/R2_server_146853_attempt_0/R2`。
- 本地取回：`ssvc_flow/runs/NEXT_20260909/verified_146853/extracted/R2_server_146853_attempt_0/R2`。
- 原始 `samples.jsonl`、`diagnosis_report.md`、`condition_metrics.csv`、`paired_condition_effects.csv` 和 `invalid_taxonomy.csv` 均保留；此摘要不替换原始表格。
