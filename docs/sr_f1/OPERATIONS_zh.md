# SR-F1 运行命令与身份边界

下列入口均已实现。实际执行状态与作业号以 [EXECUTION_STATUS_zh.md](EXECUTION_STATUS_zh.md) 和服务器回执为准；命令存在不表示运行完成。

```bash
export PYTHONPATH=<CODE_ROOT>/src
PY=/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python
RUN=/projects/_ssd/varunssd/louis-ssvc/sr_f1_20261009
MODEL=/projects/_ssd/varunssd/louis-ssvc/cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
$PY scripts/sr_f1/prepare.py --root "$RUN" --model-path "$MODEL" --font "$RUN/private_assets/fonts/DejaVuSans.ttf" --bold-font "$RUN/private_assets/fonts/DejaVuSans-Bold.ttf" --operator codex --defer-freeze
$PY -m sr_f1.chartqa --root "$RUN"
$PY scripts/sr_f1/freeze.py --root "$RUN" --operator codex
$PY scripts/sr_f1/preflight.py --root "$RUN"
```

这些CPU操作在实际调度分配中运行。ChartQA网络/权限失败按外部资产阻塞保留，仍允许主矩阵；已有可用资产必须完整验收和评价。

冻结要求：源码已提交且无脏文件、包与固定清单相符、全体图和原生processor输入记录、模型逐文件与复合hash、环境与字体身份、实际资源许可。非Git部署以 `code/SOURCE_DEPLOYMENT.json` 记录真实source_commit和逐文件hash并核对实际源码；未冻结或任何身份不一致均不可进入GPU。

调度入口：

```bash
$PY scripts/sr_f1/submit_matrix.py --plan "$RUN/config/SR_F1.json" --run-root "$RUN" --code-root <CODE_ROOT> --python "$PY" --engine-lease-minutes 4320 --lease-minutes 4320 --watch --controller-requeue-on-lease
```

CPU controller使用 `scripts/sr_f1/controller.sbatch`，其账户及CPU QoS由当前获批权限决定。worker使用记录在 `manifests/ALLOCATION_PERMISSION.json` 的teacher QoS；总计所有账号当前排队/运行GPU和不明提交，最多共享5单卡。提交先hold、持久化绑定身份、再release，UNKNOWN不重复提交。

每条序列原始token、原文、实际采样概率、图像/输入/adapter身份及失败记录持久保存。租约信号在微批边界停止未完成更新，持完整参数/Adam/RNG/游标续跑；ENGINE中断保留旧轨迹并重新完成同一4对2+2资格试验，增加的技术成本单列。

TEST汇总直至15条科学路径均有终态才开放；不以中途reward、未见测试或外部结果调整96步、奖励、种子、数据根或检查点。`TECHNICAL_FAILED`记录为缺失证据，不能作回答错误填零。科学完整矩阵与技术收尾分别标记。

字体、权重、全部图像、ChartQA数据、原始模型回答及检查点不进入Git；交付须明确服务器保留项和本地包省略项。
