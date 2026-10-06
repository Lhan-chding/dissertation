# 当前正式运行入口（run_v2）

2026-10-06 17:04已授权恢复后，当前worker为 `v2-lane0=187958`、`v2-lane1=187959`、`v2-lane2=187960`、`v2-lane3=187961`、`v2-lane4=187962`。只读查询使用 `squeue -j 187958,187959,187960,187961,187962`。旧187115/187116/187154/187155/187156均FAILED/1:0，原五个工作项已显式retry一次；不要再次retry或提交。17:06:33为1运行4排队，实际增量见 [恢复核验](RECOVERY_20261006_zh.md)。以下初次提交ID保留为历史。

当前源码 `code_v2/ssvc_flow`，运行目录 `run_v2`，机器配置 `machine.v2.json`；绝对项目根仍为 `/projects/varunssd/louis-ssvc/verified_discovery_transfer_20261006`。QOS 为 `soujanya-poria-startfund-2026-03`，account 为 `rose`，每worker一张PRO6000/4 CPU/33G。正式worker时限为分区允许的3天，这是作业上限，不是完成时间预测。

`bridge-numerics` lane 已提交并完成为187112；它先运行真实bridge，再运行固定桥接题的0生成0更新精度诊断。随后原子发布绑定当前bridge哈希的接受记录。

已提交五个正式lane：`v2-lane0=187115`、`v2-lane1=187116`、`v2-lane2=187154`、`v2-lane3=187155`、`v2-lane4=187156`。沿用下方launch.sbatch的5参数接口，传入上述新code/run/machine路径，最后参数为对应lane。不要重复提交或重新生成数据。

只读查询：`squeue -j 187115,187116,187154,187155,187156`。报告使用 `cli report --run .../run_v2`；完整登记矩阵尚未终态，不能传 `--reveal-test-after-completion`。

以下保留最初命令及QOS更正记录。

---

# QOS 更正

账户已确认可以使用 `soujanya-poria-startfund-2026-03`，MaxSubmitJobsPerUser 为 5（运行加排队），没有设置该 QOS 的 MaxJobsPerUser。集群禁止原地修改已提交作业 QOS。实际执行 `scancel --state=PENDING 187023` 后，将下方 sbatch 的 QOS 改为 `soujanya-poria-startfund-2026-03`、CPU 改为 4、内存改为 33G，其他参数相同，重提得到 **187045**，已确认 RUNNING。不要重复提交；后续查询使用 187045。

以下保留此前实际命令作为历史记录。

# 实际服务器命令与恢复入口

以下命令在服务器已验证项目内执行。准备与矩阵登记已经完成；不是要求重新运行这些阶段。调度状态与路径以 [DEPLOYMENT.json](evidence/DEPLOYMENT.json) 为准。

```bash
cd /projects/varunssd/louis-ssvc/verified_discovery_transfer_20261006/code_v1/ssvc_flow
vdt_python=/projects/varunssd/louis-ssvc/envs/ssvc-py312/bin/python
vdt_root=/projects/varunssd/louis-ssvc/verified_discovery_transfer_20261006

# 已执行并登记 294 项；此命令本身不调用模型。
PYTHONDONTWRITEBYTECODE=1 "$vdt_python" -m src.verified_discovery_transfer.cli build-queue \
  --plan "$vdt_root/run_v1/protocol.json" \
  --machine "$vdt_root/machine.server.json" --out "$vdt_root/run_v1"

# 已执行；不揭晓 E/G。
PYTHONDONTWRITEBYTECODE=1 "$vdt_python" -m src.verified_discovery_transfer.cli report \
  --run "$vdt_root/run_v1"

# 下列 sbatch 已返回 187023，不要重复提交。
sbatch --parsable --partition=cluster02 --qos=rose --account=rose \
  --gres=gpu:pro6000:1 --cpus-per-task=8 --mem=64G --time=03:00:00 \
  --job-name=ssvc-vdt-bridge \
  --output="$vdt_root/logs/bridge-%j.out" --error="$vdt_root/logs/bridge-%j.err" \
  scripts/verified_discovery_transfer/launch.sbatch \
  "$vdt_python" "$vdt_root/code_v1/ssvc_flow" "$vdt_root/run_v1" \
  "$vdt_root/machine.server.json" bridge

# 只读检查；尚未启动时日志文件可能不存在。
scontrol show job 187023
sacct -j 187023 --format=JobID,State,ExitCode,Elapsed,AllocTRES
```

Slurm 将该请求调整为 4 CPU、33 GiB、1 张 PRO6000。后续应按实际集群资源规则请求。正式 worker 必须等真实 bridge 完成并核验数值差异后再提交，使用同一冻结 run/queue；不创建新方法、超参数或测试面板。正式提交命令应据实测吞吐设定时限，当前吞吐未知。
