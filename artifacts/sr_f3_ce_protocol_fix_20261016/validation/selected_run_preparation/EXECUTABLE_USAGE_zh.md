# 实际 CPU 初始化入口

`initialize_selected.sh` 是给已审核薄 Slurm launcher 的 Bash payload；本地只做语法检查，未执行。它有意不调用 sbatch、不提交 P0/ENGINE/science；每个阶段重新调用完整 verify_selection，只有真实 PROTOCOL_SELECTED 才能运行。

部署布局固定为：

```text
NEWBASE/deployment/selected_init/initialize_selected.sh
NEWBASE/deployment/selected_preparation/selected_run_preparation/  本目录完整准备件
NEWBASE/deployment/selected_preparation/selected_data_preparation/  data代理完整准备件
NEWBASE/retained_history_v1/SR_F3_CE_PROTOCOL_STOP_EVIDENCE_FULL_UNDER400MB_20261016.tar.xz
NEWBASE/retained_history_v1/SR_F3_CE_PROTOCOL_STOP_EVIDENCE_FULL_UNDER400MB_20261016.tar.xz.verification.json
```

NEWBASE 固定为 `/projects/_ssd/varunssd/louis-ssvc/sr_f3_ce_protocol_fix_20261016`。所有 payload/helper/allowlist/source 文件均按最后 manifest 逐 hash 核对。payload 在 register 前安装为 regular canonical 文件并 chmod 0444；执行者将其实际路径与 SHA 交给 launcher。若需改变部署目录，只能在登记前明确调整 payload、重做 bash 语法检查和 SHA 登记，不能在已登记后改动字节。

初始 CPU intent、Slurm script/stdout/stderr/allocation/terminal 在 NEWBASE/deployment/selected_init；不得提前建立 selected_* 根。CPU task 使用实际 QoS `soujanya-poria-startfund-2026-03`、account rose、cluster02/cpu-2、0 GPU，job name -LH；启动后 helper 逐次查询真实当前 RUNNING allocation/UID/host，不能在登录节点靠设 SLURM_JOB_ID 冒充。

四阶段准确调用参数：

```sh
COMMON=(--ablation-root "$ABL" --old-root "$OLD" --prep-dir "$OPS_PREP" --data-prep-dir "$DATA_PREP")
"$PYTHON" -B "$OPS_PREP/prepare_selected_run.py" --mode static "${COMMON[@]}" --history-archive "$HISTORY"
# prepare_selected_data -> audit_parents -> export_prior -> verify_parent_prior
"$PYTHON" -B "$OPS_PREP/prepare_selected_run.py" --mode external-v0 "${COMMON[@]}"
# audit_native.py --root "$NEW"
"$PYTHON" -B "$OPS_PREP/prepare_selected_run.py" --mode seal "${COMMON[@]}"
# original 4146 finalizer read-only validate, then --publish
"$PYTHON" -B "$OPS_PREP/prepare_selected_run.py" --mode typed-receipt "${COMMON[@]}"
```

实际依赖顺序已全部包含在 payload，无需手工填 RUN_IDENTITY 的 hash。static 从原 ABL/code 完整 regular-file inventory复制652源码（与原 archive等价），不会用当前 Git HEAD。static 先校验所有 source/allowlist/history 原件再排他创建；如果 selected_* 已存在，立即停止。所有 copy/write 使用排他写入；失败保留 partial，后续不得删除失败证据重跑。external-v0 会精确重建新路径的 v0，完整复核原可见输入/pixel/history/原review；seal拒绝错误revision/不完整native；typed-receipt只接受4146实际已发布的本次RUN_IDENTITY绑定。

同一真实 CPU allocation 内可串行执行 native → seal → 4146 publish → typed-receipt。RUN_IDENTITY不绑定活动 external/PANELS，不绑定当前job仍在写的日志或未来terminal；typed复用receipt明示当前job终态没有在运行中被证明。payload结束后必须由主执行者核 squeue消失、sacct COMPLETED/0:0并保留外部ledger，然后才能提交新P0。

后续仍是实际新P0 + engine--b0/b1/b2/resume全部真实终态成功，才能由652 registered driver封AMENDMENT和EXECUTION_FREEZE。这个payload的完成不是ENGINE通过，也不授权科学训练越过该门禁。
