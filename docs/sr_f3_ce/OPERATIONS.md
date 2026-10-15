# SR-F3-CE 独立 CPU dispatcher

`scripts/sr_f3_ce/dispatch_registered.py` 是单独提交、单独部署的运维工具。科学实现保持在 `ROOT/code`，Git identity 必须为 `71483fbf31d57213fdc7e6f3c881848397b45329`。运维工具不修改科学源码、`RUN_IDENTITY.json`、调度 `CONFIG.json`、科学矩阵、采样预算或评估选择。

## 资源与完成门槛

- 固定 `pool_size=1`，每个科学 GPU job 只运行一个已登记任务，完成即释放。
- 仅使用已核实允许 CPU 的 `rose` account、`cluster02` partition、`soujanya-poria-startfund-2026-03` QoS、`cpu-2` node。CPU dispatcher 请求 4 CPU、12 GiB、72 小时，零 GPU，关闭 Slurm 自动 requeue。
- 同一 teacher QoS 的所有用户，实际 GPU 和请求 GPU 取较大值，加上本实验未确定的 reservation，合计不得超过 5。当前用户该 QoS 的全部 CPU/GPU 排队与运行任务，加上未确定的 reservation，提交数不得超过 5。其他 QoS 仍出现在审计快照中，不混计 teacher 额度。
- 因 MaxSubmitPU=5，运行中的 CPU dispatcher 通常留下最多 4 个单 GPU 名额。该工具不会为第五张 GPU 改 QoS、改池大小或削弱单 GPU 恢复语义。
- 所有新名称使用 `SRF3CE-DISPATCH-N0001-LH` 这样的顺序身份并以 `-LH` 结束；科学 worker 继续使用已冻结调度器的 `-LH` 名称。

推进条件是实际完成身份，绝不读取得分来选择路径。真实 P0 和 4 个 ENGINE 全部通过后，还须验证 `external/revisions/typed_v1/PUBLICATION.json`、两位独立审核者的完整原文、qid/input identity 覆盖、原始规则与视觉证据文件、图像、原始选择与模型输入字节、registry 与确定性 reconciliation。参见 [TYPED_AUDIT.md](TYPED_AUDIT.md)。没有发布完成标记时等待；已有完成标记但文件缺失、身份或证据不符时失败关闭。不能将单方 reviewer receipt 或单纯行数当作双审通过。

满足这些条件后，dispatcher 只调用现有 `freeze_science` 一次。固定顺序随后由冻结的 `Controller.tick` 执行：12 条训练全部结束 → 15 个模型固定 endpoint evaluation → release。一个训练或模型结果不会改变其他路径的执行。freeze intent、最终 freeze 和所有 ENGINE 原回执均保留；存在未完成的 freeze intent 时不自动再生成一次。

全部 33 个登记任务真实完成后，才在当前 CPU allocation 内运行同一运维 source manifest 绑定的 `build_delivery.py`。数值 release 完成仅标为 `NUMERICAL_RELEASE_COMPLETE`；只有 builder 成功并且 ZIP 哈希、大小、完整检查回执和所有登记任务覆盖均正确，driver 才写 `DONE`。

## 部署与首次提交

三个脚本必须来自同一份已经提交并推送的运维 Git commit，保持 basename，部署到 `ROOT/operations/<40位运维commit>/`：

```text
dispatch_registered.py
finalize_external_types.py
build_delivery.py
```

发布 typed audit 时，finalizer 的 `--auxiliary-code-commit` 必须是这同一 commit。不要将这些辅助脚本覆盖到 `ROOT/code`。部署完成后检查传输文件的 SHA256 与相应 Git blob 导出的字节一致，再登记运维身份。

以下是服务器入口；`OP_COMMIT` 必须替换为真实完整 commit。登记和提交只做小型身份、源码与队列检查。数据、checkpoint、双审和交付的完整验证在获配的 CPU job 中执行。

```bash
RUN=/projects/_ssd/varunssd/louis-ssvc/sr_f3_ce_20261016
PYTHON=/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12
OP_COMMIT=REPLACE_WITH_VERIFIED_40_CHARACTER_OPERATION_COMMIT
OPS="$RUN/operations/$OP_COMMIT"

"$PYTHON" "$OPS/dispatch_registered.py" --mode register \
  --root "$RUN" --python "$PYTHON" --user "$(id -un)" \
  --operation-commit "$OP_COMMIT"

"$PYTHON" "$OPS/dispatch_registered.py" --mode submit \
  --identity "$OPS/OPERATION_IDENTITY.json" --sequence 1
```

登记生成不可变 `OPERATION_SOURCES.json` 和 `OPERATION_IDENTITY.json`，绑定三个运维脚本、原科学 source manifest、`RUN_IDENTITY.json`、原调度配置、canonical root/Python 与 CPU allocation policy。启动时在导入之前核验冻结科学文件，核验真实 `scontrol` 字段、Slurm 保存的 batch script、release 回执及实际零 GPU allocation。不要从 login 节点手动执行 `--mode run` 或后台启动长轮询。

提交顺序严格为持久化 INTENT → `sbatch --hold` → 核验实际 account/QoS/partition/CPU/memory/lease/name/comment/script → 重新查询容量 → 持久化 RELEASE_INTENT → release → 保存回执。新 CPU 提交同时使用原 `orchestration/SUBMIT.lock` 和运维序号锁；运行中的 dispatcher 使用全 run 的独占 `DRIVER.lock`。

后续科研提交仅由这个 dispatcher 执行。不要同时启动原 `--kind controller` 或手动运行原 `Controller.tick/submit/resume`：已冻结的原调度器不知道运维目录中的未知 CPU intent。共享提交锁防止同一时刻提交，但不授权多个长期调度入口。其他项目和用户的任务继续保留，不取消、不改名。

## 轮询与故障

每 60 秒查询轻量队列和已知 job accounting，保存 `operations/dispatcher/attempts/NNNN/QUEUE_SNAPSHOT.json`。事件指纹只包含容量计数、已知任务的终态或 unknown 身份、typed publication 哈希，不包含 `RunTime` 等每轮变化字段。首次、容量变化、终态变化或 typed 发布完成才执行完整调度门禁；稳定轮询不会反复遍历 checkpoint 原件。

完整验证期间新出现的终态保留到下一轮，不能通过更新轮询指纹将它漏掉。Slurm 队列消失但 accounting 尚无可信终态时，仍是未知状态；不会把任务重新提交。

- `SUBMISSION_UNKNOWN`、`RELEASE_UNKNOWN`、`ACCOUNTING_UNKNOWN`、仍 held 的不完整提交，以及非 TIMEOUT 的失败，写 `END.json` 的 `FAILED` 后停止 dispatcher。它不取消现有 worker，不覆盖失败，不自动重试未知提交。
- 仅对实际终态为 `TIMEOUT` 的单 GPU science train/eval，且 ENGINE、既有科学 freeze、typed audit 与 source identity 均有效时，调用冻结的 `Controller.resume`。checkpoint、完整已提交 raw chain、采样缓存、同种子 partial replay 授权都由原 CPU verifier 核验。任何 verifier 失败继续阻塞，不重新采样来掩盖问题。
- ENGINE 的预算有严格上限，不自动恢复。多 GPU pooled worker、PREEMPTED、普通异常也不由这个辅助工具自动恢复。
- 若失败的 job 已有任务 COMPLETE receipt，进入 receipt-only reconciliation，停止而不重跑；原件和失败终态一起保留。
- CPU 端恢复后每个旧 attempt 与新 attempt 的关系保留在原调度目录和 driver `recoveries/`；运行器自己保存 partial 原文与额外技术采样账本。
- `STOP` 文件会停止 dispatcher 后续操作，现有 worker 的停止语义仍由冻结实现决定。

容量不足发生在任何 INTENT 写入前时，可以在容量实际变化后再尝试同一序号。只要该序号已有 INTENT，就不能重复提交；应检查实际 Slurm job 与所有回执。不要删除 INTENT、STATE 或 failed/unknown 目录以重置状态。

## 72 小时 CPU lease 的显式续接

这是可重启的单实例 driver，**当前版本不会自动提交下一个 CPU driver，也不承诺永久无人干预运行**。离 CPU lease 结束还有 15 分钟时，保存 `END.json: HANDOFF_REQUIRED` 并正常退出；已在运行的科学任务继续运行。72 小时仅是调度租约，不是科研总时间或 GPU 小时预算。

等待前一个 CPU job 从队列消失，并有可信 `sacct` 终态后，使用相同 operation identity 显式提交下一序号。例如：

```bash
"$PYTHON" "$OPS/dispatch_registered.py" --mode submit \
  --identity "$OPS/OPERATION_IDENTITY.json" --sequence 2 --previous 1
```

工具核验序号无缺口、无分叉、前身 operation identity/intent 不变、真实 user/account/QoS/name 对齐。前身 `COMPLETED` 必须同时有 `HANDOFF_REQUIRED`；前身 CPU 直接 `TIMEOUT` 可续接，但不能已有记录为 FAILED 或 DONE 的 END。active、unknown、失败或其他终态一律需要人工核对，不会借续接忽略错误。前身目录、Slurm 终态和失败记录始终保留。

在 full checkpoint 验证或最终 ZIP 构建期间达到 CPU lease，Slurm 可能直接结束进程。新 driver 会重新执行身份检查；同一科学 TIMEOUT 的恢复使用原确定性 successor key，不重复创建科学 attempt。交付构建器按自己的严格恢复规则保留未完成临时 ZIP，或全量核验已发布 ZIP 后只补缺失回执；无模型调用。

## 状态与验证

每个 `operations/dispatcher/attempts/NNNN/` 保留 INTENT、STATE、真实 allocation、脚本、submit/release 输出、STARTED、QUEUE_SNAPSHOT、PROGRESS、恢复/交付回执与 END。`DONE` 必须伴随 `DELIVERY_VERIFIED.json` 及输出 ZIP 的 `.verification.json`。`HANDOFF_REQUIRED`、`FAILED`、`WAIT_TYPED_REVIEW` 或 `NUMERICAL_RELEASE_COMPLETE` 均不是最终科学交付完成。

本地无 GPU 验证命令：

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/louis/Documents/ChatGPT/dissertation-ntu/.venv/bin/python \
  -m pytest -q -p no:cacheprovider tests/sr_f3_ce/test_dispatch_registered.py
/Users/louis/Documents/ChatGPT/dissertation-ntu/.venv/bin/python \
  -m ruff check --no-cache scripts/sr_f3_ce/dispatch_registered.py \
  tests/sr_f3_ce/test_dispatch_registered.py
```

这些 fixture 验证门禁和恢复行为，不代表真实 ENGINE、科学矩阵或最终交付已经完成。
