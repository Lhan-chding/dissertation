# 可执行准备辅助脚本

这两个脚本是科学提交 652 之外的部署辅助 artifact，只调用该提交既有 API；不更改科学源码或计划。它们不提交作业、不执行 audit_parents/export_prior、不调用模型、不选择候选。必须由主任务在完整 PROTOCOL_SELECTION 后的真实 CPU Slurm allocation 中执行。

共同 CLI 参数均为显式 `--old-root`、`--new-root`、`--ablation-root`、`--prep-dir`。所有路径必须已存在；NEW 预先放好被选中的 plan、code、原字节 deployment/SOURCE_MANIFEST.json。NEW 必须为 ABLATION 的同级 selected_v2 或 selected_c2，并与完整 selection 一致。SOURCE_MANIFEST 固定 SHA dae3a506dda4fffee8da0e25a3e0881f4b45fa3259342c4160ecd0c09e03c9e0；科学提交 652e52d530520a276b6420ff0f4ee479dcde6db1。两脚本每次先认证原 PREPARATION_MANIFEST 及其四个文件，再验证并导入 NEW/code 中的科学源码。

1. `prepare_selected_data.py`：要求 NEW/data 尚不存在；源文件全量预检查后用原子 mkdir 保留本次目录，复制 4102 条 data allowlist，重建 FROZEN_DATA_AUDIT、PREPARE_DATA_RECEIPT、PROTOCOL_REVISION_REUSE。最后重新验证实际 CPU allocation 和完整 selection，再发布最终复用回执。失败保留已经生成的 partial，后续调用拒绝覆盖；没有删除/回滚功能。
2. 主任务随后独立执行已有 audit_parents.py 与 export_prior.py，或按照整体依赖安排它们。辅助脚本不会代调。
3. `verify_parent_prior.py`：只读核对 fresh 结果：父审计只允许 package_manifest_sha256 变化，prior 导出只允许 parent_audit_sha256 变化；重新哈希 27 个父复制件、历史身份/FILE_MANIFEST，以及历史 RECORD_INDEX。保留原 MISSING 状态；不声称重新逐个哈希所有 historical raw payload。结果 JSON 输出到 stdout，交给主任务保存新的不可覆盖日志/回执。

```bash
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
"$SRF3_PYTHON" -B "$SRF3_PREP_DIR/prepare_selected_data.py" \
  --old-root "$SRF3_OLD_ROOT" --new-root "$SRF3_NEW_ROOT" \
  --ablation-root "$SRF3_ABLATION_ROOT" --prep-dir "$SRF3_PREP_DIR"

# 这里由主任务执行既有 audit_parents.py 与 export_prior.py。

"$SRF3_PYTHON" -B "$SRF3_PREP_DIR/verify_parent_prior.py" \
  --old-root "$SRF3_OLD_ROOT" --new-root "$SRF3_NEW_ROOT" \
  --ablation-root "$SRF3_ABLATION_ROOT" --prep-dir "$SRF3_PREP_DIR"
```

导入前完整性检查：`verify_code_tree(code, source_manifest) -> dict` 要求清单路径唯一且规范，目录内实际 regular-file 集合与清单完全相等；拒绝 code 根、内部文件或目录符号链接，拒绝 .pyc/.pyo、额外文件、缺失文件、非 regular 文件及哈希/大小变化。`authenticated_context` 在修改 sys.path 和导入科学模块之前执行该检查；static 阶段建过精确目录不能代替每次入口复验。

共享 guard：`prepare_selected_data.require_cpu_allocation() -> dict`，无参数。静态准备阶段 NEW 尚未存在时，revision_ops 可先自行认证 ABLATION/code 与原 SOURCE_MANIFEST，再从本辅助目录 import 此函数。guard 使用已认证的 `sr_f3_ce.orchestration.Slurm.show` 和 `physical_gpu_count`；验证实际 JobId、RUNNING、当前进程 UID/用户、rose account、job name 末尾 -LH、请求和分配 CPU 均正数、请求和分配 GPU 均为零、CUDA_VISIBLE_DEVICES 显式为空，并通过 `scontrol show hostnames` 证明当前 hostname 属于实际分配节点。它只读调度器，不提交或修改作业。

共享 bootstrap：`authenticated_context(args, fresh_data=False) -> dict` 供本目录的只读核验脚本使用；revision_ops 的 static 阶段只复用 CPU guard，不调用这个要求 NEW 已存在的函数。主任务仍负责提交 intent/held/release/terminal 等调度链；guard 不取代那条链。

验证边界：本地仅 AST/CLI/guard 测试；14 个 mocked CPU guard case 与 14 个 synthetic exact-tree case 通过，后者含额外 Python、.pyc/.pyo、重复 manifest row、内部文件/目录 symlink、root symlink、缺失/改写文件、FIFO，以及 authenticated_context 在科学 import 前拒绝的集成断言，两个脚本的真实本地 CLI 都在读取不存在路径之前拒绝非 Slurm 执行。没有查询真实 Slurm 或执行 data 复制。原 PREPARATION_MANIFEST 字节保持不变，新增执行层由 EXECUTABLE_PREPARATION_MANIFEST.json 单独绑定。
