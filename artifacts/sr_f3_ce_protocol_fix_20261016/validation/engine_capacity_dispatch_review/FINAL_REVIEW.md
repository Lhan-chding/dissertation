# ENGINE capacity dispatcher independent review

结论：**READY_FOR_SCOPED_OPERATIONAL_DEPLOYMENT_REVIEW；最终版无剩余 blocking finding。**

基线为 `617dd00`，科学源码固定为 `652e52d530520a276b6420ff0f4ee479dcde6db1`。只读检查新的 operational dispatcher、直接匹配测试、相关文档及冻结 Controller/observer 调用链；没有服务器命令、真实 Slurm 查询、模型调用、训练或采样，没有改源码、他人 helper、selection、Git 状态。真实作业 207422/207434 等状态来自父任务的执行背景，本审查未独立查询其当前状态。

## 审查对象和最终字节

| 文件 | SHA256 |
|---|---|
| dispatch_registered.py（最终） | `be997834324f7a48ca8a314344ff161af9dcae3622ea092dae17d59c8bc9e955` |
| test_dispatch_registered.py（最终） | `4e3d3e23523c1d8bd58aa193fef136076d9394b99b529cfe9f7c90542b27f1ca` |
| docs/sr_f3_ce/PROTOCOL_REVISION.md | `8ae0c418730c03404c024b3d514f04772aa248c27ec615f4df158220b452cfe8` |

测试前后 source/test SHA 一致。`FINAL_SCOPED_DIFF.patch` 保存相对617dd00的实际审查 diff，`FINAL_SOURCE_IDENTITY.json` 保存 bytes/hash。整个 `src/` 相对652及617dd00均无diff；冻结 orchestration/bootstrap/instrumented_execution/observe_runtime 的逐文件比较见 `FROZEN_DEPENDENCIES.json`。

## 初始发现和修复复验

**F1：初版允许 LOCAL_P0 自动提交 ENGINE（已修复）。**

初版脚本 SHA 为 `e5594ac1d0164507e238ea167120de507a3046ece2b6f5b10386e0ded52752c7`。冻结 Controller.completed() 有意支持合法 LOCAL_P0 回执，新 Driver.act 初版只看 completed 是否含 p0，因此未证明真实 P0 Slurm allocation。独立用未修改的 Controller 和现有 complete_local_p0 fixture 复现：零真实 P0 allocation，合法本地 P0 receipt，Driver.poll 返回 DISPATCHING_ENGINE，创建原四个 ENGINE intent 并发出四次 **fake** sbatch。没有真实 sbatch、模型输出或 SCIENCE freeze。

该失败保留于 `LOCAL_P0_e5594ac1d016.json`；`candidate_e5594ac1_dispatch_registered.py.txt` 保存精确初版源字节，文本快照不作为执行源码。初版独立测试真实结果为1 failed，不删除或改写为通过。此缺陷不是现场已完成的真实 P0 作业重新失败，而是新 helper 必须收紧的通用前提。

最终版增加 require_slurm_p0，位置在 no-freeze 公共入口：如果 status.completed 含p0，则必须找到唯一 COMPLETE 的p0 allocation；调用原 Controller.verify_job 完整核验 held allocation、脚本、release、job/claim、worker launch/exit、task receipt与保存的COMPLETED/0:0终态；再要求实时 Slurm.terminal(job_id) 与保存的认证终态完全一致。LOCAL_P0、缺失实时 accounting、失败状态、exit或用户变化均在任何新 ENGINE submit、typed核验或FREEZE_INTENT之前拒绝。此检查覆盖 ENGINE 未全完成和全部完成两条路径。

**原样**复跑同一 `test_independent_boundary.py`：最终版1 passed，PermissionError，零 ENGINE intent、零 fake sbatch、零 science freeze，见 `LOCAL_P0_be997834324f.json`。保留的独立测试文件 SHA 为 `0b427acecd8ba782fbd3ef3e4a827cc3322e20d0d699f7c9924d4e5f956e71bf`，复验没有修改失败用例。

补充边界：原652观测扩展 seal_amendment 自身已经明确拒绝所有 ENGINE_IDS（含p0）中任何 LOCAL_P0，所以初版在真实完整扩展中不会因此放行 SCIENCE；但是公共helper门禁仍有必要，既防提前生成 ENGINE，又避免全完成路径先写FREEZE_INTENT再由扩展拒绝。bootstrap.freeze_science仅靠Controller.completed的检查本身不能被当作排除LOCAL_P0的独立证明。

## 最终行为核对

- **任务范围和重试**：新分支只调用原 `Controller.submit("engine")`，循环至多四次，每次只在原四个 ENGINE 的 ready 集合非空时调用。冻结 Controller 在提交锁内重算 ready/claimed，从所有已有 intent 排除已认领任务；不新增任务、模型、seed、更新步或采样预算。None 表示容量或可提交任务已耗尽，停止本轮。SUBMISSION_UNKNOWN、RELEASE_UNKNOWN、HELD、FAILED等非SUBMITTED返回立即报错；刷新后任何blocked也停止，不重试。Driver.recover仍只允许既有训练/评价TIMEOUT恢复，ENGINE失败和ENGINE TIMEOUT均不能借此自动重试。
- **真实容量**：每次sbatch前由原Controller重新取queue并refresh；teacher物理GPU按所有用户请求/分配较大值计数，用户submitted上限还包含CPU dispatcher；pending/unknown intent保留占用。提交后仍走原hold → 实际字段/脚本校验 → 重新计容量 → release。未触碰无关任务。测试明确覆盖3个无关teacher GPU任务＋手动ENGINE＋CPU dispatcher的五个提交位，以及空位释放后只补未认领的固定任务。
- **SCIENCE顺序**：no-freeze路径先校验真实P0；ENGINE未齐即返回WAIT_ENGINE或DISPATCHING_ENGINE，不调用typed、seal、freeze或scientific tick。全部ENGINE完成后仍依次typed独立核验 → 原seal_amendment → 原freeze_science → 验完整extension → 固定科学任务。原extension和科学源保持字节不变。
- **snapshot可继续推进**：poll保留action前snapshot签名；teacher usage变化、新的已登记job/intent、已登记job终态变化或typed publication变化会再次调用act。满额等待时不会循环重复提交，新增提交后下一轮会观测到其登记状态；测试覆盖WAIT → 空位释放 → DISPATCH → WAIT/UNCHANGED → 再释放补位。未改变poll/72h租期/交接逻辑。
- **v1及已冻结边界**：原source/operation identity验证未改变，测试仍以原v1身份走实际Controller；未冻结的LOCAL_P0路径现在有意拒绝，真实P0路径继续工作。已有EXECUTION_FREEZE路径不新增这一前瞻P0检查，继续原typed/extension/freeze链。STOP由既有run和Controller.identity路径拒绝；本次不修改或恢复旧STOP根。

## 实际独立验证

1. `python -B -m pytest -q -p no:cacheprovider --basetemp=/private/tmp/srf3-engine-capacity-review/pytest_final tests/sr_f3_ce/test_dispatch_registered.py tests/sr_f3_ce/test_orchestration.py`：**132 passed in 5.05s**。
2. 同一独立 LOCAL_P0 复现测试：**1 passed in 0.09s**。
3. 上述132项含本次10个真实P0公共入口回归：LOCAL_P0两阶段，live accounting缺失/FAILED/exit/user变化×两阶段，均验证不产生FREEZE_INTENT；并覆盖未知提交、实际held、提交间失败、完整四ENGINE后才freeze、既有容量/恢复行为。
4. 只读比较确认整个scientific src相对652和617dd00无diff，冻结native extension/observer也完全相同。

日志见 `FINAL_TESTS.txt`、`INDEPENDENT_LOCAL_P0_FIXED.txt` 和 `FINAL_TEST_RESULTS.json`。这是本地CPU scheduler fixture：Slurm传输和worker输出为fake，部分run/source验证被明确fixture替代，未运行真实GPU、真实任务或新输出；不能用这些PASS代替实际CPUdriver的allocation、held/release和terminal证据。实现代理报告的Ruff/format结果没有冒称本审查另行执行。

## 部署和证据边界

最终辅助源应按父任务既定方式提交后独立部署到 `operations/<新commit>/` 并登记对应source identity；已有652 code/native extension、RUN_IDENTITY、旧operation目录保持原字节。本审查未执行部署、commit或push，不证明远端helper已更新。真实CPUdriver仍须保留其名字末尾-LH、held/no-requeue、allocation/script/release、单实例锁、72小时租期及最终terminal证据；本地审查不增加实验预算，也不解除原科学门禁。
