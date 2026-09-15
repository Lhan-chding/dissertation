# CPU 建模资格实验

入口原件：[START_HERE_FOR_CODEX.md](design/START_HERE_FOR_CODEX.md)。
本模块实施四事件表示、CPU 有限动作模型、局部响应回归及离线跟踪。
所有原始实验写入新的 `runs/modeling_qualification/<run_id>/`，保持历史实验和源码不变。

本轮结论：[MODELING_DECISION_zh.md](results_20260915/MODELING_DECISION_zh.md)。
工程总门禁为 BLOCKED（完整回归触及资源预算），科学结论为 NO_ACCEPTABLE_R。
新增模块94项、服务器针对性155项通过；本机适用全套完成1080项后中断，未宣称全套通过。
实际阶段命令与回执见 [ACTUAL_COMMANDS_zh.md](ACTUAL_COMMANDS_zh.md)，逐项验收见 [CPU_ACCEPTANCE_RESULTS_zh.md](CPU_ACCEPTANCE_RESULTS_zh.md)。

## 执行

从本 worktree 的 `ssvc_flow/` 工作目录执行，`PY` 指向现有、已核验的 CPU Python。
配置有完整的冻结校验；修改科学协议需要另行版本化，不能静默修改样本、门槛或划分。

```bash
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
"$PY" -m src.modeling_qualification.cli validate
"$PY" -m src.modeling_qualification.cli math --out runs/modeling_qualification/<run_id>/M0
"$PY" -m src.modeling_qualification.cli witnesses --out runs/modeling_qualification/<run_id>/M1
"$PY" -m src.modeling_qualification.cli collect --profile smoke --out runs/modeling_qualification/<run_id>/smoke
# 依据实际 smoke 和拟合 benchmark 形成同层 preflight.json 后，才允许 core。
"$PY" -m src.modeling_qualification.cli collect --profile core --out runs/modeling_qualification/<run_id>/M2
"$PY" -m src.modeling_qualification.cli fit-evaluate --input runs/modeling_qualification/<run_id>/M2 --out runs/modeling_qualification/<run_id>/M3
"$PY" -m src.modeling_qualification.cli track --input runs/modeling_qualification/<run_id>/M2 --models runs/modeling_qualification/<run_id>/M3 --out runs/modeling_qualification/<run_id>/M4
"$PY" -m src.modeling_qualification.cli audit-real --readonly-root <existing_evidence_root> --out runs/modeling_qualification/<run_id>/M5
"$PY" -m src.modeling_qualification.cli report --input runs/modeling_qualification/<run_id> --out docs/modeling_qualification/<report_id>
```

所有尖括号是待替换参数。不存在的真实证据不能作为输入。每个阶段拒绝覆盖现有输出；
命令、版本、配置、源码/输出哈希和资源回执另存于输出同层 `.receipts/`。

`preflight.json` 必须含实测来源文件哈希、配置对象哈希以及完整阶段的时间、RAM、磁盘推算；
CLI 核对来源未变和 4 小时 / 8 GiB / 2 GiB 预算。不以缩减 test seeds 或删基线解决预算超限。

## 服务器 CPU 授权

用户在本轮实施中追加授权可以使用服务器 CPU 跑测试。
`scripts/modeling_cpu_tests.sbatch` 仅供显式提交本轮 CPU 回归；CLI 不自动提交作业。
执行目录和输出必须是新目录，环境沿用现有 Python，不更新正在运行旧实验的 checkout。
结果须核验调度器退出状态、CPU 节点、TRES 中无 GPU、测试日志和源码哈希。
该追加授权不扩展为 Qwen 调用、GPU、在线检测或 SSVC 控制。

## 解释边界

`REAL_CPU_TOY` 是共享的 737 参数有限动作网络；EVIDENCE_PROXY 是额外数值观测。
工程通过、CPU 建模是否达到预设目标、真实 Qwen 数据准备度分别报告。
所有模型预测针对已发生的实际参数更新；有限步一阶误差和付费校准成本均需计入。
