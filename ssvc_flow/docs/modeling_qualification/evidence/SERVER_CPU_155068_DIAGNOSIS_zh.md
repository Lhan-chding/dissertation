# 首次服务器 CPU 回归失败归类

## 结论

原始记录位于 `runs/modeling_qualification/qualification_20260915/server_CPU_155068/`。JUnit 共 1,341 项：1,123 passed、55 failed、163 errors。218 项失败或错误中，215 项由隔离 checkout 缺少历史 fixture 引起；其余为新 W3 跨平台失败 2 项和旧 R4 测试的遍历顺序依赖 1 项。这些计数属于代码回归测试，不是 M2–M4 实验失败数。

| 原因 | failed | errors | 合计 |
|---|---:|---:|---:|
| 缺 `data/generated/train.jsonl` | 5 | 87 | 92 |
| 缺 `data/generated/manifest.json` | 29 | 42 | 71 |
| 缺 `data/generated/calibration.jsonl` | 3 | 7 | 10 |
| 缺历史 R3-warm parent metadata | 15 | 27 | 42 |
| 新 W3 `same_theta` 断言及其 runner 连带失败 | 2 | 0 | 2 |
| 旧 R4 `lstat` 重试测试依赖文件遍历顺序 | 1 | 0 | 1 |

## 历史 fixture 恢复范围

[相对路径清单](server_CPU_155068_fixture_paths.txt) 以 `/Users/louis/Documents/ChatGPT/dissertation-ntu` 为源根目录，共 1,307 个原文件、55,669,574 字节。可据此制作仅包含明确成员的传输归档；[逐文件清单](server_CPU_155068_fixture_manifest.json) 保存 SHA-256 和字节数。

- `ssvc_flow/data/generated/`：仅恢复 train、dev、ood、calibration、control 五个 JSONL、原 manifest/schema 和这五个 split 引用的 1,296 张图，共 1,303 个文件、21,226,341 字节。所有关联图像均已核对 scene 中原 `image_hash`。
- `reports/SSVC_GPT_PRO_DELIVERY_20260914/02_data/R3_warm/`：原 `runtime_lock.json`、`bank_manifest.json`、`candidate_manifest.json`，其 SHA-256 与原 loader 的固定值一致。
- `reports/SSVC_GPT_PRO_DELIVERY_20260914.zip`：原归档，供旧测试比较目录和 ZIP metadata 入口。检查中央目录未发现含 `confirm` 的成员名。
- 既有 contact-sheet manifest、legacy reward-fiber 文件和旧源码已受 Git 跟踪，不属于本次缺项。

本次不读取或复制真实 `confirm.jsonl`，也不恢复 natural_pool、diagnostic、access 及其额外图像。旧 `test_cross_split_passes_generated_dataset` 会读取目录中每个现存 JSONL；五个 split 的通过不能充当完整历史 split 隔离验证，因此服务器重跑应明确排除：

```text
tests/test_audit_r0_remaining.py::test_cross_split_passes_generated_dataset
```

这项排除属于 sealed-data 边界；不修改旧隔离检查源码，也不补造隔离证据。历史 manifest 保留其原始哈希声明。旧 R4 loader 原本仅加载上述五个 split，并继承历史 R0 的 confirm 隔离证据。

## R4 测试的 RED / GREEN

变更仅在 `tests/test_r4_continuation.py`。`src/r4_continuation.py` 保持原字节，前后 SHA-256 见 [验证记录](server_CPU_155068_r4_test_red_green.json)。

原测试假定 `os.walk` 在 `checkpoint.pt` 之前访问 `identity.json`。Linux 文件遍历顺序不同，第一次扫描可在访问 identity 前遇到注入的 ENOENT；第二次完整重试正确完成，但 identity 的访问次数是 1。原始服务器日志中，完整 inventory 相等和 checkpoint 两次访问的断言已经通过。

本地先用外层 `os.walk` 包装器强制 checkpoint 优先，实际复现 `assert 1 == 2`：**1 failed in 0.20s**。随后只在该测试的 `lstat` 分支显式安排 identity 在故障前访问，保留完整 inventory 相等、checkpoint 重试两次、identity 重验两次的全部断言。

在相同 checkpoint 优先的外层环境下重跑整个测试文件：**61 passed in 0.47s**；该文件 Ruff 检查通过。记录附有复现 harness 和具体测试参数。该验证不替代父任务负责的服务器全套 CPU 重跑；W3 修复由其负责的模块另行验证。
