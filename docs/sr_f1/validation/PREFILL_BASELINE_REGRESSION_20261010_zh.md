# SR-F1.1 旧输出只读回归记录

数据为已交付的 `SR_F1_CLAUDE_FACTS_AND_CODE_20261010` 固定快照：before 256 条、confirm 256 条、after 113 条。未加入后续 after 记录。评分器为该包内经原始 `PACKAGE_SHA256.json` 认证的官方实现；覆盖定义保持 `L_json == 1 && L_answer == 1 && L_evidence == 1`。

| 面板 | 原始记录数 | 以 `{` 开头 | 原文严格覆盖 | 配平截断后严格覆盖 |
| --- | ---: | ---: | ---: | ---: |
| before | 256 | 96 | 93 | 93 |
| confirm | 256 | 104 | 100 | 100 |
| after 固定快照 | 113 | 47 | 44 | 44 |
| 合计 | 625 | 247 | 237 | 237 |

对方案所述 247 条记录，官方严格覆盖为 **237/247 = 95.9514%**；方案写出的 **239/247 未复现**。Wilson 95% 区间为 `[92.7091%, 97.7863%]`。这些比例来自旧输出的条件子集，不是预填协议新生成结果。

与方案数字相关的两条严格规则失败：

| 原始记录 | 官方结果 | 原始数据及错误信息 | 原文件 SHA-256 |
| --- | --- | --- | --- |
| `before/format-0011-line-v0-6.json` | `L_json=1, L_answer=0, L_evidence=1` | `"answer": "56.0"`；`ValueError: Unsupported numeric representation` | `2cfa7d6ff61d48ba58ec9a52cdd59d78d48e39ccde35d280d89b51f2be94d57a` |
| `confirm/format_confirm-0008-line-v0-1.json` | `L_json=1, L_answer=1, L_evidence=0` | 证据实体重复，官方评分器在第一个重复实体处停止，`duplicate_count=1`；完整原文含 `Alpha/February`、`Beta/January`、`Beta/February`、`Alpha/March` 重复 | `9601805b263abc394d2285ef58c6a7a4a57f7418e075143ab70175f823ba0514` |

confirm 的该条差异是证据实体重复，非方案 §1.1 表格所称答案类型问题。247 条中有 2 条未获得顶层配平点；10 条严格覆盖失败的原文、评分结果和文件标识均保存在本地完整回执。原文完整尾巴含非空白字符的有 2 条，二者在配平截断后的官方评分中也不通过，因此整个旧输出尾巴检查后的覆盖仍为 237/247。

旧输出已继续生成到 EOS/长度限制。对它们检查“整个旧输出的尾巴”，无法直接判定新协议“完成配平的单个 token 内尾巴”；本回归没有加载 tokenizer，也未把该检查标记为新协议 token 边界验收。

方案提及的 `/Users/louis/Downloads/f1_analyze.py` 和 `/Users/louis/Downloads/balanced_json_stop.py` 在本次指定路径检查中均不存在，作者的近似评分脚本未执行。方案 Markdown SHA-256 为 `113efeaf2a5f3ad367e3f16577a9ca0be5abdc7d1658604a85a6eb3f075854b5`。

复验脚本：`scripts/sr_f1/audit_prefill_baseline.py`。跟踪摘要：`docs/sr_f1/validation/PREFILL_BASELINE_REGRESSION_20261010.json`，含统计、247 条记录标识/哈希/官方分类及完整回执哈希，不含原始回答。完整回执：`artifacts/sr_f1_amendment_20261010/PREFILL_BASELINE_REGRESSION_FULL.json`，保存本地、由 Git 忽略。在仓库根执行：

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/louis/Documents/ChatGPT/dissertation-ntu/.venv/bin/python scripts/sr_f1/audit_prefill_baseline.py
```

脚本验证 625 条原始记录文件 SHA-256、record_hash、任务清单哈希和官方评分器哈希。纯配平函数 10 条自检通过；当前生产 `json_protocol.balanced_cut` 对 247 条记录的返回值与独立审计实现一致。`ruff check scripts/sr_f1/audit_prefill_baseline.py` 通过。模型调用、tokenizer 加载、新生成、TEST 记录评分均为 0。
