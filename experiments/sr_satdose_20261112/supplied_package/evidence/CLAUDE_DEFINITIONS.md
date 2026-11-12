# 定义与字段映射（Stage 0 复算）

路径缩写：`LOCAL` = `/Users/louis/Documents/ChatGPT/dissertation-ntu/.worktrees/sr-ord-train-20261103/sr_ord_train`；`BUN` = `LOCAL/execution/final_delivery_20261111/bundle`；`S` = 本目录下的 `scripts/`。

所有定义都已经在全部数据上做过机器核对，核对结果附在每条之后。标【新增口径】的条目是本次复算自己加的，冻结合同里没有，报告中与冻结口径并列给出。

## 1. 基础字段

| 量 | 实际字段 / 计算 | 来源（路径:行） | 核对 |
|---|---|---|---|
| 请求身份 | `KEY=(panel, root_id, structure, parent, train_seed, arm, step, view, draw)`；`eval_raw` 另有 `request_id` | `LOCAL/analyze.py:16`；`S/common.py:16` | `eval_scores` 与 `eval_raw` 的 KEY 集合完全一致（22,880/22,880，`outputs/s00_consistency.txt`） |
| A | 解析出的 `answer` 与 `task.gold.answer` 精确相等 | `LOCAL/score_eval.py:79-84` | 直接使用 `eval_scores.A` |
| `selected` 的解析 | 对 `full_text`（prefill+后缀）做 `json.loads`，拒绝重复键，Decimal；只有 `finish∈{eos,balanced}` 且 `selected` 是字符串列表时才算已知 | `LOCAL/score_eval.py:71-90`；`LOCAL/vendor/probe_ref/order_probe/measurement.py:22-29`；本次重写 `S/common.py:33-44,147-156` | 我的解析结果与 `S_known` 22,880/22,880 一致，与 `S_hit` 22,880/22,880 一致，与 `list_choice_hit` 9,741/9,741 一致，与 `report_choice_hit` 9,809/9,809 一致 |
| S_hit | `set(selected)==set(gold.selected)`，且标签不重复 | `LOCAL/score_eval.py:90-93` | 同上 |
| B | `bindings` 与 gold 操作数字典完全相等 | `LOCAL/score_eval.py:106-117` | 直接使用 `B_hit` |
| U | Alpha 证据覆盖全部类别，并且顺序等于 `audit.expected_report_order` | `LOCAL/score_eval.py:118-132` | 只在 transfer8、scale10、context8/report_conflict 适用（`LOCAL/analyze.py:35-36`） |
| J（评估） | `J_native`：原生 `sr_f6.measurement.score_response(...)["J"]`；不兼容的面板为 null | `LOCAL/score_eval.py:152-164` | authority8/list、prefix10、scale10 为 null，从不当作 0 |
| J（训练） | `training_native_scores_projection.jsonl` 中的 `native_score.J`；J=1 要求 V=1 且 finished=1 | `LOCAL/adapter/training.py:162-164` | 36,864 行全部是 0/1 |
| 证书充分性 | `native_score.certificate_sufficient`（0/1/None），只有训练投影里有 | `LOCAL/vendor/native_src/sr_f6/measurement.py:206` 附近 | None 共 202 行，保留为单独一类 |
| 训练中的选择 | `native_score.witness.selected`；已知 ⇔ `known_mask.S` | 原生回放 `BUN/training/native_replays/*.json` | 我的 exact 判定与原生 `S` 36,864/36,864 一致（`outputs/taskC_training.txt` 第 1 行） |
| 训练组 | `BUN/frozen/training_groups.jsonl`（parent, arm, train_seed, step, slot, channel, family, group_seed）通过 group_id 与投影连接 | — | 4,608 组 × 8；`case_id` 全部一致 |
| 阶段 | `phase ∈ {early_01_08, middle_09_16, late_17_24}` | 投影字段 | 各 1,536 组 |

## 2. 来源四分类（第 5 节定义的实现）

- **T（平局边界块）** = `task.boundary_indices`，按图序；**B（严格更优）** = `task.strictly_better_indices`；**r** = `task.r`，没有时取 `task.required_from_boundary`。与 `LOCAL/score_eval.py:99-103` 的用法相同。
- **chart 集合** = B ∪（T 按图序的前 r 个）。代码：`S/common.py:77-121`。对于图表有权限的题，断言 chart 集合 = `task.gold.selected`；对于 authority8/list，断言 list 集合 = gold。断言在 22,880 条评估回答和 36,864 条训练回答上全部成立。
- **list 集合**：
  - LIST/ENUM/SWAP 渠道：B ∪（T 按 `audit.list_order` 的前 r 个）。断言它等于 `audit.list_induced_selected`。
  - REPORT 渠道：用 `audit.expected_report_order`。
  - transfer8：用 `audit.expected_report_order`，即逆字母序。
  - prefix10：用 `audit.prefix_induced_selected`，即强制前缀诱导的集合。
- **合法（admissible）**：标签不重复，包含全部 B，只取自 B∪T，并且恰好从 T 中取 r 个（`S/common.py:123-145`）。【新增口径】与 `order_probe/measurement.py:103-106` 的 `boundary_choice_admissible` 基本相同，但不另外要求 `len==k`。由于已要求 ⊇B 且 |∩T|=r，这一条自动成立。
- **分类**：`chart`、`list`、`other_admissible`、`inadmissible`、`unknown`（S_known=0）。chart 集合与 list 集合相同的题记为 `same`；训练中这类题记为 `chart_eq_list`，不计入冲突统计。
- **一致题与冲突题**：评估中，集合相同即为一致题。训练中按 `audit.list_conflict` 分层（见第 4 节）。
- **exact authorised**：`set(selected)` 与当前规则指定来源的集合完全相同，即 S_hit。

## 3. 列举顺序与自列举耦合（只读输出文本）

- 列举序列：`evidence` 中 `series=="Alpha"`、类别名合法的条目，按首次出现排序（`S/common.py:159-171`）。不要求数值正确。【新增口径】U 要求完整覆盖且顺序精确，这里只看平局块。
- 平局块列举分类（`S/common.py:174-189`）：
  - `chart`：T 的列举顺序等于图序。
  - `list`：等于清单序、报告序或逆字母序。
  - `chart_eq_list`：两种顺序相同。
  - `other`：其他顺序。
  - `uncovered`：没有覆盖 T 的全部成员。
  - `unknown`：JSON 不可解析。
  - prefix10 没有 `T_alt`，强制颠倒的前缀被记为 `other`。这是强制前缀，不是模型自发的列举。
- own-first 集合 = B ∪（T 按回答自身列举顺序的前 r 个）（`S/common.py:192-193`）。
- 自列举耦合 = 在合法且平局块完整覆盖的回答中，`selected == own-first` 的比例。另外报告"列举顺序 ≠ 图序"的子集。

## 4. 训练分层与组级量

- 分层（`S/s03_taskC_training.py: stratum_of`）：LIST_conflict、LIST_aligned、REPORT_conflict、REPORT_aligned、BASE_TIES、BASE_other。依据是私有题卡中 `audit.list_conflict` 记录的实际排列，不看计划槽位。BASE 中的 TIES 题提示里有一份与图一致的公开清单（已核对 prompt 文本），即"标准"条件。
- 优势：`(J − mean)/总体 SD`，不加 epsilon，SD=0 时为 0。实现见 `LOCAL/adapter/training.py:38-48`，调用处 `:167`。`native_replays` 没有存优势，只存了 J，所以按此公式重算。Task F 代理已在 4,608 组回执上核对，与公式不一致的组数为 0（`outputs/taskF_code_facts.md` §2）。
- 【新增口径】组内定向剂量：`dose = Σ_i a_i (c_i − c̄)`，c = exact authorised 指示量，与 `f22_checks.py` 的 `dose/grp` 定义相同。
- 【新增口径】η²_bin = 按 c 分组的组间平方和 / Σ a²，只在 J 混合组上计算。η²_cls 用全部来源类（含 unknown）代替 c。"signal 组"指 J 混合且 c 混合，即 f22 的定义。
- R_source = 1[selected == 授权集合]。unknown 有两种处理：U0 记为 0；UX 把含 unknown 的组整组排除。

## 5. 面板组合与统计口径

- **冲突面板（176 条/端点）**：context8/list_conflict（24 条）+ swap8/p0l1（64）+ swap8/p1l0（64）+ authority8/chart（24）。与 GPT 第二轮审查使用的组合相同。
- **配对与加权**【新增口径】：
  - response 加权：每条回答等权。
  - root 加权：先在 (panel, root) 内求均值，再对 root 等权。
  - panel 加权、stratum 加权（与 `LOCAL/analyze.py` 的 equal-weight 一致）。
- **bootstrap**：按 root 重抽，固定种子（各脚本文件头已注明）。authority 的 bootstrap 同时抽同一 root 的两个规则视图和两个 seed。
- **α/γ**：`logit P(chart) = α + γ·1[规则指定图表]`（0/1 编码），所以 α = logit P(chart | 规则=清单)。另外并列报告 ±½ 编码下的 `α_mid`（GPT 引用的 −1.151 用的就是这种编码）。分母为全部回答，unknown 计作"非 chart"；另附只看已知回答的版本。
- **Δ** = P(chart | 规则=图表) − P(chart | 规则=清单)，单位为概率（pp）。
