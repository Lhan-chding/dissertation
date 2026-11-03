# Part 2 独立代理复核

审核者：独立代理 `/root/audit_part2`。这是代理复核，不是人类双盲。24 条 BLIND 原文均实际逐条阅读（分 1–8、9–16、17–24 三批无截断显示）。未读取或调用生产 scorer/measurement。

阶段顺序：先阅读评分定义与 BLIND，落盘 EXTRACTION 与 SHA/阶段回执；之后读取 PRIVATE 的 world/query，独立按 Alpha 排序、同值使用原图类别位置，计算正确集合及 Beta 精确有理数聚合，核验全部 24 条提供 gold；INDEPENDENT_FINDINGS 与 SHA 落盘后才读取 FROZEN_SCORES。

## 结果

- 全部 24 条提供 gold 的 indices、selected、operands、answer 均与独立 world/query 推导一致。
- 严格完整 JSON 18 条；6 条前缀回答（第 7–12 条）均缺最终根对象右花括号，不补齐、不从 work 或残缺文本挽救字段。未在可解析对象检测到重复键；原文人工检查未见重复键。
- 独立 A：15/24；S：16 PASS、2 FAIL、6 UNKNOWN。
- 第 2 条只有 9 个 Alpha 证据类别，遗漏 Clock；A/S/B 可确认正确，但顺序和完整证据数值真值 UNKNOWN。
- 第 5 条 selected 正确但 Mug Beta 错报 48（实际 78），最终 49.5 错（真值 57），故 A=0、S=PASS、B=0。
- 第 17 条把 Table 替代 Desk，最终 198 错（真值 174）；第 24 条把 Chair 替代 Stool，最终 20 错（真值 38）；两条 A=0、S=FAIL、B=0。
- 第 24 条 work 的自我纠正不改变错误最终 selected/answer；第 11 条 work 与 binding 数值不一致，但全 JSON 本来非法，不做字段挽救。

## 原始分歧及处理

逐条比较 9 个字段，共 216 个单元，初始一致 210 个；6 条各有一个 B 状态分歧，原始独立判断已保留，不回写。

第 7–12 条的独立脚本把非法 JSON 的 B 编成 0；冻结评分为 null（UNKNOWN）。评分定义未在该简短文档显式给出 B 的未知编码，导致独立复核使用了错误状态约定。非法 JSON 没有可判 bindings，应该保留 UNKNOWN，冻结 null 合理。本分歧属于复核方状态编码差异，并未发现生产评分错误。INDEPENDENT_FINDINGS 及其 SHA 仍保存初始判断；COMPARISON 逐条保留原值、冻结值及上述解释。其余 A、S、rank_order_known、values_true、print、alphabetical、numeric order、order_compliance 全部一致。

EXTRACTION.complete 指 JSON 内容完整且结束状态正常；冻结 complete 指生成结束状态。因此 EOS 但缺根括号可能 frozen complete=1、json_valid=0，这两个意义不混同。未把 complete 的异名语义当生产错误。

C_PRINT/C_ALT 指令服从与 Alpha 数值真值分开判断；前缀字段标 forced，但非法整体 JSON 时顺序 UNKNOWN。B=1 的 15 条、B=0 的 3 条、B UNKNOWN 的 6 条是状态解释后的结果。

样本为固定 24 条，不能证明全体零评分错误，不是科学实验通过门；未改变任何冻结评分、原始回答或任务数据。

## 文件与校验

- `EXTRACTION.jsonl`、`EXTRACTION.sha256`、`STAGE1_RECEIPT.json`
- `INDEPENDENT_FINDINGS.jsonl`、`INDEPENDENT_FINDINGS.sha256`、`STAGE2_RECEIPT.json`
- `COMPARISON.json`：逐条字段对照和分歧保留
- 三个独立辅助脚本仅使用标准库；不导入项目评分器。

## 逐条事实依据

1. `MAIN_ORDER_0038_view1/b1/C_REF/d1` — World/query smallest 4 by Alpha with original category index tie-break gives ['Frame', 'Plate', 'Shelf', 'Mug']; mean Beta ['72', '36', '24', '12'] = 36. Final answer=36; S=PASS; B=1. All 10 Alpha names unique; values_true=True; print=False; alphabetical=False.
2. `MAIN_ORDER_0046_view1/b1/C_REF/d1` — World/query largest 4 by Alpha with original category index tie-break gives ['Plate', 'Lamp', 'Desk', 'Watch']; mean Beta ['24', '66', '72', '30'] = 48. Final answer=48; S=PASS; B=1. Only 9 Alpha names: order/value completeness UNKNOWN.
3. `MAIN_ORDER_0047_view1/b1/C_PRINT/d0` — World/query largest 4 by Alpha with original category index tie-break gives ['Shelf', 'Spoon', 'Plate', 'Lamp']; sum Beta ['42', '66', '54', '36'] = 198. Final answer=198; S=PASS; B=1. All 10 Alpha names unique; values_true=True; print=False; alphabetical=False.
4. `MAIN_ORDER_0007_view0/b1/C_PRINT/d1` — World/query smallest 3 by Alpha with original category index tie-break gives ['Bowl', 'Chair', 'Frame']; mean Beta ['30', '42', '78'] = 50. Final answer=50; S=PASS; B=1. All 10 Alpha names unique; values_true=True; print=True; alphabetical=False.
5. `MAIN_ORDER_0041_view0/b1/C_ALT/d1` — World/query smallest 4 by Alpha with original category index tie-break gives ['Bowl', 'Mug', 'Table', 'Frame']; mean Beta ['30', '78', '84', '36'] = 57. Final answer='49.5'; S=PASS; B=0. Binding defects: [{"category": "Mug", "expected": "78", "observed": "48"}]. All 10 Alpha names unique; values_true=True; print=False; alphabetical=False.
6. `MAIN_ORDER_0042_view0/b1/C_ALT/d0` — World/query largest 4 by Alpha with original category index tie-break gives ['Chair', 'Desk', 'Stool', 'Tray']; sum Beta ['24', '60', '72', '18'] = 174. Final answer=174; S=PASS; B=1. All 10 Alpha names unique; values_true=True; print=False; alphabetical=False.
7. `MAIN_ORDER_0007_view1/b1/P_STABLE/d0` — World/query smallest 3 by Alpha with original category index tie-break gives ['Bowl', 'Chair', 'Table']; mean Beta ['30', '42', '78'] = 50. Missing final root brace: invalid JSON; A=0, S=UNKNOWN, B=0, order metrics UNKNOWN. 
8. `MAIN_ORDER_0021_view0/b1/P_STABLE/d0` — World/query largest 3 by Alpha with original category index tie-break gives ['Tray', 'Vase', 'Plate']; sum Beta ['60', '12', '48'] = 120. Missing final root brace: invalid JSON; A=0, S=UNKNOWN, B=0, order metrics UNKNOWN. 
9. `MAIN_ORDER_0040_view1/b1/P_REVTIE/d0` — World/query smallest 4 by Alpha with original category index tie-break gives ['Bowl', 'Mug', 'Book', 'Stool']; sum Beta ['78', '36', '72', '24'] = 210. Missing final root brace: invalid JSON; A=0, S=UNKNOWN, B=0, order metrics UNKNOWN. 
10. `MAIN_ORDER_0017_view0/b1/P_REVTIE/d0` — World/query smallest 3 by Alpha with original category index tie-break gives ['Plate', 'Table', 'Stool']; mean Beta ['78', '48', '84'] = 70. Missing final root brace: invalid JSON; A=0, S=UNKNOWN, B=0, order metrics UNKNOWN. 
11. `MAIN_ORDER_0001_view0/b1/P_PLACEBO/d0` — World/query largest 3 by Alpha with original category index tie-break gives ['Shelf', 'Mug', 'Bowl']; mean Beta ['54', '72', '24'] = 50. Missing final root brace: invalid JSON; A=0, S=UNKNOWN, B=0, order metrics UNKNOWN. 
12. `MAIN_ORDER_0025_view0/b1/P_PLACEBO/d0` — World/query smallest 4 by Alpha with original category index tie-break gives ['Table', 'Clock', 'Desk', 'Shelf']; mean Beta ['30', '66', '84', '48'] = 57. Missing final root brace: invalid JSON; A=0, S=UNKNOWN, B=0, order metrics UNKNOWN. 
13. `MAIN_ORDER_0020_view1/b2/C0/d1` — World/query largest 3 by Alpha with original category index tie-break gives ['Stool', 'Book', 'Table']; sum Beta ['18', '66', '30'] = 114. Final answer=114; S=PASS; B=1. All 10 Alpha names unique; values_true=True; print=True; alphabetical=False.
14. `MAIN_ORDER_0006_view0/b2/C0/d0` — World/query largest 3 by Alpha with original category index tie-break gives ['Vase', 'Shelf', 'Bowl']; sum Beta ['66', '24', '42'] = 132. Final answer=132; S=PASS; B=1. All 10 Alpha names unique; values_true=True; print=True; alphabetical=False.
15. `MAIN_ORDER_0020_view0/b2/C_NOLIST/d1` — World/query largest 3 by Alpha with original category index tie-break gives ['Bowl', 'Book', 'Table']; sum Beta ['18', '66', '30'] = 114. Final answer=114; S=PASS; B=1. All 10 Alpha names unique; values_true=True; print=True; alphabetical=False.
16. `MAIN_ORDER_0034_view0/b2/C_NOLIST/d0` — World/query smallest 4 by Alpha with original category index tie-break gives ['Plate', 'Table', 'Frame', 'Vase']; sum Beta ['12', '24', '36', '48'] = 120. Final answer='120'; S=PASS; B=1. All 10 Alpha names unique; values_true=True; print=True; alphabetical=False.
17. `MAIN_ORDER_0042_view0/b2/C_ENUM/d0` — World/query largest 4 by Alpha with original category index tie-break gives ['Chair', 'Desk', 'Stool', 'Tray']; sum Beta ['24', '60', '72', '18'] = 174. Final answer=198; S=FAIL; B=0. Binding defects: [{"category": "Desk", "expected": "60", "observed": "None"}]. All 10 Alpha names unique; values_true=True; print=False; alphabetical=False.
18. `MAIN_ORDER_0000_view1/b2/C_ENUM/d0` — World/query smallest 3 by Alpha with original category index tie-break gives ['Lamp', 'Chair', 'Vase']; sum Beta ['42', '54', '12'] = 108. Final answer=108; S=PASS; B=1. All 10 Alpha names unique; values_true=True; print=False; alphabetical=False.
19. `MAIN_ORDER_0038_view1/b2/C_REF/d1` — World/query smallest 4 by Alpha with original category index tie-break gives ['Frame', 'Plate', 'Shelf', 'Mug']; mean Beta ['72', '36', '24', '12'] = 36. Final answer='36'; S=PASS; B=1. All 10 Alpha names unique; values_true=True; print=True; alphabetical=False.
20. `MAIN_ORDER_0020_view0/b2/C_REF/d1` — World/query largest 3 by Alpha with original category index tie-break gives ['Bowl', 'Book', 'Table']; sum Beta ['18', '66', '30'] = 114. Final answer=114; S=PASS; B=1. All 10 Alpha names unique; values_true=True; print=True; alphabetical=False.
21. `MAIN_ORDER_0033_view0/b2/C_PRINT/d0` — World/query smallest 4 by Alpha with original category index tie-break gives ['Bowl', 'Shelf', 'Frame', 'Chair']; sum Beta ['36', '66', '78', '18'] = 198. Final answer=198; S=PASS; B=1. All 10 Alpha names unique; values_true=True; print=True; alphabetical=False.
22. `MAIN_ORDER_0017_view0/b2/C_PRINT/d0` — World/query smallest 3 by Alpha with original category index tie-break gives ['Plate', 'Table', 'Stool']; mean Beta ['78', '48', '84'] = 70. Final answer=70; S=PASS; B=1. All 10 Alpha names unique; values_true=True; print=True; alphabetical=False.
23. `MAIN_ORDER_0026_view1/b2/C_ALT/d1` — World/query largest 4 by Alpha with original category index tie-break gives ['Lamp', 'Table', 'Tray', 'Bowl']; mean Beta ['12', '42', '18', '36'] = 27. Final answer='27'; S=PASS; B=1. All 10 Alpha names unique; values_true=True; print=False; alphabetical=True.
24. `MAIN_ORDER_0005_view1/b2/C_ALT/d0` — World/query smallest 3 by Alpha with original category index tie-break gives ['Watch', 'Stool', 'Spoon']; mean Beta ['18', '66', '30'] = 38. Final answer=20; S=FAIL; B=0. Binding defects: [{"category": "Stool", "expected": "66", "observed": "None"}]. All 10 Alpha names unique; values_true=True; print=False; alphabetical=True.
