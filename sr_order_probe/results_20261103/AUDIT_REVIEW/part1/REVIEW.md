# part1 独立代理复核

审核身份：独立 Codex 代理 `/root/audit_part1`。这是代理复核，不是人类双盲。样本复核不能证明总体零错误。

## 阶段与完整性

已实际阅读 BLIND.jsonl 全部24条原文（包含 work、完整JSON结构和异常结尾）。先封存 EXTRACTION.jsonl，再读取 PRIVATE.jsonl 的 world/query；按真实 Alpha 与图上位置打破同值，独立计算选择集合、Beta绑定和精确有理数答案；24条提供的gold全部核对一致。保存 INDEPENDENT_FINDINGS.jsonl 后才首次读取 FROZEN_SCORES.jsonl。未读取或调用生产scorer/measurement。

阶段1 SHA256：`c37a89cd24aeaa714f017c1a9bcae2f14b4743065de9aa5f41dca6724268345d`。
阶段2 SHA256：`2dcfd3dde891c336b78bf5a6c92316c26fc7ce13b4b49f2995e9e3d4b06d1853`。

## 逐字段比较

24条 × 9字段 = 216项比较；首次独立判断有2项分歧，涉及2条，全部为B的UNKNOWN编码；其余214项一致。原始独立判断已封存未覆盖，COMPARISON.json保留分歧和裁定。

第16、17条缺最后闭合大括号。独立首判将“无法核验B”编码为0；冻结评分使用null。裁定：应保留B=UNKNOWN/null，不能把无可判绑定当成已知绑定错误。这是本代理初判状态编码不够细，不是生产评分器错误。二者A=0、S=UNKNOWN、顺序UNKNOWN均一致。

EXTRACTION中的complete表示完整可解析对象；冻结complete表示生成终止（eos也计完成）。这两个字段含义不同，不能据此指认生产完成率错误；本比较的主要字段不混用它们。

独立A=1共15条；S为{'PASS': 15, 'FAIL': 6, 'UNKNOWN': 3}；顺序完整可判19条。

## 需要保留的事实

- 第18条缺selected：S未知；不能从work补抽。
- 第21条重复Alpha类别；第22、23条仅提供4个Alpha类别：顺序均未知，但其A/S/B可独立正确。
- 第11、12条C_ALT均未严格按全局字母序，order_compliance=False；数值仍真实。
- 第14条work先写174后改216；最终answer=216独立正确，不挑work中的数字评分。
- 第4、6、15、20条边界同值选择错误；第5条选Clock而不是Desk；第24条选Mug而不是Book，均保留为失败。
- P_* evidence标为forced，不把它们当作模型自发证据真实性。

## 每条事实依据

1. `MAIN_ORDER_0015_view0/b0/C0/d1` — A=1, S=PASS, B初判=1, rank_order_known=True. Sort Alpha largest with original plot position as tie-break; required ['Bowl', 'Plate', 'Stool'], Beta {'Bowl': '60', 'Plate': '42', 'Stool': '72'}, mean=58. Final answer='58'; selected=['Stool', 'Plate', 'Bowl']; missing/wrong gold bindings=[]. Alpha category count=10, complete unique coverage=True.
2. `MAIN_ORDER_0030_view1/b0/C0/d0` — A=1, S=PASS, B初判=1, rank_order_known=True. Sort Alpha largest with original plot position as tie-break; required ['Vase', 'Shelf', 'Stool', 'Spoon'], Beta {'Vase': '60', 'Shelf': '66', 'Stool': '24', 'Spoon': '72'}, sum=222. Final answer=222; selected=['Vase', 'Shelf', 'Stool', 'Spoon']; missing/wrong gold bindings=[]. Alpha category count=10, complete unique coverage=True.
3. `MAIN_ORDER_0012_view1/b0/C_NOLIST/d1` — A=1, S=PASS, B初判=1, rank_order_known=True. Sort Alpha largest with original plot position as tie-break; required ['Vase', 'Plate', 'Tray'], Beta {'Vase': '54', 'Plate': '84', 'Tray': '48'}, sum=186. Final answer=186; selected=['Plate', 'Vase', 'Tray']; missing/wrong gold bindings=[]. Alpha category count=10, complete unique coverage=True.
4. `MAIN_ORDER_0016_view1/b0/C_NOLIST/d1` — A=0, S=FAIL, B初判=0, rank_order_known=True. Sort Alpha largest with original plot position as tie-break; required ['Tray', 'Clock', 'Spoon'], Beta {'Tray': '30', 'Clock': '60', 'Spoon': '54'}, mean=48. Final answer=32; selected=['Tray', 'Spoon', 'Frame']; missing/wrong gold bindings=['Clock']. Alpha category count=10, complete unique coverage=True.
5. `MAIN_ORDER_0025_view1/b0/C_ENUM/d1` — A=0, S=FAIL, B初判=0, rank_order_known=True. Sort Alpha smallest with original plot position as tie-break; required ['Table', 'Watch', 'Desk', 'Shelf'], Beta {'Table': '30', 'Watch': '66', 'Desk': '84', 'Shelf': '48'}, mean=57. Final answer=54; selected=['Shelf', 'Table', 'Watch', 'Clock']; missing/wrong gold bindings=['Desk']. Alpha category count=10, complete unique coverage=True.
6. `MAIN_ORDER_0016_view1/b0/C_ENUM/d0` — A=0, S=FAIL, B初判=0, rank_order_known=True. Sort Alpha largest with original plot position as tie-break; required ['Tray', 'Clock', 'Spoon'], Beta {'Tray': '30', 'Clock': '60', 'Spoon': '54'}, mean=48. Final answer=32; selected=['Tray', 'Spoon', 'Frame']; missing/wrong gold bindings=['Clock']. Alpha category count=10, complete unique coverage=True.
7. `MAIN_ORDER_0042_view1/b0/C_REF/d1` — A=1, S=PASS, B初判=1, rank_order_known=True. Sort Alpha largest with original plot position as tie-break; required ['Chair', 'Watch', 'Table', 'Tray'], Beta {'Chair': '24', 'Watch': '60', 'Table': '72', 'Tray': '18'}, sum=174. Final answer=174; selected=['Tray', 'Chair', 'Watch', 'Table']; missing/wrong gold bindings=[]. Alpha category count=10, complete unique coverage=True.
8. `MAIN_ORDER_0030_view0/b0/C_REF/d1` — A=1, S=PASS, B初判=1, rank_order_known=True. Sort Alpha largest with original plot position as tie-break; required ['Vase', 'Shelf', 'Frame', 'Spoon'], Beta {'Vase': '60', 'Shelf': '66', 'Frame': '24', 'Spoon': '72'}, sum=222. Final answer=222; selected=['Vase', 'Shelf', 'Frame', 'Spoon']; missing/wrong gold bindings=[]. Alpha category count=10, complete unique coverage=True.
9. `MAIN_ORDER_0018_view0/b0/C_PRINT/d1` — A=1, S=PASS, B初判=1, rank_order_known=True. Sort Alpha smallest with original plot position as tie-break; required ['Mug', 'Shelf', 'Plate'], Beta {'Mug': '48', 'Shelf': '42', 'Plate': '18'}, mean=36. Final answer='36'; selected=['Shelf', 'Plate', 'Mug']; missing/wrong gold bindings=[]. Alpha category count=10, complete unique coverage=True.
10. `MAIN_ORDER_0027_view0/b0/C_PRINT/d0` — A=1, S=PASS, B初判=1, rank_order_known=True. Sort Alpha smallest with original plot position as tie-break; required ['Book', 'Tray', 'Shelf', 'Frame'], Beta {'Book': '54', 'Tray': '60', 'Shelf': '84', 'Frame': '72'}, sum=270. Final answer=270; selected=['Book', 'Frame', 'Shelf', 'Tray']; missing/wrong gold bindings=[]. Alpha category count=10, complete unique coverage=True.
11. `MAIN_ORDER_0022_view0/b0/C_ALT/d1` — A=1, S=PASS, B初判=1, rank_order_known=True. Sort Alpha smallest with original plot position as tie-break; required ['Bowl', 'Frame', 'Vase'], Beta {'Bowl': '12', 'Frame': '72', 'Vase': '30'}, sum=114. Final answer=114; selected=['Frame', 'Vase', 'Bowl']; missing/wrong gold bindings=[]. Alpha category count=10, complete unique coverage=True.
12. `MAIN_ORDER_0023_view1/b0/C_ALT/d0` — A=1, S=PASS, B初判=1, rank_order_known=True. Sort Alpha smallest with original plot position as tie-break; required ['Table', 'Frame', 'Spoon'], Beta {'Table': '48', 'Frame': '12', 'Spoon': '30'}, sum=90. Final answer=90; selected=['Frame', 'Table', 'Spoon']; missing/wrong gold bindings=[]. Alpha category count=10, complete unique coverage=True.
13. `MAIN_ORDER_0012_view0/b0/P_STABLE/d0` — A=1, S=PASS, B初判=1, rank_order_known=True. Sort Alpha largest with original plot position as tie-break; required ['Vase', 'Plate', 'Frame'], Beta {'Vase': '54', 'Plate': '84', 'Frame': '48'}, sum=186. Final answer=186; selected=['Plate', 'Vase', 'Frame']; missing/wrong gold bindings=[]. Alpha category count=10, complete unique coverage=True.
14. `MAIN_ORDER_0028_view0/b0/P_STABLE/d0` — A=1, S=PASS, B初判=1, rank_order_known=True. Sort Alpha largest with original plot position as tie-break; required ['Shelf', 'Desk', 'Spoon', 'Bowl'], Beta {'Shelf': '54', 'Desk': '18', 'Spoon': '78', 'Bowl': '66'}, sum=216. Final answer=216; selected=['Bowl', 'Desk', 'Shelf', 'Spoon']; missing/wrong gold bindings=[]. Alpha category count=10, complete unique coverage=True. Work mentions 174 before self-correction; final answer 216 alone determines A.
15. `MAIN_ORDER_0013_view0/b0/P_REVTIE/d0` — A=0, S=FAIL, B初判=0, rank_order_known=True. Sort Alpha largest with original plot position as tie-break; required ['Bowl', 'Table', 'Tray'], Beta {'Bowl': '84', 'Table': '36', 'Tray': '24'}, mean=48. Final answer=34; selected=['Tray', 'Table', 'Watch']; missing/wrong gold bindings=['Bowl']. Alpha category count=10, complete unique coverage=True.
16. `MAIN_ORDER_0041_view0/b0/P_REVTIE/d0` — A=0, S=UNKNOWN, B初判=0, rank_order_known=False. Sort Alpha smallest with original plot position as tie-break; required ['Bowl', 'Mug', 'Table', 'Frame'], Beta {'Bowl': '30', 'Mug': '78', 'Table': '84', 'Frame': '36'}, mean=57. Final answer=None; selected=None; missing/wrong gold bindings=['Bowl', 'Mug', 'Table', 'Frame']. Alpha category count=unparsed, complete unique coverage=False. Whole raw JSON invalid (missing closing brace); no repair/field salvage.
17. `MAIN_ORDER_0023_view1/b0/P_PLACEBO/d0` — A=0, S=UNKNOWN, B初判=0, rank_order_known=False. Sort Alpha smallest with original plot position as tie-break; required ['Table', 'Frame', 'Spoon'], Beta {'Table': '48', 'Frame': '12', 'Spoon': '30'}, sum=90. Final answer=None; selected=None; missing/wrong gold bindings=['Table', 'Frame', 'Spoon']. Alpha category count=unparsed, complete unique coverage=False. Whole raw JSON invalid (missing closing brace); no repair/field salvage.
18. `MAIN_ORDER_0015_view0/b0/P_PLACEBO/d0` — A=0, S=UNKNOWN, B初判=0, rank_order_known=True. Sort Alpha largest with original plot position as tie-break; required ['Bowl', 'Plate', 'Stool'], Beta {'Bowl': '60', 'Plate': '42', 'Stool': '72'}, mean=58. Final answer=42; selected=None; missing/wrong gold bindings=['Bowl']. Alpha category count=10, complete unique coverage=True. Missing selected remains UNKNOWN despite work text.
19. `MAIN_ORDER_0047_view1/b1/C0/d1` — A=1, S=PASS, B初判=1, rank_order_known=True. Sort Alpha largest with original plot position as tie-break; required ['Shelf', 'Spoon', 'Plate', 'Lamp'], Beta {'Shelf': '42', 'Spoon': '66', 'Plate': '54', 'Lamp': '36'}, sum=198. Final answer=198; selected=['Plate', 'Lamp', 'Shelf', 'Spoon']; missing/wrong gold bindings=[]. Alpha category count=10, complete unique coverage=True.
20. `MAIN_ORDER_0005_view1/b1/C0/d1` — A=0, S=FAIL, B初判=0, rank_order_known=True. Sort Alpha smallest with original plot position as tie-break; required ['Watch', 'Stool', 'Spoon'], Beta {'Watch': '18', 'Stool': '66', 'Spoon': '30'}, mean=38. Final answer=20; selected=['Watch', 'Spoon', 'Chair']; missing/wrong gold bindings=['Stool']. Alpha category count=10, complete unique coverage=True.
21. `MAIN_ORDER_0021_view1/b1/C_NOLIST/d0` — A=1, S=PASS, B初判=1, rank_order_known=False. Sort Alpha largest with original plot position as tie-break; required ['Tray', 'Book', 'Plate'], Beta {'Tray': '60', 'Book': '12', 'Plate': '48'}, sum=120. Final answer=120; selected=['Tray', 'Plate', 'Book']; missing/wrong gold bindings=[]. Alpha category count=16, complete unique coverage=False.
22. `MAIN_ORDER_0044_view1/b1/C_NOLIST/d1` — A=1, S=PASS, B初判=1, rank_order_known=False. Sort Alpha smallest with original plot position as tie-break; required ['Watch', 'Tray', 'Table', 'Spoon'], Beta {'Watch': '12', 'Tray': '60', 'Table': '72', 'Spoon': '30'}, mean=87/2. Final answer=43.5; selected=['Table', 'Spoon', 'Watch', 'Tray']; missing/wrong gold bindings=[]. Alpha category count=4, complete unique coverage=False.
23. `MAIN_ORDER_0044_view0/b1/C_ENUM/d0` — A=1, S=PASS, B初判=1, rank_order_known=False. Sort Alpha smallest with original plot position as tie-break; required ['Clock', 'Frame', 'Table', 'Spoon'], Beta {'Clock': '12', 'Frame': '60', 'Table': '72', 'Spoon': '30'}, mean=87/2. Final answer='43.5'; selected=['Table', 'Spoon', 'Clock', 'Frame']; missing/wrong gold bindings=[]. Alpha category count=4, complete unique coverage=False.
24. `MAIN_ORDER_0000_view0/b1/C_ENUM/d0` — A=0, S=FAIL, B初判=0, rank_order_known=True. Sort Alpha smallest with original plot position as tie-break; required ['Lamp', 'Book', 'Vase'], Beta {'Lamp': '42', 'Book': '54', 'Vase': '12'}, sum=108. Final answer=72; selected=['Lamp', 'Vase', 'Mug']; missing/wrong gold bindings=['Book']. Alpha category count=10, complete unique coverage=True.
