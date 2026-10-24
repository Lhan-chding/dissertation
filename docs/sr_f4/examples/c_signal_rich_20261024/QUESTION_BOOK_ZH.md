# C 信号丰富题组：24 道开发原型

状态：`DESIGN_PROTOTYPE_NOT_FROZEN_NOT_MODEL_TESTED`。9 个事实世界、6 个共同构造谱系。

以下表格与答案供设计审核，不是模型输入，也不是自然生成的回答。图片尚未生成；每道题有独立的原生 CONTRACT_V2 prompt。所有同谱系变体必须属于同一数据分区。

公共语义：并列按图中印刷顺序处理，表格从上到下就是印刷顺序。均值按实际入选项数计算，接受精确分数；差值指 Alpha 减 Beta。模型的完整 prompt 另含原生 JSON 证据与答案契约，见 QUESTION_BANK.json。

## BINDING：排序列与取值列：完整二乘二对照

事实表 `C_DESIGN_20261024_BINDING_0`（从上到下为印刷顺序）：

| 类别 | Alpha | Beta |
|---|---:|---:|
| January | 84 | 19 |
| February | 68 | 47 |
| March | 55 | 68 |
| April | 43 | 83 |
| May | 29 | 72 |
| June | 16 | 54 |

**Alpha_to_Alpha**：Select the 3 categories with the largest Alpha values. What is the arithmetic mean of the Alpha values over the selected categories?

正确采用集合：January, February, March；聚合值：84, 68, 55；精确答案：**69**。

**Alpha_to_Beta**：Select the 3 categories with the largest Alpha values. What is the arithmetic mean of the Beta values over the selected categories?

正确采用集合：January, February, March；聚合值：19, 47, 68；精确答案：**134/3**。

**Beta_to_Alpha**：Select the 3 categories with the largest Beta values. What is the arithmetic mean of the Alpha values over the selected categories?

正确采用集合：April, May, March；聚合值：43, 29, 55；精确答案：**127/3**。

**Beta_to_Beta**：Select the 3 categories with the largest Beta values. What is the arithmetic mean of the Beta values over the selected categories?

正确采用集合：April, May, March；聚合值：83, 72, 68；精确答案：**223/3**。

## IMPACT：固定可读边界，改变错选对答案的影响

事实表 `C_DESIGN_20261024_IMPACT_zero`（从上到下为印刷顺序）：

| 类别 | Alpha | Beta |
|---|---:|---:|
| January | 88 | 29 |
| February | 73 | 46 |
| March | 60 | 55 |
| April | 48 | 55 |
| May | 27 | 69 |
| June | 12 | 54 |

**zero**：Select the 3 categories with the largest Alpha values. What is the arithmetic mean of the Beta values over the selected categories?

正确采用集合：January, February, March；聚合值：29, 46, 55；精确答案：**130/3**。

事实表 `C_DESIGN_20261024_IMPACT_small`（从上到下为印刷顺序）：

| 类别 | Alpha | Beta |
|---|---:|---:|
| January | 88 | 29 |
| February | 73 | 46 |
| March | 60 | 42 |
| April | 48 | 41 |
| May | 27 | 69 |
| June | 12 | 54 |

**small**：Select the 3 categories with the largest Alpha values. What is the arithmetic mean of the Beta values over the selected categories?

正确采用集合：January, February, March；聚合值：29, 46, 42；精确答案：**39**。

事实表 `C_DESIGN_20261024_IMPACT_medium`（从上到下为印刷顺序）：

| 类别 | Alpha | Beta |
|---|---:|---:|
| January | 88 | 29 |
| February | 73 | 46 |
| March | 60 | 62 |
| April | 48 | 42 |
| May | 27 | 69 |
| June | 12 | 54 |

**medium**：Select the 3 categories with the largest Alpha values. What is the arithmetic mean of the Beta values over the selected categories?

正确采用集合：January, February, March；聚合值：29, 46, 62；精确答案：**137/3**。

事实表 `C_DESIGN_20261024_IMPACT_large`（从上到下为印刷顺序）：

| 类别 | Alpha | Beta |
|---|---:|---:|
| January | 88 | 29 |
| February | 73 | 46 |
| March | 60 | 82 |
| April | 48 | 17 |
| May | 27 | 69 |
| June | 12 | 54 |

**large**：Select the 3 categories with the largest Alpha values. What is the arithmetic mean of the Beta values over the selected categories?

正确采用集合：January, February, March；聚合值：29, 46, 82；精确答案：**157/3**。

## TIES：多元素并列与印刷顺序

事实表 `C_DESIGN_20261024_TIES_0`（从上到下为印刷顺序）：

| 类别 | Alpha | Beta |
|---|---:|---:|
| May | 80 | 31 |
| June | 58 | 47 |
| April | 58 | 68 |
| January | 58 | 83 |
| February | 22 | 72 |
| March | 11 | 54 |

**top3_mean**：Select the 3 categories with the largest Alpha values. What is the arithmetic mean of the Beta values over the selected categories?

正确采用集合：May, June, April；聚合值：31, 47, 68；精确答案：**146/3**。

**top3_sum**：Select the 3 categories with the largest Alpha values. What is the sum of the Beta values over the selected categories?

正确采用集合：May, June, April；聚合值：31, 47, 68；精确答案：**146**。

**top2_mean**：Select the 2 categories with the largest Alpha values. What is the arithmetic mean of the Beta values over the selected categories?

正确采用集合：May, June；聚合值：31, 47；精确答案：**39**。

**bottom3_mean**：Select the 3 categories with the smallest Alpha values. What is the arithmetic mean of the Beta values over the selected categories?

正确采用集合：March, February, June；聚合值：54, 72, 47；精确答案：**173/3**。

## SIGNED：选择与有符号运算分开控制

事实表 `C_DESIGN_20261024_SIGNED_0`（从上到下为印刷顺序）：

| 类别 | Alpha | Beta |
|---|---:|---:|
| January | 87 | 89 |
| February | 74 | 47 |
| March | 61 | 82 |
| April | 49 | 16 |
| May | 31 | 68 |
| June | 17 | 54 |

**diff_mean**：Select the 3 categories with the largest Alpha values. What is the arithmetic mean of the Alpha minus Beta differences over the selected categories?

正确采用集合：January, February, March；聚合值：-2, 27, -21；精确答案：**4/3**。

**diff_sum**：Select the 3 categories with the largest Alpha values. What is the sum of the Alpha minus Beta differences over the selected categories?

正确采用集合：January, February, March；聚合值：-2, 27, -21；精确答案：**4**。

**beta_mean**：Select the 3 categories with the largest Alpha values. What is the arithmetic mean of the Beta values over the selected categories?

正确采用集合：January, February, March；聚合值：89, 47, 82；精确答案：**218/3**。

**rank_beta_diff_mean**：Select the 3 categories with the largest Beta values. What is the arithmetic mean of the Alpha minus Beta differences over the selected categories?

正确采用集合：January, March, May；聚合值：-2, -21, -37；精确答案：**-20**。

## INTERVAL：区间端点与条件筛选

事实表 `C_DESIGN_20261024_INTERVAL_0`（从上到下为印刷顺序）：

| 类别 | Alpha | Beta |
|---|---:|---:|
| January | 70 | 20 |
| February | 67 | 26 |
| March | 58 | 22 |
| April | 41 | 41 |
| May | 66 | 37 |
| June | 28 | 50 |
| July | 81 | 30 |
| August | 20 | 65 |

**main**：Select the categories from February to July, excluding the start and including the end, retaining only categories where Alpha exceeds Beta. What is the arithmetic mean of the Alpha minus Beta differences over the selected categories?

正确采用集合：March, May, July；聚合值：36, 29, 51；精确答案：**116/3**。

**include_start**：Select the categories from February to July, including the start and including the end, retaining only categories where Alpha exceeds Beta. What is the arithmetic mean of the Alpha minus Beta differences over the selected categories?

正确采用集合：February, March, May, July；聚合值：41, 36, 29, 51；精确答案：**157/4**。

**exclude_end**：Select the categories from February to July, excluding the start and excluding the end, retaining only categories where Alpha exceeds Beta. What is the arithmetic mean of the Alpha minus Beta differences over the selected categories?

正确采用集合：March, May；聚合值：36, 29；精确答案：**65/2**。

**no_filter**：Select the categories from February to July, excluding the start and including the end. What is the arithmetic mean of the Alpha minus Beta differences over the selected categories?

正确采用集合：March, April, May, June, July；聚合值：36, 0, 29, -22, 51；精确答案：**94/5**。

## K_DIRECTION：数量与方向：完整二乘二对照

事实表 `C_DESIGN_20261024_K_DIRECTION_0`（从上到下为印刷顺序）：

| 类别 | Alpha | Beta |
|---|---:|---:|
| January | 42 | 66 |
| February | 86 | 24 |
| March | 57 | 78 |
| April | 21 | 43 |
| May | 74 | 51 |
| June | 35 | 87 |
| July | 63 | 19 |
| August | 12 | 60 |

**largest_2**：Select the 2 categories with the largest Alpha values. What is the arithmetic mean of the Beta values over the selected categories?

正确采用集合：February, May；聚合值：24, 51；精确答案：**75/2**。

**largest_3**：Select the 3 categories with the largest Alpha values. What is the arithmetic mean of the Beta values over the selected categories?

正确采用集合：February, May, July；聚合值：24, 51, 19；精确答案：**94/3**。

**smallest_2**：Select the 2 categories with the smallest Alpha values. What is the arithmetic mean of the Beta values over the selected categories?

正确采用集合：August, April；聚合值：60, 43；精确答案：**103/2**。

**smallest_3**：Select the 3 categories with the smallest Alpha values. What is the arithmetic mean of the Beta values over the selected categories?

正确采用集合：August, April, June；聚合值：60, 43, 87；精确答案：**190/3**。

## 构造的错误规则对照

这些是确定性的错误规则演算，不计为模型回答、训练信号或训练数据。

| 问题 | 错误规则 | 错误集合 | 沿该集合计算 | 与正确答案相同 |
|---|---|---|---:|---|
| C_DESIGN_20261024_BINDING_Alpha_to_Beta | rank_by_wrong_column | March, April, May | 223/3 | False |
| C_DESIGN_20261024_IMPACT_zero | swap_boundary_March_for_April | January, February, April | 130/3 | True |
| C_DESIGN_20261024_IMPACT_small | swap_boundary_March_for_April | January, February, April | 116/3 | False |
| C_DESIGN_20261024_IMPACT_medium | swap_boundary_March_for_April | January, February, April | 39 | False |
| C_DESIGN_20261024_IMPACT_large | swap_boundary_March_for_April | January, February, April | 92/3 | False |
| C_DESIGN_20261024_TIES_top3_mean | calendar_order_breaks_tie | May, January, April | 182/3 | False |
| C_DESIGN_20261024_TIES_top3_mean | include_entire_tie_block | May, June, April, January | 229/4 | False |
| C_DESIGN_20261024_SIGNED_diff_mean | rank_by_wrong_column | January, March, May | -20 | False |
| C_DESIGN_20261024_INTERVAL_main | include_excluded_start | February, March, May, July | 157/4 | False |
| C_DESIGN_20261024_INTERVAL_main | exclude_included_end | March, May | 65/2 | False |
| C_DESIGN_20261024_INTERVAL_main | omit_positive_filter | March, April, May, June, July | 94/5 | False |

验证：24 道题均与手工答案及两套 oracle 一致；11 条错误规则演算一致。这仅证明题目定义与算术正确，尚未证明模型会自然产生足够的组内分歧。
