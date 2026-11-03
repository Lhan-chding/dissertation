# 一个真实冻结题组的四格演算（不是模型结果）

root：`OT1_swap8_n8_00000_m2r1_ls_2026110301`。以下完全由题目事实和两个独立 oracle 推导。对同一张图只换清单，金答案不变；对调跨边界对象在图上的位置，物体的两个数值随它一起移动。两份清单在两张图之间都按实名固定。

| 视图 | 图 | 清单 | 金集合 | 对应的 Beta | 金答案 | 若误跟随清单会选 |
|---|---|---|---|---|---:|---|
|p0l0|[查看图](../frozen/images/b1e4cb583abdfbf557a59b5b1699d405f57fdc1d006ef51e162e2c71832b4787.png)|Vase, Chair, Clock, Lamp, Watch, Spoon, Book, Tray|Chair, Clock, Spoon|30, 24, 18|72|Clock, Spoon, Chair|
|p0l1|[查看图](../frozen/images/b1e4cb583abdfbf557a59b5b1699d405f57fdc1d006ef51e162e2c71832b4787.png)|Vase, Watch, Clock, Lamp, Chair, Spoon, Book, Tray|Chair, Clock, Spoon|30, 24, 18|72|Clock, Spoon, Watch|
|p1l0|[查看图](../frozen/images/3fa50b74db8841a04fa76a1b809655264cd50d839e6ebc40b56f72179cb54f6d.png)|Vase, Chair, Clock, Lamp, Watch, Spoon, Book, Tray|Watch, Clock, Spoon|60, 24, 18|102|Clock, Spoon, Chair|
|p1l1|[查看图](../frozen/images/3fa50b74db8841a04fa76a1b809655264cd50d839e6ebc40b56f72179cb54f6d.png)|Vase, Watch, Clock, Lamp, Chair, Spoon, Book, Tray|Watch, Clock, Spoon|60, 24, 18|102|Clock, Spoon, Watch|

这四格的图表原始类别和数值：

**p0l0图**（{'aggregate': 'sum', 'direction': 'largest', 'family': 'TOPK', 'k': 3, 'selection_series': 'Alpha', 'value': 'Beta'}）

| 类别（从左到右） | Alpha | Beta |
|---|---:|---:|
|Vase|30|54|
|Chair|42|30|
|Clock|78|24|
|Lamp|18|12|
|Watch|42|60|
|Spoon|60|18|
|Book|18|42|
|Tray|12|36|

**p1l0图**（{'aggregate': 'sum', 'direction': 'largest', 'family': 'TOPK', 'k': 3, 'selection_series': 'Alpha', 'value': 'Beta'}）

| 类别（从左到右） | Alpha | Beta |
|---|---:|---:|
|Vase|30|54|
|Watch|42|60|
|Clock|78|24|
|Lamp|18|12|
|Chair|42|30|
|Spoon|60|18|
|Book|18|42|
|Tray|12|36|

核对模型是否做到正确的“该变/不该变”须用相同 draw 编号的各格分别评分；不能每格挑一个最好回答。示例未向模型提供图表事实表或正确选择。
