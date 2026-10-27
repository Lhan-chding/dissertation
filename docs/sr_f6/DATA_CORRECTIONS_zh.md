# SR-F6 数据与随机流：执行前纠错记录

2026-10-27。原始实施包保留在 `docs/sr_f6/package/`，原参考模块未改。
本文件记录生产实现对科学合同的落实与实际 CPU 验证，不代表模型实验完成。

## 已修正的问题

1. **COMPOSITION 接受条件意外依赖未使用的 P1。** 原参考
   `design.make_root(..., zero_alias=True)` 先生成 P0/P1，再要求两者所有视图都没有
   活跃候选答案碰撞。正式主测试只使用普通 P0 四视图，因此这个额外拒绝条件
   会让未进入测试的优化题组改变主测试分布。生产 `make_composition_root`
   完全不调用 P1 优化，复刻原 P0 桶选择、桶内随机排列和抽样随机流，仅核验
   实际四视图。种子、最大 256 次尝试、保留组合、有限小数、非空集合及活跃候选
   零碰撞条件不变。192 个 roots 中 7 个不再被未使用 P1 推迟接受；全部发生在
   任何模型输出之前，既不按模型效果筛选，也不换 seed。
2. **父模型间训练顺序随机流未独立。** 参考调度只由 repeat 派生顺序，三个父
   模型共享同一随机顺序，但主统计方差公式未计入由此可能产生的父间协方差。
   生产调度在 `task_order` 随机流中登记 parent 和 split。同一父 × repeat 六臂
   仍共享顺序；每 12 更新包含各家族 8 个 roots；每条正式路径一遍曝光。
   单题 draw seed 仍逐字继承原参考函数，不含 arm/factor。先导明确使用
   `DEV_PILOT` 根，正式训练使用 `TRAIN` 根。
3. **正式执行顺序。** 原 `paths()` 仅按固定臂序枚举。生产
   `formal_paths()` 用 `seed_for(master, "arm_order", parent, repeat)`
   固定块内六臂随机顺序，包含两个控制臂，不以运行结果改变顺序。
4. **图像尺寸不能照搬历史 renderer。** 旧 F3/F5 renderer 是 1024×768；
   F6 合同要求 1280×896。独立 F6 renderer 使用固定画布、0–100 纵轴、
   固定 37 像素柱宽、固定字体尺寸和图例位置。绘图区为
   `(110,125,1225,755)`，每 6 单位对应 37.8 像素。字体从系统读取并记录
   字体内容 hash，不分发字体、不修改历史 renderer，也不插入逐柱数值标签。

7 个变化 roots：`BINDING_0007`、`TIES_0000`、`INTERVAL_0013`、
`INTERVAL_0020`、`K_DIRECTION_0016`、`K_DIRECTION_0018`、
`K_DIRECTION_0027`（完整前缀均为 `TEST_COMPOSITION_`）。前后题目 hash 和
接受 attempt 见实际生成的 `DATA/COMPOSITION_CORRECTION.json`。

## 已实现和验证的范围

`src/sr_f6/data.py` 独立生成并审核 4,704 条任务记录：DEV_CAL 384、
DEV_PILOT 384、DEV_EVAL 96、TRAIN 2,304、ID 384、COMPOSITION 768、
MECHANISM 192、CLASSOUT 192。DEV_CAL/DEV_PILOT/TRAIN 保留 P0/P1；
DEV_EVAL 与每个正式面板只发布一个固定 P0 四视图集合。训练每个数据因子
仍为 1,152 题。自然接口从 ID 配对抽取 128 题，使总输入路由为 4,832。

自然接口先取每家族按 qid 排序的前 16 题，再按
`seed_for(master, "natural_interface", qid)` 与 qid 排序取剩余 32 题；
此规则在模型访问之前确定。它只保留题意，不保留 JSON/bindings 输出要求，
然后追加合同规定的唯一末行 `Answer:` 约定。

全库审核核对两个参考 oracle、两个历史 native 答案实现及 native 选择集合；
重算编辑池、真实改动和成本描述、活跃候选与 coverage；核对存储金标和答案
变化量；检查全局保留组合隔离及跨 split 忽略名称/顺序的世界指纹。CLASSOUT
严格为不同类别的两个格子各 ±12。可见 prompt 由类别名与查询白名单生成；
更改 gold、factor、root、碰撞标记和不可见数值不会改变 prompt。

本地实测：4,704 条任务全部通过 oracle 与隔离检查；训练 288 roots 的
P1−P0 平均有限候选 coverage 为 **0.13145805776014108**，严格增加比例
**0.5729166666666666**；登记强造碰撞 anchor 为 96。两个操纵门均通过。
这些是生成器数学属性，不是模型学习收益。

实际本地构建目录为 `.local/sr_f6_cpu_preparation/`；生成 3,887 张唯一原图，
PNG 总字节 127,357,409。`DATA/MANIFEST.json` 含 3,924 个内容条目，
完整重新读取验证后大小/hash 不匹配为 0。构建保留 48 张开发 anchor 的
processor 审核计划（每族 8 张），当前所有视觉字段仍为 `NOT_RUN`。
本地生成图片属于预备产物；服务器冻结前必须明确采用这些精确图片字节，
或在服务器一次生成并冻结新的内容身份。不可在实验中无记录更换系统字体
或重渲染输入。

针对性测试命令：

```bash
PYTHONPATH=src python -m pytest -q tests/sr_f6/test_data.py tests/sr_f6/test_planning.py
python -m ruff check src/sr_f6/data.py src/sr_f6/planning.py tests/sr_f6/test_data.py tests/sr_f6/test_planning.py
PYTHONPATH=src python -m sr_f6.data --root NEW_PREPARATION_ROOT --config configs/sr_f6.json
```

首次实际针对性测试为 **11 passed**；补充 CLASSOUT 实际视图互异与真实编辑
描述核验后为 **13 passed**，原已生成 4,704 题重新审核仍通过。Ruff 检查通过。
测试覆盖 P1 调用被禁止时
COMPOSITION 仍可生成、7 根差异复现、真实柱像素/尺寸、隐私字段不进入提示、
父随机流/每窗家族平衡/先导根名前缀/完整六臂顺序。

## 尚未完成的门

`BUILD_RECEIPT.status=CPU_PREPARED_NOT_RELEASED`。历史 F4/F5 世界排除、
真实 LEGACY/ChartQA 文件绑定、真实 processor 后的 48 图视觉审核、真实模型
生成和所有 GPU 运行仍须各自证据。`source_visual_review`、
`processor_readability`、`historical_world_exclusion` 和
`external_retention_ids` 均保持 `NOT_RUN`。程序几何检查不冒称人类或视觉代理
审核；这些本地产物不签发最终 DATA_AND_MEASUREMENT 门。
