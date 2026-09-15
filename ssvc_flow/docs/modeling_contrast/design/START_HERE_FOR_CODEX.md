# Codex 启动说明：建模第二轮

## 工作目标

按 `CODEX_MODELING_CONTRAST_V2_zh.md` 完成实现、真实CPU验证和建模决定。当前任务是把概率变化的建模基础做扎实；不启动新的Qwen评估、GPU训练、在线检验器或SSVC。

阅读顺序：本文件 → 主实施方案 → `REVISED_IDEA_AND_THEORY_zh.md` → `protocol.json` → `CPU_ACCEPTANCE_CHECKLIST_zh.md` → `SOURCES_AND_NOVELTY_zh.md`。

## 起点

- 仓库：`Lhan-chding/dissertation`。
- 工作目录：`ssvc_flow/`。
- 本次确认的 modeling 父提交：`c71c60b4dad2d4432869eb195bb82069ebde374f`。
- 新分支建议：`codex/ssvc-modeling-contrast-v2-20260915`。
- 先检查最新HEAD与dirty状态，使用独立worktree，不修改历史正在使用的分支，不reset任何现有修改。
- 原始建模附件：`d48b4ebd-5316-46e8-abf3-cc8a0231a764.zip`；独立分析附件：`SSVC_Modeling_Independent_Review_20260915.zip`。路径由本地实际文件解析，不能把缺失服务器文件视为已验证。

## 必须推进到哪里

1. 实现 `src/modeling_contrast/`，保留原建模模块。
2. 数学见证、估计器/协方差单元测试、非零消融和CPU smoke真实运行。
3. 原件齐备时，复用旧轨迹完成N0–N3，不先重训全部历史轨迹。
4. N3科学门禁和资源门禁同时通过，才按固定新seed运行N4的32条微型CPU轨迹；否则输出具体失败决定，不扩预算追结果。
5. 形成 `MODELING_DECISION_V2_zh.md` 和源码/测试/数据回执后停止。

本包独立NumPy参考32项测试已运行通过，仅证明对应公式和契约，未执行仓库新方法效果实验。Codex仍必须完成生产实现与回归，不可复制一个PASS字段作为验收。

## 最重要的实现规则

- 固定四事件语义和题目身份；三坐标是表示自由度，不是预先确定的动态秩。
- 直接拟合候选相对主更新的差异；总体响应相减保留为基线。
- 相同起点计数在同bank差异中代数消去；剩余candidate/packet协方差要如实建模。
- 观测法至少包含独立计数、共同随机数、共同样本似然差及候选混合。
- 使用logp的方案须与同权限直接评分比较。当前16动作toy的已知X/I动作评分很强，禁止排除该基线以制造效率优势。
- alias只代表当前推理相同，不合并不同Adam历史；不把范数相近当作向量相同。
- 所有消融保持非零有效处理；没有合格秩就写NO_ACCEPTABLE_MODEL，不把消融改成r0。
- 旧locked-test已被分析，所有新调参属于开发。新seed结果产生之前冻结选择。
- 观测样本、训练bank、拟合/测试packet、oracle四个层次分开；禁止在测试响应已知后“预测”。
- sample-only和sample+logp访问权限分别评价；打分、标签、校准、候选计算全计费。

## 最终交付

代码commit（或本地patch）、实际测试日志、源文件hash、所有阶段状态、估计器比较表、候选差异预测表、成本收支、独立验证结果，以及下一步最小真实数据需求。

首次需要新的真实模型调用/GPU操作，或CPU预计超出4小时/8GiB RAM/3GiB新增结果预算时停止。不要自动sbatch，不修改外部已启动作业。
