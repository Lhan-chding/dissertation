# SSVC 建模阶段 Codex 交接包

## 阅读顺序

1. `START_HERE_FOR_CODEX.md`：任务与执行边界。
2. `CODEX_MODELING_EXPERIMENT_PLAN_zh.md`：M0–M6 的完整实验和实现规范。
3. `protocol.json`：机器可读配置及推导预算。
4. `CPU_ACCEPTANCE_CHECKLIST_zh.md`：需要实际验证的事项。
5. `SOURCES_AND_REPO_SNAPSHOT_zh.md`：项目来源、最新代码快照与相关工作。

`reference/` 包含22项已执行的独立代数/契约测试及日志，用于校对数学，不替代完整实验。`source_file_hashes.json` 绑定本会话先前设计文件；这些文件不是本轮执行结果。

主方案要求先做表示和维度资格验证，在固定观察窗口下检查模型适用范围。在线检验器、反馈控制、新 Qwen 训练均不在本轮范围内。

最终应交付一份有证据的 `MODELING_DECISION_zh.md`，说明什么表示、多少维、多少采样与多长局部窗口可用，或说明为什么目前不可用。
