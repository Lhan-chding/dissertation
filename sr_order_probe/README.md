# SR-ORDER-PROBE｜Codex实施包

**2026-11-02 / v1.0 / INFERENCE ONLY**

入口：[`CODEX_START_HERE.md`](CODEX_START_HERE.md)。完整方案为根目录 `SR_ORDER_PROBE_完整实施方案_zh.md`。

科学配置已定版；生产运行身份、真实processor和长assistant前缀仍须在服务器验证。本包不是已运行的模型实验，也不是训练授权。

## 目录

| 路径 | 内容 |
|---|---|
| docs/ | 8章实施合同：范围、条件、数据、后端、指标、统计、任务、交付 |
| config/plan.json | 固定模型路径、科学矩阵、采样、种子、统计、权限 |
| data/ | 48主世界＋4烟测世界，每世界两命名；104图、公开请求、私有oracle、manifest |
| src/order_probe/ | 可运行CPU参考：数据、输入、评分、配对统计 |
| scripts/ | 构造/核验、历史排重、评分、分析、技术预检及纯合成QA |
| backend/interface.py | 待绑定的生产后端接口，空实现默认拒绝 |
| tests/ | 52项单元测试 |
| qa/ | 本次真实CPU日志和边界；没有真实模型结果 |
| sources/ | Claude第四轮、独立回复、历史配置/提示/测量/绘图源码 |
| templates/ | 未执行回执模板，不能伪签PASS |

## CPU核验

```bash
python -m unittest discover -s tests -v
python scripts/validate_inputs.py --inputs data --out /tmp/order_probe_validation.json
```

依赖Python标准库、NumPy、Pillow；复用现有环境，不升级生产torch/transformers。字体文件从本地安装读取，包中不提供字体。

主推理4320、独立烟测216、更新0。非法模型输出保留在分母；模型效果不是技术准入门。下一步训练仅能在新合同下讨论。
