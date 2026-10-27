# SR-F6 面向 Codex 的实验实施包

**2026-10-27｜v1.0｜科学设计＋已实际运行的CPU参考；服务器训练未运行。**

入口：`CODEX_START_HERE.md`。完整中文规范在`docs/00`至`docs/11`。配置在`configs/experiment.json`。

核心：共同任务与模型管线下，匹配合法题组P0/区分覆盖题组P1 × 原J/固定oracle语义信用；以S最终集合和B任务相关取值绑定为反馈，执行与跨题关系只作诊断。两个必要对照是P1上的简单全组信用和同J/known模式置换。

## 目录
- `docs/`：12章决定、数据、标签、训练、门、统计、实施、文献与边界。
- `src/srf6_ref/`：精确oracle、题组、评分、信用、损失、统计、门与服务器接口参考。
- `scripts/`：实际可运行CPU检查、数据审计、预算、普查汇总、绘图、启动防护。
- `tests/`：单元测试；数量与结果看`qa/unit_tests.log`。
- `examples/`：仅开发题的私有oracle与可见prompt/图像；绝不可把PRIVATE文件给模型。
- `qa/`：实际本地日志；没有真实模型回答或服务器PASS。
- `sources/`：Claude原文/数学脚本、F5只读来源与相关工作短阅读记录。
- `templates/`：NOT_RUN回执、冻结合同和报告模板。

## 本地运行
```bash
python -m pip install -r requirements.txt
PYTHONPATH=src python -m unittest discover -s tests -v
python scripts/audit_data.py
python scripts/render_examples.py
python scripts/budget.py
python scripts/math_checks.py
python scripts/launch_guard.py --receipts templates/gates --config configs/experiment.json
```
最后一项应拒绝启动并exit 2。CPU依赖仅供本地参考；生产环境不得自动pip升级。字体只从系统读取，本包不含任何字体文件。

服务器适配与真实回执是必须工作，不是假定已完成的接口。未执行的GPU训练、processor审核、旧J完整生产回归、正式模型效果全部标NOT_RUN。
