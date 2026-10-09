# SR-F1 问题、数据和代码包

用途：用户将此包交给Claude，要求其根据代码和数据产出解决方案文件。包内新增说明仅记录实际问题、数据、错误信息和代码位置。

取证时间：新加坡时间 **2026-10-10 00:12:02—00:12:06**。该时点服务器仍在完成原题复测。本包是时间点快照，不是完整科学实验结果或自包含重训练包。

## 当前问题

唯一公共桥接16/16步、128条样本已完成。独立FORMAT_CONFIRM完成256/256条，格式覆盖100/256（39.0625%），低于原合同95%门槛。原题复测收录113/256条，最终协议阻塞回执在取证时尚未产生。ENGINE及正式科学训练尚未开始。

## 文件

- [问题、执行过程及错误数据](PROBLEM_DESCRIPTION_zh.md)
- [代码位置与行为](CODE_MAP_zh.md)
- [数据文件及收录数量](EVIDENCE_INDEX_zh.md)
- [原始回答示例](analysis/REPRESENTATIVE_CASES_zh.md)
- [给Claude的请求](CLAUDE_REQUEST_zh.md)
- [原执行合同](code/docs/sr_f1/package/CODEX_NEXT_PLAN_SR_F1_zh.md)
- [原机器配置](code/docs/sr_f1/package/config/SR_F1.json)
- [收录、省略与来源](OMISSIONS_AND_PROVENANCE_zh.md)

`code/`是实际部署源码及相关测试；`evidence/`是服务器记录、模板配置和实际依赖源码；`history/`记录已发生的代码修复。

## 离线校验

```bash
python3 tools/verify_bundle.py .
python3 tools/recompute_format_coverage.py .
```

校验命令仅依赖Python标准库；复算命令需Python 3.10+和NumPy（原参考评分模块的依赖）。两条命令均不联网、不加载模型、不训练。第二条从收录原文重算字段覆盖；after只含113条部分记录。
