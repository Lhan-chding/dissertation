# Codex1 接收说明：MM-DEV F2 / Qwen3.5-9B

**先读主计划：`CODEX1_MM_DEV_FINAL_PLAN_zh.md`。** 本包不是已经运行的实验结果，也不包含已完成的GPU训练器；它是完整的科学执行合同、CPU参考代码和固定数据/训练清单。

## 执行目标

新GRPO实现验收通过后，直接完成4条32步准备、5个起点的测量、30条32步响应、35个模型的固定终点评价与开发分析。不要只返回另一份审计。使用固定Qwen3.5-9B，不设置累计GPU小时或研究墙钟停止阈值。不得恢复SER-J23。

模型/processor/软件环境沿用原审计已记录身份，重新在服务器核验。旧工程adapter均不作为公共起点。最多5张已授权Pro6000并发；单作业租约到期按相同科学身份续跑。

## 文件关系

| 文件 | 用途 |
|---|---|
| `CODEX1_MM_DEV_FINAL_PLAN_zh.md` | 研究目标、证据依据、全部新参数、实施顺序、验收、统计、恢复和停止规则 |
| `config/MM_DEV_F2.json` | 同一合同的机器字段；GPU时间阈值为null |
| `config/run_matrix.json` / `seed_registry.json` | 4准备＋30响应的身份、依赖、配对与seed |
| `config/numeric_sources.jsonl` | 原数值构造函数产生的584个新数值源；尚未渲染图像 |
| `config/schedules/` | 9份固定输入流：1工程、2准备、6续训 |
| `EVIDENCE_REVIEW_zh.md` / `evidence/` | 原始结果与本轮设计的边界、关键原文副本与来源hash |
| `reference/` | CPU参考合同、清单构造及原回答复算，不是生产训练代码 |
| `tests/test_plan.py` | 66项CPU测试；不表示GPU工程已通过 |
| `validation/` | 本地计数、测试、生成与完整性核验记录 |
| `BUNDLE_MANIFEST.json` | 本包全部其他文件的SHA和长度 |

主计划与JSON不得产生不同科学配方。技术实现存在冲突时报告具体字段，不能自行选一个更容易运行的值。最后的用户明确指示高于本包。

## 本地CPU核验

在本包根目录执行：

```bash
python reference/verify_bundle.py .
python -m unittest discover -s tests -v
```

测试使用Python标准库及CPU版PyTorch，不加载模型、不访问网络、不提交作业。

独立核对原始证据（`/path/to/extracted_review`必须是用户原ZIP的解压目录）：

```bash
python reference/review_evidence.py /path/to/extracted_review \
  --out /tmp/mm_dev_audit_recount.json
```

验证数值源和训练流可重复生成（使用新临时目录，避免修改签收清单）：

```bash
python reference/build_manifests.py /path/to/extracted_review /tmp/mm_dev_manifest_rebuild
```

构造器首先检查原`src/mm_core/generator.py` SHA，不接受随意复制的同名生成器。重建输出必须与已签收清单逐字节一致。不要以重建过程为机会换seed或筛题。

`reference/write_config.py`用于从同一证据生成当前JSON和来源副本，参数也是上述原证据目录；执行不应改变任何已有文件字节。它不是训练入口。

## Codex1 必须新增的实际执行层

在原工程中新建`mm_dev`代码和独立run根，具体CLI目标在主计划第15节。必须实现：图像/输入落地、reward一般化、实际采样old logprob、冻结reference、192序列等权累积、完整恢复和状态机、评价与独立复算。禁止升级环境、暗改LoRA范围或以旧零梯度ENGINE代替新实现验证。

有自然非恒定奖励却没有正确policy梯度是技术阻塞；隔离压力测试不能绕过此错误。压力测试只在固定自然工程轨迹全部奖励恒定时按合同执行，不能冒称自然非零GRPO。

## 本轮的最终交付

34条科学训练路径完整，所有固定probe/eval槽位可追溯；报告起点是否真的不同、动作响应、绝对损害与相对均衡差异、奖励对比/归一化奖励的支持、几何缺失、未来重复差异和全部实际成本。

本轮是开发响应实验。不要从5个开发起点宣布BGeo超越强基线、控制器安全或新起点泛化；不要自动启动MM-LOCK、MM-CAL、在线控制、新奖励搜索或额外符号实验。
