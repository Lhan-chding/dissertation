# 4–5张PRO6000执行手册

## 1. 资源策略

一个进程、一张已分配GPU、一个常驻checkpoint。最多5个进程；4卡时减少并发，不改变采样量。不要默认多卡模型并行或改变精度。本轮只做4整数短输出，优先利用跨checkpoint/题目并行。

读取`nvidia-smi`中的型号、显存、可见卡；Slurm环境只使用已分配的CUDA_VISIBLE_DEVICES。不要扫描后占用其他人的空闲卡。检查Python、torch、transformers、peft与旧实测环境是否一致。模型与tokenizer用原缓存快照，不自动下载latest。

## 2. 检查点加载

`manifests/checkpoints.json`来自旧导出遗漏列表；路径是导出时的记录，不是当前存在证明。逐个查COMMIT指向与文件可读性/大小。软链接或挂载根变化可以显式映射，但必须记录路径映射。

一次加载时核对基座revision、LoRA目标模块、rank/alpha、参数名和形状、buffer/位置配置。严格加载，禁用dropout，eval+no_grad。不要把仅adapter文件误当完整源状态，或在加载后重置LoRA。

推理不更新Adam，也不需要占用其GPU显存。原COMMIT/step/reward/history仍保存为身份。源61001 t96、R4 H32、DIRECT H32是三种不同策略。

## 3. raw-only适配

可复用原`HuggingFaceAdapter.prepare/generate`以及`validate_action`；不能复用提前按正序分类的`GenerationBackend.generate`结果作为主标签。

最小接口：

```python
raw = backend.generate_public(messages, seed=seed, max_new_tokens=64)
# raw contains text, token_ids, exact stop_reason, generated length and timing
labels = scorer(raw, frozen_case, audit_record)  # inverse permutation first
```

模型接收的对象只有system/user（符号任务无image）；truth/j/DPE/PTLC预测不进入任何LLM消息。`audit_record`只由本地评分器使用。

只保留生成自带的chosen-token logprobs即可，不重评分所有输出、不导出vocab logits。若实现流式减少输出logit缓存，先与原调用进行技术等价检查，不把它作为采样算法变化。

## 4. 初始smoke

每个worker用2道copy-order、2道duplicate任务，在正/反两个条件各生成2次，共16条。role为SMOKE，永不混入主数据。

smoke检查：正确checkpoint、64token与EOS、原始文本可保留、反序还原、任务复用、重启一次不会重复计数。smoke的数学正确率不是全实验门禁；若新顺序不遵守，原样记录，依主文档做解释。

如果硬件OOM，降低每个worker的并发batch，不改最大输出长度和模型精度。如果加载单个9B也无法完成，则该worker不可用，使用其余已分配卡。不要为了填满卡数改用另一模型。

## 5. 调度

建议首轮：GPU0 S96、GPU1 S32、GPU2 R4_128、GPU3 DIRECT_128、GPU4 REP96或公共准备。随后跑REP32及剩余U/因子任务。资源只有4卡时，REP排队。

一台worker一次处理一个case的8-draw提交块；多个worker分片同一checkpoint时使用任务lease或预分片，不能同时写同一个case。逻辑job清单1944个单元，不需要1944个模型加载。

所有核心对照应覆盖完整面板；不要优先只跑历史负例或只跑会支持理论的B1翻正题。

## 6. 恢复

`sample_key=(experiment,checkpoint,canonical_case_owner,role,draw_index)`，seed由稳定摘要生成。GPU编号、执行时间和重启次数不参与seed。

每8条原子写入后提交；写失败的片段与成功片段分开。返回错误答案或非法文本不是技术失败。只能重试OOM/I/O/进程崩溃等无有效观测的请求，最多2次，保留相同seed与失败原因。

已成功但重复计算的同key：若内容相同只计一次；不同则记录冲突停止该单元，不选择更正确的一个。

## 7. 时间与容量记录

首个完整8题块后记录每输出耗时、P50/P90 token长度、峰值显存；据逻辑job未完成量估算剩余时间，报告GPU小时与墙钟区别。

每个阶段开始查输出空间，写失败或低容量报警再查。不逐样本扫描目录或对100MB状态文件重算hash。正式结果保存全部短输出与token，checkpoint不复制到每个任务目录。

## 8. 最终状态

新生成的角色仅为frozen_probe/smoke，不存在train。程序退出时检查backward_calls=0、optimizer_updates=0。缺失checkpoint、未完成单元、技术失败单元都进入最终报告，不能把它们当0准确率填表。

完成后打包短输出、语义行、case与协议、预测注册表、统计结果、运行脚本和报告；模型大张量继续留服务器，附路径与已提交身份即可。
