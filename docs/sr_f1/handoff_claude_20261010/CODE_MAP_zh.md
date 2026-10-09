# 代码地图

下面的行号对应包内实际部署源码e61e9de。完整文件随包保留。

| 环节 | 文件与行号 | 代码行为 |
|---|---|---|
| 原生视觉聊天输入 | [runtime.py](code/src/sr_f1/runtime.py)，171—263 | apply_chat_template、assistant后缀、图像tokens、尺寸及路由 |
| 关闭thinking与gold EOS | [vl_runtime.py](code/src/mm_core/vl_runtime.py)，13、568—570 | CHAT_TEMPLATE_KWARGS、encode_completion |
| 模型可见字段白名单 | [data.py](code/src/sr_f1/data.py)，48—59 | 不传隐藏world/qid/path给prompt |
| 固定采样与输出切片 | [runtime.py](code/src/sr_f1/runtime.py)，47—59、328—413 | generation config、prompt_len、raw token/logprob、EOS/截断 |
| 字段覆盖 | [runtime.py](code/src/sr_f1/runtime.py)，689—694 | L_json与L_answer与L_evidence |
| 严格JSON解析和评分 | [semantic_contract.py](code/docs/sr_f1/package/reference/semantic_contract.py)，262—325；[contract.py](code/src/sr_f1/contract.py) | 整段JSON、数值边界、解析健壮性补丁；合法错误不等于格式失败 |
| 一次公共桥接 | [training.py](code/src/sr_f1/training.py)，758—887 | 16×8、gold+EOS、NLL、等权、梯度累积、clip、更新与检查点 |
| LoRA与AdamW | [training.py](code/src/mm_core/training.py)，242—305 | full-attention q_proj/v_proj、r8/alpha16、FP32 adapter、lr1e-5 |
| SR缓存teacher forcing | [runtime.py](code/src/sr_f1/runtime.py)，426—494 | 每token前向的cache计算图、位置/掩码和logprob |
| 自定义线性cache状态 | [runtime.py](code/src/mm_dev/runtime.py)，139—196 | FunctionalLinearLayer、卷积/循环状态函数式替换、临时禁用checkpointing |
| 桥接后发布真实身份 | [runtime.py](code/src/sr_f1/runtime.py)，808—821 | 清除零LoRA标记、发布SRF1_FORMAT_BRIDGED |
| 旧guard修复与新gate | [runtime.py](code/src/sr_f1/runtime.py)，825—963；[format_review.py](code/src/sr_f1/format_review.py) | 原回答复用、审计授权、先确认再同题复测、低于95%阻塞 |
| worker返回值处理 | [run_worker.py](code/scripts/sr_f1/run_worker.py)，155—186 | PROTOCOL_BLOCKED不会成为COMPLETE，不放行下游 |
| 冻结和源码修订身份 | [freeze.py](code/src/sr_f1/freeze.py)；[orchestration.py](code/src/sr_f1/orchestration.py) | 原冻结不覆盖、一次受控恢复、UNKNOWN不重复提交 |
| GPU ENGINE验收实现 | [engine.py](code/src/sr_f1/engine.py) | 当前未执行，无通过回执 |

## 本地测试的真实覆盖范围

- [test_cached_teacher_forcing.py](code/tests/mm_dev/test_cached_teacher_forcing.py)：21—73为小型CPU F2Runtime fixture；114—136检查FP32/BF16采样logprob；157—191比较cached/full-prefix LoRA梯度；194—245检查prompt梯度回传和detach负例。
- [test_training.py](code/tests/sr_f1/test_training.py)：主要使用TinyRuntime；实际9B/CUDA ENGINE尚未执行。
- [test_runtime.py](code/tests/sr_f1/test_runtime.py)、[test_format_review.py](code/tests/sr_f1/test_format_review.py)、[test_technical_repair.py](code/tests/sr_f1/test_technical_repair.py)覆盖记录身份、审计和恢复边界。
- [本地真实检查回执](history/project_docs/validation/REPAIR_LOCAL_CHECKS.json)记录276 tests与3 subtests通过。包构建时另外从打包源码运行同一目标集合，结果见 [包内源码测试](analysis/BUNDLED_SOURCE_TESTS.txt)。

## 实际依赖源码

[evidence/installed_dependency_source/transformers/models/qwen3_5/modeling_qwen3_5.py](evidence/installed_dependency_source/transformers/models/qwen3_5/modeling_qwen3_5.py)及同目录配置/processor源码、Qwen3-VL的相关实现、cache_utils、generation/utils、masking_utils等来自服务器实际安装的Transformers 5.14.1。

[evidence/installed_dependency_source/peft/tuners/lora/layer.py](evidence/installed_dependency_source/peft/tuners/lora/layer.py)及同目录配置/模型、peft_model、保存/加载等来自实际PEFT 0.19.1。相应分发元数据和许可证一并保留。所收录的是部分依赖源码；具体每个文件的来源、字节数和hash见 [服务器取证清单](evidence/CAPTURE_MANIFEST.json)。
