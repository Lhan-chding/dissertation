# SR-F6 来源绑定与普查生产适配

`runtime.py` 提供新的 `structured_v4` 接口，原 F5 与更早模块未修改。实际采样方法复制被冻结的 `SRF3CERuntime.generate_group` 方法；AST 回归保证所有算法语句与原方法一致，只有函数所调用的 **F6 本地 generation recipe** 将共同输出上限从 1024 改为 1536。不修改旧模块的全局变量，也不在参数被原入口拒绝后静默退回 1024。

整序列 teacher forcing 同样有独立的 1536 上限适配，使用保存的真实 token IDs，不重编码文本。原独立 draw RNG、JSON prefill、停止规则、原始 sampler logits、FP32 输入路径及 reference forward 保留。自然接口和外部评测尚未由此普查适配器提供；不能把此接口称为完整 54 路训练及评测系统。

`source_binding.bind` 逐文件绑定 F5 冻结部署、当前部署继承模块、模型 snapshot（包括权重）、三个父 adapter、tokenizer/processor 配置和冻结 LEGACY/ChartQA 题目及全部图像。当前部署的继承模块必须与冻结 F5 文件字节一致。HF snapshot 只允许解析至本项目内的 blob；读取其它路径会拒绝。配置字节 hash 与规范化对象 hash 分别记录。来源绑定不加载模型，返回 `PARTIAL`；真实加载后的可训练参数与实际处理器审查未完成前，不签署 SOURCE_BINDING 的 PASS。

`processor_preflight` 只调用真实本地模型目录的 tokenizer/processor，不生成回答、不更新参数。它保存登记的 48 张开发图经过实际 processor patch tensor 逆变换后的图像、尺寸、图像/token 路由、词表与模板身份，以及真实 tokenizer JSON 往返测试。程序回执明确为 `PROGRAM_COMPLETE_VISUAL_REVIEW_PENDING`，视觉审核保持 `NOT_RUN`；程序几何不替代人工或视觉模型审核。

普查冻结要求 DEV_CAL 全部 384 个题目/因子输入，每父 3,072 条、每父×因子 1,536 条，三个父合计 9,216 条。每组严格为同 qid、同父、同冻结 policy 的 G8。逐组缓存仅用于恢复相同完整组；已有部分原始输出会阻断盲目重采。三种信用反事实均保存，标签和信用可由原文重算。完整汇总拒绝缺组、重复组、越界题目、变更的数据/配置身份及被修改的标签。

普查 seeds 由 master、parent、root、variant、draw 派生，不含 factor；因此相同父模型下相同两因子输入可以得到相同轨迹。这些重复行属于配对设计，不能算独立抽样。当前实现不主动合并相同输入调用，故报告实际采样调用与数据行数，并保持最多 9,216 回答上限。

生产命令：

```text
python -m sr_f6.source_binding --config RUN/CONFIG.json --output RUN/SOURCE_BINDING.json
python -m sr_f6.runtime processor-preflight --root RUN
python -m sr_f6.census run --root RUN --parent b0 --shard 0 --shards 2
python -m sr_f6.census summarize --root RUN
```

普查入口先调用生产 `require_census_ready`，不能靠修改模板 status 跨过来源、数据、原生 J、真实处理器与可见图审核。提交脚本统一 `LH-SRF6-` 前缀；普查作业使用两张 GPU 执行两个独立单 GPU shard。没有自动提交、自动升级依赖、测试访问或优化器更新入口。

本地 runtime/census 测试仅证明上述代码边界，未证明真实 GPU 数值一致性或模型效果。实际服务器执行记录须单列，不能用本文件代替。

2026-10-27 实施补充：首次来源绑定完整计算内容 hash；随后模型/处理器加载先核验真实路径和文件元数据，未变化直接复用首次 hash。仅元数据变化的文件重新读内容；内容不符立即阻断。`st_dev` 不作为缓存键，避免共享存储跨节点挂载编号变化引起整套模型重复哈希；缓存元数据不进入科学内容身份。加载前后均做轻量检查，保留实际可训练参数与父 adapter 的验证。

父普查固定 policy 复用加载时已经计算的真实 tensor hash；训练仅在已提交 step 变化时刷新一次，组内及同一步多题均不重复拷贝 GPU 参数。普查标注另外强制核验 `parent_content_hash` 与冻结父文件内容身份一致、同 G8 的实际 `policy_tensor_hash` 唯一且完整；不能只凭父名称重新贴标签。首组概率回放的前后参数 hash 属于只读验证证据，继续保留。

## 来源绑定事故 229909 与修复

首次服务器准备作业 `229909` 在绑定保留面板图片时失败，没有模型采样或参数更新。根因是本轮实现把 F5 复制的 `image_file` 当成相对 `source_root`，但原 F4 的两种图片清单都相对 `source_root/DATA`：`sr_f4.prepare` 调用 `build_dataset(..., root / "DATA")`，ChartQA 同样写入 `root / "DATA" / relative`。修复使用冻结 `legacy_source` 所在的 DATA 目录，并核对它恰好等于 `source_root/DATA`；不搜索替代图片、不改题目 ID 或图像内容。

实际只读核验：原 DATA 目录的 LEGACY 128 张图片共 3,888,361 字节，ChartQA 250 张共 10,982,037 字节，全部存在且与原清单 SHA256 一致，缺失和不一致均为零。

新增 `bind(..., cache_path=run / "SOURCE_BINDING_CACHE.json")`。每个文件完成内容绑定后立即原子持久保存，后面的单个对象失败也不丢失已校验对象；重启仍先核验真实路径和 stat，未变化复用原 SHA256，变化对象单独重哈希且拒绝内容漂移。缓存只写新 SR-F6 运行根，原 F4/F5/模型对象仍只读；缓存本身不是阶段 PASS。

作业 229909 的旧实现只保留了内存中的 hash，没有生成这份持久缓存。因此此次修复后的首次重新绑定仍需重新读模型一次，不能声称恢复了不存在的校验记录；之后如果再次遇到来源缺失，就能复用已落盘对象。原失败日志和准备开始回执由主准备入口保留。
