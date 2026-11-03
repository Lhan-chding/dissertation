# Codex 执行入口｜SR-ORD-TRAIN-1

**目标**：保持 Qwen3.5-9B 的历史 F6 J-only on-policy 更新，验证基于顺序诊断的 D 是否比同预算通用增强 G 更能迁移。R 是继续训练对照。只执行本包登记的项目。先读 EXPERIMENT_zh.md，再按本页动手。此包由 CPU 准备，**不含远程训练结果或一键集群启动器**。

## 已有、需实现、需实测

| 已可在本机运行 | 集群 Codex 的实施任务 | 必须在真实运行中验证 |
|---|---|---|
| 实际训练题/变体、8/10类图、所有公开提示、私有题卡、逐组计划、四格澄清、评测清单；原生 F6 评分子集及夹具；严格评分、统计 | 建立独立实验登记与只读历史源码绑定；从现有 native runtime 接入处理器、on-policy采样/反传、原子checkpoint与调度；输出所规定的 raw、概率和回执 | 实际张量/适配器/参考模型身份，真实processor图像和长前缀，原生J与源码版本，数值稳定，预算足额及真实终点收益 |

### A. 本地复核（从解压后的包根目录运行）
使用 Python 3.11 或3.12、NumPy和Pillow，见 requirements-cpu.txt；不需 GPU 或联网。运行：
```bash
python -B verify_release.py
python -B -m unittest discover -s tests -v
python -B export_runtime_plans.py --frozen frozen
python -B audit_frozen.py --frozen frozen
python -B runtime_contract.py check-native --src vendor/native_src
python -B tools/cpu_pipeline_check.py --frozen frozen
```
frozen/FREEZE.json 是科学输入清单摘要；frozen/receipt_plans/INDEX.json 记录每条路径计划摘要。不要按模型表现重新出题。build_plan.py 仅供生成到**另一个不存在的目录**并核对；默认执行直接使用已附 frozen/。如运行字体/Pillow不同，PNG的文件字节可以不同，不得用重绘产物覆盖正式已冻结的804张新图。公开图文索引 inputs.jsonl 中仅将图像、prompt和assistant_prefill送给模型；root_id、case_id、view、seed、私有世界、gold和诊断标记都是控制面或评分数据。

### B. 在新目录实现服务器适配，先输出绑定回执
1. 为这次实验创建独立运行目录、代码登记、调度与日志。不要覆盖历史F6/ORDER_PROBE，不调用旧的注册命令启动旧任务。不能通过移除历史访问限制借用旧阶段身份。
2. 通过原历史回执定位源码、真实J6父适配器、snapshot、FP32推理与训练kernels、tokenizer/processor/图像通道。固定新代码提交及依赖；用 runtime_contract.py check-native 的 --historical-kernel 对目标源码作指纹核验。详见 docs/03_RUNTIME.md。不同来源的优化细节有冲突时，先核对实际绑定，不得无声换奖励或优化器。
3. 仅只读已开放 DEV_EVAL/P0 加载历史96个题目的原始PNG和v3实际提示（含原哈希）。包内已从冻结的原构造器复原题卡、题号与文本。运行 tools/bind_retention.py：
```bash
python tools/bind_retention.py --frozen frozen --private /BOUND/F6/DATA/DEV_EVAL/TASKS_PRIVATE.jsonl --public-v3 /BOUND/PILOT_V3/DEV_EVAL/MODEL_INPUTS.jsonl --old-root /BOUND/F6 --out run/bound_assets
```
上例 /BOUND 路径是需由回执解析的占位，禁止猜路径或拿原v1 prompt冒充v3。私有题卡、v3 overlay和源图片的证据写入run/bound_assets/LEGACY_BINDING.json；旧图片从这个独立目录解析，新图从 frozen/images/ 解析。找不到或哈希、题意、提示不合等均为前置技术缺口，记录并修复真实绑定，不重画替代。历史封存测试不参与当前保持和主终点。

4. 将新训练调度接入F6原生J-only内核。旧驱动假设一批中4root×4variant，此处是16个独立root×8 draws。实现针对新计划的验证和事务层，不能把新请求伪装成旧U10。每臂每个step必须依据该臂自己的当前策略生成，128条序列等权、原始token路径与固定父参考；优化器更新前执行数值检查，详见03。
5. 在实际模型读取前生成 RUN_BINDING.json：本包登记摘要、代码版本、源/模型/父张量快照、配置、可复查的采样器/processor输入、资源/恢复策略。新数据与曾公开的F6 DEV及ORDER-PROBE、旧C、其他能合法访问的历史诊断库按身份与事实表核对；保留旧保持集的预定复用和同一root内部变体。包内已查323个可用历史数值指纹；远端尚有未随包附带的历史公开数据需补核。绝不为查重读取封存的历史结果。没有条件取得的材料明确列作未核。

### C. 烟测与规则澄清
192条开发烟测全部保存，24个G8，零参数更新。先核对输入、停止/结束、真实概率、评分、回执和图像处理（至少一张代表新8/10类别图做processor前后对比，不多调用模型）；前缀使用旧已验证的same-assistant拼接与CPU序列化夹具，不另开几百次试跑。若做梯度核验，在隔离副本上反传并丢弃，不执行Adam、不携带状态入正式路径。运行回执与 frozen/receipt_plans/smoke_slots.json 对齐。另按 clarification_manifest 采集384条真实回答，输出全部四格；既不看正式评测也不据澄清结果改方案。语义错误、模型输出非法JSON、全J0/1、混合组比例低均是有效观测；硬件/身份/计数/输入损坏按技术失败处理。

### D. 训练与评估
依次按12条已登记路径独立运行；调度并发由当前可用授权资源决定，不改变题目和优化器步数。每步保存完整G8、准确生成版本、原生J、优势/概率/更新和每次耗时等；将12、24步与父端点固定为不可变身份。所有12条路径24步完成、两个检查点及回执锁定后才解封本次正式评测清单并评估26个端点（包含两父各一次）。不以12步效果选checkpoint，不从未完成路径拼总结果，不追加训练或因GPU小时数到限提前终止科学预算。独立作业时限按SLURM正常恢复。
 
端点raw须保留完整九键与case_id、input_id、实际seed、原始full_text/assistant_prefill/suffix_text、finish_reason、模型/预处理/提示/图片/输出token身份。长前缀是同一个assistant消息，扫描范围包含prefill，计费记录区分预置与新生成token；预置事实不能当作模型感知。
```bash
python score_eval.py --manifest frozen/clarification_manifest.jsonl --cases frozen/private_cases.jsonl --inputs frozen/inputs.jsonl --raw run/clarification_raw.jsonl --out run/clarification_scores.jsonl
python score_eval.py --manifest frozen/evaluation_manifest.jsonl --cases frozen/private_cases.jsonl --inputs frozen/inputs.jsonl --raw run/eval_raw.jsonl --out run/eval_scores.jsonl
python analyze.py --manifest frozen/evaluation_manifest.jsonl --scores run/eval_scores.jsonl --manifest-sha256 4885ebc75fda00ed8dfde2e34ee0c6db60846a8e2418119524c2d6d8d72947ac --out run/analysis.json
```
单独用 runtime_contract.py check-smoke / check-path 逐项核验真实回执；传入的 --manifest-sha256 使用交付时 frozen/FREEZE.json 的SHA，--plan-sha256 从 frozen/receipt_plans/INDEX.json 预先读取对应文件摘要，不以实际输出回填。上面的分析命令则使用 **evaluation_manifest.jsonl** 的文件摘要。这两个摘要用途不同。

### E. 最终交付后停止
按 docs/05_REPORT.md 汇总：主比较与每父/seed、保持与所有不利结果、语义诊断、对照实际输入差异、资源和技术历史。写一份中文报告和轻量可复算ZIP。其后不自动运行M、b2、奖励消融、旧formal TEST、额外seed或大规模训练。本包用于把已拆清的问题交给有集群权限的执行端；本次生成文件没有提交任何SLURM作业。
