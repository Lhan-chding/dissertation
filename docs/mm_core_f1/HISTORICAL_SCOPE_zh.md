# MM-CORE F1 历史资产边界

本轮选择 **路径 B：新版本同次 `readings → answer` 协议**。这是按证据完整性作出的选择；没有使用新模型测量结果挑选协议。历史结果保持其原始定义，不把执行器答案或不同请求的字段拼成新的 P/C/A 表。

## 本次实际检查

`inventory_assets.py` 对指定仓库源码和明确指定的历史压缩包进行只读检查，不遍历用户其他目录，不解压或改写历史文件。私有运行目录的 `ASSET_CONTRACTS.csv` 按九个分支列出六字段；`ASSET_MANIFEST.json` 保存源码和每个归档成员的 SHA-256、原始行字段名、缺失项和服务器证据引用。代码实现与实际执行证据分别标为 `source_only` 和 `source_and_archived_records`。

本次读到 EOF 并验证 gzip 完整性的历史包包括：V4 原始表 18 个文件、V4 事实包 42 个文件、C2 证据包 26 个文件、C3 证据包 65 个文件。归档路径安全与成员唯一性同时检查；这些完整性检查不认证历史结论，也不补齐归档未包含的图像、权重或 optimizer 状态。

| 历史分支 | 已检查的证据 | 仍未证明的部分 | 对本轮的影响 |
|---|---|---|---|
| compbias 三段式视觉输出 | `qwen_smoke.py` 的图像路由、同次三标签输出代码和 parser | 实际运行的原图、processor tensor、原始回答/token 与权重的完整对应关系 | 有实现不等于已恢复协议；六字段缺证部分保留 unknown |
| compbias recoverability bridge | 第一阶段视觉调用与第二阶段独立文本调用的代码 | 历史运行载荷完整性 | 不满足同次双字段 |
| V4 视觉观察 | 原始 scene/observation 表包含真值、图像引用、`image_grid_thw`、visual token count；源码确有图像输入调用 | 本次包不含原图字节、完整权重及所有原始观察 token | 仅能恢复部分视觉契约 |
| V4 自由答案链 | 原始逐题指标及图像/模型 hash；代码分别生成观察、修正、运算及最终答案 | 原始最终答案 token 的完整恢复 | 最终答案属于后续独立请求，不能与读数拼成本轮联合事件 |
| V4 确定性答案端点 | 原始指标和确定性计算代码 | 独立模型答案字段不存在于这个端点定义 | `executor_h`，与自由生成端点分别保留 |
| V5 Study A | 图片打开、视觉观察与 raw/token 保存代码 | 实际 capture 运行文件未在指定资产中恢复 | unknown，不从 VL 名称推出视觉实测 |
| V5 Study B/C | 文本 common-world backend 和答案运算代码 | 本次未恢复完整原始执行包 | 只记代码契约，不冒称原始测量 |
| Study C2 | 原始 completion、support token IDs、truth/operation、配置和哈希；调用代码仅准备文本 | adapter 字节与完整训练状态 | 四整数 action；答案奖励来自执行器，旧 X/S/F/U 保持原定义 |
| Study C3 | 原始 completion、token IDs、decoder、prompt hash、训练/评价清单；文本输入 backend | adapter 字节与完整训练状态 | free 与 FSA decoder 分开；执行器 reward 不是模型自行生成答案 |

原始表中存在非法回答、越界四元组和解释性长文本；它们仍保留在归档中。本次不通过宽松解析或重新选取 token 修饰历史成功率。C3 曾使用 FSA 是历史事实，本轮新协议不继承该约束解码。

## 当前公共基座与环境

历史 ModelScope 路径在当前注册服务器上已检查为不存在。注册缓存起初只有其他模型的完整目录；Qwen2.5-VL 的锁目录不能视为权重。随后本轮执行端下载了固定的 `Qwen/Qwen2.5-VL-3B-Instruct` revision `66285546d2b821cf421d4f5eb2576359d3770cd3`，服务器回执记录全部 14 个文件的字节数与 SHA-256。实际模型和 processor 路径只存私有清单。

这是本轮新取得的公共基座，不声称与旧 ModelScope snapshot 字节相同。`ASSET_MANIFEST.json` 的模型状态 `server_receipt_verified` 表示已读入并核验服务器下载哈希回执；没有把它写成本地重算权重，也没有把下载完成写成 GPU 推理或 processor 可读性通过。权重和 processor 汇总哈希按排序后的 `name/bytes/sha256` 条目进行 canonical JSON SHA-256，逐文件值仍完整保留。

注册 Python 环境的只读检查记录 torch `2.13.0+cu130`、transformers `5.14.1`、PEFT `0.19.1`、accelerate `1.14.0`，未安装 TRL。实际 tensor shape、attention backend、processor 重采样、图像 token 数及训练器能力须由各自工程回执确认；本表不以软件版本推断它们通过。

## 旧作业和五张 PRO 6000

检查发现 SER-J23 的停止及服务器数据删除发生在本轮之前，已有用户停止回执。本轮只读复核了五个已登记作业的所有者和调度终态，以及旧运行目录不存在；不再次取消作业或删除数据。原状态是用户取消，不是科学完成或科学失败。详细身份、既有删除范围、保留回执和本轮观察时间在私有 `ASSET_SER_J23_PRIOR_*` 与 `ASSET_SERVER_INVENTORY.json` 中；正式本轮停止判定由 `SER_J23_STOP_RECEIPT.json` 给出。

用户指定五张 PRO 6000 作为资源规划上限。资产检查时，所查询可用节点合计只有三张未分配 PRO 6000，且存在账户/QoS 的 GPU、并行作业及提交数限制。因此“五张”不是实时独占承诺，也不自动扩大 F1 的 completion、更新或 GPU-hour 硬预算。精确调度快照带观察时间保存在私有清单，提交前仍需复核；不取消其他项目的作业来取得资源。

## 可重建入口和状态边界

运行 `scripts/mm_core/inventory_assets.py --output-dir <私有清单目录>`。可用 `--archive v4_raw=<文件>`、`--archive v4_facts=<文件>`、`--archive c3=<文件>` 指定原始包；C2 默认读取仓库已有证据包。通过 `--server-evidence <JSON>` 和 `--model-receipt <JSON>` 绑定本轮服务器观察和模型下载回执。缺少所需归档或原始字段时，该分支退回 unknown，不沿用上述历史事实冒充当前核验。

本文件仅记录资产审计与环境观察。格式检查、测量、桥接、恢复试验、可读性及费用以本轮各阶段回执为准；MM-DEV 和 MM-LOCK 均不由该清单授权。
