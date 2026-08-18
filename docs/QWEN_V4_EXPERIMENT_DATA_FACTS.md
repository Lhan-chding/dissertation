# Qwen v4 实验数据与运行事实汇编

## 文档范围与证据规则

- 文档范围：Qwen2.5-VL v4 的既有审计、能力链、候选打分、层间同化、缓存一致性和接口阶梯运行记录，以及计划中后续阶段的执行状态。
- 本文只记录输入、配置、命令、哈希、计数、数值、输出路径、状态和控制台错误；不包含对结果的解释、推断或结论。
- 记录时间：2026-08-18；本地代码修订：`1bb39955711f3065c6d9479cb65df908117d3f75`。
- 项目根目录：`/cloud/cloud-ssd1/dissertation`。
- 计划文件：`docs/Qwen25VL_Constraint_Assimilation_Natural_State_RL_Codex_Plan_v4.md`，SHA-256 为 `01e532f8c7fe5439c70e8cd8de81ff3448465d8d401dbbf478c76aebaf49641e`。
- 下表中的“本地可见”仅指本报告生成时的当前工作区；服务器上已生成、但未同步到当前工作区的文件，保留其服务器路径和已记录哈希。

## 固定模型、数据与运行环境

| 字段 | 值 |
| --- | --- |
| 模型 | `Qwen2.5-VL-3B-Instruct` |
| 本地模型目录 | `/model/ModelScope/Qwen/Qwen2.5-VL-3B-Instruct` |
| 模型快照 SHA-256 | `e104df572eab7267bc2a63c11d70f7c8b1ebf8f85aa835d17e2c2641447bca87` |
| 离线环境变量 | `HF_HUB_OFFLINE=1`；`TRANSFORMERS_OFFLINE=1` |
| Python 源码路径 | `PYTHONPATH=/cloud/cloud-ssd1/dissertation/src` |
| 固定图像尺寸 | 280 × 280 pixels |
| 图像边长约束 | 两个边长均为 28 的倍数 |
| Phase-C screen records 路径 | `outputs/recoverability_v1/cva_recoverability_causal_v2/phase_c_screen/screen_records.jsonl` |
| Phase-C screen records SHA-256 | `f964dd6c005bd7344804aca8c33de2f621cc8e171f8d0f4ccc73a08081f2414a` |
| Phase-C 完整视觉数据 manifest SHA-256 | `bc57389dc3164b6aeba8d4565aecfaea3fa7ba171b4df4843c8ec86cbee8a19f` |
| Phase-C 完整视觉数据 records SHA-256 | `36e09f7e15107057fd1b942875d12259b1f281e0354b87c82ed17f420693c766` |
| S1 model class | `Qwen2_5_VLForConditionalGeneration` |
| S1 language layers | 36 |
| S1 vision depth | 32 |
| S1 module count | 839 |
| S1 introspection SHA-256 | `ed96d19a238d68497617071e29604313e0aae9a41a9e3bd24dbad451d87a0640` |
| S1 module manifest SHA-256 | `1c98fd8ba74fa5c30b8f585ffee5020544baf5be61f23e0c28c61a132973e8f0` |

S1 运行时清单要求的模块名：`model.visual.blocks.0`、`model.visual.blocks.31`、`model.visual.merger`、`model.language_model.layers.0`、`model.language_model.layers.35`、`model.language_model.norm`、`lm_head`。

## 阶段总览

| 阶段 | 脚本/产物 | 已记录状态 | 场景/调用数量 | 产物 SHA-256 |
| --- | --- | --- | --- | --- |
| S0 / Phase 0 | `00_audit_legacy.py` | 已生成审计文件 | 不加载模型 | 见 S0 表 |
| S1 | `01_introspect_qwen.py` | 已有运行时证据 | 839 modules | 见固定环境表 |
| S2 / Phase 1 | `02_run_capability_chain.py` | 已完成；作为后续哈希绑定输入 | 579 scenes；3,474 model calls | 见 S2 表 |
| S3 / Phase 2 candidate scoring | `03_score_candidates.py` | 已完成；作为后续哈希绑定输入 | 579 scenes；2,316 forwards | 见 S3 表 |
| S4 / Phase 2 layerwise | `04_layerwise_assimilation.py` | `PHASE_2_LAYERWISE_ASSIMILATION_EXECUTED` | 579 scenes；2,316 forwards；36 layers | 见 S4 表 |
| S5 / Phase 3 cache | `05_validate_cache_runner.py` | `PHASE_3_EXECUTED_WITH_DIAGNOSTICS` | 579 scenes；2,316 parity calls | `c52cb71d42c83e3a32c57c00e006f5117631b9cab25a2ef8fbe62001ff572351` |
| S6 / Phase 3 interface ladder | `06_run_interface_ladder.py` | 首次服务器执行受阻；无 S6 输出产物 | 预定 579 scenes；9,843 cells | 无 |
| Phase 4 LoRA | 计划中的支持注入式训练 | 本报告记录时未执行 | 无已记录训练调用 | 无 |
| Phase 5 policy support | 计划中的 checkpoint 测量 | 未执行 | 无 | 无 |
| Phase 6 RL | 计划中的 RL 实验 | 未执行 | 无 | 无 |
| Phase 7 multimodal | 计划中的真实多模态评估 | 未执行 | 无 | 无 |

## S0：冻结遗留证据审计

### 已生成文件

| 文件 | SHA-256 | 本地可见 |
| --- | --- | --- |
| `artifacts/v4/audit/claim_evidence_matrix.csv` | `ebb313a1893814373100918d8e4124ced0cdab23d3af6240609d7409000f52f4` | 是 |
| `artifacts/v4/audit/legacy_experiment_registry.csv` | `e735087be8ed45855194e39b27d7ba84ce39494991975fc3556910b21643c818` | 是 |
| `artifacts/v4/audit/legacy_hash_manifest.json` | `7891a6e8101b1bd72cdd59b0217c95d447d3b18715fcc61cce1d26dc298a8753` | 是 |
| `artifacts/v4/audit/scoring_contract.md` | `4f433bf75f5867ff85af15f531c64a3004288539e1ee3456e4a5a1a8334b6871` | 是 |

### 遗留实验注册表的数值字段

| experiment_id | interface family | model calls | true-world recoveries | observation copies | evidence status |
| --- | --- | ---: | ---: | ---: | --- |
| `small_neural_natural_replay_v2` | `controlled_exact_natural_fork` | 未填 | 未填 | 未填 | `frozen_plan_summary` |
| `stage2_v2_forward_dsl` | `trusted_state_forward_program` | 24 | 未填 | 未填 | `frozen_legacy_result` |
| `phase_c_v3_dsl` | `strict_result_program` | 27,840 | 未填 | 未填 | `frozen_legacy_result_interface_invalid` |
| `qwen_world_only_v1r1_12` | `text_replay` | 12 | 1 | 未填 | `frozen_repository_summary` |
| `qwen_world_only_valid_cue_50` | `text_replay` | 50 | 0 | 41 | `frozen_plan_summary_raw_summary_not_local` |
| `qwen_world_only_no_cue_100` | `text_replay` | 100 | 未填 | 未填 | `awaiting_hash_bound_server_evidence` |

### 遗留数据中明确记录的缺失项

| experiment_id | required payload | status |
| --- | --- | --- |
| `qwen_world_only_no_cue_100` | raw no-cue/valid-cue records plus aggregate summary | `awaiting_hash_bound_server_evidence` |

遗留评分契约还记录了 `qwen_world_only_valid_cue_50` 的聚合计数：0/50 true recoveries、41/50 complete copies、9/50 non-recovering edits。当前工作区没有该 100-call no-cue/valid-cue 原始聚合文件。

## S2：能力链执行（Phase 1）

| 字段 | 值 |
| --- | --- |
| 源场景数 | 580 |
| 纳入的可恢复场景数 | 579 |
| 排除的歧义场景数 | 1（`trend`） |
| 纳入 family counts | `cross_series=208`；`duplicate_encoding=182`；`trend=189` |
| 排除 family counts | `trend=1` |
| 模型调用数 | 3,474（579 × 6） |
| T1 分配 | YES=290；NO=289 |
| T5 true-label slots | 145、145、145、144 |
| 数值域 | `2..18` |
| `max_new_tokens` | 32 |
| decoding | `do_sample=false` |
| seed | `2026081701` |
| bootstrap resamples | 10,000 |

| S2 输出文件 | SHA-256 | 本地可见 |
| --- | --- | --- |
| `artifacts/v4/capability_chain/per_scene.csv` | `d01c391e136ed0e5c0ed52e50fe70f6ec128d221d218e7012c9adbcb4293929f` | 否 |
| `artifacts/v4/capability_chain/summary_by_family.csv` | `8837a7275915f5a90c91eae8378ff6bd0466819842381996db1be8e4925705c7` | 否 |
| `artifacts/v4/capability_chain/paired_gaps.json` | `a2a5acb5b203719e7e5225e56643a25a4189277d13704cccd4ec86d61573e256` | 否 |

## S3：候选世界 teacher-forced 打分（Phase 2）

| 字段 | 值 |
| --- | --- |
| scenes | 579 |
| cue conditions | `no_cue`、`valid_cue`、`sham_cue`、`counterfactual_cue` |
| candidate count | 4 |
| standard forward passes | 2,316（579 × 4） |
| generation | 不调用 generation |
| labels and token IDs | `A=32`、`B=33`、`C=34`、`D=35` |
| true-label slots | 145、145、145、144 |
| seed | `2026081701` |
| bootstrap resamples | 10,000 |

| S3 输出文件 | SHA-256 | 本地可见 |
| --- | --- | --- |
| `artifacts/v4/tokenizer/candidate_labels.json` | `a7a448f230038698c4127b220362c95d47f57cef90cc7904e71b1dacacc04dbd` | 否 |
| `artifacts/v4/candidate_scoring/per_scene.jsonl` | `c303731438760a30fb2f78a489d20465a1c1e292b01941795f29351a0e234a62` | 否 |
| `artifacts/v4/candidate_scoring/summary.json` | `5e366dfecb2a4fd530407f896326bc99d3605385cfdadea34c0f478392280c73` | 否 |

## S4：层间约束同化（Phase 2）

### 执行元数据与输出

| 字段 | 值 |
| --- | --- |
| status | `PHASE_2_LAYERWISE_ASSIMILATION_EXECUTED` |
| scenes | 579 |
| conditions | 4 |
| layerwise forwards | 2,316 |
| language layers | 36 |
| `phase_2_revision` | `fa8f9e64cf37190ffa8ba70206691fb043be3f1f` |
| `phase_2_config_sha256` | `39ac4534cf2786f18ea26bfa84d3230edfdd205f3397502934a90b792724401f` |
| `phase_2_package_lock_sha256` | `75fc91ef1fa1b217c07485242b3036bb3a789e4a9ebbd715f2e61639541c6c7a` |
| `generation_invoked` | false |
| `training_invoked` | false |
| `rl_invoked` | false |
| `subjective_success_threshold_applied` | false |
| `profile_counts.successful_revision` | 81 |
| `profile_counts.transient_assimilation` | 31 |
| `profile_counts.persistent_but_insufficient_assimilation` | 467 |

| S4 输出文件 | SHA-256 | 本地可见 |
| --- | --- | --- |
| `artifacts/v4/layerwise_assimilation/per_scene.jsonl` | `e696d12bb8cb3e6142a3d6ecc6de9474c3e72e3ac85e0c7334005a249556a4af` | 否 |
| `artifacts/v4/layerwise_assimilation/summary.json` | `53eab07dcd70fce6970a63ce1831ec6369164e92320f051e717decdeb1b790c0` | 否 |

### S4 final-layer paired effects

所有下表 `number_of_scenes=579`，`confidence=0.95`。

| effect | estimate | ci_low | ci_high |
| --- | ---: | ---: | ---: |
| `counterfactual_minus_no_cue_target_margin` | 1.4540155440414508 | 1.3877374784110534 | 1.5209412780656304 |
| `counterfactual_minus_sham_target_margin` | 0.24330742659758203 | 0.1584628670120898 | 0.32599309153713296 |
| `sham_minus_no_cue_margin` | 1.1319084628670122 | 1.0451208981001727 | 1.2182642487046633 |
| `valid_minus_no_cue_margin` | 1.113341968911917 | 1.0572107081174438 | 1.168825561312608 |
| `valid_minus_sham_margin` | -0.018566493955094993 | -0.09542314335060449 | 0.059153713298791016 |

### S4 各层均值（layer 0 至 layer 35）

```text
counterfactual_minus_no_cue_target_margin =
[0.030592211787564768, 0.022196135578583766, 0.018998272884283247, 0.017345369170984455, 0.00966105354058722, 0.013655008635578584, 0.007614465986920556, -0.004408867659758204, -0.008254122034049816, -0.011855927567819651, 0.0018215673575129533, -0.005262305699481865, 0.0014125580202936096, -0.0031742497841105353, -0.02223513904630829, -0.022039024735155913, -0.02145064227115717, -0.0034140124016269293, -0.0070433937823834196, 0.00025468210276338515, -0.0030441218295451477, -0.0007215641529037133, 0.03159494350611237, 0.05429818593158623, -0.060238760781823456, -0.022471261559784517, 0.19211192287517542, 0.5552811960276338, 0.5784960600172712, 0.6150420984455959, 0.8112316493955095, 2.028929188255613, 2.2905332469775477, 2.4735535405872193, 2.2194516407599307, 1.4540155440414508]

counterfactual_minus_sham_target_margin =
[0.004439227115716753, 0.0028065630397236616, -0.0005397236614853195, -0.0002023963730569948, 0.00016191709844559586, 0.0015989313471502591, -0.0022991844203599787, 0.001720369170984456, -0.008264228675855462, -2.573438258986399e-05, 0.0049047387737478415, 0.0008230785837651123, 0.0031196976366850175, -0.0004958711139896373, -0.0024871562634930914, -0.004664037305868145, -0.003086544689119171, -0.0030432422749942655, -0.005105448510362694, -0.002713798035405872, -0.0020569124370041288, -0.0051236852668933505, 0.0007216563908341429, -0.010692607965288179, -0.044889631254891246, -0.0026257832845052085, 0.028329878891070272, 0.01829663212435233, 0.017655710276338516, -0.011678270725388601, 0.03923791018998273, 0.06363341968911918, 0.26155008635578586, 0.24805699481865284, 0.3590241796200345, 0.24330742659758203]

sham_minus_no_cue_margin =
[0.017051894430051815, 0.013128778065630398, 0.010821459412780657, 0.008129587651122625, 0.0015112262521588947, 0.0065239097582038, -0.004137905562273991, -0.0067195595854922276, 0.0030622136407565574, -0.006999343582179674, -0.00182831390328152, -0.0015888115284974093, 0.0017197795070720667, -0.0031371437823834196, -0.011110717562607945, -0.01644352598305605, -0.009726832361830742, -0.0003722064029779253, -0.0016967562607944733, 0.009971394645941278, 0.00020182976862721287, 0.010489192033678757, 0.027378413343676633, 0.04552669228667422, -0.02019957969019862, -0.033932410785365394, 0.14046445329366364, 0.49241688255613125, 0.5194705310880829, 0.6098877374784111, 0.7906411917098446, 2.0329231433506045, 2.0786917098445596, 2.3356001727115716, 1.93566493955095, 1.1319084628670122]

valid_minus_no_cue_margin =
[0.014285389005829015, 0.008649071675302246, 0.01277795768566494, 0.00948564335060449, 0.004871006044905008, 0.006213568652849741, 0.006477178069593993, 0.002624406303972366, 0.004084050140644396, 0.00011214814655521374, 0.005174600604490501, -0.008944233052677029, -0.0070179361135848444, -0.00484064658894646, -0.011970902148100173, -0.018623345462357648, -0.009822970639032815, -0.002454108730704879, -0.0017321756260794473, 0.0008635578583765112, -0.004222494332901555, 0.0032229843534953852, 0.018815944437741823, 0.05045211212210911, -0.041083340817782546, -0.021822051474879235, 0.1382981664364408, 0.41249730138169255, 0.4322646804835924, 0.49993253454231434, 0.6672333765112263, 1.8116364421416236, 1.9781411917098446, 2.1499352331606216, 1.805699481865285, 1.113341968911917]

valid_minus_sham_margin =
[-0.0027665054242227978, -0.004479706390328152, 0.0019564982728842834, 0.0013560556994818653, 0.0033597797927461138, -0.0003103411053540587, 0.010615083631867984, 0.009343965889464595, 0.0010218364998878387, 0.007111491728734888, 0.0070029145077720205, -0.00735542152417962, -0.008737715620656911, -0.0017035028065630396, -0.000860184585492228, -0.0021798194793015975, -9.613827720207254e-05, -0.0020819023277269537, -3.541936528497409e-05, -0.009107836787564766, -0.0044243241015287675, -0.007266207680183371, -0.00856246890593481, 0.004925419835434882, -0.020883761127583927, 0.012110359310486155, -0.002166286857222852, -0.07991958117443869, -0.0872058506044905, -0.10995520293609672, -0.1234078151986183, -0.221286701208981, -0.10055051813471502, -0.18566493955094993, -0.12996545768566495, -0.018566493955094993]
```

## S5：缓存与完整历史的精确一致性（Phase 3）

### 配置与执行元数据

| 字段 | 值 |
| --- | --- |
| status | `PHASE_3_EXECUTED_WITH_DIAGNOSTICS` |
| schema version | 2 |
| scenes | 579 |
| parity calls | 2,316 |
| calls per scene | 4 |
| cue conditions | `no_cue`、`valid_cue`、`sham_cue`、`counterfactual_cue` |
| decoding | deterministic greedy；`do_sample=false`；`temperature=0.0` |
| `max_new_tokens` | 32 |
| MRoPE axes | 3 |
| exact generated logits | 不要求；记录全词表差异 |
| exact generated tokens | 要求；发生 token divergence 的 call 记为 diagnostic-only |
| `generation_invoked` | false |
| `training_invoked` | false |
| `rl_invoked` | false |
| `subjective_success_threshold_applied` | false |
| `i4_primary_eligible`（整个 artifact） | false |

### S5 总计数

| `call_counts` 字段 | 计数 |
| --- | ---: |
| `total` | 2,316 |
| `exact_token_parity` | 2,283 |
| `mismatched_token` | 33 |
| `primary_eligible` | 2,283 |
| `diagnostic_only` | 33 |

### S5 按 family 的计数

| family | total | exact_token_parity | mismatched_token | primary_eligible | diagnostic_only |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cross_series` | 832 | 819 | 13 | 819 | 13 |
| `duplicate_encoding` | 728 | 719 | 9 | 719 | 9 |
| `trend` | 756 | 745 | 11 | 745 | 11 |

### S5 按 cue condition 的计数

| cue condition | total | exact_token_parity | mismatched_token | primary_eligible | diagnostic_only |
| --- | ---: | ---: | ---: | ---: | ---: |
| `counterfactual_cue` | 579 | 575 | 4 | 575 | 4 |
| `no_cue` | 579 | 574 | 5 | 574 | 5 |
| `sham_cue` | 579 | 564 | 15 | 564 | 15 |
| `valid_cue` | 579 | 570 | 9 | 570 | 9 |

### S5 输出文件

| 文件 | SHA-256 | 本地可见 |
| --- | --- | --- |
| `artifacts/v4/cache/cache_parity.json` | `c52cb71d42c83e3a32c57c00e006f5117631b9cab25a2ef8fbe62001ff572351` | 否 |

该 JSON 中记录了 `mismatch_call_ids` 的 33 个元素、每步 tensor shape/dtype、realized-token logits、argmax IDs、maximum absolute/relative difference、nonzero count、L2 difference 和最大绝对差异的 token ID。当前工作区没有该服务器 JSON 文件，因此本报告不复制未在当前工作区或控制台记录中可核验的逐 call 数值。

## S6：接口阶梯（Phase 3）

### 设计参数

| 字段 | 值 |
| --- | --- |
| interfaces | `I0_hard_text_symbolic_recovery`；`I1_soft_report_diagnostic`；`I2_candidate_world_diagnostic`；`I3_same_conversation_visual_revision`；`I4_exact_cached_natural_continuation` |
| cells per scene | 17 |
| planned interface cells | 9,843（579 × 17） |
| I0 runtime calls | 2,316 |
| I1 runtime calls | 579 |
| I1 pre-cue condition | `no_cue` |
| I1 soft-report top-k | 4 |
| I2 source | S3 candidate decision |
| I3 source | S5 full-history output |
| I4 source | S5 cached-continuation output |
| decoding | `do_sample=false`；`temperature=0.0`；`max_new_tokens=32` |
| bootstrap resamples | 10,000 |
| I4 token-divergence 处理 | 保留所有 diagnostic cells；同一 scene 从 I0/I3/I4 paired primary complete-case 集排除 |

### S6 首次阻断后的代码修订记录

| 字段 | 记录 |
| --- | --- |
| 首次阻断消息 | `BLOCKED: S6 numeric values must each be a stable single token` |
| 首次阻断时已完成的模型加载 | `Loading weights: 824/824` |
| 首次阻断后的改动 | 数值 literal 使用完整 token 序列；payload 中记录完整 `generated_token_ids` 和 candidate `token_ids`；`score_basis=first_token_logit` |
| 新增预检位置 | tokenizer 加载后、579 个 scene runtime 调用前 |
| 本地联合测试 | `65 passed`（S6 interface、Phase 4、audit gates、server scripts 测试集合） |
| 首次阻断后服务器重跑 | 本报告生成时未记录 |
| S6 `per_scene.jsonl` / `summary.json` | 本报告生成时仍不存在 |

### 首次 S6 服务器运行记录

| 字段 | 值 |
| --- | --- |
| 代码修订 | `1bb39955711f3065c6d9479cb65df908117d3f75` |
| 模型加载 | `Loading weights: 100%`；824/824 权重分片 |
| 记录的阻断信息 | `BLOCKED: S6 numeric values must each be a stable single token` |
| `artifacts/v4/interface_ladder/per_scene.jsonl` | 未生成 |
| `artifacts/v4/interface_ladder/summary.json` | 未生成 |
| 后续哈希/JSON 命令 | 对上述两个不存在文件均返回 `No such file or directory` |

## 已记录的运行错误时间线

| 阶段 | 记录的 `BLOCKED` 信息/异常 | 后续记录状态 |
| --- | --- | --- |
| S4 | `final-layer candidate logits do not match the standard forward pass for 'C': 28.25 versus 28.125` | 后续修订后重新运行 |
| S4 | `final-layer candidate logits do not match the standard forward pass for 'A': 31.0 versus 27.375` | 后续完成并产生 S4 两个输出哈希 |
| S5 | `'image_grid_thw'` | 后续修订后重新运行 |
| S5 | `S5 runtime MRoPE trace is malformed` | 后续修订后重新运行 |
| S5 | `S5 runtime exposed no cache_position trace` | 后续修订后重新运行 |
| S5 | `S5 generated-logit parity failed at step 0` | 后续修订后重新运行 |
| S5 | `S5 generated-token parity failed` | 后续修订后改为保存逐 call diagnostics 并继续其余 calls |
| S5 | `S5 generated-token parity failed: call_id=phase-c-screen-000118.sham_cue, ... cached_token=16, full_token=23, ...` | 后续修订后完成全部 2,316 calls 和 579 visual states |
| S6 | `S6 numeric values must each be a stable single token` | 本报告生成时 S6 尚无输出文件 |

## 计划中 Phase 4--7 的执行状态与已冻结参数

### Phase 4：支持注入式 LoRA

| 字段 | 已冻结计划值/状态 |
| --- | --- |
| 执行状态 | 未执行；未记录 checkpoint 或训练 artifact |
| 可训练范围 | 语言侧 attention/MLP LoRA |
| 冻结范围 | vision tower；patch merger/projector；base language weights |
| target list 生成方式 | 通过 `named_modules()` 生成精确 target list |
| 预定 manifest | `artifacts/v4/training/trainable_parameter_manifest.json` |
| 预定 frozen hashes | `artifacts/v4/training/frozen_hashes.json` |
| C0 | `Format-only LoRA`；最小四整数输出 |
| C1 | `Forward-Arithmetic LoRA`；给定正确变量后的 sum/difference/max-minus-min 或 fact verification |
| T | `Constraint-Recovery LoRA`；fact verification、conflict detection、error index、replacement value、global fact verification、free world recovery |
| 最终训练输出格式 | `a,b,c,d` |
| `precision` | `bf16` |
| `lora_rank` | 16 |
| `lora_alpha` | 32 |
| `lora_dropout` | 0.0 |
| `gradient_checkpointing` | true |
| `vision_frozen` | true |
| `merger_frozen` | true |
| 训练数据类别 | `symbolic_support_train`；`natural_error_support_train` |
| `support_dev` | 用于预先冻结 learning rate、batch、epoch |

### Phase 5--7

| 阶段 | 计划输出路径 | 本报告生成时的执行状态 |
| --- | --- | --- |
| Phase 5 policy support | `artifacts/v4/support/policy_support_by_scene.parquet`；`artifacts/v4/support/informative_group_rate.json`；`artifacts/v4/support/pass_at_k.csv` | 未执行；无输出文件 |
| Phase 6 RL | Phase 6 训练与诊断输出未在当前工作区列出 | 未执行；无输出文件 |
| Phase 7 real multimodal | Phase 7 评估输出未在当前工作区列出 | 未执行；无输出文件 |

## 哈希绑定的执行输入清单

| 输入 | SHA-256 | 使用阶段 |
| --- | --- | --- |
| Phase-C screen records | `f964dd6c005bd7344804aca8c33de2f621cc8e171f8d0f4ccc73a08081f2414a` | S2、S3、S4、S6 |
| S2 per-scene | `d01c391e136ed0e5c0ed52e50fe70f6ec128d221d218e7012c9adbcb4293929f` | S3、S4、S6 |
| S2 summary | `8837a7275915f5a90c91eae8378ff6bd0466819842381996db1be8e4925705c7` | S3、S4、S6 |
| S2 paired gaps | `a2a5acb5b203719e7e5225e56643a25a4189277d13704cccd4ec86d61573e256` | S3、S4、S6 |
| S3 candidate labels | `a7a448f230038698c4127b220362c95d47f57cef90cc7904e71b1dacacc04dbd` | S4、S6 |
| S3 per-scene scores | `c303731438760a30fb2f78a489d20465a1c1e292b01941795f29351a0e234a62` | S4、S6 |
| S3 summary | `5e366dfecb2a4fd530407f896326bc99d3605385cfdadea34c0f478392280c73` | S4、S6 |
| S4 per-scene | `e696d12bb8cb3e6142a3d6ecc6de9474c3e72e3ac85e0c7334005a249556a4af` | S5、S6 |
| S4 summary | `53eab07dcd70fce6970a63ce1831ec6369164e92320f051e717decdeb1b790c0` | S5、S6 |
| S5 cache parity | `c52cb71d42c83e3a32c57c00e006f5117631b9cab25a2ef8fbe62001ff572351` | S6 |

## 运行命令与参数

所有服务器阶段使用的环境初始化：

```bash
cd /cloud/cloud-ssd1/dissertation
source .venv/bin/activate
export PYTHONPATH=/cloud/cloud-ssd1/dissertation/src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

| 阶段 | 可执行脚本 | `--execute` | 所需哈希绑定输入数 |
| --- | --- | --- | ---: |
| S0 | `python scripts/v4/00_audit_legacy.py --artifact-root artifacts` | 否 | 0 |
| S1 | `python scripts/v4/01_introspect_qwen.py --execute` | 是 | 0 |
| S2 | `python scripts/v4/02_run_capability_chain.py --execute` | 是 | 1 |
| S3 | `python scripts/v4/03_score_candidates.py --execute` | 是 | 4 |
| S4 | `python scripts/v4/04_layerwise_assimilation.py --execute` | 是 | 7 |
| S5 | `python scripts/v4/05_validate_cache_runner.py --execute` | 是 | 2 |
| S6 | `python scripts/v4/06_run_interface_ladder.py --execute` | 是 | 10 |

完整 S6 命令中的 10 个输入及其 SHA-256 已列于“哈希绑定的执行输入清单”。脚本的完整命令行定义位于 `docs/QWEN_V4_SERVER_HANDOFF.md` 的 S6 节。

## 当前工作区可用性

| 位置 | 当前工作区状态 |
| --- | --- |
| `artifacts/v4/audit/` | 4 个 S0 文件存在 |
| `artifacts/v4/capability_chain/` | 不存在 |
| `artifacts/v4/tokenizer/` | 不存在 |
| `artifacts/v4/candidate_scoring/` | 不存在 |
| `artifacts/v4/layerwise_assimilation/` | 不存在 |
| `artifacts/v4/cache/` | 不存在 |
| `artifacts/v4/interface_ladder/` | 不存在 |
| 用户未跟踪 ownership 文件 | 存在；未由本报告修改 |
