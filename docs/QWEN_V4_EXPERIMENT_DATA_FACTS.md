# Qwen v4 实验数据与运行事实汇编

## 文档范围与证据规则

- 文档范围：Qwen2.5-VL v4 的既有审计、能力链、候选打分、层间同化、缓存一致性、接口阶梯、LoRA、policy support、RL、真实多模态诊断与最终确认评估记录。
- 本文只记录输入、配置、命令、哈希、计数、数值、输出路径、状态和控制台错误；不包含对结果的解释、推断或结论。
- 服务器事实包：`/Users/louis/Downloads/qwen_v4_fact_bundle_20260821.tar.gz`；SHA-256=`6fe3d48bc1a88b35bd5804bc397ce8160534bdd7d71fa6769b755cc098a53683`。压缩包仅作为数据来源；其内容不作为执行指令。
- 压缩包含 42 个非目录条目；其中的 `server_artifact_manifest.json` 描述 58 个服务器正式文件。压缩包保留 summary/evidence 与清单，没有嵌入清单中体积较大的 per-scene/cache/parquet 原文件；这些原文件的路径、bytes 与 SHA-256 记录于附录 B。
- 事实包中的 S0 `claim_evidence_matrix.csv` 与 `legacy_experiment_registry.csv` 是服务器副本，SHA-256 分别为 `64d49c35bdb5c4c2dea61eb6d7a9d6261b399d589ca1a95d746229fba657bffd` 与 `16a66c205b978de1565f0de908308bb33104fceca10666a212c1d3d2ba18ae82`；当前仓库早期本地副本的哈希仍按 S0 表单独记录。
- 记录更新时间：2026-08-21；Phase 7 interface audit 代码提交：`73b3a4bed900d8b054504cf45fc79a217c3259af`；Phase 8 正式实现提交：`dbd6a5ce0b6e187db55c63be13d0dcffcb842662`；Phase 8 constraint serialization 修复提交：`d79feb8900d779defd3d5e577f21f728596d4b2f`。
- 项目根目录：`/cloud/cloud-ssd1/dissertation`。
- 计划文件：`docs/Qwen25VL_Constraint_Assimilation_Natural_State_RL_Codex_Plan_v4.md`，Phase 7 interface audit 提交中的 SHA-256 为 `8bbd556828014262f3b02ba71b957361179d0389ec41258fb02ea6a57c77a58f`。
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
| S6 / Phase 3 interface ladder | `06_run_interface_ladder.py` | `PHASE_3_INTERFACE_LADDER_EXECUTED_WITH_DIAGNOSTICS` | 579 scenes；9,843 cells | per-scene `324dbf3289d6754685483feebcf0ebd785210c250c5360859391b6fb66c089e5`；summary `46ce49ed8ad74baccfba6cc5d2b474cd0260c4a852c6d94915bae7a8bcc99bcc` |
| Phase 4 LoRA | source preparation、support build 与 C0/C1/T LoRA 训练 | 服务器执行完成 | 564/564 steps；1 epoch | 输出根目录见 Phase 4 表 |
| Phase 5 policy support | Base/C0/C1/T 的 held-out policy-support 测量 | 服务器执行完成 | 4 个 checkpoint 的 128 行 scene-level 测量 | 32 个 held-out natural errors |
| Phase 6 RL | execution manifest、双 reward 数据、3 个 GRPO arm 与 5 checkpoint 评估 | `PHASE_6_RL_EVALUATED` | 173 个 RL 场景；3 × 64 steps；32 个评估场景 | 见 Phase 6 表 |
| Phase 7 multimodal | 七 checkpoint 完整多模态链的 support-dev diagnostic 与冻结 trace interface audit | `PHASE_7_MULTIMODAL_DIAGNOSTIC_EVALUATED`；`PHASE_7_INTERFACE_DIAGNOSTIC_AUDITED` | 32 scenes × 7 checkpoints = 224 rows；896 deterministic generation calls | interface audit SHA-256 `abbd06e6d1f76bbd549009f4993a748d6be5feac2d8b8f1a899aea438cea004b` |
| Phase 8 confirmatory | 四轴独立 confirm 数据冻结与七 checkpoint 最终评估 | `PHASE_8_CONFIRMATORY_EVALUATED` | 128 fixed candidates；98 selected natural-error scenes；686 rows | per-scene `f897cc66f2c9162b3d034d387ec6a09e84939357eedc1f860af18d12747e8fe8`；summary `a46bd1a494a1468cbd7f62878474e7fc877a10f88a09cc6e88a611f1de6bc8af` |

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

### S2 按 family 的六项能力链结果

`parse_rate` 与 `accuracy` 均以 scene 为统计行；`number_of_scenes` 为各 family 的纳入场景数。

| family | scenes | task | parse_rate | accuracy |
| --- | ---: | --- | ---: | ---: |
| `cross_series` | 208 | T1 | 1.0 | 0.4951923076923077 |
| `cross_series` | 208 | T2 | 1.0 | 0.0 |
| `cross_series` | 208 | T3 | 1.0 | 0.3269230769230769 |
| `cross_series` | 208 | T4 | 1.0 | 0.1778846153846154 |
| `cross_series` | 208 | T5 | 1.0 | 0.2548076923076923 |
| `cross_series` | 208 | T6 | 0.9807692307692307 | 0.0 |
| `duplicate_encoding` | 182 | T1 | 1.0 | 0.9120879120879121 |
| `duplicate_encoding` | 182 | T2 | 1.0 | 0.4065934065934066 |
| `duplicate_encoding` | 182 | T3 | 1.0 | 0.7362637362637363 |
| `duplicate_encoding` | 182 | T4 | 1.0 | 0.1043956043956044 |
| `duplicate_encoding` | 182 | T5 | 1.0 | 0.8461538461538461 |
| `duplicate_encoding` | 182 | T6 | 1.0 | 0.10989010989010989 |
| `trend` | 189 | T1 | 1.0 | 0.5185185185185185 |
| `trend` | 189 | T2 | 1.0 | 0.13227513227513227 |
| `trend` | 189 | T3 | 1.0 | 0.5396825396825397 |
| `trend` | 189 | T4 | 1.0 | 0.07407407407407407 |
| `trend` | 189 | T5 | 1.0 | 0.2804232804232804 |
| `trend` | 189 | T6 | 0.49206349206349204 | 0.0 |

### S2 配对 gaps

| family | effect | estimate | 95% CI |
| --- | --- | ---: | --- |
| overall | `G_loc` | -0.40414507772020725 | [-0.4542314335060449, -0.3540587219343696] |
| overall | `G_search` | 0.41450777202072536 | [0.3747841105354059, 0.4542314335060449] |
| `cross_series` | `G_loc` | -0.14903846153846154 | [-0.22596153846153846, -0.0673076923076923] |
| `cross_series` | `G_search` | 0.2548076923076923 | [0.1971153846153846, 0.3125] |
| `duplicate_encoding` | `G_loc` | -0.6318681318681318 | [-0.7087912087912088, -0.5494505494505495] |
| `duplicate_encoding` | `G_search` | 0.7362637362637363 | [0.6703296703296703, 0.7967032967032966] |
| `trend` | `G_loc` | -0.4656084656084656 | [-0.5502645502645502, -0.37566137566137564] |
| `trend` | `G_search` | 0.2804232804232804 | [0.21693121693121692, 0.3439153439153439] |

`T5_establishes_full_recovery=false`。被排除场景为 `phase-c-screen-006172`；该场景记录两个 supported worlds。

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

### S3 按 cue condition 的均值

| cue condition | observed-world mean log-probability | true-world mean log-probability | observed-world mean rank | true-minus-observed margin | true-world mean rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| `counterfactual_cue` | -0.8292209294832025 | -1.6347045218838934 | 1.2089810017271156 | -0.8054835924006909 | 2.4110535405872193 |
| `no_cue` | -0.36599764467746715 | -2.175151357976258 | 1.0207253886010363 | -1.809153713298791 | 2.3333333333333335 |
| `sham_cue` | -0.890182905055699 | -1.5672122660228838 | 1.4110535405872193 | -0.6770293609671848 | 2.2642487046632125 |
| `valid_cue` | -0.7892231452499056 | -1.4854666685659679 | 1.1347150259067358 | -0.6962435233160622 | 2.174438687392055 |

### S3 target-margin 配对效应

| stratum | effect | estimate | 95% CI | scenes |
| --- | --- | ---: | --- | ---: |
| overall | `counterfactual_minus_no_cue_target_margin` | 1.454447322970639 | [1.3879533678756477, 1.5211571675302245] | 579 |
| overall | `sham_minus_no_cue_margin` | 1.1321243523316062 | [1.0453367875647668, 1.2184801381692574] | 579 |
| overall | `valid_minus_no_cue_margin` | 1.112910189982729 | [1.0567789291882557, 1.1683937823834196] | 579 |
| `cross_series` | `counterfactual_minus_no_cue_target_margin` | 1.4308894230769231 | [1.3263221153846154, 1.5348557692307692] | 208 |
| `cross_series` | `sham_minus_no_cue_margin` | 1.0973557692307692 | [0.9429086538461539, 1.2493990384615385] | 208 |
| `cross_series` | `valid_minus_no_cue_margin` | 0.9921875 | [0.9020432692307693, 1.0829326923076923] | 208 |
| `duplicate_encoding` | `counterfactual_minus_no_cue_target_margin` | 1.434065934065934 | [1.3125, 1.558379120879121] | 182 |
| `duplicate_encoding` | `sham_minus_no_cue_margin` | 1.1696428571428572 | [1.0233516483516483, 1.3186813186813187] | 182 |
| `duplicate_encoding` | `valid_minus_no_cue_margin` | 1.4237637362637363 | [1.3221153846153846, 1.5254120879120878] | 182 |
| `trend` | `counterfactual_minus_no_cue_target_margin` | 1.5 | [1.376984126984127, 1.6223544973544974] | 189 |
| `trend` | `sham_minus_no_cue_margin` | 1.1342592592592593 | [0.9900793650793651, 1.2764550264550265] | 189 |
| `trend` | `valid_minus_no_cue_margin` | 0.9464285714285714 | [0.8578042328042328, 1.0324074074074074] | 189 |

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

该 JSON 中记录了 `mismatch_call_ids` 的 33 个元素、每步 tensor shape/dtype、realized-token logits、argmax IDs、maximum absolute/relative difference、nonzero count、L2 difference 和最大绝对差异的 token ID。事实包中提取的 summary 路径为 `extracted/cache_parity_summary.json`；正式完整文件大小为 69,215,306 bytes。

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
| S6 `per_scene.jsonl` / `summary.json` | 该次阻断运行未生成；后续第三次运行已生成 |

### S6 第二次服务器执行记录

| 字段 | 记录 |
| --- | --- |
| 已完成场景数 | 25 / 579 |
| 已完成模型加载 | `Loading weights: 824/824` |
| 阻断消息 | `BLOCKED: S6 Stage-1 output lies outside the frozen numeric domain` |
| 输出 artifacts | 未生成 `artifacts/v4/interface_ladder/per_scene.jsonl`；未生成 `artifacts/v4/interface_ladder/summary.json` |
| 阻断后的代码修订 | I1 payload 保留 `raw_output`、`output_format_valid=true` 和 `numeric_domain_valid=false`；域外数值作为 I1 diagnostic 记录，不再终止 S6 |

### S6 第二次阻断后的追加代码记录

| 字段 | 记录值 |
| --- | --- |
| 修改对象 | I1 Stage-1 soft-report payload 与 I1 payload 结构验证 |
| 非四数字格式输出 | 保存原始 `raw_output`；`output_format_valid=false`；`numeric_domain_valid=false`；`positions=[]` |
| 域外但四数字格式输出 | 保存原始 `raw_output`；`output_format_valid=true`；`numeric_domain_valid=false`；保留 4 个生成位置的候选 logit 记录 |
| I1 接口角色 | `diagnostic_only=true`；不计入 I0/I3/I4 主配对估计 |
| 本地回归集合 | S6 interface、server scripts、Phase 4 training、audit gates：`68 passed` |
| 服务器重跑状态 | 该次代码修订记录产生时尚未执行第三次 S6；后续第三次运行已完成 |

### S6 第三次服务器执行记录（完成）

| 字段 | 记录值 |
| --- | --- |
| status | `PHASE_3_INTERFACE_LADDER_EXECUTED_WITH_DIAGNOSTICS` |
| schema version | 1 |
| number of source scenes | 579 |
| number of cells | 9,843 |
| I4 exact eligible call count | 2,283 |
| I4 token diagnostic call count | 33 |
| intervention diagnostic cell count | 2,895 |
| primary analysis cell count | 6,552 |
| primary paired scene count | 546 |
| statistical unit | scene (`scene_is_statistical_unit=true`) |
| training invoked | false |
| RL invoked | false |
| subjective success threshold applied | false |
| model snapshot SHA-256 | `e104df572eab7267bc2a63c11d70f7c8b1ebf8f85aa835d17e2c2641447bca87` |
| package lock SHA-256 | `2ae9d3da275f3fa7292cddc73d49502c15f2e907ad1abab822a565ef7132e194` |
| S3 candidate source SHA-256 | `c303731438760a30fb2f78a489d20465a1c1e292b01941795f29351a0e234a62` |
| S5 cache source SHA-256 | `c52cb71d42c83e3a32c57c00e006f5117631b9cab25a2ef8fbe62001ff572351` |
| S6 runtime source SHA-256 | `c1a0fe7800952c05d8761c2347a3ffd4d58d5e667b8c0fdfeb5d4cb179252f36` |
| source-stage cell counts | S3 candidate: 2,316; S5 cache: 4,632; S6 runtime: 2,895 |
| server output paths | `artifacts/v4/interface_ladder/per_scene.jsonl`; `artifacts/v4/interface_ladder/summary.json` |
| per-scene output SHA-256 | `324dbf3289d6754685483feebcf0ebd785210c250c5360859391b6fb66c089e5` |
| summary output SHA-256 | `46ce49ed8ad74baccfba6cc5d2b474cd0260c4a852c6d94915bae7a8bcc99bcc` |
| config SHA-256 | `3e8ed5f3f82274ba1bb7ac0d8083c9fb1a984b8ab4b3dffef271a892429baac7` |

### S6 注册效应

| effect | estimate | 95% CI | paired scenes |
| --- | ---: | --- | ---: |
| `fact_conditioned_revision` | 0.0695970695970696 | [0.0347985347985348, 0.1043956043956044] | 546 |
| `spontaneous_visual_revision` | 0.6684981684981685 | [0.63003663003663, 0.7087912087912088] | 546 |

### S6 按 interface 的计数与指标

下表的 exact、copy 与 counterfactual compliance 均为 estimate `[95% CI]`；I1/I2 为 intervention diagnostic，因此对应三项主分析指标为 null。

| interface | cells | primary cells | diagnostic cells | parse successes | exact | copy | counterfactual compliance |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| `I0_hard_text_symbolic_recovery` | 2,316 | 2,184 | 0 | 2,214 | 0.016025641025641024 [0.01098901098901099, 0.021062271062271063] | 0.5517399267399268 [0.5233516483516484, 0.5792124542124543] | 0.3131868131868132 [0.27472527472527475, 0.3516483516483517] |
| `I1_soft_report_diagnostic` | 579 | 0 | 579 | 0 | null | null | null |
| `I2_candidate_world_diagnostic` | 2,316 | 0 | 2,316 | 2,316 | null | null | null |
| `I3_same_conversation_visual_revision` | 2,316 | 2,184 | 0 | 2,316 | 0.5128205128205128 [0.48855311355311354, 0.5370879120879121] | 0.1565934065934066 [0.1346153846153846, 0.17857142857142858] | 0.8882783882783883 [0.8608058608058609, 0.9139194139194139] |
| `I4_exact_cached_natural_continuation` | 2,316 | 2,184 | 33 | 2,316 | 0.5128205128205128 [0.48855311355311354, 0.5370879120879121] | 0.1565934065934066 [0.1346153846153846, 0.17857142857142858] | 0.8882783882783883 [0.8608058608058609, 0.9139194139194139] |

### S6 I0/I3/I4 按 cue condition

| interface | cue | exact estimate [95% CI] | copy estimate [95% CI] | counterfactual compliance estimate [95% CI] |
| --- | --- | --- | --- | --- |
| I0 | `counterfactual_cue` | 0.0 [0.0, 0.0] | 0.4597069597069597 [0.4194139194139194, 0.5] | 0.3131868131868132 [0.27472527472527475, 0.3516483516483517] |
| I0 | `no_cue` | 0.0 [0.0, 0.0] | 0.5567765567765568 [0.5164835164835165, 0.5970695970695971] | null |
| I0 | `sham_cue` | 0.005494505494505495 [0.0, 0.01282051282051282] | 0.6868131868131868 [0.6483516483516484, 0.7252747252747253] | null |
| I0 | `valid_cue` | 0.05860805860805861 [0.040293040293040296, 0.07875457875457875] | 0.5036630036630036 [0.46153846153846156, 0.5457875457875457] | null |
| I3 / I4 | `counterfactual_cue` | 0.01098901098901099 [0.003663003663003663, 0.020146520146520148] | 0.09340659340659341 [0.0695970695970696, 0.11904761904761904] | 0.8882783882783883 [0.8608058608058609, 0.9139194139194139] |
| I3 / I4 | `no_cue` | 0.6684981684981685 [0.6282051282051282, 0.7087912087912088] | 0.2271062271062271 [0.19230769230769232, 0.26373626373626374] | null |
| I3 / I4 | `sham_cue` | 0.6336996336996337 [0.5934065934065934, 0.673992673992674] | 0.18498168498168497 [0.15201465201465203, 0.21794871794871795] | null |
| I3 / I4 | `valid_cue` | 0.7380952380952381 [0.6996336996336996, 0.7728937728937729] | 0.12087912087912088 [0.09523809523809523, 0.14835164835164835] | null |

### S6 按 family 汇总

| family | source scenes | primary paired scenes | cells | primary cells | diagnostic cells | parse successes | exact estimate [95% CI] | copy estimate [95% CI] | compliance estimate [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `cross_series` | 208 | 195 | 3,536 | 2,340 | 1,053 | 3,316 | 0.3230769230769231 [0.2931623931623932, 0.352991452991453] | 0.31794871794871793 [0.28717948717948716, 0.3491452991452991] | 0.6700854700854701 [0.629059829059829, 0.7111111111111111] |
| `duplicate_encoding` | 182 | 173 | 3,094 | 2,076 | 919 | 2,906 | 0.3998073217726397 [0.378131021194605, 0.4210019267822736] | 0.2615606936416185 [0.23892100192678228, 0.28516377649325625] | 0.7186897880539499 [0.6763005780346821, 0.7591522157996147] |
| `trend` | 189 | 178 | 3,213 | 2,136 | 956 | 2,940 | 0.32256554307116103 [0.29260299625468167, 0.35205992509363293] | 0.2818352059925094 [0.25374531835205993, 0.31086142322097376] | 0.704119850187266 [0.6629213483146067, 0.7434456928838952] |

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
| S6 | `S6 numeric values must each be a stable single token` | 后续改为完整 token sequence 后重跑 |
| S6 | `S6 Stage-1 output lies outside the frozen numeric domain` | 后续保留为 I1 diagnostic，第三次服务器执行完成 579 scenes / 9,843 cells |

## Phase 4--7 的执行状态与已冻结参数

### Phase 4：支持注入式 LoRA

| 字段 | 已冻结计划值/状态 |
| --- | --- |
| 执行状态 | 服务器执行完成；控制台记录 `READY: Phase 4 C0/C1/T LoRA adapters written below /cloud/cloud-ssd1/dissertation/artifacts/v4/training/runs/phase4-r1` |
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
| source preparation | `scripts/v4/07_prepare_phase4_support_sources.py --execute` |
| symbolic source | 程序生成 579 个 `symbolic_support_train` scenes；seed=`2026081804`；value domain=`2..18` |
| natural source | 完成的 S6 artifact 中 579 个 I1/no-cue frozen Stage-1 raw outputs；仅保留单位置、域内错误 |
| source trace | `artifacts/v4/training/sources/selection_trace.jsonl` 保存每个 I1 candidate 的 accepted/rejected status |
| prepared source outputs | `symbolic_scenes.jsonl`；`natural_scenes.jsonl`；`natural_observations.jsonl`；`selection_trace.jsonl`；`source_summary.json` |
| support build | `scripts/v4/07_build_support_data.py --execute --prepared-sources` |
| GPU preflight | `scripts/v4/08_train_phase4_lora.py --execute --preflight-only --prepared-support` |
| training command | 设置固定 acknowledgement 后执行 `scripts/v4/08_train_phase4_lora.py --execute --prepared-support` |

### Phase 4 服务器输入与输出记录

| 字段 | 记录值 |
| --- | --- |
| 训练代码基线 | `b12c88d1c0129f9bba5c9e20871d18b3c5c6033b` 之后的 Phase 4 冻结/merger 修订；服务器训练完成画面未显示新的 `git rev-parse HEAD` |
| symbolic scenes | 579 |
| natural single-error scenes | 173 |
| symbolic scenes SHA-256 | `852c4e26c87d0f4afb34737e59ee840a17bf2e0a2a5813f956d3e9bc7e175c80` |
| natural scenes SHA-256 | `b3dacf9515ff85e8e8af27777af28fe1969161ad45ed8400d8fd8a627c5a1418` |
| natural observations SHA-256 | `c7d5d7d67d11f8b8e3a552c73c69ea02cbf3a05085e8bbfd45d78188155f7ee0` |
| selection trace SHA-256 | `5b47ab6d4848f297557ab45d721ee6b2258ba7637f66349aac6741e8322c433f` |
| source summary SHA-256 | `32763a9648d12f841f9231e0a16db15597fac1a50c545f8871696108c41c8322` |
| support corpus SHA-256 | `0bc003d7fdf4f2f67aecef42a25e8caf2f33433fa969aea1962df5ee397a9c5e` |
| support summary SHA-256 | `ac0d3426eb0f9705b2e4af5d7096dc2c6c2e9b0ea681fe17600864d872b2d292` |
| 最后显示的 checkpoint steps | 564/564 |
| 最后显示的 epoch | 1 |
| 最后显示的 train runtime | 1,509 秒；画面进度为 25:08 |
| 最后显示的 train samples/second | 2.991 |
| 最后显示的 train steps/second | 0.374 |
| 最后显示的 train loss | 0.1395 |
| 输出根目录 | `/cloud/cloud-ssd1/dissertation/artifacts/v4/training/runs/phase4-r1` |
| C0 adapter | `C0_format_only/final_adapter`；tree SHA-256=`ff4fdf711b0b80e5effe6362b96f2d11dc4cf92e91523e6799c27d6c89fc194c` |
| C1 adapter | `C1_forward_arithmetic/final_adapter`；tree SHA-256=`0568be719b4077790c86ef1e83a4980eef8d5d5896af0a1aa40d1beefd320d65` |
| T adapter | `T_constraint_recovery/final_adapter`；tree SHA-256=`807a61c2e3f7b532b162554dee6e7df83d654fb1f10cc464e9dcb5f6f8efd5c7` |

### Phase 4 source selection 与 support corpus 计数

| 字段 | 计数或状态 |
| --- | ---: |
| symbolic candidates / accepted | 579 / 579 |
| natural candidates | 579 |
| natural accepted | 173 |
| natural rejected: multiple-error | 1 |
| natural rejected: outside numeric domain | 3 |
| natural rejected: zero-error | 402 |
| support rows total | 6,016 |
| C0 rows | 752 |
| C1 rows | 752 |
| T rows | 4,512 |
| confirmatory data used | false |

### Phase 4 冻结参数与可训练参数 manifest

| component / manifest field | parameter tensors | SHA-256 / value |
| --- | ---: | --- |
| frozen language base | 434 | `20b002cf302a0718d2bbba1e45740e5682188fda95a2d6410e72a5736f3b203f` |
| frozen merger | 5 | `073be98ebb97f62c9a41dcd8589ef5f75404aacc5719df1498b8af27c7bb1371` |
| frozen vision | 385 | `316310eada6f3f6734036c4e6a09dd5627909866302374344409b124dbf3ec69` |
| target modules | 252 | 36 language layers × 7 target leaf modules |
| trainable LoRA parameter tensors | 504 | base language frozen=true；vision frozen=true；merger frozen=true |

| Phase 4 evidence file | SHA-256 |
| --- | --- |
| `artifacts/v4/training/frozen_hashes.json` | `f04189c34e65b4472dee80a4c3c72aea2f82f1681247809603112a7e46d593d8` |
| `artifacts/v4/training/trainable_parameter_manifest.json` | `3c1327efd60379ac89c5b71c4ba9d1e9ddae2a7750a2b5cb435cea356c34ee1a` |
| `C0_format_only/metrics.json` | `5eef36865797076abfbeb20a1a39c0cbfb257b70f3b30b9dcb3e7b78b86ebc1d` |
| `C1_forward_arithmetic/metrics.json` | `a2b97aa8cd54d3164252aa0dac46843b5ffde39def9e2072f66a6f98b6386d1c` |
| `T_constraint_recovery/metrics.json` | `dcde0eb56a6aaf73fb9af60681eebe2af0ffd5ddeebdfb1844c9e43a23ea84fa` |

### Phase 4 三个 LoRA 训练运行的最终指标

| arm | steps | epoch | train loss | runtime seconds | samples/second | steps/second | total FLOPs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C0 | 94 | 1.0 | 0.0029778568487598294 | 250.9792 | 2.996 | 0.375 | 880796539551744 |
| C1 | 94 | 1.0 | 0.008817935858307 | 250.6551 | 3.0 | 0.375 | 2190299912699904 |
| T | 564 | 1.0 | 0.13946974536703619 | 1508.5529 | 2.991 | 0.374 | 12643485874618368 |

### Phase 5--7

| 阶段 | 计划输出路径 | 本报告生成时的执行状态 |
| --- | --- | --- |
| Phase 5 policy support | `artifacts/v4/support/policy_support_by_scene.parquet`；`artifacts/v4/support/informative_group_rate.json`；`artifacts/v4/support/pass_at_k.csv` | 服务器执行完成；正式文件哈希与 summary 已由事实包采集 |
| Phase 6 RL | execution manifest、RL data、3 个 GRPO run roots、5-checkpoint evaluation | 服务器训练与评估完成；正式文件哈希与 summary/evidence 已由事实包采集 |
| Phase 7 real multimodal | `artifacts/v4/phase7/evaluation/per_scene.jsonl`；`summary.json`；`artifacts/v4/phase7/interface_audit.json` | support-dev diagnostic 与 interface audit 均在服务器执行完成；confirmatory 未授权 |

### Phase 5：policy-support 服务器执行记录

| 字段 | 记录值 |
| --- | --- |
| status | `PHASE_5_POLICY_SUPPORT_EXECUTED` |
| schema version | 1 |
| number of held-out natural errors | 32 |
| held-out family counts | `cross_series=16`；`duplicate_encoding=5`；`trend=11` |
| number of checkpoint scene rows | 128 |
| sampling rollouts per scene | 16 |
| sampling temperature | 0.7 |
| sampling seed | `2026082005` |
| pass@K grid | `1, 2, 4, 8, 16` |
| informative group size | 8 |
| confirmatory data used | false |
| scene is statistical unit | true |
| subjective success threshold applied | false |
| training invoked | false |
| RL invoked | false |
| model snapshot SHA-256 | `e104df572eab7267bc2a63c11d70f7c8b1ebf8f85aa835d17e2c2641447bca87` |
| package lock SHA-256 | `e3fb68f121bfebfc238d896cd907561455a361d1292f38df965ad23f1f5bc152` |
| support-dev summary SHA-256 | `f9dd52a363a97a961c8eb55fd69f305507a5b5ba0ad171aae1996642f330a9cf` |
| Base source SHA-256 | `e104df572eab7267bc2a63c11d70f7c8b1ebf8f85aa835d17e2c2641447bca87` |
| C0 adapter tree SHA-256 | `ff4fdf711b0b80e5effe6362b96f2d11dc4cf92e91523e6799c27d6c89fc194c` |
| C1 adapter tree SHA-256 | `0568be719b4077790c86ef1e83a4980eef8d5d5896af0a1aa40d1beefd320d65` |
| T adapter tree SHA-256 | `807a61c2e3f7b532b162554dee6e7df83d654fb1f10cc464e9dcb5f6f8efd5c7` |
| support-dev source SHA-256 | `cafbd5e5a0face1622d62faac4bb6b9bfc1fa6cc8d4353ade82b9144d8b18136` |
| config SHA-256 | `b06d242cf7a683f8c7b0ce14843b165035dfad2b1348e580fef22da135ae664a` |

Phase 5 formal output SHA-256：`policy_support_by_scene.parquet=196896097b5a261733529a903e1ea8cc1ee1cbeffeda397a233b7044e01c77b7`；`informative_group_rate.json=2cdcb288986a4769aaa6caf477e51595a7648788cbd5471331b12fb049ffd358`；`pass_at_k.csv=4de8876cc16b723c888e0a8db5a131978eb175ba54624735e69bc1a4a739ec49`。

Phase 5 support-dev 冻结记录：576 candidates；32 natural errors；family counts `cross_series=16`、`duplicate_encoding=5`、`trend=11`；excluded correct=544；included errors=32。`candidates.jsonl` SHA-256=`33150ff9edcb89246468af688126520075dd127095b7866ca32a1dbcb9aeabbb`；`errors.jsonl` SHA-256=`cafbd5e5a0face1622d62faac4bb6b9bfc1fa6cc8d4353ade82b9144d8b18136`；`selection_trace.jsonl` SHA-256=`cd3e05daff2f16d462dee92805c15290523bad9fa3194f092089038db09ac5ce`；`summary.json` SHA-256=`f9dd52a363a97a961c8eb55fd69f305507a5b5ba0ad171aae1996642f330a9cf`。

### Phase 5 checkpoint 级结果

| checkpoint | greedy success | greedy copy | mean rollout success probability | informative-group rate | sampled copy rate | true-minus-observed logit margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 0.0 | 0.8125 | 0.021484375 | 0.0940495384420501 | 0.546875 | -6.478977203369141 |
| C0 | 0.0 | 0.96875 | 0.001953125 | 0.012602516435435973 | 0.84375 | -7.938401132822037 |
| C1 | 0.0 | 1.0 | 0.0 | 0.0 | 0.9296875 | -8.159707255661488 |
| T | 0.90625 | 0.0 | 0.8671875 | 0.14764408432529308 | 0.0 | 6.697406638413668 |

| checkpoint | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 0.021484375 | 0.0386962890625 | 0.06405496597290039 | 0.09405049213819439 | 0.12224163803529327 |
| C0 | 0.001953125 | 0.0037841796875 | 0.007110118865966797 | 0.01260251644271193 | 0.020122683423381475 |
| C1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| T | 0.8671875 | 0.893798828125 | 0.9030656814575195 | 0.9059367423906224 | 0.9062468608616231 |

### Phase 5 mean rollout success 与 sampled copy 按 family

| checkpoint | family | mean success probability | sampled copy rate |
| --- | --- | ---: | ---: |
| Base | `cross_series` | 0.0 | 0.75390625 |
| Base | `duplicate_encoding` | 0.125 | 0.45 |
| Base | `trend` | 0.005681818181818182 | 0.2897727272727273 |
| C0 | `cross_series` | 0.0 | 0.97265625 |
| C0 | `duplicate_encoding` | 0.0125 | 0.8625 |
| C0 | `trend` | 0.0 | 0.6477272727272727 |
| C1 | `cross_series` | 0.0 | 0.9921875 |
| C1 | `duplicate_encoding` | 0.0 | 0.9875 |
| C1 | `trend` | 0.0 | 0.8125 |
| T | `cross_series` | 0.94140625 | 0.0 |
| T | `duplicate_encoding` | 1.0 | 0.0 |
| T | `trend` | 0.6988636363636364 | 0.0 |

### Phase 6：冻结执行参数与服务器结果

| 字段 | 记录值 |
| --- | --- |
| 当前状态 | 服务器执行完成；evaluation status=`PHASE_6_RL_EVALUATED` |
| 比较组 | `Base`；`Base_AnswerOnly_RL`；`Recovery_LoRA`；`Recovery_LoRA_RecoveryOutcome_RL`；`Recovery_LoRA_AnswerOnly_RL` |
| 实际 GRPO 训练组 | `Base_AnswerOnly_RL`；`Recovery_LoRA_RecoveryOutcome_RL`；`Recovery_LoRA_AnswerOnly_RL` |
| Base initialization | 模型快照 SHA-256 `e104df572eab7267bc2a63c11d70f7c8b1ebf8f85aa835d17e2c2641447bca87` |
| Recovery initialization | Phase 4 T adapter tree SHA-256 `807a61c2e3f7b532b162554dee6e7df83d654fb1f10cc464e9dcb5f6f8efd5c7` |
| reward modes | `answer_only`；`recovery_outcome` |
| precision | `bf16` |
| learning rate | `0.000001` |
| max steps per GRPO arm | 64 |
| rollout group size | 8 |
| temperature | 0.7 |
| top-p | 1.0 |
| top-k | 0 |
| KL beta | 0.04 |
| per-device train batch size | 1 |
| gradient accumulation steps | 8 |
| maximum prompt length | 512 |
| maximum completion length | 32 |
| checkpoint interval | 16 steps |
| training seed | `2026082006` |
| evaluation scenes | frozen support-dev 32 natural errors |
| evaluation rollouts per scene | 16 |
| evaluation seed | `2026082005` |
| scene-clustered bootstrap resamples | 10,000 |
| bootstrap seed | `2026082007` |
| confidence interval | 95% |
| registered-effect multiplicity | paired sign-flip p-values with Holm adjustment |
| training seed count | 1；seed=`2026082006` |
| confirmatory evaluation authorized | false |
| downloads authorized | false |
| execution-manifest output | `artifacts/v4/phase6/execution_manifest.json` |
| RL data outputs | `artifacts/v4/rl/data/recovery_outcome.jsonl`；`answer_only.jsonl`；`summary.json` |
| GRPO run root | `artifacts/v4/rl/runs/phase6-r1` |
| formal evaluation outputs | `artifacts/v4/rl/evaluation/by_scene.jsonl`；`summary.json` |
| execution manifest inputs | `artifacts/v4/support/informative_group_rate.json`；Phase 4 `C0/C1/T` adapter trees |
| RL data gate | 当前 Phase 5 summary SHA-256 与 `source_sha256` 必须同时匹配 execution manifest |
| GRPO preflight gate | Base snapshot 与 Phase 4 `C0/C1/T` adapter tree SHA-256 必须匹配 execution manifest |
| evaluation cache gate | checkpoint hash、support-dev input hash、config hash、package-lock hash、execution-manifest hash 必须同时匹配 |

### Phase 6：服务器数据、训练信号与注册效应

| 字段 | 记录值 |
| --- | --- |
| execution manifest | `artifacts/v4/phase6/execution_manifest.1759bc7.json` |
| execution manifest SHA-256 | `47b9f10638dae1957d03beae0a227cdbc48c922f76554bd69ce02b86a6fd73ef` |
| later revision-tagged execution manifest | `artifacts/v4/phase6/execution_manifest.7c44b58.json` |
| later revision-tagged execution manifest SHA-256 | `af8acece46c67bd593eed778289012f51ce46fe668a927d718ee5774897ec9d3` |
| untagged execution manifest SHA-256 in fact bundle | `f8d5b2a71bd06833ea803dbbf916fe56ea220bdaef97291bd70f5f2f480a644e` |
| recovery-outcome RL data count | 173 |
| answer-only RL data count | 173 |
| recovery-outcome data SHA-256 | `7af9f17c1febd3e0bfa2cddb0d6f5cde35500c12405952f9ad071e278b4b1993` |
| answer-only data SHA-256 | `3e72aa47c74f8e02191d949f968612d54bca9a81bfeb18f3cebd5d2394680d8e` |
| RL data summary SHA-256 | `337a32fac096c6034f29f96a2a1e7f7217474511839b9b8de2cc3764e856d171` |
| evaluation status | `PHASE_6_RL_EVALUATED` |
| evaluation scenes | 32；`cross_series=16`、`duplicate_encoding=5`、`trend=11` |
| training seeds | `[2026082006]` |
| scene statistical unit | true |
| subjective success threshold applied | false |
| formal evaluation by-scene SHA-256 | `6dbf0d2900628227228b59bd7f7b0cfad6d907664e23d3a14c802cc193a35eb8` |
| formal evaluation summary SHA-256 | `cfcd8aa5882ea9719f0acaad1da5a9b0f5776739f1140b35b59c6eeafbf3a9d9` |
| evaluation-bound execution manifest SHA-256 | `af8acece46c67bd593eed778289012f51ce46fe668a927d718ee5774897ec9d3` |
| evaluation config SHA-256 | `13a1f6adfc89b17c6883fd7bf727668b31729a66b5cd89f82d7b9f5c450e4648` |
| evaluation package-lock SHA-256 | `6b93547144d9268126206827a4d6c1b466fb816bbc5e1e8ae1bf649d333cd5ad` |

### Phase 6 五个 checkpoint 的全局评估率

每个 checkpoint 均为 32 scenes、每 scene 16 sampling rollouts。

| checkpoint | greedy answer exact [95% CI] | greedy recovery exact [95% CI] | greedy observation copy [95% CI] | sampled recovery exact [95% CI] | sampled observation copy rate |
| --- | --- | --- | --- | --- | --- |
| Base | 0.1875 [0.0625, 0.34375] | 0.0 [0.0, 0.0] | 0.8125 [0.65625, 0.9375] | 0.021484375 [0.00390625, 0.044921875] | 0.546875 |
| `Base_AnswerOnly_RL` | 0.1875 [0.0625, 0.34375] | 0.0 [0.0, 0.0] | 0.84375 [0.71875, 0.96875] | 0.025390625 [0.00390625, 0.05078125] | 0.53515625 |
| `Recovery_LoRA` | 0.28125 [0.125, 0.4375] | 0.90625 [0.78125, 1.0] | 0.0 [0.0, 0.0] | 0.8671875 [0.755859375, 0.95703125] | 0.0 |
| `Recovery_LoRA_AnswerOnly_RL` | 0.28125 [0.125, 0.4375] | 0.90625 [0.78125, 1.0] | 0.0 [0.0, 0.0] | 0.8671875 [0.751953125, 0.958984375] | 0.001953125 |
| `Recovery_LoRA_RecoveryOutcome_RL` | 0.28125 [0.125, 0.4375] | 0.90625 [0.8125, 1.0] | 0.0 [0.0, 0.0] | 0.8671875 [0.75390625, 0.95703125] | 0.0 |

### Phase 6 checkpoint 指标按 family

| checkpoint | family | scenes | answer exact rate | greedy exact-world recovery rate |
| --- | --- | ---: | ---: | ---: |
| Base | cross_series | 16 | 0.125 | 0.0 |
| Base | duplicate_encoding | 5 | 0.0 | 0.0 |
| Base | trend | 11 | 0.36363636363636365 | 0.0 |
| Base_AnswerOnly_RL | cross_series | 16 | 0.125 | 0.0 |
| Base_AnswerOnly_RL | duplicate_encoding | 5 | 0.0 | 0.0 |
| Base_AnswerOnly_RL | trend | 11 | 0.36363636363636365 | 0.0 |
| Recovery_LoRA | cross_series | 16 | 0.1875 | 1.0 |
| Recovery_LoRA | duplicate_encoding | 5 | 0.4 | 1.0 |
| Recovery_LoRA | trend | 11 | 0.36363636363636365 | 0.7272727272727273 |
| Recovery_LoRA_AnswerOnly_RL | cross_series | 16 | 0.1875 | 1.0 |
| Recovery_LoRA_AnswerOnly_RL | duplicate_encoding | 5 | 0.4 | 1.0 |
| Recovery_LoRA_AnswerOnly_RL | trend | 11 | 0.36363636363636365 | 0.7272727272727273 |
| Recovery_LoRA_RecoveryOutcome_RL | cross_series | 16 | 0.1875 | 1.0 |
| Recovery_LoRA_RecoveryOutcome_RL | duplicate_encoding | 5 | 0.4 | 1.0 |
| Recovery_LoRA_RecoveryOutcome_RL | trend | 11 | 0.36363636363636365 | 0.7272727272727273 |

三个 GRPO 训练分支的服务器诊断记录：

| variant | adapter tree SHA-256 | groups | all-zero rate | all-one rate | non-degenerate rate | mean reward variance | mean KL | mean entropy | exact recovery rate | copy rate | train loss | runtime seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Base_AnswerOnly_RL` | `e1189e577b48a3f6cfe0d69c7e3b3c9d8e1e5e135c5dd1fec503f55a357c6b18` | 64 | 0.921875 | 0.0 | 0.078125 | 0.009765625 | 0.012808018779882304 | 0.8926465229855644 | 0.0 | 0.0 | 0.02018016355589032 | 313.9627 |
| `Recovery_LoRA_RecoveryOutcome_RL` | `4d628eeb84ece36f992d684324f757789adc834e0605b5af99256c50968b6b2f` | 64 | 0.265625 | 0.21875 | 0.515625 | 0.089599609375 | 0.03233168982791222 | 0.2670032273172327 | 0.474609375 | 0.248046875 | 0.018073564395308495 | 271.4134 |
| `Recovery_LoRA_AnswerOnly_RL` | `5d011bc40cba39c94b6593587f9b2b76d45480eeb397a9fa86c83baa0eb3a515` | 64 | 0.875 | 0.0 | 0.125 | 0.01611328125 | 0.011673480690704414 | 0.9363073353611288 | 0.0 | 0.0 | 0.0022307709550952736 | 275.4734 |

与各自初始 checkpoint 的 32-scene greedy 输出差异计数：

| 比较 | recovery token sequence changed | answer token sequence changed |
| --- | ---: | ---: |
| `Base -> Base_AnswerOnly_RL` | 2/32 | 0/32 |
| `Recovery_LoRA -> Recovery_LoRA_RecoveryOutcome_RL` | 0/32 | 0/32 |
| `Recovery_LoRA -> Recovery_LoRA_AnswerOnly_RL` | 0/32 | 0/32 |

全局四个注册效应的服务器输出均记录：estimate=`0.0`、95% CI=`[0.0, 0.0]`、two-sided sign-flip p-value=`1.0`、Holm-adjusted p-value=`1.0`。`cross_series`（16 scenes）、`duplicate_encoding`（5 scenes）与 `trend`（11 scenes）分层对同四项效应也均记录 estimate=`0.0`、95% CI=`[0.0, 0.0]`、p-value=`1.0`；family-level 对象未包含 Holm 字段。

### Phase 7：support-dev multimodal diagnostic 与 interface audit 服务器记录

| 字段 | 冻结值或状态 |
| --- | --- |
| config | `configs/recoverability/v4_phase_7.yaml` |
| authorization status | `PHASE_7_MULTIMODAL_DIAGNOSTIC_AUTHORIZED` |
| multimodal evaluation status | `PHASE_7_MULTIMODAL_DIAGNOSTIC_EVALUATED` |
| interface audit status | `PHASE_7_INTERFACE_DIAGNOSTIC_AUDITED` |
| confirmatory evaluation authorized | false |
| support-dev diagnostic authorized | true |
| confirmatory data used | false |
| training / RL / downloads authorized | false / false / false |
| evaluation scenes | 32 个 Phase 5 frozen support-dev natural errors |
| checkpoints | `Base`、`C0`、`C1`、`T`、`Base_AnswerOnly_RL`、`Recovery_LoRA_RecoveryOutcome_RL`、`Recovery_LoRA_AnswerOnly_RL` |
| chain | image -> natural observation -> revision/recovery -> chart operation -> final answer |
| deterministic generation calls | 32 × 7 × 4 = 896 |
| multimodal / audit rows | 224 / 224 |
| image resize | 280 × 280 |
| max new tokens | Stage-1=32；recovery=32；operation=8；answer=8 |
| generation seed | `2026082102` |
| bootstrap | 10,000 scene-clustered resamples；seed=`2026082101` |
| main confidence intervals | 95% |
| equivalence interval | 90% scene-clustered percentile bootstrap interval；margin=`0.02` |
| multiplicity | two-sided paired sign-flip p-values；Holm adjustment |
| subjective success threshold | null |
| formal output paths | `artifacts/v4/phase7/evaluation/per_scene.jsonl`；`summary.json`；`artifacts/v4/phase7/interface_audit.json` |
| interface audit SHA-256 | `abbd06e6d1f76bbd549009f4993a748d6be5feac2d8b8f1a899aea438cea004b` |
| Phase 7 execution-manifest SHA-256 | `1fb6e640350c37dc966cbd76a6e8b6d7c388ebc3e2c55a0b0b8fd3e6dacb9010` |
| Phase 6 evaluation source SHA-256 | `cfcd8aa5882ea9719f0acaad1da5a9b0f5776739f1140b35b59c6eeafbf3a9d9` |
| Phase 7 per-scene SHA-256 | `487af12e77f8d2592046b0a6bdc681a14953fcd382902b716fc0a8563c3f602e` |
| Phase 7 summary SHA-256 | `fde66bae90f58eb64de5d6cb32e0a0f38f866c439775b17e7de64a31987d86db` |
| Phase 7 config SHA-256 | `f1baf59c0cced81300c88639a7b09f78f15fbc78f05d810cbcdf12fdacbaa4d9` |
| Phase 7 package-lock SHA-256 | `47c2282ca4356f9b4a9023a696ac6e64c3dd2c43994430b1f3779da14f0544ca` |

九个冻结指标：`stage1_visual_exact`、`post_revision_world_exact`、`reasoning_operator_exact`、`final_answer_exact`、`operator_invariant_correct`、`genuine_recovery`、`error_cancellation`、`trace_mismatch`、`error_mechanism_shift`。

四个注册比较：`T_minus_C0`、`T_minus_C1`、`seeded_rl_minus_base_rl`、`recovery_reward_rl_minus_answer_only_rl`。当前 support-dev diagnostic 的 `ood_axis` 为 `iid`；`style_ood`、`constraint_graph_ood`、`error_mechanism_ood` 已注册但未由该服务器命令测量。

服务器 summary 的全局 224-row 指标：

| metric | estimate | 95% CI | rows | scenes |
| --- | ---: | --- | ---: | ---: |
| `error_cancellation` | 0.008928571428571428 | [0.0, 0.026785714285714284] | 224 | 32 |
| `error_mechanism_shift` | 0.24107142857142858 | [0.15625, 0.33035714285714285] | 224 | 32 |
| `final_answer_exact` | 0.11160714285714285 | [0.049107142857142856, 0.18303571428571427] | 224 | 32 |
| `genuine_recovery` | 0.10714285714285714 | [0.040178571428571425, 0.17410714285714285] | 224 | 32 |
| `operator_invariant_correct` | 0.05357142857142857 | [0.013392857142857142, 0.10267857142857142] | 224 | 32 |
| `stage1_visual_exact` | 0.24107142857142855 | [0.15625, 0.33482142857142855] | 224 | 32 |
| `post_revision_world_exact` | 0.24553571428571427 | [0.16964285714285712, 0.32589285714285715] | 224 | 32 |
| `reasoning_operator_exact` | 1.0 | [1.0, 1.0] | 224 | 32 |
| `trace_mismatch` | 0.8348214285714286 | [0.7544642857142857, 0.90625] | 224 | 32 |

### Phase 7 按 checkpoint 的九项指标

各单元格式为 `estimate [95% CI]`；每个 checkpoint 均为 32 scenes / 32 rows。

| checkpoint | stage1 exact | post-world exact | operator exact | final answer exact | genuine recovery | operator-invariant correct | error cancellation | mechanism shift | trace mismatch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Base | 0 [0, 0] | 0 [0, 0] | 1 [1, 1] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 1 [1, 1] |
| `Base_AnswerOnly_RL` | 0.03125 [0, 0.09375] | 0.03125 [0, 0.09375] | 1 [1, 1] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0.03125 [0, 0.09375] | 1 [1, 1] |
| C0 | 0.09375 [0, 0.21875] | 0.09375 [0, 0.1875] | 1 [1, 1] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0.09375 [0, 0.21875] | 1 [1, 1] |
| C1 | 0.125 [0.03125, 0.25] | 0.125 [0.03125, 0.25] | 1 [1, 1] | 0.09375 [0, 0.21875] | 0 [0, 0] | 0.0625 [0, 0.15625] | 0.03125 [0, 0.09375] | 0.125 [0.03125, 0.25] | 0.8125 [0.65625, 0.9375] |
| `Recovery_LoRA_AnswerOnly_RL` | 0.5 [0.34375, 0.6875] | 0.5 [0.34375, 0.6875] | 1 [1, 1] | 0.25 [0.09375, 0.40625] | 0.25 [0.09375, 0.40625] | 0.125 [0.03125, 0.25] | 0 [0, 0] | 0.5 [0.3125, 0.65625] | 0.65625 [0.5, 0.8125] |
| `Recovery_LoRA_RecoveryOutcome_RL` | 0.5 [0.34375, 0.6875] | 0.5 [0.34375, 0.6875] | 1 [1, 1] | 0.21875 [0.09375, 0.375] | 0.25 [0.09375, 0.40625] | 0.09375 [0, 0.21875] | 0 [0, 0] | 0.5 [0.3125, 0.65625] | 0.65625 [0.5, 0.8125] |
| T | 0.4375 [0.28125, 0.625] | 0.46875 [0.3125, 0.65625] | 1 [1, 1] | 0.21875 [0.09375, 0.375] | 0.25 [0.09375, 0.40625] | 0.09375 [0, 0.21875] | 0.03125 [0, 0.09375] | 0.4375 [0.28125, 0.59375] | 0.71875 [0.5625, 0.875] |

服务器自由生成注册效应：

| effect | estimate | 95% CI | sign-flip p | Holm p | 90% TOST CI | equivalent at margin 0.02 |
| --- | ---: | --- | ---: | ---: | --- | --- |
| `T_minus_C0` | 0.21875 | [0.09375, 0.375] | 0.016998300169983 | 0.067993200679932 | [0.09375, 0.34375] | false |
| `T_minus_C1` | 0.125 | [0.0, 0.28125] | 0.21357864213578642 | 0.6407359264073593 | [0.0, 0.25] | false |
| `recovery_reward_rl_minus_answer_only_rl` | 0.0 | [0.0, 0.0] | 1.0 | 1.0 | [0.0, 0.0] | true |
| `seeded_rl_minus_base_rl` | 0.03125 | [0.0, 0.09375] | 1.0 | 1.0 | [0.0, 0.09375] | false |

### Phase 7 注册效应按 family 与 OOD axis

| stratum | scenes | effect | estimate | 95% CI | sign-flip p | Holm p |
| --- | ---: | --- | ---: | --- | ---: | ---: |
| `cross_series` | 16 | `T_minus_C0` | 0.0625 | [0.0, 0.1875] | 1.0 | 1.0 |
| `cross_series` | 16 | `T_minus_C1` | 0.0 | [-0.1875, 0.1875] | 1.0 | 1.0 |
| `cross_series` | 16 | `recovery_reward_rl_minus_answer_only_rl` | 0.0 | [0.0, 0.0] | 1.0 | 1.0 |
| `cross_series` | 16 | `seeded_rl_minus_base_rl` | 0.0 | [0.0, 0.0] | 1.0 | 1.0 |
| `duplicate_encoding` | 5 | `T_minus_C0` | 0.2 | [0.0, 0.6] | 1.0 | 1.0 |
| `duplicate_encoding` | 5 | `T_minus_C1` | 0.2 | [0.0, 0.6] | 1.0 | 1.0 |
| `duplicate_encoding` | 5 | `recovery_reward_rl_minus_answer_only_rl` | 0.0 | [0.0, 0.0] | 1.0 | 1.0 |
| `duplicate_encoding` | 5 | `seeded_rl_minus_base_rl` | 0.0 | [0.0, 0.0] | 1.0 | 1.0 |
| `trend` | 11 | `T_minus_C0` | 0.45454545454545453 | [0.18181818181818182, 0.7272727272727273] | 0.0625 | 0.25 |
| `trend` | 11 | `T_minus_C1` | 0.2727272727272727 | [0.0, 0.5454545454545454] | 0.25 | 0.75 |
| `trend` | 11 | `recovery_reward_rl_minus_answer_only_rl` | 0.0 | [0.0, 0.0] | 1.0 | 1.0 |
| `trend` | 11 | `seeded_rl_minus_base_rl` | 0.09090909090909091 | [0.0, 0.2727272727272727] | 1.0 | 1.0 |
| `iid` | 32 | `T_minus_C0` | 0.21875 | [0.09375, 0.375] | 0.014198580141985802 | 0.05679432056794321 |
| `iid` | 32 | `T_minus_C1` | 0.125 | [0.0, 0.28125] | 0.21637836216378362 | 0.6491350864913509 |
| `iid` | 32 | `recovery_reward_rl_minus_answer_only_rl` | 0.0 | [0.0, 0.0] | 1.0 | 1.0 |
| `iid` | 32 | `seeded_rl_minus_base_rl` | 0.03125 | [0.0, 0.09375] | 1.0 | 1.0 |

`seed_level_variability` 仅包含 seed `2026082102`；其九项全局估计与上述 global metrics 相同。

冻结 trace 的 parser 与答案计数：

| checkpoint | answer parse | free-generation answer exact | post-revision world exact | trace mismatch |
| --- | ---: | ---: | ---: | ---: |
| `Base` | 0/32 | 0/32 | 0/32 | 32/32 |
| `Base_AnswerOnly_RL` | 0/32 | 0/32 | 1/32 | 32/32 |
| `C0` | 0/32 | 0/32 | 3/32 | 32/32 |
| `C1` | 32/32 | 3/32 | 4/32 | 26/32 |
| `Recovery_LoRA_AnswerOnly_RL` | 32/32 | 8/32 | 16/32 | 21/32 |
| `Recovery_LoRA_RecoveryOutcome_RL` | 32/32 | 7/32 | 16/32 | 21/32 |
| `T` | 32/32 | 7/32 | 15/32 | 23/32 |

服务器 `interface_audit.json` 的 checkpoint 计数和率：

| checkpoint | parse | free exact | deterministic chain exact | parsed trace consistent |
| --- | ---: | ---: | ---: | ---: |
| `Base` | 0/32 (0.0) | 0/32 (0.0) | 7/32 (0.21875) | 0/32 (0.0) |
| `Base_AnswerOnly_RL` | 0/32 (0.0) | 0/32 (0.0) | 7/32 (0.21875) | 0/32 (0.0) |
| `C0` | 0/32 (0.0) | 0/32 (0.0) | 8/32 (0.25) | 0/32 (0.0) |
| `C1` | 32/32 (1.0) | 3/32 (0.09375) | 11/32 (0.34375) | 6/32 (0.1875) |
| `Recovery_LoRA_AnswerOnly_RL` | 32/32 (1.0) | 8/32 (0.25) | 24/32 (0.75) | 11/32 (0.34375) |
| `Recovery_LoRA_RecoveryOutcome_RL` | 32/32 (1.0) | 7/32 (0.21875) | 23/32 (0.71875) | 11/32 (0.34375) |
| `T` | 32/32 (1.0) | 7/32 (0.21875) | 22/32 (0.6875) | 9/32 (0.28125) |

服务器 deterministic-chain 配对效应：

| effect | executor estimate | 95% CI | free-generation estimate | interface contribution | p | Holm p | 90% TOST CI | equivalent |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| `T_minus_C0` | 0.4375 | [0.1875, 0.65625] | 0.21875 | -0.21875 | 0.0021997800219978004 | 0.004399560043995601 | [0.25, 0.625] | false |
| `T_minus_C1` | 0.34375 | [0.125, 0.5625] | 0.125 | -0.21875 | 0.0156984301569843 | 0.0156984301569843 | [0.15625, 0.53125] | false |

Base、Base-AnswerOnly-RL 与 C0 的 `final_answer_raw` 频率表中记录了以下非整数开头：`To apply the "max_minus_min"`、`To apply the "difference" chart operation`、`The sum of the recovered values (`、`The chart operation "difference" subtracts`、`The recovered values are`。上述三个 checkpoint 的 `final_answer_parse_success` 均为 0/32；interface audit 记录 `post_hoc_parser_relaxation_applied=false` 与 `free_generation_evidence_preserved=true`。

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
| Phase 4 source preparation | `python scripts/v4/07_prepare_phase4_support_sources.py --execute` | 是 | 默认读取已完成 S6 与冻结视觉数据集 |
| Phase 4 support build | `python scripts/v4/07_build_support_data.py --execute --prepared-sources` | 是 | 读取 3 个 prepared JSONL 与 source summary |
| Phase 4 GPU preflight | `python scripts/v4/08_train_phase4_lora.py --execute --preflight-only --prepared-support` | 是 | 读取 support corpus 与 summary |
| Phase 4 LoRA training | `python scripts/v4/08_train_phase4_lora.py --execute --prepared-support` | 是；另需固定 ACK 环境变量 | 读取 support corpus 与 summary |
| Phase 5 support-dev freeze | `python scripts/v4/09_prepare_phase5_support_dev.py --execute` | 是 | 默认读取冻结 8,000-scene 视觉数据和 Phase 4 selection trace/summary |
| Phase 5 policy-support measurement | `python scripts/v4/10_measure_policy_support.py --execute` | 是 | 读取冻结 support-dev pool 和 C0/C1/T adapter trees |
| Phase 6 execution-manifest freeze | `python scripts/v4/11_prepare_phase6_rl.py --execute` | 是 | 读取 Phase 5 `informative_group_rate.json` 与 Phase 4 adapter trees |
| Phase 6 RL data freeze | `python scripts/v4/12_prepare_phase6_rl_data.py --execute` | 是 | 读取 execution manifest、Phase 4 natural sources、Phase-C records 与 Phase 5 formal artifacts |
| Phase 6 GRPO preflight/training | `python scripts/v4/13_train_phase6_grpo.py --execute` | 是；真实训练另需固定 ACK 环境变量 | 读取 execution manifest、RL data、Base model 与 Phase 4 T adapter |
| Phase 6 formal evaluation | `python scripts/v4/14_evaluate_phase6_rl.py --execute` | 是 | 读取 execution manifest、3 个 GRPO adapters、Base、T 与 frozen support-dev |
| Phase 7 manifest freeze | `python scripts/v4/15_prepare_phase7_multimodal.py --execute` | 是 | 5 个命名且逐项 SHA-256 绑定的 Phase 4--6 / dataset / support-dev 输入 |
| Phase 7 seven-checkpoint preflight/evaluation | `python scripts/v4/16_evaluate_phase7_multimodal.py --execute` | 是；preflight 可加 `--preflight-only` | 读取 Phase 7 execution manifest、7 个 checkpoint、32-scene image bundle 与 Stage-1 prompt |
| Phase 7 frozen-trace interface audit | `python scripts/v4/17_audit_phase7_interface.py --execute` | 是 | 读取 Phase 7 summary 与 7 个 checkpoint trace JSONL；不加载模型且不执行新推理 |
| Phase 8 confirm 数据冻结 | `python scripts/v4/18_freeze_phase8_confirm_data.py --execute` | 是；另需固定 confirm ACK 环境变量 | 5 个命名且逐项 SHA-256 绑定的 legacy/train/support-dev/Phase 7 输入；执行 128 次 Base Stage-1 调用 |
| Phase 8 seven-checkpoint preflight/evaluation | `python scripts/v4/19_evaluate_phase8_confirmatory.py --execute` | 是；preflight 可加 `--preflight-only`；另需固定 confirm ACK | 读取 Phase 8 execution manifest、冻结自然错误、图片 bundle、Stage-1 prompt 与 7 个 checkpoint |

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

## Phase 8：独立最终确认评估

| 字段 | 冻结值 |
| --- | --- |
| 状态 | `PHASE_8_CONFIRMATORY_EVALUATED` |
| confirm data freeze status | `PHASE_8_CONFIRM_DATA_FROZEN` |
| checkpoint | `Base`、`C0`、`C1`、`T`、`Base_AnswerOnly_RL`、`Recovery_LoRA_RecoveryOutcome_RL`、`Recovery_LoRA_AnswerOnly_RL` |
| confirm axes | `iid`、`style_ood`、`constraint_graph_ood`、`error_mechanism_ood` |
| 每轴固定候选数 | 32 |
| Base Stage-1 固定候选调用总数 | 128 |
| 纳入的自然 Stage-1 error scenes | 98 |
| 正式评估行数 | 686（98 scenes × 7 checkpoints） |
| generation seed | 2026082103 |
| evaluation seed | 2026082104 |
| bootstrap seed | 2026082105 |
| bootstrap resamples | 10,000 |
| bootstrap confidence | 0.95 |
| TOST margin | 0.02 |
| 统计单位 | scene |
| 经验成功阈值 | 未设置 |
| 训练 / RL | 不执行 |
| 正式输出覆盖 | 拒绝 |
| confirmatory data used | true |
| statistical unit | scene |
| frozen candidate counts | `iid=32`；`style_ood=32`；`constraint_graph_ood=32`；`error_mechanism_ood=32` |
| selected natural errors by family | `cross_series=34`；`duplicate_encoding=34`；`trend=30` |
| selected natural errors by OOD axis | `iid=31`；`style_ood=28`；`constraint_graph_ood=27`；`error_mechanism_ood=12` |
| all natural Stage-1 errors included | true |
| selection uses model-outcome threshold | false |
| config SHA-256 | `548b1ef76ef7dfc7f7ec523dcbd6793bd41db69886d6d12fe02d5fbc176ea254` |
| package-lock SHA-256 | `e376828b49b21d6e45b9864f105ece4b8f65fecb6f5be7ca0e8d683ab31cb283` |

Phase 8 数据冻结在任何 Base Stage-1 结果产生前固定四轴各 32 个候选。四轴使用新的 semantic scene ID、numeric table ID 与 constraint graph ID，并对 legacy diagnostic、`symbolic_support_train`、`natural_error_support_train` 与 `support_dev` 执行隔离检查。Base Stage-1 输出不可解析时，数据冻结返回 `BLOCKED`；不会以 ground truth 代替。固定候选中的全部可解析自然 Stage-1 error 都进入评估，不按错误数量或后续模型结果筛选。

正式结果同时保存 free-generation answer endpoint 与 deterministic-chain answer endpoint。每个 checkpoint 的正式行包含 9 个已注册指标、`answer_source`、family、OOD axis、checkpoint SHA-256、image SHA-256 与 seed。summary 保存 global/checkpoint/family/OOD 分层、配对效应、scene-clustered bootstrap interval、sign-flip p-value、Holm correction、TOST 与 seed variability。

Phase 8 所需固定确认字符串为：

```text
COMPBIAS_V4_PHASE8_CONFIRM_ACK=I_UNDERSTAND_THIS_CONSUMES_THE_FROZEN_PHASE_8_CONFIRM_SET
```

### Phase 8 已记录 source SHA-256

| source | SHA-256 |
| --- | --- |
| confirm image bundle | `df38cb939b906d4690d96dad2b6f3e27fcfe720e93497942d85afca37600779b` |
| confirm observations | `7f0a91ca81d9764581b47c3d0ef94632812adee6aae14d5918c63e10cd41c2e2` |
| confirm scenes | `7432bb9f3b0ceda7364efdda1e13594a71595eed62312062b12b5670cb3d9946` |
| confirm summary | `3ac1c430149f464d15693da42f1c70fbc17fbb4f385f937049432f1016a38ca7` |
| execution manifest | `52fe08b3b5db5e99327302f0e4e5f86e4f403684b136d275d179eeda1c52c776` |
| legacy diagnostic | `36e09f7e15107057fd1b942875d12259b1f281e0354b87c82ed17f420693c766` |
| natural-error support train | `b3dacf9515ff85e8e8af27777af28fe1969161ad45ed8400d8fd8a627c5a1418` |
| Phase 7 evaluation | `fde66bae90f58eb64de5d6cb32e0a0f38f866c439775b17e7de64a31987d86db` |
| prompt config | `c4fa65062527b76bd0b29bbad6a5bd35e596fcc2bfef4dd0c81d7fe008610d10` |
| support dev | `cafbd5e5a0face1622d62faac4bb6b9bfc1fa6cc8d4353ade82b9144d8b18136` |
| symbolic-support train | `852c4e26c87d0f4afb34737e59ee840a17bf2e0a2a5813f956d3e9bc7e175c80` |
| selection trace | `650364dacf62d3fde9b2d8e11a9606ea575470e4f4fd8bb7f1ab78ebbd2cc1e3` |
| evaluation per-scene | `f897cc66f2c9162b3d034d387ec6a09e84939357eedc1f860af18d12747e8fe8` |
| evaluation summary | `a46bd1a494a1468cbd7f62878474e7fc877a10f88a09cc6e88a611f1de6bc8af` |

### Phase 8 已记录的全局指标

下表的 `number_of_rollouts=686`、`number_of_scenes=98`、`confidence=0.95`。

| metric | estimate | ci_low | ci_high |
| --- | ---: | ---: | ---: |
| `deterministic_chain_answer_exact` | 0.3498542274052478 | 0.27988338192419826 | 0.42128279883381925 |
| `error_cancellation` | 0.02040816326530612 | 0.0058309037900874635 | 0.039358600583090375 |
| `error_mechanism_shift` | 0.26239067055393583 | 0.2084548104956268 | 0.3163265306122449 |
| `final_answer_exact` | 0.08309037900874636 | 0.05102040816326531 | 0.11807580174927114 |
| `free_generation_answer_exact` | 0.08309037900874636 | 0.05102040816326531 | 0.11807580174927114 |
| `genuine_recovery` | 0.12099125364431486 | 0.07580174927113702 | 0.17201166180758018 |
| `operator_invariant_correct` | 0.02769679300291545 | 0.01020408163265306 | 0.048104956268221574 |
| `post_revision_world_exact` | 0.1588921282798834 | 0.10204081632653061 | 0.22011661807580174 |
| `reasoning_operator_exact` | 1.0 | 1.0 | 1.0 |
| `stage1_visual_exact` | 0.0597667638483965 | 0.02915451895043732 | 0.09329446064139942 |
| `trace_mismatch` | 0.8440233236151603 | 0.8002915451895044 | 0.8848396501457725 |

`answer_source_counts`：`error_cancellation=11`；`genuine_recovery=16`；`operator_invariance=15`；`visual_reread=8`；`unresolved=636`。

`seed_level_variability` 仅包含 evaluation seed `2026082104`；其十一项 estimate/CI 与上述 global metrics 相同。

### Phase 8 按 checkpoint 的十一项指标

各单元格式为 `estimate [95% CI]`；每 checkpoint 为 98 scenes / 98 rows。

| checkpoint | deterministic answer | free answer | stage1 exact | post-world exact | operator exact | genuine recovery | operator-invariant | cancellation | mechanism shift | trace mismatch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Base | 0.29591836734693877 [0.20408163265306123, 0.3877551020408163] | 0 [0, 0] | 0 [0, 0] | 0.11224489795918367 [0.05102040816326531, 0.1836734693877551] | 1 [1, 1] | 0.11224489795918367 [0.05102040816326531, 0.1836734693877551] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 1 [1, 1] |
| `Base_AnswerOnly_RL` | 0.2755102040816326 [0.19387755102040816, 0.3673469387755102] | 0 [0, 0] | 0 [0, 0] | 0.10204081632653061 [0.05102040816326531, 0.16326530612244897] | 1 [1, 1] | 0.10204081632653061 [0.05102040816326531, 0.16326530612244897] | 0 [0, 0] | 0 [0, 0] | 0.061224489795918366 [0.02040816326530612, 0.11224489795918367] | 1 [1, 1] |
| C0 | 0.24489795918367346 [0.16326530612244897, 0.32653061224489793] | 0 [0, 0] | 0.030612244897959183 [0, 0.07142857142857142] | 0.09183673469387756 [0.04081632653061224, 0.15306122448979592] | 1 [1, 1] | 0.061224489795918366 [0.02040816326530612, 0.11224489795918367] | 0 [0, 0] | 0 [0, 0] | 0.17346938775510204 [0.10204081632653061, 0.25510204081632654] | 1 [1, 1] |
| C1 | 0.25510204081632654 [0.17346938775510204, 0.3469387755102041] | 0.061224489795918366 [0.02040816326530612, 0.11224489795918367] | 0.04081632653061224 [0.01020408163265306, 0.08163265306122448] | 0.09183673469387756 [0.04081632653061224, 0.15306122448979592] | 1 [1, 1] | 0.05102040816326531 [0.01020408163265306, 0.10204081632653061] | 0.02040816326530612 [0, 0.05102040816326531] | 0.030612244897959183 [0, 0.07142857142857142] | 0.2755102040816326 [0.19387755102040816, 0.3673469387755102] | 0.8367346938775511 [0.7551020408163265, 0.9081632653061225] |
| `Recovery_LoRA_AnswerOnly_RL` | 0.4387755102040816 [0.3469387755102041, 0.5408163265306123] | 0.14285714285714285 [0.08163265306122448, 0.21428571428571427] | 0.11224489795918367 [0.05102040816326531, 0.17346938775510204] | 0.21428571428571427 [0.1326530612244898, 0.29591836734693877] | 1 [1, 1] | 0.15306122448979592 [0.08163265306122448, 0.22448979591836735] | 0.05102040816326531 [0.01020408163265306, 0.10204081632653061] | 0.030612244897959183 [0, 0.07142857142857142] | 0.42857142857142855 [0.32653061224489793, 0.5306122448979592] | 0.7040816326530612 [0.6122448979591837, 0.7857142857142857] |
| `Recovery_LoRA_RecoveryOutcome_RL` | 0.47959183673469385 [0.3877551020408163, 0.5816326530612245] | 0.19387755102040816 [0.12244897959183673, 0.2755102040816326] | 0.12244897959183673 [0.061224489795918366, 0.19387755102040816] | 0.25510204081632654 [0.17346938775510204, 0.3469387755102041] | 1 [1, 1] | 0.1836734693877551 [0.11224489795918367, 0.2653061224489796] | 0.061224489795918366 [0.02040816326530612, 0.11224489795918367] | 0.04081632653061224 [0.01020408163265306, 0.08163265306122448] | 0.45918367346938777 [0.35714285714285715, 0.5612244897959183] | 0.673469387755102 [0.5816326530612245, 0.7653061224489796] |
| T | 0.45918367346938777 [0.35714285714285715, 0.5612244897959183] | 0.1836734693877551 [0.11224489795918367, 0.2653061224489796] | 0.11224489795918367 [0.05102040816326531, 0.17346938775510204] | 0.24489795918367346 [0.16326530612244897, 0.32653061224489793] | 1 [1, 1] | 0.1836734693877551 [0.11224489795918367, 0.2653061224489796] | 0.061224489795918366 [0.02040816326530612, 0.11224489795918367] | 0.04081632653061224 [0.01020408163265306, 0.08163265306122448] | 0.4387755102040816 [0.336734693877551, 0.5408163265306123] | 0.6938775510204082 [0.6020408163265306, 0.7755102040816326] |

`final_answer_exact` 与 `free_generation_answer_exact` 在七个 checkpoint 的 estimate/CI 相同。

### Phase 8 deterministic-chain endpoint 注册效应

| effect | estimate | 95% CI | sign-flip p | Holm p | 90% TOST CI | equivalent | scenes |
| --- | ---: | --- | ---: | ---: | --- | --- | ---: |
| `T_minus_C0` | 0.21428571428571427 | [0.10204081632653061, 0.3163265306122449] | 0.0007999200079992001 | 0.0027997200279972004 | [0.12244897959183673, 0.30612244897959184] | false | 98 |
| `T_minus_C1` | 0.20408163265306123 | [0.10204081632653061, 0.30612244897959184] | 0.0006999300069993001 | 0.0027997200279972004 | [0.11224489795918367, 0.29591836734693877] | false | 98 |
| `recovery_reward_rl_minus_answer_only_rl` | 0.04081632653061224 | [0.01020408163265306, 0.08163265306122448] | 0.1282871712828717 | 0.2565743425657434 | [0.01020408163265306, 0.07142857142857142] | false | 98 |
| `seeded_rl_minus_base_rl` | 0.0 | [-0.05102040816326531, 0.05102040816326531] | 1.0 | 1.0 | [-0.04081632653061224, 0.04081632653061224] | false | 98 |

### Phase 8 free-generation endpoint 注册效应

所有比较的 `paired_scene_count=98`；主区间为 95% scene-clustered percentile bootstrap CI；TOST 区间为 90%，margin=`0.02`。

| effect | estimate | 95% CI | sign-flip p | Holm p | 90% TOST CI | equivalent |
| --- | ---: | --- | ---: | ---: | --- | --- |
| `T_minus_C0` | 0.1836734693877551 | [0.11224489795918367, 0.2653061224489796] | 0.00009999000099990002 | 0.00039996000399960006 | [0.12244897959183673, 0.25510204081632654] | false |
| `T_minus_C1` | 0.12244897959183673 | [0.05102040816326531, 0.20408163265306123] | 0.004199580041995801 | 0.012598740125987402 | [0.061224489795918366, 0.1836734693877551] | false |
| `recovery_reward_rl_minus_answer_only_rl` | 0.04081632653061224 | [0.01020408163265306, 0.08163265306122448] | 0.12058794120587941 | 0.24117588241175883 | [0.01020408163265306, 0.07142857142857142] | false |
| `seeded_rl_minus_base_rl` | -0.04081632653061224 | [-0.08163265306122448, -0.01020408163265306] | 0.12238776122387761 | 0.24117588241175883 | [-0.07142857142857142, -0.01020408163265306] | false |

### Phase 8 free-generation 注册效应按 OOD axis

| OOD axis | scenes | effect | estimate | 95% CI | sign-flip p | Holm p |
| --- | ---: | --- | ---: | --- | ---: | ---: |
| `constraint_graph_ood` | 27 | `T_minus_C0` | 0.07407407407407407 | [0.0, 0.18518518518518517] | 0.49745025497450257 | 1.0 |
| `constraint_graph_ood` | 27 | `T_minus_C1` | 0.037037037037037035 | [-0.07407407407407407, 0.14814814814814814] | 1.0 | 1.0 |
| `constraint_graph_ood` | 27 | `recovery_reward_rl_minus_answer_only_rl` | 0.0 | [0.0, 0.0] | 1.0 | 1.0 |
| `constraint_graph_ood` | 27 | `seeded_rl_minus_base_rl` | 0.0 | [0.0, 0.0] | 1.0 | 1.0 |
| `error_mechanism_ood` | 12 | `T_minus_C0` | 0.3333333333333333 | [0.08333333333333333, 0.5833333333333334] | 0.125 | 0.5 |
| `error_mechanism_ood` | 12 | `T_minus_C1` | 0.16666666666666666 | [0.0, 0.4166666666666667] | 0.5 | 1.0 |
| `error_mechanism_ood` | 12 | `recovery_reward_rl_minus_answer_only_rl` | 0.08333333333333333 | [0.0, 0.25] | 1.0 | 1.0 |
| `error_mechanism_ood` | 12 | `seeded_rl_minus_base_rl` | 0.0 | [0.0, 0.0] | 1.0 | 1.0 |
| `iid` | 31 | `T_minus_C0` | 0.22580645161290322 | [0.0967741935483871, 0.3870967741935484] | 0.013598640135986401 | 0.054394560543945605 |
| `iid` | 31 | `T_minus_C1` | 0.16129032258064516 | [0.0, 0.3225806451612903] | 0.12638736126387362 | 0.37916208379162086 |
| `iid` | 31 | `recovery_reward_rl_minus_answer_only_rl` | 0.03225806451612903 | [0.0, 0.0967741935483871] | 1.0 | 1.0 |
| `iid` | 31 | `seeded_rl_minus_base_rl` | -0.03225806451612903 | [-0.0967741935483871, 0.0] | 1.0 | 1.0 |
| `style_ood` | 28 | `T_minus_C0` | 0.17857142857142858 | [0.03571428571428571, 0.32142857142857145] | 0.06189381061893811 | 0.24757524247575244 |
| `style_ood` | 28 | `T_minus_C1` | 0.14285714285714285 | [0.03571428571428571, 0.2857142857142857] | 0.12218778122187782 | 0.3665633436656335 |
| `style_ood` | 28 | `recovery_reward_rl_minus_answer_only_rl` | 0.07142857142857142 | [0.0, 0.17857142857142858] | 0.5013498650134987 | 0.5013498650134987 |
| `style_ood` | 28 | `seeded_rl_minus_base_rl` | -0.10714285714285714 | [-0.25, 0.0] | 0.24477552244775522 | 0.48955104489551043 |

### Phase 8 free-generation 注册效应按 family

| family | scenes | effect | estimate | 95% CI | sign-flip p | Holm p |
| --- | ---: | --- | ---: | --- | ---: | ---: |
| `cross_series` | 34 | `T_minus_C0` | 0.11764705882352941 | [0.029411764705882353, 0.23529411764705882] | 0.1217878212178782 | 0.4871512848715128 |
| `cross_series` | 34 | `T_minus_C1` | 0.11764705882352941 | [0.029411764705882353, 0.23529411764705882] | 0.12708729127087293 | 0.4871512848715128 |
| `cross_series` | 34 | `recovery_reward_rl_minus_answer_only_rl` | 0.029411764705882353 | [0.0, 0.08823529411764706] | 1.0 | 1.0 |
| `cross_series` | 34 | `seeded_rl_minus_base_rl` | -0.029411764705882353 | [-0.08823529411764706, 0.0] | 1.0 | 1.0 |
| `duplicate_encoding` | 34 | `T_minus_C0` | 0.23529411764705882 | [0.08823529411764706, 0.38235294117647056] | 0.007599240075992401 | 0.030396960303969604 |
| `duplicate_encoding` | 34 | `T_minus_C1` | 0.17647058823529413 | [0.058823529411764705, 0.3235294117647059] | 0.0327967203279672 | 0.0983901609839016 |
| `duplicate_encoding` | 34 | `recovery_reward_rl_minus_answer_only_rl` | 0.08823529411764706 | [0.0, 0.20588235294117646] | 0.24397560243975602 | 0.48795120487951205 |
| `duplicate_encoding` | 34 | `seeded_rl_minus_base_rl` | -0.058823529411764705 | [-0.14705882352941177, 0.0] | 0.48785121487851213 | 0.48795120487951205 |
| `trend` | 30 | `T_minus_C0` | 0.2 | [0.06666666666666667, 0.3333333333333333] | 0.03469653034696531 | 0.13878612138786123 |
| `trend` | 30 | `T_minus_C1` | 0.06666666666666667 | [-0.1, 0.23333333333333334] | 0.682931706829317 | 1.0 |
| `trend` | 30 | `recovery_reward_rl_minus_answer_only_rl` | 0.0 | [0.0, 0.0] | 1.0 | 1.0 |
| `trend` | 30 | `seeded_rl_minus_base_rl` | -0.03333333333333333 | [-0.1, 0.0] | 1.0 | 1.0 |

### Phase 8 正式服务器产物

| path | 状态 |
| --- | --- |
| `artifacts/v4/phase8/confirm_data/confirm_scenes.jsonl` | 已生成；SHA-256 见 source 表 |
| `artifacts/v4/phase8/confirm_data/confirm_observations.jsonl` | 已生成；SHA-256 见 source 表 |
| `artifacts/v4/phase8/confirm_data/selection_trace.jsonl` | SHA-256=`650364dacf62d3fde9b2d8e11a9606ea575470e4f4fd8bb7f1ab78ebbd2cc1e3` |
| `artifacts/v4/phase8/confirm_data/summary.json` | 已生成；SHA-256=`3ac1c430149f464d15693da42f1c70fbc17fbb4f385f937049432f1016a38ca7` |
| `artifacts/v4/phase8/confirm_data/execution_manifest.json` | 已生成；SHA-256=`52fe08b3b5db5e99327302f0e4e5f86e4f403684b136d275d179eeda1c52c776` |
| `artifacts/v4/phase8/evaluation/per_scene.jsonl` | SHA-256=`f897cc66f2c9162b3d034d387ec6a09e84939357eedc1f860af18d12747e8fe8` |
| `artifacts/v4/phase8/evaluation/summary.json` | SHA-256=`a46bd1a494a1468cbd7f62878474e7fc877a10f88a09cc6e88a611f1de6bc8af` |

### Phase 8 已记录的执行错误与修复

| revision | server message | subsequent state |
| --- | --- | --- |
| `dbd6a5ce0b6e187db55c63be13d0dcffcb842662` | `BLOCKED: Phase 8 'PairSumConstraint' object is not iterable` | `d79feb8900d779defd3d5e577f21f728596d4b2f` 将三种 constraint dataclass 显式序列化为 recovery-scene fact mapping；随后完成数据冻结与正式评估 |

## 附录 A：Phase 7 与 Phase 8 完整分层指标

以下所有单元均为 scene-clustered estimate 与 95% CI；数值直接取自事实包中的正式 summary。

### Phase 7 指标按 family

| metric | family | scenes | estimate | 95% CI |
| --- | --- | ---: | ---: | --- |
| error_cancellation | cross_series | 16 | 0.0 | [0.0, 0.0] |
| error_cancellation | duplicate_encoding | 5 | 0.0 | [0.0, 0.0] |
| error_cancellation | trend | 11 | 0.025974025974025972 | [0.0, 0.07792207792207792] |
| error_mechanism_shift | cross_series | 16 | 0.1875 | [0.08035714285714285, 0.3125] |
| error_mechanism_shift | duplicate_encoding | 5 | 0.45714285714285713 | [0.2, 0.6857142857142857] |
| error_mechanism_shift | trend | 11 | 0.22077922077922077 | [0.09090909090909091, 0.3506493506493506] |
| final_answer_exact | cross_series | 16 | 0.03571428571428571 | [0.0, 0.09821428571428571] |
| final_answer_exact | duplicate_encoding | 5 | 0.08571428571428572 | [0.0, 0.2571428571428571] |
| final_answer_exact | trend | 11 | 0.23376623376623373 | [0.10389610389610389, 0.36363636363636365] |
| genuine_recovery | cross_series | 16 | 0.1607142857142857 | [0.05357142857142857, 0.26785714285714285] |
| genuine_recovery | duplicate_encoding | 5 | 0.08571428571428572 | [0.0, 0.2571428571428571] |
| genuine_recovery | trend | 11 | 0.03896103896103896 | [0.0, 0.11688311688311687] |
| operator_invariant_correct | cross_series | 16 | 0.008928571428571428 | [0.0, 0.026785714285714284] |
| operator_invariant_correct | duplicate_encoding | 5 | 0.0 | [0.0, 0.0] |
| operator_invariant_correct | trend | 11 | 0.14285714285714285 | [0.03896103896103896, 0.2597402597402597] |
| post_revision_world_exact | cross_series | 16 | 0.26785714285714285 | [0.16964285714285712, 0.3571428571428571] |
| post_revision_world_exact | duplicate_encoding | 5 | 0.5428571428571428 | [0.42857142857142855, 0.7142857142857142] |
| post_revision_world_exact | trend | 11 | 0.07792207792207792 | [0.012987012987012986, 0.1688311688311688] |
| reasoning_operator_exact | cross_series | 16 | 1.0 | [1.0, 1.0] |
| reasoning_operator_exact | duplicate_encoding | 5 | 1.0 | [1.0, 1.0] |
| reasoning_operator_exact | trend | 11 | 1.0 | [1.0, 1.0] |
| stage1_visual_exact | cross_series | 16 | 0.1875 | [0.08035714285714285, 0.3125] |
| stage1_visual_exact | duplicate_encoding | 5 | 0.45714285714285713 | [0.2, 0.6857142857142857] |
| stage1_visual_exact | trend | 11 | 0.22077922077922077 | [0.09090909090909091, 0.3506493506493506] |
| trace_mismatch | cross_series | 16 | 0.8928571428571428 | [0.7857142857142857, 0.9910714285714286] |
| trace_mismatch | duplicate_encoding | 5 | 0.8857142857142858 | [0.7142857142857142, 1.0] |
| trace_mismatch | trend | 11 | 0.7272727272727273 | [0.5974025974025974, 0.8571428571428572] |

### Phase 7 指标按 OOD axis

| metric | OOD axis | scenes | estimate | 95% CI |
| --- | --- | ---: | ---: | --- |
| error_cancellation | iid | 32 | 0.008928571428571428 | [0.0, 0.026785714285714284] |
| error_mechanism_shift | iid | 32 | 0.24107142857142855 | [0.15625, 0.33035714285714285] |
| final_answer_exact | iid | 32 | 0.11160714285714285 | [0.049107142857142856, 0.18303571428571427] |
| genuine_recovery | iid | 32 | 0.10714285714285714 | [0.040178571428571425, 0.17410714285714285] |
| operator_invariant_correct | iid | 32 | 0.05357142857142857 | [0.013392857142857142, 0.10267857142857142] |
| post_revision_world_exact | iid | 32 | 0.24553571428571427 | [0.16964285714285712, 0.32589285714285715] |
| reasoning_operator_exact | iid | 32 | 1.0 | [1.0, 1.0] |
| stage1_visual_exact | iid | 32 | 0.24107142857142855 | [0.15625, 0.33482142857142855] |
| trace_mismatch | iid | 32 | 0.8348214285714286 | [0.7544642857142857, 0.90625] |

### Phase 8 指标按 family

| metric | family | scenes | estimate | 95% CI |
| --- | --- | ---: | ---: | --- |
| error_cancellation | cross_series | 34 | 0.0 | [0.0, 0.0] |
| error_cancellation | duplicate_encoding | 34 | 0.0 | [0.0, 0.0] |
| error_cancellation | trend | 30 | 0.06666666666666667 | [0.019047619047619046, 0.1238095238095238] |
| error_mechanism_shift | cross_series | 34 | 0.20588235294117646 | [0.13025210084033612, 0.2857142857142857] |
| error_mechanism_shift | duplicate_encoding | 34 | 0.3025210084033613 | [0.21008403361344535, 0.3949579831932773] |
| error_mechanism_shift | trend | 30 | 0.28095238095238095 | [0.18095238095238095, 0.380952380952381] |
| final_answer_exact | cross_series | 34 | 0.050420168067226885 | [0.012605042016806721, 0.09663865546218486] |
| final_answer_exact | duplicate_encoding | 34 | 0.10084033613445377 | [0.04201680672268907, 0.1680672268907563] |
| final_answer_exact | trend | 30 | 0.1 | [0.03809523809523809, 0.1714285714285714] |
| genuine_recovery | cross_series | 34 | 0.012605042016806721 | [0.0, 0.037815126050420166] |
| genuine_recovery | duplicate_encoding | 34 | 0.3235294117647059 | [0.21848739495798317, 0.43277310924369744] |
| genuine_recovery | trend | 30 | 0.014285714285714285 | [0.0, 0.04285714285714285] |
| operator_invariant_correct | cross_series | 34 | 0.046218487394957986 | [0.012605042016806721, 0.08823529411764706] |
| operator_invariant_correct | duplicate_encoding | 34 | 0.004201680672268907 | [0.0, 0.012605042016806721] |
| operator_invariant_correct | trend | 30 | 0.03333333333333333 | [0.0, 0.07619047619047618] |
| post_revision_world_exact | cross_series | 34 | 0.025210084033613443 | [0.0, 0.058823529411764705] |
| post_revision_world_exact | duplicate_encoding | 34 | 0.4033613445378151 | [0.2773109243697479, 0.5294117647058824] |
| post_revision_world_exact | trend | 30 | 0.03333333333333333 | [0.0, 0.07619047619047618] |
| reasoning_operator_exact | cross_series | 34 | 1.0 | [1.0, 1.0] |
| reasoning_operator_exact | duplicate_encoding | 34 | 1.0 | [1.0, 1.0] |
| reasoning_operator_exact | trend | 30 | 1.0 | [1.0, 1.0] |
| stage1_visual_exact | cross_series | 34 | 0.06302521008403361 | [0.012605042016806721, 0.12184873949579833] |
| stage1_visual_exact | duplicate_encoding | 34 | 0.07983193277310925 | [0.02100840336134454, 0.15546218487394958] |
| stage1_visual_exact | trend | 30 | 0.03333333333333333 | [0.0, 0.07619047619047618] |
| trace_mismatch | cross_series | 34 | 0.8781512605042017 | [0.8067226890756302, 0.9453781512605043] |
| trace_mismatch | duplicate_encoding | 34 | 0.861344537815126 | [0.7857142857142857, 0.9327731092436975] |
| trace_mismatch | trend | 30 | 0.7857142857142857 | [0.7047619047619047, 0.861904761904762] |

### Phase 8 指标按 OOD axis

| metric | OOD axis | scenes | estimate | 95% CI |
| --- | --- | ---: | ---: | --- |
| error_cancellation | constraint_graph_ood | 27 | 0.015873015873015872 | [0.0, 0.047619047619047616] |
| error_cancellation | error_mechanism_ood | 12 | 0.0 | [0.0, 0.0] |
| error_cancellation | iid | 31 | 0.03225806451612903 | [0.0, 0.07373271889400922] |
| error_cancellation | style_ood | 28 | 0.02040816326530612 | [0.0, 0.05612244897959184] |
| error_mechanism_shift | constraint_graph_ood | 27 | 0.2857142857142857 | [0.19047619047619047, 0.37566137566137564] |
| error_mechanism_shift | error_mechanism_ood | 12 | 0.32142857142857145 | [0.16666666666666666, 0.48809523809523814] |
| error_mechanism_shift | iid | 31 | 0.15668202764976957 | [0.08294930875576036, 0.23963133640552994] |
| error_mechanism_shift | style_ood | 28 | 0.33163265306122447 | [0.22959183673469385, 0.4387755102040816] |
| final_answer_exact | constraint_graph_ood | 27 | 0.037037037037037035 | [0.0, 0.08465608465608465] |
| final_answer_exact | error_mechanism_ood | 12 | 0.16666666666666666 | [0.03571428571428571, 0.3095238095238095] |
| final_answer_exact | iid | 31 | 0.10138248847926266 | [0.04147465437788018, 0.1705069124423963] |
| final_answer_exact | style_ood | 28 | 0.07142857142857142 | [0.02040816326530612, 0.13265306122448978] |
| genuine_recovery | constraint_graph_ood | 27 | 0.005291005291005291 | [0.0, 0.015873015873015872] |
| genuine_recovery | error_mechanism_ood | 12 | 0.2738095238095238 | [0.10714285714285714, 0.4761904761904762] |
| genuine_recovery | iid | 31 | 0.13364055299539168 | [0.046082949308755755, 0.2350230414746544] |
| genuine_recovery | style_ood | 28 | 0.15306122448979592 | [0.07142857142857142, 0.25] |
| operator_invariant_correct | constraint_graph_ood | 27 | 0.021164021164021163 | [0.0, 0.0582010582010582] |
| operator_invariant_correct | error_mechanism_ood | 12 | 0.03571428571428571 | [0.0, 0.09523809523809523] |
| operator_invariant_correct | iid | 31 | 0.027649769585253454 | [0.0, 0.06912442396313363] |
| operator_invariant_correct | style_ood | 28 | 0.030612244897959183 | [0.0, 0.07142857142857142] |
| post_revision_world_exact | constraint_graph_ood | 27 | 0.042328042328042326 | [0.005291005291005291, 0.08994708994708994] |
| post_revision_world_exact | error_mechanism_ood | 12 | 0.44047619047619047 | [0.19047619047619047, 0.6904761904761906] |
| post_revision_world_exact | iid | 31 | 0.14746543778801843 | [0.04608294930875576, 0.2626728110599078] |
| post_revision_world_exact | style_ood | 28 | 0.16326530612244897 | [0.07653061224489796, 0.25510204081632654] |
| reasoning_operator_exact | constraint_graph_ood | 27 | 1.0 | [1.0, 1.0] |
| reasoning_operator_exact | error_mechanism_ood | 12 | 1.0 | [1.0, 1.0] |
| reasoning_operator_exact | iid | 31 | 1.0 | [1.0, 1.0] |
| reasoning_operator_exact | style_ood | 28 | 1.0 | [1.0, 1.0] |
| stage1_visual_exact | constraint_graph_ood | 27 | 0.037037037037037035 | [0.0, 0.08465608465608465] |
| stage1_visual_exact | error_mechanism_ood | 12 | 0.2261904761904762 | [0.07142857142857142, 0.39285714285714285] |
| stage1_visual_exact | iid | 31 | 0.027649769585253454 | [0.0, 0.06912442396313363] |
| stage1_visual_exact | style_ood | 28 | 0.04591836734693877 | [0.0, 0.10714285714285714] |
| trace_mismatch | constraint_graph_ood | 27 | 0.8677248677248677 | [0.783068783068783, 0.9417989417989417] |
| trace_mismatch | error_mechanism_ood | 12 | 0.75 | [0.6071428571428571, 0.8928571428571428] |
| trace_mismatch | iid | 31 | 0.847926267281106 | [0.7649769585253456, 0.9262672811059908] |
| trace_mismatch | style_ood | 28 | 0.8571428571428571 | [0.7806122448979592, 0.9285714285714286] |

### Phase 8 两个答案 endpoint 按 family

| endpoint | family | scenes | estimate | 95% CI |
| --- | --- | ---: | ---: | --- |
| deterministic_chain_answer_exact | cross_series | 34 | 0.2647058823529412 | [0.15966386554621848, 0.37815126050420167] |
| deterministic_chain_answer_exact | duplicate_encoding | 34 | 0.5126050420168067 | [0.39075630252100835, 0.638655462184874] |
| deterministic_chain_answer_exact | trend | 30 | 0.2619047619047619 | [0.15238095238095237, 0.380952380952381] |
| free_generation_answer_exact | cross_series | 34 | 0.050420168067226885 | [0.012605042016806721, 0.09663865546218486] |
| free_generation_answer_exact | duplicate_encoding | 34 | 0.10084033613445377 | [0.04201680672268907, 0.1680672268907563] |
| free_generation_answer_exact | trend | 30 | 0.1 | [0.03809523809523809, 0.1714285714285714] |

### Phase 8 两个答案 endpoint 按 OOD axis

| endpoint | OOD axis | scenes | estimate | 95% CI |
| --- | --- | ---: | ---: | --- |
| deterministic_chain_answer_exact | constraint_graph_ood | 27 | 0.25925925925925924 | [0.14814814814814814, 0.37566137566137564] |
| deterministic_chain_answer_exact | error_mechanism_ood | 12 | 0.6904761904761906 | [0.4761904761904762, 0.8809523809523809] |
| deterministic_chain_answer_exact | iid | 31 | 0.3087557603686636 | [0.1889400921658986, 0.4377880184331797] |
| deterministic_chain_answer_exact | style_ood | 28 | 0.336734693877551 | [0.22448979591836735, 0.45918367346938777] |
| free_generation_answer_exact | constraint_graph_ood | 27 | 0.037037037037037035 | [0.0, 0.08465608465608465] |
| free_generation_answer_exact | error_mechanism_ood | 12 | 0.16666666666666666 | [0.03571428571428571, 0.3095238095238095] |
| free_generation_answer_exact | iid | 31 | 0.10138248847926266 | [0.04147465437788018, 0.1658986175115207] |
| free_generation_answer_exact | style_ood | 28 | 0.07142857142857142 | [0.02040816326530612, 0.13265306122448978] |

## 附录 B：服务器事实包与产物清单

| 字段 | 值 |
| --- | --- |
| supplied archive | /Users/louis/Downloads/qwen_v4_fact_bundle_20260821.tar.gz |
| archive SHA-256 | 6fe3d48bc1a88b35bd5804bc397ce8160534bdd7d71fa6769b755cc098a53683 |
| manifest schema version | 1 |
| formal files in manifest | 58 |

| server artifact path | bytes | SHA-256 |
| --- | ---: | --- |
| artifacts/v4/audit/claim_evidence_matrix.csv | 819 | 64d49c35bdb5c4c2dea61eb6d7a9d6261b399d589ca1a95d746229fba657bffd |
| artifacts/v4/audit/legacy_experiment_registry.csv | 1165 | 16a66c205b978de1565f0de908308bb33104fceca10666a212c1d3d2ba18ae82 |
| artifacts/v4/audit/legacy_hash_manifest.json | 2168 | 7891a6e8101b1bd72cdd59b0217c95d447d3b18715fcc61cce1d26dc298a8753 |
| artifacts/v4/audit/scoring_contract.md | 1440 | 4f433bf75f5867ff85af15f531c64a3004288539e1ee3456e4a5a1a8334b6871 |
| artifacts/v4/cache/cache_parity.json | 69215306 | c52cb71d42c83e3a32c57c00e006f5117631b9cab25a2ef8fbe62001ff572351 |
| artifacts/v4/candidate_scoring/per_scene.jsonl | 2358983 | c303731438760a30fb2f78a489d20465a1c1e292b01941795f29351a0e234a62 |
| artifacts/v4/candidate_scoring/summary.json | 5965 | 5e366dfecb2a4fd530407f896326bc99d3605385cfdadea34c0f478392280c73 |
| artifacts/v4/capability_chain/paired_gaps.json | 2981 | a2a5acb5b203719e7e5225e56643a25a4189277d13704cccd4ec86d61573e256 |
| artifacts/v4/capability_chain/per_scene.csv | 324887 | d01c391e136ed0e5c0ed52e50fe70f6ec128d221d218e7012c9adbcb4293929f |
| artifacts/v4/capability_chain/summary_by_family.csv | 830 | 8837a7275915f5a90c91eae8378ff6bd0466819842381996db1be8e4925705c7 |
| artifacts/v4/interface_ladder/per_scene.jsonl | 7783149 | 324dbf3289d6754685483feebcf0ebd785210c250c5360859391b6fb66c089e5 |
| artifacts/v4/interface_ladder/summary.json | 27393 | 46ce49ed8ad74baccfba6cc5d2b474cd0260c4a852c6d94915bae7a8bcc99bcc |
| artifacts/v4/layerwise_assimilation/per_scene.jsonl | 9396795 | e696d12bb8cb3e6142a3d6ecc6de9474c3e72e3ac85e0c7334005a249556a4af |
| artifacts/v4/layerwise_assimilation/summary.json | 33026 | 53eab07dcd70fce6970a63ce1831ec6369164e92320f051e717decdeb1b790c0 |
| artifacts/v4/phase6/execution_manifest.1759bc7.json | 7494 | 47b9f10638dae1957d03beae0a227cdbc48c922f76554bd69ce02b86a6fd73ef |
| artifacts/v4/phase6/execution_manifest.7c44b58.json | 7494 | af8acece46c67bd593eed778289012f51ce46fe668a927d718ee5774897ec9d3 |
| artifacts/v4/phase6/execution_manifest.json | 7494 | f8d5b2a71bd06833ea803dbbf916fe56ea220bdaef97291bd70f5f2f480a644e |
| artifacts/v4/phase7/evaluation/per_scene.jsonl | 139546 | 487af12e77f8d2592046b0a6bdc681a14953fcd382902b716fc0a8563c3f602e |
| artifacts/v4/phase7/evaluation/summary.json | 39866 | fde66bae90f58eb64de5d6cb32e0a0f38f866c439775b17e7de64a31987d86db |
| artifacts/v4/phase7/execution_manifest.json | 2681 | 1fb6e640350c37dc966cbd76a6e8b6d7c388ebc3e2c55a0b0b8fd3e6dacb9010 |
| artifacts/v4/phase7/interface_audit.json | 5676 | abbd06e6d1f76bbd549009f4993a748d6be5feac2d8b8f1a899aea438cea004b |
| artifacts/v4/phase8/confirm_data/confirm_observations.jsonl | 39444 | 7f0a91ca81d9764581b47c3d0ef94632812adee6aae14d5918c63e10cd41c2e2 |
| artifacts/v4/phase8/confirm_data/confirm_scenes.jsonl | 95093 | 7432bb9f3b0ceda7364efdda1e13594a71595eed62312062b12b5670cb3d9946 |
| artifacts/v4/phase8/confirm_data/execution_manifest.json | 2521 | 52fe08b3b5db5e99327302f0e4e5f86e4f403684b136d275d179eeda1c52c776 |
| artifacts/v4/phase8/confirm_data/selection_trace.jsonl | 97090 | 650364dacf62d3fde9b2d8e11a9606ea575470e4f4fd8bb7f1ab78ebbd2cc1e3 |
| artifacts/v4/phase8/confirm_data/summary.json | 2455 | 3ac1c430149f464d15693da42f1c70fbc17fbb4f385f937049432f1016a38ca7 |
| artifacts/v4/phase8/evaluation/per_scene.jsonl | 518021 | f897cc66f2c9162b3d034d387ec6a09e84939357eedc1f860af18d12747e8fe8 |
| artifacts/v4/phase8/evaluation/summary.json | 68915 | a46bd1a494a1468cbd7f62878474e7fc877a10f88a09cc6e88a611f1de6bc8af |
| artifacts/v4/rl/data/answer_only.jsonl | 154860 | 3e72aa47c74f8e02191d949f968612d54bca9a81bfeb18f3cebd5d2394680d8e |
| artifacts/v4/rl/data/recovery_outcome.jsonl | 154374 | 7af9f17c1febd3e0bfa2cddb0d6f5cde35500c12405952f9ad071e278b4b1993 |
| artifacts/v4/rl/data/summary.json | 1696 | 337a32fac096c6034f29f96a2a1e7f7217474511839b9b8de2cc3764e856d171 |
| artifacts/v4/rl/evaluation/by_scene.jsonl | 382569 | 6dbf0d2900628227228b59bd7f7b0cfad6d907664e23d3a14c802cc193a35eb8 |
| artifacts/v4/rl/evaluation/summary.json | 13876 | cfcd8aa5882ea9719f0acaad1da5a9b0f5776739f1140b35b59c6eeafbf3a9d9 |
| artifacts/v4/rl/runs/phase6-r1/Base_AnswerOnly_RL/execution_evidence.json | 64877 | 39443ba4f14786034be9f55194c539c0f7d3546b6a6a4f12ef505b7e9b0d558b |
| artifacts/v4/rl/runs/phase6-r1/Base_AnswerOnly_RL/grpo_signal_diagnostics.json | 406 | 15431c3adfa0eb5403210c01325a31081d11e98ef314e157c80ad38f4d0f82dc |
| artifacts/v4/rl/runs/phase6-r1/Base_AnswerOnly_RL/metrics.json | 59778 | 3093ab382a3d214674bef3dbd3806f4b6332e51956329cbb31b32c248b868bf0 |
| artifacts/v4/rl/runs/phase6-r1/Recovery_LoRA_AnswerOnly_RL/execution_evidence.json | 64883 | a3d563bcce7b4fa3b5c5e6d749d7b2b3e4a302d1d0bd8dcd8a28b01ede2664b9 |
| artifacts/v4/rl/runs/phase6-r1/Recovery_LoRA_AnswerOnly_RL/grpo_signal_diagnostics.json | 402 | d8501e96908ea0a57fc7bb6a5d30d10280634ca163fd66d6298e8aaeba39c8ca |
| artifacts/v4/rl/runs/phase6-r1/Recovery_LoRA_AnswerOnly_RL/metrics.json | 60418 | 23f07a6f59b20bf92f54c663791e934d4ede2b4ae5ce9bb2ac85413f68926610 |
| artifacts/v4/rl/runs/phase6-r1/Recovery_LoRA_RecoveryOutcome_RL/execution_evidence.json | 64893 | 2243eec5fe86c55a6f94b57a82c0cf20355ef00fca08f335971287a55b6660fb |
| artifacts/v4/rl/runs/phase6-r1/Recovery_LoRA_RecoveryOutcome_RL/grpo_signal_diagnostics.json | 428 | c195f82b704dc42e46dcf1d5f18716703c69c4728491c2fd7d14060241681421 |
| artifacts/v4/rl/runs/phase6-r1/Recovery_LoRA_RecoveryOutcome_RL/metrics.json | 62250 | 1dca1fc99e13cf1cbe4df7246aa3b8723ef66f43d29fbe6d229226f184230ed3 |
| artifacts/v4/support/informative_group_rate.json | 5351 | 2cdcb288986a4769aaa6caf477e51595a7648788cbd5471331b12fb049ffd358 |
| artifacts/v4/support/pass_at_k.csv | 510 | 4de8876cc16b723c888e0a8db5a131978eb175ba54624735e69bc1a4a739ec49 |
| artifacts/v4/support/policy_support_by_scene.parquet | 31177 | 196896097b5a261733529a903e1ea8cc1ee1cbeffeda397a233b7044e01c77b7 |
| artifacts/v4/support_dev/candidates.jsonl | 417007 | 33150ff9edcb89246468af688126520075dd127095b7866ca32a1dbcb9aeabbb |
| artifacts/v4/support_dev/held_out_natural_errors.jsonl | 24133 | cafbd5e5a0face1622d62faac4bb6b9bfc1fa6cc8d4353ade82b9144d8b18136 |
| artifacts/v4/support_dev/selection_trace.jsonl | 98678 | cd3e05daff2f16d462dee92805c15290523bad9fa3194f092089038db09ac5ce |
| artifacts/v4/support_dev/summary.json | 1425 | f9dd52a363a97a961c8eb55fd69f305507a5b5ba0ad171aae1996642f330a9cf |
| artifacts/v4/tokenizer/candidate_labels.json | 1908 | a7a448f230038698c4127b220362c95d47f57cef90cc7904e71b1dacacc04dbd |
| artifacts/v4/training/frozen_hashes.json | 454 | f04189c34e65b4472dee80a4c3c72aea2f82f1681247809603112a7e46d593d8 |
| artifacts/v4/training/runs/phase4-r1/C0_format_only/metrics.json | 16723 | 5eef36865797076abfbeb20a1a39c0cbfb257b70f3b30b9dcb3e7b78b86ebc1d |
| artifacts/v4/training/runs/phase4-r1/C1_forward_arithmetic/metrics.json | 16678 | a2b97aa8cd54d3164252aa0dac46843b5ffde39def9e2072f66a6f98b6386d1c |
| artifacts/v4/training/runs/phase4-r1/T_constraint_recovery/metrics.json | 99106 | dcde0eb56a6aaf73fb9af60681eebe2af0ffd5ddeebdfb1844c9e43a23ea84fa |
| artifacts/v4/training/sources/source_summary.json | 1292 | 32763a9648d12f841f9231e0a16db15597fac1a50c545f8871696108c41c8322 |
| artifacts/v4/training/support.jsonl | 4440583 | 0bc003d7fdf4f2f67aecef42a25e8caf2f33433fa969aea1962df5ee397a9c5e |
| artifacts/v4/training/support_summary.json | 692 | ac0d3426eb0f9705b2e4af5d7096dc2c6c2e9b0ea681fe17600864d872b2d292 |
| artifacts/v4/training/trainable_parameter_manifest.json | 60104 | 3c1327efd60379ac89c5b71c4ba9d1e9ddae2a7750a2b5cb435cea356c34ee1a |
