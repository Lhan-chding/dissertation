# 历史源码与 compact 证据核对

本地 HEAD：`7e405faa241e0085f701cdc858ba423ea76ba5a8`。历史 warm runtime 记录提交：`9cb60cd8e4e26f881e83f80f20b55999f8b8ecde`。

逐文件比较 runtime_lock.source.source_files：共 63 个原有文件，全部文件 SHA-256 相同；没有缺失或差异。提交标识不同与原有源码文件内容相同可以同时成立。

本轮新增 followup 模块不在历史源码表中，其本地 CPU 检查独立记录。以下 Git blob OID 属于 SHA-1 命名空间，文件 SHA-256 属于字节摘要命名空间；两者不互相比较。模型参数 hash、完整 state hash 只来自原元数据，未核验缺失的原始张量。

| 文件 | 历史与当前文件 SHA-256 | 当前 HEAD Git blob OID | 结果 |
|---|---|---|---|
| `src/__init__.py` | `50a703cc52bc34025bddb66d0bfab19235d4a38e3342b2db96806f0a1db45f49` | `9b87194ae1a8b33d9c696a70084394ed84a14bdb` | MATCH |
| `src/audit_legacy.py` | `ecda38e0297687977e4ee03e9f6469db66a363cd830aaa27d00b20782d7a3816` | `846510fa54452b2fa13041ee949b35bc97d68221` | MATCH |
| `src/audit_p3_artifacts.py` | `98ede4831da8ca6e73404e58a16c7eced60e48fea7b33e6e366eb53b05c8bf1a` | `ddc026eddde4dd2d2a444876f80effe8785f98ff` | MATCH |
| `src/audit_r0_remaining.py` | `af989fdf5098e13168c30d934d4f37ced6dd02f5c5990109bce9ad3078fa07ca` | `3bd588f1bea2ac3ebb9cea123d4eeeddbf0b11bf` | MATCH |
| `src/audit_r1_runtime.py` | `29a361617282137153df6b8859491d6e66700d7befd0fe4b5bee078c40e277cf` | `a0f916f1ab31e359ef248780e36c95d18205688f` | MATCH |
| `src/audit_sampling_and_runtime.py` | `1e8499c40f93fb45bc90a001ddb716f019c76181999556e7885a46a77fcb485e` | `93e0e4c931026f417218ae15d68d1b83df1c0409` | MATCH |
| `src/constraint_solver.py` | `eab20b128ddf2abeca5f115dbc8c9d51c318d810188423fb44370efdd7443b7b` | `674e539a6e955f7194faba3b5d786aa1c3e28c38` | MATCH |
| `src/core.py` | `7ef83bf265510205e52d517e150e278966bb34711c272a2b9f45c995694e7c45` | `a9a85def3f04577fddc27e36d6996e1e2e749b28` | MATCH |
| `src/decoding_audit.py` | `6e19fb0234e89d57601f37647bfd4709a6aaee5fb4b992b8fdef827b54c24a55` | `12f2c5ea81e14385ba7b398cbc674d92910ce66d` | MATCH |
| `src/diagnose_repair_inputs.py` | `dc6884a2e7478a9f4d57ea1a63f189faead59cda20d93e996ba594abb9d4abf6` | `ad4d8cdf183c9cff37d5ef8c597a83e28d232714` | MATCH |
| `src/finalize_r0.py` | `13047551928b20b5b205e2aff095dcfc4ee548a9f07fb1a13b8e4eaf3091460d` | `2c36306703764b5c5067d8ae8997887e63041039` | MATCH |
| `src/fork_gradients.py` | `4b535626ade944940e4e1f63da54ac1b48d6527ec4b3b5e50c2da48d44b21a44` | `cc38bedac8e759d454f8304378547b3c2f624e4a` | MATCH |
| `src/frozen_comparison.py` | `28708f8df09bc7544371dcb987f83afbcdad3289ffb06ab1012d27a01fc32a13` | `00d4d3a38a71c96b969eba6cdd53ffc33baec0db` | MATCH |
| `src/frozen_metrics.py` | `d1fd06c57f3d9f5eec53adab3c5cfc1ce6ee0a29d3299efa57a43c124e38f139` | `6361789fff04aa1f2c0bedb3d412056f350192b3` | MATCH |
| `src/frozen_report.py` | `1d12cac7ad00f7f42ee47ed10cfde2bdc34ec6dd002e37c9c355cc0650d9eb9c` | `5d9912e4aa53381e0d7d0717e75516b5efbc9143` | MATCH |
| `src/frozen_runtime.py` | `d79bac6459cef728954c893a40149c655a075b44a2c3ce2ed852bfdb4415bfc6` | `b94252654b84fa0657736efceb5a70009ecd1dcc` | MATCH |
| `src/generate_worlds.py` | `1741bf1443c56d0016e0134e7aa1b5e94f9388a64ad0c5dc7999db721284b4b0` | `8a12284dc1d1b34850449d9b4505cf243417f8b8` | MATCH |
| `src/group_support_audit.py` | `e26dcedf9ed31a7bf521acca7813fbf647838a800e0eb893d470dac065362035` | `18766a4a1389dfdc10ce12e44bd8e2c2bc66d81e` | MATCH |
| `src/grpo_update.py` | `54e47aeb6fb46fc957180d51e5060ba2155a3748f11ef0b0e6b3c20baecc677c` | `f0952906b382277148213ac5286dfc87e689c019` | MATCH |
| `src/joint_advantage_reference.py` | `09f0121a0966e815593ac01e19c50120a9ec56f92cdb8b2dfaf4d4503f7317ac` | `0329aaf5a09eca548450e6c23639df22cb4685bd` | MATCH |
| `src/legacy_frozen.py` | `237b58eb68683d75eb42804871289cd75105a092c086fe31a1ed1ab71bee260c` | `01ed598a5d0565c02aa960156b4a12514590ce04` | MATCH |
| `src/likelihood.py` | `87499716e3f32d6b769b320b6b948e6898043d9775ad7d5698beb7ed8085203d` | `97ba743f925deb54169b96a79b9907966b7576e4` | MATCH |
| `src/model_adapters/__init__.py` | `052ccfef958643975b67e2ba08c131381dd90c7142f46d3de329d453849e02ca` | `72854cb805306e9dd3cc0188a068ec067ab37520` | MATCH |
| `src/model_adapters/base.py` | `ce229251b732e0e961271da51e8442599d198a691070ac773e6a36e6c9056c47` | `56f6c658034fddb7d46d30593000be4f3c1be6b6` | MATCH |
| `src/model_adapters/qwen25vl.py` | `253d92a71e80b1e44ebd5c21629d6953e89a9dea01c953e7c1990d29db92bcda` | `5beb8761cf289772176ff71472c3c3a8520fd9bb` | MATCH |
| `src/model_adapters/qwen35.py` | `c427c06aaac15374394839374e3535d2773b49942226c7a3b9d70338b48de7af` | `5edb238bd92da32d214258da70947bd4198bfa8c` | MATCH |
| `src/model_adapters/qwen3vl.py` | `81c1aaa578c5e60c01bafcb4e490ca78d3f50e51314f1d4798f7bf05a629a3f3` | `3336d9289304a672f4d706025cbe961fa240b4e6` | MATCH |
| `src/next_stage_common.py` | `28eedc29c8c8ca60d98cec3a25b115e84300d4c6426677d10e4062f3ca503f74` | `dbb8fa1471a62e5e9a6057e39d328cb42da28670` | MATCH |
| `src/next_stage_preflight.py` | `f22afd1cdf062ebb25586e701c3bce09599c338aa9874dcb2d377184bbf4f9ac` | `e90655c60c804ab63d65e7014592d6d498ffaabc` | MATCH |
| `src/next_stage_runtime.py` | `c57ca321c83de210881930fb89e1e39acd3b528ada0342297504ce1853f86a16` | `8a42db75d20436fff666d0596c4511b8cf42e7ec` | MATCH |
| `src/optimizer_fork.py` | `fde42487ff532577bc841d686b8b80ae129b2035494d37fc4e8ed61f4640e08a` | `75fc88e1da8d4d927bd317fe7adb712965947493` | MATCH |
| `src/prompts.py` | `40a68267cde748fbb52624a8f91d19e108b469d2ab89d9dbdba60e4bd5bc318a` | `70ce44df53d8ae83f71b12aa7eb0a3288e78c6fe` | MATCH |
| `src/r1_parity.py` | `81978d04e2731527c73d5fb4bc83969fe5af2c888f236c67e41de3c6124dda62` | `000801b0e3535bb048f665869d25dbad3e4f963a` | MATCH |
| `src/r1_reference_smoke.py` | `1f1779a34955dde154974c82dcb4f8f633c8b544038cf83805bf5edf9e8a9e7c` | `35a240955f6839177f1757b8db2a9634a61ab11c` | MATCH |
| `src/r1_supplement.py` | `75cd568bea48732233718c977345b8fd4c1fc72477dfdf252336a350d0a3870f` | `b10b74db1ba92f229e040e6afefa3eb044137ba9` | MATCH |
| `src/r2_inputs.py` | `a62c2678397890345b83bbf066476a4b8d77fe1494ce610bddcf37052bd4293d` | `bdb681eed085dda8d815acc213c599cb9cb91fa7` | MATCH |
| `src/r2_result_audit.py` | `66c5f674187b01fa1e5f84cedeea8290f36ed4c501b102a44e9738cfecc155fc` | `851b5bc5569f96c283b0d60f1e68558b2e3b5ba9` | MATCH |
| `src/r2_runtime.py` | `9d6d28457a379bbcc170feaa57485ae5290858238d16d3086e018761121b3e7a` | `4bb355b2b5ba167703ac67d9c994609837945380` | MATCH |
| `src/r3_gate.py` | `f7570a0d499978083b2506e631148aa050c4f6b9742e9fa58c63fe84fe72cf14` | `47b5b97ab537474f00b09b66c06c071172434211` | MATCH |
| `src/r3_inputs.py` | `3954c8848cd71df9fe8b201eb79d3e287c774c7c562a46c19ecb887fcd0b5e89` | `5d4a8d814ce9d95b6eae802e970a11efbc477dc8` | MATCH |
| `src/r3_report_gate.py` | `3c7310ffdbd8bf9f3b9e58355f3d797e6a912956b59434f033baba4a26978606` | `85d927e1a8ed39ed9628a933ed5ef8ea9fa6a267` | MATCH |
| `src/r3_response.py` | `d84dd3d5c2fa555b253536ef130817be1d96f21a426cea61163d8520a4d075f0` | `0c41077b146ac59161e9a700055f25872fc8aacd` | MATCH |
| `src/r3_runtime.py` | `689aae702af070c85fef56d475a37fccf7726f89377afc2cbaae760c57a0e704` | `447d64d7b3c399284272c905768bba421b3a6b53` | MATCH |
| `src/r3_updates.py` | `748dbc5c1820fb21a223048a3a57e82cf74df8f87d90e19a7b710852f5730279` | `534a428739059acd2dee49a3ef8e4ee820ba58ff` | MATCH |
| `src/r3_warm_response.py` | `e16418aa39b142553552cd6f26c252fc02c99af4eb38483adc1d0f46c1e1ef9f` | `b03fd9f672d3e122ada9d1d025082159971e3153` | MATCH |
| `src/r3_warm_runtime.py` | `976c10ef878cc4da63c5c4b09c3175693717af71bad510bb50437dc3b1a9411f` | `7e90c189fae716ad356128ad6924bbabdae3bdc8` | MATCH |
| `src/r4_continuation.py` | `9f57ff16f0ccfea4bb2f8cf0817e0f37a0dbd08f60383f14c37a0b317c44ccba` | `f37a8fe9418e4b3712f7783c2b1ce0c3ecc8c74c` | MATCH |
| `src/r4_gate.py` | `2672039f55eebe7cd90cfa4df9708db56d8fd83001f62834aba140c5ec8743f6` | `26eb38fd88c25de348b342bb4e37f43eb2942f53` | MATCH |
| `src/r4_inputs.py` | `f6ca4667a8e79a83dca4214515acd91e3df0c765769d370edcfe0eda01825a5a` | `861842b15d35300c179cb2b612623ff904b75863` | MATCH |
| `src/r4_metrics.py` | `5632b5d3d7355d15bb1bf163e095432d0ccba488361401427b954edfdd267ea3` | `2270f1d657eca5f9bafdd87692111d81c29fb329` | MATCH |
| `src/r4_report_gate.py` | `f39da9bf89a4040b012fb149f23b6b05b4304ce2fa58023dadd6c5dedf85fce6` | `43fe169ac1f69b8929f7a4b230c74be69c5af129` | MATCH |
| `src/r4_runtime.py` | `f691dc69309b8c7b1b226671dd880285029964b66172fb8fd3f416c9bce2369b` | `d4e0c19b3136f32f4014d6b0da92559d3816989a` | MATCH |
| `src/render_charts.py` | `610644abe443279f4908a91cddc1b69396b8402243f49082190cf3fb7ca45b98` | `8a5c15abfa4d8d1b68079562977d960ddef3af7d` | MATCH |
| `src/report.py` | `97b7c35c9b7055d8280ac53adb4e09454bfa99b1b1ca4d9cefaa2cb424573b94` | `f3665706995ea7c9c3782ad495bef5aec5997c25` | MATCH |
| `src/report_next_stage.py` | `fa9fa6015f2622d0e2b5f73f8f6fccb59cb6b1ccbf9f8857e888ea85776600fc` | `3636772ff1e997b482cb9924395daae85b71e567` | MATCH |
| `src/rollout.py` | `112542fe940c5527ee7cc4144ce98baf02ec4768edd1300bcee297f576593db4` | `71e1eb4d0ba204f00f054140dbe790fbc183724f` | MATCH |
| `src/sensitivity.py` | `9fe51c37083052e53072cc68a8ca446fbda8bba5baa4af5d28bb26a072da3ce6` | `8e450763d7e20298d69d1c4696602d62e29ed6e9` | MATCH |
| `src/smoke_runtime.py` | `b48ee46f204ab8db936787b0ad8556615ae6f79974441133f02d21cf9d48b915` | `cf02c700a1e9d5199bd3bf5b21eeb5fd3b67a844` | MATCH |
| `src/statistics.py` | `b2ee5bbe2eec15fbf93273a7f70229792cbaff766232b1ad76e888fadbc8bb60` | `31fdde1e8a6aca4a6070b0eb5197ce311a43d552` | MATCH |
| `src/statistics_scene_cluster.py` | `226b36d9e265fdeecbec89122c7caec4fac410e4625df0f6601bd8757a152a13` | `c2542e42e32c2face2500e9f049861eb863c15a1` | MATCH |
| `src/tabular_flow.py` | `83b9cd0fba92c518b348df545e077fc8947908caaace3551a9803d71698b8a10` | `1086a9121a2e30e895fc9028cc7bb5abe4c0d844` | MATCH |
| `src/train_two_arm_pilot.py` | `da0b1d5701cfffe16d44a0c84f2630e037fdd6d8e19eb7b747781265729c669f` | `ead62c2b368e10f9b545ebc764c5b40f1a0b4de8` | MATCH |
| `src/verifiers.py` | `03e3eb703707262f17b3e2581c450a05ce8200d7d169788bcf4834cbf39b1147` | `420ae684523d0552491928cf06e4773180b27c9c` | MATCH |

已分别只读核对旧结果目录与 ZIP 中的三份固定 SHA-256 JSON；两种来源一致。另读取 warm gradients_summary.parquet 的 60 条候选记录，并逐项匹配 candidate manifest 中的 checkpoint/parameter/optimizer/state hash 元数据；读取 R4 runtime lock、continuation decision 和 learning_curves.csv，核对两组各 64 个 step、模型与 seed/B/K。

R4 continuation 的 `decision_sha256` 按实际 `src/r4_continuation.py:369` 定义为 `canonical_hash(decision)`。复算值 `da335d76bcbcc2fc42f4a12ebf9527da833b971b7ccf9f2577a3c44d73e001c3` 与 lock 相同；对应文件字节 SHA-256 是 `babc5264d7ea6d44bc23dd618db36d8dba53dc64e5f80c6435fef56280c402d0`，两种摘要不同不表示绑定失败。

本地 compact 包缺少 warm origin.pt 与 samples.jsonl，未核验候选 checkpoint、原始 rollout、数据/图片与 prepared tensors。真实原件门禁保持 `BLOCKED_MISSING_PARENT_RAW`；raw_tensors_verified、gpu_smoke_passed、gpu_started、training_started 均为 false。

复现本地扩展审计：在仓库根目录执行 `.venv/bin/python ssvc_flow/runs/mechanism_followup_cpu/parent_inputs/write_s0_audit.py`。结果保存在 `ssvc_flow/runs/mechanism_followup_cpu/s0/`；该目录记录本机检查，不是服务器实验结果。
