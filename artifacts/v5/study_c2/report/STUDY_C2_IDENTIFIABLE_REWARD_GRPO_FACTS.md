# Study C2 Identifiable Reward GRPO Facts

```json
{
  "gpu_invoked": false,
  "optimizer_step_invoked": false,
  "report_scope": "registered_stage23_through_stage26_facts_only",
  "rl_invoked": false,
  "schema_version": 2,
  "source_sha256": {
    "artifacts/v5/study_c2/evaluation/manifest.json": "c98f03a9b23d5f424e756280b434d805cce59ee477cb967672d3c70f930e721b",
    "artifacts/v5/study_c2/evaluation/raw_rows.jsonl": "2e684df3cbaf20feda9fa876dd7003565c372a183d12fb44a31ac68c4457fa4f",
    "artifacts/v5/study_c2/evaluation/summary.json": "dc168b37ea855f0afa813a611d5b30088f81a00d8c176b62e3c8cdc5da605508",
    "artifacts/v5/study_c2/frozen_policy_support/manifest.json": "aff10e9d31b7275fcfee815065de86ac35c70bfd4526f6dc9cce0d3f756eef3a",
    "artifacts/v5/study_c2/frozen_policy_support/raw_rows.jsonl": "954a6133d013ee6fa324d09a44aa82d769ef30a89d6d700f471e8f7f00810274",
    "artifacts/v5/study_c2/frozen_policy_support/summary.json": "44c86d517edd2ef5ca1cb558e2730c00f84b2a3e7851c473baddd810aa40cca1",
    "artifacts/v5/study_c2/shared_gradient_audit/manifest.json": "8856eb4d8e841ba3965b7745b28e6ac0f03bca2ccdd396e91fc99dcf8048dc4a",
    "artifacts/v5/study_c2/shared_gradient_audit/per_group.jsonl": "1a9fdef6981137f9748156799762b02dac2fe8568473de69c7f257e36d5d69bc",
    "artifacts/v5/study_c2/shared_gradient_audit/summary.json": "8e652de6f9ecad994e5fb4d41856109c4ddb86fea96cba8020e7cef1540de689",
    "artifacts/v5/study_c2/stage24_execution_contract.json": "b91834069e821a47fdcad7834db6df367ccf13bb41002a8618aa4e7c34f5efaa",
    "artifacts/v5/study_c2/stage25_execution_contract.json": "f8c2530d0594e0b6671070f3c44c64fd4bb07c2ba3e97dce0f8fd1f7bde55f7a",
    "artifacts/v5/study_c2/training/C2_answer_reward/arm_config.json": "ffa9d3497d988bd9c19bf8cd6087d2c8ee76b27c833c6e82158ca585292b2b47",
    "artifacts/v5/study_c2/training/C2_answer_reward/group_diagnostics.jsonl": "2b08f2d30e43ac6195cead4ab5c61882058a16e7a92d361559b5f926d10972b7",
    "artifacts/v5/study_c2/training/C2_answer_reward/manifest.json": "d0d1746c5ca7cd33ea1d66fe41204e1321867d7be549de86eefa0527e59dce16",
    "artifacts/v5/study_c2/training/C2_answer_reward/raw_reward_trace.jsonl": "dc4492915f08c05bc6a84066913806a92a59d01508114c784fe3e713ea64f689",
    "artifacts/v5/study_c2/training/C2_answer_reward/summary.json": "5840f317d984ef3c92aef8b04a44eb80a76414c069142aa9f1f870e91e570527",
    "artifacts/v5/study_c2/training/C2_answer_reward/trainer_log_history.json": "a6fa230aa63f50f35b52b053f76eacf48cc21be73d7b73d4dd9ded26b4000436",
    "artifacts/v5/study_c2/training/C2_exact_state_reward/arm_config.json": "723880ff0a8f97089840e14834b45e725c973cb7bb777d160d8a763f274a92f6",
    "artifacts/v5/study_c2/training/C2_exact_state_reward/group_diagnostics.jsonl": "a52c098c07b98f7b52b64b8598776a0ca924df767399cb55ce8da30603db96a9",
    "artifacts/v5/study_c2/training/C2_exact_state_reward/manifest.json": "02467193983717d35f12c4703cde2c3ab14eaa6366cf9e76f9abf6ec6f4786f2",
    "artifacts/v5/study_c2/training/C2_exact_state_reward/raw_reward_trace.jsonl": "01fb491bf2857ad800e5efb8380f6564efe54275c01b88fbca4be545580ab47e",
    "artifacts/v5/study_c2/training/C2_exact_state_reward/summary.json": "b1dc3ace55a6a89572aae5bf5a323581204892b24950782a7b21ba2dff089d1d",
    "artifacts/v5/study_c2/training/C2_exact_state_reward/trainer_log_history.json": "fb7d2fe1bb9dc1f8a62699d8782c6b2c847af69552971d338756d23b93fee66e",
    "artifacts/v5/study_c2/training/manifest.json": "4fc30748f8c9396745baafa63015354e8eec7c871cb6eafe8e0dc6a9adfd8569",
    "configs/v5/server_package_lock.yaml": "a8f351db7cadc904f6feecdd9cddb9e0d782c8356226cd866661710623b2e544",
    "configs/v5/study_c2_identifiable_reward.yaml": "03be95231d3eb8351078a77943bcce66d0d457dca3b0c3f1ab4a94d4d52cf417"
  },
  "stages": {
    "stage23": {
      "manifest": {
        "action_protocol": "anchored_first_line_world_v1",
        "b3_adapter_sha256": "863b70b420daa5267a4c1517df0200833594ade285d1bdfa7b02e5952f9cfe9b",
        "config_sha256": "03be95231d3eb8351078a77943bcce66d0d457dca3b0c3f1ab4a94d4d52cf417",
        "fiber_rows_sha256": "ebfe4dd7b2acc2afeaea7aa633dcf2f0f7af1eaaceda8af808720c087dd36719",
        "gpu_invoked": true,
        "prompt_count": 96,
        "raw_rows_sha256": "954a6133d013ee6fa324d09a44aa82d769ef30a89d6d700f471e8f7f00810274",
        "rl_invoked": false,
        "rollout_count": 6144,
        "rollouts_per_prompt": 64,
        "schema_version": 2,
        "status": "STUDY_C2_FROZEN_SUPPORT_COMPLETE",
        "stopping_rule": "newline_or_eos_with_max_16_tokens",
        "summary_sha256": "44c86d517edd2ef5ca1cb558e2730c00f84b2a3e7851c473baddd810aa40cca1",
        "training_invoked": false
      },
      "summary": {
        "counts": {
          "F": 655,
          "S": 635,
          "U": 4711,
          "X": 143
        },
        "gpu_invoked": true,
        "k_selection": {
          "efficiency_by_k": {
            "16": 0.00881163343605933,
            "32": 0.006901408819160686,
            "8": 0.008979362032536695
          },
          "selected_k": 8
        },
        "per_scene_count": 96,
        "rollout_count": 6144,
        "schema_version": 2,
        "status": "REWARD_CONTRAST_IDENTIFIED"
      }
    },
    "stage24": {
      "manifest": {
        "b3_adapter_sha256": "863b70b420daa5267a4c1517df0200833594ade285d1bdfa7b02e5952f9cfe9b",
        "config_sha256": "03be95231d3eb8351078a77943bcce66d0d457dca3b0c3f1ab4a94d4d52cf417",
        "continue_to_main_rl": true,
        "execution_contract_sha256": "b91834069e821a47fdcad7834db6df367ccf13bb41002a8618aa4e7c34f5efaa",
        "fiber_rows_sha256": "ebfe4dd7b2acc2afeaea7aa633dcf2f0f7af1eaaceda8af808720c087dd36719",
        "gpu_invoked": true,
        "gradient_definition": "sum_group_centered_advantage_times_sequence_log_probability",
        "group_count": 768,
        "group_size": 8,
        "max_completion_length": 16,
        "max_prompt_length": 512,
        "optimizer_step_invoked": false,
        "package_lock_sha256": "a8f351db7cadc904f6feecdd9cddb9e0d782c8356226cd866661710623b2e544",
        "per_group_sha256": "1a9fdef6981137f9748156799762b02dac2fe8568473de69c7f257e36d5d69bc",
        "rl_invoked": false,
        "rollout_count": 6144,
        "same_rollouts_for_both_rewards": true,
        "schema_version": 2,
        "scientific_status": "STUDY_C2_SHARED_GRADIENT_CONTRAST_IDENTIFIED",
        "status": "STUDY_C2_SHARED_GRADIENT_AUDIT_COMPLETE",
        "summary_sha256": "8e652de6f9ecad994e5fb4d41856109c4ddb86fea96cba8020e7cef1540de689",
        "support_manifest_sha256": "aff10e9d31b7275fcfee815065de86ac35c70bfd4526f6dc9cce0d3f756eef3a",
        "support_raw_rows_sha256": "954a6133d013ee6fa324d09a44aa82d769ef30a89d6d700f471e8f7f00810274",
        "support_summary_sha256": "44c86d517edd2ef5ca1cb558e2730c00f84b2a3e7851c473baddd810aa40cca1",
        "training_invoked": false
      },
      "summary": {
        "ESGR_group_count": 56,
        "RDGR_group_count": 356,
        "by_condition": {
          "collision": {
            "ESGR_group_count": 25,
            "RDGR_group_count": 198,
            "counts": {
              "F": 274,
              "S": 373,
              "U": 2354,
              "X": 71
            },
            "gradient_answer_norm_mean": 53.577053561671484,
            "gradient_cosine_mean": 0.5242795647928241,
            "gradient_difference_norm_max": 198.84986357843127,
            "gradient_difference_norm_mean": 47.8223922905396,
            "gradient_state_norm_mean": 8.038065154761087,
            "group_count": 384,
            "reward_hamming_distance": 373
          },
          "separating": {
            "ESGR_group_count": 31,
            "RDGR_group_count": 158,
            "counts": {
              "F": 381,
              "S": 262,
              "U": 2357,
              "X": 72
            },
            "gradient_answer_norm_mean": 41.73100240322696,
            "gradient_cosine_mean": 0.6374212025069926,
            "gradient_difference_norm_max": 271.4385559212199,
            "gradient_difference_norm_mean": 35.22992240466083,
            "gradient_state_norm_mean": 8.535236168451789,
            "group_count": 384,
            "reward_hamming_distance": 262
          }
        },
        "by_family": {
          "cross_series": {
            "ESGR_group_count": 20,
            "RDGR_group_count": 140,
            "counts": {
              "F": 158,
              "S": 200,
              "U": 2647,
              "X": 67
            },
            "gradient_answer_norm_mean": 38.7444567513588,
            "gradient_cosine_mean": 0.661126242123187,
            "gradient_difference_norm_max": 271.4385559212199,
            "gradient_difference_norm_mean": 32.16783669954431,
            "gradient_state_norm_mean": 8.520570660994755,
            "group_count": 384,
            "reward_hamming_distance": 200
          },
          "trend": {
            "ESGR_group_count": 36,
            "RDGR_group_count": 216,
            "counts": {
              "F": 497,
              "S": 435,
              "U": 2064,
              "X": 76
            },
            "gradient_answer_norm_mean": 56.563599213539646,
            "gradient_cosine_mean": 0.5005745251766298,
            "gradient_difference_norm_max": 183.1438201644181,
            "gradient_difference_norm_mean": 50.88447799565611,
            "gradient_state_norm_mean": 8.05273066221812,
            "group_count": 384,
            "reward_hamming_distance": 435
          }
        },
        "continue_to_main_rl": true,
        "counts": {
          "F": 655,
          "S": 635,
          "U": 4711,
          "X": 143
        },
        "gpu_invoked": true,
        "gradient_answer_norm_mean": 47.65402798244923,
        "gradient_cosine_mean": 0.5808503836499084,
        "gradient_difference_norm_max": 271.4385559212199,
        "gradient_difference_norm_mean": 41.52615734760021,
        "gradient_state_norm_mean": 8.286650661606439,
        "group_count": 768,
        "group_size": 8,
        "optimizer_step_invoked": false,
        "reward_hamming_distance": 635,
        "reward_hamming_rate": 0.10335286458333333,
        "rl_invoked": false,
        "rollout_count": 6144,
        "schema_version": 2,
        "status": "STUDY_C2_SHARED_GRADIENT_CONTRAST_IDENTIFIED",
        "training_invoked": false,
        "zero_tolerance": 1e-12
      }
    },
    "stage25": {
      "arms": {
        "C2_answer_reward": {
          "manifest": {
            "action_protocol": "anchored_first_line_world_v1",
            "arm": "C2_answer_reward",
            "arm_config_sha256": "ffa9d3497d988bd9c19bf8cd6087d2c8ee76b27c833c6e82158ca585292b2b47",
            "b3_adapter_sha256": "863b70b420daa5267a4c1517df0200833594ade285d1bdfa7b02e5952f9cfe9b",
            "backend": {
              "first_line_generation_override": true,
              "grpo_config_required_fields": true,
              "grpo_trainer_required_fields": true,
              "reference_adapter_copy": true
            },
            "checkpoint_steps": [
              48,
              96,
              144,
              192
            ],
            "config_sha256": "03be95231d3eb8351078a77943bcce66d0d457dca3b0c3f1ab4a94d4d52cf417",
            "execution_contract_sha256": "f8c2530d0594e0b6671070f3c44c64fd4bb07c2ba3e97dce0f8fd1f7bde55f7a",
            "expected_optimizer_steps": 192,
            "fiber_rows_sha256": "ebfe4dd7b2acc2afeaea7aa633dcf2f0f7af1eaaceda8af808720c087dd36719",
            "final_adapter_sha256": "ed8faebdf5dd43f693dc8f11ec247a11a346978a37f85d1994919b733ce6fa98",
            "gpu_invoked": true,
            "group_diagnostics_sha256": "2b08f2d30e43ac6195cead4ab5c61882058a16e7a92d361559b5f926d10972b7",
            "group_size": 8,
            "matched_pair_count": 96,
            "model_snapshot_sha256": "e104df572eab7267bc2a63c11d70f7c8b1ebf8f85aa835d17e2c2641447bca87",
            "optimizer_step_invoked": true,
            "package_lock_sha256": "a8f351db7cadc904f6feecdd9cddb9e0d782c8356226cd866661710623b2e544",
            "raw_reward_trace_sha256": "dc4492915f08c05bc6a84066913806a92a59d01508114c784fe3e713ea64f689",
            "reference_initialization": "frozen_copy_of_B3_adapter",
            "resumed_from_checkpoint": null,
            "reward_function_id": "answer_reward_v1",
            "reward_only_pair_verified": true,
            "rl_invoked": true,
            "schema_version": 2,
            "stage24_manifest_sha256": "8856eb4d8e841ba3965b7745b28e6ac0f03bca2ccdd396e91fc99dcf8048dc4a",
            "stage24_per_group_sha256": "1a9fdef6981137f9748156799762b02dac2fe8568473de69c7f257e36d5d69bc",
            "stage24_summary_sha256": "8e652de6f9ecad994e5fb4d41856109c4ddb86fea96cba8020e7cef1540de689",
            "status": "STUDY_C2_ARM_TRAINING_COMPLETE",
            "stopping_rule": "newline_or_eos_with_max_16_tokens",
            "summary_sha256": "5840f317d984ef3c92aef8b04a44eb80a76414c069142aa9f1f870e91e570527",
            "trainer_log_sha256": "a6fa230aa63f50f35b52b053f76eacf48cc21be73d7b73d4dd9ded26b4000436",
            "training_invoked": true,
            "training_prompt_count": 192
          },
          "summary": {
            "arm": "C2_answer_reward",
            "checkpoint_steps": [
              48,
              96,
              144,
              192
            ],
            "counts": {
              "F": 416,
              "S": 470,
              "U": 366,
              "X": 284
            },
            "epochs": 1,
            "group_size": 8,
            "optimizer_steps": 192,
            "reward_function_id": "answer_reward_v1",
            "reward_hamming_distance": 470,
            "rollout_count": 1536,
            "schema_version": 2,
            "status": "STUDY_C2_ARM_TRAINING_SUMMARIZED",
            "training_prompt_count": 192
          }
        },
        "C2_exact_state_reward": {
          "manifest": {
            "action_protocol": "anchored_first_line_world_v1",
            "arm": "C2_exact_state_reward",
            "arm_config_sha256": "723880ff0a8f97089840e14834b45e725c973cb7bb777d160d8a763f274a92f6",
            "b3_adapter_sha256": "863b70b420daa5267a4c1517df0200833594ade285d1bdfa7b02e5952f9cfe9b",
            "backend": {
              "first_line_generation_override": true,
              "grpo_config_required_fields": true,
              "grpo_trainer_required_fields": true,
              "reference_adapter_copy": true
            },
            "checkpoint_steps": [
              48,
              96,
              144,
              192
            ],
            "config_sha256": "03be95231d3eb8351078a77943bcce66d0d457dca3b0c3f1ab4a94d4d52cf417",
            "execution_contract_sha256": "f8c2530d0594e0b6671070f3c44c64fd4bb07c2ba3e97dce0f8fd1f7bde55f7a",
            "expected_optimizer_steps": 192,
            "fiber_rows_sha256": "ebfe4dd7b2acc2afeaea7aa633dcf2f0f7af1eaaceda8af808720c087dd36719",
            "final_adapter_sha256": "324d9782901868332fbcfb8529ee91fdd8087204771c70e74d544f76b38636a7",
            "gpu_invoked": true,
            "group_diagnostics_sha256": "a52c098c07b98f7b52b64b8598776a0ca924df767399cb55ce8da30603db96a9",
            "group_size": 8,
            "matched_pair_count": 96,
            "model_snapshot_sha256": "e104df572eab7267bc2a63c11d70f7c8b1ebf8f85aa835d17e2c2641447bca87",
            "optimizer_step_invoked": true,
            "package_lock_sha256": "a8f351db7cadc904f6feecdd9cddb9e0d782c8356226cd866661710623b2e544",
            "raw_reward_trace_sha256": "01fb491bf2857ad800e5efb8380f6564efe54275c01b88fbca4be545580ab47e",
            "reference_initialization": "frozen_copy_of_B3_adapter",
            "resumed_from_checkpoint": null,
            "reward_function_id": "exact_state_reward_v1",
            "reward_only_pair_verified": true,
            "rl_invoked": true,
            "schema_version": 2,
            "stage24_manifest_sha256": "8856eb4d8e841ba3965b7745b28e6ac0f03bca2ccdd396e91fc99dcf8048dc4a",
            "stage24_per_group_sha256": "1a9fdef6981137f9748156799762b02dac2fe8568473de69c7f257e36d5d69bc",
            "stage24_summary_sha256": "8e652de6f9ecad994e5fb4d41856109c4ddb86fea96cba8020e7cef1540de689",
            "status": "STUDY_C2_ARM_TRAINING_COMPLETE",
            "stopping_rule": "newline_or_eos_with_max_16_tokens",
            "summary_sha256": "b1dc3ace55a6a89572aae5bf5a323581204892b24950782a7b21ba2dff089d1d",
            "trainer_log_sha256": "fb7d2fe1bb9dc1f8a62699d8782c6b2c847af69552971d338756d23b93fee66e",
            "training_invoked": true,
            "training_prompt_count": 192
          },
          "summary": {
            "arm": "C2_exact_state_reward",
            "checkpoint_steps": [
              48,
              96,
              144,
              192
            ],
            "counts": {
              "F": 350,
              "S": 376,
              "U": 439,
              "X": 371
            },
            "epochs": 1,
            "group_size": 8,
            "optimizer_steps": 192,
            "reward_function_id": "exact_state_reward_v1",
            "reward_hamming_distance": 376,
            "rollout_count": 1536,
            "schema_version": 2,
            "status": "STUDY_C2_ARM_TRAINING_SUMMARIZED",
            "training_prompt_count": 192
          }
        }
      },
      "pair_manifest": {
        "arms": {
          "C2_answer_reward": {
            "final_adapter_sha256": "ed8faebdf5dd43f693dc8f11ec247a11a346978a37f85d1994919b733ce6fa98",
            "manifest_sha256": "d0d1746c5ca7cd33ea1d66fe41204e1321867d7be549de86eefa0527e59dce16",
            "raw_reward_trace_sha256": "dc4492915f08c05bc6a84066913806a92a59d01508114c784fe3e713ea64f689"
          },
          "C2_exact_state_reward": {
            "final_adapter_sha256": "324d9782901868332fbcfb8529ee91fdd8087204771c70e74d544f76b38636a7",
            "manifest_sha256": "02467193983717d35f12c4703cde2c3ab14eaa6366cf9e76f9abf6ec6f4786f2",
            "raw_reward_trace_sha256": "01fb491bf2857ad800e5efb8380f6564efe54275c01b88fbca4be545580ab47e"
          }
        },
        "gpu_invoked": true,
        "optimizer_steps_per_arm": 192,
        "reward_only_pair_verified": true,
        "rl_invoked": true,
        "schema_version": 2,
        "status": "STUDY_C2_TWO_ARM_TRAINING_COMPLETE",
        "training_invoked": true,
        "training_prompt_count_per_arm": 192
      }
    },
    "stage26": {
      "manifest": {
        "arm_manifests": {
          "C2_answer_reward": {
            "final_adapter_sha256": "ed8faebdf5dd43f693dc8f11ec247a11a346978a37f85d1994919b733ce6fa98",
            "manifest_sha256": "d0d1746c5ca7cd33ea1d66fe41204e1321867d7be549de86eefa0527e59dce16"
          },
          "C2_exact_state_reward": {
            "final_adapter_sha256": "324d9782901868332fbcfb8529ee91fdd8087204771c70e74d544f76b38636a7",
            "manifest_sha256": "02467193983717d35f12c4703cde2c3ab14eaa6366cf9e76f9abf6ec6f4786f2"
          }
        },
        "b3_adapter_sha256": "863b70b420daa5267a4c1517df0200833594ade285d1bdfa7b02e5952f9cfe9b",
        "config_sha256": "03be95231d3eb8351078a77943bcce66d0d457dca3b0c3f1ab4a94d4d52cf417",
        "evaluation_pair_count": 88,
        "evaluation_scene_count": 176,
        "fiber_rows_sha256": "ebfe4dd7b2acc2afeaea7aa633dcf2f0f7af1eaaceda8af808720c087dd36719",
        "gpu_invoked": true,
        "model_snapshot_sha256": "e104df572eab7267bc2a63c11d70f7c8b1ebf8f85aa835d17e2c2641447bca87",
        "optimizer_step_invoked": false,
        "package_lock_sha256": "a8f351db7cadc904f6feecdd9cddb9e0d782c8356226cd866661710623b2e544",
        "raw_row_count": 5632,
        "raw_rows_sha256": "2e684df3cbaf20feda9fa876dd7003565c372a183d12fb44a31ac68c4457fa4f",
        "reward_only_pair_verified": true,
        "rl_invoked": false,
        "rollout_seed_algorithm": "phase5_rollout_seed_v1",
        "rollout_seed_base": 2026082401,
        "sampled_rollouts": 16,
        "schema_version": 2,
        "status": "STUDY_C2_POST_TRAINING_EVALUATION_COMPLETE",
        "summary_sha256": "dc168b37ea855f0afa813a611d5b30088f81a00d8c176b62e3c8cdc5da605508",
        "training_invoked": false,
        "training_pair_manifest_sha256": "4fc30748f8c9396745baafa63015354e8eec7c871cb6eafe8e0dc6a9adfd8569"
      },
      "summary": {
        "action_protocol": "anchored_first_line_world_v1",
        "b3_adapter_sha256": "863b70b420daa5267a4c1517df0200833594ade285d1bdfa7b02e5952f9cfe9b",
        "by_arm": {
          "C2_answer_reward": {
            "answer_mean": 0.42329545454545453,
            "exact_mean": 0.11079545454545454,
            "parse_rate_mean": 0.6977982954545454,
            "scene_count": 176
          },
          "C2_exact_state_reward": {
            "answer_mean": 0.3671875,
            "exact_mean": 0.1629971590909091,
            "parse_rate_mean": 0.5625,
            "scene_count": 176
          }
        },
        "config_sha256": "03be95231d3eb8351078a77943bcce66d0d457dca3b0c3f1ab4a94d4d52cf417",
        "evaluation_pair_count": 88,
        "evaluation_scene_count": 176,
        "fiber_rows_sha256": "ebfe4dd7b2acc2afeaea7aa633dcf2f0f7af1eaaceda8af808720c087dd36719",
        "gpu_invoked": true,
        "model_snapshot_sha256": "e104df572eab7267bc2a63c11d70f7c8b1ebf8f85aa835d17e2c2641447bca87",
        "optimizer_step_invoked": false,
        "package_lock_sha256": "a8f351db7cadc904f6feecdd9cddb9e0d782c8356226cd866661710623b2e544",
        "pair_bootstrap": {
          "bootstrap_95_ci": [
            -0.032670454545454544,
            0.014204545454545454
          ],
          "bootstrap_resamples": 10000,
          "bootstrap_seed": 2026082403,
          "estimate": -0.009232954545454546,
          "pair_count": 88
        },
        "raw_row_count": 5632,
        "reward_only_pair_verified": true,
        "rl_invoked": false,
        "rollout_seed_algorithm": "phase5_rollout_seed_v1",
        "rollout_seed_base": 2026082401,
        "sampled_rollouts": 16,
        "schema_version": 2,
        "status": "STUDY_C2_POST_TRAINING_EVALUATION_COMPLETE",
        "stopping_rule": "newline_or_eos_with_max_16_tokens",
        "training_invoked": false,
        "training_pair_manifest_sha256": "4fc30748f8c9396745baafa63015354e8eec7c871cb6eafe8e0dc6a9adfd8569"
      }
    }
  },
  "status": "STUDY_C2_IDENTIFIABLE_REWARD_GRPO_FACTS_COMPLETE",
  "training_invoked": false
}
```
