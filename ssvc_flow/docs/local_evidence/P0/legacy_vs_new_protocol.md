# Legacy versus new protocol

Historical evidence is read-only. No legacy code was executed.
Archived results describe the historical experiment only; no new model/GPU result was produced.

legacy_exact_reproduction=false

| Field | L: observed legacy evidence | N: proposed new protocol |
|---|---|---|
| checkpoint_binaries | missing | "requires phase-specific verification" |
| effective_chat_template | missing | "requires phase-specific verification" |
| group_size | 8 | 8 |
| group_std_convention | missing | "sqrt(population_variance + 1e-8)" |
| initial_adapter_config | missing | "requires phase-specific verification" |
| learning_rate | 1e-06 | 1e-05 |
| loss_reduction | missing | "fixed token constant Lnorm=64, denominator B*K*64" |
| model | "Qwen2.5-VL-3B-Instruct" | "Qwen/Qwen3.5-9B" |
| package_versions | {"torch": "2.8.0+cu128", "transformers": "5.14.1", "peft": "0.19.1", "trl": "1.9.0", "accelerate": "1.14.0", "datasets": "5.0.0", "qwen-vl-utils": "0.0.14", "pyarrow": "25.0.0", "pillow": "12.3.0"} | "requires phase-specific verification" |
| parser_runtime_qualification | missing | "requires phase-specific verification" |
| parser_source | ["parse_complete_action_any_domain", "parse_p0", "parse_p1", "parse_p2"] | "strict four-key JSON, exact keys a,b,c,d, integer values" |
| prompt_data | {"prompt_count": 464} | "requires phase-specific verification" |
| prompts_per_optimizer_update | missing | 4 |
| reward_vectors | {"A_BIN": [1, 1, 0, 0], "X_BIN": [1, 0, 0, 0], "A_LEX": [3, 3, 1, 0], "X_LEX": [3, 1, 1, 0]} | {"A_BASE": [2, 2, 0, 0], "X_BASE": [2, 0, 0, 0], "A_VALID": [3, 3, 1, 0], "X_VALID": [3, 1, 1, 0]} |
| source_commit_checkout_verified | missing | "requires phase-specific verification" |
| training_logs | ["archive:study_c3_resolution_validity_evidence_20260823.tar.gz!/artifacts/v5/study_c3/training/A_BIN/raw_reward_trace.jsonl", "archive:study_c3_resolution_validity_evidence_20260823.tar.gz!/artifacts/v5/study_c3/training/A_BIN/trainer_log_history.json", "archive:study_c3_resolution_validity_evidence_20260823.tar.gz!/artifacts/v5/study_c3/training/A_LEX/raw_reward_trace.jsonl", "archive:study_c3_resolution_validity_evidence_20260823.tar.gz!/artifacts/v5/study_c3/training/A_LEX/trainer_log_history.json", "archive:study_c3_resolution_validity_evidence_20260823.tar.gz!/artifacts/v5/study_c3/training/X_BIN/raw_reward_trace.jsonl", "archive:study_c3_resolution_validity_evidence_20260823.tar.gz!/artifacts/v5/study_c3/training/X_BIN/trainer_log_history.json", "archive:study_c3_resolution_validity_evidence_20260823.tar.gz!/artifacts/v5/study_c3/training/X_LEX/raw_reward_trace.jsonl", "archive:study_c3_resolution_validity_evidence_20260823.tar.gz!/artifacts/v5/study_c3/training/X_LEX/trainer_log_history.json"] | "requires phase-specific verification" |
| training_seeds | [2026082501] | [17] |

Missing exact-reproduction evidence: checkpoint_binaries, effective_chat_template, group_std_convention, initial_adapter_config, loss_reduction, parser_runtime_qualification, prompts_per_optimizer_update, source_commit_checkout_verified

Source hashes and field/function provenance are recorded in audit.json. Source files alone do not verify the archived checkout or effective library defaults.

Observed data: 464 rows; splits {"dev": 48, "positive_control": 32, "support_audit": 96, "test": 96, "train": 192}; evaluation 176 scenes / 88 pairs; two-condition pairing verified=true.

Observed data: 464 rows; splits {"dev": 48, "positive_control": 32, "support_audit": 96, "test": 96, "train": 192}; evaluation 176 scenes / 88 pairs; two-condition pairing verified=true.
