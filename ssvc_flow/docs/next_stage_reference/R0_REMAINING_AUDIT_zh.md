# R0 剩余机器审计运行说明

`src/audit_r0_remaining.py` 完成不生成新模型输出的剩余审计：

- `decode_raw_completion` 只移除末尾真实 EOS token；不使用 `skip_special_tokens`，因此 `<think>`、额外文本、截断和非法 JSON 保持可见。
- `audit_cross_split` 跟随数据目录中的 JSONL，按 `base_scene_id` 与 `truth_structure_hash` 检查跨 split 重复，并记录每个 split 的行数、字节哈希。
- `select_36` 按固定 opaque `base_scene_id` 选取 family×chart×operation 每格两个场景；`compare_contact_manifest` 逐字段比较既有联系图清单，只有 36 个 ID、图片哈希、路径和分层完全相同时才允许复用人工审阅。
- `audit_images` 验证图片路径边界、PNG SHA-256 和原始 768×512 尺寸。
- `audit_symbolic_prompts` 用当前版本 prompt renderer 检查 SYMBOLIC 输入没有图片路径、truth JSON 或 R2 oracle 标签。

示例（CPU 检查）：

```bash
cd ssvc_flow
PYTHONPATH=. ../.venv/bin/python -m src.audit_r0_remaining \
  --dataset data/generated \
  --contact-manifest docs/local_evidence/P0/contact_sheets_manifest.json \
  --out runs/NEXT_20260909/R0_remaining_local
```

提供 `--samples` 和服务器缓存的 `--tokenizer` 时，会对每条原始 P3 `token_ids` 解码并与 `raw_completion` 比较。缺少 tokenizer/processor 时结果保持 `PENDING_*`，不会伪造 PASS。processor 的实测尺寸、`image_grid_thw` 与视觉 token 需在锁定 Qwen3.5 processor 上运行后才可决定 R0 最终状态。

截至本地数据副本：跨 split、36 图哈希与尺寸、SYMBOLIC prompt 均通过；tokenizer 与 processor 仍待服务器 runtime 证据。既有 `/reviews/P0_human_review_20260908.json` 与相同 SHA 绑定的 36 图清单可复用，模块不会替代或伪造人工可视确认。
