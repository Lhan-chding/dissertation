# Part 3 independent agent audit

This is an independent agent review, not human double-blind review. All 24 fixed raw responses were actually read in three displayed batches before PRIVATE or FROZEN_SCORES was opened. Only this part3 directory was written. No production scorer or measurement code was read or called.

## Preserved sequence

1. Read docs/04_评分与审核.md and all BLIND raw text; freeze EXTRACTION.jsonl and STAGE1_RECEIPT.json.
2. Open PRIVATE; independently rank Alpha by question direction with printed-position ties, select k, bind Beta, and compute exact Fraction sum/mean. All 24 supplied gold answers, sets, indices and operand mappings agree. Freeze INDEPENDENT_FINDINGS.jsonl and STAGE2_RECEIPT.json.
3. Open FROZEN_SCORES and compare every requested metric for every row. Preserve initial disagreement, then append adjudication without overwriting independent files.

## Findings

- 6 strict complete JSON objects; 18 invalid JSON responses (15 missing final closing brace, 1 illegal ** marker, 1 length-truncated repetitive evidence array, 1 malformed work string).
- A: 1 correct, 23 incorrect/unparseable. S: 1 PASS, 2 FAIL, 21 UNKNOWN.
- Initial B draft: 1 pass and 23 zero. Comparison revealed 20 B disagreements: 18 invalid JSON and 2 missing bindings are unavailable, so B is UNKNOWN/null. This was an audit-side mistake; no producer defect was demonstrated. The original zero judgments and reasons remain in INDEPENDENT_FINDINGS.jsonl.
- Adjudicated B: 1 pass, 3 fail, 20 UNKNOWN. ADJUDICATED_FINDINGS.jsonl records only this correction.
- A, S, S_hit, rank_order_known, values_true, print order, alphabetical order, numerical order, compliance, critical tie-block order and forced-evidence status agree on all 24 rows. Six arrays have known order and true Alpha values; all six are forced prefix evidence. No valid C_PRINT/C_ALT row is present here.
- order_compliance remains null for these conditions; no compliance rate is inferred.
- Extraction complete means syntactically complete JSON plus non-length stop. Frozen complete instead records termination status: an eos response may still be malformed. These metadata concepts are distinct and were not silently substituted.

## Per-row factual basis

| Row | Request | Derived answer | A | S | B adjudicated | Evidence order known |
|---|---|---:|---:|---|---|---|
| 1 | MAIN_ORDER_0018_view0/b2/P_STABLE/d0 | 36 | 1 | PASS | 1 | True |
| 2 | MAIN_ORDER_0046_view0/b2/P_STABLE/d0 | 48 | 0 | FAIL | 0 | True |
| 3 | MAIN_ORDER_0044_view0/b2/P_REVTIE/d0 | 87/2 | 0 | UNKNOWN | UNKNOWN | False |
| 4 | MAIN_ORDER_0002_view1/b2/P_REVTIE/d0 | 44 | 0 | FAIL | 0 | True |
| 5 | MAIN_ORDER_0011_view1/b2/P_PLACEBO/d0 | 96 | 0 | UNKNOWN | UNKNOWN | True |
| 6 | MAIN_ORDER_0006_view1/b2/P_PLACEBO/d0 | 132 | 0 | UNKNOWN | UNKNOWN | False |
| 7 | MAIN_ORDER_0003_view0/b1/P_STABLE/d0 | 168 | 0 | UNKNOWN | UNKNOWN | False |
| 8 | MAIN_ORDER_0033_view1/b2/P_PLACEBO/d0 | 198 | 0 | UNKNOWN | UNKNOWN | False |
| 9 | MAIN_ORDER_0002_view1/b1/P_PLACEBO/d0 | 44 | 0 | UNKNOWN | UNKNOWN | False |
| 10 | MAIN_ORDER_0011_view0/b0/P_REVTIE/d0 | 96 | 0 | UNKNOWN | UNKNOWN | False |
| 11 | MAIN_ORDER_0043_view0/b1/P_REVTIE/d0 | 114 | 0 | UNKNOWN | UNKNOWN | False |
| 12 | MAIN_ORDER_0003_view1/b0/P_PLACEBO/d0 | 168 | 0 | UNKNOWN | UNKNOWN | False |
| 13 | MAIN_ORDER_0038_view0/b1/P_REVTIE/d0 | 36 | 0 | UNKNOWN | UNKNOWN | False |
| 14 | MAIN_ORDER_0018_view0/b1/P_REVTIE/d0 | 36 | 0 | UNKNOWN | UNKNOWN | False |
| 15 | MAIN_ORDER_0046_view1/b1/P_REVTIE/d0 | 48 | 0 | UNKNOWN | UNKNOWN | False |
| 16 | MAIN_ORDER_0021_view1/b1/P_PLACEBO/d0 | 120 | 0 | UNKNOWN | UNKNOWN | False |
| 17 | MAIN_ORDER_0014_view0/b0/P_STABLE/d0 | 234 | 0 | UNKNOWN | UNKNOWN | False |
| 18 | MAIN_ORDER_0022_view1/b1/C_NOLIST/d0 | 114 | 0 | UNKNOWN | UNKNOWN | False |
| 19 | MAIN_ORDER_0047_view1/b1/P_STABLE/d0 | 198 | 0 | UNKNOWN | UNKNOWN | False |
| 20 | MAIN_ORDER_0009_view1/b1/P_REVTIE/d0 | 48 | 0 | UNKNOWN | 0 | True |
| 21 | MAIN_ORDER_0014_view1/b0/P_PLACEBO/d0 | 234 | 0 | UNKNOWN | UNKNOWN | True |
| 22 | MAIN_ORDER_0047_view0/b0/P_PLACEBO/d0 | 198 | 0 | UNKNOWN | UNKNOWN | False |
| 23 | MAIN_ORDER_0025_view1/b0/C_REF/d1 | 57 | 0 | UNKNOWN | UNKNOWN | False |
| 24 | MAIN_ORDER_0002_view1/b0/P_PLACEBO/d0 | 44 | 0 | UNKNOWN | UNKNOWN | False |

Full per-row derivations, selected categories, exact operand bindings, and reasons are retained in the JSONL files; all original field differences and reasons are in COMPARISON.json.

## Integrity and limits

EXTRACTION SHA256: `7a02d93cea983dfc57df9dc02802331fd24149914a46e3003faf93330eae3290`

INDEPENDENT_FINDINGS SHA256: `176a23eeea9739274bab72d39f05d58d63ee629408b48aa62a6a279bf11e3e48`

This fixed sample cannot demonstrate that the entire corpus has zero scoring errors. Initial 20 B disagreements must be reported even though they are resolved as audit-side unknown-state handling. These files are review artifacts, not training or new model-answer data.
