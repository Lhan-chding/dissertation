# -*- coding: utf-8 -*-
import json,hashlib
from collections import Counter
from pathlib import Path
F=[json.loads(l) for l in Path('INDEPENDENT_FINDINGS.jsonl').read_text().splitlines()]
S={r['request_id']:r for r in map(json.loads,Path('FROZEN_SCORES.jsonl').read_text().splitlines())}
fields={'A':'A','S':'S','B':'B','rank_order_known':'rank_order_known','values_true':'rank_values_true','print':'rank_order_print','alphabetical':'rank_order_alphabetical','value_sorted':'rank_order_numeric_world','order_compliance':'order_compliance'}
rows=[];diffs=[]
for f in F:
 s=S[f['request_id']];checks={}
 for a,b in fields.items():
  expected={'PASS':1,'FAIL':0,'UNKNOWN':None}[f[a]] if a=='S' else f[a]
  match=expected==s[b]
  checks[a]={'independent':f[a],'frozen':s[b],'matches':match}
  if not match:
   d={'ordinal':f['ordinal'],'request_id':f['request_id'],'field':a,'independent':f[a],'frozen':s[b],'reason':'Initial independent B encoded unparseable bindings as 0, while frozen uses null for UNKNOWN. Raw JSON is invalid; no binding verdict can be established. This is reviewer status encoding disagreement, not evidence of a frozen scorer bug; retain original independent result and resolve B as UNKNOWN.' if a=='B' and 7<=f['ordinal']<=12 else 'Unresolved disagreement.'}
   diffs.append(d)
 rows.append({'request_id':f['request_id'],'ordinal':f['ordinal'],'checks':checks})
obj={'review_mode':'independent_agent_not_human_double_blind','rows':24,'fields_per_row':len(fields),'matched_cells':sum(v['matches'] for r in rows for v in r['checks'].values()),'initial_disagreement_count':len(diffs),'initial_disagreement_rows':len(set(d['request_id'] for d in diffs)),'disagreements':diffs,'comparisons':rows,'resolution':'Retain six initial B=0 versus null disagreements. Invalid JSON makes B UNKNOWN; frozen null is appropriate. No independently established production scoring defect in this sample. No extrapolation to zero error in the population.'}
Path('COMPARISON.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
lines=['# Part 2 独立代理复核','', '审核者：独立代理 `/root/audit_part2`。这是代理复核，不是人类双盲。24 条 BLIND 原文均实际逐条阅读（分 1–8、9–16、17–24 三批无截断显示）。未读取或调用生产 scorer/measurement。','', '阶段顺序：先阅读评分定义与 BLIND，落盘 EXTRACTION 与 SHA/阶段回执；之后读取 PRIVATE 的 world/query，独立按 Alpha 排序、同值使用原图类别位置，计算正确集合及 Beta 精确有理数聚合，核验全部 24 条提供 gold；INDEPENDENT_FINDINGS 与 SHA 落盘后才读取 FROZEN_SCORES。','', '## 结果','', '- 全部 24 条提供 gold 的 indices、selected、operands、answer 均与独立 world/query 推导一致。', '- 严格完整 JSON 18 条；6 条前缀回答（第 7–12 条）均缺最终根对象右花括号，不补齐、不从 work 或残缺文本挽救字段。未在可解析对象检测到重复键；原文人工检查未见重复键。', '- 独立 A：15/24；S：16 PASS、2 FAIL、6 UNKNOWN。', '- 第 2 条只有 9 个 Alpha 证据类别，遗漏 Clock；A/S/B 可确认正确，但顺序和完整证据数值真值 UNKNOWN。', '- 第 5 条 selected 正确但 Mug Beta 错报 48（实际 78），最终 49.5 错（真值 57），故 A=0、S=PASS、B=0。', '- 第 17 条把 Table 替代 Desk，最终 198 错（真值 174）；第 24 条把 Chair 替代 Stool，最终 20 错（真值 38）；两条 A=0、S=FAIL、B=0。', '- 第 24 条 work 的自我纠正不改变错误最终 selected/answer；第 11 条 work 与 binding 数值不一致，但全 JSON 本来非法，不做字段挽救。', '', '## 原始分歧及处理','', f'逐条比较 9 个字段，共 216 个单元，初始一致 {obj["matched_cells"]} 个；6 条各有一个 B 状态分歧，原始独立判断已保留，不回写。', '', '第 7–12 条的独立脚本把非法 JSON 的 B 编成 0；冻结评分为 null（UNKNOWN）。评分定义未在该简短文档显式给出 B 的未知编码，导致独立复核使用了错误状态约定。非法 JSON 没有可判 bindings，应该保留 UNKNOWN，冻结 null 合理。本分歧属于复核方状态编码差异，并未发现生产评分错误。INDEPENDENT_FINDINGS 及其 SHA 仍保存初始判断；COMPARISON 逐条保留原值、冻结值及上述解释。其余 A、S、rank_order_known、values_true、print、alphabetical、numeric order、order_compliance 全部一致。', '', 'EXTRACTION.complete 指 JSON 内容完整且结束状态正常；冻结 complete 指生成结束状态。因此 EOS 但缺根括号可能 frozen complete=1、json_valid=0，这两个意义不混同。未把 complete 的异名语义当生产错误。', '', 'C_PRINT/C_ALT 指令服从与 Alpha 数值真值分开判断；前缀字段标 forced，但非法整体 JSON 时顺序 UNKNOWN。B=1 的 15 条、B=0 的 3 条、B UNKNOWN 的 6 条是状态解释后的结果。', '', '样本为固定 24 条，不能证明全体零评分错误，不是科学实验通过门；未改变任何冻结评分、原始回答或任务数据。','', '## 文件与校验','', '- `EXTRACTION.jsonl`、`EXTRACTION.sha256`、`STAGE1_RECEIPT.json`','- `INDEPENDENT_FINDINGS.jsonl`、`INDEPENDENT_FINDINGS.sha256`、`STAGE2_RECEIPT.json`','- `COMPARISON.json`：逐条字段对照和分歧保留','- 三个独立辅助脚本仅使用标准库；不导入项目评分器。','', '## 逐条事实依据','']
for f in F:lines += [f"{f['ordinal']}. `{f['request_id']}` — {f['facts']}"]
Path('REVIEW.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:obj[k] for k in ['rows','matched_cells','initial_disagreement_count','resolution']},ensure_ascii=False))
for path in ['EXTRACTION','INDEPENDENT_FINDINGS']:
 actual=hashlib.sha256(Path(path+'.jsonl').read_bytes()).hexdigest();assert actual==Path(path+'.sha256').read_text().split()[0]
print('Both pre-comparison hashes verified unchanged')
