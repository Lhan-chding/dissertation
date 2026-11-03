# coding: utf-8
import json,pathlib,hashlib,datetime,collections
p=pathlib.Path(__file__).parent
ind=[json.loads(x) for x in (p/'INDEPENDENT_FINDINGS.jsonl').read_text().splitlines()];scores={x['request_id']:x for x in map(json.loads,(p/'FROZEN_SCORES.jsonl').read_text().splitlines())}
keys={'A':'A','S':'S','B':'B','rank_order_known':'rank_order_known','values_true':'rank_values_true','print':'rank_order_print','alphabetical':'rank_order_alphabetical','numeric_sorted':'rank_order_numeric_world','order_compliance':'order_compliance'}
rows=[];diffs=[]
for x in ind:
 s=scores[x['request_id']];checks={};d=[]
 for k,sk in keys.items():
  v={'PASS':1,'FAIL':0,'UNKNOWN':None}[x[k]] if k=='S' else x[k]
  eq=v==s[sk];checks[k]={'independent':v,'frozen':s[sk],'matches':eq}
  if not eq:
   cause='Independent first-pass B encoded inability to verify invalid JSON as 0, whereas frozen B uses null (UNKNOWN). This is reviewer state-encoding disagreement, not evidence that malformed JSON should pass. Preserve initial finding; adjudicated B=null for both malformed records.' if k=='B' and v==0 and s[sk] is None else 'UNRESOLVED'
   dd={'request_id':x['request_id'],'index':x['index'],'field':k,'independent':v,'frozen':s[sk],'reason':cause,'adjudicated':s[sk] if cause!='UNRESOLVED' else None};d.append(dd);diffs.append(dd)
 rows.append({'request_id':x['request_id'],'index':x['index'],'checks':checks,'disagreements':d,'factual_reason':x['reason']})
report={'reviewer':'independent Codex agent /root/audit_part1','method':'Agent review, not human double blind; all 24 original texts actually read before world/query and frozen scores. Independent Python standard-library calculations, no production scorer.','rows':len(ind),'field_count':len(keys),'checked_cells':len(ind)*len(keys),'initial_disagreement_count':len(diffs),'initial_disagreement_rows':len(set(d['request_id'] for d in diffs)),'unresolved_count':sum(d['reason']=='UNRESOLVED' for d in diffs),'gold_verified_rows':sum(all(x['provided_gold_verified'].values()) for x in ind),'disagreements':diffs,'comparisons':rows}
(p/'COMPARISON.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
counts=collections.Counter(x['S'] for x in ind)
lines=['# part1 独立代理复核','', '审核身份：独立 Codex 代理 `/root/audit_part1`。这是代理复核，不是人类双盲。样本复核不能证明总体零错误。','', '## 阶段与完整性','', '已实际阅读 BLIND.jsonl 全部24条原文（包含 work、完整JSON结构和异常结尾）。先封存 EXTRACTION.jsonl，再读取 PRIVATE.jsonl 的 world/query；按真实 Alpha 与图上位置打破同值，独立计算选择集合、Beta绑定和精确有理数答案；24条提供的gold全部核对一致。保存 INDEPENDENT_FINDINGS.jsonl 后才首次读取 FROZEN_SCORES.jsonl。未读取或调用生产scorer/measurement。', '', '阶段1 SHA256：`'+(p/'EXTRACTION.sha256').read_text().split()[0]+'`。', '阶段2 SHA256：`'+(p/'INDEPENDENT_FINDINGS.sha256').read_text().split()[0]+'`。','', '## 逐字段比较','',f'24条 × 9字段 = {24*9}项比较；首次独立判断有{len(diffs)}项分歧，涉及2条，全部为B的UNKNOWN编码；其余214项一致。原始独立判断已封存未覆盖，COMPARISON.json保留分歧和裁定。', '', '第16、17条缺最后闭合大括号。独立首判将“无法核验B”编码为0；冻结评分使用null。裁定：应保留B=UNKNOWN/null，不能把无可判绑定当成已知绑定错误。这是本代理初判状态编码不够细，不是生产评分器错误。二者A=0、S=UNKNOWN、顺序UNKNOWN均一致。', '', 'EXTRACTION中的complete表示完整可解析对象；冻结complete表示生成终止（eos也计完成）。这两个字段含义不同，不能据此指认生产完成率错误；本比较的主要字段不混用它们。','',f"独立A=1共{sum(x['A'] for x in ind)}条；S为{dict(counts)}；顺序完整可判{sum(x['rank_order_known'] for x in ind)}条。",'', '## 需要保留的事实','', '- 第18条缺selected：S未知；不能从work补抽。', '- 第21条重复Alpha类别；第22、23条仅提供4个Alpha类别：顺序均未知，但其A/S/B可独立正确。', '- 第11、12条C_ALT均未严格按全局字母序，order_compliance=False；数值仍真实。', '- 第14条work先写174后改216；最终answer=216独立正确，不挑work中的数字评分。', '- 第4、6、15、20条边界同值选择错误；第5条选Clock而不是Desk；第24条选Mug而不是Book，均保留为失败。', '- P_* evidence标为forced，不把它们当作模型自发证据真实性。', '', '## 每条事实依据','']
for x in ind: lines.append(f"{x['index']}. `{x['request_id']}` — A={x['A']}, S={x['S']}, B初判={x['B']}, rank_order_known={x['rank_order_known']}. {x['reason']}")
(p/'REVIEW.md').write_text('\n'.join(lines)+'\n')
(p/'STAGE3_RECEIPT.json').write_text(json.dumps({'stage':3,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'comparisons_sha256':hashlib.sha256((p/'COMPARISON.json').read_bytes()).hexdigest(),'initial_disagreement_count':len(diffs),'unresolved_count':report['unresolved_count']},indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['comparisons','disagreements']}));print(json.dumps(diffs,indent=2))
