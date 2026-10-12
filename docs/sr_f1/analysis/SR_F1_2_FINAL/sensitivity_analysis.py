"""Explicitly post-hoc whitespace sensitivity; never replaces official scores."""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from analyze_released import chart_ci

source=Path(sys.argv[1]);out=Path(__file__).resolve().parent
rows=[json.loads(line) for line in source.open()]
assert len(rows)==32500 and all(r['label']=='POST_HOC_WHITESPACE_SENSITIVITY' for r in rows)
models=defaultdict(list)
for r in rows:models[r['model_id']].append(dict(r,score=r['whitespace_stripped']))
assert len(models)==13 and all(len(v)==2500 for v in models.values())
base='SRF1_2_COMMON_ZERO';arms=['A','J','GATE','DEC']
changes=[chart_ci(models,a,s,k) for a in arms for s in ['all','human','augmented'] for k in ['relaxed_accuracy','exact_match']]
absolute=[]
for m,rs in sorted(models.items()):
    for split in ['all','human','augmented']:
        sub=[r for r in rs if split=='all' or r['variant']==split]
        absolute.append(dict(model=m,split=split,n=len(sub),metrics={k:float(np.mean([r['score'][k] for r in sub])) for k in ['relaxed_accuracy','exact_match']}))
result=dict(label='POST_HOC_WHITESPACE_SENSITIVITY',records=len(rows),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            trailing_lf=sum(r['trailing_lf'] for r in rows),absolute=absolute,changes=changes,
            limitation='Post-hoc all-whitespace strip only. Official scores preserved. No primary endpoint replacement, number extraction, or regeneration.')
(out/'CHARTQA_WHITESPACE_SENSITIVITY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
lines=['### 明确事后的空白敏感性分析','',f"32,500条回答中{result['trailing_lf']:,}条带尾LF。仅将双方字符串`.strip()`后按同一评分函数重算；无数字抽取、内容修复或重新生成。下表是事后敏感性，不是替换后的正式成绩。",'',
       '| 模型 | human relaxed | augmented relaxed | 合计relaxed | 合计exact |','|---|---:|---:|---:|---:|']
for a in ['共同起点']+arms:
    selected=[r for r in absolute if (r['model']==base if a=='共同起点' else r['model'].startswith('SRF1_2_'+a+'_'))]
    values=[np.mean([r['metrics']['relaxed_accuracy'] for r in selected if r['split']==s])*100 for s in ['human','augmented','all']]
    values.append(np.mean([r['metrics']['exact_match'] for r in selected if r['split']=='all'])*100)
    lines.append('| '+a+' | '+' | '.join(f'{v:.2f}' for v in values)+' |')
lines+=['','| 臂−共同起点 | relaxed变化pp | 95%图像簇CI |','|---|---:|---:|']
for r in changes:
    if r['split']=='all' and r['metric']=='relaxed_accuracy':
        lo,hi=r['CI95_pp'];lines.append(f"| {r['arm']} | {r['effect_pp']:.2f} | [{lo:.2f}, {hi:.2f}] |")
lines+=['','去除空白后，GATE与DEC相对起点的下降方向仍存在，不能把全部差异解释为尾换行。另一方面，绝对准确率对归一化十分敏感，正式exact不具鉴别力，保持性结论必须连同该限制和种子差异呈现。次要区间依然仅条件于这三个已实现seed，不是通用保持保证。',
        '', '逐模型/分集/两指标及其图像簇区间见[事后敏感性机器表](CHARTQA_WHITESPACE_SENSITIVITY.json)；冻结正式表保留在[ChartQA正式结果](CHARTQA_RESULTS.json)。']
(out/'CHARTQA_SENSITIVITY_SECTION_zh.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
