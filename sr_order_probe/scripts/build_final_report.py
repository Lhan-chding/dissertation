#!/usr/bin/env python3
"""Read-only delivery projection; never changes the frozen scorer or inference."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def readl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def writej(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def writel(path, rows):
    path.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows))


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def percent(n, d):
    return f'{100*n/d:.2f}%' if d else 'NA'


def pp(x):
    return f'{100*x:+.2f}'


def interval(x):
    return '[' + ', '.join(f'{100*v:.2f}' for v in x) + ']'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    r = args.results
    out = r / 'REPORT'
    out.mkdir(exist_ok=True)
    scores = readl(r / 'SCORES_MAIN.jsonl')
    assert len(scores) == len({x['request_id'] for x in scores}) == 4320
    contrasts = json.loads((r / 'ANALYSIS_MAIN/CONTRASTS.json').read_text())
    groups = defaultdict(list)
    for x in scores:
        groups[(x['parent'], x['condition'])].append(x)
        groups[('POOLED', x['condition'])].append(x)
    conds = ['C0', 'C_NOLIST', 'C_ENUM', 'C_REF', 'C_PRINT', 'C_ALT', 'P_STABLE', 'P_REVTIE', 'P_PLACEBO']
    descriptive = []
    for (parent, cond), rows in sorted(groups.items()):
        n = len(rows); hit = sum(x['S_hit'] for x in rows); unknown = sum(x['S'] is None for x in rows)
        descriptive.append(dict(parent=parent, condition=cond, N=n,
            A=sum(x['A'] for x in rows), S_hit=hit, S_unknown=unknown,
            S_known=n-unknown, S_given_known=hit/(n-unknown) if n>unknown else None,
            S_identification_lower=hit/n, S_identification_upper=(hit+unknown)/n,
            B_pass=sum(x['B']==1 for x in rows), B_unknown=sum(x['B'] is None for x in rows),
            V_probe=sum(x['V_probe'] for x in rows), A_parseable=sum(x['A_parseable'] for x in rows),
            complete=sum(x['complete'] for x in rows), order_known=sum(x['rank_order_known'] for x in rows),
            values_true=sum(x['rank_values_true']==1 for x in rows),
            order_compliant=sum(x['order_compliance']==1 for x in rows),
            order_unknown=sum(x['order_compliance'] is None for x in rows),
            reversed_choice=sum(x['reversed_boundary_choice_hit'] for x in rows),
            enum_choice=sum(x['enumeration_choice_hit'] for x in rows),
            alpha_choice=sum(x['alphabetical_choice_hit'] for x in rows)))
    writej(out / 'CONDITION_SUMMARY.json', descriptive)
    with (out / 'CONDITION_SUMMARY.csv').open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(descriptive[0]));w.writeheader();w.writerows(descriptive)
    bygroup={(x['parent'],x['condition']):x for x in descriptive}
    original=json.loads((r/'ANALYSIS_MAIN/DESCRIPTIVE.json').read_text())
    supplement=json.loads((r/'SUPPLEMENT_MAIN/SUPPLEMENTARY_DESCRIPTIVE.json').read_text())
    supplement_by={(x['parent'],x['condition'],x['rename']):x for x in supplement}
    full=[]
    for row in original:
        extra=supplement_by[(row['parent'],row['condition'],row['rename'])]
        assert row['N']==extra['N']
        merged=dict(row)
        for key,value in extra.items():
            if key in merged:assert merged[key]==value,(key,merged[key],value)
            merged[key]=value
        full.append(merged)
    assert len(full)==54
    writej(out/'FULL_54_GROUPS.json',full)
    with (out/'FULL_54_GROUPS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(full[0]));w.writeheader();w.writerows(full)

    # Keep first-pass independent findings and all disagreements, never overwrite them.
    audit=[]; differences=[]; reviewers=[]
    selected=readl(r/'SUPPLEMENT_MAIN/AUDIT/72_SELECTED_BLIND.jsonl')
    frozen={x['request_id']:x for x in scores}
    for part in range(1,4):
        folder=r/'AUDIT_REVIEW'/f'part{part}'
        findings=readl(folder/'INDEPENDENT_FINDINGS.jsonl')
        comparison=json.loads((folder/'COMPARISON.json').read_text())
        diffs=comparison.get('disagreements',comparison.get('initial_disagreements',[]))
        assert len(findings)==24
        for d in diffs:
            assert d['field']=='B' and d['independent']==0 and d['frozen'] is None
            differences.append(dict(reviewer=f'/root/audit_part{part}',**d))
        for row in findings:
            rid=row['request_id']; related=[d for d in diffs if d['request_id']==rid]
            audit.append(dict(request_id=rid,reviewer=f'/root/audit_part{part}',initial_independent_finding=row,
                disagreements=related,adjudicated_B=None if related else row['B'],
                adjudication='B unavailable is UNKNOWN; independent reviewer coding corrected, frozen score unchanged' if related else 'No reported difference',
                frozen_score=frozen[rid]))
        reviewers.append(dict(reviewer=f'/root/audit_part{part}',rows=24,
            extraction_sha256=hashlib.sha256((folder/'EXTRACTION.jsonl').read_bytes()).hexdigest(),
            finding_sha256=hashlib.sha256((folder/'INDEPENDENT_FINDINGS.jsonl').read_bytes()).hexdigest(),
            initial_disagreement_count=len(diffs)))
    assert {x['request_id'] for x in audit}=={x['request_id'] for x in selected}
    assert len(audit)==72 and len(differences)==28
    writel(out/'72_selected_rows_and_findings.jsonl',audit)
    writej(out/'AUDIT_SUMMARY.json',dict(status='COMPLETE_AGENT_REVIEW',rows=72,reviewers=reviewers,
        initial_disagreement_cells=len(differences),unresolved_disagreements=0,
        production_score_changed=False,disagreements=differences,
        limitation='Not human double blind, not proof of zero scoring errors in the population; original request IDs reveal condition.'))

    # These are post-hoc illustrations selected by sorted ID, never a new test.
    idx={(x['parent'],x['qid'],x['condition'],x['draw']):x for x in scores}
    cases=[]
    def add(label, rows, interpretation):
        cases.append(dict(case=label,selection='First lexicographic matching request; post-hoc descriptive only',
                          interpretation=interpretation,rows=rows))
    for x in sorted(scores,key=lambda x:x['request_id']):
        if x['condition']!='P_STABLE':continue
        rev=idx[(x['parent'],x['qid'],'P_REVTIE',0)];placebo=idx[(x['parent'],x['qid'],'P_PLACEBO',0)]
        if not any(c['case']=='PREFIX_NO_CHANGE' for c in cases) and x['S']==rev['S']==1 and x['rank_values_true']==rev['rank_values_true']==1:
            add('PREFIX_NO_CHANGE',[x,rev],'真实前缀关键顺序变化，但两次selected均正确；不能说每题都会跟随前缀。')
        if not any(c['case']=='PREFIX_SUPPORT' for c in cases) and x['S']==placebo['S']==1 and rev['S']==0 and rev['reversed_boundary_choice_hit']==1:
            add('PREFIX_SUPPORT',[x,rev,placebo],'正确与placebo前缀均选对，关键反转后跟随反转错误集合；单例支持而非总体效应证明。')
    for x in sorted(scores,key=lambda x:x['request_id']):
        if x['condition']!='C_ALT' or x['rename']!='CONFLICT':continue
        pr=idx[(x['parent'],x['qid'],'C_PRINT',x['draw'])]
        if not any(c['case']=='NATURAL_NONCOMPLIANCE' for c in cases) and x['order_compliance']==0 and pr['order_compliance']==1:
            add('NATURAL_NONCOMPLIANCE',[x,pr],'字母报告要求未使证据达到完整字母序；自然提示的ITT不能当作完全服从的序列中介效应。')
        mates=[y for y in scores if y['parent']==x['parent'] and y['root_id']==x['root_id'] and y['condition']=='C_ALT' and y['draw']==x['draw'] and y['rename']=='ALIGNED']
        if mates and not any(c['case']=='NAMING_CROSSOVER' for c in cases):
            aligned=mates[0]
            if aligned['order_compliance']==x['order_compliance']==1 and aligned['S']==1 and x['S']==0:
                add('NAMING_CROSSOVER',[aligned,x,idx[(x['parent'],aligned['qid'],'C_PRINT',x['draw'])],pr],
                    '同一root、父与draw，两命名的字母约束下结果不同；对应打印序条件一起展示，不能把两视图当独立世界。')
    for x in sorted(scores,key=lambda x:x['request_id']):
        if x['parent']!='b0' or x['condition']!='C_REF':continue
        base=idx[('b0',x['qid'],'C0',x['draw'])]; other=idx[('b1',x['qid'],'C_REF',x['draw'])]; obase=idx[('b1',x['qid'],'C0',x['draw'])]
        if x['A']<base['A'] and other['A']>obase['A']:
            add('PARENT_OPPOSITE_C_REF',[base,x,obase,other],'同题同draw的澄清处理在b0下降、b1上升。父级总体主比较同向不代表逐题或次比较同向。');break
    assert len(cases)==5
    wanted={row['request_id'] for c in cases for row in c['rows']}
    raw={}
    for line in (r/'SERVER_CAPTURE/RAW/MAIN.jsonl').open():
        d=json.loads(line)
        if d['request_id'] in wanted:
            raw[d['request_id']]={k:d[k] for k in ['request_id','full_text','finish_reason','assistant_prefill','suffix_text','seed']}
    writej(out/'PAIRED_CASES.json',dict(selection_scope='Post-hoc illustrative, no significance inference',cases=cases,raw=raw))

    pooled=[x for x in contrasts if x['primary']]
    primary_table=table(['比较','指标','差值(pp)','95%区间','99.375%区间','Holm p'],[
        [x['contrast'],x['metric'],pp(x['estimate']),interval(x['ci95']),interval(x['ci99_375']),f"{x['p_holm_8']:.6f}"] for x in pooled])
    parent_table=table(['比较','指标','b0 差值 [95%区间]','b1','b2'],[
        [c,m]+[next(pp(x['estimate'])+' '+interval(x['ci95']) for x in contrasts if x['panel']==p and x['contrast']==c and x['metric']==m) for p in ['b0','b1','b2']]
        for c in ['D_ENUM','I_ALT','D_REV','D_SPEC'] for m in ['A','S_hit']])
    cond_table=table(['条件','N','A','S确认正确','S UNKNOWN','S潜在真值识别区间','B正确/UNKNOWN','V_probe'],[
        [c,bygroup[('POOLED',c)]['N'],percent(bygroup[('POOLED',c)]['A'],bygroup[('POOLED',c)]['N']),
         percent(bygroup[('POOLED',c)]['S_hit'],bygroup[('POOLED',c)]['N']),
         str(bygroup[('POOLED',c)]['S_unknown'])+' ('+percent(bygroup[('POOLED',c)]['S_unknown'],bygroup[('POOLED',c)]['N'])+')',
         interval([bygroup[('POOLED',c)]['S_identification_lower'],bygroup[('POOLED',c)]['S_identification_upper']])+'%',
         str(bygroup[('POOLED',c)]['B_pass'])+'/'+str(bygroup[('POOLED',c)]['B_unknown']),
         percent(bygroup[('POOLED',c)]['V_probe'],bygroup[('POOLED',c)]['N'])] for c in conds])
    perparent=table(['父','条件','N','A正确数','S正确数','S UNKNOWN','B正确数','V_probe数'],[
        [p,c]+[bygroup[(p,c)][k] for k in ['N','A','S_hit','S_unknown','B_pass','V_probe']]
        for p in ['b0','b1','b2'] for c in conds])
    secondary=table(['次比较','A差值 [95%区间]','S_hit差值 [95%区间]'],[
        [c]+[next(pp(x['estimate'])+' '+interval(x['ci95']) for x in contrasts if x['panel']=='POOLED_FIXED_PARENTS' and x['contrast']==c and x['metric']==m) for m in ['A','S_hit']]
        for c in ['D_NOLIST','D_NAMES','D_REF','D_PRINT','D_PLACEBO']])
    case_md='\n\n'.join('### '+c['case']+'\n\n'+c['interpretation']+'\n\n'+table(
        ['request_id','A','S','B','顺序服从'],[[x['request_id'],x['A'],str(x['S']),str(x['B']),str(x['order_compliance'])] for x in c['rows']]) for c in cases)
    report=f'''# SR-ORDER-PROBE 最终结果（2026-11-03）

**本次固定48个世界、三个父模型下，冲突清单和字母排序要求明显改变任务表现；关键同值前缀反转也降低了可确认正确的产出率。但前缀回答有大量UNKNOWN，不能把全部前缀分数差解释为已经识别的潜在语义选择效应，更不能声称训练修复或自然中介已被证明。**

## 1. 实际执行与异常

216条SMOKE与4,320条MAIN均完成，三个父b0/b1/b2分别对应上游J6 seed 71001/71002/71003；每父MAIN 1,440条，模型Qwen3.5-9B，snapshot c202236235762e1c871ad0ccb60c8ee5ba337b9a。FP32、batch=1、T=1、top_p=1、top_k=0、最多1,536新token；本轮0参数更新、无训练。MAIN运行12个单GPU worker，b0/b1/b2依次、每父4固定分片。

主作业242982于新加坡2026-11-03 06:18:03开始，10:44:21结束，4小时26分18秒，COMPLETED 0:0。全部4,536请求各START/SAVED一次，未发生请求级重采。最初CPU部署242456因快照不同步失败，恢复242476完成；最初烟测242490/491/492因缺CUBLAS配置在模型加载前失败，0输出，恢复242525/526/527完成。失败日志和事件保留，未洗成PASS。

MAIN技术验证错误0；3,821 balanced、488 eos、11 length，所有length与非法输出留在分母。balanced/eos只是停止状态，**不是JSON有效或任务正确证明**。所有输入token、真实后缀token/probability、prefill边界均保留并重新核验。实际同assistant长prefill与父身份通过，CPU评分/图像/历史排重证据见EXECUTION。原生J仅支持最多8类别，本轮10类别J不适用，V_probe不能冒充J。

## 2. 全部条件水平与未知状态

A为最终answer精确正确，非法/缺失答案记0，因此前缀A同样不是排除格式问题后的潜在能力。S_hit为可确认selected正确/全部分配回答。S未知不是语义FAIL。B检查任务正确集合上的数值绑定；V_probe为本轮形状契约。自然条件各576条、前缀条件各288条，不能将不同条件混合平均当成等权能力指标。

{cond_table}

上表S识别区间是样本缺失真值的最坏情况范围，不是置信区间。每父×条件×命名54组的A/S/V及S可判率、条件正确率在ANALYSIS_MAIN/DESCRIPTIVE.json；B UNKNOWN、答案可解析率、完成率、顺序和token在SUPPLEMENT_MAIN/SUPPLEMENTARY_DESCRIPTIVE.csv。FULL_54_GROUPS.csv将两者按父/条件/命名原样合并，未重新评分。每父条件原始分子如下，两命名分别统计见合并表：

{perparent}

## 3. 四个确认比较、两项终点

按draw→命名/root→固定三父等权汇总，聚类单位为48个基础世界；四结构各12 root，10,000次配对分层bootstrap。D_ENUM=C0−C_ENUM；I_ALT为字母指令相对打印序指令的aligned−conflict交互；D_REV=P_STABLE−P_REVTIE；D_SPEC=P_PLACEBO−P_REVTIE。8项固定三父总体确认检验一起Holm调整。

{primary_table}

全部8项主比较的方向在这些构造型世界上为正，99.375%区间也不跨0。p值是居中的cluster-bootstrap近似，不是Fisher精确检验；多个p相同是10,000次计算分辨率所致，不能把0.000800解释为精确概率。未增加种子、draw或新世界。

{parent_table}

b1的清单与前缀效应明显较小；b1 D_ENUM的99.375%区间跨0（A和S_hit均如此），不能用总体强效应掩盖父差异。b1三个前缀条件的UNKNOWN达83.33%–87.50%，尤其限制其解释。父级区间未冒充另经Holm确认的一组主检验。三个父是固定上游产物，不是本轮训练重复。C_ENUM是事先构造的特定冲突置换，不能推广为任意随机乱序的平均效应。

## 4. 操纵率、placebo分解与限制

C_ALT实际完整字母序服从231/576=40.10%，C_PRINT打印序服从492/576=85.42%。C_ALT在aligned命名A为250/288=86.81%，conflict为100/288=34.72%；C_PRINT对应237/288=82.29%与234/288=81.25%。因此自然指令存在强任务相容性交互，但并非全部回答都服从，更不能用服从后的子集重算因果效应。

P_STABLE/P_REVTIE/P_PLACEBO的A分别30.90%/6.94%/31.25%，S_hit分别32.29%/7.29%/33.33%。STABLE−PLACEBO的A为−0.35pp [−4.51,3.47]，S_hit为−1.04pp [−5.21,3.13]；D_SPEC不是仅由placebo巨大改善导致，但这里也没有证明严格等效。

三个前缀条件的S UNKNOWN分别63.19%/53.47%/61.46%。REVTIE的UNKNOWN反而更少，观察到的S_hit下降不能简单归为更多格式失败；然而潜在语义S的总体D_REV最坏情况差值区间为[−28.47,88.19]pp、D_SPEC为[−27.43,87.50]pp，均跨0。它们是样本识别界而非95%区间。b2个体样本的对应下界+8.33/+11.46pp是例外，不能推广为三父总体，也不能当作带抽样不确定性的确认结论。

前缀evidence为forced，不能记成模型自发读图正确；自然条件中完整Alpha事实正确是单独描述，不能用其筛选后子集取代固定分母。C_ENUM跟随预设错误集合及P_REVTIE跟随反转集合的全分母计数见CONDITION_SUMMARY.csv，不能把碰巧答案正确倒推为选择/绑定正确。

次比较均如实保留，未追加效果门：

{secondary}

D_NOLIST=C0−C_NOLIST的小幅差异仍不确定；C_REF澄清总体接近零，b0的A为负而b1为正。C_PRINT有小幅总体提示支架改善，但b0的S_hit反而−1.04pp；这是次比较，不能与8项主检验混同，合并99.375%区间跨0。C_ENUM−C_NOLIST为负说明冲突清单比仅删清单更差，不宜把本结果只归因于名称支持。

## 5. 成对例子与反例

以下按request_id排序选择第一个满足明确模式的例子，是事后说明材料，不是新检验或预注册样本。完整原文、前缀、suffix、seed和分数见PAIRED_CASES.json；统计仍使用全部48个root。

{case_md}

## 6. 独立审核与复现

固定seed 2026110203选72条：54条覆盖每父每条件2条，另18条按预定异常优先次序抽取，未重复。三个独立代理各实际阅读24条，先封存原文抽取，再读world/query独立推导gold，最后比较冻结分数。72条gold全部一致。

初判共28处B状态编码分歧：审核代理把坏JSON或缺失bindings的不可判状态误写0，按既定字段定义裁为UNKNOWN/null。初判、封存哈希、原文、分歧和更正均保留；没有覆盖生产评分，未发现需改变生产scorer的缺陷，无未决分歧。该审核不是人类双盲，request_id可揭示条件，同类代理也可能共享盲点；72条不能保证总体零错误。

全部4,320条已在本地从原始记录重新评分并重新运行冻结统计，字节一致性见REPRODUCTION.json。全4,536条token数量/概率数量、同assistant前缀边界、唯一ID、原文SHA及attempt账本检查见FINAL_INTEGRITY.json。未产生新模型回答。

## 7. 成本、交付和下一步边界

MAIN输入6,513,408 tokens（含强制前缀240,516），采样2,043,761 tokens；SMOKE另采样92,159 tokens，总2,135,920。MAIN分配17.7533 GPU小时，成功SMOKE0.9639 GPU小时，初始化失败另0.0089 GPU小时，合计18.7261 GPU小时；不是GPU利用率或账单费用。生成秒数跨worker重叠，不可当作作业时长。

交付包含完整SMOKE/MAIN原始JSONL（gzip，无损，包含输入/suffix token和概率）、评分、统计、54组描述表、审核原文及分歧、身份/manifest/日志/成本、冻结数据和本轮源码。不含模型权重、字体、其他项目、冗余单请求副本；服务器原件保留。另有约414MB旧传输包/CPU缓存清理记录；释放的是磁盘空间。

本轮支持当前任务/语境下清单次序与报告要求影响可观察表现，并显示关键同值前缀顺序对可确认产出率的影响。没有识别自然生成的完整中介、没有证明诊断优于等预算简单统计、没有验证任何训练修复/跨任务迁移。主比较有证据不等于后续训练已授权。

如后续继续，可另行约定提示支架撤除后的保持/迁移，或相同探查和动作库预算下诊断与简单统计的比较。本轮在交付停止；不新开训练、采样、seed、正式TEST或效果搜索。
'''
    (out/'RESULTS_REPORT_zh.md').write_text(report)
    print('Wrote final report, full-denominator tables, cases and independent audit projection')


if __name__ == '__main__':
    main()
