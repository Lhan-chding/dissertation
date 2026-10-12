"""Render the Chinese final report from audited analysis artifacts."""
import json
import sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=Path(sys.argv[1])
read=lambda p:json.loads(p.read_text())
d=read(HERE/'ANALYSIS.json'); base='SRF1_2_COMMON_ZERO'; arms=['A','J','GATE','DEC']
fmt=lambda x:f'{x:.2f}'
ci=lambda x:'['+', '.join(fmt(v) for v in x)+']'
lines=[]
def add(s=''):lines.append(s)
def table(headers,rows):
    escape=lambda x:str(x).replace('|',r'\|')
    add();add('| '+' | '.join(map(escape,headers))+' |');add('| '+' | '.join(['---']*len(headers))+' |')
    for row in rows:add('| '+' | '.join(map(escape,row))+' |')
    add()
def panel(arm,pool,protocol='evidence_answer',metric='J'):
    rr=[r for r in d['absolute'] if r['pool']==pool and r['protocol']==protocol and r['view'] is None
        and (r['model']==base if arm=='共同起点' else r['model'].startswith('SRF1_2_'+arm+'_'))]
    return [r['metrics'][metric]*100 for r in rr]
def avg(arm,pool,protocol='evidence_answer',metric='J'):return np.mean(panel(arm,pool,protocol,metric))
add('# SR-F1.2 最终中文报告')
add('\n日期：2026-10-12（新加坡）。分析对象为冻结的四臂 × 三种子 × 96步及共同起点。')
add('\n## 1. 结论与边界\n')
add('联合事件训练明显改善了本任务的未见联合正确率。GATE 相对 J 提高 **3.32 个百分点**，四对比族校正后的98.75%区间为 **[1.93, 4.88]**；但 GATE 相对 DEC 仅 +0.79pp，区间 **[−1.01, 2.57]**，且三个种子方向不一致，**不能据此宣称分层奖励优于加法组合**。J 相对 A 的增益为17.53pp，DEC 相对 J 为2.53pp。')
add('\n预登记预测没有形成一致支持：P1、P3方向相反；P2点估计方向符合但区间跨零。GATE首8步纯语义分支率的描述性机制核对在登记容差内，不能抵消预测失败，也不能证明“测量正确指导了训练”。')
add('\n收益具有任务和协议边界：纯答案协议的准确率远低于证据协议；组合偏移上DEC优于GATE。ChartQA正式评分出现尾换行口径问题，必须同时阅读第7节及独立敏感性结果；不能把正式exact=0解释为模型不会回答，也不能据此认证外部能力保持。所有不利结果均保留。')
add('\n## 2. 完成度、数据与验收\n')
add('12条科学训练均完成96步，最终完整Adam/参数/调度器/cursor/RNG、adapter及FIT32/64/96由门禁验收；13个终点评价作业全部真实COMPLETED 0:0并离队。统一释放在新加坡2026-10-12 22:51:42完成，CPU198239正常结束；最终回执为`EVALUATIONS_COMPLETE_REPORT_PENDING`，共227,700条正式端点评价记录。该状态字面上仍表示报告待交付，本报告不改写服务器不可变回执。')
table(['面板','每模型记录数','边界'],[
 ['TEST_ID证据','8,192','128根×16题×4样本；主终点'],['TEST_ID纯答案','4,096','相同题，独立协议K2'],
 ['组合/语言/外观偏移','各512','各64根，证据协议K4；分别报告'],['ChartQA','2,500','human/augmented各1,250；greedy'],
 ['五诊断视图','训练模型1,280；起点2,560','起点K8、终点K4；起点另有MONITOR4,096'],
 ['TRAIN_FIT','每次128','训练模型32/64/96；共同起点一次']])
add('正式协议测量总量238,964 = 端点227,700 + 起点语义测量6,656 + 训练中FIT4,608。原表235,892漏计中间FIT3,072；**238,964不包括训练rollout**，12条训练另生成147,456条（12×96×128），技术试跑/预检/失败另记。')
add('\n固定学习率5e−5、语言解码器全Linear LoRA r8/α16/dropout0、KL系数0.02、微批1、无checkpointing、每题8条batch生成、单次PG更新及detached TIS C=2。PART未选择。跨种子滚动填空及提前封存评价仅改变调度时刻，冻结科学设置未变。')
add('\n## 3. 主终点与四项主对比\n')
add('A表示答案正确，E表示证据集合契约正确，p_read为读取比例，P为证据与读取同时通过，J=A×P。下表百分比按题内样本→根内题→族内根→四族等权→三个种子等权；共同起点只生成一次，未伪装成三条独立重复。')
table(['模型','A','E','p_read','P','J'],[[a]+[fmt(avg(a,'TEST_ID',metric=k)) for k in ['A','E','p_read','P','J']] for a in ['共同起点']+arms])
table(['臂','seed71001 J','seed71002 J','seed71003 J','均值'],[[a]+[fmt(x) for x in panel(a,'TEST_ID')]+[fmt(avg(a,'TEST_ID'))] for a in arms])
table(['主对比','均值差pp','三seed差pp','95% CI','98.75% CI'],[[r['contrast'],fmt(r['effect_pp']),'/'.join(fmt(x) for x in r['seed_effects_pp']),ci(r['CI95_pp']),ci(r['CI98_75_pp'])] for r in d['primary']])
add('严格复用预登记统计：四族内整根配对bootstrap5,000次、随机种子812901、Bonferroni四项双侧98.75%区间。区间是**给定这三个实际训练种子后的题根测量不确定性**，不是训练随机性总体区间。2pp为关注量级而非PASS门槛；跨零既不是等价证明，也不是安全证明。GATE−J下界略低于2pp，不能声称有把握超过2pp。')
add('\n![主对比及逐种子结果](PRIMARY_FIGURE.svg)\n')
add('可复算数据：[四项主对比](PRIMARY_CONTRASTS.json)，[全模型/题族/变体/协议长表](RESULTS_LONG.csv)。')
add('\n## 4. 预登记预测与起点状态\n')
sem=read(ROOT/'SEMANTIC_STATE.json');pred=read(ROOT/'SEMANTIC_PREDICTIONS.json')
table(['题族','起点A−J pp','s_A','s_J','s_G','D=P(A=0|P=1)','D分子/分母','gold_values A增益pp'],[
 [r['family'],fmt((r['means']['A']-r['means']['J'])*100),fmt(r['s_A']*100),fmt(r['s_J']*100),fmt(r['s_G']*100),fmt(r['D']['probability']*100) if r['D']['probability'] is not None else 'undefined',f"{r['D']['numerator']}/{r['D']['denominator']}",fmt(r['gold_values_uplift']*100)] for r in sem['families']])
add('P1/P2按冻结的(值,族名)排序分低2/中1/高1，未按本次结果重分。P3按冻结的D≥0.5且gold_values增益<0.1归类；这些类别是操作性诊断，不是内部计算机制识别。')
add('D按原始回答条件计数，保留分子分母；其权重口径不同于族/根平衡的主指标。s_A/s_J/s_G及D列均以百分比显示。')
table(['预测','冻结分组：正组 − 负组','预期方向','实际效应pp','95% CI','解释'],[[r['hypothesis'],','.join(r.get('positive_families',[]))+' − '+','.join(r.get('negative_families',[])),r.get('predicted_direction','undefined'),fmt(r['effect_pp']) if 'effect_pp' in r else 'undefined',ci(r['CI95_pp']) if 'CI95_pp' in r else 'undefined',('方向相反' if not r.get('point_estimate_matches_direction',False) else '方向符合但区间跨零')] for r in d['predictions']])
add('P1比较J−A在高档与低档的差，P2比较GATE−J高档与低档的差，P3比较GATE−J在计算瓶颈与读取瓶颈的差。终点减共同起点后比较臂间增益，共同起点代数抵消。次要95%区间未额外多重校正，不作为选择奖励或改模型依据。')
add('\n## 5. 训练拟合与泛化的区别\n')
fit=read(HERE.parent/'SR_F1_2_THREE_SEED_TRAIN_FIT/ANALYSIS.json')
table(['臂','step32 J','step64 J','step96 J','step96三个seed J'],[[a]+[fmt(np.mean([fit['seeds'][str(s)]['arms'][a]['fit'][str(step)]['balanced_means']['J'] for s in [71001,71002,71003]])*100) for step in [32,64,96]]+['/'.join(fmt(fit['seeds'][str(s)]['arms'][a]['fit']['96']['balanced_means']['J']*100) for s in [71001,71002,71003])] for a in arms])
add(f"共同起点同一32题FIT面板的根/族平衡J为{avg('共同起点','TRAIN_FIT'):.2f}%。上述拟合表采用层次平衡权重；与此前128条原始均值表不同，不能混用。拟合来源4,608条已核验记录，全部三个seed保留；它只用于学习曲线和训练可解释性，不替代TEST泛化。")
add('\n## 6. 普通问答迁移与三类偏移\n')
table(['模型','纯答案A','组合J','语言J','外观J'],[[a,fmt(avg(a,'TEST_ID','answer_only','A'))]+[fmt(avg(a,p)) for p in ['TEST_COMPOSITION','TEST_LANGUAGE','TEST_RENDER']] for a in ['共同起点']+arms])
table(['面板','指标','对比','效应pp','次要95% CI'],[[r['pool'],r['metric'],r['contrast'],fmt(r['effect_pp']),ci(r['CI95_pp'])] for r in d['secondary'] if (r['pool']=='TEST_ID' and r['protocol']=='answer_only' and r['contrast'] in ['GATE-J','GATE-DEC','GATE-BASELINE']) or (r['pool']!='TEST_ID' and r['metric']=='J' and r['contrast'] in ['GATE-J','GATE-DEC'])])
add('纯答案协议独立生成K2，不能从证据协议抽取answer代替。各臂相对共同起点有一定纯答案提升，但没有复制证据协议的大幅收益；GATE不在所有协议和偏移面板领先。组合偏移的GATE−DEC劣势应与主终点并列保留。完整A/J、四对比及各臂对起点的次要根区间见[迁移结果](TRANSFER_RESULTS.json)。')
add('\n## 7. ChartQA保持性与评分口径异常\n')
add('本节首先列出冻结评分器的正式relaxed accuracy。2,500题按题等权，区间按1,509个图像簇配对重抽样，整簇保留所有问题与模型；human/augmented各1,250题。')
def chartmean(a,s,k):
    rr=[r for r in d['chartqa_absolute'] if r['split']==s and (r['model']==base if a=='共同起点' else r['model'].startswith('SRF1_2_'+a+'_'))]
    return np.mean([r['metrics'][k] for r in rr])*100
table(['模型','human relaxed','augmented relaxed','合计 relaxed','正式exact'],[[a]+[fmt(chartmean(a,s,'relaxed_accuracy')) for s in ['human','augmented','all']]+[fmt(chartmean(a,'all','exact_match'))] for a in ['共同起点']+arms])
table(['臂−起点','relaxed变化pp','三个seed变化pp','次要95%图像簇CI'],[[r['arm'],fmt(r['effect_pp']),'/'.join(fmt(v) for v in r['seed_effects_pp']),ci(r['CI95_pp'])] for r in d['chartqa_changes'] if r['split']=='all' and r['metric']=='relaxed_accuracy'])
add('冻结评分器只执行`.strip(" ")`，不会剥离尾换行。数值relaxed分支通过float()仍可接受换行，字符串exact及非数值relaxed则受影响。**全部模型正式exact=0不具能力鉴别力**；正式relaxed也是该特定归一化口径下的结果。独立审核保留原评分，并另提供只去首尾空白的事后敏感性分析；它不能替换冻结正式成绩，也不能变成重新选定的主终点。')
if (HERE/'CHARTQA_SENSITIVITY_SECTION_zh.md').exists():add('\n'+(HERE/'CHARTQA_SENSITIVITY_SECTION_zh.md').read_text())
add('\n来源固定为ChartQA数据revision `af8b6f5c08c95085271561c2a3f9d15f2b5a9031`、UniChart评分revision `56dd59af1c3ebec1773fe8a4bf43694cd565130c`、评分源SHA256 `22f627ab2396fc43c3edaf568148ea6c1f3edfbbc592fdef14668bfdf2b5e4ab`。数值容差为相对5%，零真值按作者字符串回退；不做百分号换算、数值提取或答案修复。不声称无预训练污染，不套用旧F2的保持认证。')
add('\n## 8. 结构、诊断视图及奖励信用\n')
if (HERE/'DIAGNOSTIC_SECTION_zh.md').exists():
    add('\n'.join(('### '+line[3:] if line.startswith('## ') else line)
                  for line in (HERE/'DIAGNOSTIC_SECTION_zh.md').read_text().splitlines()[1:]))
else:add('辅助分析正在独立核验，本文尚未作为最终交付。')
add('\n## 9. 训练稳定性、成本与技术事件\n')
if (HERE/'TRAINING_SECTION_zh.md').exists():add('\n'.join((HERE/'TRAINING_SECTION_zh.md').read_text().splitlines()[1:]))
add('\n历史技术事件完整保留：初始TECHNICAL196825因batch/single数值门槛失败且无optimizer更新；候选恢复曾因旧scontrol记录被清除失败，后经真实sacct及不可变提交身份验证修复；最终TECHNICAL196877通过8步及独立进程step4恢复。旧主CPU196875和端点196876是授权切换而有意取消，不是训练失败；主CPU198167末端BLOCKED_ENDPOINT_EVALUATION_NOT_CONFIGURED是旧末端边界，由独立端点接续，不得伪写为训练失败或全部流程一直无异常。VPN/SSH中断按未知处理，未导致重复提交。')
add('\nHDD只接管预先登记的后两种子八条checkpoint路径，原链接未重指；终验显式访问真实目标。滚动调度、提前封存评价和checkpoint存储均有独立源码、CPU验证及一次性回执，不能重跑这些部署脚本。科学冻结未据中途拟合或终点评价调节。')
add('\n## 10. 可支持的结论与后续建议\n')
add('本轮支持“在固定模型、任务生成器、预算和三条训练重复上，联合事件反馈相对答案奖励改善未见联合正确率；GATE相对单J进一步改善”。本轮不足以支持“分层普遍胜过DEC”“测量预测成功指导了奖励选择”“内部感知与计算已被因果分离”或“外部能力安全保持”。正确信息下仍失败仅称任务执行失败。')
add('\n若继续研究，应先在新登记中解决ChartQA空白归一化口径，并检验GATE−DEC的种子异质性、组合偏移劣势及预测失败。这里是建议，未自动增加任何训练、种子、奖励、样本或后续实验。')
add('\n## 11. 复现与证据索引\n')
add('入口：[START_HERE_zh.md](START_HERE_zh.md)。机器结果包括主对比、完整长表、预测、迁移、ChartQA、训练支持/失败、GPU计账、诊断与结构、原始输出索引及独立评分审计。`analyze_released.py`严格要求最终验收、13个release对象hash、records hash及授权一致后才统计；文件字节SHA与JSON对象hash分别标注。')
add('\n原始回答、模型权重、完整训练checkpoint留在服务器原路径（含HDD目标），本地交付包含已释放逐条分数、审计元数据、脚本、冻结证据和内容哈希，**不是自包含重训练包**。完整原始数据未删除；索引允许定位复核。包内SHA256清单及ZIP逐项读取验证用于校验传输完整性。')
(HERE/'FINAL_REPORT_zh.md').write_text('\n'.join(lines)+'\n')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['svg.hashsalt']='SR-F1.2-fixed-final-report'
fig,(ax,bx)=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
for i,r in enumerate(d['primary']):
    low,high=r['CI98_75_pp'];x=r['effect_pp']
    ax.errorbar(x,3-i,xerr=[[x-low],[high-x]],fmt='o',color='#165D86',capsize=5)
    ax.text(high+.4,3-i,f'{x:.2f}',va='center',fontsize=9)
ax.axvline(0,color='#777777',lw=1);ax.set_yticks(range(4),[r['contrast'] for r in d['primary']][::-1]);ax.set_xlabel('Joint accuracy difference (percentage points)');ax.set_title('Primary contrasts: 98.75% root CIs')
for a in arms:bx.plot([1,2,3],panel(a,'TEST_ID'),marker='o',label=a)
bx.set_xticks([1,2,3],['71001','71002','71003']);bx.set_ylabel('TEST_ID joint accuracy (%)');bx.set_xlabel('Training seed');bx.legend(ncol=2);bx.set_title('All three realised seeds');bx.grid(alpha=.2)
fig.savefig(HERE/'PRIMARY_FIGURE.svg',metadata={'Date':None});fig.savefig(HERE/'PRIMARY_FIGURE.png',dpi=180);plt.close(fig)
svg=HERE/'PRIMARY_FIGURE.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
