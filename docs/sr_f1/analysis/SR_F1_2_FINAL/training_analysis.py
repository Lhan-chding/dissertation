"""Reproduce training diagnostics and allocation cost without reading sealed outputs."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / 'src'))
from sr_f12.analysis import mechanism_check
from sr_f12.protocol import ARMS, SEEDS


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def stats(values):
    a = np.asarray(values, dtype=float)
    assert a.size and np.isfinite(a).all()
    return dict(count=int(a.size), mean=float(a.mean()), median=float(np.median(a)),
                p99=float(np.quantile(a, .99)), minimum=float(a.min()), maximum=float(a.max()))


def summarize(rows):
    keys = ['step_seconds', 'sampling_seconds', 'training_seconds', 'kl',
            'sampler_train_mean_absolute_difference', 'sampler_train_p99_absolute_difference',
            'sampler_train_max_absolute_difference', 'tis_truncated_fraction',
            'total_gradient_norm', 'parameter_max_absolute_change', 'peak_allocated_bytes',
            'token_count', 'format_failure_rate']
    out = {k: stats([r[k] for r in rows]) for k in keys}
    out.update(updates=len(rows), rollout_sequences=sum(r['sequence_count'] for r in rows),
               completion_tokens=sum(r['token_count'] for r in rows),
               format_failures=sum(round(r['format_failure_rate'] * 128) for r in rows),
               recorded_step_hours=sum(r['step_seconds'] for r in rows)/3600,
               finite_updates=sum(r['finite'] for r in rows),
               parameter_changed_updates=sum(r['parameters_changed'] for r in rows),
               nonzero_adam_updates=sum(r['adam_nonzero_moments'] for r in rows),
               gradient_clipped_updates=sum(r['total_gradient_norm'] > 1 for r in rows),
               zero_advantage_updates=0, nonconstant_advantage_groups=0,
               advantage_groups=len(rows)*16, branch_counts=dict(Counter(
                   b for r in rows for b in r['reward_advantage_audit']['branches'])))
    advantages = []
    dec_scale = []
    for r in rows:
        a = np.asarray(r['reward_advantage_audit']['final_advantage'])
        assert a.shape == (16, 8) and np.isfinite(a).all()
        out['zero_advantage_updates'] += int(np.all(a == 0))
        out['nonconstant_advantage_groups'] += int(np.count_nonzero(np.ptp(a, axis=1)))
        advantages.extend(a.ravel().tolist())
        if r['arm'] == 'DEC':
            audit = r['reward_advantage_audit']
            pre = np.asarray(audit['pre_batch_advantage'])
            assert np.allclose(a, (pre-pre.mean())/(pre.std()+1e-8), atol=1e-12, rtol=1e-10)
            dec_scale.append(audit['pre_batch_std'])
    out['final_advantage'] = stats(advantages)
    out['mean_absolute_advantage'] = float(np.mean(np.abs(advantages)))
    out['final_advantage_rms'] = float(np.sqrt(np.mean(np.square(advantages))))
    out['DEC_pre_batch_std'] = stats(dec_scale) if dec_scale else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence', type=Path, default=REPO/'artifacts/sr_f12_final_20261012/evidence')
    ap.add_argument('--output', type=Path, default=Path(__file__).parent)
    args = ap.parse_args()
    r, outdir = args.evidence, args.output
    outdir.mkdir(parents=True, exist_ok=True)
    all_rows, paths, branches, inputs, processes = [], [], [], {}, []
    models = [f'SRF1_2_{arm}_s{seed}' for arm in ARMS for seed in SEEDS]
    for model in models:
        d = r/'runs'/model
        manifest = read(d/'RUN_MANIFEST.json')
        seed, arm = manifest['seed'], manifest['arm']
        rows = sorted([read(p) for p in (d/'steps').glob('*.json')], key=lambda x: x['step'])
        assert [x['step'] for x in rows] == list(range(1,97)), model
        assert manifest['gpu_count']==1 and not manifest['technical_only']
        for p in list((d/'steps').glob('*.json'))+[d/'RUN_MANIFEST.json']+list(d.glob('PROCESS-*.json')):
            inputs[str(p.relative_to(r))] = sha(p)
        pp = [read(p) for p in d.glob('PROCESS-*.json')]
        assert pp and all(p['checkpoint_field_hashes']==p['restored_field_hashes'] for p in pp)
        processes.extend(dict(model_id=model, **p) for p in pp)
        for row in rows:
            assert row['arm']==arm and row['sequence_count']==128
            assert row['finite'] and row['optimizer_updates']==1 and row['microbatch_size']==1
            assert row['parameters_changed'] and row['adam_nonzero_moments']
            assert row['backward_calls']==128 and row['tis_cap']==2 and row['clip_fraction']==0
            assert row['sampler_train_mean_absolute_difference']<=.01 and row['tis_truncated_fraction']<=.01
            assert abs(row['format_failure_rate']*128-round(row['format_failure_rate']*128))<1e-9
            if arm=='GATE':
                bb = row['reward_advantage_audit']['branches']
                assert len(bb)==16
                if row['step']<=8:
                    branches.extend(dict(arm=arm,seed=seed,step=row['step'],prompt_index=i,branch=b)
                                    for i,b in enumerate(bb))
        all_rows.extend(rows)
        paths.append(dict(model_id=model, seed=seed, arm=arm,
                          job_ids=sorted(set(p['job_id'] for p in pp)), **summarize(rows)))
    assert len(all_rows)==1152
    predpath = r/'SEMANTIC_PREDICTIONS.json'
    inputs['SEMANTIC_PREDICTIONS.json'] = sha(predpath)
    mechanism = mechanism_check(branches, read(predpath))
    accounting, job_tasks = [], defaultdict(set)
    for p in sorted((r/'accounting').glob('*.jsonl')):
        counts, jobs, events = Counter(), set(), 0
        for line in p.open():
            rec=json.loads(line); counts[rec['kind']]+=rec['count']; jobs.add(str(rec['slurm_job_id'])); events+=1
        for job in jobs: job_tasks[job].add(p.stem)
        accounting.append(dict(task=p.stem,job_ids=sorted(jobs),events=events,counts=dict(counts)))
        inputs[str(p.relative_to(r))]=sha(p)
    fields=['job_id','job_name','state','exit_code','submit_utc','start_utc','end_utc','elapsed_seconds',
            'alloc_tres','qos','nodes']
    slurm_path=Path(__file__).parent/'SLURM_GPU_ACCOUNTING.psv'
    slurm=[]
    for line in slurm_path.read_text().splitlines():
        x=line.split('|'); assert len(x)==len(fields)
        record=dict(zip(fields,x)); record['elapsed_seconds']=int(record['elapsed_seconds'])
        tres=dict(v.split('=',1) for v in record['alloc_tres'].split(','))
        record['gpu_count']=int(tres['gres/gpu']); assert record['gpu_count']==1
        record['gpu_hours']=record['elapsed_seconds']*record['gpu_count']/3600
        record['queue_seconds']=(datetime.fromisoformat(record['start_utc'])-datetime.fromisoformat(record['submit_utc'])).total_seconds()
        jid=record['job_id']; assert jid in job_tasks
        record['accounting_tasks']=sorted(job_tasks[jid])
        record['category']=('scientific_training' if any(t in models for t in job_tasks[jid]) else
                            'endpoint_evaluation' if any(t.startswith('endpoint-') for t in job_tasks[jid]) else
                            'baseline_initial' if any(t.startswith('baseline-') for t in job_tasks[jid]) else
                            'technical_failure' if record['state']=='FAILED' else 'technical_and_diagnostic')
        if record['category'] in ('scientific_training','endpoint_evaluation','baseline_initial'):
            assert record['state']=='COMPLETED' and record['exit_code']=='0:0'
        slurm.append(record)
    assert set(job_tasks)=={x['job_id'] for x in slurm} and len(slurm)==31
    totals=Counter()
    for x in slurm: totals[x['category']]+=x['gpu_hours']
    counts=Counter()
    for x in accounting: counts.update(x['counts'])
    for x in accounting:
        if x['task'] in models:
            assert x['counts']['completion_attempts']==96*128+3*128
            assert x['counts']['extra_forward_sequences']==96*128*2
        if x['task'].startswith('endpoint-'):
            assert x['counts']['completion_attempts']==(16452 if x['task'].endswith('COMMON_ZERO') else 17604)
    assert sum(x['counts']['completion_attempts'] for x in accounting if x['task'].startswith('baseline-'))==6656
    assert sum(x['counts']['completion_attempts'] for x in accounting if x['task'].startswith('endpoint-'))==227700
    gpu=dict(status='COMPLETE_FOR_ACCOUNTED_SR_F1_2_GPU_JOBS',
             source='TZ=UTC sacct -X; explicit 31 project job IDs only',
             source_file_sha256=sha(slurm_path),jobs=slurm,job_count=len(slurm),
             gpu_hours_by_category=dict(totals),total_gpu_hours=sum(totals.values()),
             accounting_counters=dict(counts),accounting_files=accounting,
             allocated_job_wall_span_hours=(max(datetime.fromisoformat(x['end_utc']) for x in slurm)-min(datetime.fromisoformat(x['start_utc']) for x in slurm)).total_seconds()/3600,
             limitations=['GPU hours are allocation time, not GPU active utilization or energy.',
                          'CPU maintenance/controller jobs and user-side work are outside GPU cost.',
                          'Prior SR-F1/SR-F1.1 experiments are excluded; no unrelated user jobs inspected.',
                          'Counters are retained SR-F1.2 accounting events, not a full Transformer-forward count.'])
    compact_steps=[]
    for model in models:
        for p in sorted((r/'runs'/model/'steps').glob('*.json')):
            row=read(p)
            compact_steps.append(dict(model_id=model,**{k:v for k,v in row.items() if k!='reward_advantage_audit'}))
    extreme_keys=['kl','sampler_train_mean_absolute_difference','sampler_train_p99_absolute_difference',
                  'sampler_train_max_absolute_difference','tis_truncated_fraction','total_gradient_norm']
    extrema={k:[dict(model_id=x['model_id'],step=x['step'],value=x[k])
                for x in sorted(compact_steps,key=lambda a:a[k],reverse=True)[:5]] for k in extreme_keys}
    result=dict(status='ALL_1152_SCIENTIFIC_UPDATES_AUDITED',paths=paths,extrema=extrema,
                overall=summarize(all_rows), by_arm={a:summarize([x for x in all_rows if x['arm']==a]) for a in ARMS},
                mechanism=mechanism,process_restoration_evidence=processes,
                step_metrics=compact_steps,input_file_sha256=inputs,
                limitations=['Per-step sampler p99 summaries are not a pooled-token p99.',
                             'Step mean sampler differences are averaged over steps in summary; each step value is token-level.',
                             'parameter_max_absolute_change is one-update change, not distance from initial weights.',
                             'total_gradient_norm is the preclip combined policy+KL norm, not pure policy gradient.',
                             'step_seconds ends before checkpoint/FIT; allocation time includes those and setup/cleanup.',
                             'GATE branch rates and advantage differences do not identify a causal submechanism.',
                             'Hardware co-tenancy, node placement and scheduling differ; timing is descriptive, not an isolated feedback-runtime experiment.',
                             'DEC tiny floating-point residuals are retained; exact zero and constant-within-group differ.',
                             'No sealed raw evaluation file is read by this script.'])
    for name,data in [('TRAINING_SUPPORT_AND_FAILURES.json',result),('GPU_ACCOUNTING.json',gpu)]:
        (outdir/name).write_text(json.dumps(data,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')
    lines=['# SR-F1.2 训练、机制与实际成本','',
           f"12条路径、1152次更新日志全部完整，sequence_count合计147,456条训练rollout。全程微批1、每步128次backward和一次Adam，所有更新有限、参数发生变化，Adam矩非零。",'',
           '| 路径 | GPU小时 | 步均耗时秒 | 训练格式失败/12288 | 最后一步KL | sampler逐步均差最大 | TIS截断率最大 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    byjob={x['job_id']:x for x in slurm}
    for x in paths:
        last=read(r/'runs'/x['model_id']/'steps'/'96.json')
        hours=sum(byjob[j]['gpu_hours'] for j in x['job_ids'])
        lines.append(f"| {x['arm']} {x['seed']} | {hours:.3f} | {x['step_seconds']['mean']:.1f} | {x['format_failures']} | {last['kl']:.6f} | {x['sampler_train_mean_absolute_difference']['maximum']:.6f} | {x['tis_truncated_fraction']['maximum']:.6f} |")
    ov=result['overall']; m=mechanism
    lines += ['',f"训练格式失败总计 {ov['format_failures']}/147456（{ov['format_failures']/147456*100:.3f}%）；所有失败保留，未作为删除样本的依据。最大逐步sampler平均绝对差 {ov['sampler_train_mean_absolute_difference']['maximum']:.6f} nat，低于0.01门槛；最大单token差 {ov['sampler_train_max_absolute_difference']['maximum']:.6f} nat，最大逐步p99 {ov['sampler_train_p99_absolute_difference']['maximum']:.6f} nat。单token最大差不是技术微批TF对照的最大差，不能拿0.05 TF门槛混用。TIS截断比例最大 {ov['tis_truncated_fraction']['maximum']:.6f}。",'',
              f"全程KL范围 {ov['kl']['minimum']:.6f}–{ov['kl']['maximum']:.6f}；combined policy+KL梯度范数范围 {ov['total_gradient_norm']['minimum']:.6f}–{ov['total_gradient_norm']['maximum']:.6f}，{ov['gradient_clipped_updates']}步超过1而触发梯度裁剪。每步参数最大变化范围 {ov['parameter_max_absolute_change']['minimum']:.8f}–{ov['parameter_max_absolute_change']['maximum']:.8f}；这不是最终权重对起点距离。",'',
              f"预登记机制核对：GATE三种子前8步384组，纯语义分支 {m['pure_group_count']}/384={m['observed']*100:.3f}%，冻结预测 {m['expected']*100:.3f}%，差 {m['difference_pp']:+.3f}个百分点；±10pp条件{'符合' if m['within_registered_ten_pp'] else '不符合'}。这是次要描述核对，不是训练或科学有效性PASS。",'',
              '| 臂 | 组内最终优势非恒定组/4608 | 平均绝对优势 | RMS优势 |', '|---|---:|---:|---:|']
    for arm,x in result['by_arm'].items():
        lines.append(f"| {arm} | {x['nonconstant_advantage_groups']}/4608 | {x['mean_absolute_advantage']:.5f} | {x['final_advantage_rms']:.5f} |")
    lines += ['', 'KL最大尖峰为GATE71002第18步0.272943，DEC71003第89步为0.252012，均保留。0.05只用于技术试跑第8步学习率规则，不是科学全程KL停止线；不能据此宣称全程无尖峰或KL保证能力保持。sampler最大单token差出现在J71002第30步。各指标前5极值及路径/步号见机器表extrema。', '',
              '优势尺度、有效组比例及分支构成只描述信用信号，不等价于梯度幅度或能力收益。DEC全部288步最终优势已按保存的组合优势与批次总体标准差重算一致；未删常数组，也未重设批次。GATE全程分支计数见机器表。', '',
              '实际Slurm分配成本如下（UTC显式查询，仅31个本项目GPU作业；每作业1张PRO6000）：','',
              '| 类别 | GPU小时 |','|---|---:|']
    for k,v in totals.items(): lines.append(f'| {k} | {v:.3f} |')
    lines += [f"| 总计 | {gpu['total_gpu_hours']:.3f} |",'',
              f"从首个本轮GPU开始到最后评价GPU结束的墙钟跨度 {gpu['allocated_job_wall_span_hours']:.3f}小时；CPU最终统一释放与报告时间不在这个跨度内。节点、同机并行和调度不同，因此各臂耗时差只能作实际成本描述，不能归因于奖励算法本身。",'',
              f"逐步记录合计 {ov['recorded_step_hours']:.3f}小时，小于科学训练分配 {totals['scientific_training']:.3f} GPU小时；前者不含checkpoint、FIT、模型加载、部分校验与清理，不能当完整成本。GPU分配小时也不表示GPU始终满载，更不是能耗。",'',
              '已保留初始技术作业196825失败与只读诊断196834成本；正式技术196877成功。旧控制器的预期接续边界以及已授权取消控制器不属于科学训练失败。CPU维护/控制器与历史旧实验成本不混入本轮GPU表。', '',
              f"保留的accounting累计 completion_attempts={counts['completion_attempts']:,}、generated_tokens={counts['generated_tokens']:,}、extra_forward_sequences={counts['extra_forward_sequences']:,}。这些是事件计数，不是完整模型前向次数。科学训练每路径12,672次生成=12,288训练rollout+384 TRAIN_FIT；12条训练rollout合计147,456应与238,964条协议测量分开记账。",'',
              '复算输入hash与1152步标量详表在TRAINING_SUPPORT_AND_FAILURES.json；31作业原始Slurm记录在SLURM_GPU_ACCOUNTING.psv，分项与计数在GPU_ACCOUNTING.json。脚本未读取任何封存评价原文。']
    (outdir/'TRAINING_SECTION_zh.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(updates=len(all_rows),mechanism=mechanism,gpu_hours=gpu['total_gpu_hours'],format_failures=ov['format_failures'])))

if __name__ == '__main__': main()
