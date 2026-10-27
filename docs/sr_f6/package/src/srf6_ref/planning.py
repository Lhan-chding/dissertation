"""Registered path counts, paired RNG streams, and one-pass root scheduling."""
from __future__ import annotations
import random
from .core import FAMILIES, seed_for

ARMS={'U00':('P0','J'),'U01':('P0','COND'),'U10':('P1','J'),'U11':('P1','COND'),
      'C_GLOBAL':('P1','GLOBAL'),'C_SHUF':('P1','SHUF')}

def root_schedule(roots_per_family:int,master:int,repeat:int)->list[list[str]]:
    if roots_per_family%8:raise ValueError('roots_per_family must be multiple of 8')
    rng=random.Random(seed_for(master,'task_order',repeat));out=[]
    for window in range(roots_per_family//8):
        roots=[f'TRAIN_{f}_{i:04d}' for f in FAMILIES for i in range(window*8,(window+1)*8)]
        rng.shuffle(roots)
        out.extend([roots[i:i+4] for i in range(0,len(roots),4)])
    return out

def draw_seed(master:int,parent:str,repeat:int,step:int,slot:int,variant:int,draw:int)->int:
    # Deliberately no arm/factor: common random numbers do not imply identical rollouts after updates.
    return seed_for(master,'draw',parent,repeat,step,slot,variant,draw)

def paths(cfg:dict)->list[dict]:
    return [{'parent':p,'repeat':r,'arm':a,'factor':ARMS[a][0],'credit':ARMS[a][1],
             'run_id':f"{cfg['experiment_id']}/{p}/r{r}/{a}"}
            for p in cfg['model']['parents'] for r in cfg['training']['repeats'] for a in cfg['arms']]

def budget(cfg:dict)->dict:
    npaths=len(paths(cfg));t=cfg['training'];train_each=t['updates']*t['groups_per_update']*t['draws_per_group']
    eval_each=sum(v['questions']*v['draws'] for v in cfg['evaluation']['panels'].values())
    stage_counts={'parent_census_max':3*2*6*8*4*8,
                  'pilot_training':3*3*12*16*8,
                  'pilot_recovery_check':8*16*8,
                  'pilot_eval':(3+9)*96*2,
                  'formal_training':npaths*train_each,
                  'formal_dev_monitor':npaths*96,
                  'formal_eval':(npaths+3)*eval_each}
    return {'formal_paths':npaths,'core_paths':3*len(t['repeats'])*4,
            'control_paths':3*len(t['repeats'])*2,'answers_per_formal_path':train_each,
            'eval_answers_per_checkpoint':eval_each,'counts':stage_counts,
            'max_generated_answers':sum(stage_counts.values()),
            'note':'Census cache can reduce actual calls; recovery is validation, not an independent training replicate. Tokens/FLOPs/GPU hours are measured, never an outcome stopping rule.'}
