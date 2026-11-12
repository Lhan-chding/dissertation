#!/usr/bin/env python3
"""Root-paired primary analysis; never substitutes missing responses with zero.

Input: 128 roots x 2 starts, each with exact integers start_correct,T0_correct,
T1_correct and draws=2 for the conflict view. Unknown selected IS an observed
failure and stays in the assigned denominator; a missing raw record is an error.
CI is conditional on these four trained paths. Extra run-noise bands are explicitly
hypothetical sensitivity analyses, NOT empirically calibrated confidence intervals.
"""
from __future__ import annotations
import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from satdose_core import STARTS, STRUCTURES, effect_classification, quantile, read_jsonl, require, write_json


def analyze(rows: list[dict], B: int=10000, seed: int=2026111219) -> dict:
    require(len(rows)==256 and B>=100,'need exactly 128 roots x 2 starts')
    byroot=defaultdict(dict);strata=defaultdict(list);rootst={}
    perstart=defaultdict(list);gains=defaultdict(list)
    for r in rows:
        root,st,s=r['root_id'],r['structure'],r['start_seed']
        require(st in STRUCTURES and s in STARTS and r['draws']==2,'invalid primary allocation')
        require(rootst.setdefault(root,st)==st and s not in byroot[root], 'duplicate root/start or mixed stratum')
        for k in ('start_correct','T0_correct','T1_correct'):
            require(type(r[k]) is int and 0<=r[k]<=2,'wrong/missing primary numerator')
        d=(r['T1_correct']-r['T0_correct'])/2
        gain=(r['T1_correct']-r['start_correct'])/2
        byroot[root][s]=(d,gain);perstart[s].append(d);gains[s].append(gain)
    for root,x in byroot.items():
        require(set(x)==set(STARTS),'unpaired starting checkpoints')
        strata[rootst[root]].append((mean(x[s][0] for s in STARTS),mean(x[s][1] for s in STARTS)))
    require(all(len(strata[st])==32 for st in STRUCTURES),'need 32 roots per structure')
    rng=random.Random(seed);boot=[];gainboot=[]
    for _ in range(B):
        d=g=0.
        for st in STRUCTURES:
            vals=strata[st]
            samples=[vals[rng.randrange(32)] for _ in range(32)]
            d+=sum(v[0] for v in samples)/128;g+=sum(v[1] for v in samples)/128
        boot.append(d);gainboot.append(g)
    point=mean(mean(v[0] for v in strata[st]) for st in STRUCTURES)
    ci=[quantile(boot,.025),quantile(boot,.975)]
    sens=[]
    for sd in (0.,.03,.05):
        rr=random.Random(seed+round(sd*10000)+1)
        vals=[x+rr.gauss(0,sd/math.sqrt(2)) for x in boot] if sd else boot
        sens.append(dict(assumed_single_pair_run_SD=sd,
                         interval95=[quantile(vals,.025),quantile(vals,.975)],
                         scope='HYPOTHETICAL_ADDITIVE_RUN_NOISE_SENSITIVITY_NOT_FITTED_CI'))
    return dict(status='ANALYZED_NO_AUTOMATIC_FOLLOWUP_AUTHORIZED',primary_metric='S_hit_all_assigned',
                estimate=point,conditional_root_bootstrap_CI95=ci,
                classification=effect_classification(*ci),SESOI=.10,
                per_start_delta={str(s):mean(perstart[s]) for s in STARTS},
                per_start_absolute_gain={str(s):mean(gains[s]) for s in STARTS},
                absolute_gain_CI95=[quantile(gainboot,.025),quantile(gainboot,.975)],
                both_starts_positive=all(mean(perstart[s])>0 for s in STARTS),
                both_starts_absolute_gain_positive=all(mean(gains[s])>0 for s in STARTS),
                run_noise_sensitivity=sens,roots=128,starts=2,draws_per_view=2,B=B,bootstrap_seed=seed,
                scope='Fixed b0 R24 forks, fixed trained paths and CRN draws; no model/seed-population claim')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--rows',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path);a=ap.parse_args()
    require(not a.out.exists(),'refuse to overwrite analysis')
    result=analyze(list(read_jsonl(a.rows)));write_json(a.out,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
