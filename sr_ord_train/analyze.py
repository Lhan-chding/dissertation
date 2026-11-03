#!/usr/bin/env python3
"""Frozen endpoint analysis. Offline only; no model calls. Requires Python 3.10+, numpy."""
import argparse
import json
import hashlib
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PARENTS = ("b0", "b1")
SEEDS = (11031, 11032)
ARMS = ("R", "G", "D")
STEPS = (12, 24)
KEY = ("panel", "root_id", "structure", "parent", "train_seed", "arm", "step", "view", "draw")
BASE = ("PARENT", 0, None)
N_BOOT = 10000
BOOT_SEED = 2026110317
STRUCTURES = ("2_choose_1", "3_choose_1", "3_choose_2", "4_choose_2")
SWAP = ("p0l0", "p0l1", "p1l0", "p1l1")
PANELS = {"transfer8", "swap8", "retention8", "context8", "scale10", "authority8", "prefix10"}
FAMILIES = ("BINDING", "IMPACT", "TIES", "SIGNED", "INTERVAL", "K_DIRECTION")
LAYOUT = {
    "transfer8":(96, ("aligned","conflict"),2),
    "swap8":(32, SWAP,2),
    "retention8":(24, ("v0","v1","v2","v3"),1),
    "context8":(24, ("base","nolist","list_conflict","report_conflict"),1),
    "scale10":(24, ("aligned","conflict"),1),
    "authority8":(24, ("chart","list"),1),
    "prefix10":(32, ("p_stable","p_revtie","p_placebo"),1),
}


def order_applies(panel, view):
    return panel in {"transfer8", "scale10"} or (panel == "context8" and view == "report_conflict")


def require(test, message):
    if not test:
        raise ValueError(message)


def unique_dict(pairs):
    d = {}
    for k, v in pairs:
        require(k not in d, f"duplicate JSON field {k}")
        d[k] = v
    return d


def bad_constant(x):
    raise ValueError("non-finite JSON numeric literal: " + x)


def load_rows(path):
    with Path(path).open(encoding="utf-8-sig") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, object_pairs_hook=unique_dict, parse_constant=bad_constant)
                require(isinstance(row, dict), "record must be an object")
                yield row
            except ValueError as err:
                raise ValueError(f"{path}:{i}: {err}") from err


def isbit(x):
    return type(x) is int and x in (0, 1)


def index_rows(rows, scores=False):
    out = {}
    for ix, r in enumerate(rows, 1):
        require(all(k in r for k in KEY), f"row {ix}: incomplete evaluation key")
        for k in ("panel", "root_id", "structure", "parent", "arm", "view"):
            require(isinstance(r[k], str) and bool(r[k].strip()), f"row {ix}: empty/invalid {k}")
        require(r["parent"] in PARENTS, f"row {ix}: bad parent")
        require(type(r["step"]) is int and type(r["draw"]) is int and r["draw"] >= 0, f"row {ix}: bad step/draw")
        if r["arm"] == "PARENT":
            require((r["arm"], r["step"], r["train_seed"]) == BASE, f"row {ix}: PARENT is recorded once at step 0, null train_seed")
        else:
            require(r["arm"] in ARMS and type(r["train_seed"]) is int and r["train_seed"] in SEEDS and r["step"] in STEPS,
                    f"row {ix}: bad arm, training seed or step")
        require(r["panel"] in PANELS, f"row {ix}: panel is not in the frozen inventory")
        require(type(r.get("U_applicable")) is bool and r["U_applicable"] == order_applies(r["panel"], r["view"]),
                f"row {ix}: U_applicable must match predeclared instruction status for the panel/view")
        key = tuple(r[k] for k in KEY)
        require(key not in out, f"duplicate evaluation key {key}")
        if scores:
            require(all(k in r and isbit(r[k]) for k in ("A", "S_hit", "S_known")), f"{key}: A/S_hit/S_known require integer 0 or 1")
            require(r["S_known"] or not r["S_hit"], f"{key}: unknown S cannot be a confirmed hit")
            require("U" in r and (r["U"] is None or isbit(r["U"])), f"{key}: U must be null or integer 0 or 1")
            require(r["U_applicable"] or r["U"] is None, f"{key}: U is NA without an explicit listing instruction")
            if r["panel"]=="retention8":
                require(isbit(r.get("J_native")), f"{key}: retention must include historical native J")
            if r["panel"] in {"scale10", "prefix10"} or (r["panel"]=="authority8" and r["view"]=="list"):
                require(r.get("J_native") is None, f"{key}: this size/authority diagnostic cannot be labeled native J")
            for m in ("J_native", "J_diag"):
                require(r.get(m) is None or isbit(r[m]), f"{key}: invalid {m}")
        out[key] = r
    require(out, "empty manifest/scores")
    return out


def verify_design(rows, retention_panel):
    bypanel = defaultdict(lambda: defaultdict(set))
    root_stratum = {}
    for r in rows:
        p, rt, st = r["panel"], r["root_id"], r["structure"]
        require(root_stratum.setdefault((p, rt), st) == st, f"{p}/{rt}: root crosses strata")
        ep = (r["parent"], r["arm"], r["step"], r["train_seed"])
        bypanel[p][ep].add((rt, st, r["view"], r["draw"]))
    require(retention_panel=="retention8", "retention panel is fixed to retention8")
    require(set(bypanel)==PANELS, f"frozen panel inventory differs; needed={sorted(PANELS-set(bypanel))}")
    endpoints = {(b, a, step, sd) for b in PARENTS for a in ARMS for step in STEPS for sd in SEEDS}
    endpoints |= {(b, *BASE) for b in PARENTS}
    for p, cells in bypanel.items():
        require(set(cells) == endpoints, f"{p}: missing or extraneous parent/arm/step/train_seed endpoint, expected 26")
        support = next(iter(cells.values()))
        require(all(v == support for v in cells.values()), f"{p}: endpoints lack matching root/view/draw support")
        nro,vv,nd=LAYOUT[p]
        gr_full=defaultdict(set)
        for rt,st,v,d in support:gr_full[st,rt].add((v,d))
        require(len(gr_full)==nro and all(z=={(v,d) for v in vv for d in range(nd)} for z in gr_full.values()),
                f"{p}: fixed per-root view/draw matrix differs from preregistered design")
        if p!="retention8":
            require(dict(Counter(st for st,rt in gr_full))==dict.fromkeys(STRUCTURES,nro//4),
                    f"{p}: incorrect allocation across the four frozen boundary structures")
        if p not in ("transfer8", "swap8"):
            require(all(x[3]==0 for x in support),f"{p}: only draw 0 is registered")
        if p=="retention8":
            gr=defaultdict(set)
            for rt,st,view,dr in support: gr[st,rt].add(view)
            require(all(v=={"v0","v1","v2","v3"} for v in gr.values())
                    and dict(Counter(st for st,rt in gr))==dict.fromkeys(FAMILIES,4),
                    "retention8 requires six original families × four DEV_EVAL P0 roots × four variants × one draw")
        if p in ("transfer8", "swap8"):
            views = {"aligned", "conflict"} if p == "transfer8" else set(SWAP)
            grouped = defaultdict(set)
            for rt, st, view, d in support:
                grouped[st, rt].add((view, d))
            for srt, keys in grouped.items():
                require(keys == {(v,d) for v in views for d in (0,1)}, f"{p}/{srt}: missing view/draw or draw outside 0,1")
            if p == "transfer8":
                counts = Counter(st for st, _ in grouped)
                require(dict(counts) == dict.fromkeys(STRUCTURES, 24),
                        "primary requires 96 new roots: 24 in each of four frozen structures")
    return {p: {"n_endpoints": len(cells), "slots_per_endpoint": len(next(iter(cells.values()))),
                "n_roots": len({x[0] for x in next(iter(cells.values()))})} for p,cells in sorted(bypanel.items())}


def grouped(rs, fields):
    d=defaultdict(list)
    for r in rs:
        d[tuple(r[f] for f in fields)].append(r)
    return d


def weights(rs):
    """Equal parent / seed / stratum / root / view / draw, for rows of *one* endpoint or arm-step."""
    def visit(x, fields, weight):
        if not fields:
            for r in x:
                yield r, weight / len(x)
        else:
            g=grouped(x, (fields[0],))
            for xs in g.values():
                yield from visit(xs, fields[1:], weight / len(g))
    return list(visit(rs, ["parent", "train_seed", "structure", "root_id", "view"], 1.0))


def summarize(rs):
    require(rs, "cannot summarize empty group")
    wr=weights(rs)
    mask=lambda r: r["U"] is not None
    vals = {
        "A":lambda r:r["A"], "S_hit":lambda r:r["S_hit"], "S_known":lambda r:r["S_known"],
        "U_known":mask, "U_confirmed":lambda r:r["U"]==1, "A_and_U_confirmed":lambda r:r["A"]==1 and r["U"]==1,
        "A_and_U_unresolved":lambda r:r["A"]==1 and r["U"] is None
    }
    cnt={k:sum(int(f(r)) for r in rs) for k,f in vals.items()}
    avg={k:float(sum(w*float(f(r)) for r,w in wr)) for k,f in vals.items()}
    ratio=lambda x,y: None if not y else x/y
    applicable={r["U_applicable"] for r in rs}
    require(len(applicable)==1, "U applicability varies within reported panel/view; manifest invalid")
    uapp=applicable=={True}
    joint=avg["A_and_U_confirmed"]
    avg.update(S_pass_given_known=ratio(avg["S_hit"],avg["S_known"]), U_pass_given_known=ratio(avg["U_confirmed"],avg["U_known"]),
               A_and_U_given_U_known=ratio(joint,avg["U_known"]), A_and_U_finite_identification_bounds=[joint,joint+avg["A_and_U_unresolved"]])
    if not uapp:
        for k in ("U_known","U_confirmed","A_and_U_confirmed","A_and_U_unresolved"):
            cnt[k]=None
        for k in ("U_known","U_confirmed","A_and_U_confirmed","A_and_U_unresolved","U_pass_given_known","A_and_U_given_U_known","A_and_U_finite_identification_bounds"):
            avg[k]=None
    # Preserve any upstream optional historic native endpoint and diagnostic proxy separately.
    for m in ("J_native", "J_diag"):
        applicable=[(r,w) for r,w in wr if r.get(m) is not None]
        cnt[m+"_known"]=len(applicable)
        cnt[m+"_hit"]=sum(r[m] for r,w in applicable)
        avg[m+"_coverage"]=sum(w for r,w in applicable)
        avg[m+"_given_applicable"]=ratio(sum(w*r[m] for r,w in applicable),sum(w for r,w in applicable))
    return {"n_assigned":len(rs),"U_applicable":uapp,"unweighted_counts":cnt,"equal_weight_rates":avg}


def root_means(rs, metric):
    out={}
    for (p,sd,st,rt),g in grouped(rs, ("parent","train_seed","structure","root_id")).items():
        perview=grouped(g,("view",))
        require(all(r.get(metric) is not None for r in g), f"{metric}: cannot silently discard missing outcomes")
        out[p,sd,st,rt]=float(np.mean([np.mean([r[metric] for r in h]) for h in perview.values()]))
    return out


def stratum_mean(values):
    require(values,"empty root effect set")
    strata=defaultdict(list)
    for (st,rt),x in values.items():
        strata[st].append(x)
    return float(np.mean([np.mean(v) for v in strata.values()]))


def root_ci(values):
    require(values, "cannot form CI from zero roots")
    strata=defaultdict(dict)
    for (st,rt),x in values.items():
        require(math.isfinite(float(x)), "missing/nonfinite root effect")
        strata[st][rt]=x
    if any(len(v)<2 for v in strata.values()):
        return None, "insufficient_roots_in_a_stratum"
    rng=np.random.default_rng(BOOT_SEED)
    sim=np.zeros(N_BOOT)
    for st in sorted(strata):
        x=np.asarray([strata[st][r] for r in sorted(strata[st])],dtype=float)
        # Fixed stream per stratum; each draw selects the already paired root effect for all arms, parents and seeds.
        sim+=x[rng.integers(0,len(x),(N_BOOT,len(x)))].mean(axis=1)/len(strata)
    return np.quantile(sim,[.025,.975]).tolist(), "conditional_on_fixed_trained_paths_and_observed_draws"


def compare(rows,left,right,step=24,metric="A"):
    aa=[r for r in rows if r["arm"]==left and r["step"]==step]
    bb=[r for r in rows if r["arm"]==right and r["step"]==step]
    pairkey=lambda r:(r["parent"],r["train_seed"],r["structure"],r["root_id"],r["view"],r["draw"])
    require(aa and bb and len(aa)==len(bb) and len(aa)==len({pairkey(r) for r in aa})
            and {pairkey(r) for r in aa}=={pairkey(r) for r in bb}, "unmatched row-level root/view/draw/parent/seed keys")
    x,y=root_means(aa,metric),root_means(bb,metric)
    require(x and x.keys()==y.keys(), "contrast missing matched endpoint/root/seed")
    dk=defaultdict(dict)
    for (p,sd,st,rt),v in x.items():
        dk[p,sd][st,rt]=v-y[p,sd,st,rt]
    require(set(dk)=={(p,s) for p in PARENTS for s in SEEDS},"contrast requires all four paired parent-seed blocks")
    sup=next(iter(dk.values())).keys()
    require(all(g.keys()==sup for g in dk.values()),"root support differs across paired parent-seed blocks")
    d={k:float(np.mean([v[k] for v in dk.values()])) for k in sup}
    effect=stratum_mean(d);ci,status=root_ci(d)
    blocks={f"{p}:{s}":stratum_mean(dk[p,s]) for p in PARENTS for s in SEEDS}
    return {"arm":left, "reference":right, "step":step,"metric":metric,
            "estimate":effect,"ci95":ci,"ci_scope":status,"bootstrap_seed":BOOT_SEED,"B":N_BOOT,
            "n_roots":len(sup),"n_parent_seed_blocks":len(dk),"n_assigned_per_arm":len(aa),
            "paired_blocks":blocks,
            "by_parent":{p:float(np.mean([blocks[f"{p}:{s}"] for s in SEEDS])) for p in PARENTS},
            "by_train_seed":{str(s):float(np.mean([blocks[f"{p}:{s}"] for p in PARENTS])) for s in SEEDS},
            "training_seed_CI":None,"direction":"inconclusive" if not ci or ci[0]<=0<=ci[1] else ("left_better" if ci[0]>0 else "right_better")}


def retention_status(ci,margin=.02):
    if ci is None:
        return "inconclusive_no_conditional_interval"
    if ci[0]>-margin: return "conditional_noninferiority_supported"
    if ci[1]<-margin: return "observed_harm_exceeds_margin"
    return "inconclusive_interval_touches_or_crosses_harm_margin"


def equivalence_status(ci,preregistered=False,margin=.02):
    if not preregistered:return "not_preregistered_no_equivalence_claim"
    return "conditional_equivalence_supported" if ci is not None and -margin<ci[0] and ci[1]<margin else "inconclusive_or_not_equivalent"


def swap_summary(rs):
    """Deterministically pair the *same draw index* across four visual/list views, not any-of-two draws."""
    out=[]
    for (arm,step),eg in grouped(rs,("arm","step")).items():
        cases=grouped(eg,("parent","train_seed","structure","root_id","draw"))
        results=[]
        for (p,sd,st,rt,d),g in cases.items():
            v={r["view"]:r["A"] for r in g}
            require(len(g)==4 and set(v)==set(SWAP),"incomplete or duplicate swap quadruple")
            m={"all_four":int(all(v.values())),"p0_both_lists":v["p0l0"]*v["p0l1"],"p1_both_lists":v["p1l0"]*v["p1l1"],
               "l0_both_prints":v["p0l0"]*v["p1l0"],"l1_both_prints":v["p0l1"]*v["p1l1"]}
            results.append(dict(parent=p,train_seed=sd,structure=st,root_id=rt,draw=d,view="paired",**m))
        ww=weights(results)
        names=("all_four","p0_both_lists","p1_both_lists","l0_both_prints","l1_both_prints")
        out.append({"arm":arm,"step":step,"n_complete_same_draw_quadruples":len(results),
                    "unweighted_hits":{m:sum(r[m] for r in results) for m in names},
                    "equal_weight_rates":{m:sum(w*r[m] for r,w in ww) for m in names}})
    return out


def run(manifest,scores,retention_panel="retention8"):
    plan=index_rows(manifest); obs=index_rows(scores,True)
    a,b=set(plan),set(obs)
    require(a==b, f"manifest/scores mismatch missing={len(a-b)}, extra={len(b-a)}; example_missing={next(iter(a-b),None)}, example_extra={next(iter(b-a),None)}")
    inventory=verify_design(list(plan.values()),retention_panel)
    rows=list(obs.values())
    levels=[]
    for (panel,arm,step,view),g in sorted(grouped(rows,("panel","arm","step","view")).items()):
        levels.append({"panel":panel,"view":view,"arm":arm,"step":step,"parent":"pooled","train_seed":None,**summarize(g)})
        for (p,s),gg in sorted(grouped(g,("parent","train_seed")).items()):
            levels.append({"panel":panel,"view":view,"arm":arm,"step":step,"parent":p,"train_seed":s,**summarize(gg)})
    primary=compare([r for r in rows if r["panel"]=="transfer8" and r["view"]=="conflict"],"D","G")
    rr=[r for r in rows if r["panel"]==retention_panel]
    ret=[]
    for arm in ("D","G"):
        ob=compare(rr,arm,"R")
        ob["noninferiority_vs_R_with_minus_2pp_margin"]=retention_status(ob["ci95"])
        ret.append(ob)
    return {"status":"complete_fixed_inventory","inventory":inventory,
            "primary":{"panel":"transfer8","view":"conflict",**primary,"equivalence":equivalence_status(primary["ci95"])},
            "retention":{"panel":retention_panel,"margin":-.02,"metric":"A","comparisons":ret},
            "endpoint_levels":levels,"swap8_same_draw_answer_robustness":swap_summary([r for r in rows if r["panel"]=="swap8"]),
            "interpretation":"Intervals condition on two fixed parents, two fixed paired training seeds per parent and stored evaluations; they do not estimate training-seed, model-family or task-generator population uncertainty. Intermediate checkpoints and other panels are descriptive, not additional primary tests."}


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest",required=True)
    ap.add_argument("--scores",required=True)
    ap.add_argument("--manifest-sha256",help="If provided, require exactly this preregistered manifest file SHA256")
    ap.add_argument("--retention-panel",choices=["retention8"],default="retention8",help="Frozen historical native evaluation panel")
    ap.add_argument("--out",required=True)
    a=ap.parse_args(argv)
    try:
        digest=hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest()
        if a.manifest_sha256 is not None:
            require(digest==a.manifest_sha256.strip().lower(),"manifest SHA256 differs from prefrozen identity")
        result=run(load_rows(a.manifest),load_rows(a.scores),a.retention_panel)
        result["manifest_file_sha256"]=digest
        result["scores_file_sha256"]=hashlib.sha256(Path(a.scores).read_bytes()).hexdigest()
        path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")
    except (OSError,ValueError) as e:
        ap.exit(2, "ANALYSIS REFUSED: "+str(e)+"\n")


if __name__=="__main__": main()
