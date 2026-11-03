#!/usr/bin/env python3
"""Offline field-local evaluator. Run only after raw responses and plan identities have been saved."""
import argparse
import json
import os
import sys
import tempfile
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"vendor/probe_ref"))
sys.path.insert(0,str(ROOT/"vendor/native_src"))
from order_probe.common import digest,exact
from order_probe.measurement import extract
from sr_f6.measurement import score_response as native_score
import analyze as stat

require=stat.require
MONITOR=("A_right_S_wrong","A_right_B_wrong","A_right_S_or_B_wrong","A_right_S_unknown","A_right_B_unknown")
DEVKEY=("panel","root_id","structure","parent","view","draw")
DEVIEWS=("aligned_plain","aligned_clear","conflict_plain","conflict_clear")
PREFIX_KEYS=("A","S_hit","S_known","B","prefix_choice_hit","json_valid")


def ids(rows,field,fields):
    out={}
    for r in rows:
        require(isinstance(r.get(field),str) and r[field] not in out,"duplicate/invalid "+field)
        require(all(k in r for k in fields) and r[field]==digest({k:r[k] for k in fields}),"content digest mismatch: "+field)
        out[r[field]]=r
    return out


def index_requests(rows,keys):
    out={}
    for r in rows:
        require(all(k in r for k in keys), "incomplete raw/manifest identity")
        for k in keys:
            if k=="train_seed":
                require(r[k] is None or type(r[k]) is int,"invalid training seed type")
            elif k in ("draw","step"): require(type(r[k]) is int and r[k]>=0,f"invalid {k}")
            else: require(isinstance(r[k],str) and bool(r[k]),f"invalid {k}")
        tup=tuple(r[k] for k in keys)
        require(tup not in out, f"duplicate request key {tup}")
        out[tup]=r
    require(bool(out),"empty request set")
    return out


def validate_dev(plan):
    require(len(plan)==384,"clarification requires 384 rows: 24 roots x 4 cells x 2 draws x 2 parents")
    roots={}
    check=defaultdict(set)
    for m in plan:
        require(m["panel"]=="clarification","mixed evaluation/clarification manifests")
        rt,st=m["root_id"],m["structure"]
        require(roots.setdefault(rt,st)==st,"clarification root crosses strata")
        check[rt].add((m["view"],m["parent"],m["draw"]))
    require(dict(Counter(roots.values()))==dict.fromkeys(stat.STRUCTURES,6),"clarification must have six roots per structure")
    expected={(v,p,d) for v in DEVIEWS for p in stat.PARENTS for d in (0,1)}
    require(all(v==expected for v in check.values()),"clarification cell/root/draw support incomplete")


def exact_equal(x,y):
    try:return int(exact(x)==exact(y))
    except (ValueError,TypeError,ArithmeticError):return 0


def fields(text,t,audit,finish):
    obj,err=extract(text)
    complete=finish in {"eos","balanced"}
    result={"A":0,"A_parseable":0,"S":None,"S_known":0,"S_hit":0,"B":None,"B_known":0,"B_hit":0,
            "U":None,"U_applicable":audit.get("expected_report_order") is not None,"json_valid":int(obj is not None),
            "completed_generation":int(complete),"parse_error":err,"S_reason":None,"B_reason":None,"U_reason":None,
            "list_conflict":audit.get("list_conflict"),"list_choice_hit":None,"prefix_choice_hit":None,"report_choice_hit":None}
    names=t["world"]["categories"]
    if obj is not None and "answer" in obj:
        try:
            val=exact(obj["answer"])
            result["A_parseable"]=1
            result["A"]=int(val==exact(t["gold"]["answer"]))
        except (TypeError,ValueError,ArithmeticError):pass
    if obj is None or not complete:
        reason="INCOMPLETE_GENERATION" if not complete else err
        result.update(S_reason=reason,B_reason=reason,U_reason=reason if result["U_applicable"] else "NOT_APPLICABLE")
        return result
    sel=obj.get("selected")
    if isinstance(sel,list) and all(type(v) is str for v in sel):
        valid=len(set(sel))==len(sel)
        good=int(valid and set(sel)==set(t["gold"]["selected"]))
        result.update(S=good,S_known=1,S_hit=good)
        for k,m in [("list_induced_selected","list_choice_hit"),("prefix_induced_selected","prefix_choice_hit")]:
            target=audit.get(k)
            if target is not None:
                result[m]=int(valid and set(sel)==set(target))
        order=audit.get("expected_report_order")
        if order is not None and {"strictly_better_indices","boundary_indices"}<=set(t):
            r=t.get("r",t.get("required_from_boundary"))
            if r is not None:
                boundary={names[i] for i in t["boundary_indices"]}
                picked={names[i] for i in t["strictly_better_indices"]}|set([n for n in order if n in boundary][:r])
                result["report_choice_hit"]=int(valid and set(sel)==picked)
    else:result["S_reason"]="SELECTED_SHAPE_OR_MISSING"
    try:
        entries=obj["bindings"]
        if not isinstance(entries,list):raise ValueError("BINDINGS_SHAPE")
        got=[]
        for e in entries:
            if not isinstance(e,dict) or set(e)!={"category","operand"} or not isinstance(e["category"],str):raise ValueError("BINDING_ENTRY")
            got.append((e["category"],exact(e["operand"])))
        selected=t["gold"]["selected"]
        target=dict(zip(selected,map(exact,t["gold"]["operands"]),strict=True))
        good=int(len(got)==len(dict(got)) and dict(got)==target)
        result.update(B=good,B_known=1,B_hit=good)
    except (ValueError,TypeError,ArithmeticError,KeyError) as e:result["B_reason"]=str(e)
    if result["U_applicable"]:
        expected=audit["expected_report_order"]
        require(isinstance(expected,list) and sorted(expected)==sorted(names),"invalid expected report order in private immutable case")
        try:
            ev=obj["evidence"]
            if not isinstance(ev,list):raise ValueError("EVIDENCE_SHAPE")
            got=[]
            for e in ev:
                if not isinstance(e,dict):raise ValueError("EVIDENCE_ENTRY")
                if e.get("series")=="Alpha":
                    if type(e.get("category")) is not str or e["category"] not in names:raise ValueError("ALPHA_CATEGORY")
                    exact(e["value"]);got.append(e["category"])
            if len(got)!=len(names) or set(got)!=set(names):raise ValueError("ALPHA_NOT_FULL_UNIQUE")
            result["U"]=int(got==expected)
        except (ValueError,TypeError,ArithmeticError,KeyError) as e:result["U_reason"]=str(e)
    else:result["U_reason"]="NOT_APPLICABLE"
    return result


def score_one(man,raw,case,inp):
    panel=man["panel"]
    require(raw.get("input_id")==man["input_id"]==case["input_id"]==inp["input_id"],"input_id mismatch")
    require(raw.get("case_id")==man["case_id"]==case["case_id"],"case_id mismatch")
    require(case["task"]["root_id"]==man["root_id"] and case["task"]["structure"]==man["structure"],"private case root/stratum mismatch")
    if "seed" in raw:require(type(raw["seed"]) is int and raw["seed"]==man["seed"],"actual generation seed mismatch")
    txt=raw.get("full_text");pref=inp["assistant_prefill"]
    require(isinstance(txt,str) and isinstance(pref,str) and pref and txt.startswith(pref),"full_text does not begin with exact frozen assistant_prefill")
    if "assistant_prefill" in raw:require(raw["assistant_prefill"]==pref,"stored prefill disagrees with frozen runtime input")
    if "suffix_text" in raw:require(isinstance(raw["suffix_text"],str) and txt==pref+raw["suffix_text"],"raw suffix+prefill reconstruction failed")
    reason=raw.get("finish_reason")
    require(reason in {"balanced","eos","length"},"unrecognized finish reason/technical failure: restore original request, do not fabricate a zero")
    if "technical_validation_errors" in raw:require(not raw["technical_validation_errors"],"technical validation errors in raw response")
    if raw.get("generation_status") is not None:require(raw["generation_status"]=="COMPLETE","technical generation failed")
    t,audit=case["task"],case["audit"]
    app=audit.get("native_J_applicable")
    require(type(app) is bool,"native J applicability absent in private case")
    if panel in {"scale10","prefix10"} or (panel=="authority8" and man["view"]=="list"):require(not app,"native J not defined for this task")
    if panel=="retention8":require(app,"retention requires historical native J")
    if panel=="prefix10":require(len(pref)>1,"forced prefix must include frozen facts, not just an opening brace")
    z=fields(txt,t,audit,reason)
    if "U_applicable" in man:require(z["U_applicable"]==man["U_applicable"],"inconsistent U applicability")
    native=None
    if app:
        n=native_score(txt,t["world"],t["query"],finished=reason in {"balanced","eos"},finish_reason=reason)
        require(z["A"]==n["A"],"native/value A disagree on an applicable task")
        native=n["J"]
        require(type(native) is int and native in (0,1),"unexpected native J")
    sr,br=z["S"],z["B"]
    z.update(J_native=native,J_diag=None,A_right_S_wrong=int(z["A"]==1 and sr==0),A_right_B_wrong=int(z["A"]==1 and br==0),
        A_right_S_or_B_wrong=int(z["A"]==1 and (sr==0 or br==0)),A_right_S_unknown=int(z["A"]==1 and sr is None),
        A_right_B_unknown=int(z["A"]==1 and br is None))
    z["prefix_brace_eligible"]=None
    if panel=="prefix10":
        repair=(reason=="eos" and not z["json_valid"] and extract(txt+"}")[0] is not None)
        diag=fields(txt+"}",t,audit,reason) if repair else z
        z["prefix_brace_eligible"]=int(repair)
        for k in PREFIX_KEYS:z["brace_"+k]=diag[k]
    return {**man,"finish_reason":reason,**z}


def mean(x):return sum(x)/len(x)


def clarification_summary(rows):
    validate_dev(rows)
    result={}
    for p in ("pooled",*stat.PARENTS):
        xs=rows if p=="pooled" else [r for r in rows if r["parent"]==p]
        by=stat.grouped(xs,("structure","root_id","view"))
        result[p]={}
        for metric in ("A","S_hit"):
            rootviews=defaultdict(dict)
            for (st,rt,v),g in by.items():rootviews[(st,rt)][v]=mean([r[metric] for r in g])
            cell={v:stat.stratum_mean({k:d[v] for k,d in rootviews.items()}) for v in DEVIEWS}
            out={"cell_means":cell,"denominator_per_cell":len(xs)//4,"n_roots":len(rootviews)}
            for name in ("aligned_gain","conflict_gain","conflict_minus_aligned_gain"):
                dd={}
                for k,d in rootviews.items():
                    a=d["aligned_clear"]-d["aligned_plain"];c=d["conflict_clear"]-d["conflict_plain"]
                    dd[k]=a if name=="aligned_gain" else c if name=="conflict_gain" else c-a
                ci,scope=stat.root_ci(dd)
                out[name]={"estimate":stat.stratum_mean(dd),"ci95":ci,"ci_scope":scope,"B":stat.N_BOOT,"bootstrap_seed":stat.BOOT_SEED}
            result[p][metric]=out
    return {"status":"DESCRIPTIVE_ONLY_NOT_A_GATE","cells_and_interaction":result}


def diagnostics(rows):
    out={"by_endpoint":[],"prefix_bounds":[]}
    for k,g in stat.grouped(rows,("parent","arm","train_seed","step","panel","view")).items():
        out["by_endpoint"].append(dict(zip(("parent","arm","train_seed","step","panel","view"),k),
             n=len(g),counts={m:sum(r[m] for r in g) for m in (*MONITOR,"S_known","B_known","B_hit","json_valid","completed_generation")},
             follow_counts={m:{"known":sum(r[m] is not None for r in g),"matches":sum(r[m]==1 for r in g)} for m in ("list_choice_hit","prefix_choice_hit","report_choice_hit")}))
    for k,g in stat.grouped([r for r in rows if r["panel"]=="prefix10"],("parent","arm","train_seed","step")).items():
        cells=stat.grouped(g,("view",));ob=dict(zip(("parent","arm","train_seed","step"),k))
        for lab,prefix in (("frozen",""),("terminal_brace_sensitivity","brace_")):
            rates={}
            for (v,),rs in cells.items():
                h=sum(r[prefix+"S_hit"] for r in rs);u=sum(not r[prefix+"S_known"] for r in rs);n=len(rs)
                rates[v]={"N":n,"confirmed_S_correct":h,"S_unknown":u,"A_confirmed":sum(r[prefix+"A"] for r in rs),"S_finite_bounds":[h/n,(h+u)/n],
                          "prefix_induced_observable_hits":sum(r[prefix+"prefix_choice_hit"]==1 for r in rs),
                          "eligible_brace":sum(r["prefix_brace_eligible"] for r in rs)}
            dd={}
            for a,b in (("p_stable","p_revtie"),("p_placebo","p_revtie"),("p_stable","p_placebo")):
                lo,hi=rates[a]["S_finite_bounds"];lo2,hi2=rates[b]["S_finite_bounds"];dd[a+"-"+b]=[lo-hi2,hi-lo2]
            ob[lab]={"levels":rates,"S_contrast_finite_identification_bounds_not_CIs":dd}
        out["prefix_bounds"].append(ob)
    return out


def evaluate(plan_rows,cases,inputs,raw_rows):
    pl=list(plan_rows);require(pl,"empty manifest");dev=pl[0]["panel"]=="clarification";keys=DEVKEY if dev else stat.KEY
    p=index_requests(pl,keys);r=index_requests(raw_rows,keys)
    if dev:validate_dev(pl)
    else:
        stat.index_rows(pl);stat.verify_design(pl,"retention8")
    require(p.keys()==r.keys(),f"raw coverage mismatch missing={len(p.keys()-r.keys())}, extra={len(r.keys()-p.keys())}")
    obs=[]
    for k,m in p.items():
        require(m.get("case_id") in cases and m.get("input_id") in inputs,"manifest case/input missing")
        obs.append(score_one(m,r[k],cases[m["case_id"]],inputs[m["input_id"]]))
    return obs,(clarification_summary(obs) if dev else diagnostics(obs))


def save_atomic(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w",encoding="utf8",dir=path.parent,delete=False) as tmp:
        tmp.write(text);name=tmp.name
    os.replace(name,path)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ("manifest","inputs","cases","raw","out"):ap.add_argument("--"+k,required=True)
    ap.add_argument("--summary",help="defaults to <out stem>_summary.json")
    a=ap.parse_args(argv)
    try:
        cases=ids(stat.load_rows(a.cases),"case_id",("input_id","task","audit"))
        inp=ids(stat.load_rows(a.inputs),"input_id",("image_file","prompt","assistant_prefill"))
        obs,summary=evaluate(stat.load_rows(a.manifest),cases,inp,stat.load_rows(a.raw))
        fmt=lambda x:json.dumps(x,ensure_ascii=False,allow_nan=False,separators=(",",":"))
        dest=Path(a.out);summarypath=Path(a.summary) if a.summary else dest.with_name(dest.stem+"_summary.json")
        require(dest.resolve()!=summarypath.resolve(),"output paths collide")
        require(not {dest.resolve(),summarypath.resolve()}.intersection(Path(x).resolve() for x in (a.raw,a.manifest,a.cases,a.inputs)),"refuse to overwrite input")
        save_atomic(dest,"".join(fmt(x)+"\n" for x in obs));save_atomic(summarypath,fmt(summary)+"\n")
    except (ValueError,KeyError,OSError,TypeError) as e:ap.exit(2,"SCORING REFUSED: "+str(e)+"\n")

if __name__=="__main__":main()
