"""CPU-only fixtures for scoring identity, observability and inference. No model outputs are generated."""
import importlib.util
import json
import unittest
from pathlib import Path

root=Path(__file__).resolve().parents[1]
sp=importlib.util.spec_from_file_location("score_eval",root/"score_eval.py")
ev=importlib.util.module_from_spec(sp);sp.loader.exec_module(ev)
from data_design import make_world,build_context


def fixture(n=8):
    t=make_world("SCORETEST",0,n);ctx=build_context(t,"TRANSFER","CONFLICT",13)
    t=ctx["task"];w=t["world"]; order=ctx["audit"]["expected_report_order"]
    evidence=[{"series":s,"category":k,"value":vs[w["categories"].index(k)]} for s,vs in w["series"].items() for k in order]
    gold=t["gold"]
    obj={"evidence":evidence,"selected":gold["selected"],"bindings":[{"category":k,"operand":v} for k,v in zip(gold["selected"],gold["operands"])],
         "aggregate":t["query"]["aggregate"],"answer":gold["answer"]}
    return t,ctx["audit"],obj


def packed(t,a,obj,panel="transfer8",view="conflict",pref="{"):
    inp={"image_file":"fixture","prompt":"fixture","assistant_prefill":pref}
    inp["input_id"]=ev.digest(inp)
    data={"input_id":inp["input_id"],"task":t,"audit":a};data["case_id"]=ev.digest(data)
    m=dict(panel=panel,view=view,parent="b0",arm="D",train_seed=11031,step=24,root_id=t["root_id"],structure=t["structure"],draw=0,seed=123,
           case_id=data["case_id"],input_id=inp["input_id"],U_applicable=a.get("expected_report_order") is not None)
    raw={**m,"full_text":json.dumps(obj,separators=(",",":")),"finish_reason":"balanced"}
    return m,raw,data,inp


class TestScoreEval(unittest.TestCase):
    def test_native_gold_and_instrument(self):
        t,a,o=fixture()
        m,r,c,i=packed(t,a,o)
        z=ev.score_one(m,r,c,i);self.assertEqual((z["A"],z["S"],z["B"],z["J_native"],z["U"]),(1,1,1,1,1))
        # Correct answer, explicit wrong selection and bindings; no inference from its correct answer.
        o["selected"]=[];o["bindings"]=[];o["evidence"]=o["evidence"][::-1];r["full_text"]=json.dumps(o)
        z=ev.score_one(m,r,c,i)
        self.assertEqual((z["A"],z["S"],z["B"],z["A_right_S_or_B_wrong"],z["U"]),(1,0,0,1,0))
        o["evidence"].pop();o["evidence"]=[x for x in o["evidence"] if x["series"]!="Alpha"]
        r["full_text"]=json.dumps(o);self.assertIsNone(ev.score_one(m,r,c,i)["U"])

    def test_asterisk_and_no_work_recovery(self):
        t,a,o=fixture()
        o["work"]="selected="+str(o["selected"])+"; answer="+str(o["answer"])
        o["answer"]="**"+str(o["answer"])+"**";del o["selected"]
        z=ev.fields(json.dumps(o),t,a,"eos")
        self.assertEqual(z["A"],0);self.assertEqual(z["A_parseable"],0);self.assertIsNone(z["S"]);self.assertEqual(z["S_hit"],0)
        o["answer"]=t["gold"]["answer"]
        self.assertIsNone(ev.fields(json.dumps(o),t,a,"length")["B"])

    def test_prefix_verifies_full_prefill_and_strict_eos_brace(self):
        t,a,o=fixture(10)
        a={"expected_report_order":None,"native_J_applicable":False,"prefix_induced_selected":t["gold"]["selected"]}
        text=json.dumps(o,separators=(",",":"));pref=text[:text.index(',"selected"')+1]
        m,r,c,i=packed(t,a,o,"prefix10","p_stable",pref)
        good=ev.score_one(m,r,c,i);self.assertEqual(good["prefix_choice_hit"],1);self.assertEqual(good["J_native"],None)
        r["finish_reason"]="eos";r["full_text"]=text[:-1]
        z=ev.score_one(m,r,c,i)
        self.assertEqual((z["A"],z["S_known"],z["prefix_brace_eligible"],z["brace_A"],z["brace_S_hit"]),(0,0,1,1,1))
        r["finish_reason"]="length";z=ev.score_one(m,r,c,i);self.assertEqual(z["prefix_brace_eligible"],0)
        r["finish_reason"]="eos";r["full_text"]=text[:-1]+',"selected":[]'
        z=ev.score_one(m,r,c,i);self.assertEqual(z["prefix_brace_eligible"],0)
        r["full_text"]=text.replace('"value":','"valRejected":',1)
        with self.assertRaisesRegex(ValueError,"prefill"):ev.score_one(m,r,c,i)
        r["full_text"]=text;r["suffix_text"]="rewrite"
        with self.assertRaisesRegex(ValueError,"reconstruction"):ev.score_one(m,r,c,i)

    def test_reject_wrong_identity_and_list_authority_native(self):
        t,a,o=fixture();m,r,c,i=packed(t,a,o)
        r["case_id"]="bad"
        with self.assertRaisesRegex(ValueError,"case_id"):ev.score_one(m,r,c,i)
        m,r,c,i=packed(t,a,o,"authority8","list")
        with self.assertRaisesRegex(ValueError,"native J"):ev.score_one(m,r,c,i)
        c["audit"]={**a,"native_J_applicable":False,"expected_report_order":None};m["U_applicable"]=False
        self.assertIsNone(ev.score_one(m,r,c,i)["J_native"])

    def test_clarification_four_cells_matched(self):
        rows=[]
        for s in ev.stat.STRUCTURES:
            for n in range(6):
                for v in ev.DEVIEWS:
                    for p in ev.stat.PARENTS:
                        for d in (0,1):
                            y=int(v=="conflict_clear" and d==0)
                            rows.append(dict(panel="clarification",root_id=f"{s}-{n}",structure=s,view=v,parent=p,draw=d,A=y,S_hit=y))
        z=ev.clarification_summary(rows)
        self.assertEqual(z["cells_and_interaction"]["pooled"]["A"]["conflict_minus_aligned_gain"]["ci95"],[.5,.5])
        self.assertEqual(z["cells_and_interaction"]["b0"]["S_hit"]["denominator_per_cell"],48)
        with self.assertRaisesRegex(ValueError,"384"):ev.clarification_summary(rows[:-1])

if __name__=="__main__":unittest.main()
