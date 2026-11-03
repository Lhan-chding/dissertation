"""Offline artificial fixtures only. Never reuse these as real outputs."""
import importlib.util
import unittest
from pathlib import Path

p=Path(__file__).resolve().parents[1]/"analyze.py"
s=importlib.util.spec_from_file_location("order_train_analysis",p)
stats=importlib.util.module_from_spec(s)
s.loader.exec_module(stats)


def row(panel="transfer8",st="2_choose_1",rt="r0",view="conflict",draw=0,arm="D",seed=11031,step=24,parent="b0",A=0):
    app=stats.order_applies(panel,view)
    return dict(panel=panel,root_id=rt,structure=st,parent=parent,train_seed=seed,arm=arm,
                step=step,view=view,draw=draw,A=A,S_known=1,S_hit=A,U=1 if app else None,U_applicable=app,
                J_native=A if panel=="retention8" else None)


def complete_fixture():
    support=[]
    for pan,(n,vs,nd) in stats.LAYOUT.items():
        strata=stats.FAMILIES if pan=="retention8" else stats.STRUCTURES
        for st in strata:
            for i in range(n//len(strata)):
                for v in vs:
                    for d in range(nd):support.append((pan,st,pan+st+str(i),v,d))
    paths=[(b,"PARENT",None,0) for b in stats.PARENTS]
    paths += [(b,a,k,t) for b in stats.PARENTS for a in stats.ARMS for k in stats.SEEDS for t in stats.STEPS]
    out=[]
    for panel,st,rt,v,d in support:
        for b,a,k,t in paths:
            on=int(panel=="transfer8" and v=="conflict" and a=="D" and t==24 and d==0)
            out.append(row(panel,st,rt,v,d,a,k,t,b,on))
    return out


class TestOrderAnalysis(unittest.TestCase):
    def test_complete_paired_result_and_manifest_missing(self):
        records=complete_fixture()
        plan=[{f:r[f] for f in (*stats.KEY,"U_applicable")} for r in records]
        result=stats.run(plan,records)
        p=result["primary"]
        self.assertEqual(p["n_roots"],96)
        self.assertEqual(p["n_assigned_per_arm"],768)
        self.assertEqual(set(p["paired_blocks"].values()),{.5})
        self.assertEqual(p["ci95"],[.5,.5])
        self.assertIsNone(p["training_seed_CI"])
        self.assertEqual(result["inventory"]["retention8"]["slots_per_endpoint"],96)
        self.assertEqual(result["retention"]["comparisons"][0]["ci95"],[0.,0.])
        with self.assertRaisesRegex(ValueError,"manifest/scores mismatch"):
            stats.run(plan,records[:-1])
        with self.assertRaisesRegex(ValueError,"duplicate evaluation key"):
            stats.run(plan,records+[records[0]])
        with self.assertRaisesRegex(ValueError,"endpoint"):
            defect=[r for r in records if not (r["panel"]=="transfer8" and r["arm"]=="D" and r["train_seed"]==11032)]
            stats.verify_design(defect,"retention8")

    def test_scoring_and_seeds_fail_closed(self):
        for patch in ({"A":2},{"A":True},{"S_known":0,"S_hit":1},{"U":"unknown"},{"U_applicable":0},{"train_seed":None},{"train_seed":""},{"train_seed":11033}):
            with self.subTest(patch=patch),self.assertRaises(ValueError):
                stats.index_rows([dict(row(),**patch)],scores=True)
        with self.assertRaises(ValueError):
            stats.index_rows([dict(row(panel="scale10"),J_native=1)],True)
        with self.assertRaises(ValueError):
            stats.index_rows([dict(row(panel="swap8"),U=1)],True)
        with self.assertRaises(ValueError):
            stats.index_rows([dict(row(panel="retention8"),J_native=None)],True)

    def test_pairing_not_unmatched_draws_and_no_seed_replication(self):
        r=[row(arm=a,parent=b,seed=s,draw=d) for a in ("D","G") for b in stats.PARENTS for s in stats.SEEDS for d in (0,1)]
        r[0]["draw"]=9
        with self.assertRaisesRegex(ValueError,"row-level"):
            stats.compare(r,"D","G")
        with self.assertRaisesRegex(ValueError,"four paired"):
            stats.compare([x for x in r if x["train_seed"]==11032],"D","G")

    def test_ci_uses_roots_equal_strata_not_draw_pseudoreplication(self):
        values={("s","r1"):1.,("s","r2"):-1.,("t","r3"):.5,("t","r4"):.5,("t","r5"):.5}
        self.assertEqual(stats.stratum_mean(values),.25)
        ci,state=stats.root_ci(values)
        self.assertEqual(ci,[-.25,.75])
        self.assertIn("conditional",state)
        self.assertEqual(stats.root_ci({("s","one"):1.})[0],None)
        with self.assertRaises(ValueError):stats.root_ci({})
        with self.assertRaises(ValueError):stats.root_ci({("s","bad"):float("nan")})

    def test_retention_without_equivalence(self):
        self.assertEqual(stats.retention_status([-.0199,.12]),"conditional_noninferiority_supported")
        self.assertTrue(stats.retention_status([-.02,.12]).startswith("inconclusive"))
        self.assertTrue(stats.retention_status([-.04,-.021]).startswith("observed_harm"))
        self.assertTrue(stats.retention_status(None).startswith("inconclusive"))
        self.assertTrue(stats.equivalence_status([-.005,.005]).startswith("not_preregistered"))
        self.assertEqual(stats.equivalence_status([-.019,.019],preregistered=True),"conditional_equivalence_supported")
        self.assertTrue(stats.equivalence_status([-.03,.03],preregistered=True).startswith("inconclusive"))

    def test_joint_unknown_and_not_applicable(self):
        xs=[dict(row(draw=i),A=a,U=u) for i,(a,u) in enumerate([(1,1),(1,None),(0,None),(1,0)])]
        r=stats.summarize(xs)
        z=r["equal_weight_rates"]
        self.assertEqual(z["A_and_U_confirmed"],.25)
        self.assertEqual(z["A_and_U_finite_identification_bounds"],[.25,.50])
        self.assertEqual(z["U_known"],.5)
        self.assertEqual(z["A_and_U_given_U_known"],.5)
        r=stats.summarize([row(panel="swap8")])
        self.assertIsNone(r["equal_weight_rates"]["A_and_U_confirmed"])
        self.assertIsNone(r["equal_weight_rates"]["A_and_U_finite_identification_bounds"])

    def test_swap_uses_same_draw_only(self):
        xs=[]
        for v in stats.SWAP:
            for d in (0,1):
                xs.append(row(panel="swap8",view=v,draw=d,A=int((v in ("p0l0","p1l0"))==(d==0))))
        z=stats.swap_summary(xs)[0]
        self.assertEqual(z["n_complete_same_draw_quadruples"],2)
        self.assertEqual(z["equal_weight_rates"]["all_four"],0.)
        self.assertEqual(z["equal_weight_rates"]["p0_both_lists"],0.)
        self.assertEqual(z["equal_weight_rates"]["l0_both_prints"],.5)
        with self.assertRaises(ValueError):stats.swap_summary(xs[:-1])


if __name__=="__main__": unittest.main()
