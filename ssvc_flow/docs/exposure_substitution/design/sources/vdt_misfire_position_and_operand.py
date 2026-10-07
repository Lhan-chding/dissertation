"""Two output-side checks on cross tasks (no model calls).

Usage:
    python vdt_misfire_position_and_operand.py /path/to/unzipped/SSVC_VERIFIED_DISCOVERY_TRANSFER_RESULTS_20261007

(1) Hub-corrupted tasks (hub at position 3 or 4): among outputs with the hub fixed to truth but >=1 leaf edited,
    the share of edited-leaf positions, compared with the share of same-structure leaf-corrupted training targets
    (donors) by position, taken from project_docs/final_analysis/discovery_fit_exposures.csv (focus exposures).
(2) Leaf-corrupted tasks with the hub AFTER the leaf: among right-location single edits with a wrong value,
    classify the wrong value as: wrong partner (b_row - observed other leaf), off-by-<=2, equals hub, etc.
All positions printed 1-based.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

root = sys.argv[1]
E = {json.loads(l)["task_id"]: json.loads(l) for l in open(f"{root}/run_v2/cohort/E_test/tasks_public.jsonl")}
EA = {json.loads(l)["task_id"]: json.loads(l) for l in open(f"{root}/run_v2/cohort/E_test/audit_only.jsonl")}
TA = {json.loads(l)["task_id"]: json.loads(l) for l in open(f"{root}/run_v2/cohort/T_train/audit_only.jsonl")}

rows = []
for d in glob.glob(f"{root}/run_v2/sealed/eval.*.E_test.*"):
    _, parent, rep, arm, _, _ = os.path.basename(d).split(".")
    for f in glob.glob(d + "/chunks/*.json"):
        for r in json.load(open(f))["rows"]:
            if E[r["task_id"]]["family"] == "cross_series":
                rows.append((arm, parent, int(rep), r["task_id"], r["raw_text"]))
df = pd.DataFrame(rows, columns=["arm", "parent", "rep", "task_id", "raw"])
df["hub"] = df.task_id.map(lambda k: EA[k]["center"])
df["j"] = df.task_id.map(lambda k: EA[k]["corrupted_index"])


def parse(s):
    try:
        v = json.loads(s.strip())
        if isinstance(v, list) and len(v) == 4 and all(isinstance(x, int) and not isinstance(x, bool) for x in v):
            return v
    except Exception:
        pass
    return None


def leaf_row(H, k):
    return [i for i, rw in enumerate(H) if rw[k] != 0][0]


ARMS = ["SELF_MIX", "SELF_SINGLE", "GOLD_MATCH_MIX", "GOLD_ALL"]
pd.set_option("display.width", 250)

# (1) donor share vs misfire share
ex = pd.read_csv(f"{root}/project_docs/final_analysis/discovery_fit_exposures.csv")
ex = ex[(ex.role == "focus") & (ex.family == "cross_series")].copy()
ex["center"] = ex.task_id.map(lambda k: TA[k]["center"])
ex["j"] = ex.task_id.map(lambda k: TA[k]["corrupted_index"])
donor = ex[ex.center != ex.j].groupby(["job", "center", "j"]).exposures.sum()

out = []
hc = df[(df.hub == df.j) & (df.hub.isin([2, 3])) & (df.arm.isin(ARMS))]
for (arm, parent, rep, hub), g in hc.groupby(["arm", "parent", "rep", "hub"]):
    job = f"sft.{parent}.{rep}.{arm}"
    leaves = [k for k in range(4) if k != hub]
    dshare = {k: donor.get((job, hub, k), 0) for k in leaves}
    dtot = sum(dshare.values())
    mis = {k: 0 for k in leaves}
    for _, r in g.iterrows():
        y = parse(r.raw)
        tr, ob = EA[r.task_id]["true_world"], E[r.task_id]["observed"]
        if y is None or y == tr or y[hub] != tr[hub]:
            continue
        for k in leaves:
            if y[k] != ob[k]:
                mis[k] += 1
    mtot = sum(mis.values())
    for k in leaves:
        out.append((arm, hub + 1, k + 1, dshare[k] / dtot if dtot else np.nan, mis[k] / mtot if mtot else np.nan))
o = pd.DataFrame(out, columns=["arm", "hub_pos", "leaf_pos", "donor_share", "misfire_share"])
print("== (1) same-structure donor share vs extra-edit (misfire) share, mean over units")
print(o.groupby(["hub_pos", "leaf_pos", "arm"])[["donor_share", "misfire_share"]].mean().unstack("arm").round(2).to_string())
v = o.dropna()
print("corr over rows:", round(np.corrcoef(v.donor_share, v.misfire_share)[0, 1], 3), "n", len(v))

# (2) wrong-operand classification
lc = df[(df.hub != df.j) & (df.hub > df.j) & (df.arm.isin(ARMS))]
kinds = []
for _, r in lc.iterrows():
    t, a = E[r.task_id], EA[r.task_id]
    ob, tr, h, j = t["observed"], a["true_world"], a["center"], a["corrupted_index"]
    y = parse(r.raw)
    if y is None or [k for k in range(4) if y[k] != ob[k]] != [j]:
        continue
    if y[j] == tr[j]:
        kinds.append((r.arm, "correct"))
        continue
    H, b = np.array(t["H_original"]), np.array(t["b_original"])
    rj, v = leaf_row(H, j), y[j]
    if any(k not in (h, j) and v == b[rj] - ob[k] for k in range(4)):
        kinds.append((r.arm, "wrong_partner"))
    elif any(i != rj and v == b[i] - ob[h] for i in range(len(b))):
        kinds.append((r.arm, "other_relation_constant"))
    elif abs(v - tr[j]) <= 2:
        kinds.append((r.arm, "off_by_<=2"))
    elif v == ob[h]:
        kinds.append((r.arm, "equals_hub"))
    else:
        kinds.append((r.arm, "other"))
k = pd.DataFrame(kinds, columns=["arm", "kind"])
print("\n== (2) leaf corrupted, hub after: right-location single edits by value kind (%)")
print((k.groupby("arm").kind.value_counts(normalize=True).unstack(fill_value=0) * 100).round(1).to_string())
print(k.groupby("arm").size().to_string())
