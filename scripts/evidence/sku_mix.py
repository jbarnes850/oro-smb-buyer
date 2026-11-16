"""Per-SKU supplier receipts, local buys, expiries and account dues: final checkpoint vs frontier (TEST)."""
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import quarters, read_csv
def i(x): return int(float(x)) if x not in (None, "") else 0
arms = sys.argv[1].split(",") if len(sys.argv) > 1 else ["maxrl_final", "gpt-6-astra", "claude-fable-5-1"]
agg = {a: defaultdict(lambda: defaultdict(float)) for a in arms}
n = defaultdict(int)
for arm, split, src, d in quarters():
    if arm not in arms or split != "test": continue
    pr = Path(d) / "engine/public_records"
    if not (pr / "receipts.csv").exists() and not (pr / "receipts.csv.gz").exists(): continue
    n[arm] += 1
    pos = read_csv(pr / "purchase_orders.csv"); actor = {p["po_id"]: p["actor"] for p in pos}
    A = agg[arm]
    for r in read_csv(pr / "receipts.csv"):
        if actor.get(r["po_id"]) == "agent": A[r["sku_id"]]["recv"] += i(r["grams_accepted"])
    for l in read_csv(pr / "local_purchases.csv"):
        if i(l["day"]) >= 0: A[l["sku_id"]]["local"] += i(l["grams"])
    for x in read_csv(pr / "disposals.csv"):
        if i(x["day"]) >= 0 and x["reason"] in ("expired", "age_cull"): A[x["sku_id"]]["waste"] += i(x["grams"])
    for c in read_csv(pr / "customer_order_lines.csv"):
        if i(c["due_day"]) >= 0: A[c["sku_id"]]["acct_due"] += i(c["ordered_grams"])
    for s in read_csv(pr / "sales_daily.csv"):
        if i(s["day"]) >= 0 and s["channel"] == "pos":
            A[s["sku_id"]]["pos_sold"] += i(s["grams_sold"]); A[s["sku_id"]]["oos_h"] += i(s["hours_out_of_stock"])
skus = sorted(agg[arms[0]])
print("per-quarter means, kg (oos_h = walk-in stockout hours)")
print(f"{'sku':16s}" + "".join(f"| {a[:12]:>12s} recv  local  waste  oos_h " for a in arms))
tot = defaultdict(lambda: defaultdict(float))
for s in sorted(skus, key=lambda s: -agg[arms[0]][s]["local"]):
    line = f"{s:16s}"
    for a in arms:
        v = agg[a][s]; k = n[a]
        line += f"| {'':12s} {v['recv']/k/1000:5.0f} {v['local']/k/1000:6.0f} {v['waste']/k/1000:6.0f} {v['oos_h']/k:6.0f} "
    print(line)
