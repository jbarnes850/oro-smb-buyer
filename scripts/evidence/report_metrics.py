"""Two SMB Bench report behaviours, recomputed with one code path for our policies and the frontier replays.

1. Reaction to a bad week (report Figure 8). For every product-week out of stock at least 25% of store
   hours, the change in days of stock held the following week. Days of stock = mean on-hand grams over
   the week / the product's mean daily sales over the quarter. Weeks are days 7w..7w+6, w = 0..10, so the
   following week ends by day 83 and the end-of-quarter wind-down is excluded. Store hours = 16 per day.
2. Account products with standing shortfall cover, day by day (report Figure 10). Account products are
   the (account, product) pairs any account ordered in the quarter. A product-specific rule overrides
   the account-wide ('*') rule.

    python3 scripts/evidence/report_metrics.py   -> $SMB_EVIDENCE_OUT/report_metrics.json
"""
import glob
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT as EVIDENCE, RUNS, read_csv  # noqa: E402

STORE_HOURS = 16
OUT = EVIDENCE / "report_metrics.json"


def i(x):
    return int(float(x)) if x not in (None, "") else 0


def bad_week_reactions(pr):
    inv = defaultdict(dict)
    for x in read_csv(pr / "inventory_daily.csv"):
        if i(x["day"]) >= 0:
            inv[x["sku_id"]][i(x["day"])] = i(x["on_hand_grams"])
    sold, hours = defaultdict(int), {}
    for x in read_csv(pr / "sales_daily.csv"):
        d = i(x["day"])
        if d < 0 or d > 89:
            continue
        sold[x["sku_id"]] += i(x["grams_sold"])
        if x["channel"] == "pos":
            hours[(x["sku_id"], d)] = i(x["hours_out_of_stock"])
    out = []
    for sku, days in inv.items():
        daily = sold[sku] / 90
        if daily <= 0:
            continue
        dos = [st.mean(days.get(d, 0) for d in range(7 * w, 7 * w + 7)) / daily for w in range(12)]
        for w in range(11):
            oos = sum(hours.get((sku, d), 0) for d in range(7 * w, 7 * w + 7)) / (STORE_HOURS * 7)
            if oos >= 0.25:
                out.append(dos[w + 1] - dos[w])
    return out


def cover_share_by_day(pr):
    accounts = {x["customer_id"] for x in read_csv(pr / "customers.csv") if x["channel"] == "account"}
    pairs = {(x["customer_id"], x["sku_id"]) for x in read_csv(pr / "customer_order_lines.csv")
             if x["customer_id"] in accounts and i(x["due_day"]) >= 0}
    rules = sorted(read_csv(pr / "shortfall_cover_rules.csv"), key=lambda r: (i(r["day"]), i(r["set_t"])))
    star, spec, k, share = {}, {}, 0, []
    for day in range(90):
        while k < len(rules) and i(rules[k]["day"]) <= day:
            r = rules[k]
            on = r["cover"] == "True"
            if r["sku_id"] == "*":
                star[r["customer_id"]] = on
                for key in [p for p in spec if p[0] == r["customer_id"]]:
                    del spec[key]
            else:
                spec[(r["customer_id"], r["sku_id"])] = on
            k += 1
        covered = sum(1 for c, s in pairs if spec.get((c, s), star.get(c, False)))
        share.append(covered / len(pairs) if pairs else 0.0)
    return share


def policies(split):
    ours = {"test": {"Qwen3.8-27B + RL": "test/maxrl-001-final-v6b/s505*", "Qwen3.8-27B, no RL": "test/q38-on-v6b/s505*"},
            "val": {"Qwen3.8-27B + RL": "val/ref-maxrl-001-final-v6b/s504*", "Qwen3.8-27B, no RL": "val/ref-q38-on-v6b/s504*"}}[split]
    out = defaultdict(list)
    for name, pat in ours.items():
        for d in sorted(glob.glob(str(RUNS / pat))):
            if (Path(d) / "quarter-result.json").exists():
                out[name].append(Path(d))
    fdir = RUNS / ("frontier-test-reference" if split == "test" else "frontier-val-reference")
    model = {r["trial_id"]: r["model"] for r in json.loads((fdir / "hurdles.json").read_text())}
    for d in sorted(fdir.iterdir()):
        if d.is_dir() and d.name in model and (d / "engine/public_records").exists():
            out[model[d.name]].append(d)
    return out


def main():
    result = {}
    for split in ("test", "val"):
        res = {}
        for name, dirs in policies(split).items():
            reacts, curves = [], []
            for d in dirs:
                pr = d / "engine/public_records"
                reacts += bad_week_reactions(pr)
                curves.append(cover_share_by_day(pr))
            res[name] = {
                "quarters": len(dirs),
                "bad_weeks": len(reacts),
                "days_of_stock_added_after_bad_week": st.mean(reacts) if reacts else None,
                "cover_share_day0": st.mean(c[0] for c in curves),
                "cover_share_mean": st.mean(st.mean(c) for c in curves),
                "cover_share_by_day": [st.mean(c[t] for c in curves) for t in range(90)],
                "quarters_all_covered_day0": sum(c[0] >= 0.999 for c in curves),
            }
        result[split] = res
        for n, r in sorted(res.items(), key=lambda kv: -(kv[1]["days_of_stock_added_after_bad_week"] or -9)):
            print(f"{split} {n:28s} q={r['quarters']:2d} badweeks={r['bad_weeks']:4d} "
                  f"react={r['days_of_stock_added_after_bad_week']:+.3f} cover0={r['cover_share_day0']:.2f} "
                  f"covermean={r['cover_share_mean']:.2f} all0={r['quarters_all_covered_day0']}")
    OUT.write_text(json.dumps(result, indent=1))
    print(OUT)


if __name__ == "__main__":
    main()
