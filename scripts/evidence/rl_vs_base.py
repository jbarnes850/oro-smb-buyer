"""MaxRL final minus base v6: difference of mean-of-world-means with SE (within-world variances), TEST and VAL."""
import json
import math
import statistics as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, WORLDS
rec = json.loads((OUT / "records_metrics.json").read_text())
tr = {r["dir"]: r for r in json.loads((OUT / "transcript_metrics.json").read_text())}
rows = []
for r in rec:
    m = dict(r); m.update({k: v for k, v in tr.get(r["dir"], {}).items() if k not in m}); rows.append(m)
def mw(arm, key):
    means, var = [], 0
    for w in WORLDS:
        v = [r[key] for r in rows if r["arm"] == arm and r["world"] == w and r["status"] == "valid" and isinstance(r.get(key), (int, float))]
        means.append(st.mean(v)); var += (st.variance(v) / len(v)) if len(v) > 1 else 0
    return st.mean(means), math.sqrt(var) / 3
keys = ["return_usd", "revenue_usd", "gross_margin", "received_t", "local_t", "local_usd", "walkin_fill", "acct_short_kg",
        "cover_rules_off", "cover_rules_per_sku", "days_of_cover", "spoilage_usd", "top_supplier_share", "tool_error_rate",
        "malformed_errors", "max_consecutive_errors", "output_tokens_per_day", "calls_per_day", "accepted_pos_per_day",
        "close_days_of_cover", "late_order_ratio_85_89", "dock_refusals"]
out = {}
for split, a, b in (("test", "maxrl_final", "base_v6"), ("val", "maxrl_final_val", "base_v6_val")):
    print(f"== {split}: {a} - {b} (valid quarters; VAL base has 11)")
    out[split] = {}
    for k in keys:
        (x, sx), (y, sy) = mw(a, k), mw(b, k)
        d, se = x - y, math.sqrt(sx**2 + sy**2)
        out[split][k] = {"maxrl": x, "base": y, "diff": d, "se": se}
        print(f"  {k:24s} {x:12.3f} {y:12.3f}  diff {d:+10.3f} ± {se:8.3f}  z={d/se if se else float('nan'):+.1f}")
(OUT / "rl_vs_base.json").write_text(json.dumps(out, indent=1))
for a in ("maxrl_final", "base_v6", "maxrl_final_val", "base_v6_val"):
    print(a, "per-quarter cover_rules_off:", [r.get("cover_rules_off") for r in rows if r["arm"] == a], "max consec:", [r.get("max_consecutive_errors") for r in rows if r["arm"] == a])
