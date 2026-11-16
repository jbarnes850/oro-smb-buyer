"""Return components by world: MaxRL final vs base v6 vs Astra vs top-4 (TEST)."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as A
comp = [("return", "return_usd"), ("revenue", "revenue_usd"), ("COGS", "cogs_usd"), ("local spend", "local_usd"),
        ("local premium vs quote", "local_premium_vs_quote_usd"), ("covered-line margin", "covered_line_margin_usd"),
        ("dispatch fees", "fees_usd.dispatch"), ("spoilage", "spoilage_usd"), ("valuation adj (incl -reserve)", "valuation_adj_usd"),
        ("supplier received t", "received_t"), ("walk-in sold t", "pos_sold_t"), ("account filled t", "acct_filled_t"),
        ("walk-in fill", "walkin_fill"), ("acct short kg", "acct_short_kg"), ("gross margin", "gross_margin")]
out = {}
for w in A.WORLDS:
    out[w] = {}
    print(f"\n== {w}")
    print(f"{'component':32s} {'MaxRL':>10s} {'base':>10s} {'Astra':>10s} {'top4':>10s} {'MaxRL-Astra':>12s}")
    for label, key in comp:
        a = A.group_val(["maxrl_final"], key, w); b = A.group_val(["base_v6"], key, w)
        c = A.group_val(["gpt-6-astra"], key, w); d = A.group_val(A.TOP4, key, w)
        out[w][label] = {"maxrl": a, "base": b, "astra": c, "top4": d}
        print(f"{label:32s} {a:10.2f} {b:10.2f} {c:10.2f} {d:10.2f} {a-c:12.2f}")
(A.OUT / "components_by_world.json").write_text(json.dumps(out, indent=1))
