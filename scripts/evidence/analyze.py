"""Aggregate records_metrics.json + transcript_metrics.json into tables.md and findings.json.

Aggregation: for every metric, mean over the three worlds of the per-world mean over valid
quarters (the campaign's primary convention), unless a world is named. Frontier groups average the
per-model values with equal weight. "top4" = the four best frontier models on TEST
(gpt-6-astra, claude-fable-5-1, gpt-6.1-sol, claude-opus-5-5).

    python3 -I analyze.py
"""
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, WORLDS  # noqa: E402

REC = json.loads((OUT / "records_metrics.json").read_text())
TR = {r["dir"]: r for r in json.loads((OUT / "transcript_metrics.json").read_text())}
ROWS = []
for r in REC:
    t = TR.get(r["dir"], {})
    m = dict(r)
    for k, v in t.items():
        if k not in m:
            m[k] = v
    ROWS.append(m)

FRONTIER = sorted({r["arm"] for r in ROWS if r["source"] == "frontier"})
TOP4 = ["gpt-6-astra", "claude-fable-5-1", "gpt-6.1-sol", "claude-opus-5-5"]


def get(r, key):
    """key may be 'a.b' for nested dicts or 'a[i]' for list index."""
    if "[" in key:
        base, idx = key[:-1].split("[")
        v = r.get(base)
        return v[int(idx)] if isinstance(v, list) and len(v) > int(idx) else None
    if "." in key:
        a, b = key.split(".", 1)
        v = r.get(a)
        return v.get(b, 0.0) if isinstance(v, dict) else None
    return r.get(key)


def valid(arm):
    return [r for r in ROWS if r["arm"] == arm and r["status"] == "valid"]


def arm_val(arm, key, world=None, agg="mean"):
    rs = valid(arm)
    if world:
        v = [get(r, key) for r in rs if r["world"] == world]
        v = [x for x in v if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))]
        return st.mean(v) if v else None
    means = [arm_val(arm, key, w) for w in WORLDS]
    means = [x for x in means if x is not None]
    return st.mean(means) if len(means) == 3 else (st.mean(means) if means else None)


def group_val(arms, key, world=None):
    v = [arm_val(a, key, world) for a in arms]
    v = [x for x in v if x is not None]
    return st.mean(v) if v else None


COLS_TEST = [("MaxRL final", ["maxrl_final"]), ("base v6", ["base_v6"]), ("Astra", ["gpt-6-astra"]),
             ("top-4 frontier", TOP4), ("all 9 frontier", None)]
COLS_VAL = [("MaxRL final (VAL)", ["maxrl_final_val"]), ("base v6 (VAL)", ["base_v6_val"]),
            ("Qwen3.6 base v1 (VAL, n=3)", ["q36_base_v1_val"])]


def val(cols, key, world=None):
    out = []
    for name, arms in cols:
        arms = arms if arms is not None else FRONTIER
        out.append(group_val(arms, key, world))
    return out


def fmt(x, nd=2):
    if x is None:
        return "n/a"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 100:
        return f"{x:.0f}"
    return f"{x:.{nd}f}"


def table(title, spec, cols=COLS_TEST, world=None):
    """spec: list of (label, key, nd)"""
    lines = [f"**{title}**" + (f" (world: {world})" if world else ""), "",
             "| metric | " + " | ".join(c for c, _ in cols) + " |",
             "|---|" + "---|" * len(cols)]
    data = {}
    for label, key, nd in spec:
        vs = val(cols, key, world)
        data[label] = dict(zip([c for c, _ in cols], vs))
        lines.append(f"| {label} | " + " | ".join(fmt(v, nd) for v in vs) + " |")
    return "\n".join(lines) + "\n", data


def per_model(key, world=None):
    return {a: arm_val(a, key, world) for a in ["maxrl_final", "base_v6"] + FRONTIER}


def main():
    md = []
    J = {}

    # --- returns
    s, J["returns_test"] = table("Return ($/quarter, TEST b505)", [
        ("primary (mean of world means)", "return_usd", 0)] + [(f"{w}", "return_usd", 0) for w in []], COLS_TEST)
    md.append(s)
    rt = {}
    for a in ["maxrl_final", "base_v6"] + FRONTIER:
        rt[a] = {w: arm_val(a, "return_usd", w) for w in WORLDS}
        rt[a]["primary"] = arm_val(a, "return_usd")
        rs = valid(a)
        rt[a]["n"] = len(rs)
        rt[a]["worst"] = min(r["return_usd"] for r in rs)
        rt[a]["n_below_minus10k"] = sum(1 for r in rs if r["return_usd"] < -10000)
    J["returns_by_model_test"] = rt
    lines = ["**Per-world return, TEST ($)**", "", "| policy | primary | null | account_line | supplier_det | worst quarter | quarters < -$10K |", "|---|---|---|---|---|---|---|"]
    for a in sorted(rt, key=lambda a: -rt[a]["primary"]):
        x = rt[a]
        lines.append(f"| {a} | {x['primary']:,.0f} | {x['null']:,.0f} | {x['account_line']:,.0f} | {x['supplier_deterioration']:,.0f} | {x['worst']:,.0f} | {x['n_below_minus10k']}/{x['n']} |")
    md.append("\n".join(lines) + "\n")
    rv = {}
    for a in ["maxrl_final_val", "base_v6_val", "q36_base_v1_val"]:
        allq = [r for r in ROWS if r["arm"] == a]
        rv[a] = {w: st.mean([(r["return_usd"] if r["status"] == "valid" else -25000) for r in allq if r["world"] == w]) for w in WORLDS}
        rv[a]["primary_failure_as_-25k"] = st.mean(rv[a][w] for w in WORLDS)
        rv[a]["failures"] = [f"{r['world']}:{r.get('failure')}@d{r.get('failure_day')}" for r in allq if r["status"] != "valid"]
        rv[a]["n"] = len(allq)
    J["returns_val"] = rv
    lines = ["**Per-world return, VAL b504 ($; a model-failure quarter scored -$25K)**", "", "| policy | primary | null | account_line | supplier_det | failures |", "|---|---|---|---|---|---|"]
    for a, x in rv.items():
        lines.append(f"| {a} (n={x['n']}) | {x['primary_failure_as_-25k']:,.0f} | {x['null']:,.0f} | {x['account_line']:,.0f} | {x['supplier_deterioration']:,.0f} | {', '.join(x['failures']) or '-'} |")
    md.append("\n".join(lines) + "\n")

    # --- Q1 demand inference
    s, J["q1"] = table("Q1 demand inference and walk-in service (TEST)", [
        ("days with reasoning about stockouts (frac)", "days_flag_stockout", 2),
        ("days with explicit censoring/true-demand inference (frac)", "days_flag_infer", 2),
        ("purchase rationales mentioning stockouts (frac)", "rationale_stockout_frac", 3),
        ("next-morning order / sku mean, after >=4h walk-in stockout", "order_ratio_after_stockout", 2),
        ("next-morning order / sku mean, after no stockout", "order_ratio_after_no_stockout", 2),
        ("P(order next morning | stockout >=4h)", "p_order_after_stockout", 2),
        ("P(order next morning | no stockout)", "p_order_after_no_stockout", 2),
        ("walk-in fill (pos sold / latent walk-in demand)", "walkin_fill", 3),
        ("sku-days with any walk-in stockout (frac)", "stockout_sku_days_frac", 3),
        ("walk-in stockout hours / quarter", "stockout_hours_grade", 0),
        ("stockout-hour share, days 0-29", "stockout_hours_frac_by_phase[0]", 3),
        ("stockout-hour share, days 60-89", "stockout_hours_frac_by_phase[2]", 3),
    ])
    md.append(s)
    s, J["q1_val"] = table("Q1 on VAL (ours + Qwen3.6/v1 contrast)", [
        ("days with reasoning about stockouts (frac)", "days_flag_stockout", 2),
        ("days with explicit censoring/true-demand inference (frac)", "days_flag_infer", 2),
        ("next-morning order ratio after stockout", "order_ratio_after_stockout", 2),
        ("next-morning order ratio after no stockout", "order_ratio_after_no_stockout", 2),
        ("walk-in fill", "walkin_fill", 3),
    ], COLS_VAL)
    md.append(s)

    # --- Q2 account cover
    for w in (None, "null", "account_line", "supplier_deterioration"):
        s, J[f"q2_{w or 'all'}"] = table("Q2 account cover (TEST)", [
            ("first day any cover set", "cover_first_day", 1),
            ("accounts with cover '*' set on day 0 (of 6)", "cover_accounts_star_day0", 1),
            ("cover-off rules over the quarter", "cover_rules_off", 1),
            ("per-product cover rules", "cover_rules_per_sku", 1),
            ("account shortfall kg (ordered - delivered)", "acct_short_kg", 0),
            ("account tonnage ordered (t)", "acct_ordered_t", 2),
            ("closing demand reserve ($)", "reserve_usd", 0),
        ], COLS_TEST, w)
        if w in (None, "null", "account_line"):
            md.append(s)

    # --- Q3 suppliers
    s, J["q3"] = table("Q3 supplier management (TEST, all worlds)", [
        ("top-supplier share of ordered grams", "top_supplier_share", 3),
        ("suppliers used (>2% of grams)", "n_suppliers_used", 1),
        ("claims auto-filed per quarter (not an agent action)", "claims_filed", 0),
        ("claim recoveries ($)", "claims_approved_usd", 0),
    ])
    md.append(s)
    for w in ("supplier_deterioration", "null"):
        s, J[f"q3_{w}"] = table("Q3 discount supplier share before/after day 21", [
            ("discount-supplier share of ordered grams, days 0-20", "disc_share_pre21", 3),
            ("discount-supplier share of ordered grams, days 21-89", "disc_share_post21", 3),
            ("discount-supplier share of received grams, orders day 0-20", "disc_recv_share_pre21", 3),
            ("discount-supplier share of received grams, orders day 21-89", "disc_recv_share_post21", 3),
            ("local share of stock", "local_share", 3),
            ("return ($)", "return_usd", 0),
        ], COLS_TEST, w)
        md.append(s)
    # weekly trajectory for deterioration world
    traj = {}
    for name, arms in COLS_TEST:
        arms = arms if arms is not None else FRONTIER
        traj[name] = [group_val(arms, f"disc_share_weekly[{k}]", "supplier_deterioration") for k in range(13)]
    J["q3_weekly_disc_share_supplier_det"] = traj
    lines = ["**Discount-supplier share of ordered grams by week, supplier_deterioration world (TEST)**", "",
             "| policy | " + " | ".join(f"d{7*k}-{7*k+6}" for k in range(13)) + " |", "|---|" + "---|" * 13]
    for name, v in traj.items():
        lines.append(f"| {name} | " + " | ".join(fmt(x, 2) for x in v) + " |")
    md.append("\n".join(lines) + "\n")

    # --- Q4 local market
    s, J["q4"] = table("Q4 local market (TEST)", [
        ("local share of stock (local / (local + supplier received))", "local_share", 3),
        ("local share, days 0-29", "local_share_by_phase[0]", 3),
        ("local share, days 30-59", "local_share_by_phase[1]", 3),
        ("local share, days 60-89", "local_share_by_phase[2]", 3),
        ("days with a local purchase (of 90)", "local_days", 0),
        ("products bought locally (of 20)", "local_skus", 1),
        ("share of account grams filled from local", "acct_local_fill_share", 3),
        ("local spend ($)", "local_usd", 0),
        ("local price / cheapest same-day supplier quote", "local_price_ratio_vs_quote", 2),
        ("premium over cheapest same-day quote ($)", "local_premium_vs_quote_usd", 0),
        ("account revenue on locally covered grams ($)", "covered_line_revenue_usd", 0),
        ("covered-line margin: that revenue - local cost ($)", "covered_line_margin_usd", 0),
        ("local lines where cost > 1.05 x line revenue (frac)", "local_lines_loss_making", 2),
        ("local grams: no agent truck due that sku-day", "local_cause_share.no_truck_due", 2),
        ("local grams: a truck due was late/short/cancelled", "local_cause_share.late_or_short_truck", 2),
        ("local grams: truck on time but too small", "local_cause_share.truck_on_time_too_small", 2),
        ("days with reasoning weighing local cost/premium (frac)", "days_flag_local_econ", 2),
    ])
    md.append(s)
    s, J["q4_null"] = table("Q4 local market, null world", [
        ("local share of stock", "local_share", 3),
        ("local spend ($)", "local_usd", 0),
        ("covered-line margin ($)", "covered_line_margin_usd", 0),
        ("account shortfall kg", "acct_short_kg", 0),
        ("gross margin", "gross_margin", 3),
        ("return ($)", "return_usd", 0),
    ], COLS_TEST, "null")
    md.append(s)

    # --- Q5 volume
    s, J["q5"] = table("Q5 volume, buffers, waste (TEST)", [
        ("supplier tonnage received (t)", "received_t", 2),
        ("supplier tonnage ordered (t)", "ordered_t", 2),
        ("local tonnage (t)", "local_t", 2),
        ("(supplier + local) / (latent walk-in + account orders)", "coverage_ratio", 3),
        ("(supplier + local) / sold", "supply_over_sold", 3),
        ("share of offered supplier capacity bought", "capacity_utilisation", 3),
        ("sku-days with >=90% of offered capacity bought", "sku_days_capacity_saturated_frac", 3),
        ("orders at the offer's max_cases (frac)", "po_at_max_cases_frac", 2),
        ("mean on-hand / mean daily sales (days of cover)", "days_of_cover", 2),
        ("promised lead time, grams-weighted (days)", "promised_lead_days_wmean", 2),
        ("revenue ($)", "revenue_usd", 0),
        ("gross margin (revenue - COGS)/revenue", "gross_margin", 3),
        ("spoilage write-off ($)", "spoilage_usd", 0),
        ("expired + aged-cull grams / stock received", "expired_plus_cull_pct_received", 4),
        ("inspection-rejected tonnage (t)", "inspection_rejected_t", 3),
    ])
    md.append(s)

    # --- Q6 notebook
    s, J["q6"] = table("Q6 memory and notebook (TEST)", [
        ("final notebook (notes.md) bytes", "notebook_bytes", 0),
        ("notebook markdown headings", "notebook_headings", 1),
        ("notebook tracks in-transit/promised POs (frac of quarters)", "notebook_has.in_transit", 2),
        ("notebook has check/verify rules (frac)", "notebook_has.checklist", 2),
        ("notebook has demand estimates (frac)", "notebook_has.demand_estimates", 2),
        ("memory tool calls per day", "memory_calls_per_day", 2),
        ("total memory used at close (bytes)", "memory_used_bytes_final", 0),
    ])
    md.append(s)

    # --- Q7 RL vs base
    comp = [("revenue", "revenue_usd"), ("COGS (incl. local buys)", "cogs_usd"), ("fees: dispatch", "fees_usd.dispatch"),
            ("fees: delivery", "fees_usd.delivery"), ("fees: receiving", "fees_usd.receiving"), ("fees: overhead", "fees_usd.overhead"),
            ("spoilage", "spoilage_usd"), ("inspection write-down", "inspection_writedown_usd"),
            ("claim recoveries", "claim_recoveries_usd"), ("refunds", "refunds_usd"),
            ("valuation adj. (incl. -reserve)", "valuation_adj_usd"), ("return", "return_usd"),
            ("memo: local spend", "local_usd"), ("memo: local premium vs quote", "local_premium_vs_quote_usd"),
            ("memo: demand reserve", "reserve_usd")]
    cols = [("MaxRL final", ["maxrl_final"]), ("base v6", ["base_v6"]), ("Astra", ["gpt-6-astra"])]
    lines = ["**Q7 return components ($/quarter, TEST, mean of world means)**", "", "| component | MaxRL final | base v6 | Astra | MaxRL - base | MaxRL - Astra |", "|---|---|---|---|---|---|"]
    J["q7_components"] = {}
    for label, key in comp:
        a, b, c = [group_val(x, key) for _, x in cols]
        J["q7_components"][label] = {"maxrl": a, "base": b, "astra": c}
        lines.append(f"| {label} | {fmt(a,0)} | {fmt(b,0)} | {fmt(c,0)} | {fmt(a-b,0)} | {fmt(a-c,0)} |")
    md.append("\n".join(lines) + "\n")
    s, J["q7_behaviour"] = table("Q7 behaviour, MaxRL final vs base (TEST and VAL)", [
        ("model calls per day", "calls_per_day", 2),
        ("output tokens per day (ours)", "output_tokens_per_day", 0),
        ("reasoning chars per call (ours)", "reasoning_chars_per_call", 0),
        ("think-cap hits per quarter", "think_cap_hits", 2),
        ("tool calls per day", "tool_calls_per_day", 1),
        ("accepted purchase orders per day", "accepted_pos_per_day", 1),
        ("tool-call error rate", "tool_error_rate", 3),
        ("MALFORMED errors per quarter", "malformed_errors", 1),
        ("rejected purchase orders per quarter", "po_rejections", 1),
        ("max consecutive failed tool results, any reason", "max_consecutive_errors", 1),
    ], [("MaxRL final", ["maxrl_final"]), ("base v6", ["base_v6"]), ("MaxRL (VAL)", ["maxrl_final_val"]),
        ("base v6 (VAL)", ["base_v6_val"]), ("Astra", ["gpt-6-astra"]), ("all 9 frontier", None)])
    md.append(s)
    # MaxRL - base per world, TEST and VAL
    dd = {}
    for w in WORLDS:
        dd[w] = {"test": arm_val("maxrl_final", "return_usd", w) - arm_val("base_v6", "return_usd", w),
                 "val": rv["maxrl_final_val"][w] - rv["base_v6_val"][w]}
    J["q7_maxrl_minus_base_by_world"] = dd
    lines = ["**MaxRL final minus base v6, by world ($)**", "", "| world | TEST | VAL (failure = -$25K) |", "|---|---|---|"]
    for w, x in dd.items():
        lines.append(f"| {w} | {x['test']:+,.0f} | {x['val']:+,.0f} |")
    md.append("\n".join(lines) + "\n")

    # --- Q8 failure modes
    s, J["q8"] = table("Q8 remaining failure modes (TEST)", [
        ("engine-rejected actions per quarter (grade)", "n_rejected_grade", 1),
        ("ordered grams/day, days 80-89 vs days 10-79", "late_order_ratio_80_89", 2),
        ("ordered grams/day, days 85-89 vs days 10-79", "late_order_ratio_85_89", 2),
        ("closing on-hand / mean daily sales (days)", "close_days_of_cover", 2),
        ("closing inventory value ($)", "close_inventory_value_usd", 0),
        ("dock refusals (storage full) per quarter", "dock_refusals", 2),
        ("refused-at-dock kg per quarter", "overflow_refused_kg", 1),
    ])
    md.append(s)
    # tool error codes for ours
    codes = defaultdict(lambda: defaultdict(list))
    for a in ["maxrl_final", "base_v6", "maxrl_final_val", "base_v6_val"]:
        for r in [x for x in ROWS if x["arm"] == a]:
            for k, v in (r.get("tool_errors") or {}).items():
                codes[a][k].append(v)
    J["q8_tool_error_codes_per_quarter"] = {a: {k: sum(v) / len([x for x in ROWS if x["arm"] == a]) for k, v in c.items()} for a, c in codes.items()}
    top = sorted(J["q8_tool_error_codes_per_quarter"]["maxrl_final"].items(), key=lambda kv: -kv[1])[:10]
    lines = ["**Tool error codes per quarter (top 10 for MaxRL final, TEST)**", "", "| tool:code | MaxRL final | base v6 | MaxRL (VAL) | base v6 (VAL) |", "|---|---|---|---|---|"]
    for k, _ in top:
        lines.append(f"| {k} | " + " | ".join(fmt(J['q8_tool_error_codes_per_quarter'][a].get(k, 0), 2) for a in ["maxrl_final", "base_v6", "maxrl_final_val", "base_v6_val"]) + " |")
    md.append("\n".join(lines) + "\n")

    # quarter counts for cover on day 0
    cnt = {}
    for name, arms in [("maxrl_final", ["maxrl_final"]), ("base_v6", ["base_v6"]), ("maxrl_final_val", ["maxrl_final_val"]),
                       ("base_v6_val", ["base_v6_val"]), ("gpt-6-astra", ["gpt-6-astra"]), ("top4", TOP4), ("all9", FRONTIER)]:
        rs = [r for a in arms for r in valid(a)]
        cnt[name] = {"n": len(rs), "all6_star_day0": sum(1 for r in rs if r["cover_accounts_star_day0"] == 6),
                     "never_set_cover": sum(1 for r in rs if r["cover_first_day"] is None),
                     "switched_cover_off": sum(1 for r in rs if r["cover_rules_off"] > 0)}
    J["cover_quarter_counts"] = cnt
    for extra in ("components_by_world.json", "rl_vs_base.json"):
        if (OUT / extra).exists():
            J[extra[:-5]] = json.loads((OUT / extra).read_text())
    J["_meta"] = {"aggregation": "mean over worlds of per-world means over valid quarters; frontier groups average per-model values",
                  "top4": TOP4, "frontier_models": FRONTIER,
                  "sources": ["engine/grade.json", "engine/public_records/*.csv", "engine/model_transcript.jsonl(.gz)", "engine/model_notebook.txt"]}
    (OUT / "tables.md").write_text("\n".join(md))
    (OUT / "findings.json").write_text(json.dumps(J, indent=1, default=lambda o: None))
    print("\n".join(md))


for r in ROWS:
    if r["status"] == "valid":
        r["n_rejected_grade"] = sum((r.get("rejected_by_reason") or {}).values())

if __name__ == "__main__":
    main()
