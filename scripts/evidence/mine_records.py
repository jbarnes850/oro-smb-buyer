"""Per-quarter business metrics from engine/grade.json + engine/public_records/*.csv (final snapshots).

Same code path for our arms and the frontier TEST replays. Agent decisions only: purchase_orders
with actor == 'agent' (the prelude days -7..-1 are the 'house' policy). Agent days are 0..89.

    python3 -I mine_records.py   -> records_metrics.json
"""
import json
import statistics as st
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, quarters, read_csv, read_json, world_of  # noqa: E402

LB = 453.59237
PHASES = ((0, 29), (30, 59), (60, 89))


def i(x):
    return int(float(x)) if x not in (None, "") else 0


def phase(d):
    for k, (a, b) in enumerate(PHASES):
        if a <= d <= b:
            return k
    return None


def discount_supplier(offers):
    """Supplier with the lowest mean price ratio vs the per-(day, sku) mean quote (public offers)."""
    by = defaultdict(list)
    for o in offers:
        by[(i(o["quote_day"]), o["sku_id"])].append((o["supplier_id"], float(o["price_cents_per_lb"])))
    ratios = defaultdict(list)
    for k, v in by.items():
        if len({s for s, _ in v}) < 2:
            continue
        m = st.mean(p for _, p in v)
        for s, p in v:
            ratios[s].append(p / m)
    return min(ratios, key=lambda s: st.mean(ratios[s])) if ratios else None


def mine(args):
    arm, split, source, d = args
    d = Path(d)
    eng = d / "engine"
    row = {"arm": arm, "split": split, "source": source, "dir": str(d)}
    if source == "frontier":
        fr = read_json(d / "frontier-replay-result.json")
        row["world"] = world_of(fr["world"])
        row["status"] = "valid" if (eng / "grade.json").exists() else "no_grade"
    else:
        q = read_json(d / "quarter-result.json")
        row["world"] = world_of(d.name)
        row["status"] = q["status"]
        row["failure"] = (q.get("failure") or {}).get("code")
        row["failure_day"] = (q.get("failure") or {}).get("last_day")
    g = read_json(eng / "grade.json")
    if g is None or row["status"] != "valid":
        return row
    b, a = g["bridge"], g["availability"]["total"]
    row["return_usd"] = b["business_return_cents"] / 100
    row["revenue_usd"] = b["revenue"] / 100
    row["cogs_usd"] = b["cogs"] / 100
    row["gross_margin"] = (b["revenue"] - b["cogs"]) / b["revenue"] if b["revenue"] else None
    row["fees_usd"] = {k: v / 100 for k, v in b["fees_by_kind"].items()}
    row["spoilage_usd"] = b["spoilage_writeoff"] / 100
    row["inspection_writedown_usd"] = b["inspection_writedown"] / 100
    row["claim_recoveries_usd"] = b["claim_recoveries"] / 100
    row["refunds_usd"] = b["refunds"] / 100
    row["valuation_adj_usd"] = (b["valuation_adjustment_close"] - b["valuation_adjustment_open"]) / 100
    cc = g["value"]["close_components"]
    row["reserve_usd"] = cc.get("customer_demand_reserve", 0) / 100
    row["close_inventory_value_usd"] = cc.get("inventory_value", 0) / 100
    row["close_inventory_book_usd"] = cc.get("inventory_book", 0) / 100
    row["walkin_fill"] = a["pos_sold_grams"] / a["latent_pos_demand_grams"] if a["latent_pos_demand_grams"] else None
    row["latent_pos_t"] = a["latent_pos_demand_grams"] / 1e6
    row["pos_sold_t"] = a["pos_sold_grams"] / 1e6
    row["lost_pos_t"] = a["lost_grams"] / 1e6
    row["stockout_hours_grade"] = a["stockout_hours"]
    row["acct_ordered_t"] = a["account_ordered_grams"] / 1e6
    row["acct_short_kg"] = (a["account_ordered_grams"] - a["account_sold_grams"]) / 1000
    row["rejected_by_reason"] = g["constraints"]["rejected_actions_by_reason"]
    row["dock_refusals"] = g["constraints"]["dock_refusals"]
    row["overflow_refused_kg"] = g["constraints"]["overflow_refused_grams"] / 1000

    pr = eng / "public_records"
    pos = read_csv(pr / "purchase_orders.csv")
    agent = [p for p in pos if p["actor"] == "agent"]
    row["n_agent_pos"] = len(agent)
    by_sup_g = defaultdict(float)
    by_sup_g_phase = defaultdict(lambda: defaultdict(float))
    by_day_g = defaultdict(float)
    by_day_n = defaultdict(int)
    by_sku_day_g = defaultdict(float)
    lead = []
    po_actor = {p["po_id"]: p["actor"] for p in pos}
    for p in agent:
        gr = i(p["cases_ordered"]) * i(p["case_grams"])
        od = i(p["order_day"])
        by_sup_g[p["supplier_id"]] += gr
        by_sup_g_phase[p["supplier_id"]][od] += gr
        by_day_g[od] += gr
        by_day_n[od] += 1
        by_sku_day_g[(p["sku_id"], od)] += gr
        lead.append((i(p["promised_arrival_day"]) - od, gr))
    tot = sum(by_sup_g.values())
    row["ordered_t"] = tot / 1e6
    row["ordered_by_supplier_t"] = {s: v / 1e6 for s, v in by_sup_g.items()}
    row["top_supplier_share"] = max(by_sup_g.values()) / tot if tot else None
    row["top_supplier"] = max(by_sup_g, key=by_sup_g.get) if tot else None
    row["hhi_supplier"] = sum((v / tot) ** 2 for v in by_sup_g.values()) if tot else None
    row["n_suppliers_used"] = sum(1 for v in by_sup_g.values() if v > 0.02 * tot) if tot else 0
    row["promised_lead_days_wmean"] = sum(l * g_ for l, g_ in lead) / sum(g_ for _, g_ in lead) if lead else None
    row["orders_per_day"] = len(agent) / 90
    row["order_days"] = sum(1 for dd in range(90) if by_day_n.get(dd))
    # end-of-quarter buying: daily ordered grams on days 80-89 vs days 10-79
    mid = st.mean(by_day_g.get(dd, 0) for dd in range(10, 80))
    late = st.mean(by_day_g.get(dd, 0) for dd in range(80, 90))
    last5 = st.mean(by_day_g.get(dd, 0) for dd in range(85, 90))
    row["late_order_ratio_80_89"] = late / mid if mid else None
    row["late_order_ratio_85_89"] = last5 / mid if mid else None

    # discount supplier and deterioration response
    offers = read_csv(pr / "offers.csv")
    disc = discount_supplier(offers)
    row["discount_supplier"] = disc
    def share(lo, hi):
        num = sum(v for dd, v in by_sup_g_phase[disc].items() if lo <= dd <= hi)
        den = sum(v for s in by_sup_g_phase for dd, v in by_sup_g_phase[s].items() if lo <= dd <= hi)
        return num / den if den else None
    row["disc_share_pre21"] = share(0, 20)
    row["disc_share_post21"] = share(21, 89)
    row["disc_share_weekly"] = [share(w, w + 6) for w in range(0, 90, 7)]
    # supplier reliability by order day (agent POs), from final PO status
    rel = defaultdict(lambda: {"n": 0, "late": 0, "short": 0, "cases": 0, "cases_recv": 0})
    for p in agent:
        od = i(p["order_day"])
        k = (p["supplier_id"], "pre21" if od < 21 else "post21")
        r = rel[k]
        r["n"] += 1
        r["cases"] += i(p["cases_ordered"])
        r["cases_recv"] += i(p["cases_received"])
        if i(p["cases_cancelled"]) > 0:
            r["short"] += 1
        if p["arrival_day"] not in ("", None) and i(p["arrival_day"]) > i(p["promised_arrival_day"]):
            r["late"] += 1
    row["supplier_reliability"] = {f"{s}|{ph}": v for (s, ph), v in rel.items()}

    # receipts (agent POs, arrival day >= 0)
    rec = read_csv(pr / "receipts.csv")
    recv_g = 0
    recv_g_phase = [0, 0, 0]
    cost_sku = defaultdict(lambda: [0, 0])
    insp_rej = 0
    for r in rec:
        if po_actor.get(r["po_id"]) != "agent":
            continue
        ga = i(r["grams_accepted"])
        recv_g += ga
        insp_rej += i(r["grams_rejected_inspection"])
        ph = phase(i(r["arrival_day"]))
        if ph is not None:
            recv_g_phase[ph] += ga
        cost_sku[r["sku_id"]][0] += i(r["goods_cost_cents"])
        cost_sku[r["sku_id"]][1] += ga
    row["received_t"] = recv_g / 1e6
    row["inspection_rejected_t"] = insp_rej / 1e6
    row["late_receipt_share"] = (sum(1 for r in rec if po_actor.get(r["po_id"]) == "agent" and i(r["days_late"]) > 0)
                                 / max(1, sum(1 for r in rec if po_actor.get(r["po_id"]) == "agent")))

    # local purchases (all are account-line fills under shortfall cover)
    lps = [l for l in read_csv(pr / "local_purchases.csv") if l.get("status", "bought") == "bought" and i(l["day"]) >= 0]
    loc_g = sum(i(l["grams"]) for l in lps)
    loc_c = sum(i(l["amount_cents"]) for l in lps)
    loc_phase_g = [0, 0, 0]
    for l in lps:
        loc_phase_g[phase(i(l["day"]))] += i(l["grams"])
    row["local_t"] = loc_g / 1e6
    row["local_usd"] = loc_c / 100
    row["local_share"] = loc_g / (loc_g + recv_g) if (loc_g + recv_g) else None
    row["local_share_by_phase"] = [lg / (lg + rg) if (lg + rg) else None for lg, rg in zip(loc_phase_g, recv_g_phase)]
    row["local_days"] = len({i(l["day"]) for l in lps})
    row["local_skus"] = len({l["sku_id"] for l in lps})
    row["local_lines"] = len(lps)
    # premium vs cheapest same-day supplier quote for that sku (goods only), fallback own avg cost
    cheapest = {}
    for o in offers:
        k = (i(o["quote_day"]), o["sku_id"])
        p = float(o["price_cents_per_lb"])
        if k not in cheapest or p < cheapest[k]:
            cheapest[k] = p
    prem_quote = prem_own = 0.0
    ratio_num = ratio_den = 0.0
    for l in lps:
        gr = i(l["grams"])
        k = (i(l["day"]), l["sku_id"])
        q = cheapest.get(k)
        if q is None:
            qs = [v for (dd, s), v in cheapest.items() if s == l["sku_id"] and abs(dd - k[0]) <= 3]
            q = st.mean(qs) if qs else None
        if q is not None:
            prem_quote += i(l["amount_cents"]) - gr / LB * q
            ratio_num += i(l["amount_cents"])
            ratio_den += gr / LB * q
        c, gg = cost_sku.get(l["sku_id"], (0, 0))
        if gg:
            prem_own += i(l["amount_cents"]) - gr * c / gg
        elif q is not None:
            prem_own += i(l["amount_cents"]) - gr / LB * q
    row["local_premium_vs_quote_usd"] = prem_quote / 100
    row["local_premium_vs_owncost_usd"] = prem_own / 100
    row["local_price_ratio_vs_quote"] = ratio_num / ratio_den if ratio_den else None

    # account lines (days 0..89)
    col = [c for c in read_csv(pr / "customer_order_lines.csv") if i(c["due_day"]) >= 0]
    filled = sum(i(c["filled_grams"]) for c in col)
    row["acct_filled_t"] = filled / 1e6
    row["acct_local_fill_share"] = loc_g / filled if filled else None
    short_phase = [0, 0, 0]
    for c in col:
        ph = phase(i(c["due_day"]))
        if ph is not None:
            short_phase[ph] += i(c["cancelled_grams"])
    row["acct_cancelled_kg_by_phase"] = [x / 1000 for x in short_phase]
    row["acct_cancelled_kg_d0_89"] = sum(short_phase) / 1000
    row["n_accounts"] = len({c["customer_id"] for c in col})
    # P&L of locally covered account grams: account revenue on those grams minus local cost.
    # Shorting instead would forfeit that revenue and pay account_service_fraction (5%) of it
    # (plus future account shrink/reserve in reacting worlds, not counted here).
    price_line = {c["line_id"]: i(c["unit_price_cents_per_lb"]) for c in col}
    cov_rev = sum(i(l["grams"]) / LB * price_line.get(l["order_line_id"], 0) for l in lps)
    row["covered_line_revenue_usd"] = cov_rev / 100
    row["covered_line_margin_usd"] = (cov_rev - loc_c) / 100
    row["cover_vs_short_usd"] = (cov_rev * 1.05 - loc_c) / 100  # >0: covering beat shorting (static)
    row["local_lines_loss_making"] = (sum(1 for l in lps if i(l["grams"]) / LB * price_line.get(l["order_line_id"], 0) * 1.05 < i(l["amount_cents"]))
                                      / len(lps)) if lps else None
    # account grams due vs due-day morning on-hand is not reconstructable from final snapshots;
    # instead: share of local grams on lines whose sku had a same-day agent receipt
    recv_day_sku = {(i(r["arrival_day"]), r["sku_id"]) for r in rec if po_actor.get(r["po_id"]) == "agent"}
    row["local_share_same_day_receipt"] = (sum(i(l["grams"]) for l in lps if (i(l["day"]), l["sku_id"]) in recv_day_sku) / loc_g) if loc_g else None
    # why was local needed? classify each local gram by the agent POs promised for that sku that day
    promised = defaultdict(list)
    for p in agent:
        promised[(i(p["promised_arrival_day"]), p["sku_id"])].append(p)
    inv = {(i(x["day"]), x["sku_id"]): i(x["on_hand_grams"]) for x in read_csv(pr / "inventory_daily.csv")}
    cls = defaultdict(int)
    for l in lps:
        k = (i(l["day"]), l["sku_id"])
        ps = promised.get(k, [])
        if not ps:
            cls["no_truck_due"] += i(l["grams"])
        elif any((p["arrival_day"] not in ("", None) and i(p["arrival_day"]) > k[0]) or i(p["cases_cancelled"]) > 0
                 or i(p["cases_refused"]) > 0 or (p["arrival_day"] in ("", None)) for p in ps):
            cls["late_or_short_truck"] += i(l["grams"])
        else:
            cls["truck_on_time_too_small"] += i(l["grams"])
    row["local_cause_share"] = {kk: v / loc_g for kk, v in cls.items()} if loc_g else {}
    # stock left at the previous close for that sku (0 = shelf was already empty overnight)
    row["local_prev_close_empty_share"] = (sum(i(l["grams"]) for l in lps if inv.get((i(l["day"]) - 1, l["sku_id"]), 0) == 0) / loc_g) if loc_g else None

    # offered capacity utilisation (quotes on days 0..89): ordered grams / sum(max_cases*case_grams)
    cap = 0
    cap_day_sku = defaultdict(int)
    offer_max = {}
    for o in offers:
        qd = i(o["quote_day"])
        if 0 <= qd <= 89:
            g_ = i(o["max_cases"]) * i(o["case_grams"])
            cap += g_
            cap_day_sku[(qd, o["sku_id"])] += g_
            offer_max[(o["offer_id"], o["offer_version"])] = i(o["max_cases"])
    row["offered_capacity_t"] = cap / 1e6
    row["capacity_utilisation"] = tot / cap if cap else None
    at_max = sum(1 for p in agent if offer_max.get((p["offer_id"], p.get("offer_version", "1"))) == i(p["cases_ordered"]))
    row["po_at_max_cases_frac"] = at_max / len(agent) if agent else None
    # sku-days where the agent bought >= 90% of all offered capacity
    ordered_day_sku = defaultdict(int)
    for p in agent:
        ordered_day_sku[(i(p["order_day"]), p["sku_id"])] += i(p["cases_ordered"]) * i(p["case_grams"])
    sat = sum(1 for k, v in cap_day_sku.items() if v and ordered_day_sku.get(k, 0) >= 0.9 * v)
    row["sku_days_capacity_saturated_frac"] = sat / len(cap_day_sku) if cap_day_sku else None
    # received (not ordered) share of the discount supplier, by order day
    rs = defaultdict(lambda: [0, 0])
    for p in agent:
        od = i(p["order_day"])
        k = 0 if od < 21 else 1
        gr = i(p["cases_received"]) * i(p["case_grams"])
        rs[k][1] += gr
        if p["supplier_id"] == disc:
            rs[k][0] += gr
    row["disc_recv_share_pre21"] = rs[0][0] / rs[0][1] if rs[0][1] else None
    row["disc_recv_share_post21"] = rs[1][0] / rs[1][1] if rs[1][1] else None

    # shortfall cover rules
    sc = read_csv(pr / "shortfall_cover_rules.csv")
    on = [s for s in sc if s["cover"] == "True"]
    off = [s for s in sc if s["cover"] == "False"]
    row["cover_first_day"] = min((i(s["day"]) for s in on), default=None)
    row["cover_accounts_star_day0"] = len({s["customer_id"] for s in on if s["sku_id"] == "*" and i(s["day"]) == 0})
    row["cover_accounts_star_ever"] = len({s["customer_id"] for s in on if s["sku_id"] == "*"})
    row["cover_rules_on"] = len(on)
    row["cover_rules_off"] = len(off)
    row["cover_rules_per_sku"] = sum(1 for s in sc if s["sku_id"] != "*")

    # disposals (day >= 0)
    disp = defaultdict(int)
    for x in read_csv(pr / "disposals.csv"):
        if i(x["day"]) >= 0:
            disp[x["reason"]] += i(x["grams"])
    row["disposal_kg"] = {k: v / 1000 for k, v in disp.items()}
    row["expired_pct_received"] = disp.get("expired", 0) / (recv_g + loc_g) if (recv_g + loc_g) else None
    row["expired_plus_cull_pct_received"] = (disp.get("expired", 0) + disp.get("age_cull", 0)) / (recv_g + loc_g) if (recv_g + loc_g) else None

    # inventory and sales
    ds = {i(x["day"]): x for x in read_csv(pr / "daily_status.csv")}
    sd = read_csv(pr / "sales_daily.csv")
    sold_day = defaultdict(int)
    hours = {}
    pos_sold = {}
    for x in sd:
        dd = i(x["day"])
        if dd < 0:
            continue
        sold_day[dd] += i(x["grams_sold"])
        if x["channel"] == "pos":
            hours[(x["sku_id"], dd)] = i(x["hours_out_of_stock"])
            pos_sold[(x["sku_id"], dd)] = i(x["grams_sold"])
    oh = [i(ds[dd]["on_hand_grams"]) for dd in range(90) if dd in ds]
    mean_sold = st.mean(sold_day.get(dd, 0) for dd in range(90))
    row["mean_on_hand_t"] = st.mean(oh) / 1e6 if oh else None
    row["days_of_cover"] = st.mean(oh) / mean_sold if oh and mean_sold else None
    row["close_on_hand_t"] = i(ds[89]["on_hand_grams"]) / 1e6 if 89 in ds else None
    row["close_days_of_cover"] = i(ds[89]["on_hand_grams"]) / mean_sold if 89 in ds and mean_sold else None
    row["min_cash_usd"] = min(i(ds[dd]["cash_cents"]) for dd in range(90) if dd in ds) / 100
    row["max_credit_drawn_usd"] = max(i(ds[dd]["credit_drawn_cents"]) for dd in range(90) if dd in ds) / 100
    row["stockout_hours_pos_d0_89"] = sum(hours.values())
    skus = sorted({s for s, _ in hours})
    row["n_skus"] = len(skus)
    row["stockout_sku_days_frac"] = sum(1 for v in hours.values() if v > 0) / len(hours) if hours else None
    row["stockout_hours_frac_by_phase"] = [
        sum(v for (s, dd), v in hours.items() if a_ <= dd <= b_) / (16 * max(1, sum(1 for (s, dd) in hours if a_ <= dd <= b_)))
        for a_, b_ in PHASES]
    row["coverage_ratio"] = (recv_g + loc_g) / (a["latent_pos_demand_grams"] + a["account_ordered_grams"]) if (a["latent_pos_demand_grams"] + a["account_ordered_grams"]) else None
    row["supply_over_sold"] = (recv_g + loc_g) / sum(sold_day.values()) if sold_day else None

    # stockout -> next-morning order response (per sku, normalized by the sku's mean daily order)
    mean_order = {s: st.mean(by_sku_day_g.get((s, dd), 0) for dd in range(90)) for s in skus}
    resp = {"so": [], "noso": [], "so_any": [], "noso_any": []}
    resp3 = {"so": [], "noso": []}
    for s in skus:
        m = mean_order[s]
        if m <= 0:
            continue
        for dd in range(0, 88):
            h = hours.get((s, dd))
            if h is None:
                continue
            nxt = by_sku_day_g.get((s, dd + 1), 0)
            nxt3 = sum(by_sku_day_g.get((s, dd + k), 0) for k in (1, 2, 3)) / 3
            key = "so" if h >= 4 else ("noso" if h == 0 else None)
            if key:
                resp[key].append(nxt / m)
                resp[key + "_any"].append(1 if nxt > 0 else 0)
                resp3[key].append(nxt3 / m)
    row["order_ratio_after_stockout"] = st.mean(resp["so"]) if resp["so"] else None
    row["order_ratio_after_no_stockout"] = st.mean(resp["noso"]) if resp["noso"] else None
    row["p_order_after_stockout"] = st.mean(resp["so_any"]) if resp["so_any"] else None
    row["p_order_after_no_stockout"] = st.mean(resp["noso_any"]) if resp["noso_any"] else None
    row["order3_ratio_after_stockout"] = st.mean(resp3["so"]) if resp3["so"] else None
    row["order3_ratio_after_no_stockout"] = st.mean(resp3["noso"]) if resp3["noso"] else None
    row["n_stockout_events_ge4h"] = len(resp["so"])

    # claims (automatic at receipt; not an agent decision)
    cl = [c for c in read_csv(pr / "claims.csv") if i(c["filed_day"]) >= 0]
    row["claims_filed"] = len(cl)
    row["claims_approved_usd"] = sum(i(c["approved_cents"]) for c in cl) / 100
    return row


def main():
    qs = quarters()
    with Pool(16) as p:
        rows = p.map(mine, qs, chunksize=1)
    (OUT / "records_metrics.json").write_text(json.dumps(rows, indent=0))
    print(len(rows), "quarters ->", OUT / "records_metrics.json")


if __name__ == "__main__":
    main()
