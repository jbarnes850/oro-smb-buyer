"""Figures for the training report (reads run artifacts and the leaderboard snapshot only).

    python scripts/story_figures.py --config docs/story-config.json --out docs/figures [--json numbers.json]

The config maps figure labels to run directories (relative to --runs-dir, default SMB_RUNS_DIR) and names
the leaderboard snapshot written by scripts/leaderboard_snapshot.py (relative to the config file).
docs/story-config.json is the configuration that produced the published figures.

Figure 1  result     sealed test and validation businesses against all 16 leaderboard models
Figure 2  behaviors  the SMB Bench report's behaviours, measured the same way for every policy on the test business
Figure 3  rl         reward over the full final RL run, stability against a dollar reward, and held-out gains
Figure 4  levers     data, where the validation gain came from, and the remaining gap to GPT-6 Astra
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

from smb_buyer import analysis

plt = None  # matplotlib is imported in main() so --help works without the plots extra

INK, MUTED, GRID = "#1b2a38", "#6b7785", "#e3e7ea"
OURS, BASE, FRONT, OPEN, GOOD, BAD = "#c8691f", "#1f6fb2", "#aab3bc", "#6f7a86", "#0f8a72", "#b04a3a"
WORLDS = analysis.WORLDS
NICE = {"gpt-6-astra": "GPT-6 Astra", "gemini-4-carbon": "Gemini 4 Carbon", "claude-fable-5-2": "Claude Fable 5.2",
        "claude-fable-5-1": "Claude Fable 5.1", "gpt-6.1-sol": "GPT-6.1 Sol", "gemini-4-hydrogen": "Gemini 4 Hydrogen",
        "gemini-4-argon": "Gemini 4 Argon", "claude-opus-5-5": "Claude Opus 5.5",
        "deepseek-v4.1-flash-1017": "DeepSeek V4.1 Flash", "grok-4.7": "Grok 4.7", "muse-ember": "Muse Ember",
        "muse-spark-1.3": "Muse Spark 1.3", "glm-5.3": "GLM 5.3", "claude-haiku-5-5": "Claude Haiku 5.5",
        "muse-wonder-1.0": "Muse Wonder 1.0", "gpt-6-luna-high": "GPT-6 Luna"}
RL_ARM, BASE_ARM = "Qwen3.8-27B + RL (ours)", "Qwen3.8-27B, no RL (ours)"
STORE_HOURS = 16
CFG = {}
OUT = Path(".")


# ----------------------------------------------------------------------------------------------- helpers

def CLIMB():  # noqa: N802
    return [(lab.replace("\\n", "\n"), dirs) for lab, dirs in CFG["climb"]]


def color_of(name, open_weights=()):
    if "+ RL" in name:
        return OURS
    if "(ours)" in name:
        return BASE
    return OPEN if name in open_weights else FRONT


def short(name):
    return name.replace(" (ours)", "")


def bold_ours(ax, names):
    for tick, n in zip(ax.get_yticklabels(), names):
        if "(ours)" in n:
            tick.set_fontweight("bold")
            tick.set_color(INK)


def label(ax, name, x, y, offsets, size=7.5):
    dx, dy = offsets.get(name, (5, 3))
    ax.annotate(short(name), (x, y), textcoords="offset points", xytext=(dx, dy), fontsize=size,
                color=INK if "(ours)" in name else MUTED, ha="left" if dx >= 0 else "right")


def style(ax, title, xlabel="", ylabel="", grid="y"):
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def summary(dirs):
    rows, _ = analysis.quarters(dirs, CFG["runs"])
    return analysis.summarize(rows) if rows else None


def diff(a, b):
    return a["primary"] - b["primary"], math.sqrt(a["se"] ** 2 + b["se"] ** 2)


def read_csv(path):
    path = Path(path)
    gz = Path(str(path) + ".gz")
    if path.exists():
        return list(csv.DictReader(open(path, newline="")))
    if gz.exists():
        return list(csv.DictReader(io.TextIOWrapper(gzip.open(gz), newline="")))
    return []


def num(x):
    return int(float(x)) if x not in (None, "") else 0


# ------------------------------------------------------------------------------------ leaderboard snapshot

def leaderboard(split):
    """{model: summary} from the leaderboard's scored runs, the fixed rule, and the open-weight models."""
    snap = json.loads(Path(CFG["leaderboard"]).read_text())
    world = {task: w for w, task in snap["tasks"][split].items()}
    by, ref = defaultdict(lambda: defaultdict(list)), {}
    for r in snap["runs"]:
        w = world.get(r["task"])
        if w is None:
            continue
        if r["reference_return_cents"] is not None:
            ref[w] = r["reference_return_cents"] / 100
        if r["status"] == "valid":
            by[NICE.get(r["model"], r["model"])][w].append(r["business_return_cents"] / 100)
    out = {}
    for m, ws in by.items():
        if all(ws.get(w) for w in WORLDS):
            means = {w: st.mean(ws[w]) for w in WORLDS}
            var = sum((st.variance(ws[w]) if len(ws[w]) > 1 else 0.0) / len(ws[w]) for w in WORLDS)
            out[m] = {"primary": st.mean(means.values()), "se": math.sqrt(var) / 3, "per_world": means,
                      "n": {w: len(ws[w]) for w in WORLDS}}
    rule = {"primary": st.mean(ref[w] for w in WORLDS), "per_world": ref}
    return out, rule, {NICE.get(m, m) for m in snap["open_weights"]}


# ----------------------------------------------------------------------------------- per-quarter records

def record_metrics(engine):
    """Behaviour measures from one quarter's public records and grade (same code for every policy)."""
    pr = engine / "public_records"
    out = {}
    # stock bought, by source
    local_g = sum(num(x["grams"]) for x in read_csv(pr / "local_purchases.csv") if x["status"] == "bought")
    local_c = sum(num(x["amount_cents"]) for x in read_csv(pr / "local_purchases.csv") if x["status"] == "bought")
    recv_g = sum(num(x["cases_received"]) * num(x["case_grams"]) for x in read_csv(pr / "purchase_orders.csv")
                 if x["actor"] == "agent")
    out.update(local_t=local_g / 1e6, supplier_t=recv_g / 1e6, local_usd=local_c / 100)
    # suppliers and promised lead time over the agent's own purchase orders
    by_sup, lead = defaultdict(float), []
    for p in read_csv(pr / "purchase_orders.csv"):
        if p["actor"] != "agent":
            continue
        g = num(p["cases_ordered"]) * num(p["case_grams"])
        by_sup[p["supplier_id"]] += g
        lead.append((num(p["promised_arrival_day"]) - num(p["order_day"]), g))
    tot = sum(by_sup.values())
    out["suppliers_used"] = sum(1 for v in by_sup.values() if v > 0.02 * tot) if tot else 0
    out["lead_days"] = sum(a * g for a, g in lead) / sum(g for _, g in lead) if lead else None
    # days of stock held, and the reaction to a bad stockout week (report Figure 8)
    sold, sku_sold, hours = defaultdict(int), defaultdict(int), {}
    for x in read_csv(pr / "sales_daily.csv"):
        d = num(x["day"])
        if 0 <= d <= 89:
            sold[d] += num(x["grams_sold"])
            sku_sold[x["sku_id"]] += num(x["grams_sold"])
            if x["channel"] == "pos":
                hours[(x["sku_id"], d)] = num(x["hours_out_of_stock"])
    status = {num(x["day"]): num(x["on_hand_grams"]) for x in read_csv(pr / "daily_status.csv")}
    mean_sold = st.mean(sold.get(d, 0) for d in range(90))
    out["days_of_stock"] = st.mean(status.get(d, 0) for d in range(90)) / mean_sold if mean_sold else None
    inv = defaultdict(dict)
    for x in read_csv(pr / "inventory_daily.csv"):
        if num(x["day"]) >= 0:
            inv[x["sku_id"]][num(x["day"])] = num(x["on_hand_grams"])
    react = []
    for sku, days in inv.items():
        daily = sku_sold[sku] / 90
        if daily <= 0:
            continue
        dos = [st.mean(days.get(d, 0) for d in range(7 * w, 7 * w + 7)) / daily for w in range(12)]
        for w in range(11):
            if sum(hours.get((sku, d), 0) for d in range(7 * w, 7 * w + 7)) / (STORE_HOURS * 7) >= 0.25:
                react.append(dos[w + 1] - dos[w])
    out["bad_week_reactions"] = react
    # account products with standing shortfall cover, day by day (report Figure 10)
    accounts = {x["customer_id"] for x in read_csv(pr / "customers.csv") if x["channel"] == "account"}
    pairs = {(x["customer_id"], x["sku_id"]) for x in read_csv(pr / "customer_order_lines.csv")
             if x["customer_id"] in accounts and num(x["due_day"]) >= 0}
    rules = sorted(read_csv(pr / "shortfall_cover_rules.csv"), key=lambda r: (num(r["day"]), num(r["set_t"])))
    star, spec, k, curve = {}, {}, 0, []
    for day in range(90):
        while k < len(rules) and num(rules[k]["day"]) <= day:
            r = rules[k]
            if r["sku_id"] == "*":
                star[r["customer_id"]] = r["cover"] == "True"
                for key in [p for p in spec if p[0] == r["customer_id"]]:
                    del spec[key]
            else:
                spec[(r["customer_id"], r["sku_id"])] = r["cover"] == "True"
            k += 1
        curve.append(sum(1 for c, s in pairs if spec.get((c, s), star.get(c, False))) / len(pairs) if pairs else 0.0)
    out["cover_curve"] = curve
    # profit components from the grade
    b = json.loads((engine / "grade.json").read_text())["bridge"]
    fees = b["fees_by_kind"]
    out["bridge"] = {"revenue": b["revenue"] / 100, "cogs": b["cogs"] / 100, "dispatch": fees.get("dispatch", 0) / 100,
                     "other_fees": sum(v for k_, v in fees.items() if k_ != "dispatch") / 100,
                     "waste": (b["spoilage_writeoff"] + b["inspection_writedown"]) / 100,
                     "return": b["business_return_cents"] / 100}
    return out


def policy_metrics(engines):
    ms = [record_metrics(e) for e in engines if (e / "grade.json").exists()]
    if not ms:
        return None
    react = [v for m in ms for v in m["bad_week_reactions"]]
    out = {k: st.mean(m[k] for m in ms if m[k] is not None)
           for k in ("local_t", "supplier_t", "local_usd", "suppliers_used", "lead_days", "days_of_stock")}
    out["bad_week_reaction"] = st.mean(react) if react else None
    out["bad_weeks"] = len(react)
    out["cover_curve"] = [st.mean(m["cover_curve"][d] for m in ms) for d in range(90)]
    out["all_covered_day0"] = sum(m["cover_curve"][0] >= 0.999 for m in ms)
    out["bridge"] = {k: st.mean(m["bridge"][k] for m in ms) for k in ms[0]["bridge"]}
    out["quarters"] = len(ms)
    return out


def test_tables():
    """Ours and the nine replayed frontier models on the test business: summaries and record metrics."""
    table, metrics = {}, {}
    for lab, dirs in CFG["test_arms"].items():
        rows, _ = analysis.quarters(dirs, CFG["runs"])
        if rows:
            table[lab] = analysis.summarize(rows)
            metrics[lab] = policy_metrics([Path(r["dir"]) / "engine" for r in rows if r["status"] == "valid"])
    fdir = analysis.resolve(CFG["test_frontier_dir"], CFG["runs"])
    models = {r["trial_id"]: r["model"] for r in json.loads((fdir / "hurdles.json").read_text())}
    by = defaultdict(list)
    for d in fdir.iterdir():
        if d.name in models and (d / "engine/grade.json").exists():
            by[NICE.get(models[d.name], models[d.name])].append(d / "engine")
    fr, _ = analysis.frontier_rows(fdir)
    for m, rows in fr.items():
        table[NICE.get(m, m)] = analysis.summarize(rows)
    for m, engines in by.items():
        metrics[m] = policy_metrics(engines)
    return table, metrics


# ---------------------------------------------------------------------------------------------- figures

def standings(ax, rows, rule, open_w, title):
    names = sorted(rows, key=lambda n: rows[n]["primary"])
    ys = range(len(names))
    ax.barh(ys, [rows[n]["primary"] / 1e3 for n in names], xerr=[rows[n]["se"] / 1e3 for n in names],
            color=[color_of(n, open_w) for n in names], error_kw={"ecolor": INK, "lw": 0.8, "capsize": 2})
    ax.set_yticks(list(ys), [short(n) + ("  (open weights)" if n in open_w else "") for n in names], fontsize=8.6)
    bold_ours(ax, names)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.axvline(rule["primary"] / 1e3, color=INK, lw=1, ls=":")
    ax.text(rule["primary"] / 1e3, len(names) - 0.45, " fixed buying rule", fontsize=8, color=INK)
    style(ax, title, "$ thousands per 90-day quarter, mean ± 1 SE over the three quarter types", "", grid="x")
    ranked = sorted(rows, key=lambda n: -rows[n]["primary"])
    return sum(1 for n in ranked[ranked.index(RL_ARM) + 1:] if "(ours)" not in n)


def fig_result(test, val, lb_test, rule_test, lb_val, rule_val, open_w, dump):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 7.6))
    t_rows = {**lb_test, RL_ARM: test[RL_ARM], BASE_ARM: test[BASE_ARM]}
    v_rows = {**lb_val, RL_ARM: val[RL_ARM], BASE_ARM: val[BASE_ARM]}
    ahead_t = standings(ax1, t_rows, rule_test, open_w, "")
    ahead_v = standings(ax2, v_rows, rule_val, open_w, "")
    ax1.set_title(f"Sealed test business. Ahead of {100 * ahead_t / len(lb_test):.0f}% of leaderboard models", loc="left",
                  fontsize=11, color=INK, fontweight="bold")
    ax2.set_title(f"Validation business. Ahead of {100 * ahead_v / len(lb_val):.0f}%", loc="left", fontsize=11, color=INK,
                  fontweight="bold")
    dump["test_rank"] = {"ahead_of": ahead_t, "of": len(lb_test), "fixed_rule": rule_test["primary"]}
    dump["val_rank"] = {"ahead_of": ahead_v, "of": len(lb_val), "fixed_rule": rule_val["primary"]}
    dump["val_ours"] = {n: val[n] for n in (RL_ARM, BASE_ARM)}
    fig.text(0.01, 0.012, "Each business is played as a normal, a customer and a supplier quarter. Our policies ran 4 of "
             "each. Leaderboard models are their scored runs, up to 3 of each (SMB Bench dataset revision "
             f"{CFG['leaderboard_revision'][:7]}).\nOur replays of nine of these models through our simulator reproduce "
             "their scores to the cent. A run of ours that ends early scores -$25K. Darker grey marks the two "
             "open-weight models.", fontsize=7.8, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "fig1_result.png", dpi=180)
    plt.close(fig)


def fig_behaviors(test, metrics, lb_test, open_w, dump):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.2))
    pols = [n for n in metrics if metrics[n]]

    def barh(ax, names, vals, fmt, title, xlabel):
        ax.barh(range(len(names)), vals, color=[color_of(n, open_w) for n in names])
        ax.set_yticks(range(len(names)), [short(n) for n in names], fontsize=8)
        bold_ours(ax, names)
        for y, v in enumerate(vals):
            ax.text(v, y, " " + fmt.format(v), va="center", fontsize=7.5, color=MUTED)
        style(ax, title, xlabel, "", grid="x")

    # 1. Every model under-buys
    ax = axes[0, 0]
    names = sorted(pols, key=lambda n: metrics[n]["supplier_t"] + metrics[n]["local_t"])
    sup = [metrics[n]["supplier_t"] for n in names]
    loc = [metrics[n]["local_t"] for n in names]
    ax.barh(range(len(names)), sup, color=[color_of(n, open_w) for n in names], label="from suppliers")
    ax.barh(range(len(names)), loc, left=sup, color="#d9a066", hatch="///", edgecolor="white",
            label="from the local market")
    ax.set_yticks(range(len(names)), [short(n) for n in names], fontsize=8)
    bold_ours(ax, names)
    for y, (s, l_) in enumerate(zip(sup, loc)):
        ax.text(s + l_, y, f" {s:.1f} + {l_:.1f}", va="center", fontsize=7.5, color=MUTED)
    style(ax, "1. Buy enough", "tonnes bought per quarter\n(solid from suppliers, hatched from the local market)", "",
          grid="x")

    # 2. They notice, and don't act
    names = sorted(pols, key=lambda n: metrics[n]["bad_week_reaction"])
    barh(axes[0, 1], names, [metrics[n]["bad_week_reaction"] for n in names], "{:+.2f}",
         "2. Act on a stockout", "days of stock added the week after a product\nwas out of stock 25% of the time or more")

    # 3. Protect every account from day zero
    ax = axes[0, 2]
    others = [n for n in pols if "(ours)" not in n and n != "GPT-6 Astra"]
    for n in others:
        ax.plot(range(90), [100 * v for v in metrics[n]["cover_curve"]], color=FRONT, lw=0.9, alpha=0.8)
    ax.plot([], [], color=FRONT, lw=0.9, label="other frontier models")
    ax.plot(range(90), [100 * v for v in metrics["GPT-6 Astra"]["cover_curve"]], color=INK, lw=1.8, label="GPT-6 Astra")
    ax.plot(range(90), [100 * v for v in metrics[BASE_ARM]["cover_curve"]], color=BASE, lw=1.8, ls="--",
            label="ours, no RL")
    ax.plot(range(90), [100 * v for v in metrics[RL_ARM]["cover_curve"]], color=OURS, lw=2.4, label="ours, + RL")
    ax.set_ylim(-3, 105)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    style(ax, "3. Protect every account from day zero", "day of quarter", "account products with shortfall cover (%)")

    # 4. Hold profit when customers or suppliers turn (all leaderboard models)
    ax = axes[1, 0]
    rows = {**lb_test, RL_ARM: test[RL_ARM], BASE_ARM: test[BASE_ARM]}
    off = {RL_ARM: (7, 4), BASE_ARM: (7, -10), "GPT-6 Astra": (6, -10), "Claude Fable 5.2": (-6, -10),
           "Gemini 4 Carbon": (6, 2), "Grok 4.7": (6, -6), "GLM 5.3": (6, 2), "DeepSeek V4.1 Flash": (6, 4),
           "Claude Opus 5.5": (6, -9)}
    for n, s in rows.items():
        pw = s["per_world"]
        drop = st.mean([pw["account_line"] - pw["null"], pw["supplier_deterioration"] - pw["null"]]) / 1e3
        ours = "(ours)" in n
        ax.scatter(pw["null"] / 1e3, drop, s=80 if ours else 36, color=color_of(n, open_w),
                   marker="D" if ours else "o", zorder=3)
        if ours or n in off:
            label(ax, n, pw["null"] / 1e3, drop, off)
        dump["adapt"][n] = {"normal": pw["null"], "mean_change": drop * 1e3}
    ax.axhline(0, color=MUTED, lw=0.8)
    style(ax, "4. Hold profit when customers or suppliers turn", "normal-quarter profit ($K)",
          "change in customer and supplier quarters\nvs the normal quarter ($K, higher is better)", grid="both")

    # 5. The expert stocks for the lead time plus a buffer
    ax = axes[1, 1]
    off = {RL_ARM: (8, 2), "GPT-6 Astra": (6, -10), "Claude Fable 5.1": (-6, 3),
           "GPT-6.1 Sol": (6, 2), "Grok 4.7": (6, -9), "Muse Wonder 1.0": (6, 2), "GLM 5.3": (-6, 2),
           "Claude Opus 5.5": (-6, 6), "Gemini 4 Argon": (6, -3), "DeepSeek V4.1 Flash": (-6, -11)}
    for n in pols:
        m = metrics[n]
        ours = "(ours)" in n
        ax.scatter(m["lead_days"], m["days_of_stock"], s=80 if ours else 36, color=color_of(n, open_w),
                   marker="D" if ours else "o", zorder=3)
        if n != BASE_ARM:
            label(ax, "Qwen3.8-27B, with and without RL (ours)" if n == RL_ARM else n, m["lead_days"],
                  m["days_of_stock"], {**off, "Qwen3.8-27B, with and without RL (ours)": off[RL_ARM]})
    lo, hi = 0.6, 1.8
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.9, ls="--")
    ax.text(hi - 0.02, hi - 0.1, "stock = lead time", fontsize=8, color=MUTED, ha="right")
    style(ax, "5. Stock for the lead time plus a buffer", "promised supplier lead time (days)",
          "stock on hand (days of sales)", grid="both")

    # 6. The expert consolidates orders on one supplier
    names = sorted(pols, key=lambda n: -metrics[n]["suppliers_used"])
    barh(axes[1, 2], names, [metrics[n]["suppliers_used"] for n in names], "{:.1f}",
         "6. Consolidate on one supplier", "suppliers with more than 2% of ordered tonnage")
    axes[1, 2].axvline(1, color=INK, lw=1, ls=":")
    axes[1, 2].text(1.04, len(names) - 0.4, "expert buyer", fontsize=8, color=INK)
    for n in pols:
        m = metrics[n]
        dump["behaviors"][n] = {k: m[k] for k in ("supplier_t", "local_t", "local_usd", "bad_week_reaction",
                                                  "bad_weeks", "all_covered_day0", "lead_days", "days_of_stock",
                                                  "suppliers_used", "quarters", "bridge")}
        dump["behaviors"][n]["cover_mean"] = st.mean(m["cover_curve"])
    fig.text(0.01, 0.008, "Test business, the same quarters and grader for every policy. Panels 1-3, 5 and 6 read the "
             "simulator records of our quarters and of the nine frontier models we replayed. Panel 4 uses all sixteen "
             "leaderboard models. Panel titles follow the SMB Bench report's findings.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(OUT / "fig2_behaviors.png", dpi=180)
    plt.close(fig)


def rollout_summaries(run):
    steps = []
    for it in sorted(analysis.resolve(run, CFG["runs"]).glob("iteration_*/train_rollout_summaries.jsonl")):
        steps.append([json.loads(line) for line in it.read_text().splitlines() if line.strip()])
    return steps


def fig_rl(test, dump):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.9), gridspec_kw={"width_ratios": [1.15, 0.9, 1.05]})

    # (a) the full final run, every rollout
    ax = axes[0]
    steps = rollout_summaries(CFG["rl_run"])
    means = []
    for k, rows in enumerate(steps):
        tm = [r["trajectory_metrics"] for r in rows]
        for j, t in enumerate(tm):
            x = k + (j - (len(tm) - 1) / 2) * 0.018
            if t.get("valid"):
                ax.scatter(x, t["margin_vs_frontier_kusd"], s=26, color=GOOD if t.get("beat_frontier") else FRONT,
                           zorder=3)
            else:
                ax.scatter(x, -3.4, s=34, color=BAD, marker="x", zorder=3)
        means.append(st.mean(t["margin_vs_frontier_kusd"] for t in tm if t.get("valid")))
        won = sum(t.get("beat_frontier", 0) for t in tm)
        ax.text(k, 3.6, f"{100 * won / len(tm):.0f}% beat", ha="center", fontsize=9,
                color=GOOD, fontweight="bold")
        dump["rl_reward"][k] = {"beat": won, "rollouts": len(tm), "mean_margin_kusd": means[-1],
                                "valid": sum(t.get("valid", 0) for t in tm)}
    ax.plot(range(len(means)), means, color=OURS, lw=2.4, marker="D", ms=7, zorder=4, label="mean margin")
    ax.axhline(0, color=INK, lw=1, ls=":")
    ax.text(-0.58, 0.1, "frontier's own result", fontsize=8, color=INK)
    ax.set_xlim(-0.6, 3.4)
    ax.set_ylim(-3.7, 4.2)
    ax.set_xticks(range(len(means)), [f"update {k}" for k in range(len(means))])  # 16 new runs per update
    style(ax, "Training reward over the full final run", "policy after k updates (16 new runs each)",
          "$K above the frontier model's result\nfrom the same morning (green = reward 1)")

    # (b) the same states under a dollar reward
    ax = axes[1]
    for (lab, run), color in zip(CFG["rl_runs"], (OURS, BASE)):
        done = [100 * sum(r["trajectory_metrics"].get("valid", 0) for r in rows) / len(rows)
                for rows in rollout_summaries(run)]
        ax.plot(range(len(done)), done, marker="o", color=color, lw=2, label=f"reward = {lab}")
        for x, v in zip(range(len(done)), done):
            ax.annotate(f"{v:.0f}%", (x, v), textcoords="offset points", xytext=(0, 7 if color == OURS else -14),
                        ha="center", fontsize=8, color=color)
        dump["rl_completed"][lab] = done
    ax.axvspan(3.5, 5.5, color=GRID, alpha=0.6, zorder=0)
    ax.text(4.5, 28, "runs start earlier\n(days 35-59)", ha="center", fontsize=8, color=MUTED)
    ax.set_ylim(0, 112)
    ax.set_xlim(-0.4, 5.5)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    style(ax, "A win-or-lose reward stays stable", "update", "runs that reached the end of the quarter (%)")

    # (c) held-out gains
    ax = axes[2]
    pairs = [(lab, *diff(summary(a), summary(b))) for lab, a, b in CFG["rl_pairs"]]
    pairs.append((CFG["test_pair_label"], *diff(test[RL_ARM], test[BASE_ARM])))
    z = sum(d / s for _, d, s in pairs) / math.sqrt(len(pairs))
    for i, (lab, d, s) in enumerate(pairs):
        ax.errorbar(d / 1e3, i, xerr=s / 1e3, fmt="o", color=OURS, ecolor=INK, capsize=3, ms=8)
        ax.annotate(f"{d / 1e3:+.1f}K", (d / 1e3, i), textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=8.5, color=INK)
        dump["rl_pairs"][lab] = {"diff": d, "se": s}
    dump["rl_pairs"]["stouffer_z"] = z
    ax.set_yticks(range(len(pairs)), [p[0] for p in pairs], fontsize=8.5)
    ax.invert_yaxis()
    ax.axvline(0, color=MUTED, lw=0.8)
    style(ax, "Held out. Gains in all four matched pairs", f"RL minus no RL, $K per quarter (± 1 SE), combined z = {z:.2f}",
          "", grid="x")
    fig.text(0.01, 0.012, "Each update samples 16 runs on the training businesses that continue archived frontier quarters "
             "from a morning between day 60 and 85, graded against what that frontier model earned from the same morning. "
             "Orange is the mean margin.\nRed crosses are runs ended by a model failure. Both rewards saw the same states "
             "in updates 0 to 3. The held-out pairs use the validation and sealed test businesses.", fontsize=7.8,
             color=MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "fig3_rl.png", dpi=180)
    plt.close(fig)


def fig_levers(test, metrics, dump):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.6), gridspec_kw={"width_ratios": [0.9, 1.05, 1.05]})

    # (a) data: imitating frontier transcripts vs no SFT, continuing frontier quarters from day 60
    ax = axes[0]
    labels, fail, beat = [], [], []
    for lab, rel in CFG["panel"]:
        rows = [json.loads(p.read_text()) for p in analysis.resolve(rel, CFG["runs"]).glob("*/branch-result.json")]
        labels.append(lab)
        fail.append(sum(r["status"] == "model_failure" for r in rows) / len(rows))
        beat.append(sum(r["status"] == "valid" and r["margin_cents"] > 0 for r in rows) / len(rows))
        dump["data_panel"][lab.replace("\n", " ")] = {"n": len(rows), "failed": fail[-1], "beat_frontier": beat[-1]}
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], [100 * f for f in fail], width=0.4, color=BAD, label="run ended by a model failure")
    ax.bar([i + 0.2 for i in x], [100 * b for b in beat], width=0.4, color=GOOD,
           label="beat the frontier from the same state")
    for i, (f, b) in enumerate(zip(fail, beat)):
        ax.text(i - 0.2, 100 * f + 1, f"{100 * f:.0f}%", ha="center", fontsize=8, color=INK)
        ax.text(i + 0.2, 100 * b + 1, f"{100 * b:.0f}%", ha="center", fontsize=8, color=INK)
    ax.set_xticks(list(x), labels, fontsize=8.5)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style(ax, "Data. Imitating frontier transcripts broke tool use", "", "% of 24 runs from frontier day-60 states")

    # (c) where the validation gain came from
    ax = axes[1]
    climb = [summary(dirs)["primary"] / 1e3 for _, dirs in CLIMB()]
    srcs = CFG["attribution"]
    colors = {"model": "#7a5ea8", "prompt": BASE, "environment": GOOD, "training": OURS}
    ax.bar(0, climb[0], color=FRONT)
    ax.text(0, -1.6, f"{climb[0]:+.1f}K", ha="center", fontsize=9, color=INK)
    for i, (lab, src) in enumerate(srcs, start=1):
        lo, hi = climb[i - 1], climb[i]
        ax.bar(i, hi - lo, bottom=lo, color=colors[src])
        ax.text(i, max(lo, hi) + 0.6, f"{hi - lo:+.1f}K", ha="center", fontsize=9, color=INK)
        dump["attribution"][lab.replace("\n", " ")] = {"source": src, "gain": (hi - lo) * 1e3}
    ax.bar(len(climb), climb[-1], color=OURS, alpha=0.35)
    ax.text(len(climb), climb[-1] + 0.6, f"{climb[-1]:+.1f}K", ha="center", fontsize=9, color=INK)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(climb) + 1), ["Qwen3.6-35B\nstart"] + [s[0].replace("\\n", "\n") for s in srcs] + ["final"],
                  fontsize=7.8)
    ax.set_ylim(climb[0] - 1.5, max(climb) + 3)
    for src, c in colors.items():
        ax.bar(0, 0, color=c, label={"model": "choice of base model", "prompt": "written instructions",
                                     "environment": "rules read from the training businesses",
                                     "training": "reinforcement learning in the simulator"}[src])
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style(ax, "Where the validation gain came from", "", "$ thousands per quarter, validation business")

    # (d) the remaining gap to GPT-6 Astra on the test business
    ax = axes[2]
    a, o = metrics["GPT-6 Astra"]["bridge"], metrics[RL_ARM]["bridge"]
    la, lo_ = metrics["GPT-6 Astra"]["local_usd"], metrics[RL_ARM]["local_usd"]
    parts = [("more sales", o["revenue"] - a["revenue"]),
             ("local-market\nbuying", -(lo_ - la)),
             ("supplier\ngoods", -((o["cogs"] - lo_) - (a["cogs"] - la))),
             ("dispatch fees\n(more suppliers)", -(o["dispatch"] - a["dispatch"])),
             ("waste", -(o["waste"] - a["waste"])),
             ("other", 0.0)]
    gap = o["return"] - a["return"]
    parts[-1] = ("other", gap - sum(v for _, v in parts[:-1]))
    if abs(parts[-1][1]) < 50:
        parts = parts[:-1]
    run = 0.0
    for i, (lab, v) in enumerate(parts):
        ax.bar(i, v / 1e3, bottom=run / 1e3, color=GOOD if v >= 0 else BAD)
        y = (run + v) / 1e3 + 0.2 if v >= 0 else (run + v) / 1e3 - 0.45
        ax.text(i, y, f"{v / 1e3:+.1f}K", ha="center", fontsize=9, color=INK)
        run += v
        dump["gap_to_astra"][lab.replace("\n", " ")] = v
    ax.bar(len(parts), gap / 1e3, color=OURS, alpha=0.35)
    ax.text(len(parts), gap / 1e3 - 0.45, f"{gap / 1e3:+.1f}K", ha="center", fontsize=9, color=INK)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylim(-8.6, 3.6)
    ax.set_xticks(range(len(parts) + 1), [p[0] for p in parts] + ["gap"], fontsize=8.3)
    style(ax, "The gap to GPT-6 Astra is a buying habit", "",
          "our profit minus Astra's, $K per quarter, test business")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_levers.png", dpi=180)
    plt.close(fig)


def main():
    global OUT, plt
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--runs-dir", default=None, help="base for relative paths (default: SMB_RUNS_DIR)")
    ap.add_argument("--out", required=True, help="figure directory")
    ap.add_argument("--json", help="also write every plotted number here")
    args = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    from smb_buyer import paths

    CFG.update(json.loads(Path(args.config).read_text()))
    CFG["leaderboard"] = str((Path(args.config).resolve().parent / CFG["leaderboard"]).resolve())
    CFG["leaderboard_revision"] = json.loads(Path(CFG["leaderboard"]).read_text())["revision"]
    CFG["runs"] = Path(args.runs_dir) if args.runs_dir else paths.runs_dir()
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    dump = defaultdict(dict)
    test, metrics = test_tables()
    lb_test, rule_test, open_w = leaderboard("test")
    lb_val, rule_val, _ = leaderboard("val")
    dump["test"] = {n: {k: s[k] for k in ("primary", "se", "per_world", "n")} for n, s in {**lb_test, **test}.items()}
    dump["fixed_rule"] = {"test": rule_test, "val": rule_val}
    val = {lab: summary(dirs) for lab, dirs in CFG["val_arms"].items()}
    fig_result(test, val, lb_test, rule_test, lb_val, rule_val, open_w, dump)
    fig_behaviors(test, metrics, lb_test, open_w, dump)
    fig_rl(test, dump)
    fig_levers(test, metrics, dump)
    if args.json:
        Path(args.json).write_text(json.dumps(dump, indent=1, default=str))
    print(OUT)


if __name__ == "__main__":
    main()
