"""Figures for the full-buyer campaign (reads run artifacts only; needs matplotlib).

  sft     RUN_DIR                         SFT train/held-out NLL curve (metrics.jsonl)
  rl      RUN_DIR [RUN_DIR ...]           RL arms on matched frontier states
  panel   PANEL_DIR                       branch-eval panel: margin vs the frontier's own return
  bridge  PANEL_DIR CONFIG [CONFIG ...]   where the branch margin comes from (grader value bridge)
  val     --candidate NAME=DIRS ...       VAL selection table and the frontier-gap mechanisms
  test    --arm NAME=DIRS ...             sealed TEST arms against frontier replays

Relative paths resolve against --runs-dir (default SMB_RUNS_DIR). Figures go to --out
(default $SMB_RUNS_DIR/figures).

    python scripts/plots.py rl rl/maxrl rl/cispo
    python scripts/plots.py val --candidate base=val/base --candidate maxrl=val/maxrl \\
        --frontier-dir frontier-val-reference
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from smb_buyer import analysis

INK, MUTED, A, B, C = "#1b2a38", "#6b7785", "#1f6fb2", "#c8691f", "#0f8a72"
plt = None


def setup_matplotlib():
    global plt
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    plt = pyplot


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e3e7ea", linewidth=0.8)


def save(fig, out, name, dpi=180):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, dpi=dpi)
    print(out / name)


def sft(args):
    run = analysis.resolve(args.run_dir, args.runs)
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    steps = [r["step"] for r in rows if "train_mean_nll" in r]
    nll = [r["train_mean_nll"] for r in rows if "train_mean_nll" in r]
    smooth = [sum(nll[max(0, i - 19) : i + 1]) / len(nll[max(0, i - 19) : i + 1]) for i in range(len(nll))]
    held = [(r["step"], r["heldout/nll"]) for r in rows if "heldout/nll" in r]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(steps, nll, color=A, alpha=0.18, linewidth=0.8, label="TRAIN batch NLL")
    ax.plot(steps, smooth, color=A, linewidth=2.2, label="TRAIN, 20-step mean")
    if held:
        ax.plot([h[0] for h in held], [h[1] for h in held], "o-", color=B, linewidth=2, label="held-out episodes")
        for s, v in (held[0], held[-1]):
            ax.annotate(f"{v:.3f}", (s, v), textcoords="offset points", xytext=(6, 6), color=B, fontsize=9)
    style(ax, "Economically selected full-buyer SFT (one pass)", "update", "per-target NLL (public actions)")
    ax.legend(frameon=False, fontsize=9)
    launch = run / "launch.json"
    if launch.exists():
        meta = json.loads(launch.read_text())
        totals = meta.get("data_manifest", {})
        caption = (f"{meta.get('model', '')} rank-32 LoRA from frozen base; "
                   f"{totals.get('train_segments', '?'):,} day-packed segments, "
                   f"{totals.get('train_targets', '?'):,} supervised targets; "
                   f"LR {meta.get('args', {}).get('learning_rate')} linear decay.")
        fig.text(0.01, 0.01, caption, fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, args.out, f"{run.name}-sft-curve.png")


def panel(args):
    """Branch panel: per configuration, margin vs the frontier's own return from the same state."""
    directory = analysis.resolve(args.panel_dir, args.runs)
    configs = sorted(p for p in directory.iterdir() if p.is_dir())
    finished = {c: len(list(c.glob("*/branch-result.json"))) for c in configs}
    for c in configs:
        if finished[c] < args.min_finished:
            print(f"(not plotted, fewer than {args.min_finished} finished) {c.name}: {finished[c]}")
    configs = [c for c in configs if finished[c] >= args.min_finished]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
    summary = []
    for i, config in enumerate(configs):
        rows = [json.loads(r.read_text()) for r in config.glob("*/branch-result.json")]
        valid = [r for r in rows if r["status"] == "valid"]
        margins = [r["margin_cents"] / 100 for r in valid]
        fails = sum(r["status"] == "model_failure" for r in rows)
        beat = sum(m > 0 for m in margins)
        mean = sum(margins) / len(margins) if margins else float("nan")
        summary.append((config.name, len(rows), len(valid), fails, beat, mean))
        jitter = [i + (k % 7 - 3) * 0.04 for k in range(len(margins))]
        ax1.scatter(jitter, margins, s=14, color=A, alpha=0.6)
        if margins:
            ax1.plot([i - 0.25, i + 0.25], [mean] * 2, color=INK, linewidth=2)
        ax2.bar(i - 0.2, len(valid) / max(1, len(rows)), width=0.4, color=C, label="completed" if i == 0 else None)
        ax2.bar(i + 0.2, beat / max(1, len(rows)), width=0.4, color=B, label="beat frontier" if i == 0 else None)
    for ax in (ax1, ax2):
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels([f"{c.name} (n={finished[c]})" for c in configs], rotation=20, ha="right", fontsize=8)
    ax1.axhline(0, color=MUTED, linewidth=1)
    style(ax1, "Margin vs the frontier from the same state", "", "graded return minus frontier's ($)")
    style(ax2, "Completion and beat rates", "", "share of branched runs")
    ax2.legend(frameon=False, fontsize=8)
    fig.text(0.01, 0.01, f"{directory.name}: each run continues an archived frontier TRAIN quarter from the same "
             "day-d state to day 90; original engine and grader.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, args.out, f"{directory.name}.png")
    for row in summary:
        print("%-22s runs %2d valid %2d model_fail %2d beat %2d mean_margin_valid %s" % (*row[:5], f"${row[5]:,.0f}"))


def bridge_features(grade_path):
    g = json.loads(Path(grade_path).read_text())
    b, a, f = g["bridge"], g["availability"]["total"], g["bridge"]["fees_by_kind"]
    return {
        "revenue": b["revenue"] / 100,
        "purchase cost": -b["cogs"] / 100,
        "freight (dispatch+delivery)": -(f.get("dispatch", 0) + f.get("delivery", 0)) / 100,
        "writedowns+spoilage": -(b["inspection_writedown"] + b["spoilage_writeoff"]) / 100,
        "other": None,  # filled as residual so the parts sum to the return difference
        "return": b["business_return_cents"] / 100,
        "lost_kg": a["lost_grams"] / 1000,
    }


def panel_bridge(args):
    """Where the margin vs the frontier comes from (same day-d state, ours minus frontier)."""
    directory = analysis.resolve(args.panel_dir, args.runs)
    hurdles = analysis.resolve(args.hurdles_dir, args.runs)
    replays = {}
    for f in hurdles.glob("*/frontier-replay-result.json"):
        r = json.loads(f.read_text())
        g = f.parent / "engine/grade.json"
        if g.exists():
            replays[(r.get("world"), r.get("return_cents"))] = g
    parts = ["revenue", "purchase cost", "freight (dispatch+delivery)", "writedowns+spoilage", "other"]
    rows = []
    for name in args.configs:
        deltas = []
        for f in (directory / name).glob("*/branch-result.json"):
            r = json.loads(f.read_text())
            key = (r["world"], r.get("hurdle_cents"))
            if r["status"] != "valid" or key not in replays:
                continue
            ours, theirs = bridge_features(f.parent / "engine/grade.json"), bridge_features(replays[key])
            d = {k: ours[k] - theirs[k] for k in ours if k != "other"}
            d["other"] = d["return"] - sum(d[k] for k in parts if k != "other")
            deltas.append(d)
        if deltas:
            rows.append((name, len(deltas), {k: sum(x[k] for x in deltas) / len(deltas) for k in deltas[0]}))
    if not rows:
        raise SystemExit("no valid continuations with a matching frontier replay")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    colors = [C, B, "#8a6fb0", "#b5485a", MUTED]
    width = 0.8 / len(parts)
    for i, (name, n, m) in enumerate(rows):
        for j, part in enumerate(parts):
            ax.bar(i - 0.4 + width * (j + 0.5), m[part], width=width * 0.95, color=colors[j],
                   label=part if i == 0 else None)
        ax.plot([i - 0.42, i + 0.42], [m["return"]] * 2, color=INK, linewidth=2.4,
                label="net (return difference)" if i == 0 else None)
        ax.annotate(f"net {m['return']:+,.0f}\nlost {m['lost_kg']:+,.0f} kg", (i, max(m[p] for p in parts)),
                    textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8, color=INK)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"{n}\n(n={k})" for n, k, _ in rows], fontsize=8)
    top = max(max(m[p] for p in parts) for _, _, m in rows)
    ax.set_ylim(top=top + 1600)
    style(ax, "Where the margin vs the frontier comes from", "", "ours minus frontier, $ per continuation")
    ax.legend(frameon=False, fontsize=8, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.text(0.01, 0.01, "Same archived frontier state; the remaining days played by each configuration; original "
             "grader's value bridge. Positive = better for us (costs shown as negative values).",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, args.out, f"{directory.name}-bridge.png")
    for name, n, m in rows:
        print(name, n, {k: round(v) for k, v in m.items()})


def rl(args):
    """RL arms on matched frontier states: each step draws the same states for every arm
    (deterministic per-step selection), so arms compare within a step; steps differ in
    state difficulty, so a step-to-step change is not by itself a learning curve."""
    import matplotlib.ticker

    run_dirs = [analysis.resolve(d, args.runs) for d in args.run_dirs]
    colors = [A, B, C, MUTED]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    width = 0.8 / len(run_dirs)
    backbones = set()
    for i, run in enumerate(run_dirs):
        launch = json.loads((run / "launch.json").read_text()) if (run / "launch.json").exists() else {}
        label = launch.get("arm", run.name)
        backbones.add(launch.get("backbone", "?"))
        steps = sorted(p for p in (run / "rollouts").glob("step*") if p.is_dir())
        xs, mean_margin, beat = [], [], []
        failures = {}
        offset = (i - (len(run_dirs) - 1) / 2) * width
        for p in steps:
            step = int(p.name[4:])
            # The trainer's own record of each group's results (matches its metrics);
            # per-quarter files can include abandoned retry attempts.
            rows = [r for f in sorted(p.glob("*/group-result-*.json")) for r in json.loads(f.read_text())]
            if len(rows) < args.min_rollouts:
                continue
            valid = [r for r in rows if r["status"] == "valid"]
            margins = [(r["return_cents"] - r["frontier"]["hurdle_cents"]) / 1e5 for r in valid]
            jitter = [step + offset + (k % 5 - 2) * 0.03 for k in range(len(margins))]
            axes[0, 0].scatter(jitter, margins, s=10, color=colors[i % 4], alpha=0.35, linewidths=0)
            xs.append(step)
            mean_margin.append(sum(margins) / len(margins) if margins else float("nan"))
            beat.append(sum(bool(r.get("beat_hurdle")) for r in valid) / len(rows))
            for r in rows:
                if r["status"] != "valid":
                    code = (r.get("failure") or {}).get("code", r["status"])
                    failures.setdefault(code, {}).setdefault(step, 0)
                    failures[code][step] += 1
        metrics_path = run / "metrics.jsonl"
        metrics = []
        if metrics_path.exists():
            metrics = [json.loads(line) for line in metrics_path.read_text().splitlines() if "cost/cumulative_usd" in line]
        cost = [(m["step"], m["cost/cumulative_usd"]) for m in metrics if m["step"] in xs]
        axes[0, 0].plot([x + offset for x in xs], mean_margin, marker="o", color=colors[i % 4], label=label)
        axes[0, 1].plot(xs, beat, marker="o", color=colors[i % 4], label=label)
        bottom = [0] * len(xs)
        for j, (code, by_step) in enumerate(sorted(failures.items())):
            heights = [by_step.get(x, 0) for x in xs]
            axes[1, 0].bar([x + offset for x in xs], heights, width=width * 0.9, bottom=bottom,
                           color=colors[i % 4], alpha=max(0.2, 1.0 - 0.25 * j), edgecolor="white",
                           label=f"{label}: {code}")
            bottom = [b + h for b, h in zip(bottom, heights)]
        axes[1, 1].plot([c[0] for c in cost], [c[1] for c in cost], marker="o", color=colors[i % 4], label=label)
    axes[0, 0].axhline(0, color=INK, linewidth=0.8)
    style(axes[0, 0], "Margin vs the frontier's own return (valid quarters)", "RL step", "$ thousands")
    style(axes[0, 1], "Share of a step's quarters that beat the frontier", "RL step", "fraction")
    style(axes[1, 0], "Quarters ended by a model failure", "RL step", "count")
    style(axes[1, 1], "Cumulative cost (official tracker)", "RL step", "USD")
    for ax in axes.flat:
        ax.legend(frameon=False, fontsize=8)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    fig.suptitle(f"Native RL ({', '.join(sorted(backbones))}) from frontier mid-quarter states "
                 "(same states per step for every arm)", x=0.01, ha="left", fontsize=11, color=MUTED)
    fig.tight_layout()
    save(fig, args.out, "rl-curves.png")


def val(args):
    """VAL selection and the two mechanisms behind the gap to the frontier."""
    candidates = analysis.parse_candidates(args.candidate)
    table, points = {}, {}
    for name, dirs in candidates.items():
        rows, _ = analysis.quarters(dirs, args.runs)
        s = analysis.summarize(rows) if rows else None
        if s:
            table[name], points[name] = s, [r for r in rows if r["status"] == "valid"]
    if not table:
        raise SystemExit("no candidate has quarters in all three worlds")
    frontier_dir = analysis.resolve(args.frontier_dir, args.runs)
    fr_by_model, _ = analysis.frontier_rows(frontier_dir)
    fr_rows = [r for rows in fr_by_model.values() for r in rows]
    fr = analysis.frontier_table(frontier_dir)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    names = sorted(table, key=lambda n: table[n]["primary"])
    ys = list(range(len(names)))
    rl_names = set(args.rl or [])
    colors = [B if n in rl_names else A for n in names]
    axes[0].barh(ys, [table[n]["primary"] / 1000 for n in names], xerr=[table[n]["se"] / 1000 for n in names],
                 color=colors, alpha=0.85, error_kw={"ecolor": INK, "lw": 0.8})
    axes[0].set_yticks(ys, [f"{n} (surv {table[n]['survival']:.0%})" for n in names], fontsize=8)
    marked = args.mark_model or [m for m, _ in sorted(fr.items(), key=lambda kv: -kv[1]["primary"])[:4]]
    for m in marked:
        if m in fr:
            v = fr[m]["primary"]
            axes[0].axvline(v / 1000, color=MUTED, linestyle="--", linewidth=0.8)
            axes[0].text(v / 1000, -0.45, " " + m, rotation=90, fontsize=7, color=MUTED, va="bottom", ha="right")
    if fr:
        med = st.median(s["primary"] for s in fr.values())
        axes[0].axvline(med / 1000, color=C, linewidth=1.2)
        axes[0].text(med / 1000, len(names) - 0.55, " frontier median", fontsize=7, color=C)
    style(axes[0], f"VAL return (failures = -${-analysis.FAILURE_USD / 1000:,.0f}K), mean ± SE", "$ thousands per quarter", "")
    acct = [r for r in fr_rows if r["world"] != "null"]
    axes[1].scatter([r["acct_short_kg"] for r in acct], [r["return_usd"] / 1000 for r in acct], s=14,
                    color="#b8c0c8", label="frontier models")
    ours = [r for n in points for r in points[n] if r["world"] != "null" and "acct_short_kg" in r]
    axes[1].scatter([r["acct_short_kg"] for r in ours], [r["return_usd"] / 1000 for r in ours], s=18, color=A,
                    label="candidates (all listed)")
    style(axes[1], "Account shortfall vs return (account worlds)", "account kg shorted over the quarter", "$ thousands")
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].scatter([r["spoilage"] / 1000 for r in fr_rows], [r["return_usd"] / 1000 for r in fr_rows], s=14,
                    color="#b8c0c8", label="frontier models")
    allours = [r for n in points for r in points[n] if "spoilage" in r]
    axes[2].scatter([r["spoilage"] / 1000 for r in allours], [r["return_usd"] / 1000 for r in allours], s=18,
                    color=A, label="candidates (all listed)")
    style(axes[2], "Spoilage write-off vs return (all worlds)", "$ thousands spoiled", "$ thousands")
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle("Full-buyer VAL; frontier = exact replays of the archived leaderboard quarters through this harness",
                 x=0.01, ha="left", fontsize=10, color=MUTED)
    fig.tight_layout()
    save(fig, args.out, "val-selection.png", dpi=170)


def test(args):
    """Sealed TEST (b505): matched arms against exact replays of every archived frontier run."""
    arms = analysis.parse_candidates(args.arm)
    table, quarters = {}, {}
    for name, dirs in arms.items():
        rows, _ = analysis.quarters(dirs, args.runs)
        s = analysis.summarize(rows) if rows else None
        if s:  # an arm missing a whole world has no primary yet
            table[name], quarters[name] = s, rows
    fr, _ = analysis.frontier_rows(analysis.resolve(args.frontier_dir, args.runs))
    for m, rows in fr.items():
        s = analysis.summarize(rows)
        if s:
            table[m], quarters[m] = s, rows
    ours = [n for n in table if n in arms]
    val_table = {}
    for spec in args.val or []:
        name, _, dirs = spec.partition("=")
        rows, _ = analysis.quarters(dirs.split(","), args.runs)
        s = analysis.summarize(rows) if rows else None
        if s:
            val_table[name] = s["primary"]
    names = sorted(table, key=lambda n: table[n]["primary"])
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1.15, 1]})
    ys = list(range(len(names)))
    colors = [(A if n == args.selected else B) if n in ours else "#b8c0c8" for n in names]
    axes[0].barh(ys, [table[n]["primary"] / 1000 for n in names], xerr=[table[n]["se"] / 1000 for n in names],
                 color=colors, error_kw={"ecolor": INK, "lw": 0.8})
    for y, n in zip(ys, names):
        v = val_table.get(n)
        if v is not None:
            axes[0].plot(v / 1000, y, marker="D", mfc="white", mec=INK, ms=6, zorder=3)
    if val_table:
        axes[0].plot([], [], marker="D", mfc="white", mec=INK, ls="", label="same arm on VAL")
        axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    labels = [f"{n}{' (selected)' if n == args.selected else ''} (surv {table[n]['survival']:.0%})" for n in names]
    axes[0].set_yticks(ys, labels, fontsize=8)
    axes[0].axvline(0, color=MUTED, linewidth=0.8)
    style(axes[0], f"TEST return (failures = -${-analysis.FAILURE_USD / 1000:,.0f}K), mean ± SE", "$ thousands per quarter", "")
    show = list(reversed(names))[:12]
    for i, n in enumerate(show):
        for j, w in enumerate(analysis.WORLDS):
            vals = [(r["return_usd"] if r["status"] == "valid" else analysis.FAILURE_USD) / 1000
                    for r in quarters[n] if r["world"] == w]
            axes[1].scatter([i + (j - 1) * 0.22] * len(vals), vals, s=16, color=(A, B, C)[j],
                            label=w if i == 0 else None, marker="o" if n not in ours else "D")
    axes[1].set_xticks(range(len(show)), show, rotation=60, ha="right", fontsize=7)
    axes[1].axhline(0, color=MUTED, linewidth=0.8)
    axes[1].legend(frameon=False, fontsize=8)
    style(axes[1], "Every TEST quarter by world (diamonds = ours)", "", "$ thousands")
    fig.suptitle("Sealed TEST (b505); frontier = exact replays of the archived b505 leaderboard quarters through "
                 "this harness", x=0.01, ha="left", fontsize=10, color=MUTED)
    fig.tight_layout()
    save(fig, args.out, "test-results.png", dpi=170)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default=None, help="base for relative paths (default: SMB_RUNS_DIR)")
    ap.add_argument("--out", default=None, help="figure directory (default: $SMB_RUNS_DIR/figures)")
    sub = ap.add_subparsers(dest="figure", required=True)
    p = sub.add_parser("sft")
    p.add_argument("run_dir")
    p = sub.add_parser("rl")
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--min-rollouts", type=int, default=8, help="skip steps with fewer group results")
    p = sub.add_parser("panel")
    p.add_argument("panel_dir")
    p.add_argument("--min-finished", type=int, default=10, help="skip configurations stopped early")
    p = sub.add_parser("bridge")
    p.add_argument("panel_dir")
    p.add_argument("configs", nargs="+")
    p.add_argument("--hurdles-dir", default="frontier-hurdles")
    p = sub.add_parser("val")
    p.add_argument("--candidate", action="append", required=True, help="NAME=DIR[,DIR...]")
    p.add_argument("--frontier-dir", required=True)
    p.add_argument("--rl", action="append", help="candidate names to color as RL arms")
    p.add_argument("--mark-model", action="append", help="frontier models to mark (default: top four)")
    p = sub.add_parser("test")
    p.add_argument("--arm", action="append", required=True, help="NAME=DIR[,DIR...]")
    p.add_argument("--frontier-dir", required=True)
    p.add_argument("--selected", default=None, help="arm selected on VAL (colored)")
    p.add_argument("--val", action="append", help="NAME=VAL_DIR[,DIR...]: draw that arm's VAL primary")
    args = ap.parse_args()
    from smb_buyer import paths

    args.runs = Path(args.runs_dir) if args.runs_dir else paths.runs_dir()
    args.out = Path(args.out) if args.out else args.runs / "figures"
    setup_matplotlib()
    {"sft": sft, "rl": rl, "panel": panel, "bridge": panel_bridge, "val": val, "test": test}[args.figure](args)


if __name__ == "__main__":
    main()
