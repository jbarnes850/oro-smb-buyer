"""VAL selection table and the declared selection rule (reads run artifacts only).

Each candidate is one or more evaluation output directories (``scripts/campaign.py
eval`` / ``resume``). Candidates are listed in order of simplicity: when several are
within one pooled SE of the best primary, the earliest listed wins. Frontier rows
are exact replays of archived VAL frontier quarters (``scripts/frontier_hurdles.py``).
See ``smb_buyer/analysis.py`` for the metric.

    python scripts/select_val.py \\
        --candidate "base=val/base" \\
        --candidate "maxrl=val/maxrl-final" \\
        --frontier-dir frontier-val-reference --json selection.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from smb_buyer import analysis


def fmt(x, digits=0):
    return "-" if x is None else f"{x:,.{digits}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", action="append", required=True, help="NAME=DIR[,DIR...] (repeat; simplest first)")
    ap.add_argument("--frontier-dir", default=None, help="frontier VAL replays (relative to --runs-dir or absolute)")
    ap.add_argument("--runs-dir", default=None, help="base for relative paths (default: SMB_RUNS_DIR)")
    ap.add_argument("--json", help="write the table to this file")
    args = ap.parse_args()
    from smb_buyer import paths

    runs = Path(args.runs_dir) if args.runs_dir else paths.runs_dir()
    candidates = analysis.parse_candidates(args.candidate)
    table, pend = {}, {}
    for name, dirs in candidates.items():
        rows, pending = analysis.quarters(dirs, runs)
        pend[name] = pending
        s = analysis.summarize(rows) if rows else None
        if s:
            table[name] = s
        elif rows:
            table[name] = {"incomplete": True, "n_done": len(rows)}
    print(f"{'candidate':24s} {'primary':>9s} {'±se':>7s}  {'null':>8s} {'acct':>8s} {'supp':>8s}  "
          "n(n/a/s)  surv  valid-mean  acct-short-kg  spoil  reserve  pending")
    for name, s in sorted(table.items(), key=lambda kv: -(kv[1].get("primary") or -1e9)):
        if s.get("incomplete"):
            print(f"{name:24s} incomplete ({s['n_done']} done; {len(pend[name])} pending)")
            continue
        pw, n = s["per_world"], s["n"]
        print(f"{name:24s} {s['primary']:9,.0f} {s['se']:7,.0f}  {pw['null']:8,.0f} {pw['account_line']:8,.0f} "
              f"{pw['supplier_deterioration']:8,.0f}  {n['null']}/{n['account_line']}/{n['supplier_deterioration']}"
              f"     {s['survival']:.2f}  {fmt(s['valid_mean']):>10s}  {fmt(s['acct_short_kg']):>13s}  "
              f"{fmt(s['spoilage']):>5s}  {fmt(s['reserve']):>7s}  {len(pend[name])}")
        if s["failures"]:
            print(f"{'':24s} failures: {', '.join(s['failures'])}")
    complete = {n: s for n, s in table.items() if "primary" in s and not pend[n]}
    selected, top, ties = analysis.select(complete, list(candidates))
    if selected:
        print(f"\ntop: {top}; near-ties within one pooled SE: {', '.join(ties)}; selected (simplest): {selected}")
    incomplete = [n for n in candidates if n not in complete]
    if incomplete:
        print(f"not selectable (missing worlds or pending quarters): {', '.join(incomplete)}")
    frontier = {}
    if args.frontier_dir:
        frontier = analysis.frontier_table(analysis.resolve(args.frontier_dir, runs))
        print("\nfrontier (VAL replays, mean over worlds of per-world means):")
        for m, s in sorted(frontier.items(), key=lambda kv: -kv[1]["primary"]):
            print(f"  {m:28s} {s['primary']:9,.0f}")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"candidates": table, "pending": pend, "selection": {"selected": selected, "top": top, "near_ties": ties},
             "frontier": {m: s["primary"] for m, s in frontier.items()}}, indent=1))


if __name__ == "__main__":
    main()
