"""Sealed TEST (b505) report: policy arms against exact replays of archived frontier runs.

Same metric as VAL selection (``smb_buyer/analysis.py``). Each arm is one or more
evaluation output directories; ``--arms-dir`` instead treats every subdirectory as an
arm (a resumed quarter supersedes its source wherever its resume output lives, so
pass the resume directory with its arm: ``--arm "base=test/base,test/resume-base"``).
Frontier rows are the archived b505 quarters replayed through this harness and grader
(``scripts/frontier_hurdles.py --final-test-access``).

    python scripts/report_test.py --arm "base=test/base" --arm "maxrl=test/maxrl" \\
        --frontier-dir frontier-test-reference --json test-report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from smb_buyer import analysis


def arms_from(args, runs):
    arms = analysis.parse_candidates(args.arm or [])
    if args.arms_dir:
        base = analysis.resolve(args.arms_dir, runs)
        for d in sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith("resume-")):
            dirs = [str(d)]
            if (d.parent / f"resume-{d.name}").is_dir():
                dirs.append(str(d.parent / f"resume-{d.name}"))
            arms.setdefault(d.name, dirs)
    return arms


def our_arms(arms, runs):
    return {name: analysis.quarters(dirs, runs) for name, dirs in arms.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", help="NAME=DIR[,DIR...] (repeatable)")
    ap.add_argument("--arms-dir", default=None, help="every subdirectory is an arm; resume-<name> joins <name>")
    ap.add_argument("--frontier-dir", required=True, help="frontier TEST replays (relative to --runs-dir or absolute)")
    ap.add_argument("--runs-dir", default=None, help="base for relative paths (default: SMB_RUNS_DIR)")
    ap.add_argument("--json")
    args = ap.parse_args()
    from smb_buyer import paths

    runs = Path(args.runs_dir) if args.runs_dir else paths.runs_dir()
    arms = arms_from(args, runs)
    if not arms:
        raise SystemExit("give --arm and/or --arms-dir")
    table, pending = {}, {}
    for name, (rows, pend) in our_arms(arms, runs).items():
        pending[name] = len(pend)
        s = analysis.summarize(rows) if rows else None
        if s:
            table[name] = s
    fr, missing = analysis.frontier_rows(analysis.resolve(args.frontier_dir, runs))
    for m, rows in fr.items():
        s = analysis.summarize(rows)
        if s:
            table[m] = s
    print(f"{'policy':28s} {'primary':>9s} {'±se':>7s}  {'null':>8s} {'acct':>8s} {'supp':>8s}  "
          "n(n/a/s)  surv  spoil  acct-short-kg  pending")
    for name, s in sorted(table.items(), key=lambda kv: -kv[1]["primary"]):
        pw, n = s["per_world"], s["n"]
        f = lambda x: "-" if x is None else f"{x:,.0f}"  # noqa: E731
        print(f"{name:28s} {s['primary']:9,.0f} {s['se']:7,.0f}  {pw['null']:8,.0f} {pw['account_line']:8,.0f} "
              f"{pw['supplier_deterioration']:8,.0f}  {n['null']}/{n['account_line']}/{n['supplier_deterioration']}     "
              f"{s['survival']:.2f}  {f(s['spoilage']):>5s}  {f(s['acct_short_kg']):>13s}  {pending.get(name, '')}")
        if s["failures"]:
            print(f"{'':28s} failures: {', '.join(s['failures'])}")
    print(f"\nfrontier replays without a grade: {missing}")
    ours = [k for k in table if k in pending]
    models = [k for k in table if k not in pending]
    for o in ours:
        print(f"\n{o} minus each frontier model (difference ± SE):")
        for m in sorted(models, key=lambda m: -table[m]["primary"]):
            d, se = analysis.diff(table[o], table[m])
            print(f"  {m:28s} {d:+9,.0f} ± {se:6,.0f}")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"table": table, "pending": pending, "frontier_missing_grades": missing}, indent=1))


if __name__ == "__main__":
    main()
