"""Spend estimate at official list prices (client-side; not a provider invoice).

Training runs (SFT/RL): the official CostTracker's cumulative USD in each run's
metrics.jsonl, which already includes RL rollout sampling and training. Every other
quarter (evaluations, branch panels, in-session VAL): priced here from its recorded
token counts at the list rates of the backbone named in its plan.json
(``smb_buyer.backbones.PRICES``). RL rollout quarters are not double counted.
The ``recorded`` column is the ``sampling_usd`` each quarter wrote at run time.

    python scripts/costs.py [--runs-dir DIR] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from smb_buyer.backbones import PRICES, backbone_of_model


def quarter_usd(result, backbone):
    rates = PRICES[backbone]
    prompt = result.get("prompt_tokens") or 0
    hits = result.get("prompt_cache_hit_tokens") or 0
    out = result.get("output_tokens") or 0
    return ((prompt - hits) * rates["prefill"] + hits * rates["cached_prefill"] + out * rates["sample"]) / 1e6


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default=None, help="default: SMB_RUNS_DIR")
    ap.add_argument("--json")
    args = ap.parse_args()
    from smb_buyer import paths

    runs = Path(args.runs_dir) if args.runs_dir else paths.runs_dir()
    rows = defaultdict(float)
    recorded = defaultdict(float)
    for metrics in sorted(runs.rglob("metrics.jsonl")):
        lines = metrics.read_text().splitlines()
        costs = [json.loads(line).get("cost/cumulative_usd") for line in lines if "cost/cumulative_usd" in line]
        if costs:
            key = f"train-run {metrics.parent.relative_to(runs)}"
            rows[key] += costs[-1]
            recorded[key] += costs[-1]
    unpriced = 0
    for result in sorted(runs.rglob("quarter-result.json")):
        rel = result.relative_to(runs)
        if "rollouts" in rel.parts:
            continue  # inside RL runs: counted by the official tracker
        r = json.loads(result.read_text())
        if str(r.get("policy_id", "")).startswith("scripted"):
            continue  # no model calls
        plan = result.parent / "plan.json"
        model = (json.loads(plan.read_text()).get("native") or {}).get("model") if plan.exists() else None
        backbone = backbone_of_model(model) if model else None
        key = f"quarters {rel.parts[0]}" + (f" [{model.split('/')[1].split(':')[0]}]" if model else "")
        recorded[key] += r.get("sampling_usd") or 0.0
        if backbone in PRICES:
            rows[key] += quarter_usd(r, backbone)
        else:
            unpriced += 1
            rows[key] += r.get("sampling_usd") or 0.0
    total = rec_total = 0.0
    print(f"{'list':>10s}  {'recorded':>10s}  item")
    for key in sorted(rows):
        print(f"{rows[key]:10.2f}  {recorded[key]:10.2f}  {key}")
        total += rows[key]
        rec_total += recorded[key]
    print(f"{total:10.2f}  {rec_total:10.2f}  TOTAL (excludes in-flight requests)")
    if unpriced:
        print(f"{unpriced} quarters without a known backbone kept their recorded estimate")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"total_usd": round(total, 2), "recorded_total_usd": round(rec_total, 2),
             "items": rows, "recorded_items": recorded}, indent=1))


if __name__ == "__main__":
    main()
