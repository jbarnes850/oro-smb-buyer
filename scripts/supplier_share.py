"""TRAIN only: share of purchased cases going to the discount (lowest-price) supplier
before/after day 21 in graded frontier supplier_deterioration quarters, against their
replayed return.

Reads each TRAIN supplier_deterioration world's private truth (to name the discount
supplier) under --train-dir, the frontier index (compact transcripts) and the
frontier replays. Never run on VAL or TEST worlds.

    python scripts/supplier_share.py --train-dir WORLDS/train \\
        --frontier-index frontier-index.json --hurdles-dir runs/frontier-hurdles
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

TRAIN = ("s500", "s501", "s502", "s503")


def corr(x, y):
    mx, my = st.mean(x), st.mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return num / math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-dir", required=True, help="directory of TRAIN world directories")
    ap.add_argument("--frontier-index", default=None, help="default: SMB_FRONTIER_INDEX")
    ap.add_argument("--hurdles-dir", default=None, help="default: $SMB_RUNS_DIR/frontier-hurdles")
    ap.add_argument("--switch-day", type=int, default=21)
    args = ap.parse_args()
    from smb_buyer import campaign, paths

    worlds = Path(args.train_dir)
    disc = {}
    for w in TRAIN:
        truth = json.loads((worlds / f"{w}_experience_long_supplier_deterioration_1/private/truth.json").read_text())
        disc["b" + w[1:]] = min(truth["suppliers_private"], key=lambda s: s["price_multiplier"])["supplier_id"]
    index = campaign.read_frontier_index(args.frontier_index)
    hurdles_dir = Path(args.hurdles_dir) if args.hurdles_dir else paths.frontier_hurdles()
    hurd = {}
    for f in glob.glob(str(hurdles_dir / "*/frontier-replay-result.json")):
        hurd[Path(f).parent.name] = json.loads(Path(f).read_text()).get("return_cents")
    rows = []
    for row in index:
        if "supplier_deterioration" not in row["task_id"] or row["task_id"][3:7] not in disc:
            continue
        b = row["task_id"][3:7]
        d = disc[b]
        pre, post = defaultdict(float), defaultdict(float)
        transcript = row.get("transcript") or row["path"]
        if not str(transcript).endswith(".jsonl.gz"):
            continue  # needs the compact transcript records
        with gzip.open(transcript, "rt") as stream:
            for line in stream:
                r = json.loads(line)
                for tc in r.get("tool_calls") or []:
                    if tc["name"] != "place_purchase_order":
                        continue
                    a = tc["arguments"] if isinstance(tc["arguments"], dict) else json.loads(tc["arguments"])
                    tgt = pre if r["day"] < args.switch_day else post
                    to_discount = a.get("offer_id", "").split("-")[2:3] == [d]
                    tgt["disc" if to_discount else "other"] += float(a.get("cases") or a.get("quantity_cases") or 1)

        def share(x):
            return x["disc"] / (x["disc"] + x["other"]) if x["disc"] + x["other"] else None

        ret = hurd.get(row["trial_id"])
        rows.append((row["model"], b, share(pre), share(post), ret / 100 if ret is not None else None))
    by = defaultdict(list)
    for m, b, p, q, r in rows:
        by[m].append((p, q, r))

    def mean_of(v, i):
        vals = [x[i] for x in v if x[i] is not None]
        return st.mean(vals) if vals else float("nan")

    for m, v in sorted(by.items(), key=lambda kv: -st.mean([x[2] for x in kv[1] if x[2] is not None] or [0])):
        print(f"{m:26s} n{len(v):2d} disc-share pre {mean_of(v, 0):.2f} post {mean_of(v, 1):.2f}  "
              f"return {mean_of(v, 2):9,.0f}")
    ok = [(p, q, r) for _, _, p, q, r in rows if p is not None and q is not None and r is not None]
    if len(ok) > 2:
        print("n", len(ok),
              "corr(post share, return)", round(corr([q for _, q, _ in ok], [r for *_, r in ok]), 2),
              "corr(post-pre, return)", round(corr([q - p for p, q, _ in ok], [r for *_, r in ok]), 2))


if __name__ == "__main__":
    main()
