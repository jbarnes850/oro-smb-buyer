"""Snapshot the public SMB Bench leaderboard runs for the validation and test businesses.

The leaderboard's six businesses are played as a normal quarter (smb-bench-control-N) and two changed
quarters (smb-bench-public-*). Our TRAIN businesses are leaderboard businesses 1 to 4, validation is
business 5 and the sealed test is business 6. The task mapping below was established by matching the
nine frontier models we replayed through our simulator to the leaderboard's scored runs to the cent.

    uv run --with pandas --with pyarrow python scripts/leaderboard_snapshot.py \
        --out data/leaderboard-runs-b5-b6.json [--revision <dataset commit>]

Source: https://huggingface.co/datasets/oro-agents/smb-bench (data/runs.parquet), CC BY 4.0.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

REPO = "oro-agents/smb-bench"
TASKS = {
    "val": {"null": "smb-bench-control-5", "account_line": "smb-bench-public-9",
            "supplier_deterioration": "smb-bench-public-10"},
    "test": {"null": "smb-bench-control-6", "account_line": "smb-bench-public-11",
             "supplier_deterioration": "smb-bench-public-12"},
}
# Leaderboard models with public weights (the dataset lists them by their Hugging Face ids).
OPEN_WEIGHTS = ["deepseek-v4.1-flash-1017", "glm-5.3"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--revision", default=None, help="dataset commit (default: current main)")
    args = ap.parse_args()
    import io

    import pandas as pd

    if args.revision is None:
        with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPO}") as r:
            args.revision = json.load(r)["sha"]
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{args.revision}/data/runs.parquet"
    with urllib.request.urlopen(url) as r:
        runs = pd.read_parquet(io.BytesIO(r.read()))
    wanted = {t for split in TASKS.values() for t in split.values()}
    keep = runs[runs.public_task_id.isin(wanted)]
    rows = [{"model": x.model, "task": x.public_task_id, "repeat": int(x.repeat), "status": x.status,
             "business_return_cents": None if pd.isna(x.business_return_cents) else int(x.business_return_cents),
             "reference_return_cents": None if pd.isna(x.reference_return_cents) else int(x.reference_return_cents)}
            for x in keep.itertuples()]
    out = {"source": f"https://huggingface.co/datasets/{REPO}", "revision": args.revision, "license": "CC BY 4.0",
           "note": "Scored runs only count (status == 'valid'); reference_return_cents is the fixed buying rule.",
           "tasks": TASKS, "open_weights": OPEN_WEIGHTS, "runs": rows}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"{len(rows)} runs from {REPO}@{args.revision[:12]} -> {args.out}")


if __name__ == "__main__":
    main()
