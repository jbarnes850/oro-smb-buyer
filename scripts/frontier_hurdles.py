"""Replay archived frontier quarters through this harness to get their graded returns.

Zero model calls: every recorded frontier response is served back to the original
engine in (day, turn) order and the quarter is graded by the original grader. The
results are the hurdles for frontier-state branching (RL, branch-eval) and the
frontier reference rows for VAL selection and the TEST report.

Input: a JSON list (or ``{"archives": [...]}``) of rows with ``trial_id``, ``model``,
``task_id`` (``r2-b50N-<mechanism>``), ``reward`` and ``transcript`` (or ``path``): a
compact ``*.jsonl.gz`` of transcript records or the archived run ``*.tar.gz``.
Output: ``<out>/<trial_id>/frontier-replay-result.json`` per row, plus
``<out>/hurdles.json`` with every row's result.

    python scripts/frontier_hurdles.py INDEX.json --out runs/frontier-hurdles --workers 8
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import functools
import json
import sys
from pathlib import Path

KEYS = ("trial_id", "model", "task_id", "reward")


def one(row, *, out, final_test_access):
    from smb_buyer import campaign

    target = out / row["trial_id"]
    out.mkdir(parents=True, exist_ok=True)
    meta = {k: row[k] for k in KEYS}
    if (target / "frontier-replay-result.json").exists():
        return json.loads((target / "frontier-replay-result.json").read_text()) | meta
    try:
        result = asyncio.run(
            campaign.frontier_quarter_return(
                campaign.world_of_task(row["task_id"]),
                row.get("transcript") or row["path"],
                target,
                final_test_access=final_test_access,
            )
        )
    except Exception as error:  # noqa: BLE001 -- recorded per row, never a hurdle
        return {"error": repr(error)[:300], **meta}
    return result | meta


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("index", help="JSON list of archived frontier quarters")
    parser.add_argument("--out", required=True, help="output directory, e.g. $SMB_RUNS_DIR/frontier-hurdles")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="replay only the first N rows")
    parser.add_argument("--final-test-access", action="store_true", help="required for sealed TEST (b505) rows")
    parser.add_argument("--registry", default=None, help="world registry (sets SMB_WORLD_REGISTRY)")
    args = parser.parse_args()
    if args.registry:
        import os

        os.environ["SMB_WORLD_REGISTRY"] = str(Path(args.registry).resolve())
    loaded = json.loads(Path(args.index).read_text())
    rows = loaded["archives"] if isinstance(loaded, dict) else loaded
    if args.limit is not None:
        rows = rows[: args.limit]
    out = Path(args.out).resolve()
    work = functools.partial(one, out=out, final_test_access=args.final_test_access)
    with concurrent.futures.ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(work, rows))
    out.mkdir(parents=True, exist_ok=True)
    summary = out / "hurdles.json"
    merged = {r["trial_id"]: r for r in (json.loads(summary.read_text()) if summary.exists() else [])}
    merged.update({r["trial_id"]: r for r in results})  # earlier rows of other trials are kept
    summary.write_text(json.dumps(list(merged.values()), indent=1))
    for r in results:
        print(r.get("model"), r.get("task_id"), r.get("reward"), r.get("return_cents"), r.get("error", ""))
    if any("error" in r for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
