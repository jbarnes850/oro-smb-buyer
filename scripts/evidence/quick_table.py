"""Quick per-arm summary of records_metrics.json (TEST unless --val)."""
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT  # noqa: E402
rows = json.loads((OUT / "records_metrics.json").read_text())
keys = sys.argv[1].split(",") if len(sys.argv) > 1 else ["return_usd"]
world = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "all" else None
split = sys.argv[3] if len(sys.argv) > 3 else "test"
by = defaultdict(list)
for r in rows:
    if r["status"] != "valid" or r["split"] != split:
        continue
    if world and r["world"] != world:
        continue
    by[r["arm"]].append(r)


def m(rs, k):
    v = [r.get(k) for r in rs if isinstance(r.get(k), (int, float))]
    return st.mean(v) if v else float("nan")


order = sorted(by, key=lambda a: -m(by[a], "return_usd"))
print(f"{'arm':26s} n  " + " ".join(f"{k[:14]:>14s}" for k in keys))
for a in order:
    print(f"{a:26s} {len(by[a]):2d} " + " ".join(f"{m(by[a], k):14.3f}" for k in keys))
