"""Stop guard for a running full-buyer RL run (stop conditions declared before launch).

Polls the run directory; after each completed step checks
  - model-failure rate per rollout-day of that step (stop if > 1/--min-days-per-failure with at
    least --min-failures failures; exposure = days each rollout ran after its branch day, so
    longer continuations are not penalised for their length),
  - mean output tokens per model call vs step 0 (stop if > --max-token-ratio x),
  - cumulative official training cost from metrics.jsonl (stop if > --max-usd),
  - free disk under --disk-path (stop if < --min-free-gib).
On a violation it sends SIGTERM to the driver, writes <run>/guard-stop.json and exits 3.
It exits 0 when the driver exits on its own. Standard library only.

    python scripts/rl_guard.py RUN_DIR PID [--min-days-per-failure 120] ...
"""
import argparse
import glob
import json
import os
import shutil
import signal
import time
from pathlib import Path


def step_rows(run, step):
    rows = []
    for f in glob.glob(str(run / "rollouts" / f"step{step:03d}" / "*" / "group-result-*.json")):
        rows.extend(json.loads(Path(f).read_text()))
    return rows


def exposure_days(r):
    if r["status"] == "valid":
        return 90 - r["branch_day"]
    return max(1, ((r.get("failure") or {}).get("last_day") or r["branch_day"]) - r["branch_day"])


def tokens_per_call(rows):
    calls = sum(r.get("model_calls") or 0 for r in rows)
    return sum(r.get("output_tokens") or 0 for r in rows) / calls if calls else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="RL --log-path directory")
    ap.add_argument("pid", type=int, help="process id of the running scripts/rl.py")
    ap.add_argument("--min-days-per-failure", type=float, default=120.0)
    ap.add_argument("--min-failures", type=int, default=3)
    ap.add_argument("--max-token-ratio", type=float, default=1.5)
    ap.add_argument("--max-usd", type=float, default=2800.0)
    ap.add_argument("--min-free-gib", type=float, default=6.0)
    ap.add_argument("--disk-path", default="/", help="filesystem whose free space is checked")
    ap.add_argument("--poll", type=int, default=120)
    args = ap.parse_args()
    run = Path(args.run)
    checked, base_tpc = set(), None

    def stop(reason, detail):
        record = {"at": time.time(), "reason": reason, "detail": detail, "pid": args.pid}
        (run / "guard-stop.json").write_text(json.dumps(record, indent=1))
        print("GUARD STOP", json.dumps(record), flush=True)
        try:
            os.kill(args.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        raise SystemExit(3)

    while True:
        try:
            os.kill(args.pid, 0)
        except ProcessLookupError:
            print("driver exited", flush=True)
            return
        free = shutil.disk_usage(args.disk_path).free / 1024**3
        if free < args.min_free_gib:
            stop("disk", {"free_gib": round(free, 2)})
        metrics = run / "metrics.jsonl"
        rows = [json.loads(line) for line in metrics.read_text().splitlines()] if metrics.exists() else []
        cost = max((r.get("cost/cumulative_usd", 0) for r in rows), default=0)
        if cost > args.max_usd:
            stop("cost", {"cumulative_usd": cost})
        done_steps = sorted({r["step"] for r in rows if "env/all/valid" in r})
        for step in done_steps:
            if step in checked:
                continue
            res = step_rows(run, step)
            valid = sum(r["status"] == "valid" for r in res) / len(res) if res else None
            fails = sum(r["status"] != "valid" for r in res)
            exposure = sum(exposure_days(r) for r in res)
            tpc = tokens_per_call(res)
            if step == 0 or base_tpc is None:
                base_tpc = base_tpc or tpc
            info = {"step": step, "rollouts": len(res), "valid": valid, "failures": fails,
                    "exposure_days": exposure, "tokens_per_call": tpc,
                    "step0_tokens_per_call": base_tpc, "cumulative_usd": cost, "free_gib": round(free, 2)}
            print("step check", json.dumps(info), flush=True)
            checked.add(step)
            if fails >= args.min_failures and exposure / fails < args.min_days_per_failure:
                stop("failure_rate", info)
            if tpc and base_tpc and tpc > args.max_token_ratio * base_tpc:
                stop("output_tokens_per_call", info)
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
