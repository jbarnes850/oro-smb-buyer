"""Shared tables for VAL selection, the TEST report and the figures (artifacts only).

Primary metric: mean over the three mechanism worlds of the per-world mean graded
business return, scoring each model-failure quarter at FAILURE_USD (the declared RL
failure reward). Infrastructure failures are excluded. A quarter continued by
``scripts/campaign.py resume`` supersedes its source quarter: the resume output
directory's ``resume-plan-*.json`` names the source of each resumed quarter.
Standard library only.
"""

from __future__ import annotations

import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

FAILURE_USD = -25_000.0
WORLDS = ("null", "account_line", "supplier_deterioration")


def mechanism_of(world_key):
    for w in WORLDS:
        if f"_{w}_1" in world_key:
            return w
    raise ValueError(world_key)


def resolve(path, runs_dir):
    path = Path(path)
    return path if path.is_absolute() else Path(runs_dir) / path


def mechanisms(grade_path):
    g = json.loads(Path(grade_path).read_text())
    a = g["availability"]["total"]
    b = g["bridge"]
    return {
        "acct_short_kg": (a["account_ordered_grams"] - a["account_sold_grams"]) / 1000,
        "acct_ordered_t": a["account_ordered_grams"] / 1e6,
        "spoilage": b["spoilage_writeoff"] / 100,
        "reserve": g["value"]["close_components"].get("customer_demand_reserve", 0) / 100,
        "revenue": b["revenue"] / 100,
    }


def superseded_sources(directory):
    """Source quarter directories replaced by a resumed quarter that exists in ``directory``."""
    out = set()
    for plan in Path(directory).glob("resume-plan-*.json"):
        for row in json.loads(plan.read_text())["plan"]:
            if row.get("resumed") and Path(row["output"]).is_dir():
                out.add(Path(row["source"]).resolve())
    return out


def quarters(dirs, runs_dir="."):
    """One outcome per quarter across ``dirs``; returns (rows, pending quarter dirs)."""
    dirs = [resolve(d, runs_dir).resolve() for d in dirs]
    replaced = set().union(*(superseded_sources(d) for d in dirs)) if dirs else set()
    rows, pending = [], []
    for directory in dirs:
        for quarter in sorted(p.parent for p in directory.glob("*/plan.json")):
            if quarter.resolve() in replaced:
                continue
            result = quarter / "quarter-result.json"
            if not result.exists():
                pending.append(str(quarter))
                continue
            r = json.loads(result.read_text())
            if r["status"] not in ("valid", "model_failure"):
                continue  # infrastructure failure: not a policy outcome
            row = {
                "dir": str(quarter),
                "world": mechanism_of(r["world"]),
                "status": r["status"],
                "return_usd": r["return_cents"] / 100 if r["status"] == "valid" else None,
                "failure": (r.get("failure") or {}).get("code"),
                "failure_day": (r.get("failure") or {}).get("last_day"),
            }
            if r["status"] == "valid" and (quarter / "engine/grade.json").exists():
                row.update(mechanisms(quarter / "engine/grade.json"))
            rows.append(row)
    return rows, pending


def summarize(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["world"]].append(r["return_usd"] if r["status"] == "valid" else FAILURE_USD)
    if not all(by.get(w) for w in WORLDS):
        return None
    means = {w: st.mean(by[w]) for w in WORLDS}
    var = sum((st.variance(by[w]) if len(by[w]) > 1 else 0.0) / len(by[w]) for w in WORLDS)
    valid = [r for r in rows if r["status"] == "valid"]
    out = {
        "primary": st.mean(means.values()),
        "se": math.sqrt(var) / len(WORLDS),
        "per_world": means,
        "n": {w: len(by[w]) for w in WORLDS},
        "survival": len(valid) / len(rows),
        "valid_mean": st.mean(r["return_usd"] for r in valid) if valid else None,
        "failures": [f"{r['world']}:{r['failure']}@d{r['failure_day']}" for r in rows if r["status"] != "valid"],
    }
    for k in ("acct_short_kg", "spoilage", "reserve", "acct_ordered_t"):
        if k == "spoilage":
            vals = [r[k] for r in valid if k in r]
        else:  # account quantities only mean something in the mechanism worlds
            vals = [r[k] for r in valid if k in r and r["world"] != "null"]
        out[k] = st.mean(vals) if vals else None
    return out


def frontier_rows(frontier_dir):
    """Graded frontier replay rows per model from ``scripts/frontier_hurdles.py`` output.

    Returns ({model: [row, ...]}, number of replays without a grade).
    """
    frontier_dir = Path(frontier_dir)
    models = {}
    summary = frontier_dir / "hurdles.json"
    if summary.exists():
        models = {r.get("trial_id"): r.get("model") for r in json.loads(summary.read_text())}
    rows, missing = defaultdict(list), 0
    for f in sorted(frontier_dir.glob("*/frontier-replay-result.json")):
        grade = f.parent / "engine/grade.json"
        if not grade.exists():
            missing += 1
            continue
        r = json.loads(f.read_text())
        row = {
            "world": mechanism_of(r["world"]),
            "status": "valid",
            "return_usd": json.loads(grade.read_text())["bridge"]["business_return_cents"] / 100,
        }
        row.update(mechanisms(grade))
        rows[models.get(f.parent.name, "?")].append(row)
    return dict(rows), missing


def frontier_table(frontier_dir):
    rows, _ = frontier_rows(frontier_dir)
    return {m: s for m, s in ((m, summarize(r)) for m, r in rows.items()) if s}


def pooled_se(a, b):
    return math.sqrt(a["se"] ** 2 + b["se"] ** 2)


def diff(a, b):
    """Difference of primaries and its SE (independent arms)."""
    return a["primary"] - b["primary"], pooled_se(a, b)


def select(table, order):
    """Declared rule: the top candidate has the highest primary; a candidate within one
    pooled SE of the top is a near-tie; the simplest near-tie (earliest in ``order``) wins.
    Returns (selected, top, near_ties)."""
    ranked = [n for n in order if n in table and "primary" in table[n]]
    if not ranked:
        return None, None, []
    top = max(ranked, key=lambda n: table[n]["primary"])
    ties = [n for n in ranked if table[top]["primary"] - table[n]["primary"] <= pooled_se(table[top], table[n])]
    return ties[0], top, ties


def parse_candidates(specs):
    """``NAME=DIR[,DIR...]`` -> ordered {name: [dirs]}."""
    out = {}
    for spec in specs:
        name, _, dirs = spec.partition("=")
        if not name or not dirs:
            raise SystemExit(f"candidate must be NAME=DIR[,DIR...], got {spec!r}")
        out[name] = dirs.split(",")
    return out
