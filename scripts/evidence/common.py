"""Shared helpers: quarter discovery and gzip-or-plain readers. Read-only over SMB_RUNS_DIR; writes to SMB_EVIDENCE_OUT."""
import csv
import glob
import gzip
import json
import os
from pathlib import Path

PKG = Path(__file__).resolve().parents[2]
# The campaign run tree (same layout as docs/story-config.json) and where outputs go.
RUNS = Path(os.environ.get("SMB_RUNS_DIR", PKG / "runs"))
OUT = Path(os.environ.get("SMB_EVIDENCE_OUT", PKG / "docs/evidence"))
WORLDS = ("null", "account_line", "supplier_deterioration")


def world_of(name):
    for w in WORLDS:
        if f"_{w}_1" in name:
            return w
    raise ValueError(name)


def opener(path):
    """Open path or path.gz as text (files may be gzipped in place)."""
    p = Path(path)
    if p.exists():
        return gzip.open(p, "rt") if p.suffix == ".gz" else open(p, "rt")
    gz = Path(str(p) + ".gz")
    if gz.exists():
        return gzip.open(gz, "rt")
    return None


def read_json(path):
    f = opener(path)
    if f is None:
        return None
    with f:
        return json.load(f)


def read_csv(path):
    f = opener(path)
    if f is None:
        return []
    with f:
        return list(csv.DictReader(f))


def iter_jsonl(path):
    f = opener(path)
    if f is None:
        return
    with f:
        for line in f:
            if line.strip():
                yield json.loads(line)


ARMS = {
    # arm label -> (split, glob of quarter dirs)
    "maxrl_final": ("test", "test/maxrl-001-final-v6b/s505*"),
    "base_v6": ("test", "test/q38-on-v6b/s505*"),
    "maxrl_final_val": ("val", "val/ref-maxrl-001-final-v6b/s504*"),
    "base_v6_val": ("val", "val/ref-q38-on-v6b/s504*"),
    "q36_base_v1_val": ("val", "warmstart/base-on/s504*"),
}


def frontier_models():
    idx = json.loads((RUNS / "frontier-test-reference/hurdles.json").read_text())
    return {r["trial_id"]: r["model"] for r in idx}


def quarters():
    """Yield (arm, split, source, quarter_dir) for every quarter we analyse."""
    out = []
    for arm, (split, pat) in ARMS.items():
        for d in sorted(glob.glob(str(RUNS / pat))):
            out.append((arm, split, "ours", d))
    fm = frontier_models()
    for d in sorted(glob.glob(str(RUNS / "frontier-test-reference/*/"))):
        d = d.rstrip("/")
        tid = os.path.basename(d)
        if tid in fm:
            out.append((fm[tid], "test", "frontier", d))
    return out
