"""Per-quarter behaviour metrics from engine/model_transcript.jsonl(.gz) and model_notebook.txt.

Our arms have raw reasoning_text; the frontier TEST replays have tool calls/results only (their
reasoning is not in the replay), so reasoning metrics are ours-only while purchase-rationale and
tool-error metrics use the same code path for everyone.

    python3 -I mine_transcripts.py  -> transcript_metrics.json, quote_candidates.jsonl
"""
import json
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, opener, quarters  # noqa: E402

STOCKOUT = re.compile(r"\bOOS\b|hours?[_ ]out|out[ -]of[ -]stock|stock[- ]?outs?\b|stocked out|sold out|ran out", re.I)
INFER = re.compile(
    r"censor|true demand|real demand|underlying demand|latent demand|unconstrained|lost (?:sales|demand|walk)"
    r"|oos[- ]adj|adjusted demand|demand[- ]adjust|16\s*/\s*\(?\s*16\s*[-−]|×\s*\(?\s*16\s*/|\*\s*16\s*/|x\s*16\s*/", re.I)
LOCAL = re.compile(r"local[_ ]?(?:market|buy|purchase|price|premium|cover)|shortfall[_ ]cover|\bcover\b", re.I)
LOCAL_ECON = re.compile(r"(?:local|cover)[^.\n]{0,120}(?:premium|expensive|loss|lose|losing|margin|shelf|backstop|accept|cheaper|cost)", re.I)
SUPPLIER_BAD = r"(?:late|short|unreliab|deteriorat|cancel|missing|delay|no[- ]show|didn'?t arrive|not arrive)"
ENDQ = re.compile(r"final (?:day|buy|morning|purchas)|last (?:buy|purchas|day)|valuation|deadweight|end of (?:the )?(?:run|quarter)", re.I)
CAP = "I have reasoned for a long time"
DISC = {"test": "sup_c", "val": "sup_b"}


def sentences(text):
    for para in re.split(r"\n+", text):
        for s in re.split(r"(?<=[.!?])\s+", para):
            s = s.strip()
            if 8 <= len(s.split()) <= 70:
                yield s


def mine(args):
    arm, split, source, d = args
    d = Path(d)
    eng = d / "engine"
    disc = DISC[split]
    sup_re = re.compile(rf"{disc}[^.\n]{{0,120}}{SUPPLIER_BAD}|{SUPPLIER_BAD}[^.\n]{{0,120}}{disc}", re.I)
    row = {"arm": arm, "split": split, "source": source, "dir": str(d)}
    day_calls = Counter()
    day_reason_chars = Counter()
    day_out_tokens = Counter()
    day_tools = Counter()
    day_pos = Counter()
    day_flags = defaultdict(set)  # flag -> days
    finish = Counter()
    cap_hits = 0
    tool_names = Counter()
    errs = Counter()
    consecutive_max = 0
    consec = 0
    rationales = []
    quotes = []
    mem_bytes = {}
    mem_calls = 0
    n_model = 0
    reason_chars = []
    f = opener(eng / "model_transcript.jsonl")
    if f is None:
        return row, []
    with f:
        for line in f:
            if '"kind": "model_request"' in line[:120] or '"kind": "id_normalized"' in line[:120]:
                continue
            r = json.loads(line)
            k = r.get("kind")
            day = r.get("day")
            if k == "model":
                n_model += 1
                day_calls[day] += 1
                t = r.get("reasoning_text") or ""
                reason_chars.append(len(t))
                day_reason_chars[day] += len(t)
                day_out_tokens[day] += (r.get("usage") or {}).get("output_tokens") or 0
                finish[r.get("finish_reason")] += 1
                if CAP in t:
                    cap_hits += 1
                if t:
                    for name, pat in (("stockout", STOCKOUT), ("infer", INFER), ("local", LOCAL),
                                      ("local_econ", LOCAL_ECON), ("supplier_bad", sup_re), ("endq", ENDQ)):
                        if pat.search(t):
                            day_flags[name].add(day)
                    # quote candidates (ours only, have reasoning)
                    for s in sentences(t):
                        cat = None
                        if LOCAL_ECON.search(s):
                            cat = "local"
                        elif INFER.search(s):
                            cat = "infer"
                        elif sup_re.search(s) and day >= 21:
                            cat = "supplier"
                        elif day >= 84 and ENDQ.search(s):
                            cat = "endq"
                        if cat:
                            quotes.append({"arm": arm, "dir": str(d), "day": day, "turn": r.get("turn"), "cat": cat, "text": s})
            elif k == "tool_result":
                name = r.get("name")
                tool_names[name] += 1
                day_tools[day] += 1
                res = r.get("result") if isinstance(r.get("result"), dict) else {}
                ok = res.get("ok", True)
                if ok is False:
                    errs[f"{name}:{res.get('reason_code')}"] += 1
                    consec += 1
                    consecutive_max = max(consecutive_max, consec)
                else:
                    consec = 0
                if name == "place_purchase_order":
                    if ok is not False:
                        day_pos[day] += 1
                    a = r.get("arguments") if isinstance(r.get("arguments"), dict) else {}
                    rat = a.get("rationale")
                    if isinstance(rat, str):
                        rationales.append(rat)
                if name and name.startswith("memory"):
                    mem_calls += 1
            elif k == "memory_snapshot":
                mem_bytes[day] = r.get("used_bytes")
    days = sorted(day_calls)
    nd = max(1, len(days))
    row["n_days"] = len(days)
    row["model_calls"] = n_model
    row["calls_per_day"] = n_model / nd
    row["reasoning_chars_per_call"] = st.mean(reason_chars) if reason_chars else 0
    row["reasoning_chars_per_day"] = sum(day_reason_chars.values()) / nd
    row["output_tokens_per_day"] = sum(day_out_tokens.values()) / nd
    row["think_cap_hits"] = cap_hits
    row["finish_reasons"] = dict(finish)
    row["tool_calls_per_day"] = sum(day_tools.values()) / nd
    row["accepted_pos_per_day"] = sum(day_pos.values()) / nd
    row["tool_names"] = dict(tool_names)
    row["tool_errors"] = dict(errs)
    row["tool_error_total"] = sum(errs.values())
    row["tool_error_rate"] = sum(errs.values()) / max(1, sum(tool_names.values()))
    row["malformed_errors"] = sum(v for kk, v in errs.items() if kk.endswith(":MALFORMED"))
    row["po_rejections"] = sum(v for kk, v in errs.items() if kk.startswith("place_purchase_order:"))
    row["max_consecutive_errors"] = consecutive_max
    row["memory_calls_per_day"] = mem_calls / nd
    row["memory_used_bytes_final"] = mem_bytes.get(max(mem_bytes)) if mem_bytes else None
    row["memory_used_bytes_d30"] = mem_bytes.get(30)
    for name in ("stockout", "infer", "local", "local_econ", "supplier_bad", "endq"):
        row[f"days_flag_{name}"] = len(day_flags[name]) / nd if reason_chars and any(reason_chars) else None
    # supplier-problem mentions before vs after day 21 (fraction of days)
    sb = day_flags["supplier_bad"]
    pre = [dd for dd in days if dd < 21]
    post = [dd for dd in days if dd >= 21]
    row["supplier_bad_rate_pre21"] = (sum(1 for dd in pre if dd in sb) / len(pre)) if pre and any(reason_chars) else None
    row["supplier_bad_rate_post21"] = (sum(1 for dd in post if dd in sb) / len(post)) if post and any(reason_chars) else None
    row["supplier_bad_first_day_post21"] = min((dd for dd in sb if dd >= 21), default=None)
    # purchase rationales (same channel for all policies)
    nr = len(rationales)
    row["n_rationales"] = nr
    row["rationale_words_mean"] = st.mean(len(x.split()) for x in rationales) if rationales else None
    row["rationale_stockout_frac"] = sum(1 for x in rationales if STOCKOUT.search(x)) / nr if nr else None
    row["rationale_infer_frac"] = sum(1 for x in rationales if INFER.search(x) or re.search(r"demand", x, re.I) and STOCKOUT.search(x)) / nr if nr else None
    row["rationale_account_frac"] = sum(1 for x in rationales if re.search(r"account|acct|a\d\b|line", x, re.I)) / nr if nr else None
    row["rationale_buffer_frac"] = sum(1 for x in rationales if re.search(r"buffer|safety|cushion", x, re.I)) / nr if nr else None
    # notebook
    nbf = opener(eng / "model_notebook.txt")
    nb = nbf.read() if nbf else ""
    if nbf:
        nbf.close()
    row["notebook_bytes"] = len(nb.encode())
    row["notebook_lines"] = nb.count("\n")
    row["notebook_headings"] = len(re.findall(r"^#{1,4} ", nb, re.M))
    low = nb.lower()
    row["notebook_has"] = {
        "in_transit": bool(re.search(r"in[- ]transit|pipeline|open po|arriv|promised", low)),
        "checklist": bool(re.search(r"checklist|\[ \]|\[x\]|verify|check each|always check|re-read", low)),
        "lessons": bool(re.search(r"lesson|learned|mistake|never |always ", low)),
        "demand_estimates": bool(re.search(r"oos[- ]adj|censor|true demand|/d\b|k/d|per day", low)),
        "cover_policy": bool(re.search(r"cover", low)),
        "supplier_notes": bool(re.search(r"sup_[a-d]", low)),
        "day_plan": bool(re.search(r"plan|tomorrow|next day|d\+1", low)),
    }
    return row, quotes


def main():
    qs = quarters()
    with Pool(14) as p:
        res = p.map(mine, qs, chunksize=1)
    rows = [r for r, _ in res]
    (OUT / "transcript_metrics.json").write_text(json.dumps(rows, indent=0))
    with open(OUT / "quote_candidates.jsonl", "w") as f:
        for _, qs_ in res:
            for q in qs_:
                f.write(json.dumps(q) + "\n")
    print(len(rows), "quarters;", sum(len(q) for _, q in res), "quote candidates")


if __name__ == "__main__":
    main()
