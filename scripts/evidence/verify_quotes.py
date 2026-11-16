"""Check that every quote used in findings.md appears verbatim in its source (reasoning_text of that day, or the notebook)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUNS, iter_jsonl, opener  # noqa: E402

T = "test/maxrl-001-final-v6b/"
QUOTES = [
    # (quarter dir, day or None for notebook, text)
    (T + "s505_experience_long_null_1__s1", 2, "spinach 3.9k (16 oos! censored — true demand is much higher)"),
    (T + "s505_experience_long_account_line_1__s0", 15, "Potatoes d13: 0, 16h; d14: 7,469, 2h OOS → *16/14 = 8,534 → ~8k+"),
    (T + "s505_experience_long_account_line_1__s0", 15, "Let me estimate true demand from the sales data (d13, d14, with OOS correction)"),
    (T + "s505_experience_long_null_1__s2", 30, "Grape tomatoes: 3236, OOS 9 → censored ≈ 3236*16/7 ≈ 7394 — demand is high!"),
    (T + "s505_experience_long_null_1__s1", None, "OOS reading: hours_out_of_stock=16 = full out day (0 sales) -> that SKU's POS demand ≈ 3-day total / 2"),
    (T + "s505_experience_long_null_1__s1", None, "Day 0: sku='*' cover=true on all 6 accounts. Keep ON; never toggle (comp + future order shrink + closing reserve >> local premium)."),
    (T + "s505_experience_long_supplier_deterioration_1__s0", 45, "sup_c Alder: cheapest, arr+1, NET21, cancels ~17% (4/24 d42-43)."),
    (T + "s505_experience_long_supplier_deterioration_1__s2", 81, "Candidate sup_c (d82 arrival, cheap, but 1-day late risk + 15-20% cancellation):"),
    (T + "s505_experience_long_null_1__s2", 70, "asp 1 is enough; let the cover handle asp line gaps (accept the premium, protect the account)."),
    (T + "s505_experience_long_null_1__s3", 13, "Skip gb (cover absorbs at 720, loss $24 vs buying loss ~$47)."),
    (T + "s505_experience_long_supplier_deterioration_1__s0", 0, "Given cover is set for all accounts, even if potatoes fall short, cover would backstop (local potato price 97 vs account 77 → cover loses 20/lb, cheap!)."),
    (T + "s505_experience_long_supplier_deterioration_1__s2", 79, "tomato local 316 c/lb vs line 155: loss; but the cover rule auto-buys local if ON — the loss is unavoidable."),
    (T + "s505_experience_long_null_1__s1", None, "Local cover for these >> shelf -> expected ~$100-250/d cover losses; accept."),
    (T + "s505_experience_long_null_1__s1", None, "Local > shelf for most SKUs -> shortfall cover loses; maximize PO supply."),
    (T + "s505_experience_long_supplier_deterioration_1__s0", None, "2. New account lines due today/tomorrow -> size gaps vs on-hand + incoming."),
    (T + "s505_experience_long_supplier_deterioration_1__s0", None, "7. TOOL FIX: rejected PO burns its idempotency_key -> new key on retry."),
    (T + "s505_experience_long_supplier_deterioration_1__s2", None, "D86 checks: sup_c prom-d85 truck (23c ~268k) on time d86 or slipped to d87?"),
    (T + "s505_experience_long_null_1__s1", None, "PO promised D arrives D@05:30, BEFORE day-D account fills + sales."),
    (T + "s505_experience_long_account_line_1__s0", None, "d89 plan: NO BUY (d90+ arrivals worthless at close)."),
    (T + "s505_experience_long_supplier_deterioration_1__s2", None, "Suppliers: sup_c d+1 NET21 (15-20% pre-cancel/case), sup_d d+2 NET7 reliable"),
]


def main():
    ok_all = True
    for d, day, text in QUOTES:
        q = RUNS / d
        found = False
        if day is None:
            f = opener(q / "engine/model_notebook.txt")
            found = text in f.read()
            f.close()
        else:
            for r in iter_jsonl(q / "engine/model_transcript.jsonl"):
                if r.get("kind") == "model" and r.get("day") == day and text in (r.get("reasoning_text") or ""):
                    found = True
                    break
        words = len(text.split())
        print(("OK " if found else "MISSING ") + f"[{words}w] {d.split('/')[-1]} {'notebook' if day is None else 'd' + str(day)}: {text[:80]}")
        ok_all &= found and words < 40
    print("ALL VERIFIED" if ok_all else "SOME FAILED")


if __name__ == "__main__":
    main()
