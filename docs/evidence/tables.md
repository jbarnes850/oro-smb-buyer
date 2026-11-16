**Return ($/quarter, TEST b505)**

| metric | MaxRL final | base v6 | Astra | top-4 frontier | all 9 frontier |
|---|---|---|---|---|---|
| primary (mean of world means) | -4,495 | -5,984 | 1,969 | -297 | -3,407 |

**Per-world return, TEST ($)**

| policy | primary | null | account_line | supplier_det | worst quarter | quarters < -$10K |
|---|---|---|---|---|---|---|
| gpt-6-astra | 1,969 | 5,218 | 4,510 | -3,819 | -6,311 | 0/9 |
| claude-fable-5-1 | -339 | 3,446 | -782 | -3,680 | -7,140 | 0/9 |
| gpt-6.1-sol | -945 | 5,259 | -2,881 | -5,212 | -8,721 | 0/9 |
| claude-opus-5-5 | -1,876 | 3,888 | 224 | -9,739 | -12,234 | 2/9 |
| gemini-4-argon | -2,980 | -396 | -1,810 | -6,735 | -12,204 | 1/9 |
| maxrl_final | -4,495 | -3,557 | -1,578 | -8,351 | -12,657 | 1/12 |
| deepseek-v4.1-flash-1017 | -5,707 | 2,137 | -6,664 | -12,595 | -13,924 | 3/9 |
| base_v6 | -5,984 | -4,748 | -2,641 | -10,564 | -13,913 | 3/12 |
| muse-wonder-1.0 | -6,769 | 1,772 | -9,407 | -12,673 | -13,155 | 4/9 |
| glm-5.3 | -7,002 | -1,036 | -8,330 | -11,639 | -12,976 | 3/8 |
| grok-4.7 | -7,013 | 1,876 | -8,853 | -14,061 | -14,704 | 4/9 |

**Per-world return, VAL b504 ($; a model-failure quarter scored -$25K)**

| policy | primary | null | account_line | supplier_det | failures |
|---|---|---|---|---|---|
| maxrl_final_val (n=12) | 2,419 | 4,734 | 3,970 | -1,445 | - |
| base_v6_val (n=12) | 1,216 | 3,847 | 6,239 | -6,437 | supplier_deterioration:TOOL_CONTRACT_ERRORS@d75 |
| q36_base_v1_val (n=3) | -28,963 | -12,784 | -51,263 | -22,843 | - |

**Q1 demand inference and walk-in service (TEST)**

| metric | MaxRL final | base v6 | Astra | top-4 frontier | all 9 frontier |
|---|---|---|---|---|---|
| days with reasoning about stockouts (frac) | 0.97 | 0.98 | n/a | n/a | n/a |
| days with explicit censoring/true-demand inference (frac) | 0.86 | 0.85 | n/a | n/a | n/a |
| purchase rationales mentioning stockouts (frac) | 0.081 | 0.083 | 0.012 | 0.025 | 0.041 |
| next-morning order / sku mean, after >=4h walk-in stockout | 1.18 | 1.11 | 1.06 | 1.05 | 1.05 |
| next-morning order / sku mean, after no stockout | 0.94 | 0.97 | 1.00 | 1.01 | 1.01 |
| P(order next morning | stockout >=4h) | 0.83 | 0.79 | 0.84 | 0.78 | 0.72 |
| P(order next morning | no stockout) | 0.64 | 0.63 | 0.77 | 0.74 | 0.66 |
| walk-in fill (pos sold / latent walk-in demand) | 0.741 | 0.701 | 0.803 | 0.807 | 0.736 |
| sku-days with any walk-in stockout (frac) | 0.327 | 0.360 | 0.253 | 0.235 | 0.325 |
| walk-in stockout hours / quarter | 7,424 | 8,370 | 5,266 | 5,002 | 7,158 |
| stockout-hour share, days 0-29 | 0.215 | 0.227 | 0.148 | 0.148 | 0.235 |
| stockout-hour share, days 60-89 | 0.287 | 0.311 | 0.219 | 0.207 | 0.270 |

**Q1 on VAL (ours + Qwen3.6/v1 contrast)**

| metric | MaxRL final (VAL) | base v6 (VAL) | Qwen3.6 base v1 (VAL, n=3) |
|---|---|---|---|
| days with reasoning about stockouts (frac) | 1.00 | 0.99 | 0.45 |
| days with explicit censoring/true-demand inference (frac) | 0.95 | 0.91 | 0.11 |
| next-morning order ratio after stockout | 1.16 | 1.15 | 0.96 |
| next-morning order ratio after no stockout | 0.93 | 0.92 | 1.11 |
| walk-in fill | 0.751 | 0.732 | 0.470 |

**Q2 account cover (TEST)**

| metric | MaxRL final | base v6 | Astra | top-4 frontier | all 9 frontier |
|---|---|---|---|---|---|
| first day any cover set | 0.0 | 0.0 | 0.0 | 8.0 | 11.5 |
| accounts with cover '*' set on day 0 (of 6) | 6.0 | 5.5 | 3.3 | 1.0 | 0.8 |
| cover-off rules over the quarter | 1.0 | 5.8 | 7.8 | 5.4 | 4.0 |
| per-product cover rules | 1.0 | 14.0 | 19.2 | 13.6 | 8.6 |
| account shortfall kg (ordered - delivered) | 58 | 75 | 308 | 427 | 640 |
| account tonnage ordered (t) | 14.71 | 14.55 | 13.46 | 12.19 | 10.71 |
| closing demand reserve ($) | 624 | 785 | 585 | 1,451 | 2,052 |

**Q2 account cover (TEST)** (world: null)

| metric | MaxRL final | base v6 | Astra | top-4 frontier | all 9 frontier |
|---|---|---|---|---|---|
| first day any cover set | 0.0 | 0.0 | 0.0 | 0.1 | 8.7 |
| accounts with cover '*' set on day 0 (of 6) | 6.0 | 6.0 | 2.0 | 0.5 | 0.9 |
| cover-off rules over the quarter | 0.0 | 6.5 | 18.3 | 8.1 | 6.6 |
| per-product cover rules | 0.0 | 18.8 | 40.0 | 20.0 | 13.6 |
| account shortfall kg (ordered - delivered) | 1 | 14 | 701 | 826 | 1,279 |
| account tonnage ordered (t) | 14.95 | 14.95 | 14.95 | 14.95 | 14.95 |
| closing demand reserve ($) | 0 | 0 | 0 | 0 | 0 |

**Q2 account cover (TEST)** (world: account_line)

| metric | MaxRL final | base v6 | Astra | top-4 frontier | all 9 frontier |
|---|---|---|---|---|---|
| first day any cover set | 0.0 | 0.0 | 0.0 | 17.2 | 21.3 |
| accounts with cover '*' set on day 0 (of 6) | 6.0 | 6.0 | 6.0 | 1.5 | 1.1 |
| cover-off rules over the quarter | 0.0 | 0.8 | 0.0 | 5.4 | 3.2 |
| per-product cover rules | 0.0 | 1.2 | 0.0 | 12.4 | 7.2 |
| account shortfall kg (ordered - delivered) | 2 | 0 | 0 | 199 | 287 |
| account tonnage ordered (t) | 14.95 | 14.95 | 14.95 | 11.62 | 9.44 |
| closing demand reserve ($) | 32 | 0 | 0 | 1,960 | 2,810 |

**Q3 supplier management (TEST, all worlds)**

| metric | MaxRL final | base v6 | Astra | top-4 frontier | all 9 frontier |
|---|---|---|---|---|---|
| top-supplier share of ordered grams | 0.728 | 0.725 | 0.842 | 0.803 | 0.870 |
| suppliers used (>2% of grams) | 3.8 | 3.7 | 2.9 | 2.7 | 2.3 |
| claims auto-filed per quarter (not an agent action) | 324 | 307 | 302 | 301 | 251 |
| claim recoveries ($) | 605 | 569 | 576 | 561 | 473 |

**Q3 discount supplier share before/after day 21** (world: supplier_deterioration)

| metric | MaxRL final | base v6 | Astra | top-4 frontier | all 9 frontier |
|---|---|---|---|---|---|
| discount-supplier share of ordered grams, days 0-20 | 0.668 | 0.656 | 0.855 | 0.826 | 0.884 |
| discount-supplier share of ordered grams, days 21-89 | 0.746 | 0.748 | 0.827 | 0.765 | 0.868 |
| discount-supplier share of received grams, orders day 0-20 | 0.670 | 0.660 | 0.854 | 0.828 | 0.886 |
| discount-supplier share of received grams, orders day 21-89 | 0.721 | 0.725 | 0.811 | 0.746 | 0.856 |
| local share of stock | 0.083 | 0.108 | 0.034 | 0.026 | 0.024 |
| return ($) | -8,351 | -10,564 | -3,819 | -5,612 | -8,906 |

**Q3 discount supplier share before/after day 21** (world: null)

| metric | MaxRL final | base v6 | Astra | top-4 frontier | all 9 frontier |
|---|---|---|---|---|---|
| discount-supplier share of ordered grams, days 0-20 | 0.735 | 0.798 | 0.836 | 0.808 | 0.854 |
| discount-supplier share of ordered grams, days 21-89 | 0.738 | 0.762 | 0.865 | 0.788 | 0.840 |
| discount-supplier share of received grams, orders day 0-20 | 0.737 | 0.801 | 0.838 | 0.809 | 0.856 |
| discount-supplier share of received grams, orders day 21-89 | 0.740 | 0.766 | 0.868 | 0.791 | 0.842 |
| local share of stock | 0.068 | 0.076 | 0.015 | 0.006 | 0.016 |
| return ($) | -3,557 | -4,748 | 5,218 | 4,453 | 2,463 |

**Discount-supplier share of ordered grams by week, supplier_deterioration world (TEST)**

| policy | d0-6 | d7-13 | d14-20 | d21-27 | d28-34 | d35-41 | d42-48 | d49-55 | d56-62 | d63-69 | d70-76 | d77-83 | d84-90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MaxRL final | 0.68 | 0.59 | 0.77 | 0.69 | 0.68 | 0.78 | 0.69 | 0.79 | 0.76 | 0.80 | 0.77 | 0.80 | 0.86 |
| base v6 | 0.67 | 0.64 | 0.67 | 0.71 | 0.67 | 0.66 | 0.75 | 0.85 | 0.77 | 0.80 | 0.82 | 0.82 | 0.75 |
| Astra | 0.82 | 0.84 | 0.90 | 0.87 | 0.78 | 0.75 | 0.77 | 0.93 | 0.86 | 0.89 | 0.81 | 0.82 | 0.95 |
| top-4 frontier | 0.83 | 0.80 | 0.85 | 0.85 | 0.79 | 0.78 | 0.74 | 0.81 | 0.74 | 0.74 | 0.74 | 0.74 | 0.80 |
| all 9 frontier | 0.87 | 0.86 | 0.92 | 0.91 | 0.87 | 0.88 | 0.86 | 0.88 | 0.85 | 0.86 | 0.86 | 0.85 | 0.90 |

**Q4 local market (TEST)**

| metric | MaxRL final | base v6 | Astra | top-4 frontier | all 9 frontier |
|---|---|---|---|---|---|
| local share of stock (local / (local + supplier received)) | 0.068 | 0.079 | 0.027 | 0.014 | 0.018 |
| local share, days 0-29 | 0.047 | 0.053 | 0.020 | 0.009 | 0.018 |
| local share, days 30-59 | 0.072 | 0.093 | 0.032 | 0.015 | 0.020 |
| local share, days 60-89 | 0.085 | 0.095 | 0.030 | 0.018 | 0.016 |
| days with a local purchase (of 90) | 72 | 73 | 45 | 20 | 20 |
| products bought locally (of 20) | 18.0 | 18.0 | 14.1 | 6.8 | 5.9 |
| share of account grams filled from local | 0.101 | 0.115 | 0.043 | 0.022 | 0.028 |
| local spend ($) | 11,868 | 13,476 | 4,190 | 1,819 | 2,512 |
| local price / cheapest same-day supplier quote | 2.54 | 2.54 | 2.54 | 2.53 | 2.54 |
| premium over cheapest same-day quote ($) | 7,204 | 8,177 | 2,541 | 1,102 | 1,523 |
| account revenue on locally covered grams ($) | 6,916 | 7,909 | 2,500 | 1,094 | 1,509 |
| covered-line margin: that revenue - local cost ($) | -4,952 | -5,567 | -1,691 | -725 | -1,002 |
| local lines where cost > 1.05 x line revenue (frac) | 0.96 | 0.95 | 0.84 | 0.83 | 0.67 |
| local grams: no agent truck due that sku-day | 0.37 | 0.39 | 0.28 | 0.20 | 0.24 |
| local grams: a truck due was late/short/cancelled | 0.52 | 0.50 | 0.64 | 0.38 | 0.30 |
| local grams: truck on time but too small | 0.12 | 0.10 | 0.08 | 0.03 | 0.04 |
| days with reasoning weighing local cost/premium (frac) | 0.99 | 0.99 | n/a | n/a | n/a |

**Q4 local market, null world** (world: null)

| metric | MaxRL final | base v6 | Astra | top-4 frontier | all 9 frontier |
|---|---|---|---|---|---|
| local share of stock | 0.068 | 0.076 | 0.015 | 0.006 | 0.016 |
| local spend ($) | 11,774 | 14,046 | 2,281 | 808 | 2,371 |
| covered-line margin ($) | -4,930 | -5,905 | -826 | -270 | -912 |
| account shortfall kg | 1 | 14 | 701 | 826 | 1,279 |
| gross margin | 0.252 | 0.237 | 0.328 | 0.332 | 0.325 |
| return ($) | -3,557 | -4,748 | 5,218 | 4,453 | 2,463 |

**Q5 volume, buffers, waste (TEST)**

| metric | MaxRL final | base v6 | Astra | top-4 frontier | all 9 frontier |
|---|---|---|---|---|---|
| supplier tonnage received (t) | 20.64 | 19.81 | 20.40 | 19.47 | 16.88 |
| supplier tonnage ordered (t) | 22.24 | 21.25 | 21.86 | 20.85 | 18.10 |
| local tonnage (t) | 1.47 | 1.64 | 0.59 | 0.26 | 0.34 |
| (supplier + local) / (latent walk-in + account orders) | 0.888 | 0.866 | 0.886 | 0.878 | 0.814 |
| (supplier + local) / sold | 0.996 | 0.991 | 0.984 | 0.985 | 0.976 |
| share of offered supplier capacity bought | 0.303 | 0.290 | 0.298 | 0.284 | 0.247 |
| sku-days with >=90% of offered capacity bought | 0.071 | 0.061 | 0.042 | 0.042 | 0.027 |
| orders at the offer's max_cases (frac) | 0.96 | 0.95 | 0.93 | 0.91 | 0.89 |
| mean on-hand / mean daily sales (days of cover) | 0.89 | 0.88 | 0.76 | 0.95 | 0.84 |
| promised lead time, grams-weighted (days) | 1.32 | 1.33 | 1.22 | 1.24 | 1.15 |
| revenue ($) | 92,770 | 90,531 | 90,543 | 84,109 | 73,435 |
| gross margin (revenue - COGS)/revenue | 0.249 | 0.236 | 0.314 | 0.324 | 0.325 |
| spoilage write-off ($) | 488 | 487 | 211 | 443 | 335 |
| expired + aged-cull grams / stock received | 0.0070 | 0.0069 | 0.0032 | 0.0059 | 0.0058 |
| inspection-rejected tonnage (t) | 0.289 | 0.276 | 0.269 | 0.263 | 0.223 |

**Q6 memory and notebook (TEST)**

| metric | MaxRL final | base v6 | Astra | top-4 frontier | all 9 frontier |
|---|---|---|---|---|---|
| final notebook (notes.md) bytes | 6,759 | 9,164 | 5,365 | 7,518 | 7,529 |
| notebook markdown headings | 8.5 | 16.1 | 4.0 | 5.0 | 6.7 |
| notebook tracks in-transit/promised POs (frac of quarters) | 1.00 | 1.00 | 0.78 | 0.92 | 0.96 |
| notebook has check/verify rules (frac) | 0.83 | 0.67 | 0.22 | 0.17 | 0.29 |
| notebook has demand estimates (frac) | 1.00 | 1.00 | 0.78 | 0.78 | 0.84 |
| memory tool calls per day | 2.17 | 2.06 | 8.34 | 5.44 | 4.09 |
| total memory used at close (bytes) | 11,576 | 10,959 | 29,005 | 23,341 | 19,372 |

**Q7 return components ($/quarter, TEST, mean of world means)**

| component | MaxRL final | base v6 | Astra | MaxRL - base | MaxRL - Astra |
|---|---|---|---|---|---|
| revenue | 92,770 | 90,531 | 90,543 | 2,239 | 2,227 |
| COGS (incl. local buys) | 69,530 | 68,957 | 62,266 | 573 | 7,264 |
| fees: dispatch | 4,347 | 4,148 | 3,236 | 198 | 1,111 |
| fees: delivery | 2,342 | 2,252 | 2,309 | 90 | 33 |
| fees: receiving | 1,401 | 1,346 | 1,385 | 55 | 16 |
| fees: overhead | 17,943 | 17,943 | 17,943 | 0 | 0 |
| spoilage | 488 | 487 | 211 | 1 | 277 |
| inspection write-down | 736 | 703 | 695 | 33 | 41 |
| claim recoveries | 653 | 617 | 624 | 36 | 29 |
| refunds | 312 | 301 | 361 | 11 | -49 |
| valuation adj. (incl. -reserve) | -671 | -833 | -627 | 163 | -43 |
| return | -4,495 | -5,984 | 1,969 | 1,489 | -6,465 |
| memo: local spend | 11,868 | 13,476 | 4,190 | -1,608 | 7,678 |
| memo: local premium vs quote | 7,204 | 8,177 | 2,541 | -973 | 4,663 |
| memo: demand reserve | 624 | 785 | 585 | -161 | 39 |

**Q7 behaviour, MaxRL final vs base (TEST and VAL)**

| metric | MaxRL final | base v6 | MaxRL (VAL) | base v6 (VAL) | Astra | all 9 frontier |
|---|---|---|---|---|---|---|
| model calls per day | 7.36 | 7.23 | 8.10 | 8.32 | 25.62 | 14.03 |
| output tokens per day (ours) | 30,485 | 30,610 | 35,414 | 31,831 | 0 | 0 |
| reasoning chars per call (ours) | 7,650 | 7,909 | 7,762 | 6,718 | 0 | 0 |
| think-cap hits per quarter | 0.08 | 0.17 | 0.00 | 0.00 | 0.00 | 0.00 |
| tool calls per day | 32.9 | 32.0 | 41.9 | 42.0 | 52.0 | 38.4 |
| accepted purchase orders per day | 19.0 | 18.2 | 26.0 | 25.9 | 19.0 | 15.8 |
| tool-call error rate | 0.031 | 0.031 | 0.040 | 0.045 | 0.018 | 0.045 |
| MALFORMED errors per quarter | 4.2 | 3.4 | 8.1 | 6.1 | 0.0 | 2.4 |
| rejected purchase orders per quarter | 23.2 | 19.2 | 33.1 | 33.8 | 0.0 | 7.8 |
| max consecutive failed tool results, any reason | 7.3 | 4.6 | 9.3 | 11.6 | 2.7 | 3.9 |

**MaxRL final minus base v6, by world ($)**

| world | TEST | VAL (failure = -$25K) |
|---|---|---|
| null | +1,191 | +886 |
| account_line | +1,063 | -2,269 |
| supplier_deterioration | +2,212 | +4,992 |

**Q8 remaining failure modes (TEST)**

| metric | MaxRL final | base v6 | Astra | top-4 frontier | all 9 frontier |
|---|---|---|---|---|---|
| engine-rejected actions per quarter (grade) | 19.5 | 16.9 | 0.0 | 2.2 | 4.9 |
| ordered grams/day, days 80-89 vs days 10-79 | 0.82 | 0.72 | 0.78 | 0.72 | 0.72 |
| ordered grams/day, days 85-89 vs days 10-79 | 0.60 | 0.44 | 0.59 | 0.56 | 0.58 |
| closing on-hand / mean daily sales (days) | 1.15 | 0.87 | 0.68 | 0.81 | 0.81 |
| closing inventory value ($) | 605 | 378 | 427 | 439 | 356 |
| dock refusals (storage full) per quarter | 3.33 | 4.67 | 0.11 | 0.14 | 0.29 |
| refused-at-dock kg per quarter | 61.2 | 82.0 | 2.0 | 2.5 | 5.3 |

**Tool error codes per quarter (top 10 for MaxRL final, TEST)**

| tool:code | MaxRL final | base v6 | MaxRL (VAL) | base v6 (VAL) |
|---|---|---|---|---|
| query_records:SQL_REJECTED | 42.00 | 43.08 | 49.33 | 54.00 |
| place_purchase_order:UNKNOWN_OFFER | 12.92 | 10.50 | 13.50 | 10.33 |
| list_offers:RESULT_TOO_LARGE | 10.42 | 10.17 | 47.92 | 60.00 |
| place_purchase_order:DUPLICATE_KEY_CONFLICT | 3.92 | 3.50 | 5.75 | 11.67 |
| memory_edit:EDIT_OLD_NOT_FOUND | 3.67 | 5.50 | 2.25 | 3.75 |
| place_purchase_order:MALFORMED | 3.42 | 2.00 | 7.00 | 6.25 |
| memory_write:CALL_TEXT_TOO_LARGE | 2.92 | 2.92 | 4.58 | 1.42 |
| place_purchase_order:ABOVE_REMAINING_CAPACITY | 2.58 | 2.83 | 1.83 | 1.08 |
| memory_write:TOO_MANY_FILES | 2.58 | 0.00 | 0.00 | 0.00 |
| memory_edit:NOTES_LIMIT_EXCEEDED | 2.50 | 2.50 | 0.92 | 2.00 |
