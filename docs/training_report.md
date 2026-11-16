# Teaching an open 27B model to run a produce buying desk

SMB Bench training report, November 2026

The question behind this work is what experience and learning signal make an agent better at noticing, attributing and acting on the delayed consequences of its own decisions, in businesses it has never seen. SMB Bench tests this on a wholesaler's produce buying desk run for a 90-day quarter, where the cost of a decision often arrives weeks later and unlabelled.

## Result

Qwen3.8-27B, an open-weight model trained inside the original SMB Bench simulator, finishes ahead of 8 of the 16 models on the current leaderboard on a business it never saw during training or model selection. They include both open-weight models (DeepSeek V4.1 Flash and GLM 5.3), Grok 4.7, GPT-6 Luna, Claude Haiku 5.5 and all three Muse models. Its margin over Grok, GLM, GPT-6 Luna, Muse Wonder and Claude Haiku is $2,300 to $3,300 a quarter, about two standard errors or more. Its lead over Muse Spark, Muse Ember and DeepSeek is within noise, and the six models at the top lead it by more than two standard errors. On the validation business it is ahead of 9 of the 16. Every policy earns less on the sealed business than on validation (GPT-6 Astra makes $17,300 a quarter on one and $2,000 on the other), and ours sits in the middle of the board on both.

The starting model, Qwen3.6-35B, lost $29,000 a quarter on validation. The final policy makes $2,400 there. Four reinforcement learning updates, about $600 of compute, added $1,500 a quarter on the sealed business, with a standard error of $1,100, and pointed the same way on validation.

![Result](figures/fig1_result.png)

## Findings from the final policy

The SMB Bench report found that frontier models under-buy, notice stockouts without acting on them, and short their restaurant and catering accounts. The final policy does the opposite on all three. We measured its test quarters and the nine frontier models we replayed on the same business with one script, and mined all 16,699 of its responses across 24 validation and test quarters. Quotes are the policy's own words, verbatim.

| SMB Bench report finding | Nine replayed frontier models, test business | Final policy |
|---|---|---|
| Every model under-buys | 12 to 22 tonnes a quarter, purchases cover 81% of true demand on average | 22.1 tonnes, covers 89% |
| Models notice stockouts and barely act | next order after a four-hour stockout 6% above average for GPT-6 Astra | 18% above average |
| Few protect their accounts in time | every account covered on day zero in 8 of 80 quarters | 24 of 24 quarters |
| Changed quarters cost profit | all 16 leaderboard models lose $2,800 to $13,300 against their normal quarter | loses $1,400 |
| The expert stocks for lead time plus a buffer | only Claude Fable 5.1 holds more than its lead time | holds 0.9 days against a 1.3-day lead time |
| The expert consolidates on one supplier | 1.2 to 3.0 suppliers | 3.8 suppliers |

**It infers the demand a stockout hides, then acts on it.** A sold-out shelf records lost sales as nothing. On 86 percent of test mornings the policy corrects for this before it orders.

> "Let me estimate true demand from the sales data (d13, d14, with OOS correction)"

After a four-hour stockout its next order is 18 percent above that product's average, three times GPT-6 Astra's 6 percent. After a week out of stock a quarter of the time or more, it adds 0.23 days of stock the next week, level with Astra and GLM 5.3 and behind only Claude Fable 5.1 on our measure. Its notes turn the estimate into a morning routine, and 83 percent of its final notebooks hold explicit checks, against 22 percent for Astra.

> "2. New account lines due today/tomorrow -> size gaps vs on-hand + incoming."

**It buys enough.** The report found every model bought 5 to 42 percent less than the fixed buying rule. The policy received 20.6 tonnes from suppliers and 1.5 tonnes from the local market per quarter, the most of the eleven policies we can measure on the test business, and its purchases covered 89 percent of true demand, the same as GPT-6 Astra. Its walk-in fill is still 74 percent against Astra's 80, a gap the thin buffer below explains.

**It protects every account before the consequence appears.** Shorted accounts quietly order less weeks later. Of the 323 leaderboard runs in which this happens, the report found 12 that saw the fall, tied it to their own shortfalls and fixed it within three days. The policy never lets the chain start. In all 24 quarters it set shortfall cover, a standing instruction to fill any account shortfall from the local market, for every account on the first morning, and kept 99.8 percent of account products covered to the end. Its notes state the delayed consequence it is buying protection against, the 5 percent compensation, the accounts' smaller future orders and the end-of-quarter reserve, and weigh them against the local premium.

> "Keep ON; never toggle (comp + future order shrink + closing reserve >> local premium)."

It still prices individual lines, and lets cover fill one when that loses less than buying would.

> "Skip gb (cover absorbs at 720, loss $24 vs buying loss ~$47)."

GPT-6 Astra covered every account on day zero in 5 of 9 test quarters, and the nine replayed frontier models in 8 of 80. In the customer quarter the policy's accounts kept ordering their full 15 tonnes, against 11.6 tonnes for the four strongest replayed frontier models.

**It holds profit when customers or suppliers turn.** Against its own normal quarter it loses $1,400 on average across the customer and supplier quarters, the smallest loss of all 18 policies on the test business. GPT-6 Astra loses $4,900. Part of this is a low starting point, because always-on cover buys at the local market even when nothing goes wrong, and its normal quarter is weaker than every leaderboard model's. In absolute profit it is ahead of 11 of the 16 in the customer quarter and 9 of the 16 in the supplier quarter.

**Where it falls short, it notices and does not act.** The report's expert buyer stocks for the delivery lead time plus a buffer and consolidates orders on one supplier. The policy holds 0.9 days of stock against a 1.3-day lead time and splits orders over 3.8 suppliers, against 2.9 for Astra. So a late truck, or a day with none due, leaves an account order for cover to fill at the local market, at about 2.5 times the cheapest supplier quote. It buys there on 72 of 90 days and treats the cost as the price of loyalty.

> "asp 1 is enough; let the cover handle asp line gaps (accept the premium, protect the account)."

> "Local cover for these >> shelf -> expected ~$100-250/d cover losses; accept."

In the supplier quarter it records the cheapest supplier's cancellations in its notes and keeps leaning on it, raising that supplier's share of its orders from 67 to 75 percent while the four strongest replayed frontier models cut theirs from 83 to 77 percent.

> "sup_c d+1 NET21 (15-20% pre-cancel/case), sup_d d+2 NET7 reliable"

These two habits cost it $7,700 a quarter at the local market and $1,100 in freight against GPT-6 Astra, more than its whole $6,500 gap to Astra, which its higher sales partly offset. The gap is widest in the normal quarter.

![Behaviors](figures/fig2_behaviors.png)

## Method

**Environment.** We used the original SMB Bench simulator and grader with the benchmark's tools, information and limits. Each morning the policy can query the business's records, read supplier quotes, place orders, set shortfall cover and edit its notes, which are all it carries to the next morning. Every business is played as a normal quarter, a customer quarter where shorted accounts order less, and a supplier quarter where the cheapest supplier also starts arriving late and short. We trained on four of the leaderboard's six businesses, chose between policies on business 5, and kept business 6 sealed until the final evaluation.

**Task.** The policy runs the whole desk, every product, supplier, order size, timing and cover decision for 90 days, and is graded on the quarter's profit. An earlier attempt trained one skill, stockout-aware replenishment, as single decisions scored against the fixed buying rule. The decision reward kept rising while whole-quarter profit fell after the first 64 updates, and a model tuned on one business lost ground on the others. Narrow decisions did not add up to a better buyer.

**Experience.** The useful unit of experience was a frontier state, not a frontier transcript. Fine-tuning Qwen3.6-35B on the 44 most profitable frontier transcripts from the training businesses, 179 million tokens, broke its tool calls. Restarted from frontier states at day 60, 71 percent of its runs ended on a malformed tool call and none beat the frontier. Instead we replay an archived frontier quarter to an exact morning between day 60 and 85, let the policy finish the quarter, and compare its profit with what that frontier model earned from the same morning. Because the engine is deterministic, every example carries its own fair bar.

**Learning signal.** The reward is a single bit, 1 when the run beats the frontier model's result from the same morning. On the same states a reward in dollars drifted. Its responses grew from 3,800 to 7,100 tokens and the runs that reached the end of the quarter fell from 16 to 8 of 16. Under the win-or-lose reward, 15 or 16 of 16 runs finished at every update, the share that beat the frontier went from 6 of 16 at the first update to 8 of 16 at the last, and the mean margin against the frontier rose from minus $1,100 to even. Each update draws new states, so the training curve is noisy and the held-out comparisons carry the evidence. RL raised profit in all four matched comparisons, by $1,000 to $4,400 a quarter, including on the sealed business and under prompts it was not trained with (combined one-sided p of about 0.06, a rough guide because three of the comparisons share the validation business).

![Reward over the full final RL run](figures/fig3_rl.png)

**Optimizer and model.** The policy is Qwen3.8-27B with a rank-32 LoRA adapter, trained on Tinker, Thinking Machines' hosted training service. Each of the four updates used four start states with four runs each, a learning rate of 3e-5, the CISPO loss, and MaxRL advantages, which divide each group's centered win-or-lose reward by the group's win rate. Evaluation runs with thinking on, temperature 1, top-p 0.95, top-k 20 and a reasoning cap of 57,344 tokens. The backbone mattered most. From identical frontier states, Qwen3.8-27B earned $1,200 to $2,100 more per run than Qwen3.6-35B.

**System prompt.** The final prompt keeps the benchmark's task, tools and scoring and adds three kinds of guidance. It describes the practices the report attributes to its expert buyer, states run-safety rules (a cap on calls per response, the run-ending limits, a check on lot size), and adds two rules read off the training businesses' records. Keep cover on for every account from day zero, and size each order to what sells before markdown. Cover is not optional. Making it conditional, to cut the local-market spending, brought account shortfalls back from 8 kg to 456 kg and cost $5,700 a quarter on validation.

## What came from where

Figure 4 splits the $31,400 validation climb by source. Moving from Qwen3.6-35B to Qwen3.8-27B is worth $20,400. The report's practices and the run-safety rules add $7,600. The two rules read off the training businesses add $2,200. Reinforcement learning in the simulator adds $1,200 on validation and $1,500 on the sealed business. Each step was measured on top of the one before, so the split depends on that order.

**What the environment taught.** The simulator's training businesses hold rules that transfer. We read two of them off the records of archived frontier runs, wrote them into the prompt, and they produced the account protection and lot sizing that frontier models get wrong, on two businesses the policy never saw. The rules reached the policy as written instructions, so this shows the signal is in the environment. Whether an agent can extract it unaided is the open part of the question.

**What training put in the weights.** Against the same model and prompt without reinforcement learning, on the sealed business, RL made the policy commit to its plan. It bought 0.8 tonnes more from suppliers and $1,600 less at the local market, filled 74 percent of walk-in demand against 70, ended the quarter with more stock on hand, almost never switched cover off, and covered every account on day zero in 12 of 12 quarters against 10 of 12. It reacted harder to stockout weeks on both businesses (combined one-sided p of about 0.07). It turned three disaster quarters below a $10,000 loss into one, and on validation its whole lead comes from one quarter the untrained model ended early after ten rejected orders in a row. It did not change tool errors, tokens, the stock buffer or the number of suppliers.

**What this says about learnability.** The environment gives a usable learning signal. A win-or-lose reward, scored against archived frontier runs from the same morning, moved the weights in four updates, and the change held on a business the policy never saw and under prompts it was not trained with. The behaviors the policy carries across businesses reverse two of the report's named frontier failures and match the best frontier models on the third. Written instructions introduced them, and reinforcement learning made the policy hold them more consistently. Learning a delayed-consequence habit from the reward alone is not yet shown, because updates that start at day 60 cannot see consequences that build from the first weeks of a quarter. The remaining gap says where to look next. Claude Fable 5.1, the only replayed frontier model that stocks above its lead time, buys almost nothing at the local market. A numeric buffer rule drawn from training-business data, and reinforcement learning from days 10 to 59, when a quarter's buying pattern and stock levels are set, are the next tests.

![Levers and attribution](figures/fig4_levers.png)

## Limits

Validation and test are one business each, and 12 test quarters per policy resolve about $2,300 between two policies. The leaderboard models ran the benchmark's own prompt and ours ran the prompt above, with the same tools, information, limits and grader. We did not rerun frontier models with our prompt, so part of our margin is instructions, $9,800 of the validation climb in Figure 4. We trained on four of the six leaderboard businesses and report no 18-task average. Frontier numbers are the leaderboard's own scored runs on businesses 5 and 6, and our replays of nine of those models reproduce them to the cent. Our pre-registered selection rule picked the untrained model on a validation near-tie. Both ran on the sealed business as a declared matched pair, and the untrained one is ahead of 5 of the 16. The RL updates ran under an earlier draft of the prompt, and every evaluation uses the final prompt. Qwen's reference parser accepted four tool calls in 3 of the 12 test quarters that our stricter training parser would have rejected, a run of ours that ends early scores a $25,000 loss where the leaderboard drops failed runs, and the reasoning cap fired once in the 12 test quarters.

## Code and model

This repository holds the harness, training and evaluation code, the final prompt (`prompts/buyer_system.txt`), the leaderboard snapshot and the figure scripts. `scripts/story_figures.py` regenerates every figure from the run records and `data/leaderboard-runs-b5-b6.json`, and `docs/evidence/` holds the transcript measures. The trained adapter is stored on Tinker. `scripts/export_model.py` exports it as a PEFT adapter and a merged Hugging Face model, and runs once a billing block on our Tinker account lifts.
