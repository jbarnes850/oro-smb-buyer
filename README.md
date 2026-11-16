# oro-smb-buyer

**Results.** Read [`docs/training_report.md`](docs/training_report.md). The four figures are in
[`docs/figures/`](docs/figures/) and the transcript evidence behind the behavior findings is in
[`docs/evidence/`](docs/evidence/).

This package reproduces a post-training campaign in which Qwen3.8-27B, served and trained on
Tinker, runs the whole buying desk of the original ORO SMB Bench simulator (engine `retail_v1`
at commit `b17b864`): every product, supplier, order size, timing and shortfall-cover decision
over a 90-day quarter, graded by the engine's own independent grader on whole-business return.
It contains the evaluation harness (the unmodified engine runs in a separate, credential-free
child process; the policy samples native tokens through the Tinker Cookbook), RL from archived
frontier mid-quarter states (MaxRL on "beat the frontier from the same state", and CISPO/RLOO on
dollar return), the economically selected SFT warm start (reported as a negative result), the
VAL selection and sealed-TEST reports against exact replays of archived frontier runs, and the
figure scripts. The final system prompt is `prompts/buyer_system.txt`.

## What is not included

- The SMB Bench engine source, the world bundles (they contain private truth), archived frontier
  transcripts, run outputs, checkpoints and credentials. You supply these by path.
- Run outputs. The training report (`docs/training_report.md`) and its figures (`docs/figures/`)
  summarize them; the per-quarter records stay with the campaign.
- Model weights. The final RL checkpoint lives in Tinker storage; `scripts/export_model.py` turns it
  into a PEFT adapter and a merged Hugging Face model (section 5).

## Requirements

| what | version used |
|---|---|
| OS | Linux (tested on aarch64) |
| Python (training/eval process) | 3.12.13 (`requires-python >=3.11,<3.14`) |
| uv | 0.11.19 |
| tinker | 0.39.1 (the service refuses 0.39.0) |
| tinker-cookbook | 0.6.1, git `8486b8df2fc32e9713cf96dd5f75d43c7ef70a17` |
| torch, chz, transformers | 2.15.0, 0.4.0, 5.22.2 (all pins in `uv.lock`) |
| engine interpreter | Python 3.12 with numpy 2.5.3, pandas 2.3.3, pyarrow 22.0.0, duckdb 1.4.1 (`engine-runtime/pyproject.toml`) |
| SMB Bench engine | `retail_v1` at commit `b17b864b63a756378cff74a6b9ed718f6823cd32` |
| matplotlib | only for figures (`--extra plots`) |

Every quarter re-hashes the engine files listed in `data/engine-b17b864.json` (107 Python files and
8 closure-rule files) and refuses to run if any differs, so a different engine revision fails
loudly instead of silently changing the task.

The engine, world bundles, grader and archived frontier runs come from ORO. The SMB Bench report
(oroagents.com/benchmarks/smb-bench) links the results and traces dataset
(huggingface.co/datasets/oro-agents/smb-bench), the gated grader
(huggingface.co/datasets/oro-agents/smb-bench-grader), the Harbor Hub setup
(hub.harborframework.com/datasets/oro-agents/smb-bench) and team@oroagents.com for access.

You also need:

- **World bundles**: 12 TRAIN worlds (businesses b500-b503), 3 VAL worlds (b504) and, for the
  sealed TEST only, 3 TEST worlds (b505). Each mechanism is one of `null`, `account_line`,
  `supplier_deterioration`; each world is a directory `s50N_experience_long_<mechanism>_1/` with
  `manifest.json`, `public/scenario_public.json` and `private/*`. Keep each split in its own parent
  directory.
- **Frontier index** (branch-eval, RL, hurdles, VAL/TEST reference): a JSON list of archived
  frontier quarters, one row per run:
  `{"trial_id": ..., "model": ..., "task_id": "r2-b500-account_line", "reward": ..., "transcript": "/path/<trial>.responses.jsonl.gz"}`.
  `transcript` is a gzip of the run's transcript records (rows with `kind`, `day`, `turn`,
  `content`, `tool_calls`) or the archived run `*.tar.gz` containing `model_transcript.jsonl`
  (the key may also be `path`). Use separate index files for TRAIN (b500-b503), VAL (b504) and TEST (b505).
- **SFT selection** (warm start only): `{"archives": [rows as above with "path" = archived run *.tar.gz]}`
  for TRAIN businesses, plus a JSON list of held-out trial ids.
- **Tinker**: `TINKER_API_KEY` in the environment for anything that samples or trains. The package
  never reads a key file and never passes credentials to the engine child.

## Setup

```bash
cd oro-smb-buyer
uv sync --extra plots                    # .venv: tinker, cookbook, this package
uv sync --project engine-runtime         # engine-runtime/.venv: the engine interpreter

export SMB_ENGINE_DIR=/path/to/retail_v1                       # engine source at b17b864 (required)
export SMB_ENGINE_PYTHON=$PWD/engine-runtime/.venv/bin/python  # default shown
export SMB_RUNS_DIR=$PWD/runs                                  # default shown; all outputs go here
export SMB_FRONTIER_INDEX=/path/to/frontier-train-index.json   # TRAIN frontier quarters
export TINKER_API_KEY=...                                      # only for sampling/training steps
```

| variable | meaning | default |
|---|---|---|
| `SMB_ENGINE_DIR` | engine `retail_v1` source | required |
| `SMB_ENGINE_PYTHON` | engine interpreter (kept literal, not resolved) | `engine-runtime/.venv/bin/python` |
| `SMB_RUNS_DIR` | quarters, training runs, `ledger.jsonl`, figures | `<package>/runs` |
| `SMB_WORLD_REGISTRY` | world registry (also `--registry` on most scripts) | `$SMB_RUNS_DIR/worlds.json` |
| `SMB_FRONTIER_INDEX` | TRAIN frontier index (also `--frontier-index`) | none |
| `TINKER_API_KEY` | read by the Tinker SDK | none |

Build the world registries (hashes and relative paths only; nothing is copied), then check the
setup without sampling:

```bash
uv run python scripts/campaign.py build-registry \
    --train-dir /path/to/worlds/train --val-dir /path/to/worlds/validation
uv run python scripts/campaign.py build-registry \
    --train-dir /path/to/worlds/train --val-dir /path/to/worlds/validation \
    --test-dir /path/to/worlds/test --out runs/worlds-with-test.json   # sealed TEST only
uv run python scripts/campaign.py check-setup
PY=.venv/bin/python scripts/smoke_test.sh
```

Zero-cost prerequisites (the engine and grader run; no model is called):

```bash
# Advance-only floor per world (~30 s): the RL failure reward for the dollar-return arms.
uv run python scripts/campaign.py noop --worlds train validation --out runs/noop-floor

# Graded returns of archived frontier quarters, replayed action by action through this harness.
uv run python scripts/frontier_hurdles.py $SMB_FRONTIER_INDEX --out runs/frontier-hurdles --workers 8
uv run python scripts/frontier_hurdles.py frontier-val-index.json --out runs/frontier-val-reference
```

## 1. Evaluate the base policy and a trained checkpoint on VAL

Each quarter is the full original 90-day episode. The configuration below is the one used for
the reported VAL and TEST numbers: thinking on, temperature 1, top-p 0.95, top-k 20, Qwen's
reference tool-call parser (`--parse-mode reference`), and an evaluation-only reasoning cap of
57,344 tokens (`--think-budget`; see `ThinkingBudget` in `scripts/campaign.py`).

```bash
EVAL="--samples 4 --top-p 0.95 --top-k 20 --parse-mode reference --think-budget 57344"

uv run python scripts/campaign.py eval --policy base --worlds validation $EVAL \
    --out runs/val/base --tag val-base --purpose "VAL: frozen Qwen3.8-27B"

uv run python scripts/campaign.py eval --policy state:tinker://<training-run>/weights/final \
    --worlds validation $EVAL --out runs/val/maxrl-final --tag val-maxrl-final --purpose "VAL: MaxRL final state"

uv run python scripts/campaign.py status runs/val     # progress while quarters run
```

`state:` loads a training state weights-only and samples from an ephemeral snapshot (nothing
durable is written); `sampler:<path>` uses a saved sampler. Model-ending failures (malformed tool
call, output-token limit, repeated tool-contract errors, ...) are recorded as failures; infrastructure
errors are never turned into a return. A quarter interrupted by the operator or by a provider stall,
or rejected by the strict parser where the reference parser recovers it, can be continued from its
failing day with the same policy, once its source quarters have stopped (a source without
`quarter-result.json` is treated as interrupted; `--dry-run` lists what would be resumed without
sampling):

```bash
uv run python scripts/campaign.py resume --policy base --source runs/val/base/s5* \
    --out runs/val/resume-base --tag val-resume-base --top-p 0.95 --top-k 20 --think-budget 57344
```

At list price a Qwen3.8 quarter cost about $20-27 in this campaign and took 7-8 hours of wall
clock (quarters run concurrently, `--concurrency`). `scripts/costs.py` prices every quarter
under `$SMB_RUNS_DIR` from its recorded token counts.

## 2. Run MaxRL training

Groups of 4 continuations start from the exact day-d state of an archived frontier TRAIN quarter
(replayed through the deterministic engine) and run to day 90. Reward is 1 if the graded return
beats that frontier quarter's own return from the same state, else 0; advantages are MaxRL's
grouped estimator, the loss is CISPO. Look at the states first; `--dry-run` builds the same
config and writes every step's batch plan without contacting Tinker:

```bash
RL="--parent-state base --steps 4 --groups-per-batch 4 --group-size 4 --learning-rate 3e-5 \
    --num-substeps 2 --val-every 2 --val-samples 2 --val-skip-first --top-p 0.95 --top-k 20 \
    --train-keep 0.34 --curriculum 60-85,60-85,60-85,60-85"

uv run python scripts/rl.py --arm cispo_maxrl_binary $RL --log-path runs/rl/maxrl-dry --dry-run
.venv/bin/python scripts/rl.py --arm cispo_maxrl_binary $RL --log-path runs/rl/maxrl &
.venv/bin/python scripts/rl_guard.py runs/rl/maxrl $! --max-usd 2800   # optional stop guard
```

The dollar-return arm is `--arm cispo_rloo` (reward = return in k$; a model-ending failure gets
that world's advance-only floor from `runs/noop-floor/summary.json`). Both arms see the same
states at the same step. A continuation run uses `--parent-state tinker://.../weights/final`,
`--state-offset N` for fresh states and an earlier-day `--curriculum` such as `35-59,35-59,10-34,10-34`.
`--train-keep 0.34` subsamples each trajectory's training datums without bias (advantages are
rescaled by n/k; see `DaySubsample`). The run saves no periodic checkpoints; the final save is a
training state, recorded in `runs/rl/maxrl/checkpoints.jsonl`. In-session VAL quarters land in
`runs/rl/maxrl/validation/`; evaluate the final state with step 1.

## 3. Select on VAL and report sealed TEST against frontier replays

Primary metric: the mean over the three VAL worlds of the per-world mean graded return, with each
model-failure quarter scored at -$25,000 and infrastructure failures excluded. List candidates
simplest first; the top candidate has the highest primary, any candidate within one pooled SE of
it is a near-tie, and the simplest near-tie is selected. A resumed quarter supersedes its source.

```bash
uv run python scripts/select_val.py \
    --candidate "base=val/base,val/resume-base" \
    --candidate "maxrl-final=val/maxrl-final" \
    --frontier-dir frontier-val-reference --json runs/selection.json
```

Only after selection, run the selected configuration and the matched arms on the sealed TEST
worlds, replay the archived TEST frontier runs, and report:

```bash
export SMB_WORLD_REGISTRY=$PWD/runs/worlds-with-test.json
uv run python scripts/campaign.py eval --policy base --worlds test $EVAL --final-test-access \
    --out runs/test/base --tag test-base --purpose "sealed TEST: selected configuration"
uv run python scripts/campaign.py eval --policy state:tinker://<training-run>/weights/final \
    --worlds test $EVAL --final-test-access --out runs/test/maxrl-final --tag test-maxrl --purpose "sealed TEST: MaxRL"
uv run python scripts/frontier_hurdles.py frontier-test-index.json \
    --out runs/frontier-test-reference --final-test-access
uv run python scripts/report_test.py --arm "base=test/base" --arm "maxrl-final=test/maxrl-final" \
    --frontier-dir frontier-test-reference --json runs/test-report.json
```

TEST worlds are admitted only by a registry that lists them and only with `--final-test-access`.

## 4. Regenerate figures

```bash
P="uv run --extra plots python scripts/plots.py"
$P rl rl/maxrl rl/cispo                                   # RL arms on matched frontier states
$P val --candidate base=val/base,val/resume-base --candidate maxrl-final=val/maxrl-final \
    --rl maxrl-final --frontier-dir frontier-val-reference
$P test --arm base=test/base --arm maxrl-final=test/maxrl-final --selected base \
    --val base=val/base,val/resume-base --val maxrl-final=val/maxrl-final --frontier-dir frontier-test-reference
$P sft sft-econ                                           # SFT NLL curve
$P panel panel-d60                                        # branch-eval panel (below)
$P bridge panel-d60 <config> [<config> ...]               # where the branch margin comes from
```

The four figures in the training report come from one config that maps labels to run directories
(`docs/story-config.json` is the one used for the campaign's run tree) and from a snapshot of the
public leaderboard's scored runs on the validation and test businesses
(`data/leaderboard-runs-b5-b6.json`, SMB Bench dataset revision `4ca0b55`, CC BY 4.0). Our replays
of the nine frontier models in the first leaderboard release reproduce those scores to the cent
(80 of 80 test quarters, 81 of 81 validation quarters).

```bash
uv run --with pandas --with pyarrow python scripts/leaderboard_snapshot.py \
    --out data/leaderboard-runs-b5-b6.json                # refresh from the current leaderboard
uv run --extra plots python scripts/story_figures.py --config docs/story-config.json \
    --out docs/figures --json docs/figures/numbers.json
```

Figures are written to `$SMB_RUNS_DIR/figures`. A branch-eval panel is a directory of
`scripts/campaign.py branch-eval --day 60 --k 2 --out runs/panel-d60/<config> ...` runs.

## 5. Export the final model

Tinker trains a LoRA adapter (rank 32) and keeps it in Tinker storage. `scripts/export_model.py`
follows the Tinker cookbook deployment path: download the training checkpoint, convert it to a PEFT
adapter for vLLM or SGLang, and merge it into `Qwen/Qwen3.8-27B` as a standalone bf16 model.

```bash
export TINKER_API_KEY=...
uv run python scripts/export_model.py --tinker-path tinker://<training-run>/weights/final \
    --out exports/final --steps download,adapter          # adapter only, a few GB
uv run python scripts/export_model.py --tinker-path tinker://<training-run>/weights/final \
    --out exports/final --steps merge                     # needs ~120 GB free
uv run python scripts/export_model.py ... --hf-repo <org>/<name>   # optional private upload
```

Downloading a training checkpoint starts a short Tinker session, so the account must be in good
standing. The merge downloads the 54 GB base model and writes a model of the same size. The export
of the reported checkpoint has not been run yet (see the training report's limits).

## SFT warm start (negative result)

The warm start fine-tunes Qwen3.6-35B-A3B (the backbone it was run on) on day-packed public actions
from top-reward frontier TRAIN runs, then is evaluated like any other policy:

```bash
uv run python scripts/build_sft_corpus.py --selection selected-archives.json --holdout holdout.json \
    --out runs/sft-corpus
uv run python scripts/econ_sft.py --data runs/sft-corpus --log-path runs/sft-econ --dry-run
uv run python scripts/econ_sft.py --data runs/sft-corpus --log-path runs/sft-econ
uv run python scripts/campaign.py eval --policy state:tinker://<sft-run>/weights/final --backbone qwen3.6 ...
```

## Reproduction notes

- The engine is deterministic given the actions. With this package, the advance-only quarters of
  all 15 TRAIN/VAL worlds and replays of archived frontier runs return the campaign's graded
  returns to the cent. Policy quarters sample at temperature 1, so their returns vary run to run.
- The reported MaxRL checkpoint (four steps, curriculum 60-85) was trained under an earlier
  revision of the system prompt that is not shipped; its VAL and TEST evaluations used
  `prompts/buyer_system.txt`. The MaxRL command above trains with the shipped prompt, as the
  campaign's continuation run did. The original SFT corpus was also rendered with an earlier
  prompt; rebuilding it with the shipped prompt changes token counts, not the supervised targets.
- Training uses the strict parser (a malformed call ends the quarter with the failure reward);
  evaluation uses the reference parser. `quarter-result.json` records `parse_mode` and the number
  of reference-parser recoveries.
- Every quarter directory holds its hash-bound `plan.json` and `adoption.json`, the compact
  `journal.jsonl.gz`, sampled tokens/logprobs/sequence ids (`actions.jsonl.gz`), the engine's own
  outputs and grade under `engine/`, and `quarter-result.json`.

## Layout

```
prompts/buyer_system.txt     final system prompt (replaces the engine's v11 prompt; v11 tools kept)
data/engine-b17b864.json     SHA-256 of every engine file the harness checks
data/leaderboard-runs-b5-b6.json  leaderboard scored runs on the validation and test businesses
engine-runtime/              pyproject for the engine interpreter
src/smb_buyer/               harness, RL environment, algorithms, SFT packing, analysis
scripts/                     command-line entry points (above), export_model.py and smoke_test.sh
scripts/evidence/            transcript and record mining behind docs/evidence (set SMB_RUNS_DIR)
tests/test_algorithms.py     offline tests of the MaxRL/RLOO advantage seam
docs/training_report.md      training report; docs/figures/ and docs/story-config.json reproduce its figures
```
