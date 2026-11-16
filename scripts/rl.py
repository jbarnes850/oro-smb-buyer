"""Full-buyer native RL with the official Cookbook runner (``rl/train.main``).

Each group is G continuations of the original 90-day buyer from one exact day-d
state of an archived frontier TRAIN quarter (order-matched prefix replay through the
deterministic engine), each ending with the original independently graded 90-day
return. Arms (advantage and loss from ``smb_buyer.algorithms``):

  cispo_maxrl_binary  reward 1 if the return beats that frontier quarter's own graded
                      return from the same state, else 0 (model failure = 0); MaxRL
                      grouped advantage, CISPO loss
  cispo_rloo          reward = return in k$ (model failure = that world's advance-only
                      floor); RLOO advantage, CISPO loss
  sr_rloo             same reward as cispo_rloo with the StableReinforce loss

Branch days follow a reverse curriculum (``--curriculum``). Each step draws its
worlds, frontier sources and days deterministically from the step index (shifted by
``--state-offset``), so different arms see the same states at the same step.
Storage: no periodic or rolling saves; the final save is state-only. Validation
quarters run in the same session from ephemeral snapshots (no durable writes).
``--dry-run`` writes every step's batch plan without contacting Tinker.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

import chz
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.rl import train
from tinker_cookbook.rl.rollout_strategy import RetryOnFailure
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder

from smb_buyer import algorithms, campaign, paths, storage
from smb_buyer.backbones import BACKBONES, DEFAULT_BACKBONE

# Set once in main() from the command line.
SETTINGS = {"reasoning": "on", "prompt": None, "backbone": DEFAULT_BACKBONE, "noop": None}
DEFAULT_CURRICULUM = "60-85,60-85,35-59,35-59,10-34,0-9"
STATE_SEED = 20261114


class FloorObjective(campaign.DollarObjective):
    """Valid quarter: return in k$. Model-ending failure: that world's advance-only floor."""

    def __init__(self, world):
        super().__init__(failure_kusd=SETTINGS["noop"][world]["return_cents"] / 100000.0)
        self.declaration.update(
            model_failure_reward_rule="advance-only (do-nothing) graded return of the same world, k$",
            world=world,
        )


class HurdleObjective(campaign.DollarObjective):
    """Binary success: graded return strictly above the hurdle (the frontier's own
    realized return from the identical branch state). Model failure = 0."""

    def __init__(self, world, hurdle_cents):
        super().__init__(failure_kusd=0.0)
        self.hurdle_cents = int(hurdle_cents)
        self.declaration.update(
            valid_quarter_reward="1 if business_return_cents > hurdle_cents else 0",
            model_failure_reward=0,
            hurdle_cents=self.hurdle_cents,
            world=world,
        )

    def __call__(self, completion):
        return 1.0 if completion.return_cents > self.hurdle_cents else 0.0

    def failure(self, model_failure):
        return 0.0


class QuarterGroup(EnvGroupBuilder):
    def __init__(self, *, world, group_dir, group_size, formatter, branch_day, policy_id, objective, frontier):
        self.world, self.group_dir, self.group_size = world, Path(group_dir), group_size
        self.formatter, self.branch_day = formatter, branch_day
        self.policy_id = policy_id
        self.objective = objective
        self.frontier = frontier  # bank row of the frontier quarter this group branches from
        self.created, self.attempt = [], 0

    async def make_envs(self):
        prefix = (
            campaign.load_frontier_prefix(self.frontier["transcript"], self.branch_day)
            if self.branch_day
            else None
        )
        envs = []
        for k in range(self.group_size):
            output = self.group_dir / f"a{self.attempt}-r{k}"
            spec = campaign.QuarterSpec(
                world=self.world,
                output=output,
                run_id=f"{self.group_dir.name}-a{self.attempt}-r{k}",
                policy_id=self.policy_id,
                policy_label="label:rl-ephemeral",
                purpose="native RL rollout",
                prefix=prefix,
                reasoning=SETTINGS["reasoning"],
                prompt_path=SETTINGS["prompt"],
                backbone=SETTINGS["backbone"],
            )
            output.mkdir(parents=True, exist_ok=False)
            journal = campaign.Journal(output / "journal.jsonl.gz")
            env = campaign.make_quarter_env(
                spec,
                self.formatter,
                objective=self.objective,
                failure_objective=self.objective.failure,
                journal=journal,
            )
            env._campaign_journal, env._campaign_output, env._campaign_started = journal, output, time.time()
            if prefix is not None:
                campaign.write(
                    output / "prefix.json",
                    {k: prefix[k] for k in ("day", "source")} | {"records": len(prefix["records"])},
                )
            envs.append(env)
        self.attempt += 1
        self.created.extend(envs)
        return envs

    async def compute_group_rewards(self, trajectory_group, env_group):
        rows = []
        for trajectory, env in zip(trajectory_group, env_group, strict=True):
            usage = campaign.trajectory_usage(trajectory, SETTINGS["backbone"])
            if env.model_failure is not None:
                row = {"status": "model_failure", "failure": env.model_failure, "return_cents": None}
            else:
                row = {
                    "status": "valid",
                    "return_cents": env.completion.return_cents,
                    "terminal_status": env.completion.terminal_status,
                    "grade": env.completion.grade_ref,
                }
            frontier = {k: self.frontier[k] for k in ("trial_id", "model", "task_id", "reward", "hurdle_cents")}
            row.update(
                world=self.world,
                branch_day=self.branch_day,
                prefix_source=None,
                frontier=frontier,
                beat_hurdle=row.get("return_cents") is not None
                and row["return_cents"] > self.frontier["hurdle_cents"],
                reward=env.terminal_reward,
                elapsed_seconds=time.time() - env._campaign_started,
                **usage,
            )
            env._campaign_journal.close()
            campaign.write_actions(trajectory, env._campaign_output / "actions.jsonl.gz")
            campaign.write(env._campaign_output / "quarter-result.json", row)
            if row["status"] == "valid":
                campaign.compress_engine_output(env._campaign_output)
            rows.append(row)
        index = len(list(self.group_dir.glob("group-result-*.json")))
        campaign.write(self.group_dir / f"group-result-{index}.json", rows)
        return [
            (
                0.0,
                {
                    "valid": float(r["status"] == "valid"),
                    "return_kusd": (r["return_cents"] or 0) / 1e5,
                    "beat_frontier": float(bool(r["beat_hurdle"])),
                    "margin_vs_frontier_kusd": (
                        (r["return_cents"] - r["frontier"]["hurdle_cents"]) / 1e5
                        if r["return_cents"] is not None
                        else 0.0
                    ),
                },
            )
            for r in rows
        ]

    def logging_tags(self):
        return [self.world.split("_experience")[0], f"branch{self.branch_day}"]

    async def cleanup(self):
        for env in self.created:
            try:
                await env.close()
            except Exception:  # noqa: BLE001
                pass
            journal = getattr(env, "_campaign_journal", None)
            if journal is not None:
                journal.close()


class CampaignDataset(RLDataset):
    """Groups branch from frontier TRAIN states; hurdle = that frontier's own return."""

    def __init__(self, *, steps, groups_per_batch, group_size, log_path, policy_id, binary,
                 curriculum, state_offset, frontier_index, hurdles_dir, formatter=None):
        self.steps, self.groups_per_batch, self.group_size = steps, groups_per_batch, group_size
        self.log_path, self.policy_id, self.binary = Path(log_path), policy_id, binary
        self.curriculum, self.state_offset = curriculum, state_offset
        self.train_worlds = campaign.worlds_in_split("train")
        self.bank = campaign.frontier_bank(frontier_index, hurdles_dir, order="model")
        self.formatter = formatter or campaign.make_formatter(SETTINGS["reasoning"], SETTINGS["backbone"])

    def __len__(self):
        return self.steps

    def get_batch(self, index):
        # The state offset shifts which worlds/frontier sources/days a step draws
        # (0 = the original draws); the branch-day window follows this run's step index.
        draw = index + self.state_offset
        rng = random.Random(STATE_SEED + draw)
        low, high = self.curriculum.get(index, self.curriculum[max(self.curriculum)])
        builders, plan = [], []
        for j in range(self.groups_per_batch):
            world = self.train_worlds[(draw * self.groups_per_batch + j) % len(self.train_worlds)]
            options = self.bank.get(world, [])
            if not options:
                continue
            source = options[(draw * 7 + j * 3) % len(options)]
            day = rng.randint(low, high)
            objective = HurdleObjective(world, source["hurdle_cents"]) if self.binary else FloorObjective(world)
            name = (
                f"g{j}-{world.split('_experience')[0]}-{world.split('long_')[1][:4]}"
                f"-d{day}-{source['model'][:10]}"
            )
            group_dir = self.log_path / "rollouts" / f"step{index:03d}" / name
            group_dir.mkdir(parents=True, exist_ok=True)
            builders.append(
                QuarterGroup(
                    world=world,
                    group_dir=group_dir,
                    group_size=self.group_size,
                    formatter=self.formatter,
                    branch_day=day,
                    policy_id=self.policy_id,
                    objective=objective,
                    frontier=source,
                )
            )
            plan.append(
                {
                    "world": world,
                    "branch_day": day,
                    "frontier_trial": source["trial_id"],
                    "frontier_model": source["model"],
                    "hurdle_cents": source["hurdle_cents"],
                    "objective": "binary-beat-frontier" if self.binary else "kusd-return",
                }
            )
        campaign.write(self.log_path / "rollouts" / f"step{index:03d}" / "batch-plan.json", plan)
        return builders


@chz.chz
class CampaignDatasetBuilder(RLDatasetBuilder):
    steps: int
    groups_per_batch: int
    group_size: int
    log_path: str
    policy_id: str
    binary: bool
    curriculum: tuple[tuple[int, int], ...]
    state_offset: int
    frontier_index: str
    hurdles_dir: str

    async def __call__(self):
        dataset = CampaignDataset(
            steps=self.steps,
            groups_per_batch=self.groups_per_batch,
            group_size=self.group_size,
            log_path=self.log_path,
            policy_id=self.policy_id,
            binary=self.binary,
            curriculum=dict(enumerate(self.curriculum)),
            state_offset=self.state_offset,
            frontier_index=self.frontier_index,
            hurdles_dir=self.hurdles_dir,
        )
        return dataset, None


@chz.chz
class CampaignRLConfig(train.Config):
    def records_stability_info(self) -> bool:
        return True  # identical native evidence for every arm


class DaySubsample:
    """Unbiased per-trajectory subsampling of training datums (declared estimator change).

    The official assembler turns each trajectory into datums, one per run of turns whose
    observations extend each other (in practice about one or two per simulated day). Every
    datum of a trajectory carries the same scalar advantage. For each trajectory with n
    datums we keep k = max(1, round(keep * n)) chosen uniformly without replacement and
    multiply their advantages by n / k, so each datum enters with probability k / n and
    weight n / k: the expected policy-gradient (linear in advantages for CISPO and MaxRL)
    is unchanged while forward_backward tokens, the dominant cost, drop by about 1 - keep.
    Only ``train.assemble_training_data`` is rebound, and always restored.
    """

    def __init__(self, keep, log_path, seed=20261115):
        self.keep, self.log_path, self.rng = float(keep), Path(log_path), random.Random(seed)
        self.calls = 0

    def __call__(self, groups, advantages, include_provenance=True):
        import dataclasses
        from collections import defaultdict

        from tinker import TensorData

        data, meta = self.official(groups, advantages, include_provenance=include_provenance)
        if self.keep >= 1.0:
            return data, meta
        by_traj = defaultdict(list)
        for i, m in enumerate(meta):
            by_traj[(m["group_idx"], m["traj_idx"])].append(i)
        kept_data, kept_meta = [], []
        tokens_before = tokens_after = 0
        for key in sorted(by_traj):
            idx = by_traj[key]
            n = len(idx)
            k = max(1, round(self.keep * n))
            chosen = sorted(self.rng.sample(idx, k))
            scale = n / k
            for i in idx:
                tokens_before += data[i].model_input.length
            for i in chosen:
                d = data[i]
                adv = d.loss_fn_inputs["advantages"].to_torch() * scale
                kept_data.append(
                    dataclasses.replace(
                        d, loss_fn_inputs={**d.loss_fn_inputs, "advantages": TensorData.from_torch(adv)}
                    )
                )
                kept_meta.append(meta[i])
                tokens_after += d.model_input.length
        with (self.log_path / "train-subsample.jsonl").open("a") as f:
            f.write(
                json.dumps(
                    {
                        "call": self.calls,
                        "keep": self.keep,
                        "datums_before": len(data),
                        "datums_after": len(kept_data),
                        "tokens_before": tokens_before,
                        "tokens_after": tokens_after,
                        "trajectories": len(by_traj),
                    }
                )
                + "\n"
            )
        self.calls += 1
        return kept_data, kept_meta

    def __enter__(self):
        self.official = train.assemble_training_data
        train.assemble_training_data = self
        return self

    def __exit__(self, *exc):
        train.assemble_training_data = self.official
        return False


class ValidationLauncher:
    """Official SamplingClientEvaluator hook: start VAL quarters in the background.

    Returns immediately so training is not blocked; each quarter writes its own
    quarter-result.json under log_path/validation/eval*/.
    """

    tasks: list = []

    def __init__(self, log_path, samples, skip_first=False, final_call=None):
        self.log_path, self.samples, self.calls = Path(log_path), samples, 0
        self.skip_first = skip_first
        # The cookbook evaluates at every eval_every step and once more at end_step,
        # then leaves its ``with tinker.ServiceClient(...)`` block, which closes the
        # session: an ephemeral snapshot's in-flight requests then hang. The final
        # weights are saved as a state, so the final VAL runs standalone from that
        # state instead (scripts/campaign.py eval --policy state:<path>).
        self.final_call = final_call

    async def __call__(self, sampling_client):
        from tinker_cookbook.completers import TinkerTokenCompleter

        step = self.calls
        self.calls += 1
        if step == 0 and self.skip_first:
            return {"val/launched": 0.0}  # parent evaluated separately
        if step == self.final_call:
            return {"val/launched": 0.0, "val/final_deferred_to_standalone": 1.0}
        formatter = campaign.make_formatter(SETTINGS["reasoning"], SETTINGS["backbone"])
        policy = TinkerTokenCompleter(
            sampling_client, max_tokens=campaign.MAX_OUTPUT, temperature=1.0, context_window=campaign.CONTEXT
        )
        out = self.log_path / "validation" / f"eval{step:02d}"
        worlds = campaign.worlds_in_split("validation")
        for world in worlds:
            for sample in range(self.samples):
                objective = FloorObjective(world)
                spec = campaign.QuarterSpec(
                    world=world,
                    output=out / f"{world}__s{sample}",
                    run_id=f"val-eval{step}-{world}-s{sample}",
                    policy_id="rl-ephemeral",
                    policy_label="label:rl-ephemeral",
                    purpose="in-session VAL selection evaluation",
                    reasoning=SETTINGS["reasoning"],
                    prompt_path=SETTINGS["prompt"],
                    backbone=SETTINGS["backbone"],
                )
                ValidationLauncher.tasks.append(
                    asyncio.create_task(
                        campaign.run_quarter(
                            spec, policy, formatter, objective=objective, failure_objective=objective.failure
                        )
                    )
                )
        return {"val/launched": float(len(worlds) * self.samples)}


def parse_curriculum(text):
    windows = []
    for window in text.split(","):
        low, high = (int(x) for x in window.split("-"))
        if not 0 <= low <= high <= 89:
            raise SystemExit(f"bad curriculum window {window!r}")
        windows.append((low, high))
    return tuple(windows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=["cispo_maxrl_binary", "cispo_rloo", "sr_rloo"], required=True)
    parser.add_argument("--parent-state", required=True, help="base | tinker:// training state to continue")
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--groups-per-batch", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-substeps", type=int, default=2)
    parser.add_argument("--val-every", type=int, default=3)
    parser.add_argument("--val-samples", type=int, default=1)
    parser.add_argument("--val-skip-first", action="store_true")
    parser.add_argument("--reasoning", choices=["on", "off"], default="on")
    parser.add_argument("--prompt", default=None, help="system prompt file (default: prompts/buyer_system.txt)")
    parser.add_argument("--backbone", choices=sorted(BACKBONES), default=DEFAULT_BACKBONE)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--parse-mode", choices=["strict", "reference"], default="strict",
                        help="strict: malformed call ends the quarter; reference: Qwen serving parser recovery")
    parser.add_argument("--train-keep", type=float, default=1.0,
                        help="per-trajectory datum keep fraction (unbiased, advantages rescaled)")
    parser.add_argument("--state-offset", type=int, default=0,
                        help="shift frontier-state draws by this many steps (fresh states for a continuation run)")
    parser.add_argument("--curriculum", default=DEFAULT_CURRICULUM,
                        help="comma-separated branch-day windows, one per step; the last repeats")
    parser.add_argument("--frontier-index", default=None, help="frontier quarter index (default: SMB_FRONTIER_INDEX)")
    parser.add_argument("--hurdles-dir", default=None,
                        help="frontier replays from scripts/frontier_hurdles.py (default: $SMB_RUNS_DIR/frontier-hurdles)")
    parser.add_argument("--noop-floor", default=None,
                        help="advance-only summary from scripts/campaign.py noop (default: $SMB_RUNS_DIR/noop-floor/summary.json)")
    parser.add_argument("--registry", default=None, help="world registry (sets SMB_WORLD_REGISTRY)")
    parser.add_argument("--owned-checkpoints", default=None,
                        help="optional owned-checkpoint budget registry; the final save waits until it fits")
    parser.add_argument("--skip-final-save", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="write every step's batch plan and exit; no Tinker calls")
    args = parser.parse_args()
    if args.registry:
        os.environ["SMB_WORLD_REGISTRY"] = str(Path(args.registry).resolve())
    os.environ["SMB_PARSE_MODE"] = args.parse_mode
    if not args.dry_run:
        paths.require_tinker_key()
    campaign.set_decoding(args.top_p, args.top_k)
    curriculum = parse_curriculum(args.curriculum)
    frontier_index = str(Path(args.frontier_index).resolve() if args.frontier_index else paths.frontier_index())
    hurdles_dir = str(Path(args.hurdles_dir).resolve() if args.hurdles_dir else paths.frontier_hurdles())
    noop_path = Path(args.noop_floor) if args.noop_floor else paths.noop_floor()
    SETTINGS.update(
        reasoning=args.reasoning,
        backbone=args.backbone,
        prompt=str(Path(args.prompt).resolve()) if args.prompt else None,
        noop=json.loads(noop_path.read_text()),
    )
    train_worlds = campaign.worlds_in_split("train")
    val_worlds = campaign.worlds_in_split("validation")
    missing = [w for w in train_worlds + val_worlds if w not in SETTINGS["noop"]]
    if missing:
        sys.exit(f"advance-only floor missing for {missing}; run scripts/campaign.py noop first")
    log_path = Path(args.log_path)
    log_path.mkdir(parents=True, exist_ok=False)
    loss_config = (
        {"kl_coeff": 0.0, "kl_estimator": "k3"}
        if args.arm.startswith("sr")
        else {"clip_low_threshold": 0.0, "clip_high_threshold": 4.0}
    )
    launch = {
        "arm": args.arm,
        "reasoning": args.reasoning,
        "backbone": args.backbone,
        "prompt": campaign.ref(SETTINGS["prompt"] or paths.PROMPT),
        "decoding": dict(campaign.DECODING),
        "curriculum": {str(i): list(w) for i, w in enumerate(curriculum)},
        "train_keep": args.train_keep,
        "parse_mode": args.parse_mode,
        "loss_fn_config": loss_config,
        "advantage": algorithms.ARMS[args.arm][1],
        "parent_state": args.parent_state,
        "args": vars(args),
        "train_worlds": train_worlds,
        "val_worlds": val_worlds,
        "frontier_index": campaign.ref(frontier_index),
        "hurdles_dir": hurdles_dir,
        "objective": (
            "1 if graded return beats the frontier's own return from the same state, else 0"
            if args.arm == "cispo_maxrl_binary"
            else "k$ graded 90-day return; model failure = same-world advance-only floor"
        ),
        "noop_floor_k$": {w: SETTINGS["noop"][w]["return_cents"] / 1e5 for w in train_worlds + val_worlds},
        "dry_run": args.dry_run,
        "started_at": time.time(),
    }
    cadence = [s for s in range(args.steps) if args.val_every > 0 and s % args.val_every == 0]
    validator = ValidationLauncher(log_path, args.val_samples, skip_first=args.val_skip_first,
                                   final_call=len(cadence))  # call index of the end_step evaluation
    config = CampaignRLConfig(
        log_path=str(log_path),
        model_name=BACKBONES[args.backbone][0],
        recipe_name=f"full-buyer-native-rl-{args.arm}",
        renderer_name=BACKBONES[args.backbone][2][args.reasoning],
        load_checkpoint_path=None if args.parent_state == "base" else args.parent_state,
        dataset_builder=CampaignDatasetBuilder(
            steps=args.steps,
            groups_per_batch=args.groups_per_batch,
            group_size=args.group_size,
            log_path=str(log_path),
            policy_id=f"rl:{args.arm}",
            binary=args.arm == "cispo_maxrl_binary",
            curriculum=curriculum,
            state_offset=args.state_offset,
            frontier_index=frontier_index,
            hurdles_dir=hurdles_dir,
        ),
        learning_rate=args.learning_rate,
        max_tokens=campaign.MAX_OUTPUT,
        temperature=1.0,
        lora_rank=32,
        num_substeps=args.num_substeps,
        save_every=0,
        rolling_save_every=0,
        eval_every=args.val_every,
        evaluator_builders=[lambda: validator],
        rollout_error_tolerance=RetryOnFailure(max_retries=2),
        remove_constant_reward_groups=False,
        num_groups_to_log=0,
        max_steps=args.steps,
    )
    config = algorithms.configure_algorithm(
        config, arm=args.arm, loss_fn_config=loss_config, reference_kl_coef=0.0, reference_kl_discount_factor=0.0
    )
    launch["loss"] = config.loss_fn
    (log_path / "launch.json").write_text(json.dumps(launch, indent=1, default=str))
    if args.dry_run:
        # Same config and dataset builder as a real run; only the trainer is not started.
        dataset, _ = asyncio.run(config.dataset_builder())
        for step in range(len(dataset)):
            dataset.get_batch(step)
            plan = log_path / "rollouts" / f"step{step:03d}" / "batch-plan.json"
            print(json.dumps({"step": step, "plan": str(plan)}), flush=True)
        return
    checkpoint_utils.CheckpointManager = storage.state_only_final(
        checkpoint_utils, backbone=args.backbone, owned_registry=args.owned_checkpoints, skip=args.skip_final_save
    )

    async def run():
        with algorithms.algorithm_context(config, arm=args.arm), DaySubsample(args.train_keep, log_path):
            await train.main(config)
        # Training's ServiceClient is closed now; unfinished in-session VAL quarters
        # cannot get another response. Record and cancel them instead of hanging.
        stranded = [task for task in ValidationLauncher.tasks if not task.done()]
        if stranded:
            (log_path / "validation" / "stranded-at-session-close.json").write_text(
                json.dumps(
                    {
                        "at": time.time(),
                        "count": len(stranded),
                        "note": "in-session VAL quarters still sampling when training's ServiceClient closed; cancelled",
                    },
                    indent=1,
                )
            )
            for task in stranded:
                task.cancel()
        if ValidationLauncher.tasks:
            await asyncio.gather(*ValidationLauncher.tasks, return_exceptions=True)

    asyncio.run(run())


if __name__ == "__main__":
    try:
        main()
    except paths.ConfigError as error:
        sys.exit(f"configuration: {error}")
