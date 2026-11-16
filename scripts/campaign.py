"""Full-buyer evaluation driver: original 90-day quarters for one policy.

  build-registry  hash the TRAIN/VAL (and optionally TEST) world directories into a registry
  check-setup     verify the engine source, engine interpreter, tools and registry (no sampling)
  noop            advance-only quarters (zero model calls): the economic floor per world
  eval            N original 90-day quarters per world for one policy, concurrently
  branch-eval     K continuations from a shared archived frontier state per TRAIN world
  resume          continue strict-parser false failures or provider stalls from their failing day
  status          summarize running/finished quarters under a directory

Policies: ``base`` (frozen backbone), ``sampler:<tinker path>`` (durable sampler),
``state:<tinker path>`` (load a training state weights-only, then sample from an
ephemeral snapshot; nothing durable is written). The Tinker SDK reads
TINKER_API_KEY from the environment.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from smb_buyer import campaign, paths
from smb_buyer.backbones import BACKBONES, DEFAULT_BACKBONE


def ledger(row):
    path = paths.runs_dir() / "ledger.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps({"t": time.time(), **row}, default=str) + "\n")


class RetryingCompleter:
    """Official TinkerTokenCompleter with bounded resubmission of failed requests.

    A request that raised returned no sample, so resubmitting the identical prompt
    draws a fresh sample from the same policy; attempts are logged.
    """

    def __init__(self, inner, *, attempts=4, timeout=2400, log=None):
        self.inner, self.attempts, self.timeout, self.log = inner, attempts, timeout, log

    async def __call__(self, model_input, stop, *, max_tokens=None):
        delay = 15
        for attempt in range(1, self.attempts + 1):
            try:
                return await asyncio.wait_for(
                    self.inner(model_input, stop, max_tokens=max_tokens), self.timeout
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001
                if self.log:
                    self.log({"event": "sample_retry", "attempt": attempt, "error": repr(error)[:300]})
                if attempt == self.attempts:
                    raise
                await asyncio.sleep(delay)
                delay *= 2


THINK_STOP_TEXT = (
    "\n\nI have reasoned for a long time. I must stop here and act on what I have already worked out, "
    "keeping this response well within its output limit.\n</think>\n\n"
)


class ThinkingBudget:
    """Evaluation-only reasoning cap (Qwen's documented early-stop method; off by default).

    The original harness ends the whole quarter when one response reaches the
    65,536-token output limit. Phase 1 samples up to ``budget`` tokens. If it stops
    on length with the reasoning still open, the fixed THINK_STOP_TEXT tokens are
    appended and the answer is sampled with the remaining allowance; a length stop
    after </think> continues unchanged (the same distribution as one long sample).
    Injected tokens carry logprob 0.0 and are logged; these traces are never
    training data.
    """

    def __init__(self, inner, tokenizer, budget, *, total=None, log=None):
        self.inner, self.budget, self.log = inner, int(budget), log
        self.total = int(total or campaign.MAX_OUTPUT)
        self.think_end = tokenizer.encode("</think>", add_special_tokens=False)
        self.inject = tokenizer.encode(THINK_STOP_TEXT, add_special_tokens=False)
        if len(self.think_end) != 1 or self.think_end[0] not in self.inject:
            raise ValueError("thinking budget needs a single-token </think> inside the stop text")
        if not 0 < self.budget < self.total - len(self.inject) - 1024:
            raise ValueError("thinking budget must leave room for an answer")

    async def __call__(self, model_input, stop, *, max_tokens=None):
        from tinker_cookbook.completers import TokensWithLogprobs

        total = min(self.total, max_tokens or self.total)
        first = await self.inner(model_input, stop, max_tokens=min(self.budget, total))
        if first.stop_reason != "length" or len(first.tokens) >= total:
            return first
        closed = self.think_end[0] in first.tokens
        bridge = [] if closed else list(self.inject)
        prefix = tinker_model_input(model_input.to_ints() + list(first.tokens) + bridge)
        second = await self.inner(prefix, stop, max_tokens=total - len(first.tokens) - len(bridge))
        if self.log:
            self.log(
                {
                    "event": "thinking_budget",
                    "injected": not closed,
                    "prompt_tokens": model_input.length,
                    "phase1_tokens": len(first.tokens),
                    "inject_at": len(first.tokens),
                    "inject_tokens": len(bridge),
                    "phase2_tokens": len(second.tokens),
                    "phase2_stop": second.stop_reason,
                    "sequence_ids": [first.sequence_id, second.sequence_id],
                }
            )
        return TokensWithLogprobs(
            tokens=list(first.tokens) + bridge + list(second.tokens),
            maybe_logprobs=list(first.logprobs) + [0.0] * len(bridge) + list(second.logprobs),
            stop_reason=second.stop_reason,
            sequence_id=second.sequence_id,
            prompt_cache_hit_tokens=(first.prompt_cache_hit_tokens or 0)
            + (second.prompt_cache_hit_tokens or 0),
        )


def tinker_model_input(ids):
    import tinker

    return tinker.ModelInput.from_ints(ids)


def budgeted(policy, formatter, args, log):
    budget = getattr(args, "think_budget", 0) or 0
    if not budget:
        return policy
    if args.reasoning != "on":
        raise SystemExit("--think-budget applies to reasoning-on policies only")
    return ThinkingBudget(policy, formatter.renderer.tokenizer, budget, log=log)


async def make_policy(spec, *, keep, backbone=DEFAULT_BACKBONE):
    """Return (TokenCompleter, label). ``keep`` holds clients alive for the session."""
    import tinker
    from tinker_cookbook.completers import TinkerTokenCompleter

    paths.require_tinker_key()
    service = tinker.ServiceClient()
    keep.append(service)
    if spec == "base":
        client = await service.create_sampling_client_async(base_model=BACKBONES[backbone][0])
        label = None
    elif spec.startswith("sampler:"):
        path = spec.split(":", 1)[1]
        client = await service.create_sampling_client_async(model_path=path)
        label = path
    elif spec.startswith("state:"):
        path = spec.split(":", 1)[1]
        trainer = await service.create_training_client_from_state_async(path)
        keep.append(trainer)
        client = await trainer.save_weights_and_get_sampling_client_async()
        label = "label:ephemeral-from:" + path
    else:
        raise ValueError("policy must be base | sampler:<path> | state:<path>")
    keep.append(client)
    completer = TinkerTokenCompleter(
        client, max_tokens=campaign.MAX_OUTPUT, temperature=1.0, context_window=campaign.CONTEXT
    )
    return completer, label


async def run_eval(args):
    out = Path(args.out).absolute()
    out.mkdir(parents=True, exist_ok=False)
    keep = []
    completer, label = await make_policy(args.policy, keep=keep, backbone=args.backbone)
    formatter = campaign.make_formatter(args.reasoning, args.backbone)
    objective = campaign.DollarObjective()
    retry_log = campaign.Journal(out / "sampling-retries.jsonl.gz")
    policy = RetryingCompleter(completer, log=retry_log)
    policy = budgeted(policy, formatter, args, retry_log)
    args.worlds = campaign.expand_worlds(args.worlds)
    specs = []
    for world in args.worlds:
        for sample in range(args.samples):
            specs.append(
                campaign.QuarterSpec(
                    world=world,
                    output=out / f"{world}__s{sample}",
                    run_id=f"{args.tag}-{world}-s{sample}",
                    policy_id=f"{args.tag}:{args.policy}",
                    policy_label=label,
                    purpose=args.purpose,
                    final_test_access=args.final_test_access,
                    reasoning=args.reasoning,
                    backbone=args.backbone,
                    prompt_path=args.prompt,
                )
            )
    (out / "eval-config.json").write_text(
        json.dumps(
            {
                "tag": args.tag,
                "policy": args.policy,
                "policy_label": label,
                "worlds": args.worlds,
                "samples": args.samples,
                "purpose": args.purpose,
                "model": BACKBONES[args.backbone][0],
                "parse_mode": os.environ.get("SMB_PARSE_MODE", "strict"),
                "decoding": dict(campaign.DECODING),
                "max_output_tokens": campaign.MAX_OUTPUT,
                "temperature": 1.0,
                "reasoning": args.reasoning,
                "think_budget": getattr(args, "think_budget", 0) or 0,
                "think_stop_text": THINK_STOP_TEXT if getattr(args, "think_budget", 0) else None,
                "prompt": campaign.ref(Path(args.prompt or paths.PROMPT).resolve()),
                "registry": campaign.ref(paths.registry()),
                "started_at": time.time(),
            },
            indent=2,
        )
    )
    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(spec):
        async with semaphore:
            result = await campaign.run_quarter(
                spec, policy, formatter, objective=objective, failure_objective=objective.failure
            )
            ledger(
                {
                    "kind": "eval_quarter",
                    "tag": args.tag,
                    "world": spec.world,
                    "status": result["status"],
                    "return_cents": result.get("return_cents"),
                    "sampling_usd": result.get("sampling_usd"),
                    "model_calls": result.get("model_calls"),
                    "output": str(spec.output),
                }
            )
            keys = ("world", "status", "return_cents", "model_calls", "sampling_usd", "elapsed_seconds")
            print(json.dumps({k: result.get(k) for k in keys}), flush=True)
            return result

    results = await asyncio.gather(*(one(s) for s in specs), return_exceptions=True)
    retry_log.close()
    summary = []
    for spec, result in zip(specs, results):
        if isinstance(result, BaseException):
            summary.append({"world": spec.world, "status": "driver_exception", "error": repr(result)})
        else:
            summary.append({k: v for k, v in result.items() if k not in ("traceback",)})
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    total = sum((r.get("sampling_usd") or 0) for r in summary)
    print(json.dumps({"done": len(summary), "sampling_usd": round(total, 4)}), flush=True)


async def run_branch_eval(args):
    """K continuations from a shared frontier state per TRAIN world; margin vs that frontier."""
    out = Path(args.out).absolute()
    out.mkdir(parents=True, exist_ok=False)
    bank = campaign.frontier_bank(args.frontier_index, args.hurdles_dir, order="hurdle")
    keep = []
    completer, label = await make_policy(args.policy, keep=keep, backbone=args.backbone)
    formatter = campaign.make_formatter(args.reasoning, args.backbone)
    retry_log = campaign.Journal(out / "sampling-retries.jsonl.gz")
    policy = RetryingCompleter(completer, log=retry_log)
    worlds = args.worlds or sorted(w for w in bank if w.startswith("s50"))
    plan, specs = [], []
    for world in worlds:
        options = bank.get(world) or []
        if not options:
            continue
        pick = {"median": options[len(options) // 2], "top": options[-1], "low": options[0]}
        source = pick[args.pick]
        prefix = campaign.load_frontier_prefix(source["transcript"], args.day)
        plan.append(
            {
                "world": world,
                "frontier_trial": source["trial_id"],
                "frontier_model": source["model"],
                "hurdle_cents": source["hurdle_cents"],
                "day": args.day,
            }
        )
        for k in range(args.k):
            spec = campaign.QuarterSpec(
                world=world,
                output=out / f"{world}__k{k}",
                run_id=f"{args.tag}-{world}-k{k}",
                policy_id=f"{args.tag}:{args.policy}",
                policy_label=label,
                purpose=f"branch eval from frontier state day {args.day}",
                prefix=prefix,
                reasoning=args.reasoning,
                backbone=args.backbone,
                prompt_path=args.prompt,
            )
            specs.append((source, spec))
    (out / "branch-plan.json").write_text(
        json.dumps({"args": vars(args), "states": plan}, indent=1, default=str)
    )
    objective = campaign.DollarObjective()
    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(source, spec):
        async with semaphore:
            r = await campaign.run_quarter(
                spec, policy, formatter, objective=objective, failure_objective=objective.failure
            )
            r["hurdle_cents"] = source["hurdle_cents"]
            r["frontier_model"] = source["model"]
            r["margin_cents"] = (
                (r["return_cents"] - source["hurdle_cents"]) if r.get("return_cents") is not None else None
            )
            (spec.output / "branch-result.json").write_text(json.dumps(r, indent=1, default=str))
            keys = ("world", "status", "return_cents", "hurdle_cents", "margin_cents", "model_calls", "sampling_usd")
            print(json.dumps({k: r.get(k) for k in keys}), flush=True)
            return r

    results = await asyncio.gather(*(one(s, sp) for s, sp in specs), return_exceptions=True)
    retry_log.close()
    rows = [r for r in results if isinstance(r, dict)]
    valid = [r for r in rows if r["status"] == "valid"]
    summary = {
        "runs": len(rows),
        "valid": len(valid),
        "model_failures": sum(r["status"] == "model_failure" for r in rows),
        "infra_failures": sum(r["status"] == "infrastructure_failure" for r in rows)
        + sum(not isinstance(r, dict) for r in results),
        "beat_frontier": sum(1 for r in valid if r["margin_cents"] > 0),
        "mean_margin_usd_valid": (sum(r["margin_cents"] for r in valid) / len(valid) / 100) if valid else None,
        "sampling_usd": sum(r.get("sampling_usd") or 0 for r in rows),
    }
    (out / "summary.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1, default=str)
    )
    print(json.dumps(summary), flush=True)


def last_action(source):
    rows = campaign.journal_rows(Path(source) / "actions.jsonl.gz")
    return rows[-1] if rows else None


def failure_day(source):
    """Day of the last model request: the failing (or interrupted) day, resampled live."""
    rows = campaign.journal_rows(Path(source) / "journal.jsonl.gz")
    days = [r["day"] for r in rows if r.get("event") == "model_request"]
    return days[-1] if days else None


# Provider-side stalls: the sampler stopped answering (or refused service), so the
# quarter ended without any model output to judge. Model-caused infrastructure
# terminals (e.g. "Rendered context exceeds declared model bound") are not on this
# list and are never resumed.
PROVIDER_STALL_MARKERS = (
    "deadline exhausted",
    "Parent ended this original logical request",
    "TimeoutError",
    "APITimeoutError",
    "APIConnectionError",
    "InternalServerError",
    "ServiceUnavailable",
    "BrokenPipeError",
    "BillingError",
    "Error code: 402",
)


def provider_stall(result):
    text = f"{result.get('error_type', '')} {result.get('error', '')}"
    return "exceeds declared model bound" not in text and any(m in text for m in PROVIDER_STALL_MARKERS)


async def run_resume(args):
    """Continue strict-parser false failures in reference mode from their failing day.

    Each source quarter's own recorded responses are replayed, request-matched, for
    every request before the failing day; that day and the rest of the quarter are
    sampled live from the same policy (the source's exact weights). Only sources whose
    final response the strict parser rejected and the reference parser recovers are
    resumed, plus quarters interrupted by the operator or ended by a provider stall;
    genuine model failures remain failures and are listed as such.
    """
    if os.environ.get("SMB_PARSE_MODE") != "reference":
        raise SystemExit("resume continues in reference mode; pass --parse-mode reference")
    out = Path(args.out).absolute()
    out.mkdir(parents=True, exist_ok=True)
    formatter = campaign.make_formatter(args.reasoning, args.backbone)
    plan, specs = [], []
    for source in map(Path, args.source):
        source = source.absolute()
        if not (source / "plan.json").exists():
            continue  # not a quarter directory (e.g. eval-config.json, summary.json)
        if (source / "quarter-result.json").exists():
            result = json.loads((source / "quarter-result.json").read_text())
        else:  # stopped by the operator mid-quarter: every completed day is a valid draw
            plan_json = json.loads((source / "plan.json").read_text())
            world = plan_json.get("world_key") or source.name.split("__")[0]
            result = {"world": world, "status": "interrupted"}
        row = {
            "source": str(source),
            "world": result["world"],
            "source_status": result["status"],
            "source_failure": (result.get("failure") or {}).get("code"),
        }
        if result["status"] not in ("model_failure", "interrupted", "infrastructure_failure"):
            plan.append({**row, "resumed": False, "why": "not a model failure"})
            continue
        if result["status"] == "infrastructure_failure" and not provider_stall(result):
            plan.append(
                {
                    **row,
                    "resumed": False,
                    "why": "infrastructure failure not caused by a provider stall",
                    "error": str(result.get("error", ""))[:200],
                }
            )
            continue
        recovery = {"strict_reason": "interrupted by operator"}
        if result["status"] == "infrastructure_failure":
            error = str(result.get("error", ""))[:120]
            recovery = {"strict_reason": f"provider stall: {result.get('error_type')}: {error}"}
        if result["status"] == "model_failure":
            last = last_action(source)
            _, diag = formatter.decode_response(last["tokens"], last["stop_reason"], last["prompt_tokens"])
            recovery = diag.get("reference_parser_recovery")
        if recovery is None:
            plan.append({**row, "resumed": False, "why": "genuine failure: reference parser does not recover it"})
            continue
        day = failure_day(source)
        prefix = campaign.load_prefix(source, day)
        output = out / source.name
        if (output / "quarter-result.json").exists():
            plan.append({**row, "resumed": False, "why": "already resumed", "output": str(output)})
            continue
        plan.append(
            {
                **row,
                "resumed": True,
                "day": day,
                "replayed_requests": len(prefix["records"]),
                "strict_reason": recovery["strict_reason"],
                "output": str(output),
            }
        )
        if result["status"] == "model_failure":
            purpose = f"reference-mode resume from day {day} of strict-parser false failure {source}"
        else:
            purpose = (
                f"reference-mode resume from day {day} of {result['status']} {source} "
                f"({recovery['strict_reason']})"
            )
        specs.append(
            campaign.QuarterSpec(
                world=result["world"],
                output=output,
                run_id=f"{args.tag}-{source.name}",
                policy_id=f"{args.tag}:{args.policy}",
                policy_label=None,
                purpose=purpose,
                prefix=prefix,
                reasoning=args.reasoning,
                backbone=args.backbone,
                prompt_path=args.prompt,
                final_test_access=args.final_test_access,
            )
        )
    plan_name = "resume-dry-run" if args.dry_run else "resume-plan"
    (out / f"{plan_name}-{int(time.time())}.json").write_text(
        json.dumps({"args": vars(args), "decoding": dict(campaign.DECODING), "plan": plan}, indent=1, default=str)
    )
    for row in plan:
        print(json.dumps({k: row.get(k) for k in ("world", "source_failure", "resumed", "day", "why")}), flush=True)
    if not specs or args.dry_run:
        return
    keep = []
    completer, label = await make_policy(args.policy, keep=keep, backbone=args.backbone)
    retry_log = campaign.Journal(out / f"sampling-retries-{int(time.time())}.jsonl.gz")
    policy = RetryingCompleter(completer, log=retry_log)
    policy = budgeted(policy, formatter, args, retry_log)
    objective = campaign.DollarObjective()

    async def one(spec):
        spec.policy_label = label
        r = await campaign.run_quarter(
            spec, policy, formatter, objective=objective, failure_objective=objective.failure
        )
        ledger(
            {
                "kind": "resume_quarter",
                "tag": args.tag,
                "world": spec.world,
                "status": r["status"],
                "return_cents": r.get("return_cents"),
                "sampling_usd": r.get("sampling_usd"),
                "output": str(spec.output),
            }
        )
        print(json.dumps({k: r.get(k) for k in ("world", "status", "return_cents", "model_calls", "sampling_usd")}), flush=True)
        return r

    await asyncio.gather(*(one(s) for s in specs), return_exceptions=True)
    retry_log.close()


async def run_noop(args):
    from tinker_cookbook.completers import TokenCompleter, TokensWithLogprobs

    formatter = campaign.make_formatter("on", args.backbone)
    text = "<think>\n\n</think>\n\n<tool_call>\n<function=advance_day>\n</function>\n</tool_call>"
    ids = formatter.renderer.tokenizer.encode(text, add_special_tokens=False)
    ids += [formatter.renderer.get_stop_sequences()[0]]

    class AdvanceOnly(TokenCompleter):
        async def __call__(self, model_input, stop, *, max_tokens=None):
            return TokensWithLogprobs(tokens=list(ids), maybe_logprobs=[0.0] * len(ids), stop_reason="stop")

    out = Path(args.out).absolute()
    out.mkdir(parents=True, exist_ok=False)
    objective = campaign.DollarObjective()
    specs = [
        campaign.QuarterSpec(
            world=w,
            output=out / w,
            run_id=f"noop-{w}",
            policy_id="scripted:advance-only",
            policy_label="label:advance-only",
            purpose="economic floor: advance_day only, zero model calls",
            backbone=args.backbone,
            final_test_access=args.final_test_access,
        )
        for w in campaign.expand_worlds(args.worlds)
    ]
    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(spec):
        async with semaphore:
            return await campaign.run_quarter(
                spec, AdvanceOnly(), formatter, objective=objective, failure_objective=objective.failure
            )

    results = await asyncio.gather(*(one(s) for s in specs))
    table = {
        r["world"]: {"status": r["status"], "return_cents": r.get("return_cents"), "terminal": r.get("terminal_status")}
        for r in results
    }
    (out / "summary.json").write_text(json.dumps(table, indent=2))
    print(json.dumps(table, indent=1))


def status(args):
    root = Path(args.dir)
    rows = []
    for quarter in sorted(p.parent for p in root.rglob("plan.json")):
        result = quarter / "quarter-result.json"
        if result.exists():
            r = json.loads(result.read_text())
            rows.append(
                (
                    quarter.name,
                    r["status"],
                    r.get("return_cents"),
                    r.get("model_calls"),
                    r.get("sampling_usd"),
                    round(r.get("elapsed_seconds", 0) / 60),
                )
            )
            continue
        day = calls = None
        stderr = quarter / "worker.stderr.log"
        if stderr.exists():
            for line in stderr.read_text(errors="replace").splitlines()[::-1]:
                if '"morning_done"' in line:
                    row = json.loads(line)
                    day, calls = row["day"], row["provider_calls"]
                    break
        pace = None
        receipts = quarter / "engine/provider_receipts.jsonl"
        if receipts.exists():
            import datetime as dt

            starts = {}
            for line in receipts.read_text(errors="replace").splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("request_started_at_utc"):
                    stamp = r["request_started_at_utc"].replace("Z", "+00:00")
                    starts.setdefault((r["day"], r["turn"]), dt.datetime.fromisoformat(stamp))
            times = sorted(starts.values())
            if len(times) > 1:
                pace = f"{(times[-1] - times[0]).total_seconds() / (len(times) - 1):.1f}s/call"
        rows.append((quarter.name, "running", f"day {day} done", calls, pace, None))
    for row in rows:
        print(*row, sep="\t")


def build_registry(args):
    result = campaign.build_registry(
        Path(args.out or paths.registry()),
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        force=args.force,
    )
    registry = json.loads(Path(result["path"]).read_text())
    counts = {}
    for world in registry["worlds"].values():
        counts[world["split"]] = counts.get(world["split"], 0) + 1
    print(json.dumps({"registry": result, "worlds_by_split": counts}, indent=1))


def check_setup(args):
    report = {"engine_dir": str(paths.engine_dir()), "engine_python": str(paths.engine_python())}
    campaign.check_engine()
    report["engine_source_hashes"] = "match"
    report["tools"] = [t["name"] for t in campaign.original_tools()]
    registry = paths.registry()
    report["registry"] = str(registry)
    if registry.exists():
        report["worlds"] = {
            split: campaign.worlds_in_split(split) for split in ("train", "validation", "test")
        }
    else:
        report["registry_missing"] = "run: scripts/campaign.py build-registry ..."
    report["prompt"] = campaign.ref(paths.PROMPT)
    report["runs_dir"] = str(paths.runs_dir())
    report["tinker_api_key_set"] = bool(os.environ.get("TINKER_API_KEY"))
    print(json.dumps(report, indent=1))


def add_policy_flags(p, *, parse_default="strict", think_budget=True):
    p.add_argument("--reasoning", choices=["on", "off"], default="on")
    p.add_argument("--backbone", choices=sorted(BACKBONES), default=DEFAULT_BACKBONE)
    p.add_argument("--prompt", default=None, help="system prompt file (default: prompts/buyer_system.txt)")
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument(
        "--parse-mode",
        choices=["strict", "reference"],
        default=parse_default,
        help="strict: a malformed call ends the quarter; reference: Qwen serving-parser recovery",
    )
    if think_budget:
        p.add_argument(
            "--think-budget", type=int, default=0, help="evaluation-only reasoning cap in tokens (0 = off)"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry", default=None, help="world registry (sets SMB_WORLD_REGISTRY)")
    sub = parser.add_subparsers(dest="mode", required=True)

    g = sub.add_parser("build-registry", help="hash world directories into a registry")
    g.add_argument("--train-dir", required=True, help="directory of TRAIN world directories (b500-b503)")
    g.add_argument("--val-dir", required=True, help="directory of VAL world directories (b504)")
    g.add_argument("--test-dir", default=None, help="directory of sealed TEST world directories (b505)")
    g.add_argument("--out", default=None, help="registry path (default: SMB_WORLD_REGISTRY)")
    g.add_argument("--force", action="store_true", help="replace an existing registry")

    sub.add_parser("check-setup", help="verify engine, interpreter, tools and registry; no sampling")

    n = sub.add_parser("noop", help="advance-only quarters (zero model calls)")
    n.add_argument("--worlds", nargs="+", required=True, help="world keys or split names (train, validation, test)")
    n.add_argument("--out", required=True)
    n.add_argument("--concurrency", type=int, default=8)
    n.add_argument("--backbone", choices=sorted(BACKBONES), default=DEFAULT_BACKBONE)
    n.add_argument("--final-test-access", action="store_true")

    e = sub.add_parser("eval", help="original 90-day quarters for one policy")
    e.add_argument("--policy", required=True, help="base | sampler:<tinker path> | state:<tinker path>")
    e.add_argument("--worlds", nargs="+", required=True, help="world keys or split names (train, validation, test)")
    e.add_argument("--samples", type=int, default=1)
    e.add_argument("--out", required=True)
    e.add_argument("--tag", required=True)
    e.add_argument("--purpose", required=True)
    e.add_argument("--concurrency", type=int, default=16)
    e.add_argument("--final-test-access", action="store_true")
    add_policy_flags(e)

    b = sub.add_parser("branch-eval", help="continuations from archived frontier states")
    b.add_argument("--policy", required=True)
    add_policy_flags(b, think_budget=False)
    b.add_argument("--frontier-index", default=None, help="frontier quarter index (default: SMB_FRONTIER_INDEX)")
    b.add_argument("--hurdles-dir", default=None, help="frontier replays (default: $SMB_RUNS_DIR/frontier-hurdles)")
    b.add_argument("--day", type=int, default=60)
    b.add_argument("--k", type=int, default=2)
    b.add_argument("--pick", choices=["median", "top", "low"], default="median")
    b.add_argument("--worlds", nargs="*", default=None)
    b.add_argument("--out", required=True)
    b.add_argument("--tag", required=True)
    b.add_argument("--concurrency", type=int, default=32)

    r = sub.add_parser("resume", help="continue false failures / stalls from their failing day")
    r.add_argument("--source", nargs="+", required=True, help="quarter output directories")
    r.add_argument("--policy", required=True)
    add_policy_flags(r, parse_default="reference")
    r.add_argument("--out", required=True)
    r.add_argument("--tag", required=True)
    r.add_argument("--dry-run", action="store_true", help="write the resume plan only; no sampling")
    r.add_argument("--final-test-access", action="store_true")

    s = sub.add_parser("status", help="summarize quarters under a directory")
    s.add_argument("dir")

    args = parser.parse_args()
    if args.registry:
        os.environ["SMB_WORLD_REGISTRY"] = str(Path(args.registry).resolve())
    if getattr(args, "parse_mode", None):
        os.environ["SMB_PARSE_MODE"] = args.parse_mode
    if getattr(args, "top_p", 1.0) != 1.0 or getattr(args, "top_k", -1) != -1:
        campaign.set_decoding(args.top_p, args.top_k)
    try:
        if args.mode == "build-registry":
            build_registry(args)
        elif args.mode == "check-setup":
            check_setup(args)
        elif args.mode == "eval":
            asyncio.run(run_eval(args))
        elif args.mode == "branch-eval":
            asyncio.run(run_branch_eval(args))
        elif args.mode == "resume":
            asyncio.run(run_resume(args))
        elif args.mode == "noop":
            asyncio.run(run_noop(args))
        else:
            status(args)
    except paths.ConfigError as error:
        sys.exit(f"configuration: {error}")


if __name__ == "__main__":
    main()
