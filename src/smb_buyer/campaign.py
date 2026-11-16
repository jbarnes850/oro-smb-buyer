"""Full-buyer campaign runtime: one code path for evaluation and RL.

Each quarter runs the original engine worker (``worker.py`` through
``owned_worker.run_worker``) behind the ``FullBuyerEnv``/``WorkerTransport``
adapter, with the native Qwen renderer and schema-aware tool parser. Actions come
from the official Cookbook ``TinkerTokenCompleter`` driven by the official
``run_rollout`` loop, so sampled tokens, logprobs and sequence IDs are the SDK's
own objects.

Plans use the ``full-buyer-quarter-plan-v1`` schema checked by ``contracts.verify``.

Original benchmark model failures (a malformed call, turn limit, consecutive
tool-contract errors, ...) end the quarter without a profit. Evaluation records
them as failures. RL may assign them an explicitly declared failure reward;
infrastructure failures always raise and never become a reward.
"""

from __future__ import annotations

import asyncio
import functools
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from . import contracts, paths
from .backbones import BACKBONES, CONTEXT, DEFAULT_BACKBONE, MAX_OUTPUT, PRICES, REASONING_REPLAY
from .contracts import ROOT, SPLIT_BUSINESSES, read, ref, require, sha, write

HERE = Path(__file__).resolve().parent
PROVIDERS = "retailsim/harness/providers.py"

# Original harness RolloutFailure codes that end a quarter because of the model's
# own output or behavior. Everything else is infrastructure and never becomes an
# economic or training outcome.
MODEL_FAILURE_CODES = (
    "PROVIDER_INCOMPLETE_OUTPUT",
    "OUTPUT_TOKEN_LIMIT",
    "LIMITS_EXCEED_ENDPOINT_CONTEXT",
    "MALFORMED_PROVIDER_RESPONSE",
    "MALFORMED_PROVIDER_TOOL_CALLS",
    "PROVIDER_REFUSAL",
    "TOOL_CONTRACT_ERRORS",
    "DAILY_MODEL_TURN_LIMIT",
    "REQUEST_CONTEXT_LIMIT",
    "REQUEST_PROTOCOL_VIOLATION",
    "RUN_PROVIDER_CALL_LIMIT",
)


# ---------------------------------------------------------------- world registry


def mechanism_of(world_key: str) -> str:
    return world_key.split("experience_long_")[1].rsplit("_", 1)[0]


def build_registry(out: Path, *, train_dir, val_dir, test_dir=None, force=False):
    """Hash every world under the given split directories into one registry.

    World directories are named ``s50N_experience_long_<mechanism>_1`` and hold
    ``manifest.json`` plus ``public/`` and ``private/`` inputs. Only hashes and
    relative paths are recorded; no world content is copied.
    """
    engine = read(paths.ENGINE_HASHES)
    dirs = {"train": Path(train_dir).resolve(), "validation": Path(val_dir).resolve()}
    if test_dir is not None:
        dirs["test"] = Path(test_dir).resolve()
    root = Path(os.path.commonpath([str(d) for d in dirs.values()]))
    worlds = {}
    for split, base in dirs.items():
        for directory in sorted(p for p in base.iterdir() if p.is_dir()):
            manifest = directory / "manifest.json"
            business = "b" + directory.name.split("_")[0][1:]
            require(business in SPLIT_BUSINESSES[split], f"{directory} is not {split}")
            inputs = {
                str(f.relative_to(directory)): sha(f)
                for f in sorted(directory.rglob("*"))
                if f.is_file() and f.parent.name in ("private", "public")
            }
            manifest_sha = sha(manifest)
            worlds[manifest_sha] = {
                "key": directory.name,
                "business": business,
                "mechanism": mechanism_of(directory.name),
                "split": split,
                "path": str(directory.relative_to(root)),
                "manifest": {"path": str(manifest.relative_to(root)), "sha256": manifest_sha},
                "input_sha256": inputs,
            }
    registry = {
        "schema": "full-buyer-world-registry-v2",
        "paths_relative_to": str(root),
        "engine": {k: engine[k] for k in ("commit", "python_files_sha256", "closure_files_sha256")},
        "engine_hashes": ref(paths.ENGINE_HASHES),
        "worlds": worlds,
    }
    out = Path(out)
    if out.exists():
        require(force, f"{out} exists; quarters bind the registry by hash (pass --force to replace)")
        out.unlink()
    write(out, registry)
    return ref(out)


def worlds_by_key(registry_path=None):
    registry = read(registry_path or paths.registry())
    return {w["key"]: (manifest, w) for manifest, w in registry["worlds"].items()}


def worlds_in_split(split, registry_path=None):
    return sorted(k for k, (_, w) in worlds_by_key(registry_path).items() if w["split"] == split)


def expand_worlds(names, registry_path=None):
    """World keys; a split name (train, validation, test) stands for all its worlds."""
    out = []
    for name in names:
        out += worlds_in_split(name, registry_path) if name in SPLIT_BUSINESSES else [name]
    return out


# ---------------------------------------------------------------- engine


@functools.lru_cache(maxsize=1)
def original_tools():
    """The engine's own v11 tool schemas, read with the engine interpreter."""
    code = (
        "import json, sys; sys.dont_write_bytecode = True; sys.path.insert(0, sys.argv[1]); "
        "from retailsim.harness.rollout import agent_tools; "
        "sys.stdout.write(json.dumps(agent_tools('v11')))"
    )
    output = subprocess.run(
        [str(paths.engine_python()), "-I", "-B", "-c", code, str(paths.engine_dir())],
        check=True,
        capture_output=True,
        text=True,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8"},
    ).stdout
    tools = json.loads(output)
    require(tuple(t["name"] for t in tools) == contracts.TOOLS, "Original16 tool order differs")
    return tools


def check_engine():
    """Engine files match the campaign's hashes; return the engine directory."""
    source = paths.engine_dir()
    contracts.verify_engine_source(source, read(paths.ENGINE_HASHES))
    require(
        sha(source / "retailsim/harness/prompts/buyer-prompt-v11.txt") == contracts.PROMPT_SHA,
        "Original prompt asset differs",
    )
    return source


# ---------------------------------------------------------------- plans


def source_hashes(prompt_path: Path):
    files = [HERE / name for name in contracts.IPC_MODULES]
    files += [HERE / name for name in ("campaign.py", "rl_env.py", "qwen_tools.py", "native.py")]
    files.append(Path(prompt_path))
    return {str(p.resolve()): sha(p) for p in files}


def prepare_quarter(
    world: str,
    output: Path,
    *,
    run_id: str,
    policy_id: str,
    policy_label: str | None,
    purpose: str,
    final_test_access: bool = False,
    response_seconds: int = 3000,
    whole_seconds: int = 43200,
    reasoning: str = "on",
    backbone: str = DEFAULT_BACKBONE,
    prompt_path: Path | None = None,
):
    """Write plan.json and adoption.json beside a fresh quarter output."""
    model, _, renderers_by_mode = BACKBONES[backbone]
    renderer_name = renderers_by_mode[reasoning]
    prompt_path = Path(prompt_path or paths.PROMPT).resolve()
    output = Path(output).absolute()
    require(not (output / "engine").exists(), "Fresh quarter output required")
    output.mkdir(parents=True, exist_ok=True)
    registry_path = paths.registry()
    registry = read(registry_path)
    manifest_sha, entry = worlds_by_key(registry_path)[world]
    split = entry["split"]
    require(split != "test" or final_test_access, "TEST requires explicit final access")
    plan = {
        "schema": "full-buyer-quarter-plan-v1",
        "environment_contract": contracts.ENVIRONMENT,
        "prompt_sha256": contracts.PROMPT_SHA,
        "tool_names": list(contracts.TOOLS),
        "engine_source": str(paths.engine_dir()),
        "engine_python": str(paths.engine_python()),
        "asset_manifest": ref(registry_path),
        "world_registry": ref(registry_path),
        "world_asset_root": registry["paths_relative_to"],
        "world_ref": manifest_sha,
        "world_key": world,
        "split": split,
        "final_test_access": bool(final_test_access),
        "history": None,
        "archived_replay_only": False,
        "output": str(output),
        "run_id": run_id,
        "policy_id": policy_id,
        "purpose": purpose,
        "sampler_model_path": policy_label,
        "native": {
            "model": model,
            "context_window": CONTEXT,
            "max_output_tokens": MAX_OUTPUT,
            "max_prompt_tokens": CONTEXT - MAX_OUTPUT,
            "renderer_name": renderer_name,
            "reasoning_mode": reasoning,
            "temperature": 1.0,
        },
        "adapter": {
            "model": model,
            "max_output_tokens": MAX_OUTPUT,
            "temperature": 1.0,
            "reasoning_request": None,
            "reasoning_replay": REASONING_REPLAY[reasoning],
        },
        "limits": {
            "max_provider_calls": 3600,
            "max_physical_attempts": 3,
            "response_seconds": response_seconds,
            "max_turns_per_day": 100,
            "max_tool_calls_per_day": 60,
            "max_request_bytes": 512000,
            "ipc_max_bytes": 4194304,
            # The engine's own conservative byte-based reservation; set so it
            # cannot end a quarter. Real spend is tracked from SDK token counts.
            "accounting_ceiling_usd": 5000,
        },
        "resources": {
            "workers": 1,
            "whole_seconds": whole_seconds,
            "output_bytes_cap": 3 * 1024**3,
            "free_floor_bytes": 12 * 1024**3,
        },
        "accounting": "list-estimate-after-call-v1",
        "reservation_rates": {
            "input_upper": PRICES[backbone]["prefill"],
            "output": PRICES[backbone]["sample"],
        },
        "system_prompt": ref(prompt_path),
        "source_hashes": source_hashes(prompt_path),
    }
    write(output / "plan.json", plan)
    plan_ref = ref(output / "plan.json")
    write(
        output / "adoption.json",
        {
            "status": "adopted_full_buyer_quarter",
            "plan": plan_ref,
            "output": str(output),
            "sampler_model_path": policy_label,
            "engine_quarters": 1,
            "training_calls": 0,
            "checkpoint_writes": 0,
            "ceiling_usd": "5000",
        },
    )
    adoption_ref = ref(output / "adoption.json")
    contracts.verify(plan_ref, adoption_ref)
    return plan_ref, adoption_ref


# ---------------------------------------------------------------- native pieces


def make_formatter(reasoning="on", backbone=DEFAULT_BACKBONE):
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    from .native import NativeBuyerConfig, NativeBuyerRenderer, load_original_provider_functions
    from .qwen_tools import with_tool_parser

    model, tokenizer_model, renderers_by_mode = BACKBONES[backbone]
    config = NativeBuyerConfig(
        model=model,
        renderer_name=renderers_by_mode[reasoning],
        reasoning_mode=reasoning,
        max_output_tokens=MAX_OUTPUT,
        context_window=CONTEXT,
    )
    engine = read(paths.ENGINE_HASHES)
    arguments, body = load_original_provider_functions(
        paths.engine_dir() / PROVIDERS, engine["python_files_sha256"][PROVIDERS]
    )
    formatter = NativeBuyerRenderer(config, get_tokenizer(tokenizer_model), arguments, body)
    formatter.renderer = with_tool_parser(formatter.renderer, original_tools())
    return formatter


def failure_code(message: str) -> str:
    return (message or "").split(":", 1)[0].strip()


def classify_failure(failure_ref):
    """Read the original worker failure receipt; return (code, is_model_failure)."""
    failure = read(failure_ref["path"])
    code = failure_code(failure.get("message", ""))
    model = failure.get("type") == "RolloutFailure" and code in MODEL_FAILURE_CODES
    return code, model, failure


def _compact(row):
    """Journal row without full message bodies or prompt token arrays."""
    event = row.get("event")
    if event == "child_event":
        value = row["value"]
        if value.get("event") == "model_request":
            return {
                "event": "model_request",
                "request_key": value["request_key"],
                "day": value["day"],
                "turn": value["turn"],
                "request_sha256": value["request_sha256"],
            }
        return {"event": "child_event", "value": value}
    if event == "native_prompt":
        ids = row["prompt_token_ids"]
        return {
            "event": "native_prompt",
            "request_key": row["request_key"],
            "day": row["day"],
            "turn": row["turn"],
            "prompt_tokens": len(ids),
            "prompt_token_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            "policy_binding_ref": row.get("policy_binding_ref"),
        }
    return row


class Journal:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.stream = gzip.open(self.path, "xt")
        self.count = 0

    def __call__(self, row):
        row = _compact(row)
        row = {"t": time.time(), **row}
        self.stream.write(json.dumps(row, default=str) + "\n")
        self.count += 1
        if self.count % 50 == 0:
            self.stream.flush()

    def close(self):
        if not self.stream.closed:
            self.stream.close()


def _make_env_class():
    from .rl_env import FullBuyerEnv

    class QuarterEnv(FullBuyerEnv):
        """FullBuyerEnv with an explicit, optional model-failure terminal reward."""

        def __init__(self, *args, failure_objective=None, prefix=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.failure_objective = failure_objective
            self.model_failure = None
            self.infra_failure = None
            # Optional prefix: recorded responses for every request before
            # ``prefix["day"]`` are served back verbatim (deterministic engine), so
            # the live episode starts from that exact day-d state.
            self.prefix = prefix
            self.replayed = 0

        async def _replay_prefix(self):
            """Serve recorded prefix responses; return once the next request is live."""
            while self.prefix is not None:
                peek = await asyncio.wait_for(self.transport.recv(), self.io_timeout_seconds)
                if peek.get("event") != "model_request" or peek["day"] >= self.prefix["day"]:
                    require(
                        self.replayed == len(self.prefix["records"]),
                        "Prefix replay ended before all recorded requests were served",
                    )
                    self.record(
                        {
                            "event": "prefix_replay_complete",
                            "replayed": self.replayed,
                            "branch_day": self.prefix["day"],
                            "source": self.prefix["source"],
                        }
                    )
                    self.prefix = None
                    return peek
                require(
                    self.replayed < len(self.prefix["records"]),
                    "Prefix ran out of recorded responses",
                )
                record = self.prefix["records"][self.replayed]
                if self.prefix.get("match", "request") == "request":
                    require(
                        peek["request_sha256"] == record["request_sha256"],
                        "Own-trajectory prefix replay diverged from the recorded request",
                    )
                else:
                    # Frontier prefix: the request text differs (other prompt and
                    # provider), but the engine actions are the recorded ones, so the
                    # original (day, turn) sequence must line up exactly.
                    require(
                        (peek["day"], peek["turn"]) == (record["day"], record["turn"]),
                        "Frontier prefix replay diverged from the recorded day/turn sequence",
                    )
                self.replayed += 1
                await asyncio.wait_for(
                    self.transport.send(
                        {
                            "event": "model_response",
                            "request_key": peek["request_key"],
                            "response": record["response"],
                        }
                    ),
                    self.io_timeout_seconds,
                )
            return None

        async def _receive(self):
            try:
                pending_event = await self._replay_prefix() if self.prefix is not None else None
                if pending_event is not None:
                    original_recv = self.transport.recv

                    async def first_then_original():
                        self.transport.recv = original_recv
                        return pending_event

                    self.transport.recv = first_then_original
                    try:
                        return await self._receive_live()
                    finally:
                        self.transport.recv = original_recv
                return await self._receive_live()
            except Exception as error:  # noqa: BLE001 -- classify, never invent profit
                failure_ref = getattr(error, "worker_failure_ref", None)
                if failure_ref is None:
                    self.infra_failure = f"{type(error).__name__}: {error}"
                    raise
                code, is_model, failure = classify_failure(failure_ref)
                if not is_model:
                    self.infra_failure = f"{code}: {failure.get('message', '')[:300]}"
                    raise
                self.model_failure = {
                    "code": code,
                    "message": failure.get("message", "")[:500],
                    "failure_ref": failure_ref,
                    "model_requests": len(self.requests),
                    "last_day": self.requests[-1]["day"] if self.requests else None,
                }
                if self.failure_objective is None:
                    raise
                reward = self.failure_objective(self.model_failure)
                require(isinstance(reward, (int, float)), "Declared failure reward required")
                self.terminal_reward = float(reward)
                self.completion = "model_failure"
                self.record(
                    {"event": "terminal_model_failure", **self.model_failure, "reward": reward}
                )
                return None

        def _share_prefix(self, prompt):
            """Same tokens, less memory: reuse the previous observation's chunks
            when the new prompt extends it (one morning's growing conversation)."""
            import tinker

            if prompt is None:
                return prompt
            ids = prompt.to_ints()
            previous = getattr(self, "_shared_prompt", None)
            if (
                previous is not None
                and len(ids) > previous.length
                and ids[: previous.length] == previous.to_ints()
            ):
                prompt = tinker.ModelInput(
                    chunks=[*previous.chunks, tinker.EncodedTextChunk(tokens=ids[previous.length :])]
                )
            self._shared_prompt = prompt
            if self.requests and isinstance(self.requests[-1].get("prompt_token_ids"), list):
                # Journaled already; keep only the length for ordering/accounting.
                self.requests[-1]["prompt_token_ids"] = len(self.requests[-1]["prompt_token_ids"])
            return prompt

        async def _receive_live(self):
            return self._share_prefix(await self._receive_live_inner())

        async def _receive_live_inner(self):
            if self.replayed:
                # The original worker counts replayed logical requests too.
                original_requests = self.requests
                padded = _PaddedRequests(original_requests, self.replayed)
                self.requests = padded
                original_actions = self.actions
                self.actions = _PaddedRequests(original_actions, self.replayed)
                try:
                    return await super()._receive()
                finally:
                    self.requests = original_requests
                    self.actions = original_actions
            return await super()._receive()

    return QuarterEnv


class _PaddedRequests(list):
    """len() includes replayed requests only for the original completion count check."""

    def __init__(self, base, extra):
        super().__init__()
        self.base, self.extra = base, extra

    def __len__(self):
        return len(self.base) + self.extra

    def __getitem__(self, index):
        return self.base[index]

    def __bool__(self):
        # Turn-order checks compare live requests only; replayed ones are verified by hash.
        return bool(self.base)

    def append(self, value):
        self.base.append(value)

    def __iter__(self):
        return iter(self.base)


def journal_rows(path: Path):
    """Complete JSON lines of a gzip journal; tolerates a quarter stopped mid-write
    (missing end-of-stream marker, partial last line)."""
    rows = []
    try:
        with gzip.open(path, "rt") as stream:
            for line in stream:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    break
    except EOFError:
        pass
    return rows


def load_prefix(source_output: Path, day: int):
    """Recorded (request_sha256, response) pairs for every request before ``day``."""
    rows = journal_rows(Path(source_output) / "journal.jsonl.gz")
    records, pending = [], None
    for row in rows:
        if row.get("event") == "prefix_replay_complete":
            raise ValueError("Nested prefix sources are not supported; use a full-quarter source")
        if row.get("event") == "model_request":
            pending = row
        elif row.get("event") == "decoded_action" and pending is not None:
            if pending["day"] >= day:
                break
            records.append(
                {
                    "request_sha256": pending["request_sha256"],
                    "day": pending["day"],
                    "turn": pending["turn"],
                    "response": row["response_before_original_bookkeeping"],
                }
            )
            pending = None
    return {"day": day, "records": records, "source": str(source_output)}


_ENV_CLASS = None


def env_class():
    global _ENV_CLASS
    if _ENV_CLASS is None:
        _ENV_CLASS = _make_env_class()
    return _ENV_CLASS


@dataclass
class QuarterSpec:
    world: str
    output: Path
    run_id: str
    policy_id: str
    policy_label: str | None
    purpose: str
    final_test_access: bool = False
    prefix: dict | None = None
    reasoning: str = "on"
    backbone: str = DEFAULT_BACKBONE
    prompt_path: str | None = None


def make_quarter_env(spec: QuarterSpec, formatter, *, objective, failure_objective, journal):
    from .native import original_completion_validator
    from .owned_worker import run_worker
    from .rl_env import WorkerTransport

    plan_ref, adoption_ref = prepare_quarter(
        spec.world,
        spec.output,
        run_id=spec.run_id,
        policy_id=spec.policy_id,
        policy_label=spec.policy_label,
        purpose=spec.purpose,
        final_test_access=spec.final_test_access,
        reasoning=spec.reasoning,
        backbone=spec.backbone,
        prompt_path=spec.prompt_path,
    )
    plan = read(plan_ref["path"])
    arguments = [
        "--plan",
        plan_ref["path"],
        "--plan-sha256",
        plan_ref["sha256"],
        "--adoption",
        adoption_ref["path"],
        "--adoption-sha256",
        adoption_ref["sha256"],
    ]

    async def run(callback):
        return await run_worker(
            arguments,
            callback,
            engine_python=plan["engine_python"],
            repository=ROOT,
            stderr_path=Path(spec.output) / "worker.stderr.log",
        )

    transport = WorkerTransport(
        run, whole_seconds=plan["resources"]["whole_seconds"], close_seconds=60, record=journal
    )
    objective_path = Path(spec.output) / "objective.json"
    if not objective_path.exists():
        write(objective_path, getattr(objective, "declaration", {"objective": repr(objective)}))
    binding_path = Path(spec.output) / "policy-binding.json"
    if not binding_path.exists():
        write(
            binding_path,
            {
                "policy_id": spec.policy_id,
                "policy_label": spec.policy_label,
                "model": BACKBONES[spec.backbone][0],
                "reasoning": spec.reasoning,
                "prompt": str(spec.prompt_path or paths.PROMPT),
            },
        )
    return env_class()(
        transport,
        formatter,
        terminal_validator=original_completion_validator(
            result_path=Path(spec.output) / "engine/result.json",
            plan_ref=plan_ref,
            adoption_ref=adoption_ref,
        ),
        objective=objective,
        objective_ref=ref(objective_path),
        task_identity={"world": spec.world, "split": plan["split"]},
        policy_binding_ref=ref(binding_path),
        max_model_requests=3600,
        io_timeout_seconds=3600,
        record=journal,
        failure_objective=failure_objective,
        prefix=spec.prefix,
    )


# ---------------------------------------------------------------- objectives


class DollarObjective:
    """Declared RL reward: graded whole-business return in thousands of dollars.

    A failed quarter (original model-ending failure) receives ``failure_kusd``.
    Group centering removes any per-world constant; this scale weights a dollar
    equally in every business.
    """

    def __init__(self, failure_kusd: float = -25.0):
        self.failure_kusd = float(failure_kusd)
        self.declaration = {
            "schema": "full-buyer-objective-v1",
            "valid_quarter_reward": "business_return_cents / 100000 (thousands of USD)",
            "model_failure_reward_kusd": self.failure_kusd,
            "infrastructure_failure": "raises; never a reward",
        }

    def __call__(self, completion):
        return completion.return_cents / 100000.0

    def failure(self, model_failure):
        return self.failure_kusd


# ---------------------------------------------------------------- costs


def usage_cost(prompt_tokens, cache_hits, output_tokens, train_tokens=0, *, backbone):
    rates = PRICES[backbone]
    return (
        (prompt_tokens - cache_hits) * rates["prefill"]
        + cache_hits * rates["cached_prefill"]
        + output_tokens * rates["sample"]
        + train_tokens * rates["train"]
    ) / 1e6


def trajectory_usage(trajectory, backbone):
    prompt = cache = output = 0
    for transition in trajectory.transitions:
        prompt += transition.ob.length
        output += len(transition.ac.tokens)
        cache += getattr(transition.ac, "prompt_cache_hit_tokens", 0) or 0
    return {
        "model_calls": len(trajectory.transitions),
        "prompt_tokens": prompt,
        "prompt_cache_hit_tokens": cache,
        "output_tokens": output,
        "sampling_usd": round(usage_cost(prompt, cache, output, backbone=backbone), 6),
    }


def write_actions(trajectory, path: Path):
    """Sampled tokens, behavior logprobs and SDK sequence IDs for every response."""
    with gzip.open(path, "xt") as stream:
        for index, transition in enumerate(trajectory.transitions):
            action = transition.ac
            stream.write(
                json.dumps(
                    {
                        "index": index,
                        "prompt_tokens": transition.ob.length,
                        "tokens": action.tokens,
                        "logprobs": action.maybe_logprobs,
                        "stop_reason": action.stop_reason,
                        "sequence_id": action.sequence_id,
                        "prompt_cache_hit_tokens": getattr(action, "prompt_cache_hit_tokens", 0),
                        "reward": transition.reward,
                        "episode_done": transition.episode_done,
                    }
                )
                + "\n"
            )


def compress_engine_output(output: Path):
    """After validation, gzip large engine JSONL files in place (bytes preserved)."""
    engine = Path(output) / "engine"
    if not engine.exists():
        return {}
    receipt = {}
    for path in sorted(engine.glob("*.jsonl")):
        if path.stat().st_size < 2 * 1024**2:
            continue
        digest = sha(path)
        target = path.with_suffix(".jsonl.gz")
        with path.open("rb") as source, gzip.open(target, "xb", compresslevel=6) as sink:
            shutil.copyfileobj(source, sink, 1 << 20)
        receipt[path.name] = {"sha256": digest, "bytes": path.stat().st_size}
        path.unlink()
    if receipt:
        write(Path(output) / "compressed-engine-files.json", receipt)
    return receipt


# ---------------------------------------------------------------- one quarter


async def run_quarter(
    spec: QuarterSpec, policy, formatter, *, objective=None, failure_objective=None
):
    """Run one original 90-day quarter with ``policy``; always writes quarter-result.json."""
    from tinker_cookbook.rl.rollout_runner import run_rollout

    output = Path(spec.output)
    output.mkdir(parents=True, exist_ok=True)
    objective = objective or DollarObjective()
    journal = Journal(output / "journal.jsonl.gz")
    started = time.time()
    result = {
        "world": spec.world,
        "run_id": spec.run_id,
        "policy_id": spec.policy_id,
        "policy_label": spec.policy_label,
        "status": "infrastructure_failure",
        "return_cents": None,
        "started_at": started,
    }
    env = None
    try:
        env = make_quarter_env(
            spec, formatter, objective=objective, failure_objective=failure_objective, journal=journal
        )
        trajectory = await run_rollout(policy, env)
        result.update(trajectory_usage(trajectory, spec.backbone))
        write_actions(trajectory, output / "actions.jsonl.gz")
        if env.model_failure is not None:
            result.update(status="model_failure", failure=env.model_failure)
        else:
            require(env.completion is not None, "Rollout ended without terminal evidence")
            result.update(
                status="valid",
                return_cents=env.completion.return_cents,
                terminal_status=env.completion.terminal_status,
                sessions=env.completion.actual_sessions,
                grade=env.completion.grade_ref,
            )
        result["trajectory_reward"] = sum(t.reward for t in trajectory.transitions)
    except BaseException as error:  # noqa: BLE001 -- retain infrastructure evidence
        if env is not None and env.model_failure is not None and failure_objective is None:
            result.update(status="model_failure", failure=env.model_failure)
        else:
            result.update(
                status="infrastructure_failure",
                error_type=type(error).__name__,
                error=str(error)[:2000],
                traceback=traceback.format_exc()[-6000:],
            )
        if isinstance(error, asyncio.CancelledError):
            raise
    finally:
        if env is not None:
            try:
                await env.close()
            except BaseException as error:  # noqa: BLE001
                result["close_error"] = f"{type(error).__name__}: {error}"
        journal.close()
        result["elapsed_seconds"] = time.time() - started
        result["parse_mode"] = os.environ.get("SMB_PARSE_MODE", "strict")
        try:  # malformed outputs recovered by the reference parser (reference mode only)
            recovered = 0
            with gzip.open(output / "journal.jsonl.gz", "rt") as rows:
                for line in rows:
                    if '"reference_parser_recovery"' in line:
                        recovered += 1
            result["reference_parser_recoveries"] = recovered
        except Exception as error:  # noqa: BLE001
            result["reference_parser_recoveries_error"] = repr(error)
        try:  # every finished quarter: validation already read the plain files
            result["compressed"] = compress_engine_output(output)
        except Exception as error:  # noqa: BLE001
            result["compress_error"] = repr(error)
        write(output / "quarter-result.json", result)
    return result


# ---------------------------------------------------------------- frontier quarters


def world_of_task(task_id: str) -> str:
    """Leaderboard task id ``r2-b50N-<mechanism>`` -> world key."""
    business, mechanism = task_id[3:7], task_id.split("-", 2)[2]
    return f"s{business[1:]}_experience_long_{mechanism}_1"


def frontier_responses(archive_path):
    """Public responses of an archived frontier quarter, in (day, turn) order.

    Each is the harness-normalized model output the original buyer consumed:
    public text and ordered tool calls only. Provider reasoning is not replayed;
    it does not change engine actions. Accepts a compact ``*.jsonl.gz`` of
    transcript records or the archived run ``*.tar.gz`` (its model_transcript.jsonl).
    """
    import tarfile

    def lines():
        if str(archive_path).endswith(".jsonl.gz"):
            with gzip.open(archive_path, "rb") as stream:
                yield from stream
            return
        with tarfile.open(archive_path, "r:gz") as tar:
            member = next(m for m in tar.getmembers() if m.name.endswith("model_transcript.jsonl"))
            yield from tar.extractfile(member)

    out = []
    for line in lines():
        record = json.loads(line)
        if record.get("kind") != "model":
            continue
        calls = [
            {
                "id": c.get("id") or f"call_{record['day']}_{record['turn']}_{i}",
                "name": c["name"],
                "arguments": (
                    c.get("arguments")
                    if isinstance(c.get("arguments"), dict)
                    else json.loads(c.get("arguments") or "{}")
                ),
            }
            for i, c in enumerate(record.get("tool_calls") or [])
        ]
        content = record.get("content") or ""
        if not isinstance(content, str):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        out.append(
            {
                "day": record["day"],
                "turn": record["turn"],
                "response": {
                    "content": content,
                    "tool_calls": calls,
                    "finish_reason": "tool_calls" if calls else "stop",
                    "provider_status": "completed",
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                    "provider_metadata": {"model": "frontier-replay", "provider": "archive"},
                },
            }
        )
    return out


def load_frontier_prefix(archive_path, day):
    records = [r for r in frontier_responses(archive_path) if r["day"] < day]
    return {"day": day, "records": records, "source": str(archive_path), "match": "order"}


async def frontier_quarter_return(world, archive_path, output, *, final_test_access=False):
    """Replay an entire frontier quarter through this harness; return graded cents."""
    from .native import original_completion_validator
    from .owned_worker import run_worker

    records = frontier_responses(archive_path)
    plan_ref, adoption_ref = prepare_quarter(
        world,
        output,
        run_id="frontier-replay",
        policy_id="frontier-replay",
        policy_label="label:frontier-replay",
        purpose="frontier hurdle replay",
        final_test_access=final_test_access,
    )
    plan = read(plan_ref["path"])
    state = {"i": 0}

    async def callback(event):
        record = records[state["i"]]
        require(
            (event["day"], event["turn"]) == (record["day"], record["turn"]),
            "Frontier replay diverged",
        )
        state["i"] += 1
        return {"response": record["response"]}

    arguments = [
        "--plan",
        plan_ref["path"],
        "--plan-sha256",
        plan_ref["sha256"],
        "--adoption",
        adoption_ref["path"],
        "--adoption-sha256",
        adoption_ref["sha256"],
    ]
    event = await run_worker(
        arguments,
        callback,
        engine_python=plan["engine_python"],
        repository=ROOT,
        stderr_path=Path(output) / "worker.stderr.log",
    )
    validator = original_completion_validator(
        result_path=Path(output) / "engine/result.json", plan_ref=plan_ref, adoption_ref=adoption_ref
    )
    complete = validator(event)
    result = {
        "world": world,
        "archive": str(archive_path),
        "return_cents": complete.return_cents,
        "terminal_status": complete.terminal_status,
        "replayed": state["i"],
        "recorded": len(records),
    }
    write(Path(output) / "frontier-replay-result.json", result)
    compress_engine_output(output)
    return result


def read_frontier_index(index_path=None):
    """Rows {trial_id, model, task_id, reward, transcript|path} of archived frontier quarters."""
    rows = json.loads(Path(index_path or paths.frontier_index()).read_text())
    return rows["archives"] if isinstance(rows, dict) else rows


def frontier_bank(index_path=None, hurdles_dir=None, *, order="hurdle"):
    """Frontier quarters per world that have a graded hurdle from exact replay.

    ``hurdles_dir`` holds ``<trial_id>/frontier-replay-result.json`` written by
    ``scripts/frontier_hurdles.py``. ``order="hurdle"`` sorts each world's rows by
    hurdle (branch-eval picks low/median/top); ``order="model"`` sorts by model and
    trial id (the RL state draw indexes into this order).
    """
    hurdles_dir = Path(hurdles_dir or paths.frontier_hurdles())
    hurdles = {
        p.parent.name: json.loads(p.read_text())
        for p in hurdles_dir.rglob("frontier-replay-result.json")
    }
    bank = {}
    for row in read_frontier_index(index_path):
        transcript = row.get("transcript") or row.get("path")
        hurdle = hurdles.get(row["trial_id"])
        if hurdle is None or hurdle.get("return_cents") is None:
            continue
        if not transcript or not Path(transcript).exists():
            continue
        world = world_of_task(row["task_id"])
        bank.setdefault(world, []).append(
            {**row, "transcript": transcript, "world": world, "hurdle_cents": hurdle["return_cents"]}
        )
    key = {
        "hurdle": lambda r: (r["hurdle_cents"], r["trial_id"]),
        "model": lambda r: (r["model"], r["trial_id"]),
    }[order]
    for rows in bank.values():
        rows.sort(key=key)
    return bank


# ---------------------------------------------------------------- decoding

DECODING = {"top_p": 1.0, "top_k": -1}  # untruncated default; set via set_decoding


def set_decoding(top_p=1.0, top_k=-1):
    """Declared model-specific decoding (temperature stays 1.0).

    Qwen's recommended sampling truncates the low-probability tail (top_p 0.95,
    top_k 20). Applied by patching the official TinkerTokenCompleter's
    SamplingParams so evaluation and the official RL runner sample identically.
    Returned logprobs remain the sampler's; importance ratios absorb any
    truncation mismatch during training.
    """
    import tinker
    from tinker_cookbook import completers

    DECODING.update(top_p=float(top_p), top_k=int(top_k))
    if getattr(completers, "_full_buyer_patched", False):
        return DECODING
    original = tinker.SamplingParams

    def params(**kwargs):
        kwargs.setdefault("top_p", DECODING["top_p"])
        kwargs.setdefault("top_k", DECODING["top_k"])
        return original(**kwargs)

    class _Namespace:
        def __getattr__(self, name):
            return params if name == "SamplingParams" else getattr(tinker, name)

    completers.tinker = _Namespace()
    completers._full_buyer_patched = True
    return DECODING
