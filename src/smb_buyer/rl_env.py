"""Original full-buyer IPC as an official Env; the terminal objective is injected.

No worker, renderer, sample, grade or optimizer is created here. The caller must
admit those exact sources and own the sampling/session/resource guards. A failed
quarter has no objective value: it raises, retaining its original evidence.
"""

from __future__ import annotations

import asyncio
import math
from copy import deepcopy

import tinker
from tinker_cookbook.rl.types import Env, StepResult


class FullBuyerRLContractError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise FullBuyerRLContractError(message)


class WorkerTransport:
    """Adapt an already admitted run_worker(callback) to one pending Env turn.

    ``run`` is a caller-bound async function accepting only the actor callback;
    it invokes the maintained owned_worker with literal verified arguments.
    Completion is delivered only after that owner has returned and reaped its
    child. This class never starts another worker or retries a failed one.
    """

    def __init__(self, run, *, whole_seconds, close_seconds, record):
        require(callable(run) and callable(record), "Bound worker and evidence writer required")
        require(
            all(math.isfinite(x) and x > 0 for x in (whole_seconds, close_seconds)),
            "Finite worker/cleanup bounds required",
        )
        self.run, self.record = run, record
        self.whole_seconds, self.close_seconds = whole_seconds, close_seconds
        self.queue = asyncio.Queue(maxsize=1)
        self.pending = None
        self.task = None
        self.closed = False
        self.delivered_terminal = False

    async def _callback(self, event):
        require(self.pending is None, "Overlapping original model requests")
        future = asyncio.get_running_loop().create_future()
        self.pending = (event["request_key"], future)
        try:
            await self.queue.put(event)
            return await future
        finally:
            self.pending = None

    async def _run(self):
        try:
            async with asyncio.timeout(self.whole_seconds):
                return await self.run(self._callback)
        except BaseException as error:
            self.record(
                {
                    "event": "owned_worker_failed",
                    "error_type": type(error).__name__,
                    "worker_failure_ref": getattr(error, "worker_failure_ref", None),
                }
            )
            raise

    async def recv(self):
        require(not self.closed and not self.delivered_terminal, "Transport already ended")
        if self.task is None:
            self.task = asyncio.create_task(self._run())
        get = asyncio.create_task(self.queue.get())
        try:
            done, _ = await asyncio.wait({get, self.task}, return_when=asyncio.FIRST_COMPLETED)
            if self.task in done:
                self.delivered_terminal = True
                return self.task.result()  # Original failures propagate, never become a reward.
            return get.result()
        finally:
            if not get.done():
                get.cancel()
            await asyncio.gather(get, return_exceptions=True)

    async def send(self, event):
        require(self.pending is not None and not self.closed, "No pending original request")
        key, future = self.pending
        require(
            event.get("event") == "model_response"
            and event.get("request_key") == key
            and not future.done(),
            "Response does not match its single original request",
        )
        future.set_result({"response": event["response"]})

    async def close(self):
        if self.closed:
            return
        self.closed = True
        if self.task is None:
            self.record({"event": "owned_worker_closed", "never_started": True})
            return
        if not self.task.done():
            self.task.cancel()
        # wait() bounds this await even if a defective owner suppresses cancellation.
        done, _ = await asyncio.wait({self.task}, timeout=self.close_seconds)
        self.record({"event": "owned_worker_closed", "owner_task_settled": bool(done)})
        require(done, "Owned worker cleanup uncertain; stop rather than claim closure")
        await asyncio.gather(self.task, return_exceptions=True)


class FullBuyerEnv(Env):
    """Variable tool turns and fresh daily prompts; no artificial ninety-turn cap.

    ``terminal_validator`` is the reviewed original full-result/grade verifier.
    ``objective`` receives only its verified completion, and is called once.
    There is deliberately no default return scale or failure-reward function.
    The original sample arrays are retained externally by the native observer;
    the official rollout keeps its original TokensWithLogprobs object.
    """

    def __init__(
        self,
        transport,
        formatter,
        *,
        terminal_validator,
        objective,
        objective_ref,
        task_identity,
        policy_binding_ref,
        max_model_requests,
        io_timeout_seconds,
        record,
    ):
        require(callable(terminal_validator) and callable(objective), "Explicit objective required")
        require(callable(record), "Append/flush evidence writer required")
        require(type(max_model_requests) is int and max_model_requests > 0, "Finite request cap")
        require(math.isfinite(io_timeout_seconds) and io_timeout_seconds > 0, "Finite IPC bound")
        for reference in (objective_ref, policy_binding_ref):
            require(set(reference) == {"path", "sha256"}, "Explicit caller-bound receipt required")
        self.transport, self.formatter = transport, formatter
        self.terminal_validator, self.objective = terminal_validator, objective
        self.objective_ref, self.policy_binding_ref = (
            deepcopy(objective_ref),
            deepcopy(policy_binding_ref),
        )
        self.task_identity = deepcopy(task_identity)
        self.max_model_requests, self.io_timeout_seconds = max_model_requests, io_timeout_seconds
        self.record = record
        self.requests, self.actions, self.keys = [], [], set()
        self.pending = self.completion = self.failure = self.owner_task = None
        self.terminal_reward = None
        self.started = False

    async def _receive(self):
        event = await asyncio.wait_for(self.transport.recv(), self.io_timeout_seconds)
        self.record({"event": "child_event", "value": deepcopy(event)})
        if event.get("event") == "failed":
            self.failure = deepcopy(event)
            raise FullBuyerRLContractError("Original failed quarter has no declared RL reward")
        if event.get("event") == "complete":
            require(self.requests, "Zero-request quarter unsupported; no placeholder tokens")
            require(
                event.get("status") == "valid"
                and event.get("archived_replay_only") is False
                and event.get("logical_model_requests") == len(self.requests) == len(self.actions),
                "Incomplete/replayed quarter or missing original response",
            )
            complete = self.terminal_validator(event)
            require(
                type(complete.return_cents) is int
                and complete.return_cents == event["return_cents"]
                and complete.result_ref == event["result"]
                and type(complete.actual_sessions) is int
                and 0 < complete.actual_sessions <= 90
                and complete.actual_sessions == event["sessions"]
                and complete.terminal_status in {"completed", "insolvent"},
                "Independent original terminal result differs",
            )
            reward = self.objective(complete)
            require(type(reward) in (int, float) and math.isfinite(reward), "Nonfinite objective")
            self.completion, self.terminal_reward = complete, reward
            self.record(
                {
                    "event": "terminal_objective",
                    "raw_return_cents": complete.return_cents,
                    "reward": reward,
                    "objective_ref": self.objective_ref,
                    "result_ref": complete.result_ref,
                    "grade_ref": complete.grade_ref,
                }
            )
            return None
        require(event.get("event") == "model_request", "Unexpected child event")
        require(len(self.requests) < self.max_model_requests, "Logical request cap exceeded")
        key = event["request_key"]
        require(isinstance(key, str) and key and key not in self.keys, "Repeated request")
        require(type(event.get("day")) is int and 0 <= event["day"] < 90, "Invalid day")
        require(type(event.get("turn")) is int and event["turn"] >= 0, "Invalid turn")
        if self.requests:
            previous = self.requests[-1]
            require(
                (event["day"], event["turn"]) > (previous["day"], previous["turn"]),
                "Turn regressed",
            )
        prompt, metadata = self.formatter.render_request(event)
        self.keys.add(key)
        self.pending = {
            "request_key": key,
            "day": event["day"],
            "turn": event["turn"],
            "prompt_token_ids": prompt.to_ints(),
            "prompt_tokens": prompt.length,
            "metadata": metadata,
        }
        self.requests.append(self.pending)
        self.record(
            {
                "event": "native_prompt",
                **deepcopy(self.pending),
                "policy_binding_ref": self.policy_binding_ref,
            }
        )
        return prompt

    async def initial_observation(self):
        require(not self.started, "Env is single use")
        self.started, self.owner_task = True, asyncio.current_task()
        prompt = await self._receive()
        require(prompt is not None, "No actual initial model request")
        return prompt, self.formatter.renderer.get_stop_sequences()

    async def step(self, action, *, extra=None):
        require(self.pending is not None and self.completion is None, "No pending original request")
        require(
            extra is not None and extra.get("stop_reason") in {"stop", "length"},
            "Native stop required",
        )
        pending = self.pending
        original = list(action)
        self.actions.append({"tokens": original, "stop_reason": extra["stop_reason"]})
        # Preserve malformed/truncated samples before the original decoder can reject them.
        self.record(
            {
                "event": "native_action",
                "request_key": pending["request_key"],
                "tokens": original,
                "stop_reason": extra["stop_reason"],
            }
        )
        response, diagnostics = self.formatter.decode_response(
            action, extra["stop_reason"], pending["prompt_tokens"]
        )
        require(list(action) == original, "Decoder mutated original sampled tokens")
        self.record(
            {
                "event": "decoded_action",
                "request_key": pending["request_key"],
                "diagnostics": diagnostics,
                "response_before_original_bookkeeping": response,
            }
        )
        self.pending = None
        await asyncio.wait_for(
            self.transport.send(
                {
                    "event": "model_response",
                    "request_key": pending["request_key"],
                    "response": response,
                }
            ),
            self.io_timeout_seconds,
        )
        prompt = await self._receive()
        done = self.completion is not None
        return StepResult(
            reward=self.terminal_reward if done else 0.0,
            episode_done=done,
            next_observation=tinker.ModelInput.empty() if done else prompt,
            next_stop_condition=self.formatter.renderer.get_stop_sequences(),
            metrics={"original_buyer_model_turn": 1, "terminal_quarter": int(done)},
        )

    async def close(self):
        owner = self.owner_task
        if owner is not None and owner is not asyncio.current_task() and not owner.done():
            owner.cancel()
        await self.transport.close()
        if owner is not None and owner is not asyncio.current_task():
            done, _ = await asyncio.wait({owner}, timeout=self.transport.close_seconds)
            require(done, "Owned rollout did not settle after worker cleanup")
            await asyncio.gather(owner, return_exceptions=True)

