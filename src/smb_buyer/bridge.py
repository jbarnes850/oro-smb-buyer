"""One original buyer model call over owned JSON IPC; no HTTP or Tinker client."""

import json
import select
import sys

from retailsim.harness.providers import ModelAdapter, OpenAICompatibleAdapter

from smb_buyer.contracts import digest, require


class BridgeFailure(RuntimeError):
    """Not classified as retryable by the original provider retry loop."""


class IPCModelAdapter(ModelAdapter):
    kind = "chat"
    expected_provider = None
    expected_model = None

    def __init__(
        self,
        config,
        *,
        replay_only,
        timeout_seconds,
        max_line_bytes,
        input_stream=None,
        output_stream=None,
    ):
        self.name = "original-full-buyer-ipc-v1"
        self.live = not replay_only
        self.allow_live = not replay_only
        self.model = config["model"]
        self.max_tokens = config["max_output_tokens"]
        self.temperature = config["temperature"]
        self.provider_routing = None
        self.reasoning_request = config["reasoning_request"]
        self.reasoning_replay = config["reasoning_replay"]
        self.usage_include = False
        self.replay_only = replay_only
        self.timeout_seconds, self.max_line_bytes = timeout_seconds, max_line_bytes
        self.input, self.output = input_stream or sys.stdin, output_stream or sys.stdout
        self.policy = None
        self.dispatched = set()
        self.receipts = []

    def settings(self):
        return {
            **super().settings(),
            "model": self.model,
            "transport": "owned-json-ipc",
            "archived_replay_only": self.replay_only,
            "reasoning_replay": self.reasoning_replay,
            "temperature": self.temperature,
            "reasoning_request": self.reasoning_request,
            "network_client_in_child": False,
        }

    def request_body(self, messages, tools):
        # Preserve the original chat conversion. The archived source explicitly
        # omitted temperature; omission is a transport setting, not a message edit.
        body = OpenAICompatibleAdapter.request_body(self, messages, tools)
        if self.temperature is None:
            body.pop("temperature")
        return body

    def complete(self, messages, tools):
        require(self.policy is not None, "Original buyer binding absent")
        original = self.policy.transcript[-1]
        require(original["kind"] == "model_request", "IPC must follow original fitted request")
        body = self.request_body(messages, tools)
        require(body == original["request"], "Rebuilt original request changed before dispatch")
        day, turn = original["day"], original["turn"]
        identity = (day, turn, digest(body))
        if identity in self.dispatched:
            raise BridgeFailure("Repeated original logical request; no new sample allowed")
        self.dispatched.add(identity)  # Failure is terminal; this call is never retried.
        key = f"day-{day:03d}-turn-{turn:03d}-{identity[2][:16]}"
        event = {
            "event": "model_request",
            "request_key": key,
            "day": day,
            "turn": turn,
            "messages": messages,
            "tools": tools,
            "request_body": body,
            "request_sha256": identity[2],
        }
        encoded = json.dumps(event, ensure_ascii=False) + "\n"
        if len(encoded.encode()) > self.max_line_bytes:
            raise BridgeFailure("Original request exceeds explicitly bound IPC envelope")
        self.output.write(encoded)
        self.output.flush()
        try:
            ready, _, _ = select.select([self.input], [], [], self.timeout_seconds)
            if not ready:
                raise BridgeFailure("Owned model-response deadline exhausted")
            raw = self.input.readline(self.max_line_bytes + 1)
            if not raw or not raw.endswith("\n") or len(raw.encode()) > self.max_line_bytes:
                raise BridgeFailure("Missing or oversized original model response")
            envelope = json.loads(raw)
            if envelope.get("abort"):
                raise BridgeFailure("Parent ended this original logical request")
            if envelope.get("event") != "model_response" or envelope.get("request_key") != key:
                raise BridgeFailure("Mismatched original response identity")
            response = envelope["response"]
            if not isinstance(response, dict):
                raise BridgeFailure("Original response must be an object")
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise BridgeFailure("Invalid owned model-response envelope") from error
        self.receipts.append(
            {
                "request_key": key,
                "day": day,
                "turn": turn,
                "request_sha256": identity[2],
                "response_sha256": digest(response),
                "provenance": envelope.get("provenance"),
                "archived_replay_only": self.replay_only,
            }
        )
        # Neither response text nor tool arguments are edited. The unchanged
        # LLMBuyerPolicy performs its original validity and bookkeeping checks.
        return response
