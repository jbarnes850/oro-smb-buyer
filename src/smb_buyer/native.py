"""Native token boundary and terminal validation for the unchanged full SMB buyer.

``NativeBuyerRenderer`` turns each original buyer request (messages + tools) into the
official Tinker renderer's prompt tokens and decodes sampled tokens back into the
original response shape. ``original_completion_validator`` binds a finished quarter
to the engine's own independently graded result. No service creation, sampling,
optimizer or reward lives here. Training retains the original TokensWithLogprobs
in the official rollout, never this decoded copy.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from tinker_cookbook import renderers
from tinker_cookbook.renderers.base import ToolCall, get_text_content


class NativeBuyerContractError(ValueError):
    """A local protocol failure, not a replacement sample or numeric reward."""


def require(ok, detail):
    if not ok:
        raise NativeBuyerContractError(detail)


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


@dataclass(frozen=True)
class NativeBuyerConfig:
    model: str
    renderer_name: str
    reasoning_mode: str
    max_output_tokens: int
    context_window: int
    # Original release byte fitting remains in LLMBuyerPolicy._fit. A byte
    # allowance is NOT a proof that the rendered token prompt fits this model.
    max_request_bytes: int = 512000
    temperature: float = 1.0

    def __post_init__(self):
        require(isinstance(self.model, str) and self.model, "Declare actual model")
        require(
            isinstance(self.renderer_name, str) and self.renderer_name, "Declare official renderer"
        )
        require(self.reasoning_mode in {"on", "off"}, "Declare reasoning on or off")
        require(
            type(self.max_output_tokens) is int and self.max_output_tokens >= 16000,
            "Full-buyer release needs an explicit output allowance >=16000",
        )
        require(
            type(self.context_window) is int and self.context_window > self.max_output_tokens,
            "Declare actual model context with room for the prompt",
        )
        require(self.max_request_bytes == 512000, "Original release byte-fit bound changed")
        require(self.temperature == 1.0, "Unreviewed sampling temperature")

    @property
    def reasoning_replay(self):
        return "details_and_text" if self.reasoning_mode == "on" else None


def load_original_provider_functions(providers_path: Path, expected_sha256: str):
    """Load pure argument/body conversion, without creating an HTTP adapter."""
    require(
        hashlib.sha256(providers_path.read_bytes()).hexdigest() == expected_sha256,
        "Original provider source changed",
    )
    spec = importlib.util.spec_from_file_location("full_buyer_original_providers", providers_path)
    require(spec is not None and spec.loader is not None, "Cannot load original providers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def body_builder(messages, tools, config):
        settings = SimpleNamespace(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_output_tokens,
            reasoning_replay=config.reasoning_replay,
            provider_routing=None,
            reasoning_request=None,
            usage_include=None,
        )
        return module.OpenAICompatibleAdapter.request_body(settings, messages, tools)

    return module._arguments, body_builder


class NativeBuyerRenderer:
    def __init__(
        self, config: NativeBuyerConfig, tokenizer, original_arguments, original_body_builder
    ):
        self.config = config
        self.renderer = renderers.get_renderer(
            config.renderer_name, tokenizer, model_name=config.model
        )
        require(
            bool(
                getattr(
                    self.renderer,
                    "disables_thinking",
                    not getattr(self.renderer, "support_thinking", True),
                )
            )
            == (config.reasoning_mode == "off"),
            "Reasoning mode differs from official renderer",
        )
        self.original_arguments = original_arguments
        self.original_body_builder = original_body_builder

    def conversation(self, messages, tools):
        require(
            isinstance(messages, list) and messages and messages[0]["role"] == "system",
            "Effective buyer system message must be first",
        )
        require(all(m["role"] != "system" for m in messages[1:]), "Unexpected second system")
        require(
            isinstance(tools, list) and all(isinstance(t, dict) for t in tools),
            "Original tool definitions required",
        )
        require(len({t["name"] for t in tools}) == len(tools), "Duplicate tool names")
        # Official tool declaration, preserving the actual effective system text and original schemas.
        result = self.renderer.create_conversation_prefix_with_tools(
            deepcopy(tools), system_prompt=messages[0]["content"]
        )
        for original in messages[1:]:
            role = original["role"]
            require(role in {"user", "assistant", "tool"}, "Unsupported original role")
            content = original.get("content") or ""
            require(isinstance(content, str), "Original conversation content must be text")
            message = {"role": role, "content": content}
            if role == "assistant":
                thinking = original.get("reasoning_text") or ""
                require(
                    not original.get("reasoning_details"),
                    "Opaque provider reasoning cannot be translated into native tokens",
                )
                require(
                    not thinking or self.config.reasoning_mode == "on",
                    "Thinking history supplied in explicit off mode",
                )
                if thinking:
                    message["content"] = [
                        {"type": "thinking", "thinking": thinking},
                        {"type": "text", "text": content},
                    ]
                if original.get("tool_calls"):
                    message["tool_calls"] = [
                        ToolCall(
                            id=call.get("id"),
                            function=ToolCall.FunctionBody(
                                name=call["name"],
                                arguments=(
                                    json.dumps(call["arguments"])
                                    if isinstance(call.get("arguments"), dict)
                                    else call.get("arguments", "")
                                ),
                            ),
                        )
                        for call in original["tool_calls"]
                    ]
            if role == "tool":
                message["tool_call_id"] = original["tool_call_id"]
                message["name"] = original["name"]
            result.append(message)
        return result

    def render_request(self, event):
        require(event.get("event") == "model_request", "Expected original buyer request")
        body = event["request_body"]
        require(digest(body) == event["request_sha256"], "Original request hash changed")
        require(
            body == self.original_body_builder(event["messages"], event["tools"], self.config),
            "IPC messages/tools do not reproduce original request body",
        )
        require(
            len(canonical_bytes(body)) <= self.config.max_request_bytes,
            "Original byte-fit allowance exceeded",
        )
        require(body.get("model") == self.config.model, "Requested model mismatch")
        require(
            body.get("max_tokens") == self.config.max_output_tokens,
            "Original/native output allowances differ",
        )
        require(body.get("temperature") == self.config.temperature, "Decoder mismatch")
        messages = self.conversation(event["messages"], event["tools"])
        prompt = self.renderer.build_generation_prompt(messages)
        require(
            prompt.length + self.config.max_output_tokens <= self.config.context_window,
            "Rendered context exceeds declared model bound; no extra clipping or sample",
        )
        metadata = {
            "request_key": event["request_key"],
            "request_sha256": event["request_sha256"],
            "original_messages_sha256": digest(event["messages"]),
            "original_tools_sha256": digest(event["tools"]),
            "prompt_token_ids_sha256": digest(prompt.to_ints()),
            "prompt_tokens": prompt.length,
            "max_output_tokens": self.config.max_output_tokens,
            "context_window": self.config.context_window,
            "reasoning_mode": self.config.reasoning_mode,
            "renderer": self.config.renderer_name,
            "tokenizer_class": type(self.renderer.tokenizer).__module__
            + "."
            + type(self.renderer.tokenizer).__qualname__,
            "renderer_claims_extension": self.renderer.has_extension_property,
            "original_byte_fitting": True,
            "additional_native_trimming": False,
        }
        return prompt, metadata

    def decode_response(self, tokens, stop_reason, prompt_tokens):
        require(
            isinstance(tokens, list) and all(type(t) is int and t >= 0 for t in tokens),
            "Action must be original token IDs",
        )
        require(stop_reason in {"stop", "length"}, "Actual native stop reason required")
        frozen = tuple(tokens)
        # Reference-parser alignment: a turn ended by the model's EOS token instead
        # of the chat stop token is parsed as ended; sampled tokens are unchanged in
        # the native record.
        parse_tokens = list(tokens)
        tok = self.renderer.tokenizer
        enders = {i for i in (getattr(tok, "eos_token_id", None), tok.convert_tokens_to_ids("<|endoftext|>")) if isinstance(i, int)}
        stops = [s for s in self.renderer.get_stop_sequences() if isinstance(s, int)]
        eos_terminated = bool(parse_tokens) and bool(stops) and parse_tokens[-1] in enders - set(stops)
        if eos_terminated:
            parse_tokens[-1] = stops[0]
        message, termination = self.renderer.parse_response(parse_tokens)
        require(tuple(tokens) == frozen, "Renderer modified native action tokens")
        decoded_text = self.renderer.tokenizer.decode(list(tokens))
        unclosed_thinking = (
            self.config.reasoning_mode == "on"
            and stop_reason == "stop"
            and "</think>" not in decoded_text
            and "<tool_call>" not in decoded_text
        )
        if unclosed_thinking:
            # Qwen's reference reasoning parser returns this as reasoning with empty
            # content; the original harness logs an empty reply and nudges.
            thinking_only = decoded_text.replace("<|endoftext|>", "").replace("<|im_end|>", "")
            response = {
                "content": "",
                "tool_calls": [],
                "usage": {"input_tokens": prompt_tokens, "output_tokens": len(tokens)},
                "finish_reason": "stop",
                "native_finish_reason": stop_reason,
                "provider_status": "completed",
                "provider_metadata": {"model": self.config.model, "provider": "tinker"},
                "native_identity_basis": "caller-bound sampler; not a per-response model field",
                "reasoning_text": thinking_only,
            }
            diagnostics = {
                "native_stop_reason": stop_reason,
                "parser_clean": False,
                "unclosed_thinking_treated_as_empty_reply": True,
                "eos_terminated": eos_terminated,
                "unparsed_tool_calls": [],
                "parsed_calls_before_original_id_normalization": [],
                "sampled_tokens_sha256": digest(tokens),
                "failed_response_reward_assigned": False,
            }
            return response, diagnostics
        content = message.get("content", "")
        thinking = (
            "".join(p["thinking"] for p in content if p["type"] == "thinking")
            if isinstance(content, list)
            else ""
        )
        calls = [
            {
                "id": call.id,
                "name": call.function.name,
                "arguments": self.original_arguments(call.function.arguments),
            }
            for call in message.get("tool_calls", [])
        ]
        unparsed = [
            {"raw_text": item.raw_text, "error": item.error}
            for item in message.get("unparsed_tool_calls", [])
        ]
        reason = (
            "max_output_tokens"
            if stop_reason == "length"
            else "native_tool_parse_error"
            if unparsed
            else "native_message_framing_error"
            if not termination.is_clean
            else None
        )
        reference = None
        if reason in ("native_tool_parse_error", "native_message_framing_error") and (
            os.environ.get("SMB_PARSE_MODE", "strict") == "reference"
        ):
            # Declared measurement mode: Qwen's serving-stack parser (vLLM
            # qwen3_coder) recovers these outputs instead of failing the turn.
            from .qwen_tools import reference_parse

            ref_reasoning, ref_content, ref_calls = reference_parse(
                decoded_text,
                getattr(self.renderer, "_full_buyer_schemas", {}),
                self.config.reasoning_mode == "on",
            )
            reference = {"strict_reason": reason, "strict_errors": unparsed, "calls": len(ref_calls)}
            calls = [
                {
                    "id": call.id,
                    "name": call.function.name,
                    "arguments": self.original_arguments(call.function.arguments),
                }
                for call in ref_calls
            ]
            message = {"role": "assistant", "content": ref_content}
            thinking = ref_reasoning if self.config.reasoning_mode == "on" else thinking
            reason = None
        response = {
            "content": get_text_content(message),
            "tool_calls": calls,
            "usage": {"input_tokens": prompt_tokens, "output_tokens": len(tokens)},
            "finish_reason": "length"
            if stop_reason == "length"
            else "tool_calls"
            if calls
            else "stop",
            "native_finish_reason": stop_reason,
            "provider_status": "incomplete" if reason else "completed",
            "provider_metadata": {"model": self.config.model, "provider": "tinker"},
            "native_identity_basis": "caller-bound sampler; not a per-response model field",
        }
        if thinking and self.config.reasoning_mode == "on":
            response["reasoning_text"] = thinking
        elif thinking:
            # Thinking-off actor emitted thinking text anyway: the sampled tokens stay
            # in the native record, but it is not replayed as reasoning history.
            response["unreplayed_thinking_chars"] = len(thinking)
        if reason:
            response["incomplete_details"] = {"reason": reason}
        diagnostics = {
            "native_stop_reason": stop_reason,
            "parser_clean": termination.is_clean,
            "eos_terminated": eos_terminated,
            "unparsed_tool_calls": unparsed,
            "parsed_calls_before_original_id_normalization": deepcopy(calls),
            "sampled_tokens_sha256": digest(tokens),
            "failed_response_reward_assigned": False,
        }
        if reference is not None:
            diagnostics["reference_parser_recovery"] = reference
        # No inference that parsing success proves actual served-model identity.
        # The caller must bind the observed native sampler before constructing Env.
        return response, diagnostics


# ---------------------------------------------------------------- terminal validation


@dataclass(frozen=True)
class CompletedQuarter:
    return_cents: int
    result_ref: dict
    grade_ref: dict
    actual_sessions: int
    terminal_status: str = "completed"


def original_completion_validator(*, result_path, plan_ref, adoption_ref):
    """Bind the original independently graded worker result, without regrading.

    Caller must first admit the exact worker/source/plan. This checks literal
    outputs and their grade, not invented booleans from an IPC message.
    """

    def read_bound(reference):
        path = Path(reference["path"])
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == reference["sha256"], "Changed terminal artifact")
        return json.loads(raw)

    def validate(event):
        require(
            Path(event["result"]["path"]).resolve() == Path(result_path).resolve(),
            "Different owned terminal result",
        )
        result = read_bound(event["result"])
        require(
            result["plan"] == plan_ref and result["adoption"] == adoption_ref,
            "Terminal plan/adoption mismatch",
        )
        read_bound(plan_ref)
        read_bound(adoption_ref)
        require(
            result["status"] == "valid" and result["archived_replay_only"] is False,
            "Replay or invalid terminal result",
        )
        for key in (
            "return_cents",
            "sessions",
            "logical_model_requests",
            "agent_tool_calls",
            "budgeted_agent_tool_calls",
            "archived_replay_only",
        ):
            require(result[key] == event[key], "Terminal event does not match actual result")
        require(
            result["model_calls"] == result["logical_model_requests"],
            "Original outer provider retry/replacement occurred",
        )
        manifest = read_bound(result["run_manifest"])
        require(
            type(manifest["sessions"]) is int
            and 0 <= manifest["sessions"] <= 90
            and manifest["sessions"] == result["sessions"]
            and manifest["status"] in {"completed", "insolvent"}
            and manifest["episode_finished"] is True
            and manifest["capability_score_eligible"] is True,
            "Original terminal run incomplete",
        )
        grade = result["grade"]
        require(read_bound(result["grade_ref"]) == grade, "Original external grade differs")
        require(
            type(grade["cash"]["insolvent"]) is bool
            and (manifest["status"] == "insolvent") == grade["cash"]["insolvent"]
            and manifest["insolvency_day"] == grade["cash"]["insolvency_day"],
            "Actual terminal insolvency evidence disagrees",
        )
        require(
            grade["status"] == "valid"
            and not grade["invalid_reasons"]
            and grade["audit"]
            and all(row["pass"] is True and row["failures"] == 0 for row in grade["audit"]),
            "Original full financial/physical grader failed",
        )
        require(
            type(grade["value"]["business_return_cents"]) is int
            and grade["value"]["business_return_cents"] == result["return_cents"],
            "Original grade return differs",
        )
        require(
            grade["hashes"]["run_manifest_sha256"] == result["run_manifest"]["sha256"],
            "Grader is not bound to the actual terminal manifest",
        )
        for key in (
            "model_transcript",
            "tool_calls",
            "bridge_receipts",
            "events",
            "actions",
            "journal",
        ):
            path = Path(result[key]["path"])
            with path.open("rb") as stream:
                require(
                    hashlib.file_digest(stream, "sha256").hexdigest() == result[key]["sha256"],
                    "Changed original execution evidence",
                )
        return CompletedQuarter(
            result["return_cents"],
            event["result"],
            result["grade_ref"],
            manifest["sessions"],
            manifest["status"],
        )

    return validate
