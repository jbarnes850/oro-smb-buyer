"""Schema-aware XML argument decoding beside the unchanged official Qwen renderer.

Prompt/output formatting, thinking/framing parsing and native tokens are unchanged.
Only execution-side argument conversion differs: strings retain their exact text;
boolean schema fields also recognize the pinned formatter's True/False spelling.
The original engine still validates names, argument types, bounds and business rules.
"""

from __future__ import annotations

import json
import re
from copy import copy
from types import MethodType

from tinker_cookbook.renderers.base import ToolCall, ToolSpec, UnparsedToolCall
from tinker_cookbook.renderers.qwen3_5 import _FUNCTION_BLOCK_RE, Qwen3_5Renderer

# Unlike the official regex, do not consume arbitrary whitespace from the value.
_PARAMETERS = re.compile(r"<parameter=(?P<name>[^>\n]+)>(?P<value>.*?)</parameter>", re.DOTALL)
_RESERVED = (
    "<tool_call>",
    "</tool_call>",
    "<function=",
    "</function>",
    "<parameter=",
    "</parameter>",
)


def _schemas(tools: list[ToolSpec]) -> dict:
    result = {}
    for tool in tools:
        name = tool["name"]
        if name in result:
            raise ValueError("Duplicate original tool schema")
        result[name] = tool["parameters"].get("properties", {})
    return result


def _parse_block(raw_text: str, schemas: dict) -> ToolCall | UnparsedToolCall:
    def malformed(reason):
        return UnparsedToolCall(raw_text=raw_text, error=reason)

    match = _FUNCTION_BLOCK_RE.match(raw_text)
    if not match:
        return malformed("Malformed Qwen tool-call XML")
    name, body = match.group("name").strip(), match.group("body")
    if not name:
        return malformed("Missing function name")
    properties = schemas.get(name, {})
    arguments, pos = {}, 0
    for parameter in _PARAMETERS.finditer(body):
        if body[pos : parameter.start()].strip():
            return malformed("Unexpected non-parameter content")
        key, value = parameter.group("name").strip(), parameter.group("value")
        if not key:
            return malformed("Empty parameter name")
        # A repeated parameter keeps the last value, as in Qwen's reference parser
        # (dict assignment). Rejecting identical repeats produced false model
        # failures, e.g. a duplicated idempotency_key with the same value.
        if any(delimiter in value for delimiter in _RESERVED):
            return malformed("Unescaped structural delimiter inside parameter")
        # The official formatter emits LF + value + LF. Match Qwen's reference XML
        # tool parser (Qwen3-Coder parser in vLLM/SGLang): strip at most one
        # leading and one trailing framing LF, independently. (Rejecting an
        # "unpaired" newline turned valid calls such as
        # "<parameter=sql>\nSELECT ...</parameter>" into false model failures.)
        if value.startswith("\n"):
            value = value[1:]
        if value.endswith("\n"):
            value = value[:-1]
        declared = properties.get(key, {}).get("type")
        if declared == "string":
            parsed = value
        elif declared == "boolean" and value in {"True", "False"}:
            # Exact Python spellings emitted by the installed official formatter.
            # Never turn 0/1 or a quoted string into a boolean.
            parsed = value == "True"
        else:
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = value
        arguments[key] = parsed
        pos = parameter.end()
    if body[pos:].strip():
        return malformed("Unexpected trailing content inside function")
    return ToolCall(function=ToolCall.FunctionBody(name=name, arguments=json.dumps(arguments)))


def parse_response(renderer: Qwen3_5Renderer, tokens: list[int], tools: list[ToolSpec]):
    """Parse current native tokens without changing renderer state or token arrays.

    A shallow local renderer copy changes only its XML argument conversion method;
    the installed parse_response still performs all framing/thinking/tool-block
    extraction and reports malformed/unclosed calls. No custom training renderer
    is registered, and no sampled or historical token is reconstructed.

    Fail-closed syntax constraint: unescaped nested structural delimiters inside a
    parameter value are rejected, since literal delimiters are not safely
    representable in this unescaped XML format. A repeated parameter keeps its last
    value and one framing newline is stripped on each side, as in Qwen's reference
    parser. This cannot infer intent for text that itself forms valid XML syntax.
    """
    if not isinstance(renderer, Qwen3_5Renderer):
        raise TypeError("This boundary is only for the pinned official Qwen3.5/3.6 renderer")
    schemas = _schemas(tools)
    original = tuple(tokens)
    local = copy(renderer)
    local._parse_qwen3_5_tool_call_xml = MethodType(
        lambda self, raw_text: _parse_block(raw_text, schemas), local
    )
    result = local.parse_response(tokens)
    if tuple(tokens) != original:
        raise ValueError("Official parser changed native token IDs")
    return result


def with_tool_parser(renderer: Qwen3_5Renderer, tools: list[ToolSpec]) -> Qwen3_5Renderer:
    """Bind the same conversion once on a private fresh renderer copy.

    Use for the live formatter; its official prompt/output methods and stop tokens
    remain unchanged. This is not a pickle-stable custom renderer registration.
    Construct/bind explicitly in each process that will decode responses.
    """
    if not isinstance(renderer, Qwen3_5Renderer):
        raise TypeError("This boundary is only for the pinned official Qwen3.5/3.6 renderer")
    schemas = _schemas(tools)
    local = copy(renderer)
    local._parse_qwen3_5_tool_call_xml = MethodType(
        lambda self, raw_text: _parse_block(raw_text, schemas), local
    )
    local._full_buyer_schemas = schemas  # for reference_parse in reference mode
    return local


# --- Reference-parser mode --------------------------------------------------------
# Qwen's serving-stack parser (vLLM ``--tool-call-parser qwen3_coder``, implemented as
# vllm/parser/qwen3.py: a lenient state machine) never rejects an output: a parameter
# value ends at ``</parameter>`` or at the next ``<parameter=``; text between
# parameters is ignored; ``</function>`` or the next ``<function=`` ends a call; a
# missing ``</tool_call>`` between calls is tolerated; a call still open at the end of
# the output is emitted with the parameters it has (including the open trailing one,
# as the non-streaming path keeps the streamed partial value). ``reference_parse``
# reproduces that non-streaming outcome for the original tool schemas. It is used only
# when the strict boundary above reports a malformed or unterminated call and the run
# declares SMB_PARSE_MODE=reference (``--parse-mode reference``); native tokens are
# never changed.

_REF_PARAM = re.compile(
    r"<\s*parameter\s*=\s*([^>]*)>(.*?)(?:<\s*/\s*parameter\s*>|(?=<\s*parameter\s*=))", re.DOTALL
)
_REF_PARTIAL = re.compile(r"<\s*parameter\s*=\s*([^>]+)>(.*)$", re.DOTALL)
_REF_OUTER = re.compile(r"<tool_call>|</tool_call>|<function=|</think>")
_REF_NAME = re.compile(r">|</function>|</tool_call>|<tool_call>|<function=")
_REF_ARGS = re.compile(r"</function>|<function=")
_TURN_BOUNDARIES = ("<|im_end|>", "<|im_start|>", "<|endoftext|>")


def _trim_wrapping_newlines(value: str) -> str:
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value


def _reference_arguments(raw_args: str, properties: dict) -> dict:
    params, outside, pos = {}, [], 0
    for match in _REF_PARAM.finditer(raw_args):
        outside.append(raw_args[pos : match.start()])
        params[match.group(1).strip()] = _trim_wrapping_newlines(match.group(2))
        pos = match.end()
    open_param = _REF_PARTIAL.search("".join(outside) + raw_args[pos:])
    if open_param is not None and open_param.group(1).strip():
        params[open_param.group(1).strip()] = _trim_wrapping_newlines(open_param.group(2))
    typed = {}
    for key, value in params.items():
        declared = properties.get(key, {}).get("type")
        if declared == "string":
            typed[key] = value
        elif declared == "boolean" and value in {"True", "False"}:
            typed[key] = value == "True"
        else:
            try:
                typed[key] = json.loads(value)
            except json.JSONDecodeError:
                typed[key] = value
    return typed


def reference_parse(text: str, schemas: dict, thinking: bool):
    """Return (reasoning, content, [ToolCall]) as Qwen's reference parser would."""
    for boundary in _TURN_BOUNDARIES:
        cut = text.find(boundary)
        if cut >= 0:
            text = text[:cut]
    reasoning, rest = "", text
    if thinking:
        ends = [i for i in (text.find("</think>"), text.find("<tool_call>")) if i >= 0]
        if not ends:
            return text.removeprefix("<think>"), "", []
        end = min(ends)
        reasoning = text[:end].removeprefix("<think>")
        rest = text[end + len("</think>"):] if text.startswith("</think>", end) else text[end:]
    calls, content, state, i, current = [], [], "CONTENT", 0, None

    def finish():
        name = current["name"].strip()
        if name:
            arguments = _reference_arguments(current["args"], schemas.get(name, {}))
            calls.append(ToolCall(function=ToolCall.FunctionBody(name=name, arguments=json.dumps(arguments))))

    while i < len(rest):
        if state in ("CONTENT", "PREAMBLE", "BETWEEN"):
            m = _REF_OUTER.search(rest, i)
            if m is None:
                if state == "CONTENT":
                    content.append(rest[i:])
                break
            if state == "CONTENT":
                content.append(rest[i : m.start()])
            token, i = m.group(0), m.end()
            if token == "</think>":
                continue  # a duplicate closer is dropped
            if token == "<function=":
                current, state = {"name": "", "args": ""}, "NAME"
            elif token == "<tool_call>" and state != "PREAMBLE":
                state = "PREAMBLE"
            elif token == "</tool_call>":
                if state == "CONTENT":
                    content.append(token)
                state = "CONTENT"
        elif state == "NAME":
            m = _REF_NAME.search(rest, i)
            if m is None:
                current["name"] += rest[i:]
                i = len(rest)
                break
            current["name"] += rest[i : m.start()]
            token, i = m.group(0), m.end()
            if token == ">":
                state = "ARGS"
                continue
            finish()
            current = None
            if token == "</function>":
                state = "BETWEEN"
            elif token == "</tool_call>":
                state = "CONTENT"
            elif token == "<tool_call>":
                state = "PREAMBLE"
            else:
                current, state = {"name": "", "args": ""}, "NAME"
        else:  # ARGS
            m = _REF_ARGS.search(rest, i)
            if m is None:
                current["args"] += rest[i:]
                i = len(rest)
                break
            current["args"] += rest[i : m.start()]
            token, i = m.group(0), m.end()
            finish()
            current = None
            if token == "</function>":
                state = "BETWEEN"
            else:
                current, state = {"name": "", "args": ""}, "NAME"
    if current is not None and state in ("NAME", "ARGS"):
        finish()
    return reasoning, "".join(content).strip(), calls
