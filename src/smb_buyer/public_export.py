"""Public-only conversion of an archived buyer request and response; no provider imports.

``convert_request`` keeps only the actual request's public content (system text,
tool schemas, public message text, ordered typed tool calls and tool results) and
drops provider reasoning; ``target`` turns a recorded response into the assistant
message to supervise. Chat-completions, Anthropic Messages and OpenAI Responses
request shapes are accepted; anything unrecognized raises instead of being guessed.
"""

from __future__ import annotations

import collections
import json

PRIVATE = {
    "reasoning",
    "reasoning_content",
    "reasoning_text",
    "reasoning_details",
    "reasoning_summary",
    "thinking",
    "signature",
    "redacted_thinking",
    "provider_items",
    "provider_metadata",
}


def check_keys(x, allowed, where):
    assert isinstance(x, dict), (where, type(x).__name__)
    unknown = set(x) - set(allowed)
    assert not unknown, (where, sorted(unknown))


def tool_call(cid, name, args):
    assert isinstance(cid, str) and cid and isinstance(name, str) and name
    if isinstance(args, str):
        obj = json.loads(args)
        encoded = args
    else:
        obj = args
        encoded = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
    assert isinstance(obj, dict), ("non-object arguments", name)
    return {"type": "function", "id": cid, "function": {"name": name, "arguments": encoded}}


def public_text(value, skips, where):
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    assert isinstance(value, list), (where, type(value).__name__)
    out = []
    for block in value:
        typ = block.get("type")
        if typ in ("thinking", "redacted_thinking"):
            skips[typ] += 1
            continue
        assert typ == "text", (where, "non-public-text", typ)
        check_keys(block, {"type", "text", "cache_control"}, where)
        assert isinstance(block["text"], str)
        out.append({"type": "text", "text": block["text"]})
    return out


def responses_to_public_request(request):
    """Project Responses API items to public ordered message groups; no reasoning replay."""
    tools = []
    for t in request["tools"]:
        check_keys(t, {"type", "name", "description", "parameters", "strict"}, "Responses tool")
        assert t["type"] == "function"
        tools.append(
            {"type": "function", "function": {k: t[k] for k in ("name", "description", "parameters")}}
        )
    messages = [{"role": "system", "content": request["instructions"]}]
    skipped = collections.Counter()
    pending = None

    def flush():
        nonlocal pending
        if pending is not None:
            messages.append(pending)
            pending = None

    for item in request["input"]:
        kind = item.get("type", "message")
        if kind == "reasoning":
            skipped["responses_reasoning_item"] += 1
            continue
        if kind == "function_call":
            check_keys(
                item, {"type", "arguments", "call_id", "id", "name", "status"}, "Responses function call"
            )
            if pending is None:
                pending = {"role": "assistant", "content": "", "tool_calls": []}
            pending.setdefault("tool_calls", []).append(
                tool_call(item["call_id"], item["name"], item["arguments"])
            )
        elif kind == "function_call_output":
            flush()
            check_keys(item, {"type", "call_id", "output"}, "Responses function output")
            assert isinstance(item["output"], str)
            messages.append({"role": "tool", "tool_call_id": item["call_id"], "content": item["output"]})
        elif kind == "message":
            check_keys(item, {"type", "role", "content", "id", "status", "phase"}, "Responses message")
            content = item["content"]
            role = item["role"]
            assert role in ("user", "assistant")
            if isinstance(content, list):
                parts = []
                for z in content:
                    assert z.get("type") in ("input_text", "output_text", "text"), (
                        "Responses unsupported public block",
                        z.get("type"),
                    )
                    check_keys(z, {"type", "text", "annotations", "logprobs"}, "Responses text")
                    assert not z.get("annotations"), "public text annotations need explicit mapping"
                    parts.append({"type": "text", "text": z["text"]})
                content = parts
            assert isinstance(content, (str, list))
            if role == "assistant":
                # Preserve order even if an unusual source interleaves text after calls.
                flush()
                pending = {"role": "assistant", "content": content}
            else:
                flush()
                messages.append({"role": role, "content": content})
        else:
            raise AssertionError(("Responses unknown input item", kind))
    flush()
    return {"messages": messages, "tools": tools}, dict(skipped)


def convert_request(request):
    """Keep only the actual request's public content, without rebuilding history."""
    skips = collections.Counter()
    tools, msgs, system, name_by_id = [], [], [], {}
    if "input" in request:
        request, removed = responses_to_public_request(request)
        skips.update(removed)
    # Top-level inference settings and provider routing are metadata, not conversation.
    for t in request["tools"]:
        if t.get("type") == "function":
            check_keys(t, {"type", "function"}, "chat tool")
            v = t["function"]
            schema = v["parameters"]
        else:
            v = t
            schema = v["input_schema"]
        check_keys(v, {"name", "description", "parameters", "input_schema", "cache_control"}, "tool schema")
        assert isinstance(v["name"], str) and isinstance(v["description"], str) and isinstance(schema, dict)
        tools.append({"name": v["name"], "description": v["description"], "parameters": schema})
    if "system" in request:
        s = public_text(request["system"], skips, "system")
        system.append(s if isinstance(s, str) else "".join(z["text"] for z in s))
    for source in request["messages"]:
        check_keys(
            source, {"role", "content", "tool_calls", "tool_call_id", "name", "cache_control"} | PRIVATE, "message"
        )
        skips.update(k for k in source if k in PRIVATE)
        role = source["role"]
        content = source.get("content")
        assert role in ("system", "user", "assistant", "tool")
        if role == "system":
            s = public_text(content, skips, "system message")
            system.append(s if isinstance(s, str) else "".join(z["text"] for z in s))
            continue
        if role == "assistant":
            text, calls = [], []
            if isinstance(content, list):
                seen_call = False
                for block in content:
                    typ = block.get("type")
                    if typ in ("thinking", "redacted_thinking"):
                        skips[typ] += 1
                        continue
                    if typ == "text":
                        # Official public Message represents text before calls; reject reordering.
                        assert not seen_call, "interleaved assistant text after tool call"
                        check_keys(block, {"type", "text", "cache_control"}, "assistant text")
                        text.append({"type": "text", "text": block["text"]})
                    elif typ == "tool_use":
                        seen_call = True
                        check_keys(
                            block, {"type", "id", "name", "input", "caller", "cache_control"}, "tool use"
                        )
                        calls.append(tool_call(block["id"], block["name"], block["input"]))
                    else:
                        raise AssertionError(("unknown assistant block", typ))
                c = text
            else:
                c = public_text(content, skips, "assistant")
            for tc in source.get("tool_calls", []):
                check_keys(tc, {"type", "id", "function"}, "chat call")
                assert tc.get("type") == "function"
                check_keys(tc["function"], {"name", "arguments"}, "chat call body")
                calls.append(tool_call(tc["id"], tc["function"]["name"], tc["function"]["arguments"]))
            out = {"role": "assistant", "content": c}
            if calls:
                out["tool_calls"] = calls
                for tc in calls:
                    name_by_id[tc["id"]] = tc["function"]["name"]
            msgs.append(out)
        elif role == "tool":
            cid = source["tool_call_id"]
            assert cid in name_by_id, ("unknown tool result", cid)
            name = source.get("name", name_by_id[cid])
            assert name == name_by_id[cid]
            msgs.append(
                {
                    "role": "tool",
                    "content": public_text(content, skips, "tool result"),
                    "tool_call_id": cid,
                    "name": name,
                }
            )
        elif isinstance(content, list):
            # Anthropic groups tool results into user blocks; expand in exact block order.
            pending = []

            def flush():
                if pending:
                    msgs.append({"role": "user", "content": list(pending)})
                    pending.clear()

            for block in content:
                typ = block.get("type")
                if typ == "text":
                    check_keys(block, {"type", "text", "cache_control"}, "user text")
                    pending.append({"type": "text", "text": block["text"]})
                elif typ == "tool_result":
                    flush()
                    check_keys(
                        block, {"type", "tool_use_id", "content", "is_error", "cache_control"}, "tool result"
                    )
                    cid = block["tool_use_id"]
                    assert cid in name_by_id, ("unknown Anthropic tool result", cid)
                    # is_error is provider framing. Actual public result text including
                    # ok/reason remains literal.
                    assert not block.get("is_error", False), (
                        "provider error flag requires explicit public representation"
                    )
                    msgs.append(
                        {
                            "role": "tool",
                            "content": public_text(block["content"], skips, "tool result content"),
                            "tool_call_id": cid,
                            "name": name_by_id[cid],
                        }
                    )
                else:
                    raise AssertionError(("unknown user block", typ))
            flush()
        else:
            msgs.append({"role": "user", "content": public_text(content, skips, "user")})
    assert len(system) == 1 and system[0]
    return system[0], tools, msgs, dict(skips)


def target(response):
    assert isinstance(response.get("content"), str)
    out = {"role": "assistant", "content": response["content"]}
    calls = [tool_call(z["id"], z["name"], z["arguments"]) for z in response.get("tool_calls", [])]
    if calls:
        out["tool_calls"] = calls
    assert out["content"] or calls, "empty public target"
    return out
