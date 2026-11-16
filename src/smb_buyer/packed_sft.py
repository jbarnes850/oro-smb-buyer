"""Day-packed public-action SFT from economically selected frontier TRAIN episodes.

Source rows use the public exporter (``public_export.convert_request``/``target``):
public text, ordered typed tool calls and tool results only; provider reasoning is
removed and nothing is fabricated. Within one morning, consecutive requests whose
messages are exact prefixes of the next (after normalizing empty content-block
lists to ``""``) are packed into one official rendering of the segment's last
request. Every assistant message that is an actual target of a row in that
segment is supervised with unit total weight; condensed history from before a
segment boundary is context only. The final turn's generated empty-thinking
closure is masked. Targets with any public ``ok: false`` tool result receive zero
loss. Each supervised span is parsed back with the schema-aware Qwen parser and
must reproduce the target's ordered calls exactly.

The warm start was run on Qwen3.6-35B-A3B with its thinking renderer; the corpus is
rendered with that tokenizer and renderer (``SFT_BACKBONE``).
"""

from __future__ import annotations

import collections
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np

from . import public_export
from .backbones import BACKBONES

SFT_BACKBONE = "qwen3.6"
CLOSURE_TEXT = "\n</think>\n\n"
_PARSERS = {}


def sft_renderer():
    """The official thinking renderer of the SFT backbone."""
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    _, tokenizer_model, by_mode = BACKBONES[SFT_BACKBONE]
    return renderers.get_renderer(
        by_mode["on"], get_tokenizer(tokenizer_model), model_name=tokenizer_model
    )


def norm(message):
    """Normalize representation-only differences between history and targets.

    Empty/listed content blocks become one string, and tool-call arguments are
    re-encoded canonically (the Qwen renderer parses arguments before formatting,
    so this changes no rendered token; it only makes prefix comparison exact).
    """
    message = dict(message)
    if isinstance(message.get("content"), list):
        message["content"] = "".join(part["text"] for part in message["content"])
    if message.get("tool_calls"):
        calls = []
        for call in message["tool_calls"]:
            call = {**call, "function": dict(call["function"])}
            call["function"]["arguments"] = json.dumps(
                json.loads(call["function"]["arguments"]), sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
            calls.append(call)
        message["tool_calls"] = calls
    return message


def gemini_to_chat(request):
    """Project a native Gemini request to the chat shape the exporter accepts.

    Public text and function calls only: ``thought`` parts and thought signatures
    are dropped (private reasoning). Function responses are re-serialized with the
    harness's own chat encoding, ``json.dumps(result, sort_keys=True)``.
    """
    declarations = [d for tool in request["tools"] for d in tool.get("functionDeclarations", [])]
    tools = [
        {
            "type": "function",
            "function": {
                "name": d["name"],
                "description": d["description"],
                "parameters": d.get("parametersJsonSchema") or d.get("parameters"),
            },
        }
        for d in declarations
    ]
    system = "".join(part.get("text", "") for part in request["systemInstruction"]["parts"])
    messages = []
    for content in request["contents"]:
        parts = content["parts"]
        if content["role"] == "model":
            text = "".join(p["text"] for p in parts if "text" in p and not p.get("thought"))
            calls = [
                {
                    "type": "function",
                    "id": p["functionCall"]["id"],
                    "function": {
                        "name": p["functionCall"]["name"],
                        "arguments": p["functionCall"].get("args", {}),
                    },
                }
                for p in parts
                if "functionCall" in p
            ]
            message = {"role": "assistant", "content": text}
            if calls:
                message["tool_calls"] = calls
            messages.append(message)
            continue
        texts = [{"type": "text", "text": p["text"]} for p in parts if "text" in p]
        responses = [p["functionResponse"] for p in parts if "functionResponse" in p]
        if texts and responses:
            raise ValueError("Mixed Gemini user text and function responses")
        for response in responses:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": response["id"],
                    "name": response["name"],
                    "content": json.dumps(response["response"], sort_keys=True),
                }
            )
        if texts:
            messages.append({"role": "user", "content": texts})
    return {"system": system, "tools": tools, "messages": messages}


def iter_rows(archive_path, exporter):
    """(day, turn, system, tools, messages+[target]) for each answered request."""
    with tarfile.open(archive_path, "r:gz") as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith("model_transcript.jsonl"))
        pending = None
        for line in tar.extractfile(member):
            record = json.loads(line)
            kind = record.get("kind")
            if kind == "model_request":
                pending = record
            elif kind == "model" and pending is not None:
                if (record["day"], record["turn"]) != (pending["day"], pending["turn"]):
                    raise ValueError("Request/response order differs")
                request = pending["request"]
                if "contents" in request:
                    request = gemini_to_chat(request)
                system, tools, messages, _ = exporter.convert_request(request)
                try:
                    final = exporter.target(record)
                except AssertionError:
                    pending = None  # empty public reply: original nudge, no target
                    continue
                yield {
                    "day": record["day"],
                    "turn": record["turn"],
                    "system": system,
                    "tools": tools,
                    "messages": [norm(m) for m in messages] + [norm(final)],
                }
                pending = None


def segments(rows):
    """Maximal same-day chains where each request extends the previous one."""
    by_day = collections.defaultdict(list)
    for row in rows:
        by_day[row["day"]].append(row)
    out = []
    for day in sorted(by_day):
        chain = []
        for row in sorted(by_day[day], key=lambda r: r["turn"]):
            if chain and row["messages"][: len(chain[-1]["messages"])] != chain[-1]["messages"]:
                out.append(chain)
                chain = []
            chain.append(row)
        if chain:
            out.append(chain)
    return out


def errored_call_ids(rows):
    bad = set()
    for row in rows:
        for message in row["messages"]:
            if message.get("role") == "tool":
                try:
                    result = json.loads(message["content"])
                except (TypeError, ValueError):
                    continue
                if isinstance(result, dict) and result.get("ok") is False:
                    bad.add(message.get("tool_call_id"))
    return bad


def render_segment(chain, renderer, policy_text, bad_calls, max_tokens):
    from tinker_cookbook.renderers import TrainOnWhat, ToolCall

    from .qwen_tools import with_tool_parser

    last = chain[-1]
    key = (id(renderer), hashlib.sha256(json.dumps(last["tools"], sort_keys=True).encode()).hexdigest())
    if key not in _PARSERS:
        _PARSERS[key] = with_tool_parser(renderer, last["tools"])
    parser = _PARSERS[key]
    messages = []
    for message in last["messages"]:
        message = dict(message)
        if "tool_calls" in message:
            message["tool_calls"] = [ToolCall.model_validate(c) for c in message["tool_calls"]]
        messages.append(message)
    prefix = renderer.create_conversation_prefix_with_tools(last["tools"], policy_text)
    model_input, weights = renderer.build_supervised_example(
        prefix + messages, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )
    ids = model_input.to_ints()
    if len(ids) > max_tokens:
        return None, {"dropped_over_length": len(ids)}
    w = weights.numpy().astype(np.float64)
    spans, i = [], 0
    while i < len(w):
        if w[i] > 0:
            j = i
            while j < len(w) and w[j] > 0:
                j += 1
            spans.append((i, j))
            i = j
        else:
            i += 1
    assistant_positions = [k for k, m in enumerate(last["messages"]) if m["role"] == "assistant"]
    if len(spans) != len(assistant_positions):
        raise ValueError("Assistant span count differs from assistant messages")
    target_positions = {len(row["messages"]) - 1 for row in chain}
    closure = renderer.tokenizer.encode(CLOSURE_TEXT, add_special_tokens=False)
    out = np.zeros(len(ids), dtype=np.float32)
    audit = collections.Counter()
    for (start, end), position in zip(spans, assistant_positions):
        if position not in target_positions:
            audit["context_only_assistant"] += 1
            continue
        target = last["messages"][position]
        public_start = start
        if ids[start : start + len(closure)] == closure:
            public_start = start + len(closure)  # generated closure: never supervised
            parse_tokens = ids[start:end]
        else:
            parse_tokens = closure + ids[start:end]
        parsed, termination = parser.parse_response(parse_tokens)
        expected = [
            (c["function"]["name"], json.loads(c["function"]["arguments"]))
            for c in target.get("tool_calls", [])
        ]
        actual = [
            (c.function.name, json.loads(c.function.arguments)) for c in parsed.get("tool_calls", [])
        ]
        if (
            not termination.is_clean
            or parsed.get("unparsed_tool_calls")
            or json.dumps(expected, sort_keys=True) != json.dumps(actual, sort_keys=True)
        ):
            raise ValueError("Packed rendering changed target tool semantics")
        if any(c["id"] in bad_calls for c in target.get("tool_calls", [])):
            audit["zero_loss_errored_target"] += 1
            continue
        n = end - public_start
        if n <= 0:
            raise ValueError("Empty public target span")
        out[public_start:end] = 1.0 / n
        audit["supervised_targets"] += 1
        audit["supervised_tokens"] += n
    return (np.asarray(ids, dtype=np.int32), out), dict(audit, tokens=len(ids))


_WORKER = {}


def _episode_worker(args):
    """Process-pool worker: render every segment of one archive."""
    archive, policy_text, max_tokens = args
    if not _WORKER:
        _WORKER["renderer"] = sft_renderer()
    rows = list(iter_rows(archive["path"], public_export))
    bad = errored_call_ids(rows)
    episode = collections.Counter(rows=len(rows))
    parts = []
    for chain in segments(rows):
        rendered, audit = render_segment(chain, _WORKER["renderer"], policy_text, bad, max_tokens)
        episode.update(audit)
        if rendered is None:
            episode["dropped_segments"] += 1
            continue
        ids, w = rendered
        if not (w > 0).any():
            episode["segments_without_supervision"] += 1
            continue
        parts.append((ids, w, chain[0]["day"], chain[0]["turn"]))
        episode["segments"] += 1
    return archive, parts, dict(episode)


def build_parallel(selection_path, output_dir, policy_prompt_path, *, holdout, workers=12, max_tokens=131072):
    import concurrent.futures

    policy_bytes = Path(policy_prompt_path).read_bytes()
    selection = json.loads(Path(selection_path).read_text())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "full-buyer-packed-econ-sft-v1",
        "selection": {"path": str(Path(selection_path).resolve()), "sha256": hashlib.sha256(Path(selection_path).read_bytes()).hexdigest()},
        "policy_prompt": {"path": str(Path(policy_prompt_path).resolve()), "sha256": hashlib.sha256(policy_bytes).hexdigest()},
        "backbone": SFT_BACKBONE,
        "renderer": f"{BACKBONES[SFT_BACKBONE][2]['on']} thinking; ALL_ASSISTANT_MESSAGES; final closure masked; unit mass per target; errored-call targets zero",
        "max_tokens": max_tokens,
        "holdout_trials": sorted(holdout),
        "episodes": [],
    }
    split_parts = {"train": [], "heldout": []}
    jobs = [(a, policy_bytes.decode(), max_tokens) for a in selection["archives"]]
    with concurrent.futures.ProcessPoolExecutor(workers) as pool:
        for archive, parts, episode in pool.map(_episode_worker, jobs):
            split = "heldout" if archive["trial_id"] in holdout else "train"
            split_parts[split].extend((archive["trial_id"], *p) for p in parts)
            manifest["episodes"].append({"split": split, **{k: archive[k] for k in ("trial_id", "model", "task_id", "reward")}, **episode})
            print(split, archive["model"], archive["task_id"], episode, flush=True)
    totals = collections.Counter()
    for split, parts in split_parts.items():
        offsets, cursor = [], 0
        for trial, ids, w, day, turn in parts:
            offsets.append([cursor, len(ids), trial, day, turn, int((w > 0).sum()), float(w.sum())])
            cursor += len(ids)
        np.save(output_dir / f"{split}-tokens.npy", np.concatenate([p[1] for p in parts]))
        np.save(output_dir / f"{split}-weights.npy", np.concatenate([p[2] for p in parts]))
        (output_dir / f"{split}-offsets.json").write_text(json.dumps(offsets))
        totals[f"{split}_segments"] = len(parts)
        totals[f"{split}_tokens"] = cursor
        totals[f"{split}_supervised_tokens"] = sum(o[5] for o in offsets)
        totals[f"{split}_targets"] = round(sum(o[6] for o in offsets))
    manifest["totals"] = dict(totals)
    manifest["files"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output_dir.iterdir())
        if p.suffix in (".npy", ".json") and p.name != "manifest.json"
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest
