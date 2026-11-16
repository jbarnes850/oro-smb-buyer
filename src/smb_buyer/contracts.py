"""Source and literal task binding for the original full buyer, without execution.

Imported by both the parent process and the engine child (``worker.py``), so it
depends on the standard library and ``backbones`` only, and reads every location
from the hash-bound plan rather than from the environment.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

from smb_buyer.backbones import BACKBONES, CONTEXT, REASONING_REPLAY

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ENVIRONMENT = "original-full-buyer-v11-native-ipc-v1"
ENGINE_COMMIT = "b17b864b63a756378cff74a6b9ed718f6823cd32"
# sha256 of the engine's own retailsim/harness/prompts/buyer-prompt-v11.txt. The v11
# environment and tools are kept; its system prompt is replaced by prompts/buyer_system.txt.
PROMPT_SHA = "33e67d9015816471e7b870acc8f619300f55ffb6271b1ee36b5d6993a5ded17e"
TOOLS = (
    "list_tables",
    "describe_table",
    "query_records",
    "list_offers",
    "place_purchase_order",
    "get_business_status",
    "advance_day",
    "set_shortfall_cover",
    "query_experience",
    "memory_list",
    "memory_read",
    "memory_search",
    "memory_write",
    "memory_append",
    "memory_edit",
    "memory_delete",
)
# Package modules the engine child executes; their hashes are bound in every plan.
IPC_MODULES = (
    "contracts.py",
    "backbones.py",
    "bridge.py",
    "worker.py",
    "owned_worker.py",
    "lifecycle.py",
)
SPLIT_BUSINESSES = {
    "train": ("b500", "b501", "b502", "b503"),
    "validation": ("b504",),
    "test": ("b505",),
}
sys.dont_write_bytecode = True


def require(ok, message):
    if not ok:
        raise ValueError(message)


def open_noatime(path):
    """Owned immutable source read; fail rather than silently change archive atime."""
    return os.fdopen(os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOATIME), "rb")


def sha(path):
    with open_noatime(path) as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ref(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def read(path):
    with open_noatime(path) as stream:
        return json.load(stream)


def bound(reference):
    require(ref(reference["path"]) == reference, "Changed bound file: " + reference["path"])
    return read(reference["path"])


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def verify_engine_source(source, engine):
    """Every hashed engine file under ``source`` must match the campaign's engine."""
    require(engine["commit"] == ENGINE_COMMIT, "Wrong engine revision")
    for name, checksum in {
        **engine["python_files_sha256"],
        **engine["closure_files_sha256"],
    }.items():
        require(sha(Path(source) / name) == checksum, "Original engine source changed: " + name)


def verify(plan_ref, adoption_ref=None):
    plan = bound(plan_ref)
    require(
        plan["environment_contract"] == ENVIRONMENT
        and plan["prompt_sha256"] == PROMPT_SHA
        and plan["tool_names"] == list(TOOLS),
        "Original full-buyer contract differs",
    )
    for filename, checksum in plan["source_hashes"].items():
        require(sha(filename) == checksum, "Changed execution source: " + filename)
    for name in IPC_MODULES:
        require(
            plan["source_hashes"].get(str(HERE / name)) == sha(HERE / name), "Unbound IPC source"
        )
    source = Path(plan["engine_source"])
    require(
        sha(source / "retailsim/harness/prompts/buyer-prompt-v11.txt") == PROMPT_SHA,
        "Original prompt asset differs",
    )
    assets = bound(plan["asset_manifest"])
    verify_engine_source(source, assets["engine"])
    registry = bound(plan["world_registry"])
    world = registry["worlds"][plan["world_ref"]]
    # TRAIN b500-b503 by default; VAL b504 and sealed TEST b505 only when the plan
    # declares that split (TEST additionally requires explicit final access).
    split = plan.get("split", "train")
    require(
        split in SPLIT_BUSINESSES
        and world["split"] == split
        and world["business"] in SPLIT_BUSINESSES[split],
        "World split/business differs from the plan's declared split",
    )
    require(
        split != "test" or plan.get("final_test_access") is True,
        "Sealed TEST requires explicit final access in the plan",
    )
    directory = Path(plan["world_asset_root"]) / world["path"]
    require(
        sha(Path(plan["world_asset_root"]) / world["manifest"]["path"])
        == plan["world_ref"]
        == world["manifest"]["sha256"],
        "World manifest differs",
    )
    for name, checksum in world["input_sha256"].items():
        require(sha(directory / name) == checksum, "World input differs: " + name)
    public = read(directory / "public/scenario_public.json")
    require(
        public["calendar"]["purchasing_days"] == 90
        and public["calendar"]["settlement_days"] == 0
        and public["tool_budget"] == {"max_query_rows": 200, "max_tool_calls_per_day": 60}
        and public["local_market"]["enabled"] is True
        and "stock_valuation" in public
        and not public["local_market"].get("shelf_cover", {}).get("enabled"),
        "Original v11 world pairing differs",
    )
    require(
        plan["history"] is None
        and plan["limits"]["max_turns_per_day"] == 100
        and plan["limits"]["max_tool_calls_per_day"] == 60
        and plan["limits"]["max_request_bytes"] == 512000
        and plan["adapter"]["max_output_tokens"] >= 16000,
        "Unreviewed full-buyer limits/history",
    )
    validate_scope(plan, plan_ref, bound(adoption_ref) if adoption_ref else None)
    prompt_path = Path(plan["system_prompt"]["path"])
    require(
        plan["system_prompt"] == ref(prompt_path)
        and plan["source_hashes"].get(plan["system_prompt"]["path"])
        == plan["system_prompt"]["sha256"],
        "Unbound system prompt (must be a hash-bound file)",
    )
    return plan, directory


def import_original(plan):
    source = Path(plan["engine_source"])
    prior = sys.modules.get("retailsim")
    require(
        prior is None or Path(prior.__file__).resolve().parent == source / "retailsim",
        "Another engine source is already imported",
    )
    sys.path.insert(0, str(source))
    from retailsim.harness.rollout import agent_tools

    require(tuple(t["name"] for t in agent_tools("v11")) == TOOLS, "Original16 tool order differs")


def install_system_prompt(policy, plan):
    """Use the plan-bound system prompt, retaining v11 solely for environment/tools."""
    prompt_path = Path(plan["system_prompt"]["path"])
    with open_noatime(prompt_path) as stream:
        raw = stream.read()
    effective = {"path": str(prompt_path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}
    require(
        plan["system_prompt"] == effective
        and plan["source_hashes"].get(str(prompt_path.resolve())) == effective["sha256"],
        "System prompt is not the exact plan-bound file",
    )
    text = raw.decode("utf-8")
    require(bool(text.strip()), "System prompt is empty")
    base = policy.prompt
    require(
        base.version == "v11" and base.sha256 == PROMPT_SHA,
        "Original v11 environment/tool pairing was not constructed",
    )
    resolved = prompt_path.resolve()
    identifier = str(resolved.relative_to(ROOT)) if resolved.is_relative_to(ROOT) else str(resolved)
    policy.system_prompt = text
    policy.prompt = base._replace(prompt_id=identifier, text=text, sha256=effective["sha256"])
    policy.config.update(
        prompt_version=identifier,
        system_prompt_sha256=effective["sha256"],
        system_prompt_source=effective,
        source_base_prompt_id=base.prompt_id,
        source_base_system_prompt_sha256=base.sha256,
        environment_pairing_prompt_version=base.version,
    )
    return effective


def terminal_fields(manifest, grade):
    """Original completed/insolvent outcome; infrastructure failures have no reward."""
    require(
        manifest["episode_finished"] is True
        and manifest["capability_score_eligible"] is True
        and manifest["status"] in ("completed", "insolvent")
        and type(manifest["sessions"]) is int
        and 0 <= manifest["sessions"] <= 90,
        "Original world did not reach a valid terminal boundary",
    )
    require(
        grade["status"] == "valid"
        and grade["invalid_reasons"] == []
        and bool(grade["audit"])
        and all(row["pass"] is True and row["failures"] == 0 for row in grade["audit"])
        and type(grade["value"]["business_return_cents"]) is int,
        "Original independent economic/physical grade invalid or incomplete",
    )
    require(
        type(grade["cash"]["insolvent"]) is bool
        and (manifest["status"] == "insolvent") == grade["cash"]["insolvent"]
        and manifest["insolvency_day"] == grade["cash"]["insolvency_day"],
        "Original insolvency event/manifest/grade disagree",
    )
    return grade["value"]["business_return_cents"]


def validate_scope(plan, plan_ref, adoption=None):
    """Parameterize execution, not the original business interface or score."""
    import math
    from decimal import Decimal

    require(
        plan["schema"] == "full-buyer-quarter-plan-v1"
        and plan["archived_replay_only"] is False
        and not any(k in plan for k in ("archive", "members", "archival_order_reconstruction")),
        "Fresh native quarter required",
    )
    native, adapter, limits, resources = (
        plan["native"],
        plan["adapter"],
        plan["limits"],
        plan["resources"],
    )
    # Declared backbones and their official renderers: thinking on, or the official
    # thinking-disabled renderer.
    renderers = {model: by_mode for model, _, by_mode in BACKBONES.values()}
    require(
        native["model"] in renderers
        and native["context_window"] == CONTEXT
        and renderers[native["model"]].get(native["reasoning_mode"]) == native["renderer_name"]
        and native["temperature"] == 1.0,
        "Declared backbone/renderer/reasoning configuration required",
    )
    require(
        type(native["max_output_tokens"]) is int
        and 16000 <= native["max_output_tokens"] <= 65536
        and type(native["max_prompt_tokens"]) is int
        and native["max_prompt_tokens"] > 0
        and native["max_prompt_tokens"] + native["max_output_tokens"] == native["context_window"],
        "Explicit prompt/output context partition required",
    )
    require(
        adapter
        == {
            "model": native["model"],
            "max_output_tokens": native["max_output_tokens"],
            "temperature": 1.0,
            "reasoning_request": None,
            "reasoning_replay": REASONING_REPLAY[native["reasoning_mode"]],
        },
        "Original adapter configuration differs",
    )
    require(
        type(limits["max_provider_calls"]) is int
        and 0 < limits["max_provider_calls"] <= 3600
        and limits["max_physical_attempts"] in (1, 2, 3)
        and limits["ipc_max_bytes"] == 4194304,
        "Finite native/IPC limits required",
    )
    for value in (
        limits["response_seconds"],
        resources["whole_seconds"],
        resources["output_bytes_cap"],
        resources["free_floor_bytes"],
    ):
        require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value > 0,
            "Finite positive resource limit required",
        )
    require(
        limits["response_seconds"] <= resources["whole_seconds"] <= 43200
        and resources["workers"] == 1,
        "One finite owned worker required",
    )
    require(
        type(limits["accounting_ceiling_usd"]) in (int, float),
        "Original engine needs a numeric behavior budget",
    )
    ceiling = Decimal(str(limits["accounting_ceiling_usd"]))
    require(ceiling.is_finite() and ceiling > 0, "Explicit finite behavior budget required")
    require(plan["accounting"] == "list-estimate-after-call-v1", "Original accounting mode differs")
    require(
        set(plan["reservation_rates"]) == {"input_upper", "output"},
        "Original model reservation rates required",
    )
    for rate in plan["reservation_rates"].values():
        require(math.isfinite(rate) and rate >= 0, "Finite explicit model prices required")
    engine_python = Path(plan["engine_python"])
    require(
        engine_python.is_absolute() and engine_python.exists(),
        "Literal absolute engine interpreter path required",
    )
    for key in ("run_id", "policy_id", "purpose"):
        require(isinstance(plan[key], str) and bool(plan[key]), "Missing execution metadata")
    path = plan["sampler_model_path"]
    require(
        path is None
        or (isinstance(path, str) and path.startswith("tinker://") and "/sampler_weights/" in path)
        or (isinstance(path, str) and path.startswith("label:")),
        "Literal sampler path, labeled ephemeral sampler, or frozen base required",
    )
    if adoption is not None:
        require(
            adoption["status"] == "adopted_full_buyer_quarter"
            and adoption["plan"] == plan_ref
            and adoption["output"] == plan["output"]
            and adoption["sampler_model_path"] == path
            and adoption["engine_quarters"] == 1
            and adoption["training_calls"] == adoption["checkpoint_writes"] == 0
            and Decimal(adoption["ceiling_usd"]) == ceiling,
            "Explicit single-quarter policy, source and resource adoption required",
        )
