"""Credential-free original full-buyer worker. All business behavior is upstream."""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
# Runs under the engine interpreter, where this package is not installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smb_buyer.contracts import (  # noqa: E402
    ENVIRONMENT,
    ROOT,
    import_original,
    install_system_prompt,
    ref,
    require,
    terminal_fields,
    verify,
    write,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--adoption", required=True)
    parser.add_argument("--adoption-sha256", required=True)
    args = parser.parse_args()
    require(args.repository.resolve() == ROOT, "Wrong repository")
    require(
        not any(
            k.upper().endswith("_API_KEY") or k in ("TINKER_API_KEY", "OPENROUTER_API_KEY")
            for k in os.environ
        ),
        "Credentials must not enter the engine child",
    )
    plan_ref = {"path": args.plan, "sha256": args.plan_sha256}
    adoption_ref = {"path": args.adoption, "sha256": args.adoption_sha256}
    plan, world = verify(plan_ref, adoption_ref)
    # Literal comparison: a venv interpreter resolves to its base python, so
    # resolving both sides could not tell two virtualenvs apart.
    require(
        Path(sys.executable).absolute() == Path(plan["engine_python"]),
        "Worker is not running under the plan's literal engine interpreter",
    )
    import_original(plan)
    from retailsim.grader.grade import grade_run
    from retailsim.harness.rollout import LLMBuyerPolicy
    from retailsim.harness.run import run_episode

    from smb_buyer.bridge import IPCModelAdapter

    output = Path(plan["output"]) / "engine"
    require(not output.exists(), "Refusing existing full-quarter output")
    output.mkdir()
    adapter = IPCModelAdapter(
        plan["adapter"],
        replay_only=plan["archived_replay_only"],
        timeout_seconds=plan["limits"]["response_seconds"],
        max_line_bytes=plan["limits"]["ipc_max_bytes"],
    )
    policy = LLMBuyerPolicy(
        adapter,
        policy_id=plan["policy_id"],
        prompt_version="v11",
        max_turns_per_day=100,
        max_tool_calls_per_day=60,
        max_request_bytes=512000,
        max_provider_calls=plan["limits"]["max_provider_calls"],
        max_accounted_cost_usd=plan["limits"]["accounting_ceiling_usd"],
        reservation_rates=plan["reservation_rates"],
        accounting=plan["accounting"],
    )
    effective_prompt = install_system_prompt(policy, plan)
    adapter.policy = policy
    metadata = {
        "environment_contract": ENVIRONMENT,
        "bridge_plan": plan_ref,
        "archived_replay_only": plan["archived_replay_only"],
        "training_credit": False,
        "purpose": plan["purpose"],
    }
    try:
        manifest = run_episode(
            world,
            policy,
            output,
            run_id=plan["run_id"],
            history=None,
            transport="inproc",
            meta=metadata,
            max_sessions=90,
            checkpoint_days=False,
        )
        grade = grade_run(output, world)
        terminal_fields(manifest, grade)
        result = {
            "status": "valid",
            "environment_contract": ENVIRONMENT,
            "plan": plan_ref,
            "adoption": adoption_ref,
            "grade": grade,
            "system_prompt": effective_prompt,
            "return_cents": grade["value"]["business_return_cents"],
            "sessions": manifest["sessions"],
            "terminal_status": manifest["status"],
            "insolvent": grade["cash"]["insolvent"],
            "grade_ref": ref(output / "grade.json"),
            "events": ref(output / "events.jsonl"),
            "actions": ref(output / "actions.jsonl"),
            "journal": ref(output / "journal.jsonl"),
            "logical_model_requests": len(adapter.receipts),
            "model_calls": policy.provider_calls,
            "agent_tool_calls": policy.agent_tool_calls,
            "budgeted_agent_tool_calls": policy.budgeted_agent_tool_calls,
            "memory_snapshots": len(policy.memory.files.snapshot_index),
            "archived_replay_only": plan["archived_replay_only"],
            "training_calls": 0,
            "checkpoint_writes": 0,
            "run_manifest": ref(output / "run_manifest.json"),
            "model_transcript": ref(output / "model_transcript.jsonl"),
            "tool_calls": ref(output / "tool_calls.jsonl"),
            "test_business_accessed": plan.get("split") == "test",
            "network_client_in_child": False,
        }
        write(output / "bridge-receipts.json", adapter.receipts)
        result["bridge_receipts"] = ref(output / "bridge-receipts.json")
        write(output / "result.json", result)
        print(
            json.dumps(
                {
                    "event": "complete",
                    "result": ref(output / "result.json"),
                    **{
                        k: result[k]
                        for k in (
                            "status",
                            "return_cents",
                            "sessions",
                            "logical_model_requests",
                            "agent_tool_calls",
                            "budgeted_agent_tool_calls",
                            "archived_replay_only",
                        )
                    },
                }
            ),
            flush=True,
        )
    except BaseException as error:
        # The original run_episode already finalizes failed physical/model evidence.
        write(
            output / "failure.json",
            {
                "status": "failed_no_retry",
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                "plan": plan_ref,
                "logical_requests_dispatched": len(adapter.dispatched),
            },
        )
        print(
            json.dumps({"event": "failed", "failure": ref(output / "failure.json")}),
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()
