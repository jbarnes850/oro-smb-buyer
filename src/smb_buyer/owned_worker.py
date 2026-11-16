"""Original bounded owner with only full-model IPC event/envelope translation."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from .lifecycle import CLEANUP_SECONDS, finish_despite_cancellation, reap_owned


async def run_worker(
    arguments,
    actor_callback,
    *,
    engine_python,
    repository,
    stderr_path,
    worker_script=None,
):
    """Same worker arguments, public JSON protocol and callback; only teardown is bounded."""
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LANG": "C.UTF-8",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stderr_path.open("x") as errors:
        spawn = asyncio.create_task(
            asyncio.create_subprocess_exec(
                str(engine_python),
                str(worker_script or Path(__file__).with_name("worker.py")),
                "--repository",
                str(repository),
                *arguments,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=errors,
                env=environment,
                limit=4 * 1024 * 1024,
            )
        )
        process, complete, worker_failure = None, None, None
        try:
            process = await asyncio.shield(spawn)
            while line := await process.stdout.readline():
                event = json.loads(line)
                if event["event"] == "complete":
                    complete = event
                    continue
                if event["event"] == "failed":
                    worker_failure = event["failure"]
                    failure = RuntimeError(
                        "Original full-buyer quarter failed; preserved failure receipt: "
                        + str(worker_failure)
                    )
                    failure.worker_failure_ref = worker_failure
                    raise failure
                if event["event"] != "model_request":
                    raise RuntimeError("unknown private-worker protocol event")
                try:
                    result = await actor_callback(event)
                    reply = {
                        "event": "model_response",
                        "request_key": event["request_key"],
                        **result,
                    }
                except BaseException:
                    reply = {"request_key": event["request_key"], "abort": True}
                    process.stdin.write((json.dumps(reply) + "\n").encode())
                    await asyncio.wait_for(process.stdin.drain(), CLEANUP_SECONDS)
                    raise
                process.stdin.write((json.dumps(reply) + "\n").encode())
                await process.stdin.drain()
            code = await process.wait()
            if code != 0 or complete is None:
                raise RuntimeError(
                    f"private engine worker failed (exit={code}); see its local failure receipt"
                )
            return complete
        finally:
            cleanup = {"reaped": False, "spawn_pending": not spawn.done()}

            async def finish():
                owned = process
                if owned is None:
                    owned = await asyncio.wait_for(asyncio.shield(spawn), CLEANUP_SECONDS)
                return await reap_owned(owned)

            try:
                cleanup = await finish_despite_cancellation(finish())
            except BaseException as error:
                cleanup.update(getattr(error, "owned_cleanup_report", {}))
                cleanup["error_type"] = type(error).__name__
                if not spawn.done():
                    spawn.cancel()
                raise
            finally:
                if worker_failure is not None:
                    cleanup["worker_failure"] = worker_failure
                # Never convert timeout/cancellation into a clean quarter.
                with stderr_path.with_suffix(stderr_path.suffix + ".cleanup.json").open(
                    "x"
                ) as stream:
                    json.dump(cleanup, stream, sort_keys=True, indent=2)
                    stream.write("\n")
