"""Bounded ownership of the engine child: settle through cancellation, then reap."""

from __future__ import annotations

import asyncio

CLEANUP_SECONDS = 5.0


async def finish_despite_cancellation(awaitable):
    """Await ``awaitable`` to completion even if the caller is cancelled meanwhile."""
    task = asyncio.ensure_future(awaitable)
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


async def reap_owned(process, *, grace_seconds=CLEANUP_SECONDS, kill_seconds=CLEANUP_SECONDS):
    """Keep the same wait future through TERM/KILL; report unconfirmed reaping as failure."""
    if not (0 < grace_seconds <= 5 and 0 < kill_seconds <= 5):
        raise ValueError("Finite owned-process cleanup bounds required")
    report = {
        "pid": process.pid,
        "term_sent": False,
        "kill_sent": False,
        "reaped": False,
    }
    wait = asyncio.create_task(process.wait())
    try:
        if process.returncode is None:
            try:
                process.terminate()
                report["term_sent"] = True
            except ProcessLookupError:
                pass
        try:
            await asyncio.wait_for(asyncio.shield(wait), grace_seconds)
        except TimeoutError:
            if process.returncode is None:
                try:
                    process.kill()
                    report["kill_sent"] = True
                except ProcessLookupError:
                    pass
            await asyncio.wait_for(asyncio.shield(wait), kill_seconds)
        report.update(reaped=process.returncode is not None, returncode=process.returncode)
        if not report["reaped"]:
            raise RuntimeError("Owned worker wait completed without a terminal process")
        return report
    except BaseException as error:
        report.update(
            error_type=type(error).__name__,
            returncode=process.returncode,
            reaped=process.returncode is not None,
        )
        error.owned_cleanup_report = report
        raise
    finally:
        if not wait.done():
            wait.cancel()  # Stop only the local waiter; remote/child state remains uncertain.
