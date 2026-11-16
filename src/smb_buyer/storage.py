"""State-only final checkpoint for the official Cookbook trainers, with an optional
durable-storage budget.

Training runs save no periodic or rolling checkpoints; the final save is a
training state only (samplers are created ephemerally from it). If an owned-
checkpoint registry is given, the save waits until this project's own durable
footprint plus the new state fits that budget, and the new paths are recorded.
Registry format (JSON): ``{"budget_bytes": <int>, "owned": {<tinker path>: <note>}}``.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

# Upper-bound sizes of a rank-32 LoRA training state with AdamW moments. Qwen3.6 is
# the observed size; Qwen3.8 (dense, ~220M LoRA params) is an estimate.
STATE_BYTES_BY_BACKBONE = {"qwen3.6": 6_706_370_119, "qwen3.8": 3_500_000_000}


async def owned_bytes(registry_path):
    import tinker

    registry = json.loads(Path(registry_path).read_text())
    rest, sizes, offset = tinker.RestClient(), {}, 0
    while True:  # paginate: an organization listing can exceed one page
        response = await rest.list_user_checkpoints(limit=200, offset=offset)
        sizes.update({c.tinker_path: c.size_bytes or 0 for c in response.checkpoints})
        offset += len(response.checkpoints)
        if not response.checkpoints or offset >= response.cursor.total_count:
            break
    owned = sum(sizes.get(path, 0) for path in registry["owned"])
    return owned, registry["budget_bytes"], sum(sizes.values())


def register_owned(registry_path, path, note):
    registry = json.loads(Path(registry_path).read_text())
    registry["owned"][path] = note
    Path(registry_path).write_text(json.dumps(registry, indent=1))


async def wait_for_storage(
    registry_path, needed_bytes, status_path, *, poll_seconds=300, max_seconds=48 * 3600
):
    """Block (session alive) until a durable save fits the owned-checkpoint budget."""
    start = time.time()
    while True:
        owned, budget, org = await owned_bytes(registry_path)
        status = {
            "owned_bytes": owned,
            "budget_bytes": budget,
            "org_bytes": org,
            "needed_bytes": needed_bytes,
            "checked_at": time.time(),
            "fits": owned + needed_bytes <= budget,
        }
        Path(status_path).write_text(json.dumps(status, indent=1))
        if status["fits"]:
            return status
        if time.time() - start > max_seconds:
            raise RuntimeError("Checkpoint storage did not become available")
        await asyncio.sleep(poll_seconds)


def state_only_final(checkpoint_utils, *, backbone, owned_registry=None, skip=False):
    """A CheckpointManager subclass whose final save is state-only (or skipped)."""

    class StateOnlyFinal(checkpoint_utils.CheckpointManager):
        async def save_final_async(self, loop_state):
            await self._resolve_pending_periodic_async()
            await self._resolve_pending_rolling_async()
            if skip:
                await self.finalize_async()
                return {}
            if owned_registry is not None:
                await wait_for_storage(
                    owned_registry,
                    STATE_BYTES_BY_BACKBONE[backbone],
                    Path(self._log_path) / "storage-wait.json",
                )
            paths = await checkpoint_utils.save_checkpoint_async(
                training_client=self._training_client,
                name="final",
                log_path=self._log_path,
                kind="state",
                loop_state=loop_state,
                ttl_seconds=None,
                store=self._store,
            )
            if owned_registry is not None:
                for kind_key in ("state_path", "sampler_path"):
                    if paths.get(kind_key):
                        register_owned(
                            owned_registry,
                            paths[kind_key],
                            f"{Path(self._log_path).name} final ({kind_key})",
                        )
            await self.finalize_async()
            return paths

    return StateOnlyFinal
