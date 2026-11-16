"""Where the engine, worlds, outputs and shipped files live.

Everything outside this package is named by an environment variable (or the CLI flag
that sets it); nothing is hard-coded to one machine:

  SMB_ENGINE_DIR      ORO SMB Bench ``retail_v1`` source at commit b17b864 (required)
  SMB_ENGINE_PYTHON   interpreter with the engine's own dependencies
                      (default: <package>/engine-runtime/.venv/bin/python)
  SMB_WORLD_REGISTRY  world registry written by ``scripts/campaign.py build-registry``
                      (default: $SMB_RUNS_DIR/worlds.json)
  SMB_RUNS_DIR        output root for quarters, training runs and the ledger
                      (default: <package>/runs)
  SMB_FRONTIER_INDEX  JSON list of archived frontier quarters (branch-eval, RL, hurdles)
  TINKER_API_KEY      read by the Tinker SDK from the environment only

The engine child process gets none of these variables; everything it needs is in
the hash-bound quarter plan.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROMPT = ROOT / "prompts/buyer_system.txt"
ENGINE_HASHES = ROOT / "data/engine-b17b864.json"


class ConfigError(RuntimeError):
    """A required location was not configured."""


def _path(name: str, default: Path | None = None, *, resolve: bool = True) -> Path:
    value = os.environ.get(name)
    if not value:
        if default is None:
            raise ConfigError(f"Set {name} (see README.md, 'Configuration')")
        return default
    path = Path(value).expanduser()
    # The engine interpreter stays literal (a venv python is a symlink; resolving it
    # would escape the venv). Directories and files are resolved.
    return path.resolve() if resolve else path.absolute()


def engine_dir() -> Path:
    return _path("SMB_ENGINE_DIR")


def engine_python() -> Path:
    return _path("SMB_ENGINE_PYTHON", ROOT / "engine-runtime/.venv/bin/python", resolve=False)


def runs_dir() -> Path:
    return _path("SMB_RUNS_DIR", ROOT / "runs")


def registry() -> Path:
    return _path("SMB_WORLD_REGISTRY", runs_dir() / "worlds.json")


def frontier_index() -> Path:
    return _path("SMB_FRONTIER_INDEX")


def frontier_hurdles() -> Path:
    return runs_dir() / "frontier-hurdles"


def noop_floor() -> Path:
    return runs_dir() / "noop-floor/summary.json"


def require_tinker_key() -> None:
    if not os.environ.get("TINKER_API_KEY"):
        raise ConfigError("Set TINKER_API_KEY in the environment; it is never read from a file")
