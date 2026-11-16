#!/usr/bin/env bash
# Zero-cost smoke test: never contacts Tinker (TINKER_API_KEY is unset for every step).
#
#   PY=.venv/bin/python scripts/smoke_test.sh
#
# Always: byte-compile, import every module (engine-side modules under the engine
# interpreter too), --help for every entrypoint, offline unit tests.
# If SMB_ENGINE_DIR and a world registry are configured: check-setup and one
# advance-only quarter through the real engine and grader (about 15 s).
# If SMB_FRONTIER_INDEX and the frontier hurdles are present: an RL --dry-run.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
unset TINKER_API_KEY
export PYTHONDONTWRITEBYTECODE=1
TMP=$(mktemp -d)
FAIL=0
step() {  # step NAME CMD...
  local name=$1; shift
  if "$@" >"$TMP/log" 2>&1; then echo "PASS  $name"; else echo "FAIL  $name"; tail -n 15 "$TMP/log" | sed 's/^/      /'; FAIL=1; fi
}

step "byte-compile src scripts tests" env PYTHONPYCACHEPREFIX="$TMP/pycache" "$PY" -m compileall -q src scripts tests

for m in smb_buyer smb_buyer.paths smb_buyer.backbones smb_buyer.contracts smb_buyer.lifecycle \
         smb_buyer.owned_worker smb_buyer.native smb_buyer.qwen_tools smb_buyer.rl_env smb_buyer.campaign \
         smb_buyer.algorithms smb_buyer.storage smb_buyer.analysis smb_buyer.packed_sft smb_buyer.public_export; do
  step "import $m" "$PY" -B -c "import $m"
done

if [ -n "${SMB_ENGINE_DIR:-}" ]; then
  ENGINE_PY=${SMB_ENGINE_PYTHON:-engine-runtime/.venv/bin/python}
  step "engine-side imports under $ENGINE_PY" "$ENGINE_PY" -B -c "
import sys; sys.dont_write_bytecode = True
sys.path[:0] = ['src', '$SMB_ENGINE_DIR']
import smb_buyer.contracts, smb_buyer.backbones, smb_buyer.lifecycle, smb_buyer.bridge"
fi

for s in campaign rl rl_guard frontier_hurdles build_sft_corpus econ_sft select_val report_test costs supplier_share plots story_figures leaderboard_snapshot export_model; do
  step "scripts/$s.py --help" "$PY" -B "scripts/$s.py" --help
done
for sub in build-registry check-setup noop eval branch-eval resume status; do
  step "scripts/campaign.py $sub --help" "$PY" -B scripts/campaign.py "$sub" --help
done

step "pytest tests (offline)" "$PY" -B -m pytest -q -p no:cacheprovider tests

if [ -n "${SMB_ENGINE_DIR:-}" ]; then
  step "campaign.py check-setup" "$PY" -B scripts/campaign.py check-setup
  REGISTRY=$("$PY" -B -c "from smb_buyer import paths; print(paths.registry())")
  if [ -f "$REGISTRY" ]; then
    WORLD=$("$PY" -B -c "from smb_buyer import campaign; print(campaign.worlds_in_split('validation')[0])")
    step "campaign.py noop $WORLD (real engine + grader, zero model calls)" \
      "$PY" -B scripts/campaign.py noop --worlds "$WORLD" --out "$TMP/noop"
    if [ -f "$TMP/noop/summary.json" ]; then echo "      $(cat "$TMP/noop/summary.json" | tr -d '\n ' )"; fi
  else
    echo "SKIP  engine quarter: no registry at $REGISTRY (scripts/campaign.py build-registry)"
  fi
  RUNS=$("$PY" -B -c "from smb_buyer import paths; print(paths.runs_dir())")
  if [ -n "${SMB_FRONTIER_INDEX:-}" ] && [ -d "$RUNS/frontier-hurdles" ] && [ -f "$RUNS/noop-floor/summary.json" ]; then
    step "rl.py --dry-run (batch plans, no Tinker)" "$PY" -B scripts/rl.py --dry-run --arm cispo_maxrl_binary \
      --parent-state base --log-path "$TMP/rl-dry" --steps 2 --curriculum 60-85
  else
    echo "SKIP  rl.py --dry-run: needs SMB_FRONTIER_INDEX, \$SMB_RUNS_DIR/frontier-hurdles and \$SMB_RUNS_DIR/noop-floor"
  fi
else
  echo "SKIP  engine checks: SMB_ENGINE_DIR is not set"
fi

rm -rf "$TMP"
[ "$FAIL" = 0 ] && echo "SMOKE TEST PASSED" || echo "SMOKE TEST FAILED"
exit "$FAIL"
