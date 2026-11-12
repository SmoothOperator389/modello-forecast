#!/usr/bin/env bash
# P0 h30 SKU quality runner — pensato per girare SU /opt/modello-forecast (VM).
# Avvio tipico (dal cloud dopo SSH, o in locale VM):
#   nohup bash scripts/vm_p0_h30_runner.sh all > logs/p0_h30_runner.nohup.out 2>&1 &
#   echo $! | tee logs/p0_h30_runner.pid
set -euo pipefail

ROOT="${MODELLO_ROOT:-/opt/modello-forecast}"
cd "$ROOT"
mkdir -p logs
export MODELLO_ROOT="$ROOT"
export BI_TRAINING_LOCK_PATH="${BI_TRAINING_LOCK_PATH:-$ROOT/logs/bi_training.lock}"
export PYTHONPATH="${PYTHONPATH:-$ROOT}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

PY="${PYTHON_BIN:-}"
if [[ -z "$PY" ]]; then
  for c in "$ROOT/.venv2/bin/python" "$ROOT/.venv/bin/python" python3; do
    if command -v "$c" >/dev/null 2>&1 || [[ -x "$c" ]]; then
      PY="$c"
      break
    fi
  done
fi
echo "[p0] ROOT=$ROOT PY=$PY $(date -u +%FT%TZ)" | tee -a logs/p0_h30_runner.log

CMD="${1:-all}"

release_lock() {
  "$PY" -m production.ops.bi_training_lock force-release || true
}
trap release_lock EXIT

step_policy() {
  echo "=== 1) Jul/Aug NON AFFIDABILE (no July wait) ===" | tee -a logs/p0_h30_runner.log
  "$PY" - <<'PY' | tee -a logs/p0_h30_runner.log
from production.ops.month_reliability import month_status, NON_AFFIDABILE_MONTHS
import json
print(json.dumps({m: month_status(2025, m) for m in sorted(NON_AFFIDABILE_MONTHS)}, indent=2, ensure_ascii=False))
PY
}

step_wire_tip() {
  echo "=== 3) Wire tip_budget into weekly/monthly ===" | tee -a logs/p0_h30_runner.log
  "$PY" -m production.ops.tip_budget_pipeline --wire --discover --refresh --dry-run \
    | tee logs/tip_budget_wire.json | tee -a logs/p0_h30_runner.log
  # refresh reale solo se scoperto
  "$PY" -m production.ops.tip_budget_pipeline --refresh \
    | tee logs/tip_budget_refresh.json | tee -a logs/p0_h30_runner.log || true
}

step_rebuild_train() {
  echo "=== 2) Rebuild samples (no poisoned) + train LGBM h30 + YoY ===" | tee -a logs/p0_h30_runner.log
  "$PY" -m production.ops.bi_training_lock acquire --reason h30_sku_rebuild_train
  export FORECAST_NATIVE_HORIZON=30
  export FORECAST_EXCLUDE_POISONED_AS_OF=1
  export FORECAST_ATTACH_YOY=1
  # Prefer wrapper se presente, altrimenti modulo nativo + post-hook
  if [[ -f production/run_train_h30_p0.py ]]; then
    nohup "$PY" -m production.run_train_h30_p0 > logs/train_h30_p0.out 2>&1 &
    echo $! | tee logs/train_h30_p0.pid
    echo "[p0] train pid=$(cat logs/train_h30_p0.pid) log=logs/train_h30_p0.out"
    wait "$(cat logs/train_h30_p0.pid)"
  else
    nohup env FORECAST_NATIVE_HORIZON=30 "$PY" -m production.run_train_horizon_native_real \
      > logs/train_h30_native.out 2>&1 &
    echo $! | tee logs/train_h30_native.pid
    echo "[p0] train pid=$(cat logs/train_h30_native.pid) log=logs/train_h30_native.out"
    wait "$(cat logs/train_h30_native.pid)"
  fi
}

step_holdout_scoreboard() {
  echo "=== 4/holdout+scoreboard ===" | tee -a logs/p0_h30_runner.log
  # Discovery: qualunque export holdout Sep/Oct già cablato in repo VM
  for mod in \
    production.run_export_bi_holdout \
    production.bi_engine \
    bi_engine
  do
    if "$PY" -c "import importlib; importlib.import_module('$mod')" 2>/dev/null; then
      echo "[p0] found module $mod" | tee -a logs/p0_h30_runner.log
    fi
  done
  if [[ -x scripts/regen_sep_oct_holdouts.sh ]]; then
    nohup bash scripts/regen_sep_oct_holdouts.sh > logs/holdout_sep_oct.out 2>&1 &
    echo $! | tee logs/holdout_sep_oct.pid
    wait "$(cat logs/holdout_sep_oct.pid)" || true
  fi
  echo "[p0] scoreboard helper ready via production.ops.bi_scoreboard" | tee -a logs/p0_h30_runner.log
}

case "$CMD" in
  policy) step_policy ;;
  tip) step_wire_tip ;;
  train) step_rebuild_train ;;
  holdout) step_holdout_scoreboard ;;
  all)
    step_policy
    step_wire_tip
    step_rebuild_train
    step_holdout_scoreboard
    ;;
  *)
    echo "usage: $0 {all|policy|tip|train|holdout}" >&2
    exit 2
    ;;
esac

echo "[p0] DONE $(date -u +%FT%TZ)" | tee -a logs/p0_h30_runner.log
