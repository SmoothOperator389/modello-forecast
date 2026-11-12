#!/usr/bin/env bash
# Deploy branch P0 sulla VM e avvia runner in nohup.
# Uso (dal cloud, quando 10.11.1.139 è raggiungibile):
#   SSHPASS='osit2026!' bash scripts/deploy_p0_to_vm.sh
set -euo pipefail

HOST="${VM_HOST:-modelcdm@10.11.1.139}"
ROOT_REMOTE="${MODELLO_ROOT:-/opt/modello-forecast}"
BRANCH="${DEPLOY_BRANCH:-cursor/p0-h30-sku-quality-5008}"
LOGDIR_LOCAL="${LOGDIR_LOCAL:-logs}"
mkdir -p "$LOGDIR_LOCAL"

ssh_cmd() {
  if [[ -n "${SSHPASS:-}" ]] && command -v sshpass >/dev/null 2>&1; then
    sshpass -p "$SSHPASS" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 "$@"
  else
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 "$@"
  fi
}

echo "[deploy] probe $HOST"
ssh_cmd "$HOST" "hostname; test -d $ROOT_REMOTE && echo OK_ROOT"

echo "[deploy] sync git branch $BRANCH"
ssh_cmd "$HOST" "bash -s" <<EOF
set -euo pipefail
cd $ROOT_REMOTE
git fetch origin || true
git checkout -B $BRANCH origin/$BRANCH 2>/dev/null || git pull --ff-only || true
# fallback: se il remote non ha il branch, lascia working tree e copia sarà via rsync lato chiamante
mkdir -p logs
chmod +x scripts/vm_p0_h30_runner.sh scripts/deploy_p0_to_vm.sh 2>/dev/null || true
nohup bash scripts/vm_p0_h30_runner.sh all > logs/p0_h30_runner.nohup.out 2>&1 &
echo \$! | tee logs/p0_h30_runner.pid
echo "[vm] started pid=\$(cat logs/p0_h30_runner.pid)"
EOF

echo "[deploy] done — monitor: ssh $HOST 'tail -f $ROOT_REMOTE/logs/p0_h30_runner.log'"
