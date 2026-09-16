#!/usr/bin/env bash
# Create a real Python 3.11+ venv for modello-forecast on macOS.
#
# Apple CLT Python 3.9 (/Library/Developer/CommandLineTools/...) often cannot
# run TensorFlow on recent macOS (mutex lock / import hang). Use python.org or
# Homebrew Python 3.11+ instead.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"

cd "${PROJECT_ROOT}"

find_python() {
  local candidates=(
    /usr/local/bin/python3.11
    /opt/homebrew/bin/python3.11
    /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11
    python3.11
    python3.12
    python3
  )
  local c ver major minor
  for c in "${candidates[@]}"; do
    if ! command -v "${c}" >/dev/null 2>&1; then
      continue
    fi
    if ! ver="$("${c}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)"; then
      continue
    fi
    major="${ver%%.*}"
    minor="${ver#*.}"
    if [[ "${major}" -eq 3 && "${minor}" -ge 11 ]]; then
      if "${c}" -c 'import sys; exit("CommandLineTools" in sys.executable)' 2>/dev/null; then
        echo "${c}"
        return 0
      fi
    fi
  done
  return 1
}

echo "=== modello-forecast: setup venv Mac ==="

if [[ -L "${VENV_DIR}" ]]; then
  echo "Rimuovo symlink .venv -> $(readlink "${VENV_DIR}")"
  rm "${VENV_DIR}"
elif [[ -d "${VENV_DIR}" ]]; then
  echo "Rimuovo venv esistente: ${VENV_DIR}"
  rm -rf "${VENV_DIR}"
fi

PY="$(find_python)" || {
  cat <<'EOF' >&2
ERRORE: Python 3.11+ non trovato (o punta ancora a Command Line Tools).

Installa uno di:
  • https://www.python.org/downloads/  (consigliato: Python 3.11)
  • brew install python@3.11

Poi riesegui:
  ./scripts/setup_venv_mac.sh
EOF
  exit 1
}

echo "Usando: ${PY} ($(${PY} --version))"
"${PY}" -m venv "${VENV_DIR}"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo ""
echo "Verifica GPU (Apple Silicon):"
python -c 'from testing.tensorflow_env import import_tensorflow; import_tensorflow(); import tensorflow as tf; print(tf.config.list_physical_devices())'

echo ""
echo "Venv pronto: ${VENV_DIR}"
echo ""
echo "Verifica TensorFlow:"
echo "  OMP_NUM_THREADS=1 .venv/bin/python -c 'from testing.tensorflow_env import import_tensorflow; import_tensorflow(); print(\"OK\")'"
echo ""
echo "Training:"
echo "  OMP_NUM_THREADS=1 .venv/bin/python -m testing.run_train"
echo ""
echo "Se TensorFlow fallisce ancora, usa Docker:"
echo "  ./scripts/run_train_docker.sh"
