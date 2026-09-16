#!/usr/bin/env bash
# Run TensorFlow training inside Docker (avoids Mac CLT Python + TF issues).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE="${TF_DOCKER_IMAGE:-tensorflow/tensorflow:2.15.0}"

cd "${PROJECT_ROOT}"

exec docker run --rm -it \
  -v "${PROJECT_ROOT}:/workspace" \
  -w /workspace \
  -e OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
  "${IMAGE}" \
  bash -c 'pip install -q -r requirements.txt && python -m testing.run_train "$@"' _ "$@"
