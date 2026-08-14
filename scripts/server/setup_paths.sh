#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${COMPBIAS_PROJECT_ROOT:-/cloud/cloud-ssd1/dissertation}"
if [[ "${PROJECT_ROOT}" != /cloud/cloud-ssd1/* ]]; then
  echo "ERROR: project root must stay under /cloud/cloud-ssd1" >&2
  exit 2
fi

install -d -m 700 \
  "${PROJECT_ROOT}/data" \
  "${PROJECT_ROOT}/outputs" \
  "${PROJECT_ROOT}/checkpoints" \
  "${PROJECT_ROOT}/trajectories" \
  "${PROJECT_ROOT}/cache"

if [[ ! -e "${PROJECT_ROOT}/configs/paths.yaml" ]]; then
  cp "${PROJECT_ROOT}/configs/paths.example.yaml" "${PROJECT_ROOT}/configs/paths.yaml"
  chmod 600 "${PROJECT_ROOT}/configs/paths.yaml"
fi
