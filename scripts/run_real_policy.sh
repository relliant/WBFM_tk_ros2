#!/usr/bin/env bash
set -euo pipefail

POLICY_PATH=${1:-}
MANIFEST_PATH=${2:-}
MOTION_FILE=${3:-}

ros2 launch tienkung_bringup sim2real_minimal.launch.py \
  policy_path:="${POLICY_PATH}" \
  manifest_path:="${MANIFEST_PATH}" \
  motion_file:="${MOTION_FILE}"
