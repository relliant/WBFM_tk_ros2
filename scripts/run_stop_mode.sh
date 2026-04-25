#!/usr/bin/env bash
set -euo pipefail

POLICY_PATH=${1:-}
MANIFEST_PATH=${2:-}

ros2 launch tienkung_bringup policy_runner.launch.py \
  policy_path:="${POLICY_PATH}" \
  manifest_path:="${MANIFEST_PATH}"
