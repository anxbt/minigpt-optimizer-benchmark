#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-anubrat@13.234.232.112}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/anubrat/pretrain_runs}"
LOCAL_ROOT="${LOCAL_ROOT:-gpu_benchmark/downloaded_checkpoints_from_ec2}"

mkdir -p "$LOCAL_ROOT"

# Downloads checkpoint files from EC2 while preserving run directory structure.
# Example:
#   bash backup/download_ec2_checkpoints.sh
rsync -av --progress \
  --include='*/' \
  --include='checkpoints/latest.pt' \
  --exclude='*' \
  "$HOST:$REMOTE_ROOT/" \
  "$LOCAL_ROOT/"
