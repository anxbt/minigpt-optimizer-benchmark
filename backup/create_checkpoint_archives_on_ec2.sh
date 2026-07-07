#!/usr/bin/env bash
set -euo pipefail

# Run this on EC2. It creates split tar archives below /home/anubrat/checkpoint_archives.
# Each part is <= 1900MB, suitable for GitHub Release assets.

SRC="${SRC:-/home/anubrat/pretrain_runs}"
OUT="${OUT:-/home/anubrat/checkpoint_archives}"
PART_SIZE="${PART_SIZE:-1900M}"

mkdir -p "$OUT"
cd "$(dirname "$SRC")"
base="$(basename "$SRC")"

tar --checkpoint=.1000 -cf - "$base" \
  --wildcards '*/checkpoints/latest.pt' \
  | zstd -T0 -19 \
  | split -b "$PART_SIZE" - "$OUT/pretrain_checkpoints.tar.zst.part-"

cd "$OUT"
sha256sum pretrain_checkpoints.tar.zst.part-* > SHA256SUMS.txt
ls -lh
