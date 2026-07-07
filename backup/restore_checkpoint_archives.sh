#!/usr/bin/env bash
set -euo pipefail

# Restores split checkpoint archives created by create_checkpoint_archives_on_ec2.sh.
# Put all parts in ./checkpoint_archives or set ARCHIVE_DIR.

ARCHIVE_DIR="${ARCHIVE_DIR:-checkpoint_archives}"
RESTORE_ROOT="${RESTORE_ROOT:-.}"

cd "$ARCHIVE_DIR"
sha256sum -c SHA256SUMS.txt
cat pretrain_checkpoints.tar.zst.part-* | zstd -d | tar -xf - -C "$RESTORE_ROOT"
